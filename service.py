from __future__ import annotations

import io
import typing as t

import bentoml
import torch
from bentoml.io import JSON, File, Image, Multipart
from einops import rearrange
from einops import repeat
from pydantic import BaseModel

from config import MODEL_VERSION
from config import DEFAULT_CONFIGS
from helpers import load_svd_model
from helpers import get_batch
from helpers import preprocess_image
from helpers import postprocess_samples
from helpers import get_unique_embedder_keys_from_conditioner

if t.TYPE_CHECKING:
    from PIL.Image import Image as PIL_Image
    from numpy.typing import NDArray

bento_model = bentoml.models.get(MODEL_VERSION)

class SVDRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'cpu')
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self) -> None:

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        _config_d = DEFAULT_CONFIGS[MODEL_VERSION]
        self.num_frames = _config_d["num_frames"]
        self.num_steps = _config_d["num_steps"]

        config_str = f"configs/{MODEL_VERSION}.yaml"
        model_filename: str = bento_model.info.metadata["filename"]
        model_path = bento_model.path_of(model_filename)
        self.model = load_svd_model(
            config_str,
            self.device,
            self.num_frames,
            self.num_steps,
            model_path,
        )

    @bentoml.Runnable.method(batchable=False)
    def generate(
            self,
            tensor: torch.Tensor,
            *,
            motion_bucket_id: int,
            fps_id: int,
            cond_aug: float,
            decoding_t: int,
            seed: int | None = None,
    ) -> NDArray:

        if seed is not None:
            torch.manual_seed(seed)

        tensor = tensor.to("cuda")
        H, W = tensor.shape[2:]
        assert tensor.shape[1] == 3
        F = 8
        C = 4
        shape = (self.num_frames, C, H // F, W // F)

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = tensor
        value_dict["cond_frames"] = tensor + cond_aug * torch.randn_like(tensor)
        value_dict["cond_aug"] = cond_aug

        with torch.no_grad():
            with torch.autocast(self.device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(self.model.conditioner),
                    value_dict,
                    [1, self.num_frames],
                    T=self.num_frames,
                    device=self.device,
                )
                c, uc = self.model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=self.num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=self.num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=self.num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=self.num_frames)

                randn = torch.randn(shape, device=self.device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, self.num_frames
                ).to(self.device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return self.model.denoiser(
                        self.model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = self.model.sampler(denoiser, randn, cond=c, uc=uc)
                self.model.en_and_decode_n_samples_a_time = decoding_t

                samples_x = self.model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                return samples.cpu().numpy()


runner = bentoml.Runner(
    SVDRunnable,
    name=f"{MODEL_VERSION}-runner",
    models=[bento_model]
)

svc = bentoml.Service(f"{MODEL_VERSION}-service", runners=[runner])

class Params(BaseModel):
    fps_id: t.Optional[int] = 6
    motion_bucket_id: t.Optional[int] = 127
    cond_aug: t.Optional[float] = 0.02
    decoding_t: t.Optional[int] = 1
    seed: t.Optional[int] = None

params_sample = Params()

@svc.api(
    route="/generate",
    input=Multipart(img=Image(), params=JSON.from_sample(params_sample)),
    output=File(mime_type="video/mp4"),
)
async def generate(img: PIL_Image, params: Params):
    if params.motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    params_d = params.dict()
    tensor = preprocess_image(img)

    samples = await runner.generate.async_run(tensor, **params_d)
    video_bytes = postprocess_samples(samples, params.fps_id)
    f = io.BytesIO(video_bytes)
    return f
