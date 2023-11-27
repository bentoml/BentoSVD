from __future__ import annotations

import math
import os
import tempfile
import typing as t

import bentoml
import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL.Image import Image
from torchvision.transforms import ToTensor

if t.TYPE_CHECKING:
    from numpy.typing import NDArray

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_svd_model(
    config_str: str,
    device: str,
    num_frames: int,
    num_steps: int,
    model_path: str,
    clip_model_path: str,
    lowvram_mode: bool,
):

    from sgm.util import instantiate_from_config

    config = OmegaConf.load(config_str)
    config.model.params.ckpt_path = model_path
    config.model.params.conditioner_config.params.emb_models[
        0
    ].params.open_clip_embedding_config.params.version = clip_model_path

    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        if lowvram_mode:
            model = instantiate_from_config(config.model)
            model.model.half()
            with torch.device(device):
                model = model.cuda().eval()
        else:
            with torch.device(device):
                model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
    return model


def preprocess_image(image: Image) -> torch.Tensor:
    if image.mode == "RGBA":
        image = image.convert("RGB")

    w, h = image.size
    if h % 64 != 0 or w % 64 != 0:
        width, height = map(lambda x: x - x % 64, (w, h))
        image = image.resize((width, height))
        print(
            f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
        )

    tensor = ToTensor()(image)
    tensor = tensor * 2.0 - 1.0

    tensor = tensor.unsqueeze(0)

    H, W = tensor.shape[2:]
    if (H, W) != (576, 1024):
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )

    return tensor


def postprocess_samples(samples: NDArray, fps_id: int) -> bytes:

    _, output_path = tempfile.mkstemp(".mp4")
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps_id + 1,
        (samples.shape[-1], samples.shape[-2]),
    )

    vid = (rearrange(samples, "t c h w -> t h w c") * 255).astype(np.uint8)
    for frame in vid:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()

    with open(output_path, "rb") as f:
        res = f.read()

    os.remove(output_path)
    return res
