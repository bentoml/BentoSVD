from __future__ import annotations

import os
from pathlib import Path
from PIL.Image import Image

import bentoml

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


@bentoml.service(
    resources={
        "GPU": 1,
        "memory": "16Gi",
    },
    traffic={"timeout": 300},
)
class SVDService:

    def __init__(self) -> None:
        import torch
        import diffusers

        # Load model into pipeline
        self.pipe = diffusers.StableVideoDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16"
        )
        self.pipe.to("cuda")


    @bentoml.api
    def generate(
            self, context: bentoml.Context,
            image: Image,
            decode_chunk_size: int = 2,
            seed: int | None = None,
    ) -> Path:
        import torch
        from diffusers.utils import load_image, export_to_video

        generator = torch.manual_seed(seed) if seed is not None else None
        image = image.resize((1024, 576))
        image = image.convert("RGB")
        output_path = os.path.join(context.directory, "output.mp4")

        frames = self.pipe(
            image, decode_chunk_size=decode_chunk_size, generator=generator,
        ).frames[0]
        export_to_video(frames, output_path)
        return Path(output_path)
