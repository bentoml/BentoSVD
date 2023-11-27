import huggingface_hub
import bentoml
from bentoml.models import ModelContext

from config import MODEL_VERSION

MODEL_VERSION_MAPPING = {
    "svd": (
        "stabilityai/stable-video-diffusion-img2vid",
        "svd.safetensors"
    ),
    "svd_image_decoder": (
        "stabilityai/stable-video-diffusion-img2vid",
        "svd_image_decoder.safetensors"
    ),
    "svd_xt": (
        "stabilityai/stable-video-diffusion-img2vid-xt",
        "svd_xt.safetensors"
    ),
    "svd_xt_image_decoder": (
        "stabilityai/stable-video-diffusion-img2vid-xt",
        "svd_xt_image_decoder.safetensors"
    ),
}


def import_model(model_version):
    repo_id, filename = MODEL_VERSION_MAPPING[model_version]
    model_name = model_version
    with bentoml.models.create(
            model_name,
            metadata=dict(filename=filename, model_version=model_version),
            module="bentoml.pytorch",
            context=ModelContext(framework_name="", framework_versions={}),
            signatures={},
    ) as bento_model:
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=bento_model.path_of("/"),
            local_dir_use_symlinks=False,
        )

        return bento_model


def import_clip_model():

    bento_model_name = "clip-vit-h-14-laion"
    filename = "open_clip_pytorch_model.bin"

    try:
        bentoml.models.get(bento_model_name)
    except bentoml.exceptions.NotFound:
        with bentoml.models.create(
                bento_model_name,
                metadata=dict(filename=filename),
                module="bentoml.pytorch",
                context=ModelContext(framework_name="", framework_versions={}),
                signatures={},
        ) as bento_model:
            huggingface_hub.hf_hub_download(
                repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                filename=filename,
                local_dir=bento_model.path_of("/"),
                local_dir_use_symlinks=False,
            )

            return bento_model


if __name__ == "__main__":
    import_model(MODEL_VERSION)
    import_clip_model()
