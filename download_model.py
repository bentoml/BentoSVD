import huggingface_hub

from model_version import model_version

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


if __name__ == "__main__":
    repo_id, filename = MODEL_VERSION_MAPPING[model_version]
    huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="checkpoints/",
        local_dir_use_symlinks=False
    )
