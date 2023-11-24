# model verson can be "svd", "svd_xt", "svd_image_decoder", "svd_xt_image_decoder"
MODEL_VERSION = "svd"

DEFAULT_CONFIGS = {
    "svd": dict(num_frames=14, num_steps=25),
    "svd_image_decoder": dict(num_frames=14, num_steps=25),
    "svd_xt": dict(num_frames=25, num_steps=30),
    "svd_xt_image_decoder": dict(num_frames=25, num_steps=30),
}
