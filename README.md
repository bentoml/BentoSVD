## Introduction

[Stable Video Diffusion (SVD)](https://stability.ai/news/stable-video-diffusion-open-ai-video-model) is a foundation model for generative video based on the image model Stable Diffusion. It comes in the form of two primary image-to-video models, SVD and SVD-XT, capable of generating 14 and 25 frames at customizable frame rates between 3 and 30 frames per second.

This sample project is designed to streamline the process of serving and deploying SVD models in production through BentoML, eliminating the setup and configuration complexity with such models.

This project supports the following SVD models:

- [stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/svd.safetensors): Generates video sequences of 14 frames.
- [stable-video-diffusion-img2vid-decoder](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/svd_image_decoder.safetensors): Generates video sequences of 14 frames using the standard frame-wise decoder.
- [stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/svd_xt.safetensors): Generates video sequences of 25 frames.
- [stable-video-diffusion-img2vid-xt-decoder](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/svd_xt_image_decoder.safetensors): Generates video sequences of 25 frames using the standard frame-wise decoder.

## Get started

### Prerequisites

You have installed Python 3.8 (or later) and `pip`.

### Set up the environment

Clone the GitHub repository and navigate to the project directory.

```bash
git clone https://github.com/bentoml/bentoml-svd
cd bentoml-svd
```

For dependency isolation, we suggest you create a virtual environment.

```bash
python -m venv test
source test/bin/activate
```

Install the required dependencies.

```bash
pip install -U pip && pip install -r requirements.txt
```

### Start a local SVD server

Edit the `config.py` file, where you specify the model to download and use for launching the server later.

```bash
vi config.py
```

Set your desired model version. It defaults to `svd`.

```bash
# Allowed values for MODEL_VERSION include "svd", "svd_xt", "svd_image_decoder", and "svd_xt_image_decoder"
MODEL_VERSION = "svd"
```

Run the script to download the model specified.

```bash
python import_model.py
```

The model is saved to the BentoML Model Store, a centralized repository for managing models. Run `bentoml models list` to view all available models locally.

```bash
$ bentoml models list

Tag                      Module           Size      Creation Time
svd_xt:ulk4jzem2suhrv2a  bentoml.pytorch  8.90 GiB  2023-11-27 10:25:38
svd:anbmkjem7g5fqo4n     bentoml.pytorch  8.90 GiB  2023-11-27 07:46:02
```

Run `bentoml serve` to start a server locally, which is powered by the model defined in `config.py`. The server is accessible at http://0.0.0.0:3000.

```bash
$ bentoml serve service:svc

2023-11-27T03:37:46+0000 [INFO] [cli] Environ for worker 0: set CUDA_VISIBLE_DEVICES to 0
2023-11-27T03:37:46+0000 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:svc" can be accessed at http://localhost:3000/metrics.
2023-11-27T03:37:47+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:svc" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

To interact with the server, send a request via `curl`. Replace `test_image.png` with your own image.

```bash
curl -X 'POST' \
  'http://0.0.0.0:3000/generate' \
  -H 'accept: video/mp4' \
  -H 'Content-Type: multipart/form-data' \
  -F 'img=@test_image.png;type=image/png' \
  -F 'params={
  "fps_id": 6,
  "motion_bucket_id": 127,
  "cond_aug": 0.02,
  "decoding_t": 1,
  "seed": null
}' --output generated_video.mp4
```

This is the image used in the request:

![sample](/assets/sample.png)

Returned output (leaves blowing and cloud moving):

![output-image](/assets/output.gif)

### Build a Bento

A [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html) in BentoML is a deployable artifact including all the source code, models, data files, and dependencies. Once a Bento is built, you can containerize it as a Docker image or distribute it on BentoCloud for better management, scalability and observability.

The `bentofile.yaml` file required to build a Bento is ready available in the project directory with some basic configurations, while you can also customize it as needed. See [Bento build options](https://docs.bentoml.com/en/latest/concepts/bento.html#bento-build-options) to learn more.

To build a Bento, run:

```bash
$ bentoml build

██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ███╗   ███╗██╗
██╔══██╗██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║██║
██████╔╝█████╗  ██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║██║
██╔══██╗██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║██║
██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝

Successfully built Bento(tag="svd-service:csqechem7w5fqo4n").

Possible next steps:

 * Containerize your Bento with `bentoml containerize`:
    $ bentoml containerize svd-service:csqechem7w5fqo4n  [or bentoml build --containerize]

 * Push to BentoCloud with `bentoml push`:
    $ bentoml push svd-service:csqechem7w5fqo4n [or bentoml build --push]
```

To build a Docker image for the Bento, run:

```bash
bentoml containerize BENTO_TAG
```

To push the Bento to BentoCloud, [log in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html) first and run the following command:

```bash
bentoml push BENTO_TAG
```

You can then deploy the Bento on BentoCloud.

## Contribution

We welcome contributions of all kinds to the example project! Check out the following resources to stay tuned for more example projects and announcements about BentoML.

- BentoML [gallery](https://www.bentoml.com/gallery)
- Join the [BentoML community on Slack](https://l.bentoml.com/join-slack)
- Follow us on [Twitter](https://twitter.com/bentomlai) and [Linkedin](https://www.linkedin.com/company/bentoml/)