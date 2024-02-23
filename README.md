<div align="center">
    <h1 align="center">BentoSVD</h1>
    <br>
    <strong>Serve and deploy Stable Video Diffusion (SVD) models in production without any setup hassles.<br></strong>
    <i>Powered by BentoML 🍱</i>
    <br>
</div>
<br>

Stable Video Diffusion (SVD) is a foundation model for generative video based on the image model Stable Diffusion. It comes in the form of two primary image-to-video models, SVD and SVD-XT, capable of generating 14 and 25 frames at customizable frame rates between 3 and 30 frames per second.

This project is designed to streamline the process of serving and deploying SVD models in production, eliminating the setup and configuration complexity with such models. It builds a video generation application using BentoML, powered by [diffusers](https://github.com/bentoml/BentoSD2Upscaler) and [Stable Video Diffusion (SVD)](https://stability.ai/news/stable-video-diffusion-open-ai-video-model).

## Prerequisites

- You have installed Python 3.9+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.
- If you want to test the Service locally, a Nvidia GPU with 16G VRAM is required.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoSVD.git
cd BentoSVD
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-19T07:29:04+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SVDService" listening on http://localhost:3000 (Press CTRL+C to quit)
Loading pipeline components...: 100%
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: */*' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/sample.png;type=image/png' \
  -F 'decode_chunk_size=2' \
  -F 'seed=null' \
  -o generated.mp4
```

Python client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000/") as client:
    result = client.generate(
        decode_chunk_size=2,
        image=@assets/sample.png,
        seed=0,
    )
```

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
