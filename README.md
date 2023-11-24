1. make a virtual environment and run `pip install -U pip && pip install -r requirements.txt` to install dependencies.
2. edit `model_version.py` to select the default model version.
3. run `python download_model.py` to download the default model to `checkpoints/`
4. `bentoml serve service.py` ...
5. build bento ...
6. deploy