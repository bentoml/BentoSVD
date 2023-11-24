1. make a virtual environment and run `pip install -U pip && pip install -r requirements.txt` to install dependencies.
2. edit `config.py` to select the default model version.
3. run `python import_model.py` to import the model to BentoML model store.
4. `bentoml serve service.py` ...
5. `bentoml build`
6. deploy
