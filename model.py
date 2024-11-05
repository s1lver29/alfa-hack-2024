import polars as pl
from xgboost import XGBClassifier


def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
    model = XGBClassifier(**args_model)

    model.fit(data, target)

    return model


def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
    if predict_proba_is:
        return model.predict_proba(data)[:, 1]

    return model.predict(data)
