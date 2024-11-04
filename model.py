from sklearn.linear_model import LogisticRegression
import polars as pl


def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
    model = LogisticRegression(**args_model)

    model.fit(data, target)

    return model


def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
    if predict_proba_is:
        return model.predict_proba(data)[:, 1]

    return model.predict(data)
