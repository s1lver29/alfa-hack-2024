import numpy as np
import polars as pl
from catboost import CatBoostClassifier as CatBoostModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class XGBoostClassifier:
    @staticmethod
    def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
        model = XGBClassifier(**args_model)
        model.fit(data, target)
        return model

    @staticmethod
    def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
        if predict_proba_is:
            return model.predict_proba(data)[:, 1]
        return model.predict(data)

    @staticmethod
    def save_model(model, path: str):
        model.save_model(path + ".json")

    @staticmethod
    def load_model(path: str):
        model = XGBClassifier()
        model.load_model(path)
        return model


class CatBoostClassifier:
    @staticmethod
    def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
        model = CatBoostModel(**args_model)
        model.fit(
            data.to_pandas(use_pyarrow_extension_array=True),
            target.to_pandas(use_pyarrow_extension_array=True),
            verbose=0,
        )
        return model

    @staticmethod
    def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
        if predict_proba_is:
            return model.predict_proba(
                data.to_pandas(use_pyarrow_extension_array=True)
            )[:, 1]
        return model.predict(data.to_pandas(use_pyarrow_extension_array=True))

    @staticmethod
    def save_model(model, path: str):
        model.save_model(path + ".cbm")

    @staticmethod
    def load_model(path: str):
        model = CatBoostModel()
        model.load_model(path)
        return model


class RandomForest:
    @staticmethod
    def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
        model = RandomForestClassifier(**args_model)
        model.fit(data, target.to_numpy().flatten())
        return model

    @staticmethod
    def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
        if predict_proba_is:
            return model.predict_proba(data)[:, 1]
        return model.predict(data)

    @staticmethod
    def save_model(model, path: str):
        import joblib

        joblib.dump(model, path + ".joblib")

    @staticmethod
    def load_model(path: str):
        import joblib

        return joblib.load(path)


class LRegression:
    @staticmethod
    def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
        model = LogisticRegression(**args_model)
        model.fit(data, target)
        return model

    @staticmethod
    def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
        if predict_proba_is:
            return model.predict_proba(data)[:, 1]
        return model.predict(data)


# Обучение и блендинг
def blending_ensemble_train(
    xgb_model,
    cat_model,
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    **args_model,
):
    # Получение предсказаний на обучающем наборе
    xgb_preds_train = xgb_model.predict_proba(X_train)[:, 1]
    cat_preds_train = cat_model.predict_proba(X_train)[:, 1]

    # Объединяем предсказания для мета-модели
    train_meta_features = np.column_stack((xgb_preds_train, cat_preds_train))

    # Обучение мета-модели (SVC)
    meta_model = LRegression.train_model(train_meta_features, y_train, **args_model)

    return meta_model


def blending_ensemble_predict(meta_model, xgb_model, cat_model, X_test):
    # Получение предсказаний базовых моделей на тестовом наборе
    xgb_preds_test = xgb_model.predict_proba(X_test)[:, 1]
    cat_preds_test = cat_model.predict_proba(X_test)[:, 1]

    # Объединяем предсказания тестовых данных для мета-модели
    test_meta_features = np.column_stack((xgb_preds_test, cat_preds_test))

    # Финальные предсказания мета-модели
    final_predictions = LRegression.predict(meta_model, test_meta_features)

    return final_predictions
