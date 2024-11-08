import numpy as np
import polars as pl
from catboost import CatBoostClassifier as CatBoostModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


class LightGBMClassifier:
    @staticmethod
    def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
        model = LGBMClassifier(**args_model)
        model.fit(data, target)
        return model

    @staticmethod
    def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
        if predict_proba_is:
            return model.predict_proba(data)[:, 1]
        return model.predict(data)


class LogReg:
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
    lgbm_model,
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    **args_model,
):
    # Получение предсказаний на обучающем наборе
    xgb_preds_train = xgb_model.predict_proba(X_train)[:, 1]
    cat_preds_train = cat_model.predict_proba(X_train)[:, 1]
    lgbm_pred_train = lgbm_model.predict_proba(X_train)[:, 1]

    standart_scaler = StandardScaler()
    X_train_standart_scaler = standart_scaler.fit_transform(X_train)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_standart_scaler)

    # Объединяем предсказания для мета-модели
    predict_base_models = np.column_stack(
        (xgb_preds_train, cat_preds_train, lgbm_pred_train)
    )

    train_meta_features = np.column_stack((predict_base_models, X_train_pca))

    meta_model = LogReg.train_model(train_meta_features, y_train, **args_model)

    return meta_model, standart_scaler, pca


def blending_ensemble_predict(
    pca, standart_scaler, meta_model, xgb_model, lgbm_model, cat_model, X_test
):
    X_test_pca = pca.transform(standart_scaler.transform(X_test))

    # Получение предсказаний базовых моделей на тестовом наборе
    xgb_preds_test = xgb_model.predict_proba(X_test)[:, 1]
    cat_preds_test = cat_model.predict_proba(X_test)[:, 1]
    lgbm_preds_test = lgbm_model.predict_proba(X_test)[:, 1]

    # Объединяем предсказания тестовых данных для мета-модели
    predict_base_models = np.column_stack(
        (xgb_preds_test, cat_preds_test, lgbm_preds_test)
    )

    test_meta_features = np.column_stack((predict_base_models, X_test_pca))

    # Финальные предсказания мета-модели
    final_predictions = LogReg.predict(meta_model, test_meta_features)

    return final_predictions
