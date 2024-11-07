import numpy as np
import polars as pl
from catboost import CatBoostClassifier as CatBoostModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
        model.save_model(path)

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
        model.save_model(path)

    @staticmethod
    def load_model(path: str):
        model = CatBoostModel()
        model.load_model(path)
        return model


class RandomForest:
    @staticmethod
    def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
        model = RandomForestClassifier(**args_model)
        model.fit(data, target)
        return model

    @staticmethod
    def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
        if predict_proba_is:
            return model.predict_proba(data)[:, 1]
        return model.predict(data)

    @staticmethod
    def save_model(model, path: str):
        import joblib

        joblib.dump(model, path)

    @staticmethod
    def load_model(path: str):
        import joblib

        return joblib.load(path)


class SVMClassifier:
    @staticmethod
    def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
        model = SVC(**args_model)
        model.fit(data, target)
        return model

    @staticmethod
    def predict(model, data: pl.DataFrame):
        return model.predict(data)


# Обучение и блендинг
def blending_ensemble(
    X_train: pl.DataFrame, y_train: pl.DataFrame, X_test: pl.DataFrame
):
    # Обучение базовых моделей
    xgb_model = XGBoostClassifier.train_model(X_train, y_train, n_estimators=100)
    cat_model = CatBoostClassifier.train_model(
        X_train, y_train, iterations=100, depth=6
    )
    rf_model = RandomForest.train_model(X_train, y_train, n_estimators=100)

    # Получение предсказаний на обучающем наборе
    xgb_preds_train = XGBoostClassifier.predict(xgb_model, X_train)
    cat_preds_train = CatBoostClassifier.predict(cat_model, X_train)
    rf_preds_train = RandomForest.predict(rf_model, X_train)

    # Объединяем предсказания для мета-модели
    train_meta_features = np.column_stack(
        (xgb_preds_train, cat_preds_train, rf_preds_train)
    )

    # Обучение мета-модели (SVC)
    meta_model = SVMClassifier.train_model(pl.DataFrame(train_meta_features), y_train)

    # Получение предсказаний базовых моделей на тестовом наборе
    xgb_preds_test = XGBoostClassifier.predict(xgb_model, X_test)
    cat_preds_test = CatBoostClassifier.predict(cat_model, X_test)
    rf_preds_test = RandomForest.predict(rf_model, X_test)

    # Объединяем предсказания тестовых данных для мета-модели
    test_meta_features = np.column_stack(
        (xgb_preds_test, cat_preds_test, rf_preds_test)
    )

    # Финальные предсказания мета-модели
    final_predictions = SVMClassifier.predict(
        meta_model, pl.DataFrame(test_meta_features)
    )
    return final_predictions
