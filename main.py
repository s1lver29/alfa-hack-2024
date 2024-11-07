import hydra
import numpy as np
import optuna
import polars as pl
from clearml import Logger, Task
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from metrics import calculate_classification_metrics, calculation_confusion_matrix
from model import (
    XGBoostClassifier,
    RandomForest,
    SVMClassifier,
    CatBoostClassifier,
    LGBMClassifier,
    blending_ensemble_train,
    blending_ensemble_predict,
)
from preprocessing import load_data
import pickle


class MLWorkflow:
    def __init__(self, config: DictConfig):
        self.config = config
        self.task = self.init_clearml_task()
        self.logger = Logger.current_logger()
        self.model = None
        self.type_models = LGBMClassifier()

    def init_clearml_task(self) -> Task:
        """Инициализация задачи ClearML и логирование гиперпараметров."""
        task = Task.init(
            project_name=self.config.clearml.project_name,
            task_name=self.config.clearml.task_name,
        )
        task.connect(self.config)
        return task

    def load_data(self, dataset_type: str = "train"):
        """Загрузка данных (train или test)."""
        dataset_config = (
            self.config.dataset_train
            if dataset_type == "train"
            else self.config.dataset_test
        )
        return load_data(
            dataset_project=dataset_config.project_name,
            dataset_name=dataset_config.dataset_name,
            name_file=dataset_config.file_name,
            train_test_split_is=(
                dataset_type == "train" and self.config.train_test_split_is
            ),
            target=(
                self.config.dataset_train.target_columns
                if dataset_type == "train"
                else None
            ),
        )

    def train(self, data):
        """Обучение модели"""
        self.model = self.type_models.train_model(
            data=data.drop(self.config.dataset_train.target_columns),
            target=data.select(self.config.dataset_train.target_columns),
            **self.config.model,
        )

    def evaluate_model(self, data):
        predictions = self.type_models.predict(
            self.model,
            data.drop(self.config.dataset_train.target_columns),
            predict_proba_is=False,
        )
        predict_proba = self.type_models.predict(
            self.model,
            data.drop(self.config.dataset_train.target_columns),
            predict_proba_is=True,
        )

        metrics = calculate_classification_metrics(
            data.select(self.config.dataset_train.target_columns),
            predictions,
            predict_proba,
        )
        conf_matrix = calculation_confusion_matrix(
            data.select(self.config.dataset_train.target_columns), predictions
        )

        # Логирование метрик и матрицы ошибок
        self.logger.report_text(metrics)
        self.logger.report_confusion_matrix(
            title="Confusion matrix", series="ignored", matrix=conf_matrix
        )

    def optimize_hyperparameters(
        self, train_data, val_data, n_splits: int = 3, random_state: int = 42
    ):
        """Оптимизация гиперпараметров с использованием Optuna."""
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        train_data, train_target = (
            train_data.drop(self.config.dataset_train.target_columns),
            train_data.select(self.config.dataset_train.target_columns),
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 0.00001, 1.5
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
                "class_weight": "balanced",
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 5, log=True),
                "random_state": trial.suggest_int("random_state", 1, 100),
                "device": "gpu",
            }

            fold_metrics = {
                "f1-score": [],
                "precision": [],
                "recall": [],
                "roc_auc_score": [],
            }

            for train_index, val_index in skf.split(train_data, train_target):
                X_train, X_val = train_data[train_index], train_data[val_index]
                y_train, y_val = train_target[train_index], train_target[val_index]

                model = self.type_models.train_model(X_train, y_train, **params)

                y_pred = self.type_models.predict(model, X_val, predict_proba_is=False)
                y_pred_proba = self.type_models.predict(
                    model, X_val, predict_proba_is=True
                )

                metrics = calculate_classification_metrics(y_val, y_pred, y_pred_proba)

                for name, score in metrics.items():
                    fold_metrics[name].append(score)

            for name, scores in fold_metrics.items():
                scores = np.array(scores)
                self.logger.report_scalar(
                    title="Mean metrics CV fold",
                    series=f"Mean {name}",
                    value=scores.mean(),
                    iteration=trial.number,
                )
                self.logger.report_scalar(
                    title="Mean metrics CV fold",
                    series=f"Std {name}",
                    value=scores.std(),
                    iteration=trial.number,
                )

            return np.array(fold_metrics["roc_auc_score"]).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.hyperparameters.n_trials)

        self.logger.report_text(f"Best hyperparameters: {study.best_params}")

        return study.best_params

    def predict_on_test_data(self):
        """Выполнение предсказаний на тестовом наборе данных."""
        test_data = self.load_data(dataset_type="test")
        predictions = self.type_models.predict(
            self.model, test_data.drop("id"), predict_proba_is=True
        )
        predictions_df = pl.DataFrame({"id": test_data["id"], "target": predictions})

        return predictions_df

    def save_submission(self, predictions_df):
        """Сохранение предсказаний и добавление в ClearML как артефакт."""
        predictions_df.write_csv("predictions.csv")
        self.task.upload_artifact(name="submission", artifact_object="predictions.csv")
        self.logger.report_table(
            title="Submission",
            series="Predictions",
            table_plot=predictions_df.head().to_pandas(
                use_pyarrow_extension_array=True
            ),
        )

    def run(self):
        """Запуск обучения, оценки, предсказания и сохранения."""
        if self.config.clearml.queue:
            self.task.execute_remotely(
                queue_name=self.config.clearml.queue, exit_process=True
            )
        else:
            self.logger.report_text("Starting local training")

        data = self.load_data()

        if self.config.hyperparameters_optimization_is:
            train_data, val_data = data["train"], data["test"]
            best_params = self.optimize_hyperparameters(train_data, val_data)

            # Обучаем с лучшими параметрами
            self.model = self.type_models.train_model(
                data=train_data.drop(self.config.dataset_train.target_columns),
                target=train_data.select(self.config.dataset_train.target_columns),
                **best_params,
            )
            self.evaluate_model(val_data)

        else:
            if self.config.blending:
                self.logger.report_text("Start blending")
                print(list(data.keys()))
                data = data["test"]

                task_1 = Task.get_task(
                    project_name="Alfa_hack",
                    task_name="Blending XGBoost train full_dataset 07.11.24 v1.0",
                )
                with open(
                    task_1.artifacts["xgboost_weights"].get_local_copy(), "rb"
                ) as file:
                    model_1 = pickle.load(file)

                task_2 = Task.get_task(
                    project_name="Alfa_hack",
                    task_name="Blending Catboost train full_dataset 07.11.24 v1.0",
                )
                with open(
                    task_2.artifacts["catboost_weights"].get_local_copy(), "rb"
                ) as file:
                    model_2 = pickle.load(file)

                self.logger.report_text("Train blending")
                meta_model = blending_ensemble_train(
                    model_1,
                    model_2,
                    data.drop(self.config.dataset_train.target_columns).to_pandas(
                        use_pyarrow_extension_array=True
                    ),
                    data.select(self.config.dataset_train.target_columns).to_pandas(
                        use_pyarrow_extension_array=True
                    ),
                    **self.config.model,
                )

                del data
            else:
                if self.config.train_test_split_is:
                    self.logger.report_text(
                        "Train model and evaluate on validation data"
                    )
                    self.train(data["train"])
                    self.evaluate_model(data["test"])
                else:
                    self.logger.report_text("Training model on full dataset")
                    self.train(data)

            if self.config.save_model:
                self.logger.report_text("Save model")
                self.task.upload_artifact(
                    name=self.config.save_model, artifact_object=self.model
                )

            # Предсказания и сохранение, если указано
            if self.config.predict_and_save_test_data:
                self.logger.report_text("Prediction test data and save prediction")
                if self.config.blending:
                    test_data = self.load_data(dataset_type="test")

                    predictions = blending_ensemble_predict(
                        meta_model,
                        model_1,
                        model_2,
                        test_data.drop("id").to_pandas(
                            use_pyarrow_extension_array=True
                        ),
                    )

                    predictions_df = pl.DataFrame(
                        {"id": test_data["id"], "target": predictions}
                    )
                else:
                    predictions_df = self.predict_on_test_data()
                self.save_submission(predictions_df)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    workflow = MLWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
