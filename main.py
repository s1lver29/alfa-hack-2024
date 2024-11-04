import hydra
import joblib
import polars as pl
from clearml import Logger, Task
from omegaconf import DictConfig
import optuna

from metrics import calculate_classification_metrics, calculation_confusion_matrix
from model import predict, train_model
from preprocessing import load_data


class MLWorkflow:
    def __init__(self, config: DictConfig):
        self.config = config
        self.task = self.init_clearml_task()
        self.logger = Logger.current_logger()
        self.model = None

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
        self.model = train_model(
            data=data["train"].drop(self.config.dataset_train.target_columns),
            target=data["train"].select(self.config.dataset_train.target_columns),
            **self.config.model,
        )

    def evaluate_model(self, data):
        predictions = predict(
            self.model,
            data["test"].drop(self.config.dataset_train.target_columns),
            predict_proba_is=False,
        )
        metrics = calculate_classification_metrics(
            data["test"].select(self.config.dataset_train.target_columns), predictions
        )
        conf_matrix = calculation_confusion_matrix(
            data["test"].select(self.config.dataset_train.target_columns), predictions
        )

        # Логирование метрик и матрицы ошибок
        self.logger.report_text(metrics)
        self.logger.report_confusion_matrix(
            title="Confusion matrix", series="ignored", matrix=conf_matrix
        )

    def optimize_hyperparameters(self, train_data, val_data):
        """Оптимизация гиперпараметров с использованием Optuna."""

        def objective(trial):
            params = {
                "C": trial.suggest_loguniform("C", 1e-3, 1e1),
                "random_state": self.config.model.get("random_state", 42),
            }

            model = train_model(
                data=train_data.drop(self.config.dataset_train.target_columns),
                target=train_data.select(self.config.dataset_train.target_columns),
                **params,
            )

            predictions = predict(
                model,
                val_data.drop(self.config.dataset_train.target_columns),
                predict_proba_is=False,
            )
            metrics = calculate_classification_metrics(
                val_data.select(self.config.dataset_train.target_columns), predictions
            )

            for name, score in metrics.items():
                self.logger.report_scalar(
                    title=name, series="series", value=score, iteration=trial.number
                )

            return metrics["f1-score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.hyperparameters.n_trials)

        self.logger.report_text(f"Best hyperparameters: {study.best_params}")

    def predict_on_test_data(self):
        """Выполнение предсказаний на тестовом наборе данных."""
        test_data = self.load_data(dataset_type="test")
        predictions = predict(self.model, test_data.drop("id"), predict_proba_is=True)
        predictions_df = pl.DataFrame(
            {"id": test_data["id"], "prediction": predictions}
        )

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
            # Разделение данных на train, validation, test
            train_data, val_data = data["train"], data["test"]
            self.optimize_hyperparameters(train_data, val_data)

            # Обучаем на train+validation с лучшими параметрами
            # train_val_data = pl.concat([train_data, val_data])
            # self.train_with_best_params(train_val_data)

        else:
            if self.config.train_test_split_is:
                self.logger.report_text("Train model and evaluate on validation data")
                self.train(data)
                self.evaluate_model(data)
            else:
                self.logger.report_text("Training model on full dataset")
                self.train(data)

            if self.config.save_model:
                self.logger.report_text("Save model")
                joblib.dump(self.model, self.config.save_model, compress=True)
                # self.task.upload_artifact(name="trained_model", artifact_object=self.model)

            # Предсказания и сохранение, если указано
            if self.config.predict_and_save_test_data:
                self.logger.report_text("Prediction test data and save prediction")
                predictions_df = self.predict_on_test_data()
                self.save_submission(predictions_df)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    workflow = MLWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
