from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import numpy as np


def calculate_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    average: str = "binary",
) -> dict[str, float]:
    """
    Вычисляет precision, recall, f1-score и support для бинарной или многоклассовой классификации.

    :param y_true: Истинные метки классов.
    :param y_pred: Предсказанные метки классов.
    :param average: Стратегия усреднения для многоклассового случая ('binary', 'micro', 'macro', 'weighted').
    :return: Словарь с метриками precision, recall, f1-score и support.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "roc_auc_score": roc_auc,
    }

    return metrics


def calculation_confusion_matrix(
    y_true: np.ndarray | list[int], y_pred: np.ndarray | list[int], labels: str = None
):
    """
    Отображает confusion matrix.

    :param y_true: Истинные метки классов.
    :param y_pred: Предсказанные метки классов.
    :param labels: Список имен классов (опционально).
    """
    cm = confusion_matrix(y_true, y_pred)

    return cm


def plot_roc_curve(y_true: list[int] | np.ndarray, y_scores: list[float] | np.ndarray):
    """
    Строит ROC-кривую.

    :param y_true: Истинные метки классов (для бинарной классификации).
    :param y_scores: Вероятности положительного класса или оценки для ROC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")

    return plt
