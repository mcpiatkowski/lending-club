"""Evaluation module."""

__all__ = ["evaluate", "plot_roc_curve"]

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(test: pd.Series, pred: pd.Series) -> None:
    """Evaluate model performance."""

    accuracy = accuracy_score(test, pred)
    conf_matrix = confusion_matrix(test, pred)
    class_report = classification_report(test, pred)

    print(f"Accuracy: {accuracy}")

    print("Confusion Matrix:")
    print(conf_matrix)

    print("Classification Report:")
    print(class_report)

    print(f"Precision score is {precision_score(test,pred)}")
    print(f"Recall score is {recall_score(test,pred)}")
    print(f"f1-score is {f1_score(test,pred)}")


def plot_roc_curve(test: pd.Series, prob: pd.Series) -> None:
    fpr, tpr, thresholds = roc_curve(test, prob)
    roc_auc = roc_auc_score(test, prob)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
