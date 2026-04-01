from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from telecom_churn_prediction.ml.metrics import calculate_classification_metrics


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba,
) -> dict[str, float]:
    """Return evaluation metrics for classifier outputs."""
    return calculate_classification_metrics(y_true, y_pred, y_proba)


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    output_path: Path,
) -> None:
    """Save confusion matrix figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(
    y_true,
    y_proba,
    output_path: Path,
) -> None:
    """Save ROC curve figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve_plot(
    y_true,
    y_proba,
    output_path: Path,
) -> None:
    """Save precision-recall curve figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 4))
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution_plot(
    y_true,
    output_path: Path,
) -> None:
    """Save class distribution bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_counts = y_true.value_counts().sort_index()
    labels = ["No Churn", "Churn"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, class_counts.values)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)