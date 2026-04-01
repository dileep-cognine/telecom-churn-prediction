from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_proba,
) -> dict[str, float]:
    """Calculate key classification metrics for churn evaluation."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }