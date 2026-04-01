from telecom_churn_prediction.ml.metrics import calculate_classification_metrics


def test_calculate_classification_metrics_returns_expected_keys() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    y_proba = [0.1, 0.8, 0.9, 0.2]

    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "roc_auc",
    }