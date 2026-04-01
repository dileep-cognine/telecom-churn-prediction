import numpy as np

from telecom_churn_prediction.ml.threshold_optimization import find_best_threshold


def test_find_best_threshold_returns_valid_values() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.7, 0.4, 0.9])

    threshold, score = find_best_threshold(y_true, y_proba)

    assert 0.1 <= threshold <= 0.85
    assert 0.0 <= score <= 1.0