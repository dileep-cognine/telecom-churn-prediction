from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(
    y_true,
    y_proba,
    thresholds: np.ndarray | None = None,
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)