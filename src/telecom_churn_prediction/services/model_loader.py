from __future__ import annotations

import json
from typing import Any

import joblib

from telecom_churn_prediction.exceptions import ArtifactNotFoundError
from telecom_churn_prediction.settings import settings


def load_model_pipeline() -> Any:
    """Load the trained model pipeline artifact from disk."""
    if not settings.model_path.exists():
        raise ArtifactNotFoundError(
            f"Model artifact not found: {settings.model_path}"
        )

    return joblib.load(settings.model_path)


def load_threshold() -> float:
    """Load the decision threshold artifact from disk."""
    if not settings.threshold_path.exists():
        raise ArtifactNotFoundError(
            f"Threshold artifact not found: {settings.threshold_path}"
        )

    with settings.threshold_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return float(payload["threshold"])