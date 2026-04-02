from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from telecom_churn_prediction.exceptions import PredictionError
from telecom_churn_prediction.services.model_loader import (
    load_model_pipeline,
    load_threshold,
)


@dataclass
class PredictionResult:
    """Structured prediction output for a single customer."""

    summary: str
    churn_probability: float
    prediction: dict[str, Any]
    model_info: dict[str, Any]


def _get_risk_level(probability: float) -> str:
    """Map churn probability to a human-readable risk level."""
    if probability < 0.3:
        return "Low"
    if probability < 0.6:
        return "Medium"
    return "High"


def _build_summary(predicted_label: int, risk_level: str) -> str:
    """Create a short human-friendly summary of the prediction."""
    if predicted_label == 1:
        return f"This customer is at {risk_level.lower()} risk of churn."
    return f"This customer is at {risk_level.lower()} risk and is unlikely to churn."


class PredictionService:
    """Load trained artifacts and generate churn predictions."""

    def __init__(self) -> None:
        self.pipeline = load_model_pipeline()
        self.threshold = load_threshold()

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """
        Generate churn prediction for a single customer record.

        Returns:
            PredictionResult containing a summary, probability,
            readable prediction details, and model metadata.
        """
        try:
            input_df = pd.DataFrame([payload])
            churn_probability = round(
                float(self.pipeline.predict_proba(input_df)[:, 1][0]),
                4,
            )
            predicted_label = int(churn_probability >= self.threshold)

            label_name = "Churn" if predicted_label == 1 else "No Churn"
            risk_level = _get_risk_level(churn_probability)
            summary = _build_summary(predicted_label, risk_level)

            return PredictionResult(
                summary=summary,
                churn_probability=churn_probability,
                prediction={
                    "label": predicted_label,
                    "label_name": label_name,
                    "risk_level": risk_level,
                },
                model_info={
                    "threshold": round(self.threshold, 4),
                },
            )
        except Exception as exc:
            raise PredictionError(f"Prediction failed: {exc}") from exc