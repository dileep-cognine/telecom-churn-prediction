from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from telecom_churn_prediction.api.application import app
from telecom_churn_prediction.api.dependencies import get_prediction_service


# ---------------------------------------------------------------------
# Sample Payload Fixture
# ---------------------------------------------------------------------
@pytest.fixture
def sample_payload() -> dict[str, Any]:
    """Reusable valid request payload for prediction tests."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }


# ---------------------------------------------------------------------
# Dummy Pipeline Fixture
# ---------------------------------------------------------------------
@pytest.fixture
def dummy_pipeline():
    """Mock pipeline for unit tests."""

    class DummyPipeline:
        def predict_proba(self, X):
            # Always return fixed probability for consistency
            return [[0.2, 0.8] for _ in range(len(X))]

    return DummyPipeline()

# ---------------------------------------------------------------------
# Dummy Prediction Service
# ---------------------------------------------------------------------
class DummyPredictionService:
    """Mock PredictionService for API tests."""

    def predict(self, payload: dict[str, Any]):
        class Result:
            churn_probability = 0.8
            prediction = {
                "label": 1,
                "label_name": "Churn",
                "risk_level": "High",
            }
            model_info = {
                "threshold": 0.4,
            }

        return Result()

@pytest.fixture
def override_prediction_service():
    """Override FastAPI dependency with dummy service."""
    return DummyPredictionService()

# ---------------------------------------------------------------------
# FastAPI Test Client
# ---------------------------------------------------------------------
@pytest.fixture
def client(override_prediction_service) -> TestClient:
    """Provide test client with dependency override."""
    app.dependency_overrides[get_prediction_service] = lambda: override_prediction_service
    return TestClient(app)