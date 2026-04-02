from __future__ import annotations

from fastapi import APIRouter, Depends

from telecom_churn_prediction.api.dependencies import get_prediction_service
from telecom_churn_prediction.api.schemas import (
    HealthResponse,
    ModelInfo,
    PredictionDetails,
    PredictionRequest,
    PredictionResponse,
)
from telecom_churn_prediction.services.health_service import get_health_status
from telecom_churn_prediction.services.prediction_service import PredictionService

router = APIRouter()

# @router.get("/")
# def predict():
#     return {"message":"Welcome to Churn-Prediction"}
@router.get("/")
def root() -> dict[str, object]:
    """Return a simple human-friendly overview of the application."""
    return {
        "application": "Telecom Churn Prediction API",
        "purpose": (
            "This application predicts whether a telecom customer is likely "
            "to churn and returns a human-friendly risk assessment."
        ),
        "main_endpoint": "/predict",
        "health_check": "/health",
        "documentation": "/docs",
        "sample_output": {
            "summary": "This customer is at high risk of churn.",
            "churn_probability": 0.9348,
            "prediction": {
                "label": 1,
                "label_name": "Churn",
                "risk_level": "High"
            },
            "model_info": {
                "threshold": 0.4
            }
        }
    }

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API health status for monitoring and readiness checks."""
    status = get_health_status()
    return HealthResponse(status=status.status, message=status.message)


# @router.post("/predict", response_model=PredictionResponse)
# def predict_churn(
#     request: PredictionRequest,
#     service: PredictionService = Depends(get_prediction_service),
# ) -> PredictionResponse:
#     """
#     Generate churn prediction for a single customer record.

#     Returns:
#         PredictionResponse containing probability, readable prediction details,
#         and model metadata such as decision threshold.
#     """
#     result = service.predict(request.model_dump())

#     return PredictionResponse(
#         churn_probability=result.churn_probability,
#         prediction=PredictionDetails(**result.prediction),
#         model_info=ModelInfo(**result.model_info),
#     )

@router.post("/predict", response_model=PredictionResponse)
def predict_churn(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """
    Generate churn prediction for a single customer record.

    Returns:
        PredictionResponse containing a plain-English summary,
        readable prediction details, and model metadata.
    """
    result = service.predict(request.model_dump())

    return PredictionResponse(
        summary=result.summary,
        churn_probability=result.churn_probability,
        prediction=PredictionDetails(**result.prediction),
        model_info=ModelInfo(**result.model_info),
    )