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

@router.get("/")
def predict():
    return {"message":"Welcome to Churn-Prediction"}

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API health status for monitoring and readiness checks."""
    status = get_health_status()
    return HealthResponse(status=status.status, message=status.message)


@router.post("/predict", response_model=PredictionResponse)
def predict_churn(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """
    Generate churn prediction for a single customer record.

    Returns:
        PredictionResponse containing probability, readable prediction details,
        and model metadata such as decision threshold.
    """
    result = service.predict(request.model_dump())

    return PredictionResponse(
        churn_probability=result.churn_probability,
        prediction=PredictionDetails(**result.prediction),
        model_info=ModelInfo(**result.model_info),
    )