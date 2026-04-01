from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input payload for churn prediction."""

    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionDetails(BaseModel):
    """Human-readable prediction result."""

    label: int
    label_name: str
    risk_level: str


class ModelInfo(BaseModel):
    """Model metadata included in the API response."""

    # threshold: float
    threshold: float = Field(..., description="Decision threshold used for churn classification.")


class PredictionResponse(BaseModel):
    """Structured churn prediction API response."""

    churn_probability: float
    prediction: PredictionDetails
    model_info: ModelInfo


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str