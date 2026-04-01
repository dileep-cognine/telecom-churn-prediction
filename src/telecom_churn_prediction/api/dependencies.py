from telecom_churn_prediction.services.prediction_service import PredictionService


def get_prediction_service() -> PredictionService:
    return PredictionService()