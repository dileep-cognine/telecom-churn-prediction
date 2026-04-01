import numpy as np

from telecom_churn_prediction.services.prediction_service import PredictionService


class DummyPipeline:
    def predict_proba(self, input_df):
        return np.array([[0.2, 0.8]])


def test_prediction_service_returns_prediction(monkeypatch) -> None:
    monkeypatch.setattr(
        "telecom_churn_prediction.services.prediction_service.load_model_pipeline",
        lambda: DummyPipeline(),
    )
    monkeypatch.setattr(
        "telecom_churn_prediction.services.prediction_service.load_threshold",
        lambda: 0.5,
    )

    service = PredictionService()
    result = service.predict({"tenure": 12})

    assert result.churn_probability == 0.8
    assert result.prediction["label"] == 1
    assert result.prediction["label_name"] == "Churn"
    assert result.prediction["risk_level"] == "High"
    assert result.model_info["threshold"] == 0.5