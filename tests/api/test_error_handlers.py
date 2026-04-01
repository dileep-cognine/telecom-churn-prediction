from fastapi import FastAPI
from fastapi.testclient import TestClient

from telecom_churn_prediction.api.error_handlers import register_exception_handlers
from telecom_churn_prediction.exceptions import PredictionError


def test_prediction_error_handler_returns_500() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom():
        raise PredictionError("Prediction failed.")

    client = TestClient(app)
    response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed."}