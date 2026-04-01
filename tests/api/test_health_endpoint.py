from fastapi.testclient import TestClient

from telecom_churn_prediction.api.application import app

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Service is healthy.",
    }