from fastapi.testclient import TestClient

from telecom_churn_prediction.api.application import app
from telecom_churn_prediction.api.dependencies import get_prediction_service


class FakePredictionService:
    def predict(self, payload: dict):
        class Result:
            churn_probability = 0.82
            prediction = {
                "label": 1,
                "label_name": "Churn",
                "risk_level": "High",
            }
            model_info = {
                "threshold": 0.4,
            }

        return Result()


def override_prediction_service() -> FakePredictionService:
    return FakePredictionService()


app.dependency_overrides[get_prediction_service] = override_prediction_service
client = TestClient(app)


def test_predict_endpoint_returns_prediction() -> None:
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 1050.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] == 0.82
    assert body["prediction"]["label"] == 1
    assert body["prediction"]["label_name"] == "Churn"
    assert body["prediction"]["risk_level"] == "High"
    assert body["model_info"]["threshold"] == 0.4