from __future__ import annotations

from telecom_churn_prediction.services.prediction_service import PredictionService


def main() -> None:
    service = PredictionService()

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

    result = service.predict(payload)
    print(result)


if __name__ == "__main__":
    main()