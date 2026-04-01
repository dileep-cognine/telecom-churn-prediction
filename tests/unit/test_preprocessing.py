import pandas as pd

from telecom_churn_prediction.ml.preprocessing import build_preprocessing_pipeline


def test_build_preprocessing_pipeline_fits_and_transforms() -> None:
    df = pd.DataFrame(
        {
            "tenure": [1, 12, 24],
            "MonthlyCharges": [20.5, 55.2, 80.1],
            "TotalCharges": [20.5, 662.4, 1922.4],
            "gender": ["Male", "Female", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "PhoneService": ["Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No"],
            "OnlineBackup": ["No", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No"],
            "TechSupport": ["Yes", "No", "No"],
            "StreamingTV": ["No", "Yes", "No"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
        }
    )

    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == 3