import pandas as pd

from telecom_churn_prediction.data.data_splitter import (
    split_features_and_target,
    split_train_test,
)


def test_split_features_and_target_returns_expected_outputs() -> None:
    df = pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4"],
            "tenure": [1, 2, 3, 4],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
            "TotalCharges": [10.0, 40.0, 90.0, 160.0],
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "Yes", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["Yes", "No", "Yes", "No"],
            "OnlineBackup": ["No", "Yes", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "No", "Yes"],
            "TechSupport": ["Yes", "No", "Yes", "No"],
            "StreamingTV": ["No", "Yes", "No", "Yes"],
            "StreamingMovies": ["No", "Yes", "No", "Yes"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "Churn": ["Yes", "No", "Yes", "No"],
        }
    )

    x, y = split_features_and_target(df)

    assert "Churn" not in x.columns
    assert "customerID" not in x.columns
    assert list(y) == [1, 0, 1, 0]


def test_split_train_test_returns_all_parts() -> None:
    x = pd.DataFrame({"a": range(10)})
    y = pd.Series([0, 1] * 5)

    x_train, x_test, y_train, y_test = split_train_test(x, y, test_size=0.2)

    assert len(x_train) == 8
    assert len(x_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2