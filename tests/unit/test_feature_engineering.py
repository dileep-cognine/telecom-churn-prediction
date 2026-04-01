import pandas as pd

from telecom_churn_prediction.ml.feature_engineering import apply_feature_engineering


def test_apply_feature_engineering_adds_avg_monthly_value() -> None:
    df = pd.DataFrame(
        {
            "tenure": [10, 0],
            "MonthlyCharges": [100.0, 50.0],
        }
    )

    transformed_df = apply_feature_engineering(df)

    assert "AvgMonthlyValue" in transformed_df.columns
    assert transformed_df["AvgMonthlyValue"].iloc[0] == 10.0
    assert transformed_df["AvgMonthlyValue"].iloc[1] == 50.0