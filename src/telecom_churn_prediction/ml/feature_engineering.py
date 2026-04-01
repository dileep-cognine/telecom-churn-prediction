from __future__ import annotations

import pandas as pd


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight feature engineering.

    The IBM telco dataset already contains mostly usable features,
    so this function keeps transformations minimal and reproducible.
    """
    transformed_df = df.copy()

    if "tenure" in transformed_df.columns and "MonthlyCharges" in transformed_df.columns:
        transformed_df["AvgMonthlyValue"] = (
            transformed_df["MonthlyCharges"] / transformed_df["tenure"].replace(0, 1)
        )

    return transformed_df