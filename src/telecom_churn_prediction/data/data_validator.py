from __future__ import annotations

import pandas as pd

from telecom_churn_prediction.constants import (
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
)
from telecom_churn_prediction.exceptions import DatasetValidationError


def validate_dataset(df: pd.DataFrame) -> None:
    expected_columns = set(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DROP_COLUMNS + [TARGET_COLUMN]
    )
    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        raise DatasetValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    if not unique_targets.issubset({"Yes", "No"}):
        raise DatasetValidationError(
            f"Unexpected target values: {sorted(unique_targets)}"
        )