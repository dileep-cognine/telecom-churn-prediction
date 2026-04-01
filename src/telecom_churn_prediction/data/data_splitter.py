from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from telecom_churn_prediction.constants import DROP_COLUMNS, TARGET_COLUMN


def split_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return x, y


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )