from __future__ import annotations

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to the training set only."""
    smote = SMOTE(random_state=random_state)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    return x_resampled, y_resampled