from __future__ import annotations

import pandas as pd

from telecom_churn_prediction.settings import settings
from telecom_churn_prediction.logger import get_logger

logger = get_logger(__name__)


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw telecom churn dataset."""
    logger.info("Loading dataset from %s", settings.raw_dataset_path)
    df = pd.read_csv(settings.raw_dataset_path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning required before validation."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna(subset=["TotalCharges"])
    return df