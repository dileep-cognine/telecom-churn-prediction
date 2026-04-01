from pathlib import Path

import pandas as pd

from telecom_churn_prediction.data.data_loader import clean_dataset


def test_clean_dataset_converts_total_charges_and_drops_invalid_rows() -> None:
    df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "TotalCharges": ["100.5", " "],
        }
    )

    cleaned_df = clean_dataset(df)

    assert len(cleaned_df) == 1
    assert cleaned_df["TotalCharges"].iloc[0] == 100.5