import pandas as pd
import pytest

from telecom_churn_prediction.data.data_validator import validate_dataset
from telecom_churn_prediction.exceptions import DatasetValidationError


def test_validate_dataset_raises_for_missing_columns() -> None:
    df = pd.DataFrame({"foo": [1], "bar": [2]})

    with pytest.raises(DatasetValidationError):
        validate_dataset(df)