import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from telecom_churn_prediction.ml.model_factory import create_model


def test_create_model_returns_logistic_regression() -> None:
    model = create_model("logistic_regression")
    assert isinstance(model, LogisticRegression)


def test_create_model_returns_random_forest() -> None:
    model = create_model("random_forest")
    assert isinstance(model, RandomForestClassifier)


def test_create_model_raises_for_invalid_name() -> None:
    with pytest.raises(ValueError):
        create_model("invalid_model")