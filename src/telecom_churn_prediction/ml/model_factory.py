from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def create_model(model_name: str):
    """Create and return a configured classifier by model name."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "naive_bayes":
        return GaussianNB()

    if model_name == "svm":
        return SVC(
            probability=True,
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            class_weight="balanced",
            random_state=42,
        )

    raise ValueError(f"Unsupported model name: {model_name}")