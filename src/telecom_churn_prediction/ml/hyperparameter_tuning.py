from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from telecom_churn_prediction.ml.model_factory import create_model
from telecom_churn_prediction.ml.preprocessing import build_preprocessing_pipeline


def build_smote_xgboost_pipeline() -> ImbPipeline:
    """Build a preprocessing + SMOTE + XGBoost pipeline."""
    preprocessing = build_preprocessing_pipeline()
    model = create_model("xgboost")

    return ImbPipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ]
    )


def tune_xgboost_with_cv(x_train, y_train) -> GridSearchCV:
    """Tune XGBoost hyperparameters using stratified cross-validation."""
    pipeline = build_smote_xgboost_pipeline()

    param_grid: dict[str, list[Any]] = {
        "model__max_depth": [4, 6, 8],
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search