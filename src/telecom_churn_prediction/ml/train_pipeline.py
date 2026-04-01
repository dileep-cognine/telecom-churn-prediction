from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from telecom_churn_prediction.data.data_loader import clean_dataset, load_raw_dataset
from telecom_churn_prediction.data.data_splitter import (
    split_features_and_target,
    split_train_test,
)
from telecom_churn_prediction.data.data_validator import validate_dataset
from telecom_churn_prediction.logger import get_logger
from telecom_churn_prediction.ml.metrics import calculate_classification_metrics
from telecom_churn_prediction.ml.model_factory import create_model
from telecom_churn_prediction.ml.preprocessing import build_preprocessing_pipeline
from telecom_churn_prediction.ml.threshold_optimization import find_best_threshold
from telecom_churn_prediction.settings import settings

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model_pipeline: Pipeline
    metrics: dict[str, float]
    threshold: float


def train_model_pipeline(model_name: str = "xgboost") -> TrainingResult:
    """Train the churn prediction pipeline and return training artifacts."""
    df = clean_dataset(load_raw_dataset())
    validate_dataset(df)

    x, y = split_features_and_target(df)
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    preprocessing = build_preprocessing_pipeline()
    model = create_model(model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )

    logger.info("Training model: %s", model_name)
    pipeline.fit(x_train, y_train)

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    threshold, _ = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = calculate_classification_metrics(y_test, y_pred, y_proba)

    return TrainingResult(
        model_pipeline=pipeline,
        metrics=metrics,
        threshold=threshold,
    )


# def save_feature_names(pipeline: Pipeline, output_path: Path) -> None:
#     """Save transformed feature names from the preprocessing pipeline."""
#     preprocessing = pipeline.named_steps["preprocessing"]
#     feature_names = preprocessing.get_feature_names_out().tolist()

#     with output_path.open("w", encoding="utf-8") as file:
#         json.dump(feature_names, file, indent=2)

def save_feature_names(pipeline: Pipeline, output_path: Path) -> None:
    """Save transformed feature names from the preprocessing pipeline."""
    preprocessing = pipeline.named_steps["preprocessing"]
    raw_feature_names = preprocessing.get_feature_names_out()
    feature_names = (
        raw_feature_names.tolist()
        if hasattr(raw_feature_names, "tolist")
        else list(raw_feature_names)
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)

def save_training_artifacts(result: TrainingResult) -> None:
    """Save trained pipeline, threshold, metrics, and feature names."""
    artifacts_dir = settings.model_path.parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    feature_names_path = artifacts_dir / "feature_names.json"
    save_feature_names(result.model_pipeline, feature_names_path)

    joblib.dump(result.model_pipeline, settings.model_path)

    with settings.threshold_path.open("w", encoding="utf-8") as file:
        json.dump({"threshold": result.threshold}, file, indent=2)

    with settings.metrics_path.open("w", encoding="utf-8") as file:
        json.dump(result.metrics, file, indent=2)

    logger.info("Artifacts saved successfully.")