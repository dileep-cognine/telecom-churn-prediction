import json
from pathlib import Path

from telecom_churn_prediction.ml.train_pipeline import (
    TrainingResult,
    save_training_artifacts,
)


class DummyPreprocessor:
    def get_feature_names_out(self):
        return ["numeric__tenure", "numeric__MonthlyCharges"]


class DummyPipeline:
    named_steps = {"preprocessing": DummyPreprocessor()}


def test_save_training_artifacts_creates_expected_files(monkeypatch, tmp_path: Path) -> None:
    class DummySettings:
        model_path = tmp_path / "trained_model.joblib"
        threshold_path = tmp_path / "selected_threshold.json"
        metrics_path = tmp_path / "evaluation_metrics.json"

    monkeypatch.setattr(
        "telecom_churn_prediction.ml.train_pipeline.settings",
        DummySettings(),
    )

    result = TrainingResult(
        model_pipeline=DummyPipeline(),
        metrics={"accuracy": 0.9, "f1_score": 0.8},
        threshold=0.45,
    )

    save_training_artifacts(result)

    assert (tmp_path / "trained_model.joblib").exists()
    assert (tmp_path / "selected_threshold.json").exists()
    assert (tmp_path / "evaluation_metrics.json").exists()
    assert (tmp_path / "feature_names.json").exists()

    with (tmp_path / "selected_threshold.json").open("r", encoding="utf-8") as file:
        threshold_payload = json.load(file)

    assert threshold_payload["threshold"] == 0.45