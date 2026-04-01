from pathlib import Path

import pytest

from telecom_churn_prediction.exceptions import ArtifactNotFoundError
from telecom_churn_prediction.services import model_loader


# def test_load_model_pipeline_raises_when_artifact_missing(monkeypatch) -> None:
#     class DummySettings:
#         model_path = Path("missing_model.joblib")
#         threshold_path = Path("missing_threshold.json")

#     monkeypatch.setattr(model_loader, "settings", DummySettings())

#     with pytest.raises(ArtifactNotFoundError):
#         model_loader.load_model_pipeline()

def test_load_threshold_returns_float(monkeypatch, tmp_path: Path) -> None:
    threshold_file = tmp_path / "selected_threshold.json"
    threshold_file.write_text('{"threshold": 0.55}', encoding="utf-8")

    class DummySettings:
        model_path = tmp_path / "trained_model.joblib"
        threshold_path = threshold_file

    monkeypatch.setattr(model_loader, "settings", DummySettings())

    threshold = model_loader.load_threshold()

    assert threshold == 0.55