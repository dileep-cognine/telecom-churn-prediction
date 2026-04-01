from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


# def _get_project_root() -> Path:
#     return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()

def _get_project_root() -> Path:
    if os.getenv("PROJECT_ROOT"):
        return Path(os.getenv("PROJECT_ROOT")).resolve()
    return Path(__file__).resolve().parents[2]

PROJECT_ROOT = _get_project_root()
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))


@dataclass(frozen=True)
class Settings:
    app_name: str = "telecom-churn-prediction"
    model_path: Path = ARTIFACTS_DIR / "trained_model.joblib"
    threshold_path: Path = ARTIFACTS_DIR / "selected_threshold.json"
    metrics_path: Path = ARTIFACTS_DIR / "evaluation_metrics.json"
    raw_dataset_path: Path = RAW_DATA_DIR / "telco_customer_churn.csv"
    reports_dir: Path = REPORTS_DIR
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()