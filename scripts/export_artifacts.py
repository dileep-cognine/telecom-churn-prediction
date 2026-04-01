from __future__ import annotations

from telecom_churn_prediction.ml.train_pipeline import (
    save_training_artifacts,
    train_model_pipeline,
)


def main() -> None:
    result = train_model_pipeline(model_name="xgboost")
    save_training_artifacts(result)
    print("Artifacts exported successfully.")


if __name__ == "__main__":
    main()