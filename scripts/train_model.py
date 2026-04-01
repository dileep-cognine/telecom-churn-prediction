from telecom_churn_prediction.ml.train_pipeline import (
    train_model_pipeline,
    save_training_artifacts,
)

if __name__ == "__main__":
    result = train_model_pipeline(model_name="xgboost")
    save_training_artifacts(result)
    print("Training completed successfully.")
    print(result.metrics)