from telecom_churn_prediction.ml.train_pipeline import TrainingResult


class DummyPipeline:
    def fit(self, x_train, y_train):
        return self

    def predict_proba(self, x_test):
        import numpy as np
        return np.array([[0.3, 0.7], [0.8, 0.2]])


def test_training_result_dataclass() -> None:
    result = TrainingResult(
        model_pipeline=DummyPipeline(),
        metrics={"accuracy": 0.8},
        threshold=0.5,
    )

    assert result.metrics["accuracy"] == 0.8
    assert result.threshold == 0.5