from pathlib import Path

from telecom_churn_prediction.ml.evaluation import (
    evaluate_predictions,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
)


def test_evaluate_predictions_returns_metrics() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    y_proba = [0.1, 0.8, 0.9, 0.2]

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics


def test_save_confusion_matrix_plot_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "confusion_matrix.png"

    save_confusion_matrix_plot(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        output_path=output_path,
    )

    assert output_path.exists()


def test_save_roc_curve_plot_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "roc_curve.png"

    save_roc_curve_plot(
        y_true=[0, 1, 1, 0],
        y_proba=[0.1, 0.8, 0.7, 0.2],
        output_path=output_path,
    )

    assert output_path.exists()