from __future__ import annotations

import json

import joblib
from sklearn.metrics import confusion_matrix

from telecom_churn_prediction.data.data_loader import clean_dataset, load_raw_dataset
from telecom_churn_prediction.data.data_splitter import (
    split_features_and_target,
    split_train_test,
)
from telecom_churn_prediction.explainability.shap_explainer import ShapExplainer
from telecom_churn_prediction.ml.evaluation import (
    evaluate_predictions,
    save_class_distribution_plot,
    save_confusion_matrix_plot,
    save_precision_recall_curve_plot,
    save_roc_curve_plot,
)
from telecom_churn_prediction.settings import settings


def main() -> None:
    """Load trained artifacts, evaluate the model, and generate report figures."""
    df = clean_dataset(load_raw_dataset())
    x, y = split_features_and_target(df)
    _, x_test, _, y_test = split_train_test(x, y)

    pipeline = joblib.load(settings.model_path)

    with settings.threshold_path.open("r", encoding="utf-8") as file:
        threshold_payload = json.load(file)

    threshold = float(threshold_payload["threshold"])

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_predictions(y_test, y_pred, y_proba)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    figures_dir = settings.reports_dir / "figures"

    save_class_distribution_plot(
        y_true=y,
        output_path=figures_dir / "class_distribution.png",
    )

    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=y_pred,
        output_path=figures_dir / "confusion_matrix.png",
    )

    save_roc_curve_plot(
        y_true=y_test,
        y_proba=y_proba,
        output_path=figures_dir / "roc_curve.png",
    )

    save_precision_recall_curve_plot(
        y_true=y_test,
        y_proba=y_proba,
        output_path=figures_dir / "precision_recall_curve.png",
    )

    shap_sample = x_test.head(100).copy()
    shap_explainer = ShapExplainer(pipeline)
    shap_explainer.save_summary_plot(
        shap_sample,
        figures_dir / "shap_summary_plot.png",
    )

    print("Saved evaluation and explainability figures successfully.")


if __name__ == "__main__":
    main()