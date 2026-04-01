from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap


class ShapExplainer:
    """Generate SHAP explanations for a trained tree-based model pipeline."""

    def __init__(self, model_pipeline: Any) -> None:
        self.model_pipeline = model_pipeline
        self.preprocessor = model_pipeline.named_steps["preprocessing"]
        self.model = model_pipeline.named_steps["model"]

    def transform_features(self, features: pd.DataFrame):
        """Transform raw input features using the fitted preprocessing pipeline."""
        return self.preprocessor.transform(features)

    def compute_shap_values(self, features: pd.DataFrame):
        """Compute SHAP values for the provided input features."""
        transformed_features = self.transform_features(features)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(transformed_features)
        return shap_values, transformed_features

    def save_summary_plot(self, features: pd.DataFrame, output_path: Path) -> None:
        """Generate and save a SHAP summary plot."""
        shap_values, transformed_features = self.compute_shap_values(features)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap.summary_plot(shap_values, transformed_features, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()