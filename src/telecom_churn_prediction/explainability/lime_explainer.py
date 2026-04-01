from __future__ import annotations

from typing import Any

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


class LimeExplainer:
    """Generate LIME explanations for a trained preprocessing + model pipeline."""

    def __init__(self, model_pipeline: Any, reference_features: pd.DataFrame) -> None:
        self.model_pipeline = model_pipeline
        self.preprocessor = model_pipeline.named_steps["preprocessing"]
        self.model = model_pipeline.named_steps["model"]

        transformed_reference = self.preprocessor.transform(reference_features)
        self.feature_names = self.preprocessor.get_feature_names_out().tolist()

        self.explainer = LimeTabularExplainer(
            training_data=transformed_reference,
            feature_names=self.feature_names,
            class_names=["No Churn", "Churn"],
            mode="classification",
        )

    def explain_instance(self, instance_features: pd.DataFrame, num_features: int = 10):
        """Generate a LIME explanation for a single instance."""
        transformed_instance = self.preprocessor.transform(instance_features)[0]

        explanation = self.explainer.explain_instance(
            data_row=transformed_instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
        )
        return explanation