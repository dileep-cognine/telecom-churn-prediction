from __future__ import annotations

from typing import Iterable


def format_top_reasons(
    feature_contributions: Iterable[tuple[str, float]],
) -> list[str]:
    """Convert feature contribution pairs into readable explanation strings."""
    formatted_reasons: list[str] = []

    for feature_name, contribution in feature_contributions:
        direction = "increased" if contribution > 0 else "reduced"
        formatted_reasons.append(
            f"{feature_name} {direction} churn risk by {abs(contribution):.4f}"
        )

    return formatted_reasons