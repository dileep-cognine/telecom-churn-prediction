from telecom_churn_prediction.explainability.explanation_formatter import (
    format_top_reasons,
)


def test_format_top_reasons_formats_messages() -> None:
    reasons = format_top_reasons(
        [
            ("Contract", 0.1234),
            ("TechSupport", -0.0456),
        ]
    )

    assert len(reasons) == 2
    assert "Contract increased churn risk" in reasons[0]
    assert "TechSupport reduced churn risk" in reasons[1]