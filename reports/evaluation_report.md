#  Model Evaluation Report

---

## Dataset Overview

- Dataset: Telecom Customer Churn
- Samples: ~7000+
- Target: Churn (Yes / No)

The dataset is **imbalanced**, with more non-churn customers.

---

## Data Preprocessing

- Missing value handling
- Categorical encoding
- Numerical scaling
- Pipeline-based transformation
---

## Final Model Performance

| Metric     | Value |
|-----------|------|
| Accuracy  | 0.78 |
| Precision | 0.58 |
| Recall    | 0.65 |
| F1 Score  | 0.61 |
| ROC-AUC   | 0.82 |

---

##  Threshold Optimization

The API does not use a hardcoded default threshold of 0.5. It loads the optimized threshold from **selected_threshold.json** and applies it during inference.

### Reason:
- Improves recall
- Reduces missed churn cases
- Aligns with business needs

---

## Confusion Matrix Analysis

- False Negatives → Missed churn (critical)
- False Positives → Extra intervention cost

Priority: Minimize false negatives

---

## ROC Curve

- ROC-AUC ≈ 0.82
- Indicates strong separation capability

---

## Business Insights

Key churn drivers:

- Short tenure
- Month-to-month contracts
- High monthly charges
- Lack of support services

---

## Conclusion

The model provides:
- Strong predictive performance
- Business-aligned decisions
- Explainable insights

SHAP (SHapley Additive Explanations) was used to interpret model predictions. A summary plot was generated to identify the most influential features contributing to churn risk. This improves model transparency and supports business decision-making.
