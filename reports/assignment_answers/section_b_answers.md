# Models & Evaluation

## Model Selection & Training

### Decision Tree Splitting

Decision Trees split data based on feature thresholds.

Impurity metrics:
- Gini Index
- Entropy

Example tree:
- Root: contract_months
- Next: data_usage_gb
- Next: calls_to_support

Each split reduces impurity.

---

### Random Forest vs Decision Tree

Random Forest improves over single tree by:
- Bagging (sampling data)
- Feature randomness

**Advantages:**
- reduces overfitting
- improves generalization

Optimal trees:
- tuned via cross-validation

---

### Naive Bayes Assumption

Assumes feature independence.

Not realistic:
- monthly bill and usage are correlated

Impact:
- may reduce accuracy

---

###  SVM Concepts

- Margin: separation boundary
- Support vectors: closest points
- Kernel trick: nonlinear transformation

Use RBF kernel when:
- data is nonlinear

---

### K-Fold Cross Validation

80,000 records, 5 folds:

Training size per fold:
\[
\frac{4}{5} × 80,000 = 64,000
\]

**Why CV:**
- better generalization
- reduces variance

---

## Evaluation Metrics

### Metrics Calculation

Given:
- TN = 12480
- FP = 400
- FN = 720
- TP = 2400

Accuracy:
\[
\frac{TP + TN}{Total} = 0.93
\]

Precision:
\[
\frac{TP}{TP+FP} = 0.857
\]

Recall:
\[
\frac{TP}{TP+FN} = 0.769
\]

F1:
\[
0.811
\]

Specificity:
\[
0.969
\]

---

### Precision vs Recall

Important:
- Missing churn → high loss
- False alarm → lower cost

**Conclusion:**
Recall is more important

---

### ROC & AUC

ROC curve:
- TPR vs FPR

AUC:
- probability model ranks correctly

Model B (AUC=0.91) is better generally

But Model A preferred if:
- higher precision needed
- business constraints differ