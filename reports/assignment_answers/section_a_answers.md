# Mathematics & Statistical Foundations

## Linear Algebra — Feature Representation

### 1. Feature Matrix Representation

Let the feature matrix \( X \) represent 5 customers and 4 features:
- Monthly Bill
- Calls to Support
- Data Usage (GB)
- Contract Months

\[
X =
\begin{bmatrix}
20.0 & 1 & 5.0 & 1 \\
70.5 & 3 & 15.2 & 12 \\
90.0 & 0 & 25.3 & 24 \\
45.2 & 2 & 10.1 & 6 \\
60.3 & 4 & 18.5 & 18
\end{bmatrix}
\]

Dimensions:

\[
X \in \mathbb{R}^{5 \times 4}
\]

Each row represents a customer and each column represents a feature.

---

### 2. PCA on Feature Matrix

Principal Component Analysis (PCA) transforms the dataset into a lower-dimensional space by projecting it onto directions of maximum variance.

Steps:
1. Center the data
2. Compute covariance matrix
3. Compute eigenvalues and eigenvectors
4. Select top \( k \) eigenvectors
5. Project data onto these components

Each eigenvector represents a **principal direction of variation** in churn behavior.

**Churn context:**
- One component may capture "high spend + high usage"
- Another may capture "low tenure + weak engagement"

Thus, PCA reduces dimensionality while preserving important churn-related patterns.

---

### 3. Cosine Similarity

Customer A = [250, 8, 1.2, 6]  
Customer B = [240, 9, 1.1, 6]

\[
\text{cosine similarity} = \frac{A \cdot B}{||A|| \cdot ||B||}
\]

Dot product:

\[
A \cdot B = 250×240 + 8×9 + 1.2×1.1 + 6×6 = 60000 + 72 + 1.32 + 36 = 60109.32
\]

Magnitudes:

\[
||A|| \approx \sqrt{250^2 + 8^2 + 1.2^2 + 6^2} \approx 250.2
\]
\[
||B|| \approx 240.2
\]

\[
\text{cosine similarity} \approx \frac{60109.32}{250.2 × 240.2} \approx 0.999
\]

**Interpretation:**
Customers are highly similar → likely similar churn behavior.

---

### 4. Dot Product Interpretation

\[
w = [0.4, 0.3, 0.2, 0.1], \quad x = [250, 8, 1.2, 6]
\]

\[
w \cdot x = 0.4×250 + 0.3×8 + 0.2×1.2 + 0.1×6
\]

\[
= 100 + 2.4 + 0.24 + 0.6 = 103.24
\]

**Interpretation:**
This value is the **linear model score**, representing weighted influence of features on churn risk.

---

## Calculus — Gradient Descent & Loss Functions

### 5. Binary Cross-Entropy Loss

\[
\text{Loss} = -[y \log(p) + (1-y)\log(1-p)]
\]

Given:
- \( p = 0.72 \)
- \( y = 1 \)

\[
\text{Loss} = -\log(0.72) \approx 0.328
\]

---

### 6. Batch MSE

Given MSE values:
\[
[0.12, 0.45, 0.08, 0.31]
\]

\[
\text{Batch MSE} = \frac{0.12 + 0.45 + 0.08 + 0.31}{4} = 0.24
\]

Sample:
\[
(0.65 - 0)^2 = 0.4225
\]

**Why BCE is better:**
- Designed for classification
- Penalizes confident wrong predictions
- MSE is less sensitive for probabilities

---

### 7. Gradient Descent

\[
L(\theta) = 0.5\theta^2 - 3\theta + 7
\]

\[
\frac{dL}{d\theta} = \theta - 3
\]

Learning rate \( \alpha = 0.1 \), \( \theta_0 = 0 \)

Step updates:
- \( \theta_1 = 0.3 \)
- \( \theta_2 = 0.57 \)
- \( \theta_3 = 0.813 \)

Optimal solution:
\[
\theta = 3
\]

---

### 8. GD vs SGD vs Mini-batch

| Method | Description |
|------|------------|
| Batch GD | Uses full dataset |
| SGD | Uses single sample |
| Mini-batch | Uses small batches |

**Best choice:** Mini-batch GD  
- balances speed and stability  
- suitable for large datasets (80k records)

---

## Statistics & Probability

### 9. Class Imbalance

Dataset:
- 82% non-churn
- 18% churn

**Impact:**
- model biased toward majority class
- misleading accuracy

**Solutions:**
1. SMOTE (oversampling)
2. Class weighting

---

### 10. Bayes Theorem

\[
P(3+ | churn) = \frac{P(churn | 3+) × P(3+)}{P(churn)}
\]

\[
= \frac{0.7 × 0.25}{0.18} \approx 0.972
\]

---

### 11. Normal Distribution

\[
Z_1 = \frac{730 - 850}{120} = -1
\]
\[
Z_2 = \frac{970 - 850}{120} = 1
\]

\[
P(-1 < Z < 1) \approx 0.68
\]

**Interpretation:**
~68% customers fall within this billing range.