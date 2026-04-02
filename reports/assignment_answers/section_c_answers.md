# Boosting, Pipelines & Model Explainability

## XGBoost & LightGBM

XGBoost improves over standard gradient boosting in several important ways:

1. **Regularization**
   - XGBoost adds a regularization term to the objective function.
   - This helps control model complexity and reduces overfitting.

2. **Second-order optimization**
   - It uses both first-order gradients and second-order Hessian information.
   - This makes the optimization more accurate and efficient.

3. **Shrinkage / learning rate**
   - Each tree contributes only a fraction of its output.
   - This improves generalization.

4. **Subsampling**
   - XGBoost can sample rows and columns while training.
   - This improves robustness and reduces variance.

5. **Efficient implementation**
   - It is optimized for speed and memory usage.
   - It supports parallelism and efficient handling of structured data.

The XGBoost objective can be written as:

\[
\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
\]

where:

- \(l(y_i, \hat{y}_i)\) is the training loss
- \(\Omega(f_k)\) is the regularization term for the \(k\)-th tree

A common form of the regularization term is:

\[
\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
\]

where:
- \(T\) = number of leaves
- \(w_j\) = score/weight of leaf \(j\)
- \(\gamma\) penalizes too many leaves
- \(\lambda\) penalizes large leaf weights

### Role of \(\Omega(f_k)\)
The regularization term:
- penalizes overly complex trees
- discourages too many leaves
- discourages extreme leaf scores
- improves generalization on unseen customer data

### Churn context
In telecom churn prediction, this is important because the model may otherwise overfit specific patterns in a training sample, such as unusual billing or support-behavior combinations. Regularization helps the model learn stable churn patterns instead of memorizing noise.

---
Each hyperparameter controls a different part of model complexity and training behavior.

#### 1. `max_depth = 6`
This controls the maximum depth of each tree.
- Larger values make trees more complex
- Smaller values reduce overfitting risk

For churn data:
- a deeper tree can model interactions like short tenure + high monthly charges + weak support
- but very deep trees may overfit

#### 2. `n_estimators = 200`
This is the number of boosting rounds / trees.
- More trees can improve performance
- Too many trees may overfit or increase training time

#### 3. `learning_rate = 0.1`
This controls how much each tree contributes.
- Smaller learning rate = slower learning, often better generalization
- Larger learning rate = faster learning, but can overfit

#### 4. `subsample = 0.8`
This means each tree is trained on 80% of the training rows.
- Helps reduce variance
- Improves robustness
- Acts as regularization

### Tuning strategy for the churn dataset
I would tune these hyperparameters using **Stratified K-Fold Cross-Validation**, because the churn dataset is imbalanced.

Example tuning logic:
- `max_depth`: try `[4, 6, 8]`
- `n_estimators`: try `[100, 200, 300]`
- `learning_rate`: try `[0.05, 0.1, 0.2]`
- `subsample`: try `[0.8, 1.0]`

The best combination would be selected using a validation metric such as:
- F1-score
- Recall
- ROC-AUC

### Project alignment
In the project, XGBoost is the final deployed model, and the pipeline supports hyperparameter-based configuration. This makes the model selection process realistic and aligned with production use.

---

LightGBM and XGBoost are both gradient boosting frameworks, but they differ in tree growth strategy.

#### XGBoost — Level-wise growth
- Expands trees level by level
- Grows nodes in a balanced way
- Usually more conservative
- Easier to control overfitting

#### LightGBM — Leaf-wise growth
- Expands the leaf that gives the best loss reduction
- Produces deeper, more asymmetric trees
- Often reaches lower loss faster

#### Histogram-based splitting
LightGBM also bins continuous values into discrete buckets before splitting.
This reduces:
- training time
- memory usage

### Which would train faster on 80,000 records?
**LightGBM would usually train faster** because:
- histogram-based splits are efficient
- leaf-wise growth can reduce loss quickly

### Overfitting risk
Leaf-wise growth can produce:
- very deep branches
- overly specific rules
- memorization of training noise

So the main overfitting risk is that LightGBM may become too aggressive unless parameters such as:
- `num_leaves`
- `max_depth`
- `min_data_in_leaf`

are controlled.

### Churn context
For a churn dataset with 80,000 customers, LightGBM could be very fast, but XGBoost may be preferred if better regularization control and stable generalization are more important.

---

A correct pipeline structure is:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

preprocessing = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier())
    ]
)
```

## SHAP & LIME — Explainability
SHAP (SHapley Additive exPlanations) explains a prediction by assigning each feature a contribution value based on cooperative game theory.

For a single customer, SHAP starts from a baseline prediction and then shows how each feature moves the prediction:

- upward toward churn
- or downward away from churn

The contributions add up to the final prediction.

        Final Prediction=Base Value+∑SHAP Contributions
For customer #4421 with churn probability = 89%

A SHAP waterfall plot would show:

### 1.Base value
- the average churn tendency in the dataset
### 2.Positive contributions
- features increasing churn risk, for example:
    - short tenure
    - month-to-month contract
    - high monthly charges
    - lack of tech support
    - no online security
### 3.Negative contributions
- features reducing churn risk, for example:
    - long customer history
    - automatic payment
    - bundled services
    - stronger support options
### 4.Final prediction
- all contributions combined produce the final churn probability of 0.89
## Interpretation

A waterfall plot answers the retention manager’s question:
```
“Why is this customer flagged?”
```
It shows exactly which features pushed the customer into the high-risk category.

## Project alignment

In the project, SHAP is implemented and used to generate explainability outputs, including a summary plot. This makes the churn model more transparent and business-trustworthy.

LIME and SHAP are both explainability methods, but they work differently.

# LIME

LIME explains a prediction by:

- perturbing data around one instance
- fitting a simple local surrogate model
- approximating the original model only near that point

So LIME is a local approximation method.

# SHAP

SHAP explains a prediction using:

Shapley values from cooperative game theory
additive feature contributions
a principled and consistent attribution framework

So SHAP provides a more theoretically grounded explanation.

## Key limitation of LIME

The biggest limitation of LIME is instability.

Its explanation can change depending on:

- how the neighborhood is sampled
- how the perturbed points are generated
- the local surrogate fit

This makes LIME less consistent across repeated runs or slightly different settings.

**Why SHAP is more reliable for this churn model**

SHAP is more reliable because:

- it provides additive feature attribution
- it is more consistent
- it aligns well with tree-based models like XGBoost
- it supports both local and global interpretation
