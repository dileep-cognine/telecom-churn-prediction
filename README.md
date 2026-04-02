# Telecom Churn Prediction System

A Machine Learning-powered system that:

* Predicts **customer churn probability**
* Classifies users into **risk levels**
* Provides **interpretable insights using SHAP**
* Exposes a **REST API using FastAPI**
* Runs as a **Docker containerized service**

This project demonstrates a **production-ready ML system** combining **model performance, explainability, and scalable deployment**.

---

# Project Overview

Customer churn prediction is a critical problem in the telecom industry.

This system solves it by:

1. Building an **end-to-end ML pipeline** for churn prediction.
2. Using **XGBoost** for high-performance classification.
3. Applying **threshold optimization** for business decision-making.
4. Deploying via **FastAPI REST service**.
5. Supporting **Docker-based deployment**.
6. Providing **model explainability using SHAP**.

---

# Key Features

### ML Churn Prediction

Predicts whether a customer is likely to churn.

Model Details:

| Component          | Technique            |
|------------------|--------------------|
| Algorithm        | XGBoost            |
| Feature Handling | Encoding + Scaling |
| Optimization     | Threshold tuning   |

---

### Explainability (SHAP)

Provides model transparency by:

* Identifying key churn drivers
* Explaining individual predictions
* Supporting business decisions

---

### FastAPI REST API

Endpoints allow integration with external systems.

Example use cases:

* Telecom analytics platforms  
* Customer retention systems  
* Business intelligence dashboards  

---

### Docker Deployment

The application runs inside Docker for:

* Easy deployment  
* Environment consistency  
* Scalability  

---

# System Architecture

Customer Data
│
▼
Data Preprocessing
│
▼
ML Model (XGBoost)
│
▼
Prediction + Probability
│
▼
SHAP Explainability
│
▼
FastAPI Service
│
▼
JSON Response


---

# Project Structure
```
telecom-churn-prediction/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── artifacts/
│ ├── trained_model.joblib
│ ├── preprocessing_pipeline.joblib
│ ├── feature_names.json
│ ├── selected_threshold.json
│ └── evaluation_metrics.json
│
├── reports/
│ ├── figures/
│ └── evaluation_report.md
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_model_experiments.ipynb
│ └── 03_shap_analysis.ipynb
│
├── scripts/
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── smoke_test.py
│
├── src/telecom_churn_prediction/
│ ├── api/
│ ├── data/
│ ├── ml/
│ ├── services/
│ ├── explainability/
│ └── utils/
│
├── tests/
├── Dockerfile
├── docker-compose.yml
└── README.md
```
<<<<<<< HEAD
---
The core assignment solution is the FastAPI /predict service backed by a trained XGBoost pipeline. Additional components such as Prometheus monitoring, Redis-ready caching, and explainability modules were added to demonstrate production-readiness and modular design.
---
=======
>>>>>>> e3d963a4f3efb59811d50af34abef571d79e81cd

# Installation (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/telecom-churn-prediction.git
```

### Navigate to Project
```bash
cd telecom-churn-prediction
```
### 2.Create Virtual Environment
```
python -m venv telecom-venv
```
Activate Environment:

**Windows**
```
telecom-venv\Scripts\activate
```

**Mac / Linux**
```
source telecom-venv/bin/activate
```
---
### 3.Install Dependencies
### Option 1 (Recommended - simple)
```
pip install -r requirements.txt
```
### Option 2 (Advanced / developer mode)
```
pip install -e ".[dev]"
```
---
## 4.Model Training
Run the training script:
```
python scripts/train_model.py
```
Artifacts will be saved in:
```
artifacts/
```
## Model Evaluation
Run:
```
python scripts/evaluate_model.py
```
## Generated Outputs
- confusion_matrix.png
- roc_curve.png
- precision_recall_curve.png
- class_distribution.png
Saved in:
```
reports/figures/
```
### Run Tests
```bash
pytest --cov=src --cov-report=term-missing
```
# Running the API
```bash
uvicorn telecom_churn_prediction.api.application:app --reload
```
Open in browser:
```
http://localhost:8000/docs
```
FastAPI will display interactive API documentation.

# API Endpoints
## Health Check
```
GET /health
```
Response

```
{
  "status": "ok"
}
```
## Churn Prediction
```
POST /predict
```

Example Request
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

Example Response
```json
{
  "summary": "This customer is at high risk of churn.",
  "churn_probability": 0.9348,
  "prediction": {
    "label": 1,
    "label_name": "Churn",
    "risk_level": "High"
  },
  "model_info": {
    "threshold": 0.4
  }
}
```
The core assignment solution is the FastAPI /predict service backed by a trained XGBoost pipeline. Additional components such as Prometheus monitoring, Redis-ready caching, and explainability modules were added to demonstrate production-readiness and modular design.

# Model Performance
| Metric     | Value |
|-----------|------|
| Accuracy  | 0.75 |
| Precision | 0.58 |
| Recall    | 0.71 |
| F1 Score  | 0.60 |
| ROC-AUC   | 0.82 |
## Threshold Usage

The model uses an optimized threshold (not default 0.5) loaded from:
```
artifacts/selected_threshold.json
```
This ensures predictions align with business objectives

# Running Tests
Run unit tests using **pytest**:
```
pytest --cov=src --cov-report=term-missing
```
# Docker Deployment
## Build & Run
```
docker compose up --build
```

Access API
```
http://localhost:8000/docs
```
# Monitoring & Scalability
- Prometheus configuration included
- Redis caching supported
- Docker-based scalable deployment

## Failure Cases and Handling

The system handles potential failures such as:

- Missing or invalid input fields → handled by FastAPI validation
- Incorrect data types → rejected at API level
- Missing model artifacts → raises controlled exception
- Unknown categories → handled using encoding strategy
- Runtime prediction errors → wrapped in custom PredictionError

These safeguards ensure robustness and production readiness.

# Technologies Used
* Python
* FastAPI
* Scikit-learn
* XGBoost
* SHAP
* PyTest
* Docker
---

# Author
AI/ML Engineer

Telecom-Churn Prediction System

