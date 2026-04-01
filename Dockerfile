FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY data ./data
COPY prometheus.yml ./prometheus.yml

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["uvicorn", "telecom_churn_prediction.api.application:app", "--host", "0.0.0.0", "--port", "8000"]