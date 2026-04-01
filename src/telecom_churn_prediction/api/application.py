from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from telecom_churn_prediction.api.error_handlers import register_exception_handlers
from telecom_churn_prediction.api.middleware import register_middleware
from telecom_churn_prediction.api.routes import router


def create_application() -> FastAPI:
    app = FastAPI(
        title="Telecom Churn Prediction API",
        version="1.0.0",
    )

    register_exception_handlers(app)
    register_middleware(app)

    app.include_router(router)

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_application()