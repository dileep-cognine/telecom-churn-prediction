from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from telecom_churn_prediction.exceptions import ArtifactNotFoundError, PredictionError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ArtifactNotFoundError)
    async def handle_artifact_not_found(
        request: Request,
        exc: ArtifactNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    @app.exception_handler(PredictionError)
    async def handle_prediction_error(
        request: Request,
        exc: PredictionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )