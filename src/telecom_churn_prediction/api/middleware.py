from __future__ import annotations

import time

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total API request count",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            http_status=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(elapsed)

        return response


def register_middleware(app: FastAPI) -> None:
    app.add_middleware(MetricsMiddleware)