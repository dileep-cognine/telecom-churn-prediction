from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HealthStatus:
    status: str
    message: str


def get_health_status() -> HealthStatus:
    return HealthStatus(status="ok", message="Service is healthy.")