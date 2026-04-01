from __future__ import annotations

import json
from typing import Any, Optional

import redis


class CacheService:
    def __init__(self, host: str = "redis", port: int = 6379) -> None:
        self._client = redis.Redis(host=host, port=port, decode_responses=True)

    def get(self, key: str) -> Optional[dict[str, Any]]:
        raw_value = self._client.get(key)
        if raw_value is None:
            return None
        return json.loads(raw_value)

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 300) -> None:
        self._client.setex(key, ttl_seconds, json.dumps(value, default=str))