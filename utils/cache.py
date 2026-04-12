"""
utils/cache.py
~~~~~~~~~~~~~~
Redis caching layer for research pipeline results.

Provides async get/set/delete with JSON serialization and TTL.
Falls back gracefully to a no-op in-memory dict when Redis is unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Any

import structlog

log = structlog.get_logger(__name__)

# TTL default: 1 hour
DEFAULT_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))


class CacheClient:
    """
    Async Redis cache client with JSON serialization.

    Falls back to an in-memory dict if Redis is unavailable,
    so the application continues working without Redis.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        ttl: int = DEFAULT_TTL,
    ):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.ttl = ttl
        self._redis = None
        self._fallback: dict[str, str] = {}  # In-memory fallback
        self._use_fallback = False

    async def _get_redis(self):
        """Lazily initialize Redis connection."""
        if self._use_fallback:
            return None
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(
                self.url, encoding="utf-8", decode_responses=True
            )
            await self._redis.ping()
            log.info("redis_connected", url=self.url)
        except Exception as exc:
            log.warning("redis_unavailable", error=str(exc), fallback="in-memory")
            self._use_fallback = True
            self._redis = None
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve and deserialize a cached value."""
        redis = await self._get_redis()

        if self._use_fallback:
            raw = self._fallback.get(key)
        else:
            try:
                raw = await redis.get(key)
            except Exception as exc:
                log.warning("cache_get_error", key=key, error=str(exc))
                return None

        if raw is None:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Serialize and store a value with TTL."""
        ttl = ttl or self.ttl
        serialized = json.dumps(value, default=str)
        redis = await self._get_redis()

        if self._use_fallback:
            self._fallback[key] = serialized
            return True

        try:
            await redis.set(key, serialized, ex=ttl)
            return True
        except Exception as exc:
            log.warning("cache_set_error", key=key, error=str(exc))
            return False

    async def delete(self, key: str) -> bool:
        """Remove a cached value."""
        redis = await self._get_redis()

        if self._use_fallback:
            self._fallback.pop(key, None)
            return True

        try:
            await redis.delete(key)
            return True
        except Exception as exc:
            log.warning("cache_delete_error", key=key, error=str(exc))
            return False

    async def exists(self, key: str) -> bool:
        """Check whether a key exists in cache."""
        redis = await self._get_redis()

        if self._use_fallback:
            return key in self._fallback

        try:
            return bool(await redis.exists(key))
        except Exception:
            return False

    async def flush(self) -> None:
        """Clear all cached values (use with care)."""
        redis = await self._get_redis()

        if self._use_fallback:
            self._fallback.clear()
            return

        try:
            await redis.flushdb()
        except Exception as exc:
            log.warning("cache_flush_error", error=str(exc))