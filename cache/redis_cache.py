"""
Redis Cache with In-Memory Fallback

Provides a unified caching interface backed by Redis.  When Redis is
unavailable (e.g. local dev, CI) it transparently degrades to an
in-memory Python dict.

Usage::

    cache = RedisCache()                       # auto-detects Redis
    cache.set("prompt:abc123", result, ttl=60)
    cached = cache.get("prompt:abc123")
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import redis as redis_lib

    _REDIS_AVAILABLE = True
except ImportError:
    redis_lib = None  # type: ignore[assignment]
    _REDIS_AVAILABLE = False


class RedisCache:
    """Key-value cache with Redis backend and in-memory fallback.

    Parameters
    ----------
    redis_client : redis.Redis, optional
        Pre-configured Redis client.  If ``None`` a local connection is
        attempted; on failure the cache falls back to an in-memory dict.
    default_ttl : int
        Default time-to-live in seconds for cached entries.
    prefix : str
        Key prefix to namespace all cache entries.
    """

    def __init__(
        self,
        redis_client: Any = None,
        default_ttl: int = 300,
        prefix: str = "cip:",
    ) -> None:
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._redis: Any = None
        self._memory: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

        if redis_client is not None:
            self._redis = redis_client
        elif _REDIS_AVAILABLE:
            try:
                client = redis_lib.Redis(host="localhost", port=6379, db=0)
                client.ping()
                self._redis = client
                logger.info("Connected to Redis at localhost:6379")
            except Exception:
                logger.info("Redis unavailable — using in-memory fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value by key.  Returns ``None`` on miss."""
        full_key = self.prefix + key
        if self._redis is not None:
            try:
                raw = self._redis.get(full_key)
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis GET failed: %s", exc)
                return None

        # In-memory fallback
        if full_key in self._memory:
            if full_key in self._expiry and time.time() > self._expiry[full_key]:
                del self._memory[full_key]
                del self._expiry[full_key]
                return None
            return self._memory[full_key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in the cache with an optional TTL (seconds)."""
        full_key = self.prefix + key
        ttl = ttl if ttl is not None else self.default_ttl

        if self._redis is not None:
            try:
                self._redis.setex(full_key, ttl, json.dumps(value))
                return
            except Exception as exc:
                logger.warning("Redis SET failed: %s", exc)

        # In-memory fallback
        self._memory[full_key] = value
        if ttl > 0:
            self._expiry[full_key] = time.time() + ttl

    def delete(self, key: str) -> bool:
        """Remove a key from the cache.  Returns ``True`` if it existed."""
        full_key = self.prefix + key
        if self._redis is not None:
            try:
                return bool(self._redis.delete(full_key))
            except Exception as exc:
                logger.warning("Redis DELETE failed: %s", exc)
                return False

        existed = full_key in self._memory
        self._memory.pop(full_key, None)
        self._expiry.pop(full_key, None)
        return existed

    def exists(self, key: str) -> bool:
        """Check whether a key exists in the cache."""
        full_key = self.prefix + key
        if self._redis is not None:
            try:
                return bool(self._redis.exists(full_key))
            except Exception:
                return False

        if full_key in self._memory:
            if full_key in self._expiry and time.time() > self._expiry[full_key]:
                del self._memory[full_key]
                del self._expiry[full_key]
                return False
            return True
        return False

    def flush(self) -> None:
        """Remove all entries managed by this cache (prefix-scoped)."""
        if self._redis is not None:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{self.prefix}*", count=100)
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
                return
            except Exception as exc:
                logger.warning("Redis FLUSH failed: %s", exc)

        # In-memory fallback
        self._memory.clear()
        self._expiry.clear()

    @staticmethod
    def compute_cache_key(*args: Any) -> str:
        """Compute a deterministic SHA-256 cache key from arbitrary arguments.

        All arguments are JSON-serialised and hashed so that logically
        identical requests map to the same key regardless of dict ordering.
        """
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
