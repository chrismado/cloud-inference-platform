"""
SLO-Aware GPU Router
Routes inference requests across SGLang (text prefill) and 3DGS (spatial rendering)
with dynamic TVM step-scaling based on p95 latency targets.

Reference: Zhou et al. (Luma AI), Terminal Velocity Matching, ICLR 2026.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal
import redis

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

LATENCY_KEY = "slo:latency_window"
WINDOW_SECONDS = 60


@dataclass
class SLOConfig:
    p95_target_ms: float = 600.0
    tvm_steps_normal: int = 4
    tvm_steps_degraded: int = 1
    degradation_threshold: float = 0.85


class SLORouter:
    def __init__(self, config: SLOConfig, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self._current_steps = config.tvm_steps_normal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float) -> None:
        """Push a latency observation into the Redis rolling window.

        Each entry is stored as a sorted-set member with its timestamp as
        the score so stale entries can be trimmed efficiently.
        """
        now = time.time()
        member = f"{now}:{latency_ms}"
        pipe = self.redis.pipeline()
        pipe.zadd(LATENCY_KEY, {member: now})
        pipe.zremrangebyscore(LATENCY_KEY, "-inf", now - WINDOW_SECONDS)
        pipe.execute()

    def get_nfe_steps(self) -> int:
        """Dynamically return NFE step count based on current server load."""
        p95 = self._get_current_p95_latency()
        threshold = self.config.p95_target_ms * self.config.degradation_threshold
        self._current_steps = (
            self.config.tvm_steps_degraded if p95 > threshold
            else self.config.tvm_steps_normal
        )
        return self._current_steps

    def route_request(self, request_type: Literal["text", "spatial", "video"]):
        """Route an inference request to the appropriate backend.

        Routing strategy:
          - ``text``    → SGLang RadixAttention backend (efficient KV reuse).
          - ``spatial`` → 3DGS gsplat CUDA renderer.
          - ``video``   → DiT diffusion pipeline with dynamic TVM step-scaling.
                          The NFE step count is adjusted in real-time based on
                          the current p95 latency relative to the SLO target.

        Returns a dict describing the selected backend and its parameters.
        """
        nfe_steps = self.get_nfe_steps()

        if request_type == "text":
            return {
                "backend": "sglang",
                "engine": "RadixAttention",
                "nfe_steps": None,
            }

        if request_type == "spatial":
            return {
                "backend": "gaussian",
                "engine": "gsplat_cuda",
                "nfe_steps": None,
            }

        if request_type == "video":
            return {
                "backend": "dit_tvm",
                "engine": "TVM-DiT",
                "nfe_steps": nfe_steps,
            }

        raise ValueError(f"Unknown request_type: {request_type!r}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_p95_latency(self) -> float:
        """Compute p95 latency from the Redis rolling window.

        The last ``WINDOW_SECONDS`` of latency observations are kept in a
        Redis sorted set keyed by timestamp.  We fetch all members still
        inside the window, extract their latency values and compute the
        95th-percentile.  If no observations exist we return 0.0 (healthy).
        """
        now = time.time()
        cutoff = now - WINDOW_SECONDS

        # Trim stale entries then fetch remaining ones
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(LATENCY_KEY, "-inf", cutoff)
        pipe.zrangebyscore(LATENCY_KEY, cutoff, "+inf")
        _, members = pipe.execute()

        if not members:
            return 0.0

        latencies = sorted(
            float(m.decode().split(":")[1]) if isinstance(m, bytes) else float(m.split(":")[1])
            for m in members
        )

        # p95 index (nearest-rank method)
        idx = int(math.ceil(0.95 * len(latencies))) - 1
        return latencies[max(idx, 0)]

# ======================================================================
# FastAPI application
# ======================================================================

def _build_router() -> SLORouter:
    """Create an SLORouter from environment variables or defaults."""
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    p95_target = float(os.environ.get("SLO_P95_TARGET_MS", "600"))
    degradation = float(os.environ.get("SLO_DEGRADATION_THRESHOLD", "0.85"))

    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        redis_client.ping()
        logger.info("Connected to Redis at %s:%d", redis_host, redis_port)
    except redis.ConnectionError:
        logger.warning("Redis unavailable at %s:%d — latency tracking disabled", redis_host, redis_port)
        redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

    config = SLOConfig(p95_target_ms=p95_target, degradation_threshold=degradation)
    return SLORouter(config, redis_client)


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="Cloud Inference Platform", version="0.1.0")

    class InferenceRequest(BaseModel):
        request_type: Literal["text", "spatial", "video"]
        prompt: str | None = None
        max_tokens: int = 128
        temperature: float = 0.7
        num_frames: int = 16
        resolution: list[int] = [512, 512]
        camera_pose: Dict[str, Any] | None = None

    _router_instance: SLORouter | None = None

    def _get_router() -> SLORouter:
        global _router_instance
        if _router_instance is None:
            _router_instance = _build_router()
        return _router_instance

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/infer")
    async def infer(request: InferenceRequest) -> Dict[str, Any]:
        router = _get_router()
        t_start = time.time()
        try:
            result = router.route_request(request.request_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        latency_ms = (time.time() - t_start) * 1000
        router.record_latency(latency_ms)

        return {
            "routing": result,
            "request_type": request.request_type,
            "latency_ms": round(latency_ms, 2),
        }
else:
    app = None  # uvicorn will fail with a clear error
