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

try:
    import redis  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional in tests / CPU-only envs
    redis = None  # type: ignore[assignment]

try:
    import fakeredis
except ImportError:
    fakeredis = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

LATENCY_KEY_SUFFIX = "latency_window"
WINDOW_SECONDS = 60


@dataclass
class SLOConfig:
    p95_target_ms: float = 600.0
    tvm_steps_normal: int = 4
    tvm_steps_degraded: int = 1
    degradation_threshold: float = 0.85
    latency_key_prefix: str = "slo"
    instance_id: str = "default"


class SLORouter:
    def __init__(self, config: SLOConfig, redis_client: Any):
        self.config = config
        self.redis = redis_client
        self._current_steps = config.tvm_steps_normal
        self._backend_registry: Dict[str, Any] = {}

    @property
    def latency_key(self) -> str:
        return f"{self.config.latency_key_prefix}:{self.config.instance_id}:{LATENCY_KEY_SUFFIX}"

    def record_latency(self, latency_ms: float) -> None:
        """Push a latency observation into the Redis rolling window."""
        now = time.time()
        member = f"{now}:{latency_ms}"
        pipe = self.redis.pipeline()
        pipe.zadd(self.latency_key, {member: now})
        pipe.zremrangebyscore(self.latency_key, "-inf", now - WINDOW_SECONDS)
        pipe.execute()

    def get_nfe_steps(self) -> int:
        """Return the active NFE step count based on the rolling p95 latency."""
        p95 = self._get_current_p95_latency()
        threshold = self.config.p95_target_ms * self.config.degradation_threshold
        self._current_steps = self.config.tvm_steps_degraded if p95 > threshold else self.config.tvm_steps_normal
        return self._current_steps

    def route_request(self, request_type: Literal["text", "spatial", "video"]) -> Dict[str, Any]:
        """Route an inference request to the appropriate backend."""
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

    def process_request(
        self,
        request_type: Literal["text", "spatial", "video"],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route and execute a request, recording backend latency instead of router overhead."""
        route = self.route_request(request_type)
        start = time.perf_counter()
        backend_result = self._execute_backend(route, payload)
        latency_ms = (time.perf_counter() - start) * 1000
        self.record_latency(latency_ms)

        return {
            "routing": route,
            "backend_result": backend_result,
            "request_type": request_type,
            "latency_ms": round(latency_ms, 2),
        }

    def _get_current_p95_latency(self) -> float:
        """Compute p95 latency from the Redis rolling window."""
        now = time.time()
        cutoff = now - WINDOW_SECONDS

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(self.latency_key, "-inf", cutoff)
        pipe.zrangebyscore(self.latency_key, cutoff, "+inf")
        _, members = pipe.execute()

        if not members:
            return 0.0

        latencies = sorted(
            float(member.decode().split(":")[1]) if isinstance(member, bytes) else float(member.split(":")[1])
            for member in members
        )

        idx = int(math.ceil(0.95 * len(latencies))) - 1
        return latencies[max(idx, 0)]

    def _get_backend(self, name: str) -> Any:
        if name in self._backend_registry:
            return self._backend_registry[name]

        backend: Any
        if name == "sglang":
            from serving.sglang_backend import SGLangBackend

            backend = SGLangBackend()
        elif name == "gaussian":
            from serving.gaussian_backend import GaussianSplatBackend

            backend = GaussianSplatBackend()
        else:
            raise ValueError(f"Unknown backend: {name}")

        self._backend_registry[name] = backend
        return backend

    def _execute_backend(self, route: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        backend_name = route["backend"]

        if backend_name == "sglang":
            backend = self._get_backend("sglang")
            return backend.generate(
                payload.get("prompt", ""),
                max_tokens=payload.get("max_tokens", 128),
                temperature=payload.get("temperature", 0.7),
            )

        if backend_name == "gaussian":
            backend = self._get_backend("gaussian")
            model_path = payload.get("model_path")
            if model_path and backend.health_check().get("model_path") != model_path:
                backend.load_model(model_path)
            image = backend.render(
                camera_pose=payload.get("camera_pose"),
                resolution=tuple(payload.get("resolution", [512, 512])),
            )
            return {
                "image_shape": list(image.shape),
                "engine": route["engine"],
                "model_path": backend.health_check().get("model_path"),
            }

        if backend_name == "dit_tvm":
            nfe_steps = int(route.get("nfe_steps") or self.config.tvm_steps_normal)
            # The repo does not yet expose a real video backend, so keep this
            # prototype path explicit while still timing real work rather than a
            # dict lookup.
            time.sleep(0.002 * max(nfe_steps, 1))
            return {
                "engine": route["engine"],
                "nfe_steps": nfe_steps,
                "status": "prototype_execution",
            }

        raise ValueError(f"Unsupported backend: {backend_name}")


def _build_router() -> SLORouter:
    """Create an SLORouter from environment variables or defaults."""
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    p95_target = float(os.environ.get("SLO_P95_TARGET_MS", "600"))
    degradation = float(os.environ.get("SLO_DEGRADATION_THRESHOLD", "0.85"))
    key_prefix = os.environ.get("SLO_KEY_PREFIX", "slo")
    instance_id = (
        os.environ.get("SLO_INSTANCE_ID") or os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "default"
    )

    if redis is not None:
        try:
            redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
            redis_client.ping()
            logger.info("Connected to Redis at %s:%d", redis_host, redis_port)
        except redis.ConnectionError:
            if fakeredis is not None:
                logger.warning(
                    "Redis unavailable at %s:%d; using fakeredis fallback for local operation",
                    redis_host,
                    redis_port,
                )
                redis_client = fakeredis.FakeRedis(decode_responses=False)
            else:
                logger.warning(
                    "Redis unavailable at %s:%d and fakeredis is not installed",
                    redis_host,
                    redis_port,
                )
                redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    elif fakeredis is not None:
        logger.warning("redis package unavailable; using fakeredis fallback for local operation")
        redis_client = fakeredis.FakeRedis(decode_responses=False)
    else:
        raise RuntimeError("redis or fakeredis is required to build the router")

    config = SLOConfig(
        p95_target_ms=p95_target,
        degradation_threshold=degradation,
        latency_key_prefix=key_prefix,
        instance_id=instance_id,
    )
    return SLORouter(config, redis_client)


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="Cloud Inference Platform", version="0.1.0")

    class InferenceRequest(BaseModel):
        request_type: str
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
        try:
            result = router.process_request(
                request.request_type,  # type: ignore[arg-type]
                {
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "num_frames": request.num_frames,
                    "resolution": request.resolution,
                    "camera_pose": request.camera_pose,
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return result
else:
    app = None  # uvicorn will fail with a clear error
