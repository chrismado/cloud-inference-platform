from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from router.priority_queue import InferenceRequest, PriorityRequestQueue
from router.slo_router import SLOConfig, SLORouter, app

try:
    import fakeredis
except ImportError:  # pragma: no cover - dependency added in phase 2
    fakeredis = None

pytestmark = pytest.mark.integration


def _build_fake_router() -> SLORouter:
    if fakeredis is None:
        raise RuntimeError("fakeredis must be installed for integration tests")
    fake_redis = fakeredis.FakeRedis(decode_responses=False)
    return SLORouter(SLOConfig(p95_target_ms=100.0, degradation_threshold=0.5), fake_redis)


def test_router_p95_computation_and_priority_queue_order(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _build_fake_router()

    for latency_ms in range(1, 101):
        router.record_latency(float(latency_ms))

    assert router._get_current_p95_latency() == pytest.approx(95.0, abs=1.0)

    queue = PriorityRequestQueue()
    for idx in range(50):
        queue.enqueue(InferenceRequest(id=f"r{idx}", priority=idx % 5, payload={"idx": idx}))

    priorities = []
    while queue:
        request = queue.dequeue()
        assert request is not None
        priorities.append(request.priority)

    assert priorities == sorted(priorities)


def test_fastapi_endpoints_with_fake_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _build_fake_router()
    monkeypatch.setattr("router.slo_router._router_instance", router, raising=False)

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    text_response = client.post("/infer", json={"request_type": "text", "prompt": "hello"})
    assert text_response.status_code == 200
    assert text_response.json()["routing"]["backend"] == "sglang"

    for latency_ms in range(1, 101):
        router.record_latency(float(latency_ms))

    video_response = client.post("/infer", json={"request_type": "video", "prompt": "generate"})
    assert video_response.status_code == 200
    assert video_response.json()["routing"]["nfe_steps"] == 1

    invalid_response = client.post("/infer", json={"request_type": "invalid"})
    assert invalid_response.status_code == 400
