"""
Unit tests for SLORouter with mocked Redis.

Usage::

    python -m pytest tests/test_router.py -v
    python -m unittest tests.test_router
"""

from __future__ import annotations

import unittest

from router.slo_router import SLOConfig, SLORouter


class FakeRedis:
    """Minimal Redis mock that supports sorted-set operations used by SLORouter."""

    def __init__(self):
        self._store: dict = {}

    def pipeline(self) -> "FakePipeline":
        return FakePipeline(self)

    def zadd(self, key: str, mapping: dict) -> int:
        if key not in self._store:
            self._store[key] = {}
        self._store[key].update(mapping)
        return len(mapping)

    def zremrangebyscore(self, key: str, min_score: str, max_score: float) -> int:
        if key not in self._store:
            return 0
        to_remove = [m for m, score in self._store[key].items() if score <= max_score]
        for m in to_remove:
            del self._store[key][m]
        return len(to_remove)

    def zrangebyscore(self, key: str, min_score: float, max_score: str):
        if key not in self._store:
            return []
        return [m for m, score in self._store[key].items() if score >= min_score]


class FakePipeline:
    """Fake Redis pipeline that batches commands and returns results."""

    def __init__(self, redis: FakeRedis):
        self._redis = redis
        self._commands: list = []

    def zadd(self, key, mapping):
        self._commands.append(("zadd", key, mapping))
        return self

    def zremrangebyscore(self, key, min_s, max_s):
        self._commands.append(("zremrangebyscore", key, min_s, max_s))
        return self

    def zrangebyscore(self, key, min_s, max_s):
        self._commands.append(("zrangebyscore", key, min_s, max_s))
        return self

    def execute(self):
        results = []
        for cmd in self._commands:
            if cmd[0] == "zadd":
                results.append(self._redis.zadd(cmd[1], cmd[2]))
            elif cmd[0] == "zremrangebyscore":
                results.append(self._redis.zremrangebyscore(cmd[1], cmd[2], cmd[3]))
            elif cmd[0] == "zrangebyscore":
                results.append(self._redis.zrangebyscore(cmd[1], cmd[2], cmd[3]))
        self._commands.clear()
        return results


class TestSLORouter(unittest.TestCase):
    """Tests for the SLO-aware GPU router."""

    def setUp(self):
        self.redis = FakeRedis()
        self.config = SLOConfig(
            p95_target_ms=600.0,
            tvm_steps_normal=4,
            tvm_steps_degraded=1,
            degradation_threshold=0.85,
        )
        self.router = SLORouter(self.config, self.redis)

    def test_route_text_request(self):
        result = self.router.route_request("text")
        self.assertEqual(result["backend"], "sglang")
        self.assertEqual(result["engine"], "RadixAttention")
        self.assertIsNone(result["nfe_steps"])

    def test_route_spatial_request(self):
        result = self.router.route_request("spatial")
        self.assertEqual(result["backend"], "gaussian")
        self.assertEqual(result["engine"], "gsplat_cuda")

    def test_route_video_request_normal_load(self):
        """Under normal load the router should use full NFE steps."""
        result = self.router.route_request("video")
        self.assertEqual(result["backend"], "dit_tvm")
        self.assertEqual(result["nfe_steps"], 4)

    def test_route_video_request_degraded(self):
        """When p95 exceeds the threshold the router should degrade to 1 step."""
        # Inject high-latency observations to trigger degradation
        threshold = self.config.p95_target_ms * self.config.degradation_threshold
        for _ in range(100):
            self.router.record_latency(threshold + 100)

        result = self.router.route_request("video")
        self.assertEqual(result["nfe_steps"], 1)

    def test_unknown_request_type_raises(self):
        with self.assertRaises(ValueError):
            self.router.route_request("unknown")

    def test_record_latency_stores_observations(self):
        self.router.record_latency(100.0)
        self.router.record_latency(200.0)
        p95 = self.router._get_current_p95_latency()
        self.assertGreater(p95, 0.0)

    def test_no_observations_returns_zero(self):
        p95 = self.router._get_current_p95_latency()
        self.assertEqual(p95, 0.0)

    def test_get_nfe_steps_normal(self):
        steps = self.router.get_nfe_steps()
        self.assertEqual(steps, 4)


if __name__ == "__main__":
    unittest.main()
