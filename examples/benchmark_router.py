from __future__ import annotations

import random

from router.slo_router import SLOConfig, SLORouter

try:
    import fakeredis
except ImportError as exc:  # pragma: no cover - dependency added in phase 2
    raise SystemExit("fakeredis is required for benchmark_router.py") from exc


def main() -> None:
    router = SLORouter(SLOConfig(p95_target_ms=120.0, degradation_threshold=0.75), fakeredis.FakeRedis())
    latencies = [random.uniform(40.0, 160.0) for _ in range(200)]
    for latency in latencies:
        router.record_latency(latency)

    p95 = router._get_current_p95_latency()
    selected_steps = router.get_nfe_steps()

    print("Recorded samples:", len(latencies))
    print("Approx p95 latency (ms):", f"{p95:.2f}")
    print("Selected NFE steps:", selected_steps)


if __name__ == "__main__":
    main()
