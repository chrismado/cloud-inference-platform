"""
SLO Tracker — Sliding-Window Latency Compliance Monitor

Tracks inference latency observations in a time-bounded sliding window
and computes compliance rates, percentiles, and violation status without
requiring Redis (pure in-memory).

Usage::

    tracker = SLOTracker(p95_target_ms=600.0, window_seconds=60)
    tracker.record_latency(latency_ms=123.4)
    print(tracker.get_compliance_rate())
    print(tracker.get_p95_latency())
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Tuple


@dataclass
class SLOTrackerConfig:
    """Configuration for the SLO tracker."""

    p95_target_ms: float = 600.0
    window_seconds: float = 60.0


class SLOTracker:
    """Sliding-window SLO compliance tracker.

    Thread-safe via an internal lock.  All observations older than
    ``window_seconds`` are automatically pruned on each operation.

    Parameters
    ----------
    config : SLOTrackerConfig
        Tracker configuration.  If not supplied a default is created.
    """

    def __init__(self, config: SLOTrackerConfig | None = None) -> None:
        self.config = config or SLOTrackerConfig()
        self._observations: Deque[Tuple[float, float]] = deque()  # (timestamp, latency_ms)
        self._lock = threading.Lock()
        self._violation_count: int = 0
        self._total_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency observation.

        Automatically prunes stale entries and updates violation counters.
        """
        now = time.time()
        with self._lock:
            self._observations.append((now, latency_ms))
            self._total_count += 1
            if latency_ms > self.config.p95_target_ms:
                self._violation_count += 1
            self._prune(now)

    def get_compliance_rate(self) -> float:
        """Return the fraction of in-window requests meeting the SLO target.

        Returns 1.0 when there are no observations (vacuously compliant).
        """
        with self._lock:
            self._prune(time.time())
            if not self._observations:
                return 1.0
            compliant = sum(1 for _, lat in self._observations if lat <= self.config.p95_target_ms)
            return compliant / len(self._observations)

    def get_p95_latency(self) -> float:
        """Compute the p95 latency from the current window.

        Returns 0.0 when there are no observations.
        """
        with self._lock:
            self._prune(time.time())
            if not self._observations:
                return 0.0
            latencies = sorted(lat for _, lat in self._observations)
            idx = int(math.ceil(0.95 * len(latencies))) - 1
            return latencies[max(idx, 0)]

    def is_violating(self) -> bool:
        """Return ``True`` if the current p95 exceeds the SLO target."""
        return self.get_p95_latency() > self.config.p95_target_ms

    def get_report(self) -> Dict[str, Any]:
        """Return a summary dict of current SLO health.

        Keys: ``p95_ms``, ``compliance_rate``, ``violating``,
        ``window_size``, ``total_observed``, ``total_violations``.
        """
        with self._lock:
            self._prune(time.time())
            latencies = sorted(lat for _, lat in self._observations)
            n = len(latencies)
            if n == 0:
                p95 = 0.0
            else:
                idx = int(math.ceil(0.95 * n)) - 1
                p95 = latencies[max(idx, 0)]
            compliant = sum(1 for lat in latencies if lat <= self.config.p95_target_ms)
            compliance = compliant / n if n > 0 else 1.0

        return {
            "p95_ms": p95,
            "compliance_rate": compliance,
            "violating": p95 > self.config.p95_target_ms,
            "window_size": n,
            "total_observed": self._total_count,
            "total_violations": self._violation_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Remove observations older than the sliding window."""
        cutoff = now - self.config.window_seconds
        while self._observations and self._observations[0][0] < cutoff:
            self._observations.popleft()
