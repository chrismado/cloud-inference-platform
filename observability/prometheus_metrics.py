"""
Prometheus Metrics Integration

Exposes request latency, GPU utilisation, SLO violations, and throughput
counters via the ``prometheus_client`` library.  Falls back to no-op
when the library is not installed.

Usage::

    metrics = PrometheusMetrics()
    metrics.record_request("text", latency_ms=42.5)
    metrics.record_gpu_util(0, 0.73)
    metrics.start_server(port=9090)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics will be no-ops.")


class PrometheusMetrics:
    """Manages Prometheus metric instruments for the inference platform.

    All recording methods are safe to call even when ``prometheus_client``
    is not installed — they silently become no-ops.
    """

    def __init__(self, namespace: str = "cip") -> None:
        self.namespace = namespace

        if not _PROM_AVAILABLE:
            self._request_latency = None
            self._request_count = None
            self._gpu_utilization = None
            self._slo_violations = None
            self._active_requests = None
            return

        self._request_latency = Histogram(
            f"{namespace}_request_latency_ms",
            "Inference request latency in milliseconds",
            labelnames=["backend", "request_type"],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
        )

        self._request_count = Counter(
            f"{namespace}_requests_total",
            "Total number of inference requests",
            labelnames=["backend", "request_type", "status"],
        )

        self._gpu_utilization = Gauge(
            f"{namespace}_gpu_utilization",
            "Current GPU utilization (0-1)",
            labelnames=["gpu_index"],
        )

        self._slo_violations = Counter(
            f"{namespace}_slo_violations_total",
            "Total SLO target violations",
            labelnames=["request_type"],
        )

        self._active_requests = Gauge(
            f"{namespace}_active_requests",
            "Number of currently in-flight requests",
            labelnames=["backend"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(
        self,
        request_type: str,
        latency_ms: float,
        backend: str = "unknown",
        status: str = "ok",
    ) -> None:
        """Record a completed inference request."""
        if not _PROM_AVAILABLE:
            return
        self._request_latency.labels(backend=backend, request_type=request_type).observe(latency_ms)
        self._request_count.labels(backend=backend, request_type=request_type, status=status).inc()

    def record_gpu_util(self, gpu_index: int, utilization: float) -> None:
        """Update the GPU utilization gauge for a specific device."""
        if not _PROM_AVAILABLE:
            return
        self._gpu_utilization.labels(gpu_index=str(gpu_index)).set(utilization)

    def record_slo_violation(self, request_type: str) -> None:
        """Increment the SLO violation counter."""
        if not _PROM_AVAILABLE:
            return
        self._slo_violations.labels(request_type=request_type).inc()

    def set_active_requests(self, backend: str, count: int) -> None:
        """Set the current in-flight request count for a backend."""
        if not _PROM_AVAILABLE:
            return
        self._active_requests.labels(backend=backend).set(count)

    def start_server(self, port: int = 9090) -> None:
        """Start the Prometheus HTTP metrics server.

        Parameters
        ----------
        port : int
            Port to bind the metrics endpoint on.
        """
        if not _PROM_AVAILABLE:
            logger.warning("Cannot start Prometheus server — prometheus_client not installed.")
            return
        start_http_server(port)
        logger.info("Prometheus metrics server started on :%d", port)
