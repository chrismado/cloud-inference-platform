from __future__ import annotations

from pathlib import Path

import pytest

from router.load_monitor import AsyncGPULoadMonitor, GPUStats
from serving.gaussian_backend import GaussianSplatBackend


def test_gaussian_backend_rejects_missing_model_path() -> None:
    backend = GaussianSplatBackend()

    with pytest.raises(FileNotFoundError):
        backend.load_model("missing-scene.ply")

    health = backend.health_check()
    assert health["render_mode"] == "unloaded"
    assert health["model_loaded"] is False


def test_gaussian_backend_reports_stub_mode_for_existing_model(tmp_path: Path) -> None:
    model_path = tmp_path / "scene.ply"
    model_path.write_text("ply placeholder\n", encoding="utf-8")

    backend = GaussianSplatBackend()
    backend.load_model(str(model_path))

    health = backend.health_check()
    assert health["model_loaded"] is True
    assert health["model_path"] == str(model_path)
    assert health["render_mode"] in {"deterministic_stub", "prototype_stub"}


def test_async_gpu_monitor_uses_latest_snapshot_before_repolling() -> None:
    monitor = AsyncGPULoadMonitor(overload_threshold=0.75)
    monitor.latest = {
        0: GPUStats(index=0, utilization_pct=0.90),
        1: GPUStats(index=1, utilization_pct=0.80),
    }
    monitor._monitor.is_overloaded = lambda: False  # type: ignore[method-assign]

    assert monitor.is_overloaded() is True

    monitor.latest = {
        0: GPUStats(index=0, utilization_pct=0.90),
        1: GPUStats(index=1, utilization_pct=0.20),
    }
    assert monitor.is_overloaded() is False
