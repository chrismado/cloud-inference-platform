"""
GPU Load Monitor

Monitors GPU utilization and memory usage across all available devices
using pynvml. Falls back to dummy readings when pynvml is unavailable
(e.g. in CI or CPU-only environments).

Usage::

    monitor = GPULoadMonitor(overload_threshold=0.90)
    print(monitor.get_utilization())
    print(monitor.get_available_gpus())
    print(monitor.is_overloaded())
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


@dataclass
class GPUStats:
    """Snapshot of a single GPU's utilization."""

    index: int
    name: str = "unknown"
    utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: int = 0


class GPULoadMonitor:
    """Polls GPU metrics via pynvml with a graceful dummy fallback.

    Parameters
    ----------
    overload_threshold : float
        GPU utilization percentage (0-1) above which the system is
        considered overloaded.  Default 0.90.
    """

    def __init__(self, overload_threshold: float = 0.90) -> None:
        self.overload_threshold = overload_threshold
        self._device_count = self._get_device_count()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_utilization(self) -> Dict[int, GPUStats]:
        """Return a ``{gpu_index: GPUStats}`` dict for every visible GPU."""
        stats: Dict[int, GPUStats] = {}
        for i in range(self._device_count):
            stats[i] = self._read_gpu(i)
        return stats

    def get_available_gpus(self) -> List[int]:
        """Return indices of GPUs whose utilization is below the overload threshold."""
        available: List[int] = []
        for idx, gpu in self.get_utilization().items():
            if gpu.utilization_pct < self.overload_threshold:
                available.append(idx)
        return available

    def is_overloaded(self) -> bool:
        """Return ``True`` if **all** GPUs exceed the overload threshold."""
        if self._device_count == 0:
            return False
        return len(self.get_available_gpus()) == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_device_count() -> int:
        if not _NVML_AVAILABLE:
            return 0
        try:
            return pynvml.nvmlDeviceGetCount()
        except Exception:
            return 0

    @staticmethod
    def _read_gpu(index: int) -> GPUStats:
        if not _NVML_AVAILABLE:
            return GPUStats(index=index)
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            return GPUStats(
                index=index,
                name=name,
                utilization_pct=util.gpu / 100.0,
                memory_used_mb=mem.used / (1024 * 1024),
                memory_total_mb=mem.total / (1024 * 1024),
                temperature_c=temp,
            )
        except Exception:
            return GPUStats(index=index)


class AsyncGPULoadMonitor:
    """Asynchronous wrapper that polls GPU metrics on a background loop.

    Usage::

        async_monitor = AsyncGPULoadMonitor(poll_interval=2.0)
        await async_monitor.start()
        latest = async_monitor.latest
        await async_monitor.stop()
    """

    def __init__(
        self,
        overload_threshold: float = 0.90,
        poll_interval: float = 2.0,
    ) -> None:
        self._monitor = GPULoadMonitor(overload_threshold=overload_threshold)
        self._poll_interval = poll_interval
        self._task: Optional[asyncio.Task] = None
        self.latest: Dict[int, GPUStats] = {}

    async def start(self) -> None:
        """Begin the background polling loop."""
        if self._task is None:
            self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel the background polling loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def is_overloaded(self) -> bool:
        """Check overload based on the most recent snapshot."""
        return self._monitor.is_overloaded()

    async def _poll_loop(self) -> None:
        while True:
            self.latest = self._monitor.get_utilization()
            await asyncio.sleep(self._poll_interval)
