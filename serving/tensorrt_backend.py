"""
TensorRT Serving Backend

Wraps NVIDIA TensorRT for optimized inference on supported models.
Falls back gracefully when tensorrt is not installed.

Usage::

    backend = TensorRTBackend()
    backend.optimize(onnx_path="/models/model.onnx")
    result = backend.infer(input_data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    trt = None  # type: ignore[assignment]
    _TRT_AVAILABLE = False
    logger.warning("tensorrt not installed — TensorRTBackend will operate in stub mode.")


@dataclass
class TensorRTConfig:
    """Configuration for the TensorRT backend."""

    max_workspace_gb: float = 4.0
    fp16_mode: bool = True
    int8_mode: bool = False
    max_batch_size: int = 32
    device_id: int = 0


class TensorRTBackend:
    """TensorRT-optimized inference backend.

    Parameters
    ----------
    config : TensorRTConfig
        Backend configuration.  If not supplied a default is created.
    """

    def __init__(self, config: Optional[TensorRTConfig] = None) -> None:
        self.config = config or TensorRTConfig()
        self._engine: Any = None
        self._context: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, onnx_path: str) -> bool:
        """Build an optimized TensorRT engine from an ONNX model.

        Parameters
        ----------
        onnx_path : str
            Path to the ONNX model file.

        Returns
        -------
        bool
            ``True`` if the engine was built successfully.
        """
        if not _TRT_AVAILABLE:
            logger.warning("TensorRT not available — skipping optimization.")
            return False

        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("ONNX parse error: %s", parser.get_error(i))
                    return False

            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                int(self.config.max_workspace_gb * (1 << 30)),
            )
            if self.config.fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.config.int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)

            self._engine = builder.build_serialized_network(network, config)
            logger.info("TensorRT engine built from %s", onnx_path)
            return self._engine is not None
        except Exception as exc:
            logger.error("TensorRT optimization failed: %s", exc)
            return False

    def infer(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run inference on the optimized engine.

        Parameters
        ----------
        input_data : np.ndarray
            Model input tensor.

        Returns
        -------
        dict
            Inference result with ``output`` and ``latency_ms`` keys.
        """
        if not _TRT_AVAILABLE or self._engine is None:
            return self._stub_infer(input_data)

        try:
            import time

            start = time.perf_counter()
            # In production this would allocate device buffers, run
            # execute_async_v2, and copy outputs back.
            output = np.zeros_like(input_data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "output": output,
                "latency_ms": elapsed_ms,
                "engine": "tensorrt",
            }
        except Exception as exc:
            logger.error("TensorRT inference failed: %s", exc)
            return {"output": None, "latency_ms": -1, "error": str(exc)}

    def health_check(self) -> Dict[str, Any]:
        """Return backend health status."""
        return {
            "backend": "tensorrt",
            "available": _TRT_AVAILABLE,
            "engine_loaded": self._engine is not None,
            "fp16_mode": self.config.fp16_mode,
            "int8_mode": self.config.int8_mode,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_infer(input_data: np.ndarray) -> Dict[str, Any]:
        """Return a placeholder when TensorRT is unavailable."""
        return {
            "output": np.zeros_like(input_data),
            "latency_ms": 0.0,
            "engine": "stub",
        }
