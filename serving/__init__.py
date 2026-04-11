"""
Serving backends — unified interface for heterogeneous inference engines.
"""

from __future__ import annotations

from typing import Any


def _optional_import(module_name: str, symbol_name: str) -> Any:
    try:
        module = __import__(module_name, fromlist=[symbol_name])
        return getattr(module, symbol_name)
    except ImportError:
        return None


GaussianSplatBackend = _optional_import("serving.gaussian_backend", "GaussianSplatBackend")
SGLangBackend = _optional_import("serving.sglang_backend", "SGLangBackend")
TensorRTBackend = _optional_import("serving.tensorrt_backend", "TensorRTBackend")
VLLMBackend = _optional_import("serving.vllm_backend", "VLLMBackend")

__all__ = [
    "SGLangBackend",
    "VLLMBackend",
    "GaussianSplatBackend",
    "TensorRTBackend",
]
