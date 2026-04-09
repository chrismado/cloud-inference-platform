"""
SGLang Serving Backend

Wraps the SGLang RadixAttention engine for efficient text inference with
KV-cache reuse.  Falls back gracefully when sglang is not installed.

Usage::

    backend = SGLangBackend(model_path="meta-llama/Llama-3-8B-Instruct")
    result = backend.generate("Explain quantum entanglement.")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import sglang as sgl

    _SGLANG_AVAILABLE = True
except ImportError:
    sgl = None  # type: ignore[assignment]
    _SGLANG_AVAILABLE = False
    logger.warning("sglang not installed — SGLangBackend will operate in stub mode.")


@dataclass
class SGLangConfig:
    """Configuration for the SGLang backend."""

    model_path: str = "meta-llama/Llama-3-8B-Instruct"
    radix_attention: bool = True
    max_batch_size: int = 32
    kv_cache_vram_gb: float = 7.0
    tp_size: int = 1


class SGLangBackend:
    """Text inference backend powered by SGLang with RadixAttention.

    Parameters
    ----------
    config : SGLangConfig
        Backend configuration.  If not supplied a default is created.
    """

    def __init__(self, config: Optional[SGLangConfig] = None) -> None:
        self.config = config or SGLangConfig()
        self._engine: Any = None
        self._initialize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Generate a single completion.

        Returns a dict with ``text``, ``tokens_generated``, and ``model``.
        """
        if not _SGLANG_AVAILABLE or self._engine is None:
            return self._stub_generate(prompt, max_tokens)

        try:
            result = self._engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return {
                "text": result.get("text", ""),
                "tokens_generated": result.get("usage", {}).get("completion_tokens", 0),
                "model": self.config.model_path,
            }
        except Exception as exc:
            logger.error("SGLang generation failed: %s", exc)
            return {"text": "", "tokens_generated": 0, "model": self.config.model_path, "error": str(exc)}

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Generate completions for a batch of prompts."""
        return [self.generate(p, max_tokens=max_tokens, temperature=temperature) for p in prompts]

    def health_check(self) -> Dict[str, Any]:
        """Return backend health status."""
        return {
            "backend": "sglang",
            "available": _SGLANG_AVAILABLE,
            "engine_loaded": self._engine is not None,
            "model": self.config.model_path,
        }

    def shutdown(self) -> None:
        """Gracefully shut down the engine and release resources."""
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception as exc:
                logger.warning("SGLang shutdown error: %s", exc)
            self._engine = None
            logger.info("SGLang engine shut down.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Attempt to start the SGLang runtime."""
        if not _SGLANG_AVAILABLE:
            return
        try:
            self._engine = sgl.Engine(
                model_path=self.config.model_path,
                tp_size=self.config.tp_size,
            )
            logger.info("SGLang engine initialized: %s", self.config.model_path)
        except Exception as exc:
            logger.error("Failed to initialize SGLang engine: %s", exc)
            self._engine = None

    @staticmethod
    def _stub_generate(prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Return a placeholder when SGLang is unavailable."""
        return {
            "text": f"[STUB] SGLang not available. Prompt length={len(prompt)}",
            "tokens_generated": 0,
            "model": "stub",
        }
