"""
vLLM Serving Backend

Wraps the vLLM engine for high-throughput text inference with PagedAttention.
Falls back gracefully when vllm is not installed.

Usage::

    backend = VLLMBackend(model_name="meta-llama/Llama-3-8B-Instruct")
    result = backend.generate("What is Flash Attention?")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]
    _VLLM_AVAILABLE = False
    logger.warning("vllm not installed — VLLMBackend will operate in stub mode.")


@dataclass
class VLLMConfig:
    """Configuration for the vLLM backend."""

    model_name: str = "meta-llama/Llama-3-8B-Instruct"
    paged_attention: bool = True
    max_batch_size: int = 16
    kv_cache_vram_gb: float = 21.0
    tp_size: int = 1
    max_model_len: int = 4096


class VLLMBackend:
    """Text inference backend powered by vLLM with PagedAttention.

    Parameters
    ----------
    config : VLLMConfig
        Backend configuration.  If not supplied a default is created.
    """

    def __init__(self, config: Optional[VLLMConfig] = None) -> None:
        self.config = config or VLLMConfig()
        self._llm: Any = None
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
        if not _VLLM_AVAILABLE or self._llm is None:
            return self._stub_generate(prompt, max_tokens)

        try:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            outputs = self._llm.generate([prompt], params)
            output_text = outputs[0].outputs[0].text if outputs else ""
            token_count = len(outputs[0].outputs[0].token_ids) if outputs else 0
            return {
                "text": output_text,
                "tokens_generated": token_count,
                "model": self.config.model_name,
            }
        except Exception as exc:
            logger.error("vLLM generation failed: %s", exc)
            return {"text": "", "tokens_generated": 0, "model": self.config.model_name, "error": str(exc)}

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Generate completions for a batch of prompts.

        When vLLM is available this uses its native batching; otherwise
        falls back to sequential stub calls.
        """
        if not _VLLM_AVAILABLE or self._llm is None:
            return [self._stub_generate(p, max_tokens) for p in prompts]

        try:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )
            outputs = self._llm.generate(prompts, params)
            results: List[Dict[str, Any]] = []
            for out in outputs:
                results.append({
                    "text": out.outputs[0].text,
                    "tokens_generated": len(out.outputs[0].token_ids),
                    "model": self.config.model_name,
                })
            return results
        except Exception as exc:
            logger.error("vLLM batch generation failed: %s", exc)
            return [{"text": "", "tokens_generated": 0, "model": self.config.model_name, "error": str(exc)}
                    for _ in prompts]

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model": self.config.model_name,
            "tp_size": self.config.tp_size,
            "max_model_len": self.config.max_model_len,
            "paged_attention": self.config.paged_attention,
            "kv_cache_vram_gb": self.config.kv_cache_vram_gb,
        }

    def health_check(self) -> Dict[str, Any]:
        """Return backend health status."""
        return {
            "backend": "vllm",
            "available": _VLLM_AVAILABLE,
            "engine_loaded": self._llm is not None,
            "model": self.config.model_name,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        if not _VLLM_AVAILABLE:
            return
        try:
            self._llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tp_size,
                max_model_len=self.config.max_model_len,
            )
            logger.info("vLLM engine initialized: %s", self.config.model_name)
        except Exception as exc:
            logger.error("Failed to initialize vLLM engine: %s", exc)
            self._llm = None

    @staticmethod
    def _stub_generate(prompt: str, max_tokens: int) -> Dict[str, Any]:
        return {
            "text": f"[STUB] vLLM not available. Prompt length={len(prompt)}",
            "tokens_generated": 0,
            "model": "stub",
        }
