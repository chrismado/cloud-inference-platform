"""
Prototype TVM-backed diffusion serving backend.

This backend keeps the repo honest about its current state: the routing layer can
exercise a video-generation execution path, but the actual DiT/TVM model serving
integration is still a prototype. The work is isolated here rather than hidden in
the router so the control plane and execution stub are easier to reason about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DiTTVMConfig:
    """Configuration for the prototype DiT/TVM backend."""

    base_step_latency_s: float = 0.002
    engine_name: str = "TVM-DiT"


class DiTTVMBackend:
    """Prototype diffusion backend used by the router's video path."""

    def __init__(self, config: Optional[DiTTVMConfig] = None) -> None:
        self.config = config or DiTTVMConfig()

    def execute(
        self,
        prompt: str = "",
        num_frames: int = 16,
        nfe_steps: int = 4,
    ) -> Dict[str, Any]:
        """Simulate prototype execution while keeping the limitation explicit."""
        del prompt  # Prompt wiring is part of the future real backend integration.

        clamped_steps = max(int(nfe_steps), 1)
        time.sleep(self.config.base_step_latency_s * clamped_steps)

        return {
            "engine": self.config.engine_name,
            "nfe_steps": clamped_steps,
            "num_frames": int(num_frames),
            "status": "prototype_execution",
            "execution_mode": "explicit_backend_stub",
        }

    def health_check(self) -> Dict[str, Any]:
        """Return backend status with explicit prototype framing."""
        return {
            "backend": "dit_tvm",
            "engine": self.config.engine_name,
            "status": "prototype_backend",
            "base_step_latency_s": self.config.base_step_latency_s,
        }
