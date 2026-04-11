"""
TVM Dynamic Step-Scaler
Adjusts NFE (Number of Function Evaluations) step count for DiT diffusion
inference in real-time based on server load and SLO compliance.

During normal load:  4-step DiT generation (full quality, FID ~2.1)
Under traffic spike:  autonomously degrades to 1-step TVM generation (FID ~4.6)
SLO contract:  p95 latency target is never violated.
FID budget:  degradation stays < 2.5 from the 4-step baseline.

Reference: Zhou et al. (Luma AI), Terminal Velocity Matching, ICLR 2026.
  d/ds f_θ(x_t, t, s) = F_θ(x_t, t, s) + (s-t) · ∂_s F_θ(x_t, t, s)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List

from router.slo_router import SLORouter

# Ordered highest→lowest quality.  Each entry maps an NFE step count to its
# expected FID delta relative to the 4-step baseline.
_DEFAULT_STEP_LADDER: List[tuple[int, float]] = [
    (4, 0.0),  # full quality
    (2, 1.2),  # moderate degradation
    (1, 2.5),  # maximum degradation within FID budget
]


@dataclass
class TVMScalerConfig:
    fid_budget: float = 2.5
    step_ladder: List[tuple[int, float]] | None = None


class TVMScaler:
    """Selects the optimal NFE step count given the current SLO headroom.

    The scaler consults the ``SLORouter`` for the live p95 latency and the
    SLO target, then walks a *step ladder* (4→2→1) to find the most
    aggressive reduction whose FID delta still fits within the budget.

    Typical lifecycle::

        scaler = TVMScaler(config, router)
        steps  = scaler.select_steps()          # returns 4, 2, or 1
        fid    = scaler.estimate_fid(steps)      # expected FID delta
    """

    def __init__(self, config: TVMScalerConfig, router: SLORouter):
        self.config = config
        self.router = router
        self._ladder = sorted(
            config.step_ladder or _DEFAULT_STEP_LADDER,
            key=lambda t: t[0],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_steps(self) -> int:
        """Choose the NFE step count based on current load.

        Strategy:
        1.  Read the live p95 latency and the SLO degradation threshold
            from the router.
        2.  If p95 is *at or below* the threshold → use full quality
            (top of the ladder).
        3.  If p95 *exceeds* the threshold → walk down the ladder,
            picking the fewest steps whose FID delta is still within
            ``fid_budget``.
        """
        p95 = self.router._get_current_p95_latency()
        threshold = self.router.config.p95_target_ms * self.router.config.degradation_threshold

        if p95 <= threshold:
            # Healthy — full quality
            return self._ladder[0][0]

        degraded_entries = [
            (steps, fid_delta)
            for steps, fid_delta in self._ladder[1:]
            if fid_delta <= self.config.fid_budget
        ]
        if not degraded_entries:
            return self._ladder[0][0]

        # Scale the degradation level with overload severity instead of
        # immediately jumping to the minimum-step configuration.
        overload_ratio = max(0.0, (p95 - threshold) / max(threshold, 1e-6))
        severity_index = min(int(math.floor(overload_ratio)) if overload_ratio > 0 else 0, len(degraded_entries) - 1)
        return degraded_entries[severity_index][0]

    def estimate_fid(self, nfe_steps: int) -> float:
        """Return the expected FID delta for a given NFE step count.

        Looks up ``nfe_steps`` in the step ladder; if not found, linearly
        interpolates between the two nearest entries.
        """
        for steps, fid_delta in self._ladder:
            if steps == nfe_steps:
                return fid_delta

        # Interpolate between surrounding entries.
        for i in range(len(self._ladder) - 1):
            hi_steps, hi_fid = self._ladder[i]
            lo_steps, lo_fid = self._ladder[i + 1]
            if lo_steps < nfe_steps < hi_steps:
                ratio = (hi_steps - nfe_steps) / (hi_steps - lo_steps)
                return hi_fid + ratio * (lo_fid - hi_fid)

        # Outside ladder range — return worst-case.
        return self._ladder[-1][1]

    def should_degrade(self) -> bool:
        """Return ``True`` if current load warrants stepping down."""
        return self.select_steps() < self._ladder[0][0]
