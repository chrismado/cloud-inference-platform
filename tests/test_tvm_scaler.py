"""
Unit tests for TVMScaler — NFE step selection and FID estimation.

Usage::

    python -m pytest tests/test_tvm_scaler.py -v
    python -m unittest tests.test_tvm_scaler
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from router.slo_router import SLOConfig, SLORouter
from router.tvm_scaler import TVMScaler, TVMScalerConfig


class TestTVMScaler(unittest.TestCase):
    """Tests for dynamic TVM step selection and FID estimation."""

    def _make_scaler(self, p95_latency: float = 0.0) -> TVMScaler:
        """Create a TVMScaler with a mocked router returning the given p95."""
        slo_config = SLOConfig(
            p95_target_ms=600.0,
            tvm_steps_normal=4,
            tvm_steps_degraded=1,
            degradation_threshold=0.85,
        )
        router = MagicMock(spec=SLORouter)
        router.config = slo_config
        router._get_current_p95_latency.return_value = p95_latency
        scaler_config = TVMScalerConfig(fid_budget=2.5)
        return TVMScaler(scaler_config, router)

    # ------------------------------------------------------------------
    # select_steps
    # ------------------------------------------------------------------

    def test_select_steps_normal_load(self):
        """Under low p95 the scaler should select full 4-step quality."""
        scaler = self._make_scaler(p95_latency=100.0)
        self.assertEqual(scaler.select_steps(), 4)

    def test_select_steps_at_threshold(self):
        """At exactly the threshold boundary the scaler should stay at 4 steps."""
        threshold = 600.0 * 0.85  # 510.0
        scaler = self._make_scaler(p95_latency=threshold)
        self.assertEqual(scaler.select_steps(), 4)

    def test_select_steps_above_threshold(self):
        """Above the threshold the scaler should degrade."""
        scaler = self._make_scaler(p95_latency=520.0)
        steps = scaler.select_steps()
        self.assertIn(steps, [1, 2])

    def test_select_steps_extreme_load(self):
        """Under extreme load the scaler should select minimum steps."""
        scaler = self._make_scaler(p95_latency=5000.0)
        steps = scaler.select_steps()
        self.assertEqual(steps, 1)

    # ------------------------------------------------------------------
    # estimate_fid
    # ------------------------------------------------------------------

    def test_estimate_fid_exact_match(self):
        """FID for a step count in the ladder should return the exact delta."""
        scaler = self._make_scaler()
        self.assertAlmostEqual(scaler.estimate_fid(4), 0.0)
        self.assertAlmostEqual(scaler.estimate_fid(2), 1.2)
        self.assertAlmostEqual(scaler.estimate_fid(1), 2.5)

    def test_estimate_fid_interpolation(self):
        """FID for an intermediate step count should interpolate."""
        scaler = self._make_scaler()
        fid_3 = scaler.estimate_fid(3)
        # Between 4-step (0.0) and 2-step (1.2), 3 is the midpoint
        self.assertGreater(fid_3, 0.0)
        self.assertLess(fid_3, 1.2)

    def test_estimate_fid_outside_range(self):
        """FID for a step count outside the ladder should return worst-case."""
        scaler = self._make_scaler()
        fid = scaler.estimate_fid(0)
        self.assertEqual(fid, 2.5)

    # ------------------------------------------------------------------
    # should_degrade
    # ------------------------------------------------------------------

    def test_should_degrade_false_normal(self):
        scaler = self._make_scaler(p95_latency=100.0)
        self.assertFalse(scaler.should_degrade())

    def test_should_degrade_true_overloaded(self):
        scaler = self._make_scaler(p95_latency=520.0)
        self.assertTrue(scaler.should_degrade())

    # ------------------------------------------------------------------
    # Boundary / edge cases
    # ------------------------------------------------------------------

    def test_zero_latency(self):
        scaler = self._make_scaler(p95_latency=0.0)
        self.assertEqual(scaler.select_steps(), 4)

    def test_negative_latency(self):
        """Negative p95 (shouldn't happen) should still produce full steps."""
        scaler = self._make_scaler(p95_latency=-10.0)
        self.assertEqual(scaler.select_steps(), 4)


if __name__ == "__main__":
    unittest.main()
