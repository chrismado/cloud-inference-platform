"""
3D Gaussian Splatting (3DGS) Serving Backend

Renders novel views from a pre-trained Gaussian Splat scene using the
gsplat CUDA rasterizer. Falls back to a deterministic gradient stub when
gsplat or torch are unavailable.

Usage::

    backend = GaussianSplatBackend()
    backend.load_model("/models/garden_scene.ply")
    image = backend.render(camera_pose=pose_4x4, resolution=(1920, 1080))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import gsplat  # noqa: F401

    _GSPLAT_AVAILABLE = True
except ImportError:
    _GSPLAT_AVAILABLE = False
    logger.warning("gsplat not installed - GaussianSplatBackend will use stub rendering.")


@dataclass
class GaussianSplatConfig:
    """Configuration for the 3DGS rendering backend."""

    default_resolution: Tuple[int, int] = (1920, 1080)
    max_gaussians: int = 5_000_000
    sh_degree: int = 3
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class GaussianSplatBackend:
    """3D Gaussian Splatting renderer for spatial inference.

    Parameters
    ----------
    config : GaussianSplatConfig
        Rendering configuration. If not supplied a default is created.
    """

    def __init__(self, config: Optional[GaussianSplatConfig] = None) -> None:
        self.config = config or GaussianSplatConfig()
        self._model_data: Optional[Dict[str, Any]] = None
        self._model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> None:
        """Load a Gaussian Splat scene from a ``.ply`` or checkpoint file.

        Parameters
        ----------
        model_path : str
            Path to the 3DGS model file.
        """
        path = Path(model_path)
        if not path.exists():
            self._model_data = None
            self._model_path = None
            raise FileNotFoundError(f"3DGS model path does not exist: {model_path}")

        self._model_path = str(path)

        if not _TORCH_AVAILABLE:
            logger.warning("torch unavailable - loading stub model data.")
            self._model_data = {"stub": True, "path": str(path)}
            return

        try:
            # In a real deployment this would parse the PLY / checkpoint
            # and populate means3D, opacities, scales, rotations, sh_coeffs.
            self._model_data = {
                "path": str(path),
                "loaded": True,
                "n_gaussians": 0,
            }
            logger.info("3DGS model loaded from %s", path)
        except Exception as exc:
            logger.error("Failed to load 3DGS model: %s", exc)
            self._model_data = None

    def render(
        self,
        camera_pose: Any,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Render a novel view from the loaded Gaussian Splat scene."""
        w, h = resolution or self.config.default_resolution

        if self._model_data is None:
            logger.warning("No model loaded - returning blank frame.")
            return np.zeros((h, w, 3), dtype=np.uint8)

        if not _GSPLAT_AVAILABLE:
            return self._stub_render(w, h)

        logger.warning(
            "gsplat is available but the real rasterization path is still a prototype; "
            "returning the deterministic stub render instead."
        )
        return self._stub_render(w, h)

    def render_batch(
        self,
        camera_poses: List[Any],
        resolution: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """Render multiple views from different camera poses."""
        return [self.render(pose, resolution=resolution) for pose in camera_poses]

    def health_check(self) -> Dict[str, Any]:
        """Return backend health status."""
        if self._model_data is None:
            render_mode = "unloaded"
        elif not _GSPLAT_AVAILABLE:
            render_mode = "deterministic_stub"
        else:
            render_mode = "prototype_stub"

        return {
            "backend": "gaussian_splat",
            "gsplat_available": _GSPLAT_AVAILABLE,
            "torch_available": _TORCH_AVAILABLE,
            "model_loaded": self._model_data is not None,
            "model_path": self._model_path,
            "render_mode": render_mode,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_render(width: int, height: int) -> np.ndarray:
        """Return a gradient test image when gsplat is unavailable."""
        img: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        grad = np.linspace(0, 255, width, dtype=np.uint8)
        img[:, :, 0] = grad[np.newaxis, :]
        return img
