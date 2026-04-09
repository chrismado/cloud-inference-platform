"""
Checkpoint Store — Local + S3 Model Checkpoint Management

Saves and loads model checkpoints to local disk with optional S3
synchronisation.  Falls back gracefully when boto3 is unavailable.

Usage::

    store = CheckpointStore(local_dir="/checkpoints", s3_bucket="my-bucket")
    store.save("model_v1", state_dict)
    loaded = store.load("model_v1")
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import boto3

    _BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore[assignment]
    _BOTO3_AVAILABLE = False
    logger.info("boto3 not installed — S3 checkpoint sync disabled.")


@dataclass
class CheckpointStoreConfig:
    """Configuration for the checkpoint store."""

    local_dir: str = "./checkpoints"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "checkpoints/"
    allow_unsafe_deserialization: bool = False


class CheckpointStore:
    """Manages model checkpoints on local disk and optionally in S3.

    Parameters
    ----------
    config : CheckpointStoreConfig
        Store configuration.  If not supplied a default is created.
    """

    def __init__(self, config: Optional[CheckpointStoreConfig] = None) -> None:
        self.config = config or CheckpointStoreConfig()
        os.makedirs(self.config.local_dir, exist_ok=True)
        self._s3: Any = None

        if _BOTO3_AVAILABLE and self.config.s3_bucket:
            try:
                self._s3 = boto3.client("s3")
                logger.info("S3 checkpoint sync enabled: s3://%s/%s", self.config.s3_bucket, self.config.s3_prefix)
            except Exception as exc:
                logger.warning("Failed to initialise S3 client: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, name: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a checkpoint to local disk (and optionally S3).

        Parameters
        ----------
        name : str
            Checkpoint name (used as the filename stem).
        data : Any
            Picklable checkpoint data (e.g. ``state_dict``).
        metadata : dict, optional
            JSON-serialisable metadata saved alongside the checkpoint.

        Returns
        -------
        str
            Absolute path to the saved local checkpoint file.
        """
        local_path = os.path.join(self.config.local_dir, f"{name}.ckpt")
        meta_path = os.path.join(self.config.local_dir, f"{name}.meta.json")

        with open(local_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Checkpoint saved locally: %s", local_path)

        if metadata is not None:
            with open(meta_path, "w") as mf:
                json.dump(metadata, mf, indent=2)

        # Sync to S3
        if self._s3 is not None and self.config.s3_bucket:
            try:
                s3_key = f"{self.config.s3_prefix}{name}.ckpt"
                self._s3.upload_file(local_path, self.config.s3_bucket, s3_key)
                logger.info("Checkpoint uploaded to s3://%s/%s", self.config.s3_bucket, s3_key)
            except Exception as exc:
                logger.warning("S3 upload failed: %s", exc)

        return os.path.abspath(local_path)

    def load(self, name: str, allow_unsafe: Optional[bool] = None) -> Any:
        """Load a checkpoint by name.

        Looks on local disk first; if missing and S3 is configured,
        attempts to download from S3.

        Returns
        -------
        Any
            The unpickled checkpoint data.

        Raises
        ------
        FileNotFoundError
            If the checkpoint cannot be found locally or in S3.
        PermissionError
            If unsafe pickle deserialization was not explicitly enabled.
        """
        local_path = os.path.join(self.config.local_dir, f"{name}.ckpt")

        if not os.path.exists(local_path) and self._s3 is not None:
            try:
                s3_key = f"{self.config.s3_prefix}{name}.ckpt"
                self._s3.download_file(self.config.s3_bucket, s3_key, local_path)
                logger.info("Checkpoint downloaded from S3: %s", s3_key)
            except Exception as exc:
                raise FileNotFoundError(f"Checkpoint '{name}' not found locally or in S3.") from exc

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {local_path}")

        allow_unpickle = self.config.allow_unsafe_deserialization if allow_unsafe is None else allow_unsafe
        if not allow_unpickle:
            raise PermissionError(
                "Checkpoint loading uses pickle deserialization and is disabled by default. "
                "Pass allow_unsafe=True only for checkpoints you trust."
            )

        with open(local_path, "rb") as f:
            return pickle.load(f)  # nosec B301 - explicit opt-in for trusted checkpoints only

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all locally available checkpoints.

        Returns
        -------
        list[dict]
            Each entry has ``name``, ``path``, and ``size_mb``.
        """
        checkpoints: List[Dict[str, Any]] = []
        for filename in sorted(os.listdir(self.config.local_dir)):
            if filename.endswith(".ckpt"):
                path = os.path.join(self.config.local_dir, filename)
                checkpoints.append(
                    {
                        "name": filename.removesuffix(".ckpt"),
                        "path": path,
                        "size_mb": os.path.getsize(path) / (1024 * 1024),
                    }
                )
        return checkpoints

    def delete(self, name: str) -> bool:
        """Delete a checkpoint from local disk.

        Returns ``True`` if the file existed and was removed.
        """
        local_path = os.path.join(self.config.local_dir, f"{name}.ckpt")
        meta_path = os.path.join(self.config.local_dir, f"{name}.meta.json")
        existed = os.path.exists(local_path)
        if existed:
            os.remove(local_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            logger.info("Checkpoint deleted: %s", name)
        return existed
