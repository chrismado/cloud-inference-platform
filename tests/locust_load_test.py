"""
Locust Load Test for Cloud Inference Platform

Simulates realistic traffic patterns against the inference router API,
mixing text, spatial, and video requests with configurable weights.

Usage::

    locust -f tests/locust_load_test.py --host http://localhost:8080
    python tests/locust_load_test.py  # quick smoke test

NOTE: This module is intentionally excluded from pytest discovery to avoid
gevent/SSL conflicts.  Run it directly or via ``locust -f``.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


class InferenceUser(HttpUser):
    """Simulates a user sending inference requests to the router.

    Task weights reflect a realistic traffic mix:
      - text (60%)  — most common, LLM completions
      - video (25%) — DiT diffusion generation
      - spatial (10%) — 3DGS rendering
      - health (5%)  — monitoring probes
    """

    wait_time = between(0.1, 1.0)

    # ── Sample payloads ───────────────────────────────────────────────

    _TEXT_PROMPTS = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to compute Fibonacci numbers.",
        "Summarise the key points of transformer architectures.",
        "What are the benefits of using Rust over C++?",
        "Describe the process of nuclear fusion in stars.",
    ]

    _CAMERA_POSES = [
        {"position": [0, 0, 5], "rotation": [0, 0, 0, 1]},
        {"position": [2, 1, 3], "rotation": [0.1, 0.2, 0.3, 0.9]},
        {"position": [-1, 3, 4], "rotation": [0, 0.5, 0, 0.87]},
    ]

    # ── Tasks ─────────────────────────────────────────────────────────

    @task(60)
    def text_inference(self):
        """Send a text completion request."""
        payload = {
            "request_type": "text",
            "prompt": random.choice(self._TEXT_PROMPTS),
            "max_tokens": random.choice([64, 128, 256]),
            "temperature": 0.7,
        }
        with self.client.post(
            "/infer",
            json=payload,
            name="/infer [text]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(25)
    def video_inference(self):
        """Send a video (DiT diffusion) generation request."""
        payload = {
            "request_type": "video",
            "prompt": "A timelapse of a flower blooming in a garden.",
            "num_frames": random.choice([16, 32, 64]),
            "resolution": [512, 512],
        }
        with self.client.post(
            "/infer",
            json=payload,
            name="/infer [video]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(10)
    def spatial_inference(self):
        """Send a 3DGS spatial rendering request."""
        payload = {
            "request_type": "spatial",
            "camera_pose": random.choice(self._CAMERA_POSES),
            "resolution": [1920, 1080],
        }
        with self.client.post(
            "/infer",
            json=payload,
            name="/infer [spatial]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def health_check(self):
        """Hit the health endpoint."""
        with self.client.get(
            "/health",
            name="/health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


if __name__ == "__main__":
    import subprocess
    import sys

    print("Starting Locust load test (web UI mode)...")
    print("Open http://localhost:8089 in your browser.\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "locust",
                "-f",
                __file__,
                "--host",
                "http://localhost:8080",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nLoad test stopped.")
    except FileNotFoundError:
        print("ERROR: locust not found. Install with: pip install locust", file=sys.stderr)
        sys.exit(1)
