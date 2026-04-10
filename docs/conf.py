from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "cloud-inference-platform"
author = "Chris Matteau"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
autodoc_mock_imports = [
    "boto3",
    "fakeredis",
    "fastapi",
    "kubernetes",
    "locust",
    "prometheus_client",
    "pydantic",
    "redis",
    "sglang",
    "torch",
    "triton",
    "uvicorn",
    "vllm",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
html_theme = "alabaster"
