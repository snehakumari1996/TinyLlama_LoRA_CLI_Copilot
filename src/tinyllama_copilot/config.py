"""Central configuration — paths and model ids resolved from env vars or defaults."""
from __future__ import annotations

import os
import pathlib

# Project root: env override → walk up from this file (src/tinyllama_copilot/config.py)
PROJECT_ROOT: pathlib.Path = pathlib.Path(
    os.environ.get(
        "TINYLLAMA_PROJECT_ROOT",
        pathlib.Path(__file__).resolve().parents[2],
    )
).resolve()

# Model
BASE_MODEL: str = os.environ.get(
    "TINYLLAMA_BASE_MODEL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# Standard project directories
DATA_DIR: pathlib.Path = PROJECT_ROOT / "data"
ADAPTER_DIR: pathlib.Path = PROJECT_ROOT / "lora_adapter"
LOG_DIR: pathlib.Path = PROJECT_ROOT / "logs"
OFFLOAD_DIR: pathlib.Path = PROJECT_ROOT / "offload"
EVAL_DIR: pathlib.Path = PROJECT_ROOT / "eval"
OUTPUTS_DIR: pathlib.Path = PROJECT_ROOT / "outputs"
CACHE_DIR: pathlib.Path = PROJECT_ROOT / "cache"


def ensure_dirs() -> None:
    """Create writable runtime directories. Safe to call repeatedly."""
    for d in (LOG_DIR, OFFLOAD_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
