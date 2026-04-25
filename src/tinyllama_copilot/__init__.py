"""TinyLlama LoRA CLI Copilot — natural language → shell commands with dry-run safety."""
from __future__ import annotations

__version__ = "0.2.0"

from tinyllama_copilot.agent import dry, generate, parse_steps, run

__all__ = ["dry", "generate", "parse_steps", "run", "__version__"]
