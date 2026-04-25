"""CLI agent — prints a numbered plan plus *dry-run* shell commands."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import subprocess
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from tinyllama_copilot.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    CACHE_DIR,
    LOG_DIR,
    OFFLOAD_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ── Prompt template (few-shot) ─────────────────────────────────────────────
FEWSHOT = """\
<s>[INST] You are a CLI expert.

TASK: List .txt files in the current directory

1. Identify current directory
2. Show .txt files
$ pwd
$ find . -type f -name '*.txt'
</s>"""

TEMPLATE = FEWSHOT + """

<INST> You are a CLI expert.

TASK: {task}

1. Output a numbered plan (English).
2. Then print one POSIX shell command per line, each starting with $ .
   No explanations, no extra text.
3. End with </s>. </INST>"""


# ── Lazy model loading + on-disk cache for merged weights ─────────────────
_TOK = None
_MODEL = None


def _adapter_cache_key() -> str:
    """Stable cache key from base model id + adapter config + weights mtime/size."""
    config_file = ADAPTER_DIR / "adapter_config.json"
    weights_file = ADAPTER_DIR / "adapter_model.safetensors"
    if not config_file.exists() or not weights_file.exists():
        return ""
    h = hashlib.sha256()
    h.update(BASE_MODEL.encode())
    h.update(config_file.read_bytes())
    stat = weights_file.stat()
    h.update(f"{stat.st_size}-{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _merged_cache_dir() -> Optional[pathlib.Path]:
    key = _adapter_cache_key()
    return CACHE_DIR / "merged" / key if key else None


def _load_model():
    """Load tokenizer + merged model, using an on-disk cache if available.

    The first call merges the LoRA adapter into the base weights and saves
    the result to ``cache/merged/<hash>/`` (~10-20 s). Subsequent calls
    skip the merge and load the cached model directly (a few seconds).
    """
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {ADAPTER_DIR}. "
            "Train one with `python -m tinyllama_copilot.train` "
            "or set TINYLLAMA_PROJECT_ROOT to a directory that contains lora_adapter/."
        )
    ensure_dirs()

    # Lazy heavy imports — keeps `import tinyllama_copilot` cheap.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as hf_log

    hf_log.set_verbosity_error()

    cache_dir = _merged_cache_dir()
    if cache_dir is not None and cache_dir.exists():
        logger.info("Loading cached merged model from %s", cache_dir)
        tok = AutoTokenizer.from_pretrained(str(cache_dir), use_fast=True)
        tok.eos_token_id = tok.eos_token_id or tok.convert_tokens_to_ids("</s>")
        model = AutoModelForCausalLM.from_pretrained(
            str(cache_dir),
            device_map="auto",
            offload_folder=str(OFFLOAD_DIR),
            torch_dtype="auto",
        )
        return tok, model

    logger.info("Merging LoRA adapter into base model (one-time cost)…")
    from peft import PeftModel  # heavier; only needed on cache miss

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.eos_token_id = tok.eos_token_id or tok.convert_tokens_to_ids("</s>")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER_DIR),
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
    ).merge_and_unload()

    if cache_dir is not None:
        try:
            logger.info("Caching merged model to %s", cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cache_dir))
            tok.save_pretrained(str(cache_dir))
        except Exception as e:  # caching is a best-effort optimization
            logger.warning("Failed to cache merged model: %s", e)

    return tok, model


def _ensure_loaded():
    global _TOK, _MODEL
    if _MODEL is None:
        logger.info("Loading model + LoRA adapter (first call only)…")
        _TOK, _MODEL = _load_model()
    return _TOK, _MODEL


# ── Pure helpers (no model required — easy to unit-test) ───────────────────
def dry(cmd: str) -> str:
    """Return the dry-run echo of *cmd* without executing it."""
    return subprocess.run(
        ["bash", "-c", f"echo $ {cmd}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def parse_steps(answer: str) -> list[dict]:
    """Extract shell commands (lines starting with `$`) from *answer*."""
    steps: list[dict] = []
    for line in answer.splitlines():
        m = re.match(r"^\$\s*(.+)", line.strip())
        if m:
            cmd = m.group(1).strip()
            steps.append({"cmd": cmd, "dry": dry(cmd)})
    return steps


# ── Generation ─────────────────────────────────────────────────────────────
def generate(task: str) -> str:
    """Generate the raw LLM answer for *task*."""
    tok, model = _ensure_loaded()
    ids = tok(TEMPLATE.format(task=task), return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=192,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
    )[0]
    text = tok.decode(out[ids.shape[1]:], skip_special_tokens=True)
    return text.split("</s>")[0].strip()


def run(task: str) -> dict:
    """Generate, parse, log, and return the trace record for *task*."""
    answer = generate(task)
    print(answer, flush=True)

    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "task": task,
        "raw": answer,
        "steps": parse_steps(answer),
    }
    ensure_dirs()
    with (LOG_DIR / "trace.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# ── CLI entrypoint ─────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=os.environ.get("TINYLLAMA_LOG_LEVEL", "INFO"))
    ap = argparse.ArgumentParser(
        prog="tinyllama-cli",
        description="Translate natural-language tasks into dry-run shell commands.",
    )
    ap.add_argument("task", nargs="*", help="Natural-language task description")
    args = ap.parse_args(argv)
    task = " ".join(args.task).strip() if args.task else input("Enter a task: ").strip()
    if not task:
        ap.error("empty task")
    run(task)


if __name__ == "__main__":
    main()
