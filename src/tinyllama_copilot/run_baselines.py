"""Generate candidate outputs for each model on the benchmark.

Supported models (selectable via ``--models lora,base,gemini,...``):

| name        | description                                          |
|-------------|------------------------------------------------------|
| ``lora``    | Local TinyLlama-1.1B + your LoRA adapter             |
| ``base``    | Local TinyLlama-1.1B with no adapter (pure base)     |
| ``gemini``  | Gemini 2.0 Flash via Google AI Studio (free tier)    |
| ``anthropic`` | Claude Haiku via Anthropic API (paid)              |
| ``openai``  | GPT-4o-mini via OpenAI API (paid)                    |
| ``stub``    | Echo the task back — for pipeline tests, no API/GPU  |

All models receive the same agent prompt (numbered plan + ``$``-prefixed
shell commands) so the comparison is apples-to-apples.

Outputs are written to ``eval/candidates/<model>.jsonl`` as
``{"id": int, "output": str}`` rows.

Run with::

    tinyllama-baseline --models lora,base --limit 20
    tinyllama-baseline --models lora,gemini --sleep 4   # respect 15 RPM free tier
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from tinyllama_copilot.agent import TEMPLATE
from tinyllama_copilot.config import ADAPTER_DIR, BASE_MODEL, OFFLOAD_DIR, ensure_dirs

logger = logging.getLogger(__name__)

GenerateFn = Callable[[str], str]


# ---------- Local TinyLlama (with or without adapter) -------------------

def _make_local_generator(use_adapter: bool) -> GenerateFn:
    """Return a generate(task)→output function for local TinyLlama."""
    ensure_dirs()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as hf_log

    hf_log.set_verbosity_error()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.eos_token_id = tok.eos_token_id or tok.convert_tokens_to_ids("</s>")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
        torch_dtype="auto",
    )
    if use_adapter:
        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(f"LoRA adapter not found at {ADAPTER_DIR}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            base,
            str(ADAPTER_DIR),
            device_map="auto",
            offload_folder=str(OFFLOAD_DIR),
        ).merge_and_unload()
    else:
        model = base

    def generate(task: str) -> str:
        ids = tok(TEMPLATE.format(task=task), return_tensors="pt").input_ids.to(model.device)
        out = model.generate(
            ids,
            max_new_tokens=192,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
        )[0]
        text = tok.decode(out[ids.shape[1] :], skip_special_tokens=True)
        return text.split("</s>")[0].strip()

    return generate


# ---------- Hosted API generators ---------------------------------------

def _make_gemini_generator(model: str = "gemini-2.5-flash") -> GenerateFn:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY (https://aistudio.google.com/app/apikey)")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def generate(task: str) -> str:
        prompt = TEMPLATE.format(task=task)
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return resp.text or ""
            except Exception as e:
                last_exc = e
                wait = 2.0 ** attempt
                logger.warning("Gemini error (%s); retry in %.0fs", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"Gemini generation failed: {last_exc}")

    return generate


def _make_anthropic_generator(model: str = "claude-haiku-4-5-20251001") -> GenerateFn:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY")
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def generate(task: str) -> str:
        prompt = TEMPLATE.format(task=task)
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except Exception as e:
                last_exc = e
                wait = 2.0 ** attempt
                logger.warning("Anthropic error (%s); retry in %.0fs", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"Anthropic generation failed: {last_exc}")

    return generate


def _make_openai_generator(model: str = "gpt-4o-mini") -> GenerateFn:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY")
    import openai

    client = openai.OpenAI(api_key=api_key)

    def generate(task: str) -> str:
        prompt = TEMPLATE.format(task=task)
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                wait = 2.0 ** attempt
                logger.warning("OpenAI error (%s); retry in %.0fs", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenAI generation failed: {last_exc}")

    return generate


def _make_stub_generator() -> GenerateFn:
    """Echo the task — for testing the pipeline without any model."""

    def generate(task: str) -> str:
        return f"1. Run this command\n$ # stub for: {task[:80]}"

    return generate


MODEL_FACTORIES: dict[str, Callable[..., GenerateFn]] = {
    "lora": lambda **_: _make_local_generator(use_adapter=True),
    "base": lambda **_: _make_local_generator(use_adapter=False),
    "gemini": lambda gemini_model="gemini-2.5-flash", **_: _make_gemini_generator(gemini_model),
    "anthropic": lambda anthropic_model="claude-haiku-4-5-20251001", **_: _make_anthropic_generator(anthropic_model),
    "openai": lambda openai_model="gpt-4o-mini", **_: _make_openai_generator(openai_model),
    "stub": lambda **_: _make_stub_generator(),
}


# ---------- Driver ------------------------------------------------------

def run_one_model(
    model_name: str,
    benchmark: list[dict],
    *,
    limit: int = 0,
    sleep_s: float = 0.0,
    **factory_kwargs,
) -> list[dict]:
    """Run *model_name* over *benchmark* and return candidate records."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_FACTORIES)}")
    generate = MODEL_FACTORIES[model_name](**factory_kwargs)

    rows = benchmark[:limit] if limit > 0 else benchmark
    iterator = rows
    if sys.stderr.isatty():
        try:
            from tqdm import tqdm

            iterator = tqdm(rows, desc=model_name, file=sys.stderr)
        except ImportError:
            pass

    candidates: list[dict] = []
    for r in iterator:
        try:
            output = generate(r["task"])
        except Exception as e:
            logger.error("id=%s failed: %s", r["id"], e)
            output = ""
        candidates.append({"id": r["id"], "output": output})
        if sleep_s > 0:
            time.sleep(sleep_s)
    return candidates


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tinyllama-baseline",
        description="Generate candidate outputs for each model on the benchmark.",
    )
    ap.add_argument(
        "--models",
        default="lora,base",
        help=f"Comma-separated. Available: {','.join(MODEL_FACTORIES)}",
    )
    ap.add_argument("--benchmark", default="eval/benchmark.jsonl")
    ap.add_argument("--output-dir", default="eval/candidates")
    ap.add_argument("--limit", type=int, default=0, help="Limit benchmark rows (for testing)")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between calls (rate-limit hosted APIs; ~4.0 for Gemini free tier)",
    )
    ap.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model id (free tier: gemini-2.5-flash, gemini-2.5-flash-lite)",
    )
    ap.add_argument(
        "--anthropic-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model id",
    )
    ap.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model id",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=os.environ.get("TINYLLAMA_LOG_LEVEL", "INFO"))
    args = _parse_args(argv)

    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {bench_path}")
    benchmark = [json.loads(line) for line in bench_path.read_text().splitlines() if line.strip()]
    logger.info("Benchmark: %d tasks", len(benchmark))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_FACTORIES:
            raise ValueError(f"Unknown model: {m}. Choose from {list(MODEL_FACTORIES)}")

    factory_kwargs = {
        "gemini_model": args.gemini_model,
        "anthropic_model": args.anthropic_model,
        "openai_model": args.openai_model,
    }
    for model_name in models:
        logger.info("=== Generating candidates: %s ===", model_name)
        cands = run_one_model(
            model_name, benchmark, limit=args.limit, sleep_s=args.sleep, **factory_kwargs
        )
        out_path = out_dir / f"{model_name}.jsonl"
        with out_path.open("w") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
        logger.info("Wrote %s (%d candidates)", out_path, len(cands))


if __name__ == "__main__":
    main()
