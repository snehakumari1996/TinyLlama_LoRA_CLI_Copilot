"""Static eval: compare base vs LoRA on a fixed prompt set; writes eval_static.md."""
from __future__ import annotations

import logging
from typing import Optional

from tinyllama_copilot.config import ADAPTER_DIR, BASE_MODEL, PROJECT_ROOT

logger = logging.getLogger(__name__)

PROMPTS = [
    "Create a new Git branch and switch to it.",
    "Compress the folder reports into reports.tar.gz.",
    "List all Python files in the current directory recursively.",
    "Set up a virtual environment and install requests.",
    "Fetch only the first ten lines of a file named output.log.",
]


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {ADAPTER_DIR}")

    # Lazy heavy imports
    from peft import PeftModel
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as hf_log
    from transformers import pipeline

    hf_log.set_verbosity_error()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base_m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype="auto"
    )
    pipe_b = pipeline(
        "text-generation",
        model=base_m,
        tokenizer=tok,
        max_new_tokens=64,
        do_sample=False,
    )
    lora_m = PeftModel.from_pretrained(base_m, str(ADAPTER_DIR)).merge_and_unload()
    pipe_l = pipeline(
        "text-generation",
        model=lora_m,
        tokenizer=tok,
        max_new_tokens=64,
        do_sample=False,
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows: list[tuple[str, str, str, str]] = []
    for p in PROMPTS:
        b = pipe_b(f"[INST] {p} [/INST]")[0]["generated_text"]
        l = pipe_l(f"[INST] {p} [/INST]")[0]["generated_text"]
        r = scorer.score(b, l)["rougeL"].fmeasure
        rows.append((p, b.strip(), l.strip(), f"{r:.3f}"))

    md = [
        "| Prompt | Base (trim) | LoRA (trim) | ROUGE-L |",
        "|--------|-------------|-------------|---------|",
    ]
    for p, b, l, r in rows:
        md.append(f"| {p} | `{b[:40]}…` | `{l[:40]}…` | **{r}** |")

    out = PROJECT_ROOT / "eval_static.md"
    out.write_text("\n".join(md) + "\n")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
