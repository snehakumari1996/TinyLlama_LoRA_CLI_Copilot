#!/usr/bin/env python
"""Static eval: base vs LoRA; writes eval_static.md."""
import pathlib
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from peft import PeftModel

logging.set_verbosity_error()
BASE_DIR   = "/content/drive/MyDrive/fenrir-mini-lora-agent"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER    = f"{BASE_DIR}/lora_adapter"
TOK = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# base pipeline
base_m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype="auto")
pipe_b = pipeline("text-generation", model=base_m, tokenizer=TOK, max_new_tokens=64, do_sample=False)

# LoRA‑merged pipeline
lora_m = PeftModel.from_pretrained(base_m, ADAPTER).merge_and_unload()
pipe_l = pipeline("text-generation", model=lora_m, tokenizer=TOK, max_new_tokens=64, do_sample=False)

PROMPTS = [
    "Create a new Git branch and switch to it.",
    "Compress the folder reports into reports.tar.gz.",
    "List all Python files in the current directory recursively.",
    "Set up a virtual environment and install requests.",
    "Fetch only the first ten lines of a file named output.log.",
]

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rows = []
for p in PROMPTS:
    b = pipe_b(f"[INST] {p} [/INST]")[0]["generated_text"]
    l = pipe_l(f"[INST] {p} [/INST]")[0]["generated_text"]
    r = scorer.score(b, l)["rougeL"].fmeasure
    rows.append((p, b.strip(), l.strip(), f"{r:.3f}"))

md = ["| Prompt | Base (trim) | LoRA (trim) | ROUGE‑L |", "|--------|-------------|-------------|---------|"]
for p, b, l, r in rows:
    md.append(f"| {p} | `{b[:40]}…` | `{l[:40]}…` | **{r}** |")

pathlib.Path(f"{BASE_DIR}/eval_static.md").write_text("\n".join(md))
print("eval_static.md written")