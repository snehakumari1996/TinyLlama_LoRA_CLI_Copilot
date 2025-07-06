#!/usr/bin/env python
"""
CLI agent – prints a numbered plan plus *dry-run* shell commands.
"""
# ── Silence TensorFlow & HF chatter ───────────────────────────────────────
import os, re, argparse, subprocess, json, datetime, pathlib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_log
from peft import PeftModel
hf_log.set_verbosity_error()

# ── Paths & model ids ──────────────────────────────────────────────────────
ROOT     = "/content/drive/MyDrive/fenrir-mini-lora-agent"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPT    = f"{ROOT}/lora_adapter"
OFFLOAD  = f"{ROOT}/offload"; pathlib.Path(OFFLOAD).mkdir(exist_ok=True)

# ── Load tokenizer & merged-LoRA model ─────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tok.eos_token_id = tok.eos_token_id or tok.convert_tokens_to_ids("</s>")

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", offload_folder=OFFLOAD, torch_dtype="auto"
)
model = (
    PeftModel.from_pretrained(base, ADAPT,
                              device_map="auto", offload_folder=OFFLOAD)
    .merge_and_unload()
)

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

# ── Helpers ────────────────────────────────────────────────────────────────
def dry(cmd: str) -> str:
    """Return dry-run output of *cmd* (string)."""
    return subprocess.run(
        ["bash", "-c", f"echo $ {cmd}"],
        capture_output=True, text=True
    ).stdout.strip()

def generate(task: str) -> str:
    """Generate raw LLM answer (string)."""
    ids = tok(TEMPLATE.format(task=task), return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=192, do_sample=False,
                         eos_token_id=tok.eos_token_id)[0]
    text = tok.decode(out[ids.shape[1]:], skip_special_tokens=True)
    return text.split("</s>")[0].strip()      # crop at explicit end tag

def parse_steps(answer: str):
    """Extract shell commands ($ …) and return list of {cmd,dry} dicts."""
    steps = []
    for line in answer.splitlines():
        m = re.match(r"^\$\s*(.+)", line.strip())
        if m:
            cmd = m.group(1).strip()
            steps.append({"cmd": cmd, "dry": dry(cmd)})
    return steps

# ── Driver ────────────────────────────────────────────────────────────────
def run(task: str):
    answer = generate(task)
    print(answer, flush=True)

    rec = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "task": task,
        "raw": answer,
        "steps": parse_steps(answer),
    }
    log = pathlib.Path(ROOT, "logs"); log.mkdir(exist_ok=True)
    (log / "trace.jsonl").open("a").write(json.dumps(rec) + "\n")

# ── CLI entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="+", help="Natural-language task description")
    if args.task:
      run(" ".join(ap.parse_args().task))
    else:
      task=input("enter a task")
