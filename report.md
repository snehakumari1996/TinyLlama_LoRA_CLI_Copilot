# Fenrir-Mini LoRA Agent 

A 1.1 B-parameter TinyLlama fine-tuned (LoRA rank 8) to turn natural-language
requests into **safe, dry-run POSIX shell plans**.

---

## 1 · Project Snapshot

| Item                 | Value |
|----------------------|-------|
| **Base model**       | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (4-bit qLoRA) |
| **Adapter size**     | 75 MB (`lora_adapter/`) |
| **Training steps**   | 6 (≈ 30 s on Colab T4) |
| **Dataset**          | 400 synthetic CLI-QA JSONL rows |
| **Code entry-point** | `src/agent.py` |
| **Repro command**    | `python src/agent.py "Find all JPEGs larger than 5 MB"` |

---

## 2 · Folder Layout

fenrir-mini-lora-agent/
├─ lora_adapter/ # 75 MB LoRA weights
├─ offload/ # disk off-load shards (auto-created)
├─ data/cli_qa.jsonl # 400-row training set
├─ logs/trace.jsonl # every agent run appends one JSON record
├─ src/
│ ├─ train.py # reproduce adapter
│ ├─ evaluate_static.py # static rubric (8 prompts)
│ ├─ evaluate_dynamic.py # dynamic rubric (last 7 prompts)
│ ├─ mean.py # aggregate score helper
│ └─ agent.py # the CLI assistant
├─ eval_static.md # filled by evaluate_static.py
├─ eval_dynamic.md # YOU fill scores then commit
└─ report.md # this file

---

## 3 · Training Details

* **Optimizer** : AdamW (β₁ 0.9, β₂ 0.98)  
* **LR schedule** : Constant  0.00015  
* **Batch size** : 32 tokens  16 accumulation  
* **Loss** after 6 updates : **5.88 → 5.31**

---

## 4 · Evaluation

### 4.1 Static Prompts (baseline template mismatch)

| Prompt                       | Plan | Safe | Answer |
|------------------------------|:----:|:----:|:------:|
| *(8 total — see file)*       | 0.00 | 0.00 | 0.00 |

### 4.2 Dynamic Prompts (latest seven runs)

| Prompt (7 runs)                                       | Plan | Safe | Answer |
|-------------------------------------------------------|:----:|:----:|:------:|
| List .py (recursive)                                  | 1 | 2 | 1 |
| New Git branch                                        | 1 | 1 | 1 |
| Compress `reports`                                    | 1 | 1 | 0 |
| Venv + install requests                               | 1 | 2 | 0 |
| Head 10 lines of `output.log`                         | 1 | 2 | 0 |
| Dry-run delete `*~`                                   | 1 | 2 | 0 |
| Find JPEGs > 5 MB                                     | 1 | 1 | 1 |
| **Mean**                                              | **0.71** | **1.43** | **0.57** |

*(Rubric — 0 = poor, 1 = partial, 2 = perfect)*

### 4.3 How to recompute

```bash
# at repo root
$ python src/evaluate_static.py     
$ python src/evaluate_dynamic.py     
$ python src/mean.py

##5 · Key Findings
Safety first  tagging every line that begins with $ and logging
dry-runs keeps the agent from destructive execution (Safe ≥ 1.4 / 2.0).

Few-shot injection  a single in-context example bumped Plan score
from 0.0 to 0.7 without retraining.

Answer correctness is still low; many commands exist but filenames,
flags, or ordering are occasionally off.

##6 · Future Work
Add 1 2-shot examples for Git, tar, and GNU find to raise Answer ≥ 1.5.

Sandboxed execution autograder → exact output match scoring.

Lightweight Streamlit front-end with copy-to-clipboard buttons.

7 · Reproduction Guide
git clone <repo>
cd fenrir-mini-lora-agent

# 1 · Install (CUDA-11.8 wheels already pinned)
pip install -r requirements.txt

# 2 · (Optionally) retrain
python src/train.py

# 3 · Run agent
python src/agent.py "Delete every *~ file (dry-run)"

# 4 · Inspect logs
tail -n 1 logs/trace.jsonl | jq .


8 · Credits
TinyLlama - TinyLlama Community (Apache-2.0)

LoRA & PEFT - Microsoft Research

Notebook acceleration - Google Colab

Report generated {{ now | utc }}.

---

### `src/mean.py` (unchanged, but included for completeness)

```python
#!/usr/bin/env python
"""
Compute average Plan / Safe / Answer scores across all evaluation markdowns.
"""

import re, pathlib, statistics, textwrap

SCORE_RE = re.compile(r"\|\s*([^|]+)\s*\|\s*([0-2.]+)\s*\|\s*([0-2.]+)\s*\|\s*([0-2.]+)\s*\|")

def parse(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        m = SCORE_RE.match(line)
        if m and not m.group(1).lower().startswith("mean"):
            rows.append(tuple(float(x) for x in m.groups()[1:]))
    return rows

def main():
    md_files = ["eval_static.md", "eval_dynamic.md"]
    header = f"{'File':<15}  Plan  Safe  Answer  #rows"
    print(header); print("-"*len(header))
    for file in md_files:
        if not pathlib.Path(file).exists(): continue
        rows = parse(file)
        means = [statistics.mean(col) if rows else 0 for col in zip(*rows)]
        print(f"{file:<15}", *[f"{m:5.2f}" for m in means], f"{len(rows):6d}")

if __name__ == "__main__":
    main()



