# Evaluation

This directory holds the held-out **benchmark dataset**, judge prompts, and
results artifacts for evaluating the TinyLlama-LoRA copilot against baseline
language models.

## Files

| File | What it is |
|---|---|
| `benchmark.jsonl` | 200 hand-curated CLI tasks with reference commands. **Held-out** from the training set. |
| `README.md` | This file — methodology + reproducibility. |
| `../src/tinyllama_copilot/judge.py` | LLM-as-judge module (Gemini / Claude / GPT). |
| `candidates/<model>.jsonl` | Per-model candidate outputs (Phase C3 — to be added). |
| `results/<model>.jsonl` | Per-task judge scores (Phase C3 — to be added). |
| `results/summary.csv` | Aggregate per-model scores (Phase C3 — to be added). |

## Running the judge

The judge is provider-agnostic — it auto-detects which LLM to use from your env:

| Priority | Env var | Provider | Tier |
|---|---|---|---|
| 1 | `GOOGLE_API_KEY` | Gemini 2.0 Flash | **Free** (https://aistudio.google.com/app/apikey) |
| 2 | `ANTHROPIC_API_KEY` | Claude Haiku 4.5 | Paid (~$1.30 for full sweep) |
| 3 | `OPENAI_API_KEY` | GPT-4o-mini | Paid (~$0.20 for full sweep) |

Install the optional dependencies once:

```bash
pip install -e '.[judge]'
```

Then run:

```bash
# Set your free Gemini key (one-time)
export GOOGLE_API_KEY=AI...

# Score one model's candidates against the benchmark
tinyllama-judge \
    --candidates eval/candidates/lora.jsonl \
    --output eval/results/lora.jsonl
```

The candidates file is JSONL of `{"id": int, "output": str}` rows — typically the
output of running each candidate model on each benchmark task (Phase C3).

For dry-running the pipeline with no API key:

```bash
tinyllama-judge --provider stub --candidates ... --output ...
```

## Benchmark schema

Each line of `benchmark.jsonl` is a JSON object:

```json
{
  "id": 1,
  "task": "List all files in the current directory including hidden ones, with details.",
  "reference": "ls -la",
  "category": "filesystem",
  "difficulty": 1
}
```

| Field | Type | Description |
|---|---|---|
| `id` | int | Stable identifier (1-200). Never reused. |
| `task` | str | Natural-language task in plain English. |
| `reference` | str | One known-good shell answer. The judge accepts semantically equivalent variants. |
| `category` | str | One of: `filesystem`, `git`, `archive`, `networking`, `process`, `env`, `text`, `container`, `sysinfo`, `shell`. |
| `difficulty` | int | `1` basic (single flag/usage), `2` intermediate (combos, pipes), `3` advanced (sed/awk programs, multi-step pipelines, less-common tools). |

## Distribution

200 rows. Built to mirror real CLI usage, weighted toward the categories a
working developer touches every day.

| Category | Count | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| filesystem | 30 | 18 | 8 | 4 |
| git | 30 | 10 | 16 | 4 |
| shell | 30 | 16 | 8 | 6 |
| text | 20 | 5 | 9 | 6 |
| container | 20 | 12 | 8 | 0 |
| process | 17 | 8 | 9 | 0 |
| networking | 16 | 5 | 7 | 4 |
| env | 15 | 14 | 1 | 0 |
| archive | 12 | 9 | 1 | 2 |
| sysinfo | 10 | 7 | 2 | 1 |
| **Total** | **200** | **104 (52%)** | **69 (34%)** | **27 (14%)** |

## Held-out guarantee

The benchmark is hand-written with the explicit goal of being disjoint from
the training corpus produced by `tinyllama-collect`. We verify this with a
normalized-instruction overlap check (lowercase, punctuation stripped,
whitespace collapsed):

```
benchmark ∩ train  = 0
benchmark ∩ val    = 0
benchmark ∩ test   = 0
```

Re-run the check any time:

```bash
python -c "
import json
from pathlib import Path
from tinyllama_copilot.utils import normalize_for_dedup

bench = [json.loads(l) for l in Path('eval/benchmark.jsonl').open()]
train = [json.loads(l) for l in Path('data/train.jsonl').open()]
overlap = sum(1 for b in bench
              if normalize_for_dedup(b['task'])
              in {normalize_for_dedup(r['instruction']) for r in train})
print('overlap:', overlap)
"
```

## Why these tasks

The benchmark intentionally:

- **Mirrors what a working developer types in a real terminal.** Categories
  weighted by everyday frequency (filesystem + git + shell ≈ 45%), with
  explicit coverage of containers and CI-adjacent tooling.
- **Mixes phrasings.** Imperative (`Compress …`), interrogative (`How
  do I …` was deliberately avoided to reduce style overlap with TLDR-style
  training data), and noun-led (`Disk usage of …`) variants.
- **Covers difficulty levels.** L1 to confirm the model handles the easy
  path; L2 to test flag and pipe composition; L3 to surface failures on
  multi-step pipelines and less-common tools.
- **Has a single canonical reference.** The judge accepts semantically
  equivalent answers — `git branch -d X` vs `git branch --delete X`,
  `ls -la` vs `ls -al` — but a single reference keeps grading reproducible.

## End-to-end pipeline (Phase C3)

1. **Generate candidates** — run each model on the benchmark:

   ```bash
   # Local + free hosted (Gemini)
   tinyllama-baseline --models lora,base,gemini --sleep 4

   # Adding paid models if their keys are set
   tinyllama-baseline --models anthropic,openai
   ```

   Outputs: `eval/candidates/<model>.jsonl` with `{"id": int, "output": str}` rows.

   The `--sleep 4` paces calls to ~15 RPM, staying within the Gemini free tier.

2. **Judge each candidates file**:

   ```bash
   for m in lora base gemini; do
       tinyllama-judge \
           --candidates eval/candidates/$m.jsonl \
           --output    eval/results/$m.jsonl
   done
   ```

   Outputs: `eval/results/<model>.jsonl` with judge scores per row.

3. **Aggregate into a comparison table**:

   ```bash
   tinyllama-summary --by-category
   ```

   Outputs:
   - `eval/results/summary.md` — markdown table for the README
   - `eval/results/summary.csv` — same data, machine-readable

## Methodology

1. **Generation.** For each row in `benchmark.jsonl`, run each candidate model
   (base TinyLlama, LoRA, GPT-4o-mini, Claude Haiku, Llama-3-8B) on the task
   with a fixed prompt and temperature=0.
2. **LLM-as-judge.** Hand the (task, reference, candidate) triple to a
   judge model. Score on three axes (0/1/2 each): correctness, safety,
   plan quality. Force structured JSON output.
3. **Calibration.** Hand-label ~30 rows; compute Cohen's kappa between the
   judge and the human labels. Reported in `judge_calibration.md`.
4. **Reporting.** Produce a model × metric table for the README and a
   per-category breakdown for the writeup.

## Limitations

- 200 rows is a starting point — small enough that a model could overfit to
  it if used during development. Iteration should always be done on `val`,
  not on this benchmark.
- Reference commands are biased toward GNU/Linux POSIX. macOS variants of
  some tools (BSD `find`, `sed`, `tar`) differ. The judge should accept
  POSIX-compliant variants.
- No multi-turn / agentic tasks. Each row is a single-step command. Multi-
  step orchestration (pipelines that need state across calls) is out of
  scope for v1.
- No localization. Tasks are English-only.
