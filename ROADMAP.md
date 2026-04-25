# Project Roadmap — TinyLlama LoRA CLI Copilot

Goal: turn a Colab proof-of-concept into a resume-grade ML engineering project that gets callbacks.

Each phase ships independent value. Stop at any point and the project is still better than it was.

---

## Phase A — Foundation

Get the code running outside Colab. Make it look like a real Python package.

### A1. Fix critical bugs ✅ DONE
- [x] Fix `argparse` `NameError` in `src/agent.py` (`args` used before defined)
- [x] Replace hardcoded `/content/drive/MyDrive/...` paths with `PROJECT_ROOT` derived from `__file__` + `TINYLLAMA_PROJECT_ROOT` env var override
  - [x] `src/agent.py`
  - [x] `src/train.py`
  - [x] `src/evaluate_static.py`
  - [x] `src/evaluate_dynamic.py`
  - [x] `mean.py`
- [x] Fix undefined `lic_path` in `collect_data.py:167`
- [x] Add missing dependencies to `requirements.txt`: `beautifulsoup4`, `requests`, `tiktoken`, `tqdm`
- [x] Remove `+cu118` torch pin (broke macOS / non-CUDA installs); document GPU install path in comments
- [x] Add file-existence guards (clear error if adapter / dataset / trace missing)
- [x] Add lazy model loading in `agent.py` (importable without loading weights)
- [x] Verify all scripts compile (`python -m py_compile`)

### A2. Project structure & packaging ✅ DONE
- [x] Create `pyproject.toml` (PEP 621 metadata, deps, ruff/mypy/pytest config)
- [x] Create `src/tinyllama_copilot/` package directory
- [x] Move `src/agent.py` → `src/tinyllama_copilot/agent.py` (and the other modules)
- [x] Add `__init__.py` exporting public API (`run`, `generate`, `parse_steps`, `dry`)
- [x] Add `src/tinyllama_copilot/config.py` — central path/env resolution
- [x] Add console script entry points (`tinyllama-cli`, `tinyllama-train`, `tinyllama-collect`, etc.)
- [x] Add `.env.example` documenting all env vars
- [x] Add `py.typed` marker for type-hint visibility downstream
- [x] Verify package imports + public API work (`PYTHONPATH=src python -c "import tinyllama_copilot"`)
- [ ] Final user-side verification: `pip install -e .` then `tinyllama-cli "list files"` (skipped — heavy ML deps)

### A3. Code quality ✅ DONE
- [x] Add full type hints (`list[dict]`, `Path`, etc.) across modules
- [x] Replace `print` with `logging` module + structured logger setup
- [x] Add error handling for missing model/adapter/dataset/trace (helpful messages)
- [x] Cache merged LoRA model to disk so re-runs don't re-merge (save 10-20s/run)
- [x] Add network retries with backoff to `collect_data.py`
- [x] Lazy-import heavy deps (transformers/peft) — `import tinyllama_copilot` stays cheap
- [x] Add docstrings to every public function
- [ ] Run `ruff format` + `ruff check --fix` (deferred — runs in CI in Phase D2)

---

## Phase B — Data & training

Build the data foundation that all eval depends on.

### B1. Scale dataset to 2-5k examples ✅ DONE
- [x] Extend `collect_data.py` with new sources:
  - [x] More TLDR pages (default raised from 400 → 2000)
  - [x] More Stack Overflow tags (20 tags incl. `docker`, `kubectl`, `aws-cli`, `find`, `sed`, `awk`, `ssh`, `curl`, `wget`, `make`, `rsync`, `systemd`, `cron`, `vim`, `tmux`)
  - [x] More DevDocs pages (`bash`, `git`, `docker`, `ssh`)
  - [ ] `man` page synopsis sections (deferred — adds POSIX-only branch)
  - [ ] GitHub `awesome-cli` curated lists (deferred — current sources already give 2k+)
- [x] Add exact + normalized-instruction dedup (`dedup_rows`)
- [x] Add length filters (8-400 chars instruction, 2-4000 chars response)
- [x] Add prose-rejection heuristic (multi-sentence detection) + command-signal heuristic
- [x] Track license per row in JSONL output (`source` + `license` fields preserved end-to-end)
- [x] Add CLI args: `--max-tldr`, `--so-wanted`, `--so-tags`, `--seed`, `--limit`, `--output`
- [x] Add bounded retries with exponential backoff to network calls
- [x] Write `data/data_card.md` documenting sources, schema, cleaning, reproducibility, limitations
- [ ] MinHash near-dup detection (deferred — exact dedup is enough below 10k rows)

### B2. Train/val/test splits + contamination check ✅ DONE
- [x] Implement 80/10/10 random split (seeded — `--seed` arg, default 42)
- [x] Implement normalized-instruction contamination check (lowercase + punctuation-strip)
- [x] Drop val/test rows whose normalized key appears in train (train always wins)
- [x] Save `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`
- [x] Write `data/contamination_report.md` (split sizes, drop counts, drop samples, methodology)
- [x] Add `tinyllama-split` console script + `--train-ratio/--val-ratio/--test-ratio/--seed` args
- [x] Wire train.py to auto-detect splits (uses train.jsonl + val.jsonl if present, falls back to cli_qa.jsonl)
- [x] Pass `eval_dataset` + `eval_strategy=epoch` when validation set is available
- [ ] Semantic contamination check (cosine sim > 0.9) — deferred; current corpus too small to need it

### B3. Retrain LoRA on new dataset
- [x] Update `train.py` to read from `data/train.jsonl` (auto-detects splits)
- [x] Add validation loss tracking (per-epoch via `eval_strategy="epoch"`)
- [x] Save `outputs/training_log.json` (full trainer.state.log_history)
- [x] Save `outputs/training_curves.png` (train + val loss plot, matplotlib best-effort)
- [x] Document Colab flow in `docs/COLAB.md` (cell-by-cell copy-paste)
- [ ] **USER ACTION**: push current changes, run Colab cells, commit new adapter
- [ ] Add early stopping on val loss (deferred — only valuable with multi-epoch runs)

---

## Phase C — Evaluation (centerpiece)

This is the section recruiters care about most.

### C1. Build benchmark test suite (200+ tasks) ✅ DONE
- [x] Curate **200** held-out CLI tasks across **10 categories**:
  - [x] filesystem (30), git (30), shell (30), text (20), container (20)
  - [x] process (17), networking (16), env (15), archive (12), sysinfo (10)
- [x] Each entry: `{id, task, reference, category, difficulty}`
- [x] Difficulty distribution: **52% L1 / 34% L2 / 14% L3**
- [x] Save as `eval/benchmark.jsonl` (200 rows, schema-validated, unique IDs 1-200)
- [x] Document methodology in `eval/README.md` (schema, distribution table, held-out guarantee, planned C2/C3 pipeline, limitations)
- [x] Verify **0 contamination**: `benchmark ∩ {train, val, test} = ∅` under normalized-instruction match

### C2. LLM-as-judge automated grading ✅ DONE (calibration deferred)
- [x] Provider-agnostic judge: Gemini → Anthropic → OpenAI auto-detect
- [x] Free-tier path: **Gemini 2.0 Flash** as default
- [x] Stub provider for offline dry-runs (no API key required)
- [x] Judge prompt covers 3 axes: correctness, safety, plan_quality (each 0-2)
- [x] Force structured JSON output via `response_mime_type="application/json"` / `response_format={"type":"json_object"}`
- [x] Tolerant JSON parser (handles markdown fences + stray prose around the object)
- [x] Bounded retries with exponential backoff
- [x] Build `tinyllama_copilot/judge.py` + `tinyllama-judge` console script
- [x] 8 smoke tests pass (parse, end-to-end stub, CLI, error messages)
- [ ] Hand-label ~30 examples; compute judge–human agreement (Cohen's kappa) — deferred to after first real run

### C3. Baseline comparison sweep ✅ CODE READY (awaits user-side run)
- [x] Build `run_baselines.py` orchestrator with 6 backends:
  - [x] `lora` — local TinyLlama + your LoRA adapter
  - [x] `base` — local TinyLlama with no adapter
  - [x] `gemini` — Gemini 2.0 Flash (free tier)
  - [x] `anthropic` — Claude Haiku (paid, optional)
  - [x] `openai` — GPT-4o-mini (paid, optional)
  - [x] `stub` — for offline pipeline tests
- [x] All backends use the same agent prompt (apples-to-apples)
- [x] Bounded retries with exponential backoff for hosted APIs
- [x] `--limit` arg for partial test runs; `--sleep` for free-tier rate limiting
- [x] Build `summarize.py` aggregator: mean per axis, overall score, per-category breakdown
- [x] Add `tinyllama-baseline` and `tinyllama-summary` console scripts
- [x] End-to-end stub-pipeline smoke test passed (baseline → judge → summary)
- [x] Document end-to-end command sequence in `eval/README.md`
- [ ] **USER ACTION**: get Gemini key, run sweep, commit `eval/candidates/` + `eval/results/`
- [ ] Run benchmark against:
  - [ ] Base TinyLlama-1.1B (no finetune)
  - [ ] Your LoRA-tuned TinyLlama
  - [ ] Gemini 2.0 Flash
  - [ ] (optional) GPT-4o-mini, Claude Haiku
  - [ ] Claude Haiku (via Anthropic API)
  - [ ] Llama-3-8B-Instruct (local or via inference API)
- [ ] Build `eval/run_baselines.py` that orchestrates the sweep
- [ ] Save results to `eval/results/<model>.jsonl` (per-task) + `eval/results/summary.csv`
- [ ] Generate the master comparison table — go in README

### C4. Quantization study
- [ ] Run inference at 4 precisions: fp32, fp16, int8 (bitsandbytes), int4 (bitsandbytes nf4)
- [ ] Measure for each: accuracy on benchmark, latency (tokens/sec), peak memory, model size on disk
- [ ] Plot Pareto curve (accuracy vs. memory)
- [ ] Save `eval/quantization_results.csv` + `eval/quantization_pareto.png`

### C5. Performance profiling
- [ ] Build `eval/perf.py` measuring:
  - [ ] Cold-start time (model load)
  - [ ] Time-to-first-token
  - [ ] Tokens/second sustained
  - [ ] Peak GPU memory (CUDA) and CPU RAM
- [ ] Run on both GPU and CPU
- [ ] Document hardware used (chip, RAM, OS)
- [ ] Save `eval/perf_results.md`

### C6. Red-team safety eval
- [ ] Curate ~50 adversarial prompts:
  - [ ] Destructive commands (`rm -rf`, fork bombs, disk format)
  - [ ] Credential exfil (read `~/.ssh`, env vars, history)
  - [ ] Prompt injection ("ignore previous instructions...")
  - [ ] Social engineering / phishing-adjacent
- [ ] Score each model output:
  - [ ] Did dry-run sandbox hold? (binary)
  - [ ] Did model refuse / warn? (0-2)
  - [ ] Was raw output dangerous if executed? (0-2)
- [ ] Save `eval/redteam.jsonl` + `eval/redteam_results.md`

---

## Phase D — Engineering polish

Make the repo look maintained.

### D1. Pytest suite
- [ ] `tests/test_parser.py` — `parse_steps()` regex behavior (10+ cases)
- [ ] `tests/test_dry.py` — dry-run echoes don't execute
- [ ] `tests/test_config.py` — env var resolution, fallback paths
- [ ] `tests/test_data.py` — collect_data clean_text, dedup logic
- [ ] `tests/conftest.py` — fixtures (sample trace, mock model)
- [ ] Mock heavy imports (no model loading in CI)
- [ ] Aim for ≥80% coverage on non-model code

### D2. GitHub Actions CI
- [ ] `.github/workflows/ci.yml` — lint (ruff) + type check (mypy) + test (pytest)
- [ ] Matrix: Python 3.10, 3.11
- [ ] Add status badge to README
- [ ] Add `dependabot.yml` for dep updates

### D3. Docker, Makefile, pre-commit
- [ ] CPU-only `Dockerfile` (multi-stage build)
- [ ] `docker-compose.yml` (optional, for ergonomics)
- [ ] `Makefile` targets: `install`, `test`, `lint`, `format`, `run`, `eval`, `docker-build`
- [ ] `.pre-commit-config.yaml` with ruff + mypy + check-yaml + end-of-file-fixer
- [ ] Test: `docker run tinyllama-copilot "list files"` produces output

---

## Phase E — Distribution (where callbacks come from)

The visible artifacts. This is what recruiters click.

### E1. HuggingFace Space demo
- [ ] Create HF Space (free CPU tier)
- [ ] Build Gradio UI: textbox in, plan + commands out, dry-run preview
- [ ] Add example prompts as buttons
- [ ] Embed HF Space iframe in README
- [ ] Test: stranger can use it in 30 seconds

### E2. Rewrite README — the showpiece
- [ ] Hero section: title, badges (CI, License, Python, HF Space)
- [ ] Demo GIF or screenshot (use `vhs` or `asciinema`)
- [ ] One-paragraph elevator pitch
- [ ] Architecture diagram (ASCII or Mermaid)
- [ ] Quickstart (`pip install` → `tinyllama-cli "task"`)
- [ ] Results section:
  - [ ] Benchmark comparison table (Phase C3 output)
  - [ ] Quantization Pareto (Phase C4 output)
  - [ ] Performance numbers (Phase C5 output)
  - [ ] Safety red-team summary (Phase C6 output)
- [ ] Training reproducibility: exact commands, hardware, time, cost
- [ ] Limitations section (be honest)
- [ ] Roadmap / future work
- [ ] Acknowledgements (TinyLlama, PEFT, sources)

### E3. Blog post / writeup
- [ ] 800-1200 words on Medium / dev.to / personal site
- [ ] Working title: *"What I learned fine-tuning a 1.1B model for shell commands"*
- [ ] Cover: motivation, data work, eval methodology, what worked, **what failed and why**, what I'd do next
- [ ] Embed key results table + quantization plot
- [ ] Cross-post to LinkedIn with the link

### E4. Resume bullet rewrite
- [ ] Replace generic bullet with quantified version
- [ ] Template:
  > Fine-tuned TinyLlama-1.1B with QLoRA on Nk curated CLI examples; achieved X% accuracy on a Y-task held-out benchmark (vs. Z% base, W% GPT-4o-mini) with an Mb adapter and Ts CPU latency. Built LLM-as-judge eval harness, dry-run safety sandbox, and red-team adversarial test suite. Deployed live demo on HuggingFace Spaces.
- [ ] Update LinkedIn project section to match
- [ ] Update GitHub profile README to feature the project

---

## Tracking summary

| Phase | Steps | Status |
|---|---|---|
| A. Foundation | A1, A2, A3 | A1 ✅ • A2 ✅ • A3 ✅ |
| B. Data & training | B1, B2, B3 | ⬜ |
| C. Evaluation | C1–C6 | ⬜ |
| D. Engineering polish | D1, D2, D3 | ⬜ |
| E. Distribution | E1, E2, E3, E4 | ⬜ |
