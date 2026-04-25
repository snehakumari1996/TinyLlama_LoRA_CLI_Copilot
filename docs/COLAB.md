# Training on Google Colab

End-to-end flow: clone → install → collect data → split → train → download adapter.

> **Before you start:**
> 1. Push your latest local changes to GitHub. The CLI scripts (`tinyllama-collect`, `tinyllama-split`, `tinyllama-train`, …) live in code that hasn't been pushed yet — without the push, Colab will clone an outdated version.
> 2. In Colab, set the runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`.

Each block below is one Colab cell. Copy-paste in order.

---

## Cell 1 — verify GPU is attached

```bash
!nvidia-smi
```

Should show one GPU (Tesla T4 typically). If you see "command not found" or no output, switch the runtime to GPU and try again.

## Cell 2 — clone the repo

```bash
!git clone https://github.com/snehakumari1996/TinyLlama_LoRA_CLI_Copilot.git
%cd TinyLlama_LoRA_CLI_Copilot
```

## Cell 3 — install the package

```bash
!pip install -q -e . matplotlib
```

`-e .` installs in editable mode so the `tinyllama-*` console scripts resolve. `matplotlib` is needed for the training-curves PNG (technically optional — training works without it).

## Cell 4 — collect data (skip if `data/cli_qa.jsonl` already has enough rows)

```bash
# Optional: set GITHUB_TOKEN to raise the GitHub API rate limit
# %env GITHUB_TOKEN=ghp_xxxxxxxxxxxx

!tinyllama-collect --max-tldr 2000 --so-wanted 500 --limit 5000 --seed 42
```

This pulls TLDR, Stack Overflow, and DevDocs sources. Takes 3–8 minutes depending on the network. End state: `data/cli_qa.jsonl` with up to 5000 deduped, quality-filtered rows + `data/license_map.csv`.

## Cell 5 — split into train/val/test

```bash
!tinyllama-split --seed 42
```

Writes `data/train.jsonl` (80%), `data/val.jsonl` (10%), `data/test.jsonl` (10%) plus `data/contamination_report.md` documenting what was dropped.

## Cell 6 — train

```bash
!tinyllama-train
```

Single-epoch QLoRA-style FP32 fine-tune on the new train split, with per-epoch validation loss tracking when `val.jsonl` is present. ~20–30 min on T4 for ~4000 train rows. Outputs:

- `lora_adapter/` — the trained adapter
- `outputs/training_log.json` — raw `trainer.state.log_history`
- `outputs/training_curves.png` — train (+ val if available) loss plot

## Cell 7 — view the loss curves inline

```python
from IPython.display import Image
Image("outputs/training_curves.png")
```

## Cell 8 — quick sanity check the adapter works

```bash
!tinyllama-cli "create a git branch called feature-x and switch to it"
```

Should print a numbered plan + `$`-prefixed commands. The merged model gets cached on first run; subsequent calls are fast.

## Cell 9 — package the adapter for download

```bash
!zip -qr lora_adapter.zip lora_adapter outputs/training_log.json outputs/training_curves.png
from google.colab import files
files.download("lora_adapter.zip")
```

Unzip locally into the repo root (next to `pyproject.toml`) — the `lora_adapter/` directory will overwrite the old one. Commit the result.

---

## Re-running with a different seed / hyperparameters

The hyperparameters are constants at the top of `src/tinyllama_copilot/train.py` (`BATCH`, `GRAD_ACC`, `LR`, `EPOCHS`, `MAXLEN`). To experiment, edit the file and rerun Cell 6.

For seed sweeps:

```bash
!tinyllama-split --seed 7 && tinyllama-train
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `tinyllama-collect: command not found` | Cell 3 didn't install | Re-run Cell 3 and check the output for errors |
| `CUDA out of memory` during training | Sequence too long, batch too big | Lower `BATCH` to 2 or `MAXLEN` to 384 in `train.py` |
| GitHub API rate-limited during collect | Anonymous quota exhausted | Set `GITHUB_TOKEN` and re-run Cell 4 |
| `Trace log not found` from eval scripts | Agent never ran | Run Cell 8 once to populate `logs/trace.jsonl` |
