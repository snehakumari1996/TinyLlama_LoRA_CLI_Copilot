"""FP32 LoRA fine-tuning entrypoint."""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from tinyllama_copilot.config import ADAPTER_DIR, BASE_MODEL, DATA_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)

# Hyperparameters
BATCH = 4
GRAD_ACC = 8
LR = 2e-4
EPOCHS = 1
MAXLEN = 512

TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
LEGACY_PATH = DATA_DIR / "cli_qa.jsonl"


def _resolve_data_paths() -> tuple["pathlib.Path", "pathlib.Path | None"]:
    """Prefer train/val splits if present; fall back to the un-split file."""
    if TRAIN_PATH.exists():
        return TRAIN_PATH, (VAL_PATH if VAL_PATH.exists() else None)
    if LEGACY_PATH.exists():
        logger.warning(
            "Using un-split %s. Run `tinyllama-split` to enable val tracking.",
            LEGACY_PATH.name,
        )
        return LEGACY_PATH, None
    raise FileNotFoundError(
        "No training data found. Run `tinyllama-collect` then `tinyllama-split`."
    )


def main(argv: Optional[list[str]] = None) -> None:
    import pathlib  # noqa: F401  (used in type hint above)

    logging.basicConfig(level=os.environ.get("TINYLLAMA_LOG_LEVEL", "INFO"))
    train_path, val_path = _resolve_data_paths()

    os.environ["ACCELERATE_DISABLE_TENSOR_PARALLEL"] = "1"

    # Lazy heavy imports
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    logger.info("[1] Loading train from %s", train_path)
    train_rows = [json.loads(line) for line in train_path.open()]
    raw_train = Dataset.from_list(train_rows)
    raw_val = None
    if val_path is not None:
        logger.info("    Loading val from %s", val_path)
        val_rows = [json.loads(line) for line in val_path.open()]
        if val_rows:
            raw_val = Dataset.from_list(val_rows)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token = tok.eos_token

    def fmt(r):
        return tok(
            f"<s>[INST] {r['instruction']} [/INST] {r['response']} </s>",
            truncation=True,
            max_length=MAXLEN,
        )

    train_ds = raw_train.map(fmt, remove_columns=raw_train.column_names)
    val_ds = raw_val.map(fmt, remove_columns=raw_val.column_names) if raw_val else None

    def collate(batch):
        pad = tok.pad(batch, return_tensors="pt")
        pad["labels"] = pad["input_ids"].clone()
        return pad

    logger.info("[2] Building model + LoRA (FP32)")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    lora = get_peft_model(
        base,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    lora.enable_input_require_grads()

    logger.info("[3] Training")
    steps = len(train_ds) // (BATCH * GRAD_ACC) + 1
    eval_kwargs = (
        {"eval_strategy": "epoch", "per_device_eval_batch_size": BATCH}
        if val_ds is not None
        else {}
    )
    args = TrainingArguments(
        output_dir=str(OUTPUTS_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        max_steps=steps,
        fp16=False,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        **eval_kwargs,
    )

    trainer = Trainer(
        model=lora,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train()

    logger.info("[4] Saving adapter to %s", ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    lora.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))

    _save_training_artifacts(trainer)
    logger.info("Done. Adapter saved to %s", ADAPTER_DIR)


def _save_training_artifacts(trainer) -> None:
    """Persist trainer.log_history as JSON + render a loss curve PNG (best-effort)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS_DIR / "training_log.json"
    log_path.write_text(json.dumps(trainer.state.log_history, indent=2))
    logger.info("Wrote %s", log_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed — skipping training_curves.png")
        return

    history = trainer.state.log_history
    train_steps = [h["step"] for h in history if "loss" in h and "eval_loss" not in h]
    train_losses = [h["loss"] for h in history if "loss" in h and "eval_loss" not in h]
    eval_steps = [h["step"] for h in history if "eval_loss" in h]
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(7, 4))
    if train_losses:
        ax.plot(train_steps, train_losses, label="train", marker="o", markersize=3)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="val", marker="s", markersize=4, color="tab:red")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("LoRA fine-tuning loss")
    ax.grid(alpha=0.3)
    if train_losses or eval_losses:
        ax.legend()
    fig.tight_layout()
    plot_path = OUTPUTS_DIR / "training_curves.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", plot_path)


if __name__ == "__main__":
    main()
