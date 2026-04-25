"""Split a cleaned dataset into train/val/test with a contamination check.

Reads ``data/cli_qa.jsonl``, writes ``data/train.jsonl``, ``data/val.jsonl``,
``data/test.jsonl`` and a human-readable ``data/contamination_report.md``.

Why this matters: an evaluation set that overlaps with the training set
inflates accuracy. This module enforces both exact and normalized-instruction
disjointness, drops contaminated rows from val/test (never from train), and
records every drop in the report so the result is reproducible.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
from typing import Any, Optional

from tinyllama_copilot.config import DATA_DIR
from tinyllama_copilot.utils import normalize_for_dedup

logger = logging.getLogger(__name__)


def load_jsonl(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_indices(n: int, ratios: tuple[float, float, float], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Return (train_idx, val_idx, test_idx) for *n* rows, shuffled with *seed*."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} ({sum(ratios)})")
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def find_contamination(
    holdout: list[dict[str, Any]],
    train_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition *holdout* into (clean, contaminated) using normalized-instruction keys."""
    clean: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []
    for r in holdout:
        if normalize_for_dedup(r["instruction"]) in train_keys:
            contaminated.append(r)
        else:
            clean.append(r)
    return clean, contaminated


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tinyllama-split",
        description="Split cli_qa.jsonl into train/val/test with contamination check.",
    )
    ap.add_argument("--input", default="cli_qa.jsonl", help="Input file under data/")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=os.environ.get("TINYLLAMA_LOG_LEVEL", "INFO"))
    args = _parse_args(argv)

    src = DATA_DIR / args.input
    if not src.exists():
        raise FileNotFoundError(
            f"Input not found: {src}. Run `tinyllama-collect` first."
        )

    rows = load_jsonl(src)
    logger.info("Loaded %d rows from %s", len(rows), src)
    if len(rows) < 10:
        raise RuntimeError(
            f"Only {len(rows)} rows — too few to split meaningfully. "
            "Re-run collection with more data."
        )

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train_idx, val_idx, test_idx = split_indices(len(rows), ratios, args.seed)
    train = [rows[i] for i in train_idx]
    val_raw = [rows[i] for i in val_idx]
    test_raw = [rows[i] for i in test_idx]

    train_keys = {normalize_for_dedup(r["instruction"]) for r in train}
    val, val_contam = find_contamination(val_raw, train_keys)
    test, test_contam = find_contamination(test_raw, train_keys)

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    test_path = DATA_DIR / "test.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)
    write_jsonl(test, test_path)

    logger.info(
        "Split: train=%d, val=%d (-%d contaminated), test=%d (-%d contaminated)",
        len(train),
        len(val),
        len(val_contam),
        len(test),
        len(test_contam),
    )

    report = _build_report(
        src_name=src.name,
        seed=args.seed,
        ratios=ratios,
        train=train,
        val=val,
        test=test,
        val_contam=val_contam,
        test_contam=test_contam,
    )
    report_path = DATA_DIR / "contamination_report.md"
    report_path.write_text(report)
    logger.info("Wrote %s", report_path)


def _build_report(
    *,
    src_name: str,
    seed: int,
    ratios: tuple[float, float, float],
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    val_contam: list[dict[str, Any]],
    test_contam: list[dict[str, Any]],
) -> str:
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    lines = [
        "# Contamination report",
        "",
        f"_Generated: {ts} UTC_",
        "",
        f"- Source: `data/{src_name}`",
        f"- Seed: `{seed}`",
        f"- Ratios (train/val/test): `{ratios[0]:.2f} / {ratios[1]:.2f} / {ratios[2]:.2f}`",
        "",
        "## Final split sizes (after contamination drop)",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
        f"| train | {len(train)} |",
        f"| val | {len(val)} |",
        f"| test | {len(test)} |",
        "",
        "## Dropped rows (instruction overlapped with train, normalized)",
        "",
        f"- val: {len(val_contam)} dropped",
        f"- test: {len(test_contam)} dropped",
        "",
    ]
    if val_contam or test_contam:
        lines += [
            "### Sample of dropped instructions",
            "",
        ]
        for tag, drops in [("val", val_contam), ("test", test_contam)]:
            for r in drops[:5]:
                lines.append(f"- ({tag}) `{r['instruction'][:120]}`")
        lines.append("")
    lines += [
        "## Methodology",
        "",
        "- Random shuffle, seeded for reproducibility.",
        "- Contamination key: `normalize_for_dedup(instruction)` — lowercase, "
        "punctuation stripped, whitespace collapsed.",
        "- Any val/test row whose key appears in train is dropped (train always wins).",
        "- No semantic / embedding-based check yet — see ROADMAP.md (B2 follow-up).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
