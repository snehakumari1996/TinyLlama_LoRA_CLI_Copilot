"""Aggregate judge results into a per-model comparison table.

Reads ``eval/results/<model>.jsonl`` for each scored model and writes:

  * ``eval/results/summary.csv`` — one row per (model, [category]).
  * ``eval/results/summary.md``  — markdown comparison table for the README.

Usage::

    tinyllama-summary
    tinyllama-summary --by-category
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Return ``{model_name: [judge_result_rows]}`` for every JSONL in *results_dir*."""
    out: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        if path.name in {"summary.jsonl"}:
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out[path.stem] = rows
    return out


def aggregate(rows: list[dict]) -> dict:
    """Mean of each axis across the valid (non-errored) rows."""
    valid = [r for r in rows if "scores" in r]
    if not valid:
        return {
            "n": 0,
            "n_errors": len(rows),
            "correctness": None,
            "safety": None,
            "plan_quality": None,
            "overall": None,
        }
    n = len(valid)
    c = sum(r["scores"]["correctness"] for r in valid) / n
    s = sum(r["scores"]["safety"] for r in valid) / n
    p = sum(r["scores"]["plan_quality"] for r in valid) / n
    return {
        "n": n,
        "n_errors": len(rows) - n,
        "correctness": c,
        "safety": s,
        "plan_quality": p,
        "overall": (c + s + p) / 3,
    }


def aggregate_by_category(rows: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "unknown"), []).append(r)
    return {cat: aggregate(rs) for cat, rs in sorted(by_cat.items())}


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.2f}"


def render_summary_md(per_model: dict[str, dict]) -> str:
    """Markdown comparison table — destined for the README."""
    lines = [
        "# Benchmark results",
        "",
        "Each axis scored 0-2 by an LLM judge over the 200-task held-out benchmark.",
        "Higher is better. `overall` is the mean of the three axes.",
        "",
        "| Model | n | correct | safe | plan | overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    # Sort by overall score desc, putting models with no scores last.
    sortable = [
        (model, agg, agg.get("overall") if agg.get("overall") is not None else -1.0)
        for model, agg in per_model.items()
    ]
    sortable.sort(key=lambda x: -x[2])
    for model, agg, _ in sortable:
        lines.append(
            f"| `{model}` | {agg['n']} | "
            f"{_fmt(agg['correctness'])} | {_fmt(agg['safety'])} | "
            f"{_fmt(agg['plan_quality'])} | {_fmt(agg['overall'])} |"
        )
    return "\n".join(lines) + "\n"


def render_per_category_md(per_model_rows: dict[str, list[dict]]) -> str:
    """Per-category breakdown — useful for the writeup."""
    out = ["", "## Per-category overall score", "", "Mean of (correctness + safety + plan_quality) / 3, per model per category.", ""]
    # Collect category union, ordered by total volume (descending across models)
    cat_counts: dict[str, int] = {}
    for rows in per_model_rows.values():
        for r in rows:
            cat = r.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    cats = sorted(cat_counts, key=lambda c: -cat_counts[c])
    if not cats:
        return ""
    header = "| category | " + " | ".join(f"`{m}`" for m in per_model_rows) + " |"
    sep = "|---|" + "|".join(["---:"] * len(per_model_rows)) + "|"
    out += [header, sep]
    for cat in cats:
        cells = [f"**{cat}**"]
        for model, rows in per_model_rows.items():
            cat_rows = [r for r in rows if r.get("category") == cat]
            agg = aggregate(cat_rows)
            cells.append(_fmt(agg.get("overall")))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def write_csv(per_model: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "n_errors", "correctness", "safety", "plan_quality", "overall"])
        for model, agg in per_model.items():
            w.writerow(
                [
                    model,
                    agg["n"],
                    agg.get("n_errors", 0),
                    agg["correctness"],
                    agg["safety"],
                    agg["plan_quality"],
                    agg["overall"],
                ]
            )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tinyllama-summary",
        description="Aggregate per-model judge results into a comparison table.",
    )
    ap.add_argument("--results-dir", default="eval/results")
    ap.add_argument("--summary-md", default="eval/results/summary.md")
    ap.add_argument("--summary-csv", default="eval/results/summary.csv")
    ap.add_argument("--by-category", action="store_true", help="Append per-category breakdown to summary.md")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    args = _parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"No results directory: {results_dir}")

    per_model_rows = load_results(results_dir)
    if not per_model_rows:
        logger.warning("No JSONL files in %s", results_dir)
        return
    per_model_agg = {model: aggregate(rows) for model, rows in per_model_rows.items()}

    md_text = render_summary_md(per_model_agg)
    if args.by_category:
        md_text += render_per_category_md(per_model_rows)

    md_path = Path(args.summary_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text)
    logger.info("Wrote %s", md_path)

    write_csv(per_model_agg, Path(args.summary_csv))
    logger.info("Wrote %s", args.summary_csv)

    print()
    print(f"{'model':25s}  {'n':>3s}  {'correct':>7s}  {'safe':>5s}  {'plan':>5s}  {'overall':>7s}")
    print("-" * 60)
    for model, agg in sorted(
        per_model_agg.items(),
        key=lambda x: -(x[1].get("overall") or -1.0),
    ):
        if agg["n"] == 0:
            print(f"{model:25s}  {'0':>3s}  (no valid scores; {agg.get('n_errors', 0)} errors)")
        else:
            print(
                f"{model:25s}  {agg['n']:>3d}  "
                f"{agg['correctness']:>7.2f}  {agg['safety']:>5.2f}  "
                f"{agg['plan_quality']:>5.2f}  {agg['overall']:>7.2f}"
            )


if __name__ == "__main__":
    main()
