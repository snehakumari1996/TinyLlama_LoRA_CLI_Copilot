"""Re-compute mean Plan / Safe / Answer scores from eval_*.md tables."""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Optional

from tinyllama_copilot.config import PROJECT_ROOT

TABLES = ["eval_static.md", "eval_dynamic.md"]

_ROW_RE = re.compile(r"\|\s*[^|]+\s*\|\s*\d+(\.\d+)?\s*\|")


def extract_scores(md_path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for line in md_path.read_text().splitlines():
        if _ROW_RE.match(line):
            parts = [x.strip() for x in line.split("|")[1:-1]]
            if len(parts) >= 4 and parts[1].replace(".", "").isdigit():
                rows.append(tuple(float(p) for p in parts[1:4]))  # type: ignore[arg-type]
    return rows


def main(argv: Optional[list[str]] = None) -> None:
    all_scores = {
        tbl: extract_scores(PROJECT_ROOT / tbl) if (PROJECT_ROOT / tbl).exists() else []
        for tbl in TABLES
    }

    print("File            Plan   Safe  Answer  #rows")
    print("-" * 42)
    for name, rows in all_scores.items():
        if rows:
            p, s, a = zip(*rows)
            print(
                f"{name:<15} {statistics.mean(p):5.2f} "
                f"{statistics.mean(s):5.2f} {statistics.mean(a):6.2f}   {len(rows)}"
            )
        else:
            print(f"{name:<15}  n/a   n/a   n/a     0")


if __name__ == "__main__":
    main()
