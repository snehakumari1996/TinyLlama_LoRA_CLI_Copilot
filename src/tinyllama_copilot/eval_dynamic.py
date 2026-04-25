"""Build eval_dynamic.md from the last N entries in logs/trace.jsonl.

Plan / Safe / Answer columns start at 0 — fill in by hand, then run
`python -m tinyllama_copilot.mean_score` to refresh the means.
"""
from __future__ import annotations

import datetime
import json
import logging
import pathlib
import textwrap
from typing import Optional

from tinyllama_copilot.config import LOG_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

TRACE = LOG_DIR / "trace.jsonl"
OUT = PROJECT_ROOT / "eval_dynamic.md"
N_ROWS = 7


def last_n(path: pathlib.Path, n: int) -> list[bytes]:
    """Read the last *n* lines of a file without loading the whole thing."""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        end = fh.tell()
        buf, chunk, nl = b"", 4096, 0
        while end > 0 and nl < n:
            step = min(chunk, end)
            fh.seek(end - step)
            buf = fh.read(step) + buf
            nl = buf.count(b"\n")
            end -= step
    return buf.splitlines()[-n:]


def pretty_prompt(raw: str) -> str:
    raw = raw.strip()
    return raw.split(".")[0].strip() + "."


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    if not TRACE.exists():
        raise FileNotFoundError(
            f"Trace log not found at {TRACE}. Run the agent first to populate it."
        )

    rows = [json.loads(line) for line in last_n(TRACE, N_ROWS) if line.strip()]
    prompts = [pretty_prompt(r["task"]) for r in rows]

    header = textwrap.dedent(
        f"""\
        # Dynamic evaluation

        Run date (UTC): {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S}

        Scoring rubric — 0 = poor · 1 = partial · 2 = perfect.

        | Prompt | Plan | Safe | Answer |
        | --- | --- | --- | --- |
        """
    )

    lines = []
    for p in prompts:
        p = p.replace("|", "\\|")
        lines.append(f"| {p} | 0 | 0 | 0 |")
    lines.append("| **Mean** | 0 | 0 | 0 |")

    OUT.write_text(header + "\n".join(lines) + "\n")
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
