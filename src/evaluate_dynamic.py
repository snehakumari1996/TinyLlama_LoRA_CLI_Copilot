#!/usr/bin/env python
"""
evaluate_dynamic.py
Builds **eval_dynamic.md** from the last 7 log entries in logs/trace.jsonl

• Plan column : 0/1/2 placeholder  
• Safe column : 0/1/2 placeholder  
• Answer      : 0/1/2 placeholder  

After it runs, open eval_dynamic.md and replace the zeros with your scores,
then run  src/update_mean.py  to refresh the Mean row.
"""
import json, pathlib, datetime, itertools, textwrap

ROOT  = pathlib.Path("/content/drive/MyDrive/fenrir-mini-lora-agent")
TRACE = ROOT / "logs" / "trace.jsonl"
OUT   = ROOT / "eval_dynamic.md"

N_ROWS = 7                     # how many recent tasks to include

# ── 1  load latest N_ROWS jsonl entries ───────────────────────────────────
def last_n(path: pathlib.Path, n: int):
    """Read the last *n* lines of a large file without slurping everything."""
    with path.open("rb") as fh:
        fh.seek(0, 2)          # to EOF
        end = fh.tell()
        buf, chunk, nl = b"", 4096, 0
        while end > 0 and nl < n:
            step = min(chunk, end)
            fh.seek(end - step)
            buf = fh.read(step) + buf
            nl = buf.count(b"\n")
            end -= step
    return buf.splitlines()[-n:]

rows = [json.loads(l) for l in last_n(TRACE, N_ROWS) if l.strip()]

# ── 2  extract the user prompt (first line of "task") ─────────────────────
def pretty_prompt(raw: str) -> str:
    raw = raw.strip()
    # keep it short – first sentence only
    return raw.split(".")[0].strip() + "."

prompts = [pretty_prompt(r["task"]) for r in rows]

# ── 3  compose the markdown table ─────────────────────────────────────────
header = textwrap.dedent(f"""\
    # Dynamic evaluation

    Run date (UTC): {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S}

    Scoring rubric — 0 = poor · 1 = partial · 2 = perfect.

    | Prompt | Plan | Safe | Answer |
    | --- | --- | --- | --- |
""")

lines = []
for p in prompts:
    # escape pipes inside the prompt for Markdown safety
    p = p.replace("|", "\\|")
    lines.append(f"| {p} | 0 | 0 | 0 |")

lines.append("| **Mean** | 0 | 0 | 0 |")

# ── 4  write file ─────────────────────────────────────────────────────────
OUT.write_text(header + "\n".join(lines) + "\n")
print(" Wrote", OUT.relative_to(ROOT))
