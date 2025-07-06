"""
Re-compute mean Plan / Safe / Answer scores from eval_*.md tables.

Usage:
$ python mean.py
"""
from pathlib import Path
import re, statistics

ROOT = Path("/content/drive/MyDrive/fenrir-mini-lora-agent")
tables = ["eval_static.md", "eval_dynamic.md"]

def extract_scores(md_path):
    rows = []
    for line in md_path.read_text().splitlines():
        # markdown row: | Prompt | Plan | Safe | Answer |
        if re.match(r"\|\s*[^|]+\s*\|\s*\d+(\.\d+)?\s*\|", line):
            parts = [x.strip() for x in line.split("|")[1:-1]]
            if len(parts) >= 4 and parts[1].replace(".","").isdigit():
                rows.append(tuple(float(p) for p in parts[1:4]))
    return rows

all_scores = {tbl: extract_scores(ROOT / tbl) for tbl in tables}

print("File            Plan   Safe  Answer  #rows")
print("-"*42)
for name, rows in all_scores.items():
    if rows:
        p, s, a = zip(*rows)
        print(f"{name:<15} {statistics.mean(p):5.2f} {statistics.mean(s):5.2f} {statistics.mean(a):6.2f}   {len(rows)}")
    else:
        print(f"{name:<15}  n/a   n/a   n/a     0")
