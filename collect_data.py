#!/usr/bin/env python
"""
Automated Stage-1 pipeline for Fenrir “mini-LoRA + agent” task.

Outputs
-------
data/cli_qa.jsonl      cleaned Q&A pairs (≥150 rows)
data/license_map.csv   URL → license map for audit
"""
import json, re, html, time, csv, random, urllib.parse, pathlib, sys
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os, requests, pathlib, re

# ------------- helper utils -------------------------------------------------
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

def clean_text(txt: str, max_tokens: int = 300) -> str:
    "Collapse whitespace, strip HTML, cap to ≈max_tokens GPT-2 tokens."
    import tiktoken                                  # lazy-import (≈70 KB wheel)
    enc = tiktoken.encoding_for_model("gpt2")
    txt = html.unescape(re.sub(r"\s+", " ", txt)).strip()
    tokens = enc.encode(txt)
    if len(tokens) > max_tokens:
        txt = enc.decode(tokens[:max_tokens])
    return txt

def write_jsonl(rows: List[Dict], path: pathlib.Path):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------- 1. TLDR pages  (MIT) ------------------------------------------
# --- replace the old fetch_tldr_examples() with this ------------------------

def fetch_tldr_examples(max_files: int = 400) -> list[dict]:
    """
    Grab command snippets from tldr-pages in one sweep.

    • Uses the GitHub ‘git/trees’ API to fetch the full file list.
    • Downloads raw markdown via raw.githubusercontent.com (no extra API calls).
    • Returns a list[dict] with the keys expected by the downstream pipeline.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if (tok := os.getenv("GITHUB_TOKEN")):
        headers["Authorization"] = f"token {tok}"

    TREE_URL = (
        "https://api.github.com/repos/tldr-pages/tldr/git/trees/main?recursive=1"
    )
    tree = requests.get(TREE_URL, headers=headers).json()

    if not isinstance(tree, dict) or "tree" not in tree:
        raise RuntimeError(f"GitHub tree API failed: {tree.get('message', tree)}")

    md_paths = [
        n["path"] for n in tree["tree"]
        if n["path"].startswith("pages/") and n["path"].endswith(".md")
    ][:max_files]

    examples = []
    RAW_BASE = "https://raw.githubusercontent.com/tldr-pages/tldr/main/"

    for path in md_paths:
        url = RAW_BASE + path
        md_lines = requests.get(url).text.splitlines()
        if not md_lines:
            continue

        title = md_lines[0].lstrip("# ").strip()          # e.g. "git checkout"
        snippet = next((l for l in md_lines if l.startswith("`")), None)
        if not snippet:
            continue

        instruction = f"How do I {title}?"
        response = snippet.strip("`").replace("$ ", "")
        examples.append(
            {
                "instruction": instruction,
                "response": response,
                "source": url,
                "license": "MIT",
            }
        )
    return examples

# ------------- 2. Stack Overflow (CC BY-SA 4.0) -----------------------------
SO_API = "https://api.stackexchange.com/2.3"

def fetch_so_examples(tags=("bash", "git", "grep", "tar", "virtualenv"),
                      wanted: int = 40) -> List[Dict]:
    rows, page = [], 1
    while len(rows) < wanted:
        resp = requests.get(
            f"{SO_API}/search/advanced",
            params=dict(page=page, pagesize=50, order="desc", sort="votes",
                        accepted="True", tagged=";".join(tags),
                        site="stackoverflow", filter="withbody")
        ).json()
        for item in resp.get("items", []):
            q_title = BeautifulSoup(item["title"], "html.parser").text
            ans_id  = item.get("accepted_answer_id")
            if not ans_id:
                continue
            ans = requests.get(f"{SO_API}/answers/{ans_id}",
                               params=dict(filter="withbody", site="stackoverflow")).json()
            body = BeautifulSoup(ans["items"][0]["body"], "html.parser").text
            rows.append({"instruction": q_title, "response": body,
                         "source": item["link"], "license": "CC BY-SA 4.0"})
            if len(rows) >= wanted:
                break
        page += 1
        time.sleep(0.5)                 # stay well under 300 req/day
        if not resp.get("has_more"):
            break
    return rows

# ------------- 3. (Optional) DevDocs Bash pages (MPL 2.0) -------------------
def fetch_devdocs_examples(pages=("bash", "git")) -> List[Dict]:
    """Grab first <pre><code> snippet per DevDocs page."""
    examples = []
    for topic in pages:
        url = f"https://devdocs.io/{topic}"
        soup = BeautifulSoup(requests.get(url).text, "html.parser")
        code = soup.find("pre")
        if code and code.text.strip():
            instruction = f"Give an example usage of {topic}."
            examples.append(
                {"instruction": instruction, "response": code.text.strip(),
                 "source": url, "license": "MPL-2.0"}
            )
    return examples

# ------------- 4. Crunch all sources ----------------------------------------
def main():
    print("Collecting data …")
    all_rows: List[Dict] = []
    all_rows += fetch_tldr_examples()
    print(f"✔ TLDR examples: {len(all_rows)}")

    all_rows += fetch_so_examples()
    print(f"✔ Stack Overflow examples: {len(all_rows)}")

    all_rows += fetch_devdocs_examples()
    print(f"✔ DevDocs examples: {len(all_rows)}")

    random.shuffle(all_rows)

    # ---------- cleaning ----------------------------------------------------
    print("Cleaning & token-capping …")
    clean_rows = []
    for r in tqdm(all_rows, desc="sanitize"):
        r["instruction"] = clean_text(r["instruction"], 40)
        r["response"]    = clean_text(r["response"], 300)
        clean_rows.append(
            {"instruction": r["instruction"], "response": r["response"]}
        )

    # keep first 160 (buffer) ► 150 min. guaranteed
    cli_path = DATA_DIR / "cli_qa.jsonl"
    write_jsonl(clean_rows[:400], cli_path)
    print(f" Wrote {cli_path} ({cli_path.stat().st_size/1024:.1f} KB)")

    # ---------- license audit ------------ lic_path = DATA_DIR / "license_map.csv"
    with lic_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["url", "license"])
        for r in all_rows:
            wr.writerow([r["source"], r["license"]])
    print(f" Wrote {lic_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
