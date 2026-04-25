"""Stage-1 data pipeline: scrape, clean, dedup CLI Q&A pairs from open sources.

Outputs
-------
data/cli_qa.jsonl       cleaned Q&A pairs (deduped, filtered)
data/license_map.csv    URL → license map for audit

Run with `python -m tinyllama_copilot.collect_data` or `tinyllama-collect`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import os
import pathlib
import random
import re
import sys
import time
from typing import Any, Iterable, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from tinyllama_copilot.config import DATA_DIR
from tinyllama_copilot.utils import normalize_for_dedup

logger = logging.getLogger(__name__)

# Network defaults
DEFAULT_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF = 1.5
SO_API = "https://api.stackexchange.com/2.3"
TLDR_TREE_URL = (
    "https://api.github.com/repos/tldr-pages/tldr/git/trees/main?recursive=1"
)
TLDR_RAW_BASE = "https://raw.githubusercontent.com/tldr-pages/tldr/main/"

# Default SO tags — broadened from the original 5 to cover modern toolchains.
DEFAULT_SO_TAGS: tuple[str, ...] = (
    "bash", "git", "grep", "tar", "virtualenv",
    "find", "sed", "awk", "ssh", "curl", "wget",
    "docker", "kubectl", "aws-cli", "make", "rsync",
    "systemd", "cron", "vim", "tmux",
)
DEFAULT_DEVDOCS_PAGES: tuple[str, ...] = ("bash", "git", "docker", "ssh")

# Quality thresholds
MIN_INSTR_CHARS = 8
MIN_RESP_CHARS = 2
MAX_INSTR_CHARS = 400
MAX_RESP_CHARS = 4000
# Strong shell signals: $ prompt, backticks, short/long flag, pipe, redirect.
COMMAND_LIKE_RE = re.compile(r"(\$\s|^\$|`|\s-{1,2}[\w-]|^-{1,2}[\w-]|\|\s|>>?\s|<<?\s)")
# Sentence boundary heuristic — used to reject multi-sentence prose responses.
SENTENCE_BOUNDARY_RE = re.compile(r"\.\s+[A-Z]")


def _get(url: str, *, headers: Optional[dict] = None, params: Optional[dict] = None) -> requests.Response:
    """GET with bounded exponential backoff. Raises on final failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            wait = BACKOFF ** attempt
            logger.warning("GET %s failed (%s); retry %d/%d in %.1fs", url, e, attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} attempts: {last_exc}")


def clean_text(txt: str, max_tokens: int = 300) -> str:
    """Collapse whitespace, unescape HTML, cap to ≈max_tokens GPT-2 tokens."""
    import tiktoken  # lazy-import (~70 KB wheel)

    enc = tiktoken.encoding_for_model("gpt2")
    txt = html.unescape(re.sub(r"\s+", " ", txt)).strip()
    tokens = enc.encode(txt)
    if len(tokens) > max_tokens:
        txt = enc.decode(tokens[:max_tokens])
    return txt


def write_jsonl(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------- filtering & dedup -------------------------------------------
def passes_quality(row: dict[str, Any]) -> bool:
    """Drop rows that are too short, too long, or look like multi-sentence prose.

    Accepts rows that either show a strong shell signal (``$``, flag, pipe, …)
    or are short enough not to be prose. Multi-sentence answers are dropped.
    """
    instr = row.get("instruction", "")
    resp = row.get("response", "")
    if not instr or not resp:
        return False
    if len(instr) < MIN_INSTR_CHARS or len(instr) > MAX_INSTR_CHARS:
        return False
    if len(resp) < MIN_RESP_CHARS or len(resp) > MAX_RESP_CHARS:
        return False

    resp_s = resp.strip()
    # Hard reject: multi-sentence prose (fires even if a command appears mid-text).
    if SENTENCE_BOUNDARY_RE.search(resp_s):
        return False
    if COMMAND_LIKE_RE.search(resp_s):
        return True
    # Allow short single-phrase responses (e.g. "git status").
    if len(resp_s) <= 100:
        return True
    return False


def dedup_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact + normalized-instruction duplicates. Keeps first occurrence."""
    seen_instr: set[str] = set()
    seen_pair: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        instr_key = normalize_for_dedup(r["instruction"])
        pair_key = hashlib.sha1(
            (instr_key + "\n" + normalize_for_dedup(r["response"])).encode()
        ).hexdigest()
        if instr_key in seen_instr or pair_key in seen_pair:
            continue
        seen_instr.add(instr_key)
        seen_pair.add(pair_key)
        out.append(r)
    return out


# ------------- 1. TLDR pages (MIT) -----------------------------------------
def fetch_tldr_examples(max_files: int = 400) -> list[dict[str, Any]]:
    """Grab command snippets from tldr-pages via the GitHub trees API."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if (gh_token := os.getenv("GITHUB_TOKEN")):
        headers["Authorization"] = f"token {gh_token}"

    tree = _get(TLDR_TREE_URL, headers=headers).json()
    if not isinstance(tree, dict) or "tree" not in tree:
        raise RuntimeError(f"GitHub tree API failed: {tree.get('message', tree)}")

    md_paths = [
        n["path"]
        for n in tree["tree"]
        if n["path"].startswith("pages/") and n["path"].endswith(".md")
    ][:max_files]

    examples: list[dict[str, Any]] = []
    for path in tqdm(md_paths, desc="tldr"):
        url = TLDR_RAW_BASE + path
        try:
            md_lines = _get(url).text.splitlines()
        except RuntimeError as e:
            logger.warning("Skipping %s: %s", path, e)
            continue
        if not md_lines:
            continue

        title = md_lines[0].lstrip("# ").strip()
        snippet = next((l for l in md_lines if l.startswith("`")), None)
        if not snippet:
            continue

        examples.append(
            {
                "instruction": f"How do I {title}?",
                "response": snippet.strip("`").replace("$ ", ""),
                "source": url,
                "license": "MIT",
            }
        )
    return examples


# ------------- 2. Stack Overflow (CC BY-SA 4.0) ----------------------------
def fetch_so_examples(
    tags: tuple[str, ...] = DEFAULT_SO_TAGS,
    wanted: int = 200,
) -> list[dict[str, Any]]:
    """Pull high-voted accepted answers for the given tags."""
    rows: list[dict[str, Any]] = []
    page = 1
    while len(rows) < wanted:
        try:
            resp = _get(
                f"{SO_API}/search/advanced",
                params={
                    "page": page,
                    "pagesize": 50,
                    "order": "desc",
                    "sort": "votes",
                    "accepted": "True",
                    "tagged": ";".join(tags),
                    "site": "stackoverflow",
                    "filter": "withbody",
                },
            ).json()
        except RuntimeError as e:
            logger.warning("Stack Overflow page %d failed: %s", page, e)
            break

        for item in resp.get("items", []):
            q_title = BeautifulSoup(item["title"], "html.parser").text
            ans_id = item.get("accepted_answer_id")
            if not ans_id:
                continue
            try:
                ans = _get(
                    f"{SO_API}/answers/{ans_id}",
                    params={"filter": "withbody", "site": "stackoverflow"},
                ).json()
            except RuntimeError as e:
                logger.warning("Skipping answer %s: %s", ans_id, e)
                continue
            body = BeautifulSoup(ans["items"][0]["body"], "html.parser").text
            rows.append(
                {
                    "instruction": q_title,
                    "response": body,
                    "source": item["link"],
                    "license": "CC BY-SA 4.0",
                }
            )
            if len(rows) >= wanted:
                break
        page += 1
        time.sleep(0.5)  # stay well under 300 req/day
        if not resp.get("has_more"):
            break
    return rows


# ------------- 3. DevDocs (MPL 2.0) ----------------------------------------
def fetch_devdocs_examples(pages: tuple[str, ...] = DEFAULT_DEVDOCS_PAGES) -> list[dict[str, Any]]:
    """Grab the first <pre> code snippet from each DevDocs page."""
    examples: list[dict[str, Any]] = []
    for topic in pages:
        url = f"https://devdocs.io/{topic}"
        try:
            soup = BeautifulSoup(_get(url).text, "html.parser")
        except RuntimeError as e:
            logger.warning("Skipping DevDocs %s: %s", topic, e)
            continue
        code = soup.find("pre")
        if code and code.text.strip():
            examples.append(
                {
                    "instruction": f"Give an example usage of {topic}.",
                    "response": code.text.strip(),
                    "source": url,
                    "license": "MPL-2.0",
                }
            )
    return examples


# ------------- driver ------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tinyllama-collect",
        description="Scrape, clean, dedup CLI Q&A pairs from open sources.",
    )
    ap.add_argument("--max-tldr", type=int, default=2000, help="Max TLDR pages to fetch")
    ap.add_argument(
        "--so-wanted", type=int, default=200, help="Target Stack Overflow rows to collect"
    )
    ap.add_argument(
        "--so-tags",
        type=str,
        default=",".join(DEFAULT_SO_TAGS),
        help="Comma-separated Stack Overflow tags",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    ap.add_argument("--limit", type=int, default=0, help="Cap final dataset size (0 = no cap)")
    ap.add_argument("--output", type=str, default="cli_qa.jsonl", help="Output filename")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=os.environ.get("TINYLLAMA_LOG_LEVEL", "INFO"))
    args = _parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    so_tags = tuple(t.strip() for t in args.so_tags.split(",") if t.strip())

    logger.info("Collecting data…")
    raw_rows: list[dict[str, Any]] = []
    raw_rows += fetch_tldr_examples(max_files=args.max_tldr)
    logger.info("After TLDR: %d", len(raw_rows))

    raw_rows += fetch_so_examples(tags=so_tags, wanted=args.so_wanted)
    logger.info("After Stack Overflow: %d", len(raw_rows))

    raw_rows += fetch_devdocs_examples()
    logger.info("After DevDocs: %d", len(raw_rows))

    if not raw_rows:
        raise RuntimeError("Collected zero rows — check network connectivity and try again.")

    # Stable shuffle
    rng = random.Random(args.seed)
    rng.shuffle(raw_rows)

    logger.info("Cleaning & token-capping…")
    cleaned: list[dict[str, Any]] = []
    for r in tqdm(raw_rows, desc="sanitize"):
        r["instruction"] = clean_text(r["instruction"], 40)
        r["response"] = clean_text(r["response"], 300)
        cleaned.append(r)

    before = len(cleaned)
    filtered = [r for r in cleaned if passes_quality(r)]
    logger.info("Quality filter: %d → %d (-%d)", before, len(filtered), before - len(filtered))

    deduped = dedup_rows(filtered)
    logger.info("Dedup: %d → %d (-%d)", len(filtered), len(deduped), len(filtered) - len(deduped))

    if args.limit:
        deduped = deduped[: args.limit]
        logger.info("Capped to --limit=%d", args.limit)

    # Final write keeps source + license alongside the training pair, so
    # downstream split/eval steps can audit provenance per-row.
    cli_path = DATA_DIR / args.output
    write_jsonl(deduped, cli_path)
    logger.info("Wrote %s (%d rows, %.1f KB)", cli_path, len(deduped), cli_path.stat().st_size / 1024)

    lic_path = DATA_DIR / "license_map.csv"
    with lic_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["url", "license"])
        for r in deduped:
            wr.writerow([r.get("source", ""), r.get("license", "")])
    logger.info("Wrote %s", lic_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
