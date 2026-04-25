# Dataset Card — `cli_qa.jsonl`

A curated dataset of natural-language → shell-command Q&A pairs used to fine-tune
TinyLlama-1.1B with LoRA. Pulled from open, license-compatible sources via
`tinyllama-collect`.

## Schema

Each row in `data/cli_qa.jsonl` is a JSON object with:

| Field | Type | Description |
|---|---|---|
| `instruction` | str | A natural-language description of a CLI task. |
| `response` | str | The corresponding shell snippet (or short answer). |
| `source` | str | URL of the source document (kept for audit). |
| `license` | str | SPDX-style license tag of the source document. |

## Sources

| Source | License | Approx. fraction | Notes |
|---|---|---|---|
| [tldr-pages](https://github.com/tldr-pages/tldr) | MIT | ~70% | Title + first command snippet per page. |
| Stack Overflow | CC BY-SA 4.0 | ~25% | Top-voted accepted answers for selected CLI tags (`bash`, `git`, `docker`, …). |
| [DevDocs](https://devdocs.io) | MPL-2.0 | ~5% | First `<pre>` snippet per indexed page. |

A complete `(source URL → license)` audit trail is written to
`data/license_map.csv` on every run.

## Cleaning & filters

Applied in `tinyllama_copilot.collect_data`:

1. **Whitespace + HTML normalization** (`clean_text`).
2. **Token cap** at ≈40 tokens for instructions and ≈300 tokens for responses.
3. **Length filter** — drop rows with instruction < 8 chars or > 400, response < 4 chars or > 4000.
4. **Has-command heuristic** — drop responses with no shell-command-like content (`$ `, backticks, or leading identifier).
5. **Dedup** — exact + normalized-instruction dedup keeps first occurrence.

## Reproducibility

```bash
# Default settings
python -m tinyllama_copilot.collect_data

# Custom run (more SO rows, capped final size)
python -m tinyllama_copilot.collect_data \
  --max-tldr 2000 \
  --so-wanted 500 \
  --so-tags bash,git,docker,kubectl,find,sed,awk,curl \
  --limit 5000 \
  --seed 42
```

`GITHUB_TOKEN` is read from the environment to raise the GitHub API rate limit
when scraping TLDR.

## Known limitations

- Stack Overflow snippets occasionally include prose that survives the
  command-hint filter; future work: stricter regex / parse `<code>` blocks only.
- TLDR extracts only the first command per page; a richer pipeline would
  emit one row per example block.
- No semantic dedup — two paraphrased questions with identical commands will
  both survive. Consider MinHash or embedding-based dedup once the corpus
  exceeds ~10 k rows.
- License attribution is stored at the URL level, not inlined into responses.
  Downstream redistribution must reference `license_map.csv`.

## Splits

Train / val / test splits are produced by `tinyllama_copilot.split_data`
(see Phase B2 in `ROADMAP.md`).
