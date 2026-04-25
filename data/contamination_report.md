# Contamination report

_Generated: 2026-04-25T12:29:03 UTC_

- Source: `data/cli_qa.jsonl`
- Seed: `42`
- Ratios (train/val/test): `0.80 / 0.10 / 0.10`

## Final split sizes (after contamination drop)

| Split | Rows |
| --- | ---: |
| train | 1571 |
| val | 196 |
| test | 197 |

## Dropped rows (instruction overlapped with train, normalized)

- val: 0 dropped
- test: 0 dropped

## Methodology

- Random shuffle, seeded for reproducibility.
- Contamination key: `normalize_for_dedup(instruction)` — lowercase, punctuation stripped, whitespace collapsed.
- Any val/test row whose key appears in train is dropped (train always wins).
- No semantic / embedding-based check yet — see ROADMAP.md (B2 follow-up).
