# Token Reconciliation Report

Generated: 2026-02-15T17:39:59.576948+00:00

## Primary Numbers

- `token-count-guess.md` full-file token count (`o200k_base` on raw JSON text): **458,527,657**
- `reports/validation_manifest.json` token count (validated dataset): **9,405,521**
- `reports/validation_manifest.json` sample count: **10,000**
- `kimi-updates.md` stated corpus tokens: **51.5M**

## Why The Numbers Differ

1. Different unit of analysis:
`token-count-guess.md` counts tokens in the entire raw JSON file text, including keys, punctuation, metadata, and all serialized content.
2. Different dataset scope:
`reports/token_forensics.md` is generated from `token_row_metrics.parquet` validated in `validation_manifest.json`, which currently contains 10,000 rows.
3. Different tokenizers/estimators:
`token-count-guess.md` uses `o200k_base`; validated reports use forensics pipeline token columns and benchmark stats.
4. Different method in Kimi update:
`kimi-updates.md` references Moonshine analyzer output that uses heuristic token estimation (word-based), not raw BPE tokenization.

## Ratio Check

- Full-file count / validated forensics count: **48.75x**

## Monthly Distribution From `moonshine_corpus.db`

| Month (UTC) | Conversations | Estimated Tokens |
|---|---:|---:|
| 2023-03 | 1 | 368 |
| 2023-04 | 11 | 9,073 |
| 2023-05 | 6 | 7,290 |
| 2023-07 | 1 | 503 |
| 2023-10 | 2 | 687 |
| 2024-01 | 7 | 7,530 |
| 2024-08 | 5 | 10,533 |
| 2024-09 | 2 | 6,009 |
| 2024-10 | 1 | 1,196 |
| 2024-11 | 3 | 7,979 |
| 2024-12 | 60 | 235,063 |
| 2025-01 | 51 | 289,734 |
| 2025-02 | 165 | 1,151,993 |
| 2025-03 | 254 | 4,566,336 |
| 2025-04 | 71 | 3,426,770 |
| 2025-05 | 112 | 4,018,233 |
| 2025-06 | 88 | 3,625,527 |
| 2025-07 | 105 | 5,477,873 |
| 2025-08 | 144 | 6,423,767 |
| 2025-09 | 60 | 5,793,198 |
| 2025-10 | 89 | 5,705,953 |
| 2025-11 | 40 | 2,785,659 |
| 2025-12 | 36 | 1,879,311 |
| 2026-01 | 82 | 3,300,506 |
| 2026-02 | 43 | 2,793,371 |

## June Chunk Export

- Output: `reports/june_5mb_chunk.md`
- Target size: 5,242,880 bytes
- Actual size: 5,242,742 bytes
- June conversations included: 26 / 88
- Messages included: 4,821

## Bottom Line

Tokens did not disappear. The artifacts are measuring different things (full raw export text vs validated subset vs heuristic conversation estimates).
