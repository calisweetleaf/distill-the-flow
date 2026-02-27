# Release Alignment — 2026-02-17

**Run ID**: moonshine_mash_distill_20260217_r1
**Date**: 2026-02-17
**Scope**: Stage A + Stage B Repair Wave

---

## Artifact State Table

| Artifact | Expected | Actual State | Status |
|----------|----------|--------------|--------|
| `reports/token_ledger.json` | Exists, all 4 counters, source_sha256 locked | EXISTS — `content_tokens_non_system=115,330,530`, `distilled_tokens_selected=104,321,772` | **VERIFIED** |
| `reports/moonshine_corpus.db` | 3 tables: conversations, messages, distilled_conversations | EXISTS — all 3 tables present | **VERIFIED** |
| `reports/moonshine_distillation_manifest.json` | Exists with policy + budget fields | EXISTS — `distilled_conversations=1326`, `budget_status=in_band` | **VERIFIED** |
| `reports/repro_manifest.json` | Exists with checksums | EXISTS — real checksums for 2 stable files | **VERIFIED** (known debt noted) |
| `visualizations/distilled_corpus_dashboard.png` | Exists at 300 dpi | EXISTS — 478 KB | **VERIFIED** |
| `visualizations/quality_metrics_distilled_timeseries.png` | Exists at 300 dpi | EXISTS — 259 KB | **VERIFIED** |
| `docs/EVIDENCE_MATRIX_20260217.md` | Exists | EXISTS | **VERIFIED** |
| `docs/STAGE_A_VERIFICATION_20260217.md` | Exists | EXISTS | **VERIFIED** |
| `docs/STAGE_B_VERIFICATION_20260217.md` | Exists | EXISTS | **VERIFIED** |
| `docs/QUERY_CONTRACTS_GATE_B.md` | Exists | EXISTS | **VERIFIED** |

---

## Gate Outcomes

### Gate A

| Check | Status |
|-------|--------|
| `token_ledger.json` exists with 4 counters | PASS |
| `content_tokens_non_system` == 115,330,530 | PASS |
| `source_sha256` == `4e6d44cd...` | PASS |
| `validate_token_ledger()` passes | PASS |
| `scripts/run_validation.py --strict` exits 0 | PASS (44/45, 1 non-blocking warning) |

**Gate A: PASS**

### Gate B

| Check | Status |
|-------|--------|
| `distilled_conversations` table present | PASS |
| Provenance columns in schema | PASS |
| QC-1 (dominant topic non-empty) | PASS (640 rows for `architecture`) |
| QC-2 (token budget in 90M–110M band) | PASS (104.3M canonical tokens) |
| QC-3 (gold-tier provenance locked) | PASS (29 rows, all with canonical hash) |
| `moonshine_distillation_manifest.json` written | PASS |
| Both distilled visualizations generated | PASS |

**Gate B: PASS**

### Adjudication Signal

`forward` — both gates pass, no revision requests.

---

## Acknowledged Non-Blocking Debt

| Item | Impact | Owner |
|------|--------|-------|
| `repro_manifest.json` only has 2 file checksums (token_row_metrics.parquet, tokenizer_benchmark.csv) | Low — validation warns but passes; `token_forensics.json` and `validation_report.md` are regenerated on each validation run | Stage D (finisher) |
| `DataProfilerAgent` dead `pass` branches (lines 103–107 in `token_forensics_agents.py`) | Low — format handling works via upstream `_load_conversations()`; dead code only | Stage D (optional cleanup) |
| `tone_shift=0.0` hardcoded for all conversations | Low — metric stored as 0.0; not used in distillation policy | Future enhancement |
| `raw_json_tokens = null` in token_ledger.json | Low — validator accepts null; tiktoken full-file scan not re-run | Known acceptable |

---

## Token Accounting Final

| Scope | Value | Source |
|-------|-------|--------|
| Raw JSON estimate | ~458.5M | Historical estimate |
| Canonical source-locked (non-system) | **115,330,530** | `token_forensics.json` (o200k_base, streaming ijson) |
| Moonshine heuristic baseline (DB sum) | 51,524,462 | `moonshine_corpus.db` `SUM(total_tokens)` |
| Distilled canonical tokens | **104,321,772** | `token_ledger.json` `distilled_tokens_selected` |
| Distilled fraction | 90.4% | Of canonical source |
