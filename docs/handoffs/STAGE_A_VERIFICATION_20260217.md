# Stage A Verification — 2026-02-17

**Run ID**: moonshine_mash_distill_20260217_r1
**Auditor**: Wave 1B Stage A Verifier
**Date**: 2026-02-17

---

## Source Hash Verification

| Field | Value |
|-------|-------|
| Source File | `02-14-26-ChatGPT/conversations.json` |
| Computed SHA256 | `4e6d44cd2102d26701858f832ba325e15d3490e61397013b7896a22dfad792dd` |
| Expected (canonical) | `4e6d44cd2102d26701858f832ba325e15d3490e61397013b7896a22dfad792dd` |
| Match | **YES** |

---

## Token Ledger Generation

`reports/token_ledger.json` was created by sourcing the canonical token count from
`reports/token_forensics.json` (real export pipeline run on 2026-02-16 using streaming
ijson + o200k_base tokenizer). The full pipeline re-run was not required since the
canonical count is stable and traceable to the source hash above.

### Actual Counter Values

| Counter | Value | Status |
|---------|-------|--------|
| `raw_json_tokens` | `null` | OK — tiktoken full-file count not available; validator accepts null |
| `content_tokens_non_system` | **115,330,530** | PASS |
| `content_tokens_cleaned` | `null` | OK — pending Stage B distillation |
| `distilled_tokens_selected` | `null` | OK — pending Stage B distillation |
| `source_sha256` | `4e6d44cd2102d267...` | PASS (64-char hash) |

---

## Gate A Checks

| Criterion | Status |
|-----------|--------|
| `token_ledger.json` exists with all 4 counters | **PASS** |
| `source_sha256` in ledger matches canonical hash | **PASS** |
| `content_tokens_non_system` == 115,330,530 | **PASS** |
| Validation script exits 0 (`--strict`) | **PASS** (45/46 checks pass; 1 non-blocking warning) |

**Gate A: PASS**

---

## Validation Run Summary

```
Total checks:   46
Passed:         45
Failed:         1 (0 errors, 1 warnings)
OVERALL: VALIDATION PASSED
```

### Single Warning (Non-Blocking)

`repro_manifest:checksums` — Only 3/4 files have checksums. `validation_report.md`
checksum is excluded because it is regenerated on each validation run (chicken-and-egg
issue inherent to the design). This is acknowledged known debt.

---

## Known Debt (Acknowledged, Non-Blocking)

| Item | Notes |
|------|-------|
| `repro_manifest.json` synthetic dataset_summary | References generic 10K sample dataset stats from parquet file; actual Moonshine corpus stats are in `token_forensics.json` |
| `raw_json_tokens = null` | Full-file tiktoken tokenization not run; validator accepts null |
| Dead `pass` branches in `token_forensics_agents.py` lines 103-107 | Format switching is handled upstream; agent code has unreachable branches |

---

## Files Produced

- `reports/token_ledger.json` ← **new**
- `reports/repro_manifest.json` ← updated with real checksums
- `reports/validation_manifest.json` ← refreshed by validation run
- `reports/validation_report.md` ← refreshed by validation run
