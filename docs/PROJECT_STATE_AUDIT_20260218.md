# Project State Audit - 2026-02-18

## Objective

Reconcile current workspace state against `TOKEN_FORENSICS_README.md` and the raw-source requirement: use only the local ChatGPT export (`02-14-26-ChatGPT/conversations.json`) for active forensics truth.

## Scope Read During Audit

- `docs/docs-tree.md`
- `reports/reports-tree.md`
- `README.md`
- `TOKEN_FORENSICS_README.md`
- `PROJECT_MOONSHINE_UPDATE_1.md`
- `PROJECT_MOONSHINE_UPDATE_1.json`
- `PROJECT_DATABASE_DOCUMENTATION.md`
- `CONTEXT.md`
- `MEMORY.md`
- `STATE_SNAPSHOT_20260217.md`
- `docs/STATE_SNAPSHOT_20260217.md`
- `docs/MOONSHINE_DISTRIBUTED_PLAN_20260217.md`
- `docs/HANDOFF_SCHEMA.json`
- `docs/HANDOFF_PACKET_20260217_*.json`
- `project-directory.md`
- `reports/token_forensics.json`
- `reports/token_forensics.md`
- `reports/token_ledger.json`
- `reports/validation_manifest.json`
- `reports/validation_report.md`
- `reports/parquet_forensics.json`
- `reports/repro_manifest.json`
- `token-count-guess.md`
- `scripts/generate_sample_data.py`
- `scripts/generate_real_export_forensics.py`
- `scripts/run_validation.py`
- `scripts/reconcile_tokens_and_extract_june.py`
- `token_forensics_agents.py`
- `token_forensics_orchestrator.py`
- `dataset_forensics/config.single_source.yaml`
- `dataset_forensics/cli.py`

## Confirmed State

### Canonical Raw-Export Artifacts (source-locked)

- `reports/token_ledger.json`
  - `source_path`: `02-14-26-ChatGPT/conversations.json`
  - `source_sha256`: `4e6d44cd2102d26701858f832ba325e15d3490e61397013b7896a22dfad792dd`
  - `content_tokens_non_system`: `115,330,530`
  - `content_tokens_cleaned`: `104,321,772`
  - `distilled_tokens_selected`: `104,321,772`
- `reports/token_forensics.json`
  - `report_version`: `2.1.0-real-export-streaming`
  - `total_conversations`: `1,439`
  - `total_messages`: `169,397`
  - `total_tokens`: `115,330,530`
  - `distilled_tokens`: `104,321,772`

### Synthetic/Template-Derived Artifacts (non-canonical)

- `reports/validation_manifest.json`
- `reports/validation_report.md`
- `reports/parquet_forensics.json`
- `reports/parquet_forensics.md`
- `reports/token_forensics.md`
- `reports/repro_manifest.json`

These contain the synthetic signature:

- `sources`: `documentation`, `stackoverflow`, `web_crawl`, `github_code`
- `split_distribution`: `train=9039`, `val=487`, `test=474`
- `total_samples`: `10,000`
- tokenizers labeled `gpt2`, `llama-2`, `code-llama`, `mistral`

## Code Path Mapping (Where Drift Comes From)

### Synthetic path

- `scripts/generate_sample_data.py`
  - Hardcodes synthetic sources and tokenizers
  - Defaults output to `reports/token_row_metrics.parquet`
  - This is the direct source of the 10k / 9039-487-474 profile

### Validator behavior

- `scripts/run_validation.py`
  - Reads `reports/token_row_metrics.parquet` and `reports/tokenizer_benchmark.csv`
  - Builds `validation_manifest.json` and `validation_report.md` from those files
  - Builds `parquet_forensics.json` and `parquet_forensics.md` from those same files
  - Separately builds `token_forensics.json` from `token_ledger.json + moonshine_corpus.db` via `_generate_real_token_forensics`

Result: one run can emit both raw-truth and synthetic-truth artifacts in the same folder.

### Raw-truth generation path

- `scripts/generate_real_export_forensics.py`
  - Streams `02-14-26-ChatGPT/conversations.json` with `ijson`
  - Uses `o200k_base`
  - Produces raw-based `reports/token_forensics.json` / `reports/token_forensics.md` (when invoked)

## Why Counts Diverge

Not data loss. Multiple measurement scopes coexist:

- Full raw JSON serialized text: ~`458,527,657` (`token-count-guess.md`)
- Message-content non-system canonical: `115,330,530` (`token_ledger.json` / `token_forensics.json`)
- Distilled selected content: `104,321,772` (`token_ledger.json`)
- Synthetic template parquet sample: `9,405,521` over 10k rows (`validation_manifest.json`)
- Heuristic Moonshine baseline: `51,524,462` (`PROJECT_MOONSHINE_UPDATE_1.json`)

## Gap vs TOKEN_FORENSICS_README.md

- `TOKEN_FORENSICS_README.md` still includes external ecosystem text (GitHub links, StackOverflow/web crawl examples, etc.) that conflicts with current run policy.
- Active operator truth is now local raw export only.

## Reconciliation Path (Concrete)

### Phase 1 - Policy hardening (no behavior break)

1. Mark `validation_manifest.json` and `parquet_forensics.*` as "parquet profile scope" not canonical corpus truth.
2. Mark canonical token truth source as `reports/token_ledger.json` and `reports/token_forensics.json` only.
3. Add source-scope warning in `README.md`, `CONTEXT.md`, and `MEMORY.md`.

### Phase 2 - Artifact contamination prevention

1. Change sample generator default output away from `reports/` (for example `reports/synthetic/`).
2. Add explicit guardrails before writing into canonical `reports/` paths.
3. Add validator check to flag synthetic source labels when strict raw-only mode is expected.

### Phase 3 - Single canonical profile output

1. Generate real parquet metrics from raw ChatGPT export only.
2. Regenerate `validation_manifest.json` and `validation_report.md` from real parquet metrics.
3. Regenerate `token_reconciliation.md` using updated validation inputs.
4. Keep scope tags visible in every report to prevent cross-scope arithmetic.

## Immediate Operational Rule (until Phase 3 is complete)

Treat only these as canonical for token totals:

- `reports/token_ledger.json`
- `reports/token_forensics.json`

Treat these as non-canonical profile artifacts:

- `reports/validation_manifest.json`
- `reports/validation_report.md`
- `reports/parquet_forensics.json`
- `reports/parquet_forensics.md`
- `reports/token_forensics.md`
- `reports/repro_manifest.json` (current one is tied to 10k synthetic parquet profile)
