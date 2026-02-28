# Publish Surface - 2026-02-28

## Goal

Prepare the second teaser/public push after Moonshine closeout validation without rebuilding the corpus and without exposing raw exports.

## Live Authority Verified

Primary authority artifacts:

- `reports/main/moonshine_mash_active.db`
- `reports/main/token_recount.main.postdeps.json`
- `reports/main/final_db_pass_20260228.json`
- `reports/main/final_db_pass_20260228.md`
- `reports/main/reports_authority_manifest.json`

Current live main state:

- conversations: `2788`
- messages: `179974`
- distilled_conversations: `2486`
- exact non-system tokens: `122627092`
- exact distilled non-system tokens: `110539045`

Provider composition:

- conversations: `chatgpt=1439`, `claude=954`, `deepseek=320`, `qwen=75`
- messages: `chatgpt=169397`, `claude=7726`, `deepseek=2073`, `qwen=778`
- distilled_conversations: `chatgpt=1326`, `claude=789`, `deepseek=304`, `qwen=67`

## Publish Shape

### Include

Root / docs:
- `README.md`
- `WIKI.md`
- `PROJECT_MOONSHINE_UPDATE_1.md`
- `PROJECT_DATABASE_DOCUMENTATION.md`
- `docs/`
- `distill-the-flow-filetree.md`
- `docs/PUBLISH_SURFACE_20260228.md`

Main authority lane:
- `reports/main/final_db_pass_20260228.json`
- `reports/main/final_db_pass_20260228.md`
- `reports/main/reports_authority_manifest.json`
- `reports/main/token_recount.main.postdeps.json`
- `reports/main/moonshine_mash_active.db` via Git LFS

Canonical parquet lane:
- `reports/canonical/parquet_forensics.raw.json`
- `reports/canonical/parquet_forensics.raw.md`
- `reports/canonical/token_row_metrics.raw.parquet`

Provider metadata lane:
- repaired provider token ledgers/manifests/input manifests under `reports/providers/`
- provider-local markdown reports where useful for credibility
- do not require provider-local DB binaries for this push

Visual surface:
- `visualizations/`
- `visualizations/providers/claude/`
- `visual_intelligence/`

Inventory / operator aids:
- `reports/CURRENT_REPORTS_FILETREE.md`

### Exclude

- raw exports under `exports/`
- archives under `archive/`
- synthetic quarantine as headline surface unless intentionally referenced
- oversized legacy zips not needed for the story (`merge_manifest.main.zip`, `reports.zip`, `requirements.zip`)
- duplicate provider-local generic DB copies unless a later release explicitly needs them

## GitHub Transport Constraint

- `reports/main/moonshine_mash_active.db` is ~805 MB and must not be pushed as a normal git blob.
- `.gitattributes` now tracks this DB through Git LFS.
- Large nonessential zip artifacts should remain out of this push.

## Recommended Commit Story

`docs: publish validated moonshine authority pack`

Optional follow-up commit if needed:

`data: publish main moonshine db via lfs`

## Final Push Rule

Use explicit staging paths instead of broad `git add .` so teaser/public scope stays intentional.
