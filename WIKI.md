# Distill the Flow Wiki

## Project Position

Distill the Flow is a production-oriented corpus forensics and Moonshine mash pipeline focused on:

- provenance-locked corpus processing,
- strict schema governance,
- quality-gated distillation,
- reproducible report and visualization outputs.

Current main-lane baseline is `chatgpt + claude` with authority database:

- `reports/main/moonshine_mash_active.db`

## Canonical Architecture

1. Raw provider exports are treated as immutable source inputs.
2. Provider-local analysis produces run-scoped artifacts under `reports/providers/<provider>/<run_id>/`.
3. One-provider-at-a-time merge streams into `reports/main/moonshine_mash_active.db`.
4. Pre-merge archive snapshots are mandatory.
5. Reports and visuals are generated from canonical lanes.

## Schema Authority

Schema and metadata compliance are governed by:

- `PROJECT_DATABASE_DOCUMENTATION.md`
- `docs/Moonshine-Technical-Implementation.md` (with 2026-02-27 schema authority addendum)

Production table contract:

1. `conversations`
2. `messages`
3. `distilled_conversations`

Required provenance envelope fields:

- `provider`
- `provider_run_id`
- `source_file_sha256`
- `source_path`
- `ingested_at`
- `record_uid`

`record_uid` uniqueness is a hard merge invariant.

## Token Scope Contract

Token figures must always be scope-tagged:

- heuristic/analyzer totals,
- canonical exact recount totals,
- distilled selected totals.

For merged main-lane exact token truth, use:

- `reports/main/token_recount.main.json`

## Public Teaser Surface (Recommended)

Public teaser should focus on method, governance, and observability artifacts:

- root docs (`README.md`, `WIKI.md`, `distill-the-flow-filetree.md`)
- `docs/`
- `reports/` (curated)
- `visualizations/`
- `visual_intelligence/`

Exclude raw exports, archives, and private workflow overlays.

## Operator Read Order

1. `README.md`
2. `WIKI.md`
3. `docs/Moonshine-Documentation-Index.md`
4. `PROJECT_DATABASE_DOCUMENTATION.md`
5. `docs/MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md`
6. `reports/main/merge_manifest.main.json`
7. `reports/main/token_recount.main.json`

## Mission Rhythm

- Archive first
- Merge one provider at a time
- Validate schema and provenance
- Recount tokens
- Sync docs in same pass

This keeps the core complexity intact while making the system externally legible.
