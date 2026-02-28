# Distill the Flow Wiki

<div align="center">
  <img src="visuals/logo.png" alt="Operation Moonshine Distill the Flow" width="320" />
</div>

---

## Project Position

Distill the Flow is the OpsOTA Drop 3 corpus forensics and Moonshine mash pipeline. It transforms raw multi-provider conversational exports into provenance-locked, schema-governed, quality-gated SQLite training substrates with reproducible token accounting and visual intelligence outputs.

The project is part of **Project Decentralize SOTA** alongside:

- [Drop 1: Reinforcement-Learning-Full-Pipeline](https://github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline)
- [Drop 2: SOTA-Runtime-Core](https://github.com/calisweetleaf/SOTA-Runtime-Core)
- [Drop 3: distill-the-flow](https://github.com/calisweetleaf/distill-the-flow) (this repository)

---

## Live Main-Lane State (2026-02-28)

This is the authoritative current-state summary. For machine-readable source of truth, read the authority stack listed under **Operator Read Order** below.

### Provider Composition

| Table | chatgpt | claude | deepseek | qwen | Total |
|-------|---------|--------|----------|------|-------|
| `conversations` | 1,439 | 954 | 320 | 75 | **2,788** |
| `messages` | 169,397 | 7,726 | 2,073 | 778 | **179,974** |
| `distilled_conversations` | 1,326 | 789 | 304 | 67 | **2,486** |

### Exact Token Recount (o200k\_base, non-system)

| Provider | Tokens |
|----------|--------|
| chatgpt | 115,334,978 |
| claude | 4,791,566 |
| deepseek | 1,482,829 |
| qwen | 1,017,719 |
| **main-lane total** | **122,627,092** |
| **distilled total** | **110,539,045** |

### Integrity Status

- `record_uid` collisions: `0` across all core tables
- SQLite integrity: `PRAGMA quick_check = ok`
- Authority DB: `reports/main/moonshine_mash_active.db`

---

## Merge History

| Event | Date | Provider | Conversations Added | Messages Added |
|-------|------|----------|--------------------:|---------------:|
| Bootstrap (ChatGPT-only) | 2026-02-17 | chatgpt | 1,439 | 169,397 |
| Claude Wave 1 merge | 2026-02-26 | claude | +757 | +5,589 |
| Qwen promotion | 2026-02-27 | qwen | +75 | +778 |
| DeepSeek promotion | 2026-02-27 | deepseek | +320 | +2,073 |
| Claude gap-fill (late run) | 2026-02-28 | claude | +197 | +2,137 |

Each merge was preceded by a pre-merge archive snapshot. Rollback anchors exist for every layer.

### Claude Provider Runs in Main

- `claude_20260226_065717`: 757 conversations, 5,589 messages, 652 distilled
- `claude_20260227_080825_20260226`: 197 conversations, 2,137 messages, 137 distilled (gap-fill)

---

## Canonical Architecture

```
RAW EXPORTS (immutable)
  └── per-provider ingestion
        ├── reports/providers/<provider>/<run_id>/
        │     ├── moonshine_<provider>_<run_id>.db
        │     └── token_ledger.<provider>.<run_id>.json
        │
        └── archive/<provider>/<run_id>/  (pre-merge snapshot)

MAIN MASH (rolling authority)
  └── reports/main/moonshine_mash_active.db
        ├── conversations
        ├── messages
        └── distilled_conversations

CANONICAL FORENSICS LANES
  ├── reports/canonical/        (raw-only token forensics)
  ├── reports/legacy_synthetic/ (quarantined synthetic only)
  └── reports/main/             (integration and serving authority)
```

**Execution invariants:**

1. Raw provider exports are immutable source inputs — never mutated.
2. Every provider run gets its own scoped DB under `reports/providers/`.
3. Main mash DB is archived before every merge.
4. One provider at a time, dry-run validated before live merge.
5. Idempotent upsert keys: `provider`, `provider_run_id`, `conversation_id`, `message_id`.
6. `record_uid` uniqueness is a hard merge invariant — zero collisions required.
7. Token reconciliation report is generated after every merge cycle.
8. Documentation is synced in the same pass as the merge.

---

## Schema Authority

Schema and metadata compliance are governed by:

- `PROJECT_DATABASE_DOCUMENTATION.md`
- `docs/Moonshine-Technical-Implementation.md`

**Production table contract:**

| Table | Description |
|-------|-------------|
| `conversations` | One row per conversation with 28 quality and behavioral metrics |
| `messages` | Flat message list with role, text, timestamps, and char/word counts |
| `distilled_conversations` | Distillation-selected subset with quality filtering applied |

**Required provenance envelope (per row):**

- `provider`
- `provider_run_id`
- `source_file_sha256`
- `source_path`
- `ingested_at`
- `record_uid`

`record_uid` is a deterministic merge key. Its uniqueness across all core tables is verified on every merge pass.

---

## Token Scope Contract

Token figures are scope-specific. Do not compare across scopes without explicit tagging.

| Scope Tag | Source | Current Value |
|-----------|--------|---------------|
| `heuristic_analyzer` | `reports/moonshine_corpus_report.md` | 51,524,462 |
| `canonical_source_locked` (ChatGPT only) | `reports/token_ledger.json` | 115,330,530 |
| `distilled_selected` (ChatGPT only) | `reports/token_ledger.json` | 104,321,772 |
| `parquet_row_aggregate` | `reports/canonical/parquet_forensics.raw.json` | 231,608,618 |
| `exact_main_non_system` (all providers) | `reports/main/token_recount.main.postdeps.json` | 122,627,092 |
| `exact_distilled_non_system` | `reports/main/token_recount.main.postdeps.json` | 110,539,045 |

For merged main-lane exact token truth, use:

- `reports/main/token_recount.main.postdeps.json`

The older `reports/main/token_recount.main.json` pointer is superseded by the live authority stack as of 2026-02-28.

---

## Parquet vs SQLite Role Split

Parquet artifacts are kept for forensics, dedup auditing, and stream workflows. They are not the integration contract.

| Artifact Family | Role |
|-----------------|------|
| `token_row_metrics*.parquet` | Per-row token accounting; quality/reproducibility |
| `dedup_clusters.parquet` | Row/cluster mapping for dedup signals |
| `reports/canonical/token_row_metrics.raw.parquet` | Canonical raw-only lane; 169,397 rows |
| `reports/main/moonshine_mash_active.db` | Integration authority; distillation source of truth |

SQLite is the canonical mash contract. Distillation export paths (DPO, GRPO, agentic code, conversational) all originate from the main SQLite DB.

---

## Raw-Only Enforcement

The canonical lane is locked to raw export forensics only.

- `scripts/generate_sample_data.py` requires `--allow-synthetic` to generate synthetic rows.
- Synthetic artifacts are quarantined to `reports/legacy_synthetic/` and tagged `dataset_origin="synthetic"`.
- `scripts/run_validation.py --profile raw_only` (default) enforces this boundary.
- Raw-only contamination gates R1-R5 are active and emit `reports/raw_only_gate_manifest.json`.

**Canonical enforcement command:**

```powershell
.\.venv\Scripts\python.exe run_token_forensics.py 02-14-26-ChatGPT/conversations.json --output-dir reports
.\.venv\Scripts\python.exe scripts/run_validation.py --reports-dir reports --strict --profile raw_only
```

---

## Distillation Lane Strategy

Distillation targets are provider-signal-aware:

| Provider | Primary Strength | Distillation Priority |
|----------|-----------------|----------------------|
| chatgpt | Long-form analysis, structured architecture dialog, broad coverage | Core SFT + DPO substrate |
| claude | Ideation-heavy, trace-heavy, tool-using interactions | DPO / GRPO signal lane |
| deepseek | Technical depth | Supplementary SFT |
| qwen | Supplementary conversational coverage | Supplementary |

Distillation lanes under `reports/distillations/`:

- `dpo/` — Direct Preference Optimization pairs
- `grpo/` — Group Relative Policy Optimization sequences
- `agentic_code/` — Tool-using and multi-step code execution trajectories
- `conversational/` — General dialog quality training

---

## Corpus Quality Baseline (ChatGPT Phase 1)

Established during Phase 1 analysis of 1,439 ChatGPT conversations (169,397 messages):

- Mean information gain: `0.458` (Jaccard-based)
- Mean malicious compliance (sycophancy): `0.073` — low, good signal
- Mean user entropy: `0.612` — high vocabulary diversity
- Code block density: `60.5 per conversation` — strongly technical corpus
- High-signal conversations (info gain > 0.58, compliance < 0.25): `124` (8.6%)
- Correction events (DPO candidates): `4,413` across `594` conversations (41.3%)

Topic distribution:

| Topic | Count | Pct |
|-------|-------|-----|
| Architecture | 676 | 47.0% |
| Debugging | 559 | 38.8% |
| Data Processing | 97 | 6.7% |
| Code Review | 37 | 2.6% |
| General | 27 | 1.9% |
| Deployment | 17 | 1.2% |
| RCF Theory | 11 | 0.8% |
| RLHF Implementation | 10 | 0.7% |

Full analysis findings: `docs/Moonshine-Analysis-Findings.md`

---

## Seven-Agent Forensics Pipeline

The token forensics pipeline runs 7 specialized agents in dependency-aware execution:

**Phase 1 — Parallel (6 agents):**

| Agent | Function |
|-------|----------|
| DataProfilerAgent | Schema mapping, language detection, split analysis |
| MultiTokenizerAgent | Cross-tokenizer analysis (GPT-4, Llama, fallback), context-fit modeling, drift detection |
| QualityScoringAgent | Readability, Shannon entropy, trigram repetition, boilerplate detection |
| SafetyPIIAgent | Email, phone, SSN, credit card, IP, API key detection with risk scoring |
| DedupAgent | SHA256 exact dedup, MinHash near-dup, semantic clustering |
| CostModelAgent | Per-model-family training and inference cost projections |

**Phase 2 — Serial (1 agent, requires all Phase 1 outputs):**

| Agent | Function |
|-------|----------|
| VerifierAgent | Artifact completeness, schema integrity, count consistency, reproducibility hash, quality gate enforcement |

Quality gates: `mean_quality_score >= 0.5` and `low_quality_rate < 0.3`. Gate failures emit `failure_manifest.json` and return non-zero exit.

---

## Provider-Local Ledger Repair (2026-02-28)

Provider-local Claude, Qwen, and DeepSeek ledgers and distillation manifests were repaired on 2026-02-28.

They no longer inherit the ChatGPT `115,330,530` baseline. Each provider ledger now carries exact provider-local `content_tokens_source` and `content_tokens_non_system` values derived from `exact_message_recount`.

This is required for correct per-provider token attribution in mixed-corpus environments.

---

## Merge-Manifest Caveat

`reports/main/merge_manifest.main.json` currently reflects a skip-only rerun for the late Claude run. This does not mean the late Claude layer is absent. The authoritative interpretation is encoded in:

- `reports/main/final_db_pass_20260228.json`
- `reports/main/final_db_pass_20260228.md`

Those artifacts reconcile the skip-only rerun with the verified live DB state and exact recount totals.

---

## Public Teaser Surface (2026-02-27)

This repository is in a public teaser phase focused on architecture transparency and current-state visibility.

**Included in teaser scope:**

- `README.md`
- `WIKI.md`
- `docs/`
- `visualizations/`
- `file-trees/`
- `PROJECT_DATABASE_DOCUMENTATION.md`
- `PROJECT_MOONSHINE_UPDATE_1.md`

**Excluded from teaser scope:**

- Raw exports under `exports/`
- Archives under `archive/`
- Synthetic quarantine artifacts as headline surface
- Private operational overlays

**Public-facing read path:**

1. `README.md`
2. `WIKI.md`
3. `PROJECT_DATABASE_DOCUMENTATION.md`
4. `docs/Moonshine-Documentation-Index.md`
5. `docs/Moonshine-Technical-Implementation.md`

---

## Operator Read Order

For live DB operations, schema decisions, or release-facing work, read in this order:

### Authority Stack (Machine Truth)

1. `reports/main/reports_authority_manifest.json`
2. `reports/main/final_db_pass_20260228.json`
3. `reports/main/final_db_pass_20260228.md`
4. `reports/main/token_recount.main.postdeps.json`
5. `reports/main/moonshine_mash_active.db`
6. `PROJECT_DATABASE_DOCUMENTATION.md`

### Full Operational Read Order

1. `README.md`
2. `WIKI.md`
3. `PROJECT_MOONSHINE_UPDATE_1.md`
4. `docs/PUBLISH_SURFACE_20260228.md`
5. `PROJECT_DATABASE_DOCUMENTATION.md`
6. `docs/MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md`
7. `docs/Moonshine-Project-Overview.md`
8. `docs/Moonshine-Technical-Implementation.md`
9. `docs/Moonshine-Analysis-Findings.md`
10. `reports/main/reports_authority_manifest.json`
11. `reports/main/final_db_pass_20260228.md`
12. `reports/main/token_recount.main.postdeps.json`

---

## Runtime Contract

All execution is `.venv`-first:

```powershell
.\.venv\Scripts\python.exe <script>.py ...
```

No non-venv Python runs for Moonshine or token-forensics operations.

---

## Mission Rhythm

- Archive before every merge
- Merge one provider at a time
- Validate schema, provenance, and idempotence
- Recount tokens with exact o200k_base accounting
- Repair provider-local ledgers after each promotion
- Sync documentation in the same pass as the merge

This keeps the core provenance discipline intact while making the system externally legible and auditable.

---

## Documentation Sync Contract

After every live merge, update these in the same pass to prevent drift:

1. `README.md`
2. `AGENTS.md`
3. `PROJECT_MOONSHINE_UPDATE_1.md`
4. `WIKI.md`
5. `reports/CURRENT_REPORTS_FILETREE.md`
6. `CONTEXT.md` and `MEMORY.md`

---

**Built with precision. Released with intent.**

*No MVP. Only SOTA++.*
