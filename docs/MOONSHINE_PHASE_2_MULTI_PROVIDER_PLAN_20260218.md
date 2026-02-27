# MOONSHINE PHASE 2 PLAN â€” MULTI-PROVIDER MASH LAYER

**Date:** 2026-02-18  
**Status:** Ready to execute  
**Primary Goal:** Onboard additional provider exports (starting with Claude) into a controlled archive + rolling mash workflow.

---

## 1. Current Truth Snapshot

| Item | Current State |
|------|---------------|
| Canonical source lock | `02-14-26-ChatGPT/conversations.json` |
| Canonical token ledger | `reports/token_ledger.json` |
| Canonical DB | `reports/moonshine_corpus.db` |
| Expansion DB | `reports/expansion_20260218/moonshine_corpus.db` (isolated, exploratory) |

**Answer to operator question:** yes, the current canonical production corpus is ChatGPT-only.

---

## 2. Phase 2 Operating Contract

1. Every provider gets its own **per-run DB**.
2. The rolling mash DB is versioned by **pre-merge archive snapshot** every run.
3. No synthetic sources in canonical lanes.
4. Every run is source-hash locked and manifested.
5. Distillation outputs are generated from mash DB, not directly from raw export files.

---

## 3. Required Directory Layout (New Standard)

```text
archive/
  chatgpt/
    <run_id>/
  claude/
    <run_id>/
  main/
    <run_id>/

reports/
  providers/
    chatgpt/<run_id>/
      moonshine_chatgpt_<run_id>.db
      token_ledger.chatgpt.<run_id>.json
    claude/<run_id>/
      moonshine_claude_<run_id>.db
      token_ledger.claude.<run_id>.json
  main/
    moonshine_mash_active.db
    token_ledger.main.json
    merge_manifest.main.json
  distillations/
    dpo/<run_id>/
    grpo/<run_id>/
    agentic_code/<run_id>/
    conversational/<run_id>/
```

---

## 4. Run Lifecycle (Per New Provider Export)

### Stage A â€” Preflight

- Verify export file checksum.
- Assign `run_id` (`provider_YYYYMMDD_HHMMSS`).
- Validate provider format adapter readiness.

### Stage B â€” Provider Ingestion

- Ingest raw export into provider-run DB only.
- Emit provider token ledger + provenance manifest.
- Run raw-only validation gates.

### Stage C â€” Archive Snapshot

- Copy provider run artifacts to `archive/<provider>/<run_id>/`.
- Snapshot current mash DB to `archive/main/<run_id>/moonshine_mash_premerge.db`.

### Stage D â€” Merge to Mash

- Merge provider-run DB into `reports/main/moonshine_mash_active.db`.
- Enforce idempotent upsert keys (`provider`, `provider_run_id`, `conversation_id`, `message_id`).
- Emit `merge_manifest.main.json` with inserted/updated/skipped counts.

### Stage E â€” Distill

- Generate distillation lanes from mash DB:
  - DPO lane
  - GRPO lane
  - Agentic code lane
  - Conversational lane

### Stage F â€” Documentation Sync

- Update `TOKEN_FORENSICS_README.md`.
- Append status block in `PROJECT_MOONSHINE_UPDATE_1.md`.
- Update `docs/Moonshine-Project-Overview.md` lane counts.

---

## 5. Schema Additions Required for Multi-Provider Merge

Add or enforce these columns on conversation/message records before merge:

- `provider` (`chatgpt`, `claude`, ...)
- `provider_run_id`
- `source_file_sha256`
- `source_path`
- `ingested_at`
- `record_uid` (deterministic merge key)

Without these fields, cross-provider dedup and rollback become unsafe.

---

## 6. Distillation Targets by Provider Signal

### ChatGPT (already loaded)

- Strength lane: conversational quality + broad architecture dialog.
- Use heavily for conversational and generalist planning data.

### Claude (next)

- Target lane: DPO/GRPO/agentic code trajectories.
- Prioritize correction-heavy, tool-using, debugging-heavy interactions.

### Output Strategy

- Keep provider provenance in every distillation row.
- Build mixed sets only after provider-labeled quality balancing.

---

## 7. Gates Before Accepting a Provider Run

1. Source hash present and reproducible.
2. Provider tag coverage = 100% for imported rows.
3. Merge idempotence check passes (re-run merge yields zero net new rows).
4. Token reconciliation report generated (`provider ledger` + `main ledger`).
5. No canonical contamination signatures (`documentation/stackoverflow/web_crawl/github_code`).

---

## 8. Immediate Execution Plan â€” Claude Wave 1

1. Create archive scaffold (`archive/chatgpt`, `archive/claude`, `archive/main`).
2. Move/copy current ChatGPT run package into `archive/chatgpt/<run_id>/`.
3. Snapshot current mash baseline into `archive/main/<run_id>/`.
4. Ingest Claude export into `reports/providers/claude/<run_id>/`.
5. Validate and archive Claude run.
6. Merge Claude provider DB into `reports/main/moonshine_mash_active.db`.
7. Generate first mixed-provider distillation pack (DPO/GRPO/agentic code).

---

## 9. Decision Log

- The project now transitions from single-provider corpus analytics to a multi-provider mash operating model.
- Archive-first discipline is mandatory before each merge.
- Main mash DB becomes the distillation source of truth; provider DBs remain immutable run artifacts.

---

**Execution signal:** `forward`  
**Owner for implementation pass:** Claude + subagents  
**Planning anchor:** this document + `docs/Moonshine-Documentation-Index.md`

---

## Phase 2 Execution Update (2026-02-19)

### Completed

- `reports/main/` authority lane created.
- `reports/main/moonshine_mash_active.db` bootstrapped from `reports/expansion_20260218/moonshine_corpus.db`.
- `reports/main/token_ledger.main.json` created from expansion token ledger.
- `reports/main/merge_manifest.main.json` and `reports/main/reports_authority_manifest.json` emitted.
- Legacy root moonshine artifacts archived to `archive/chatgpt/chatgpt_20260217_r1/`.

### Remaining Before First New Provider Merge

1. Add provider columns + deterministic `record_uid` to merge schema.
2. Implement merge-upsert pipeline producing real inserted/updated/skipped counts.
3. Onboard first non-ChatGPT provider run into `reports/providers/<provider>/<run_id>/`.


## Phase 2 Execution Update (2026-02-27)

### Current Main-Lane Reality (Post Claude Merge)

The Phase 2 contract has materially progressed from planning into live execution.

- Active mash DB: `reports/main/moonshine_mash_active.db`
- Current provider composition in main:
  - `conversations`: chatgpt `1439`, claude `757`
  - `messages`: chatgpt `169397`, claude `5589`
  - `distilled_conversations`: chatgpt `1326`, claude `652`
- Exact non-system token recount (`o200k_base`):
  - chatgpt `115,334,978`
  - claude `3,008,283`
  - combined `118,343,261`

Interpretation: ChatGPT remains the core substrate; Claude is additive and signal-shaping rather than volume-dominant.

### Provider Staging Status

- `reports/providers/qwen/qwen_20260226_063147/` exists and is merge-ready.
- `reports/providers/deepseek/deepseek_20260226_063139/` exists and is merge-ready.
- Both remain provider-local and have not yet been promoted into main.

### Merge Order Lock (Operator-Requested)

When main-lane expansion resumes, enforce this exact sequence:

1. `qwen`
2. `deepseek`

Maintain archive-before-merge and one-provider-per-operation invariants.

### DPO/GRPO Positioning Clarification

Given the current mixed corpus behavior:

- ChatGPT lane is long-form analysis and structured source-grounded work (core SFT + DPO substrate).
- Claude lane is ideation/trace-heavy extension and should be treated as a targeted DPO/GRPO signal lane.

### Next Practical Step

Before adding new providers to main, execute docs-first stabilization so all operators read from the same authority chain (main DB, recount tokens, merge manifest, and provider-local staging status).
