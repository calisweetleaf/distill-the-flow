# OpenCode Stage B1 Kickoff Prompt (2026-02-26)

You are `builder_executor` for **Project Moonshine**.
Your source-of-truth contract is:
- `docs/handoffs/20260226_attack_plan/HANDOFF_PACKET_20260226_STAGE_P1_CONTRACT.json`

You must execute B1 implementation and return a schema-valid handoff packet.

## Role + Objective
Implement provider-safe schema and merge readiness for the Moonshine mash pipeline so next provider onboarding (Claude first) can proceed with high confidence.

## Required Read Order
1. `AGENTS.md`
2. `CONTEXT.md`
3. `MEMORY.md`
4. `docs/HANDOFF_SCHEMA.json`
5. `docs/handoffs/20260226_attack_plan/HANDOFF_PACKET_20260226_STAGE_P0_PLAN.json`
6. `docs/handoffs/20260226_attack_plan/HANDOFF_PACKET_20260226_STAGE_P1_CONTRACT.json`
7. `docs/MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md`

## Runtime Rule
Use `.venv` for command execution.
- Python commands: `.\\venv\\Scripts\\python.exe ...`

## Implementation Scope (B1)
1. Enforce P1 schema requirements for:
- `conversations`
- `messages`
- `distilled_conversations`

2. Required fields:
- `provider`
- `provider_run_id`
- `source_file_sha256`
- `source_path`
- `ingested_at`
- `record_uid`

3. Additional required field on `messages`:
- `conversation_record_uid`

4. Deterministic key rules:
- conversations: `provider:provider_run_id:conversation_id`
- messages: `provider:provider_run_id:conversation_id:message_id`
- distilled_conversations: `provider:provider_run_id:conversation_id`

5. Add/verify indexes needed for provider-run queries and uniqueness checks.

6. Build migration path for existing main DB:
- target DB: `reports/main/moonshine_mash_active.db`
- preserve data
- backfill required fields deterministically

7. Build merge readiness plumbing:
- snapshot-before-merge behavior: `archive/main/<run_id>/moonshine_mash_premerge.db`
- merge manifest with non-null counters: `inserted`, `updated`, `skipped`
- idempotence check path (same merge twice => zero net new rows on second run)

8. Keep provider-local outputs under:
- `reports/providers/<provider>/<run_id>/`

9. Do not regress existing Moonshine analytics fidelity or break current reports/main lane.

## Evidence Required in B1 Output
You must provide command-level evidence summary for:
1. Schema parity checks (`PRAGMA table_info(...)`) for all three tables.
2. Provider coverage checks (100% rows populated on required fields).
3. Record UID uniqueness checks.
4. Snapshot existence before merge.
5. Merge counter values non-null.
6. Idempotence re-run results.
7. Token reconciliation artifact generation.

## Required Artifacts from B1
1. `docs/handoffs/20260226_attack_plan/HANDOFF_PACKET_20260226_STAGE_B1_OPENCODE.json`
2. `docs/handoffs/20260226_attack_plan/B1_EXECUTION_REPORT_20260226.md`
3. `docs/handoffs/20260226_attack_plan/B1_EXECUTION_LOG_20260226.json`

## Handoff Packet Rules
Your B1 packet must conform to:
- `docs/HANDOFF_SCHEMA.json`

Mandatory fields:
- `run_id`
- `stage_id`
- `role` = `builder_executor`
- `state_hash_in`
- `state_hash_out`
- `files_changed`
- `verification` (checks + status + evidence)
- `result`
- `next_actions`
- `signal_type`

Allowed signal types:
- `forward`
- `revision_request`
- `escalation`

If blocked by contract contradiction or missing prerequisite, emit `revision_request` with exact blocker.

## Quality Bar
No placeholders. No TODO stubs. No partial migrations.
Deliver production-grade implementation with deterministic behavior and reproducible evidence.

## Completion Condition
B1 is complete only when:
- all P1 schema/gate requirements are implemented,
- evidence is written,
- B1 packet exists and validates structurally,
- signal type is set appropriately.
