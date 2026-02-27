# Moonshine Attack Plan - 2026-02-26

## Mission
Prepare the project for next-provider onboarding with a planner-first distributed workflow:
1. Lock mission and scope.
2. Define schema and gate contract.
3. Hand major implementation to OpenCode.
4. Run Codex adjudication.
5. Finish with Claude (Sonnet/Opus + subagents).

## Directory Contract
All stage artifacts for this run live under:
- `docs/handoffs/20260226_attack_plan/`

## Stage Map

### P0 - Mission Lock (planner_reasoner, Codex)
- Objective:
  - Freeze tonight scope and non-goals.
  - Freeze ownership model (OpenCode builder, Claude finisher).
  - Lock handoff protocol and evidence expectations.
- Output:
  - `HANDOFF_PACKET_20260226_STAGE_P0_PLAN.json`

### P1 - Schema + Gate Contract (planner_reasoner, Codex)
- Objective:
  - Define exact required schema fields for provider-safe mash merge.
  - Define gate checks and evidence for merge readiness.
  - Define execution tasks for OpenCode and acceptance checks for Claude finisher.
- Output:
  - `HANDOFF_PACKET_20260226_STAGE_P1_CONTRACT.json`

### B1 - Implementation Wave (builder_executor, OpenCode)
- Objective:
  - Execute P1 contract with full file-level implementation.
  - Produce command evidence and verification summary.
- Required output:
  - `HANDOFF_PACKET_20260226_STAGE_B1_OPENCODE.json`

### C1 - Drift Adjudication (planner_reasoner, Codex)
- Objective:
  - Contract-vs-implementation audit.
  - Emit `forward` or `revision_request`.
- Required output:
  - `HANDOFF_PACKET_20260226_STAGE_C1_ADJUDICATION.json`

### F1 - Finisher Wave (finisher_polisher, Claude)
- Objective:
  - Regression safety, docs harmonization, release-quality packaging.
  - Resolve or formally accept residual caveats with explicit debt language.
- Required output:
  - `HANDOFF_PACKET_20260226_STAGE_F1_CLAUDE_FINISHER.json`

### C2 - Final Signal (planner_reasoner, Codex)
- Objective:
  - Final acceptance and next dependency publication.
- Required output:
  - `HANDOFF_PACKET_20260226_STAGE_C2_FINAL_SIGNAL.json`

## Core Gate Criteria
1. Provider schema fields exist and are populated on merge-path tables.
2. Deterministic `record_uid` key contract enforced.
3. Provider-local run artifacts stay isolated under provider directory.
4. `reports/main` snapshot-before-merge policy enforced and manifested.
5. Merge idempotence evidence present.
6. Token reconciliation evidence present.

## Collaboration Discipline
- Every stage must write a schema-valid handoff packet.
- Every major stage must close with exo reflection and memory store.
- No stage may skip signal typing (`forward`, `revision_request`, `escalation`).
