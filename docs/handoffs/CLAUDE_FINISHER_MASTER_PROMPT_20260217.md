# CLAUDE FINISHER MASTER PROMPT â€” MOONSHINE/MASH STAGE D (2026-02-17)

You are the final finisher for a distributed multi-agent run. Treat this as a production stabilization and release-hardening pass.

## Mission

Finish Stage D for Project Moonshine/Mash by:

1. Verifying no post-adjudication drift.
2. Closing residual gaps with concrete code/docs updates.
3. Producing a clean release-ready state packet.

Do not produce speculative claims. Every completion statement must map to on-disk evidence.

## Environment Reality

- You do **not** have MCP tools in this environment.
- You must use local file reads/writes and terminal commands only.
- Work from repository root: `C:/Users/treyr/Desktop/Dev-Drive/distill_the_flow`.
- OS assumptions: Windows PowerShell.

## Read-First Contract (Required, in order)

1. `AGENTS.md`
2. `CONTEXT.md`
3. `MEMORY.md`
4. `MY_Development_Style.md`
5. `STATE_SNAPSHOT_20260217.md`
6. `docs/STATE_SNAPSHOT_20260217.md`
7. `docs/MOONSHINE_DISTRIBUTED_PLAN_20260217.md`
8. `docs/HANDOFF_SCHEMA.json`
9. `docs/HANDOFF_PACKET_20260217_STAGE_A.json`
10. `docs/HANDOFF_PACKET_20260217_STAGE_B_BUILDER.json`
11. `docs/HANDOFF_PACKET_20260217_STAGE_C_ADJUDICATION.json`
12. `docs/HANDOFF_PACKET_20260217_CLAUDE_FINISHER.json`
13. `docs/STAGE_A_VERIFICATION_20260217.md`
14. `docs/STAGE_B_VERIFICATION_20260217.md`
15. `reports/token_ledger.json`
16. `reports/moonshine_distillation_manifest.json`
17. `moonshine_corpus_analyzer.py`
18. `moonshine_visualizer.py`
19. `token_forensics_agents.py`
20. `token_forensics_orchestrator.py`
21. `file-processor/moonshine_output.py`

## Long-Horizon Manifest Reference (No MCP Fallback)

Load and parse this file to anchor tool/capability awareness for planning structure:

- `.claude/skills/bb7-distributed-cognition/bb7_manifest.json`

Use this category map for long-horizon planning structure:

- `analysis`
- `execution`
- `exoskeleton`
- `files`
- `memory`
- `misc`
- `project_context`
- `sessions`
- `system`
- `visual`
- `web`

Because MCP is unavailable here, use these categories only as a planning rubric, not a runtime dependency.

## Known Ground Truth (Do Not Re-argue)

- Gate A: PASS.
- Gate B: PASS for core contracts.
- Stage C adjudication packet exists and signals `forward`.
- Distilled artifacts exist, including:
  - `reports/moonshine_corpus.db`
  - `reports/moonshine_distillation_manifest.json`
  - `reports/token_ledger.json`
  - `visualizations/distilled_corpus_dashboard.png`
  - `visualizations/quality_metrics_distilled_timeseries.png`

## Critical Residual Caveat You Must Resolve

`moonshine_corpus_analyzer.py` still contains:

- `tone_shift=0.0  # TODO: compute from tone changes`

This means at least one claim from the Stage B narrative is overstated. Stage D must do one of the following and document it explicitly:

1. Implement a real `tone_shift` computation and validate it.
2. Keep placeholder behavior but downgrade/clarify claim language in docs/handoff packets as accepted technical debt.

No silent carry-forward allowed.

## Subagent-Oriented Execution (If Your Runtime Supports It)

Spin up these lanes in parallel. If subagents are unavailable, execute in the same order sequentially while preserving deliverables.

### Lane A â€” Evidence/Drift Auditor

Responsibilities:

- Re-validate Stage A/B/C artifact existence and key values.
- Confirm checksums/timestamps where possible.
- Verify no new drift in token ledger/manifest values.

Outputs:

- `docs/STAGE_D_EVIDENCE_AUDIT_20260217.md`
- Update risk notes in handoff packet if any mismatch is found.

### Lane B â€” Code Hardener

Responsibilities:

- Resolve `tone_shift` caveat in `moonshine_corpus_analyzer.py`.
- Ensure implementation is deterministic and documented.
- Avoid placeholder/TODO behavior.

Minimum acceptance:

- `tone_shift` is computed from observed conversation signals.
- No `# TODO` remains for this field.
- Analyzer still runs without regressions.

### Lane C â€” Docs/State Integrator

Responsibilities:

- Synchronize `README.md`, `AGENTS.md`, `CONTEXT.md`, `MEMORY.md`, and both snapshot files.
- Ensure all stage status statements match actual evidence.
- Add a clear release-note style delta for Stage D.

### Lane D â€” Release Gatekeeper

Responsibilities:

- Run final validation and summarize pass/fail.
- Produce final Stage D handoff packet conforming to `docs/HANDOFF_SCHEMA.json`.

Outputs:

- `docs/HANDOFF_PACKET_20260217_CLAUDE_FINISHER.json` (updated)
- `docs/RELEASE_ALIGNMENT_20260217.md` (updated)
- `docs/EXECUTION_LOG_20260217.json` (updated)

## Required Concrete Checks

Run and capture outcomes for at least:

1. `python scripts/run_validation.py --reports-dir reports --strict`
2. Targeted SQL checks against `reports/moonshine_corpus.db`:
   - existence of `distilled_conversations`
   - QC-1 topic query
   - QC-2 token band query
   - QC-3 gold-tier provenance query
3. Verify `reports/token_ledger.json` counters remain internally consistent.
4. Confirm visual outputs exist and are non-empty files.

## Non-Negotiables

- No speculative completion claims.
- No partial placeholder implementations.
- No silent changes to canonical counts without explanation.
- Every doc status claim must point to file evidence.

## Token Scope Guardrail

Never merge these scopes without explicit labels:

- Raw serialized JSON token estimate (~458.5M)
- Source-locked message content total (115,330,530)
- Distilled canonical selected tokens (~104.3M)

## Required Final Deliverables

1. Updated code and docs with all Stage D deltas.
2. Updated `docs/HANDOFF_PACKET_20260217_CLAUDE_FINISHER.json` with final signal:
   - `forward` only if release-ready.
   - `revision_request` if any blocker remains.
3. One concise release summary including:
   - what changed
   - what was validated
   - any accepted debts
   - exact file list changed

## Final Response Format

Return your final response with these sections:

1. `Stage D Outcome`
2. `Findings and Fixes`
3. `Validation Evidence`
4. `Files Changed`
5. `Residual Risks`
6. `Signal`

Be explicit, evidence-backed, and conservative.
