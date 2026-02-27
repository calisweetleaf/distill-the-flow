# Moonshine Distributed Plan - 2026-02-17

## Mission
Distill mash into a reproducible Moonshine corpus where token accounting is explicit at every layer and final storage is a query-first database. The target operating band is:
- Pre-clean reference: ~450M tokens (raw file-level accounting)
- Post-clean reference: ~100M tokens (content-only, policy-cleaned, provenance-locked)

## Canonical Token Ledger (single source of truth)
All stages must write to one ledger artifact: `reports/token_ledger.json`.
Required counts in the same run:
1. `raw_json_tokens` (`o200k_base` over full JSON text)
2. `content_tokens_non_system` (user/assistant/tool text only)
3. `content_tokens_cleaned` (post-policy filtering)
4. `distilled_tokens_selected` (final training/query subset)

Acceptance rule: no downstream report can publish a total unless it cites one of these fields directly.

## Stage Contracts

### Stage A - Kimi Agent 1 (`builder_executor`, OpenCode lane)
Ownership:
- `token_forensics_agents.py`
- `token_forensics_orchestrator.py`
- `run_token_forensics.py`
- `scripts/run_validation.py`

Objectives:
1. Fix ingestion parity so `DataProfilerAgent` handles both ChatGPT list exports and mapping-style dict exports.
2. Add deterministic token ledger generation and include source SHA256, tokenizer, and parsing scope.
3. Replace legacy mixed synthetic checksum assumptions with real runtime checksums only.
4. Emit strict validation statuses that fail on cross-artifact token contradictions.

Deliverables:
- Updated token forensics execution path
- `reports/token_ledger.json`
- `reports/validation_manifest.json` (source-locked)
- `reports/validation_report.md` with mismatch explanations

Gate A (must pass before Stage B completion):
- Token ledger present with all four counters
- Source lock hash equals `4e6d44cd2102d26701858f832ba325e15d3490e61397013b7896a22dfad792dd`
- No placeholder checksum fields in active manifests

### Stage B - Kimi Agent 2 (`builder_executor`, Kimi Code lane)
Ownership:
- `moonshine_corpus_analyzer.py`
- `file-processor/moonshine_output.py`
- `moonshine_visualizer.py`
- `scripts/run_visual_intelligence.py`

Objectives:
1. Align analyzer output with Stage A token ledger instead of heuristic-only totals.
2. Introduce distillation policy application and write selected rows to a dedicated table in SQLite.
3. Add output schema for stable query contracts on distilled corpus slices.
4. Generate quality dashboards over cleaned/distilled subsets, not only raw corpus metrics.

Deliverables:
- Updated Moonshine DB schema with a distilled slice table
- `reports/moonshine_distillation_manifest.json`
- Updated `reports/moonshine_corpus_report.md`
- Visual pack updated for cleaned/distilled analytics

Gate B:
- Distilled token total lands in policy band (90M-110M unless policy explicitly changed)
- DB has provenance columns (`source_sha256`, `policy_version`, `run_id`)
- At least three query contracts validated on distilled data

### Stage C - Codex planner re-review (`planner_reasoner`)
Objectives:
1. Review both Kimi deltas against contracts.
2. Resolve plan drift, contradictions, or missing provenance.
3. Emit either `forward` or `revision_request` handoff packet.

Deliverables:
- Updated handoff packet with risk decision
- Consolidated gap list for finishing lane

### Stage D - Claude finisher (`finisher_polisher`)
Objectives:
1. Stabilize release candidate and remove drift between docs and implementation.
2. Run regression safety pass on reports + DB + visual outputs.
3. Finalize operator docs and release notes.

Deliverables:
- Release-ready docs and manifests
- Final state snapshot
- Final `forward` packet to human relay

## Known Risks to Watch
1. Token baseline drift across raw JSON, message-only, and heuristic analyzers.
2. Legacy artifacts still present in `reports/` can be mistaken as current authority.
3. `moonshine_corpus_analyzer.py` currently hardcodes `tone_shift=0.0`, which weakens metric trust.
4. `file-processor/moonshine_output.py` contains statements conflicting with Moonshine product intent and should be reconciled with current architecture.

## Handoff Protocol Rules
- If a builder cannot satisfy stage gate assumptions, emit `revision_request` immediately.
- Use `escalation` only for irreconcilable conflicts (data corruption, undefined policy ownership, or blocking runtime failures).
- Every stage writes one packet conforming to `docs/HANDOFF_SCHEMA.json`.

## Immediate Next Actions
1. Hand Stage A contract to Kimi OpenCode.
2. Hand Stage B contract to Kimi Code in parallel.
3. Collect both packets plus changed files list.
4. Return to Codex for Stage C drift adjudication.
