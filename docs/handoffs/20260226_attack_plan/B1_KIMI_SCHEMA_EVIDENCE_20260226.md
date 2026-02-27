# B1 Kimi Schema Migration â€” Evidence Report

**Date:** 2026-02-26  
**Agent:** Kimi (builder_executor)  
**Task:** Schema Core + Migration (G1-G3)  
**DB Target:** `reports/main/moonshine_mash_active.db`

---

## Executive Summary

Successfully migrated main mash database to Phase 2 multi-provider schema with 100% provider coverage and deterministic record_uid uniqueness.

| Gate | Status | Evidence |
|------|--------|----------|
| G1 Schema Parity | âœ… PASS | All 6 required fields present on all 3 tables |
| G2 Provider Coverage | âœ… PASS | 100% rows populated (1439/1439 conv, 169397/169397 msg, 1326/1326 dist) |
| G3 Record UID Uniqueness | âœ… PASS | All tables have unique record_uid (no collisions) |

---

## Schema Changes Applied

### New Columns Added (All Tables)

| Column | Type | Purpose |
|--------|------|---------|
| `provider` | TEXT | Source provider (e.g., "chatgpt", "claude") |
| `provider_run_id` | TEXT | Immutable run identifier |
| `source_file_sha256` | TEXT | Cryptographic source hash |
| `source_path` | TEXT | Original source file path |
| `ingested_at` | TEXT | ISO8601 ingestion timestamp |
| `record_uid` | TEXT | Deterministic merge key |

### Messages Table Additional Column

| Column | Type | Purpose |
|--------|------|---------|
| `conversation_record_uid` | TEXT | FK reference to parent conversation |

### Deterministic UID Rules Implemented

```
conversations:           {provider}:{provider_run_id}:{conversation_id}
messages:                {provider}:{provider_run_id}:{conversation_id}:{message_id}
distilled_conversations: {provider}:{provider_run_id}:{conversation_id}
```

---

## Migration Execution Log

### Phase 1: Schema Migration (`scripts/migrate_main_db_to_phase2.py`)

```
Tables migrated:
  conversations: +6 columns (provider, provider_run_id, source_file_sha256, source_path, ingested_at, record_uid)
  messages: +7 columns (+ conversation_record_uid)
  distilled_conversations: +6 columns

Rows updated:
  conversations: 1,439
  messages: 169,397
  distilled_conversations: 1,326
```

### Phase 2: Index Creation

| Index | Table | Columns | Type |
|-------|-------|---------|------|
| idx_conv_provider | conversations | provider | INDEX |
| idx_conv_provider_run | conversations | provider, provider_run_id | INDEX |
| idx_conv_record_uid | conversations | record_uid | **UNIQUE** |
| idx_msg_provider | messages | provider | INDEX |
| idx_msg_provider_run | messages | provider, provider_run_id | INDEX |
| idx_msg_conv_record_uid | messages | conversation_record_uid | INDEX |
| idx_msg_record_uid | messages | record_uid | **UNIQUE** |
| idx_distilled_provider | distilled_conversations | provider | INDEX |
| idx_distilled_provider_run | distilled_conversations | provider, provider_run_id | INDEX |
| idx_distilled_record_uid | distilled_conversations | record_uid | **UNIQUE** |

---

## Verification Evidence

### G1: Schema Parity

```json
{
  "schema_parity": {
    "conversations": "PASS",
    "messages": "PASS",
    "distilled_conversations": "PASS"
  }
}
```

All tables contain required Phase 2 columns. No missing fields.

### G2: Provider Coverage

| Table | Total Rows | With Provider | Coverage % |
|-------|-----------|---------------|------------|
| conversations | 1,439 | 1,439 | 100.0% |
| messages | 169,397 | 169,397 | 100.0% |
| distilled_conversations | 1,326 | 1,326 | 100.0% |

All rows populated with deterministic provider values from `token_ledger.main.json`.

### G3: Record UID Uniqueness

| Table | Unique UIDs | Total Rows | Is Unique |
|-------|-------------|------------|-----------|
| conversations | 1,439 | 1,439 | âœ… true |
| messages | 169,397 | 169,397 | âœ… true |
| distilled_conversations | 1,326 | 1,326 | âœ… true |

Zero UID collisions across all tables. Deterministic key generation verified.

---

## Provenance

| Field | Value |
|-------|-------|
| Provider | `chatgpt` |
| Provider Run ID | `moonshine_20260218_072146` |
| Source SHA256 | `4e6d44cd2102d26701858f832ba325e15d3490e61397013b7896a22dfad792dd` |
| Source Path | `02-14-26-ChatGPT\conversations.json` |
| Ingested At | `2026-02-26T05:50:34.252506+00:00` |

---

## Files Changed

1. `reports/main/moonshine_mash_active.db` â€” Schema migrated and data backfilled
2. `scripts/migrate_main_db_to_phase2.py` â€” Created (migration tool)
3. `scripts/backfill_phase2_data.py` â€” Created (backfill tool)
4. `scripts/fix_migration_indexes.py` â€” Created (index repair tool)
5. `scripts/check_schema.py` â€” Created (schema verification tool)

---

## Artifacts Generated

1. `docs/handoffs/20260226_attack_plan/B1_KIMI_SCHEMA_LOG_20260226.json` â€” Machine-readable verification log
2. `docs/handoffs/20260226_attack_plan/B1_KIMI_SCHEMA_EVIDENCE_20260226.md` â€” This report

---

## Notes

- Migration required 2-phase execution due to UNIQUE constraint error on `conversation_record_uid` (fixed by making index non-unique)
- All 169,397 messages now have deterministic record_uids with zero collisions
- Schema is now ready for multi-provider merge operations
- Next dependency: OpenCode B1 merge infrastructure (G4-G8)

---

**Signal:** `forward` â€” G1-G3 complete, ready for G4-G8 merge readiness phase.
