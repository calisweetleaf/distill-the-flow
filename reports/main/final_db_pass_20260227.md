# Final DB Pass Validation (2026-02-27)

Generated: 2026-02-27T06:22:18.919145+00:00

## Merge Sequence
- qwen_20260226_063147
- deepseek_20260226_063139

## Table Count Deltas
- conversations: 2196 -> 2591 (delta 395, expected 395)
- messages: 174986 -> 177837 (delta 2851, expected 2851)
- distilled_conversations: 1978 -> 2349 (delta 371, expected 371)

## Provider Presence in Main
- qwen:
  - conversations: source_rows=75, missing_in_main=0
  - messages: source_rows=778, missing_in_main=0
  - distilled_conversations: source_rows=67, missing_in_main=0
- deepseek:
  - conversations: source_rows=320, missing_in_main=0
  - messages: source_rows=2073, missing_in_main=0
  - distilled_conversations: source_rows=304, missing_in_main=0

## Integrity Checks
- quick_check: ok
- record_uid collisions: {'conversations': 0, 'messages': 0, 'distilled_conversations': 0}
- all checks pass: True

## Exact Recount (o200k_base)
- all_non_system tokens total: 120843809
- all_non_system messages total: 177837
- distilled_non_system tokens total: 109421548
- distilled_non_system messages total: 168645

JSON evidence: `reports/main/final_db_pass_20260227.json`
