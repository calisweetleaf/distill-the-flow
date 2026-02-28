# Final DB Pass Validation (2026-02-28)

Generated: 2026-02-28T02:41:46.920593+00:00

## Live Authority
- main DB: `reports/main/moonshine_mash_active.db`
- exact token recount: `reports/main/token_recount.main.postdeps.json`
- prior checkpoint: `reports/main/final_db_pass_20260227.json`

## Delta vs 2026-02-27 Checkpoint
- conversations: 2591 -> 2788 (delta 197)
- messages: 177837 -> 179974 (delta 2137)
- distilled_conversations: 2349 -> 2486 (delta 137)
- exact all_non_system tokens: 120843809 -> 122627092 (delta 1783283)
- exact distilled_non_system tokens: 109421548 -> 110539045 (delta 1117497)

## Provider Composition
- conversations: chatgpt=1439, claude=954, deepseek=320, qwen=75
- messages: chatgpt=169397, claude=7726, deepseek=2073, qwen=778
- distilled_conversations: chatgpt=1326, claude=789, deepseek=304, qwen=67

## Claude Provider-Run Breakdown In Main
- conversations: claude_20260226_065717=757, claude_20260227_080825_20260226=197
- messages: claude_20260226_065717=5589, claude_20260227_080825_20260226=2137
- distilled_conversations: claude_20260226_065717=652, claude_20260227_080825_20260226=137

## Exact Recount (o200k_base)
- all_non_system tokens total: 122627092
- all_non_system messages total: 179974
- distilled_non_system tokens total: 110539045
- distilled_non_system messages total: 170180

## Provider-Local Ledger Repair
- status: repaired
- claude_20260226_065717: content_tokens_non_system=3008283, distilled_tokens_selected=2285649, source_origin=exact_message_recount
- claude_20260227_080825_20260226: content_tokens_non_system=1854471, distilled_tokens_selected=1134446, source_origin=exact_message_recount
- qwen_20260226_063147: content_tokens_non_system=1017719, distilled_tokens_selected=953828, source_origin=exact_message_recount
- deepseek_20260226_063139: content_tokens_non_system=1482829, distilled_tokens_selected=1368910, source_origin=exact_message_recount

## Integrity Checks
- quick_check: ok
- record_uid collisions: {'conversations': 0, 'messages': 0, 'distilled_conversations': 0}
- merge_manifest_skip_only_rerun: True
- late_claude_layer_present_in_main: True
- provider_local_115330530_baseline_removed: True
- all checks pass: True

## Merge Manifest Interpretation
- latest main merge manifest: `reports/main/merge_manifest.main.json`
- note: Latest main merge manifest is a skip-only rerun artifact. The late Claude layer is still present in the live DB and exact recount, so main authority must be read from the live DB plus token_recount.main.postdeps.json rather than the skip-only merge counters alone.

JSON evidence: `reports/main/final_db_pass_20260228.json`
