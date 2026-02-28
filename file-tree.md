 ┣ dataset_forensics
 ┃ ┣ __pycache__
 ┃ ┃ ┣ cli.cpython-312.pyc
 ┃ ┃ ┣ dedup.cpython-312.pyc
 ┃ ┃ ┣ ingestion.cpython-312.pyc
 ┃ ┃ ┣ metadata.cpython-312.pyc
 ┃ ┃ ┣ normalization.cpython-312.pyc
 ┃ ┃ ┣ quality.cpython-312.pyc
 ┃ ┃ ┣ tokenization.cpython-312.pyc
 ┃ ┃ ┣ __init__.cpython-312.pyc
 ┃ ┃ ┗ __main__.cpython-312.pyc
 ┃ ┣ cli.py
 ┃ ┣ config.single_source.yaml
 ┃ ┣ config.yaml
 ┃ ┣ dedup.py
 ┃ ┣ ingestion.py
 ┃ ┣ metadata.py
 ┃ ┣ normalization.py
 ┃ ┣ quality.py
 ┃ ┣ tokenization.py
 ┃ ┣ __init__.py
 ┃ ┗ __main__.py
 ┣ docs
 ┃ ┣ Moonshine-Analysis-Findings.md
 ┃ ┣ Moonshine-Documentation-Index.md
 ┃ ┣ Moonshine-Phase-Roadmap.md
 ┃ ┣ Moonshine-Project-Overview.md
 ┃ ┣ Moonshine-Technical-Implementation.md
 ┃ ┣ MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md
 ┃ ┣ PROJECT_MOONSHINE_UPDATE_1.json
 ┃ ┣ PROJECT_MOONSHINE_UPDATE_1.md
 ┃ ┣ PROJECT_STATE_AUDIT_20260218.md
 ┃ ┣ PUBLISH_SURFACE_20260228.md
 ┃ ┣ QUERY_CONTRACTS_GATE_B.md
 ┃ ┣ RAW_ONLY_ENFORCEMENT_20260218.md
 ┃ ┗ STATE_SNAPSHOT_20260217.md
 ┣ file-trees
 ┃ ┣ docs-filetree.md
 ┃ ┣ reports-filetree.md
 ┃ ┣ scripts-filetree.md
 ┃ ┗ visualizations-filetree.md
 ┣ mash_merge_archive
 ┃ ┗ main
 ┃ ┃ ┣ chatgpt
 ┃ ┃ ┃ ┗ chatgpt_20260217_r1
 ┃ ┃ ┃ ┃ ┣ dedup_clusters.parquet
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus.legacy_root_20260217.db
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.legacy_root_20260217.md
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.legacy_root_20260217.json
 ┃ ┃ ┃ ┃ ┣ parquet_forensics.raw.json
 ┃ ┃ ┃ ┃ ┣ parquet_forensics.raw.md
 ┃ ┃ ┃ ┃ ┗ token_row_metrics.raw.parquet
 ┃ ┃ ┣ claude_20260226_065717
 ┃ ┃ ┃ ┣ merge_manifest.main.pre_claude_merge.json
 ┃ ┃ ┃ ┗ moonshine_mash_premerge.db
 ┃ ┃ ┣ claude_20260227_080825_20260226
 ┃ ┃ ┃ ┗ moonshine_mash_premerge.db
 ┃ ┃ ┣ deepseek_20260226_063139
 ┃ ┃ ┃ ┗ moonshine_mash_premerge.db
 ┃ ┃ ┣ main_20260219_bootstrap
 ┃ ┃ ┃ ┗ moonshine_mash_bootstrap_from_expansion_20260218.db
 ┃ ┃ ┣ manual_premerge_claude_20260226_before_live_merge
 ┃ ┃ ┃ ┗ moonshine_mash_active.pre_claude_merge.db
 ┃ ┃ ┗ qwen_20260226_063147
 ┃ ┃ ┃ ┗ moonshine_mash_premerge.db
 ┣ reports
 ┃ ┣ canonical
 ┃ ┃ ┣ parquet_forensics.raw.json
 ┃ ┃ ┣ parquet_forensics.raw.md
 ┃ ┃ ┗ token_row_metrics.raw.parquet
 ┃ ┣ chatgpt-expansion_20260218
 ┃ ┃ ┣ atlas
 ┃ ┃ ┃ ┣ atlas_payload.json
 ┃ ┃ ┃ ┣ strategic_command_atlas.html
 ┃ ┃ ┃ ┗ visual_manifest.json
 ┃ ┃ ┣ deep_dive
 ┃ ┃ ┃ ┣ moonshine_deep_dive.json
 ┃ ┃ ┃ ┣ moonshine_deep_dive.md
 ┃ ┃ ┃ ┗ moonshine_deep_dive_manifest.json
 ┃ ┃ ┣ moonshine_corpus.db
 ┃ ┃ ┣ moonshine_corpus_report.md
 ┃ ┃ ┣ moonshine_distillation_manifest.json
 ┃ ┃ ┗ token_ledger.json
 ┃ ┣ main
 ┃ ┃ ┣ db_baseline.pre_qwen_deepseek.json
 ┃ ┃ ┣ db_status.after_qwen.json
 ┃ ┃ ┣ db_status.final_qwen_deepseek.json
 ┃ ┃ ┣ final_db_pass_20260227.json
 ┃ ┃ ┣ final_db_pass_20260227.md
 ┃ ┃ ┣ final_db_pass_20260228.json
 ┃ ┃ ┣ final_db_pass_20260228.md
 ┃ ┃ ┣ merge_manifest.main.json
 ┃ ┃ ┣ merge_manifest.main.zip
 ┃ ┃ ┣ moonshine_mash_active.db
 ┃ ┃ ┣ reports_authority_manifest.json
 ┃ ┃ ┣ token_ledger.main.json
 ┃ ┃ ┣ token_recount.main.json
 ┃ ┃ ┣ token_recount.main.postdeps.json
 ┃ ┃ ┗ token_recount.main.pre_qwen_deepseek.json
 ┃ ┣ providers
 ┃ ┃ ┣ claude
 ┃ ┃ ┃ ┣ claude_20260226_065717
 ┃ ┃ ┃ ┃ ┣ canonical
 ┃ ┃ ┃ ┃ ┃ ┣ parquet_forensics.raw.json
 ┃ ┃ ┃ ┃ ┃ ┣ parquet_forensics.raw.md
 ┃ ┃ ┃ ┃ ┃ ┗ token_row_metrics.raw.parquet
 ┃ ┃ ┃ ┃ ┣ visuals
 ┃ ┃ ┃ ┃ ┃ ┣ atlas_payload.json
 ┃ ┃ ┃ ┃ ┃ ┣ strategic_command_atlas.html
 ┃ ┃ ┃ ┃ ┃ ┗ visual_manifest.json
 ┃ ┃ ┃ ┃ ┣ cost_projection.json
 ┃ ┃ ┃ ┃ ┣ data_profile.json
 ┃ ┃ ┃ ┃ ┣ data_profile.md
 ┃ ┃ ┃ ┃ ┣ dedup_clusters.parquet
 ┃ ┃ ┃ ┃ ┣ dedup_results.json
 ┃ ┃ ┃ ┃ ┣ failure_manifest.json
 ┃ ┃ ┃ ┃ ┣ merge_to_main.claude_20260226_065717.json
 ┃ ┃ ┃ ┃ ┣ MIGRATION_DELTA.md
 ┃ ┃ ┃ ┃ ┣ moonshine_claude_claude_20260226_065717.db
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus.db
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.claude.claude_20260226_065717.md
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.md
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.claude.claude_20260226_065717.json
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.json
 ┃ ┃ ┃ ┃ ┣ pii_safety_report.json
 ┃ ┃ ┃ ┃ ┣ pii_safety_results.json
 ┃ ┃ ┃ ┃ ┣ provider_input_manifest.claude.claude_20260226_065717.json
 ┃ ┃ ┃ ┃ ┣ quality_risk_report.json
 ┃ ┃ ┃ ┃ ┣ quality_scores.json
 ┃ ┃ ┃ ┃ ┣ raw_only_gate_manifest.json
 ┃ ┃ ┃ ┃ ┣ repro_manifest.json
 ┃ ┃ ┃ ┃ ┣ tokenization_results.json
 ┃ ┃ ┃ ┃ ┣ tokenizer_benchmark.csv
 ┃ ┃ ┃ ┃ ┣ token_forensics.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.claude.claude_20260226_065717.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.json
 ┃ ┃ ┃ ┃ ┣ token_row_metrics.parquet
 ┃ ┃ ┃ ┃ ┣ validation_manifest.json
 ┃ ┃ ┃ ┃ ┣ validation_report.md
 ┃ ┃ ┃ ┃ ┣ verification_report.json
 ┃ ┃ ┃ ┃ ┗ visualizations_manifest.claude.claude_20260226_065717.json
 ┃ ┃ ┃ ┗ claude_20260227_080825_20260226
 ┃ ┃ ┃ ┃ ┣ moonshine_claude_claude_20260227_080825_20260226.db
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus.db
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.claude.claude_20260227_080825_20260226.md
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.md
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.claude.claude_20260227_080825_20260226.json
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.json
 ┃ ┃ ┃ ┃ ┣ provider_input_manifest.claude.claude_20260227_080825_20260226.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.claude.claude_20260227_080825_20260226.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.json
 ┃ ┃ ┃ ┃ ┗ visualizations_manifest.claude.claude_20260227_080825_20260226.json
 ┃ ┃ ┣ deepseek
 ┃ ┃ ┃ ┗ deepseek_20260226_063139
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus.db
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.deepseek.deepseek_20260226_063139.md
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.md
 ┃ ┃ ┃ ┃ ┣ moonshine_deepseek_deepseek_20260226_063139.db
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.deepseek.deepseek_20260226_063139.json
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.json
 ┃ ┃ ┃ ┃ ┣ provider_input_manifest.deepseek.deepseek_20260226_063139.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.deepseek.deepseek_20260226_063139.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.json
 ┃ ┃ ┃ ┃ ┗ visualizations_manifest.deepseek.deepseek_20260226_063139.json
 ┃ ┃ ┗ qwen
 ┃ ┃ ┃ ┗ qwen_20260226_063147
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus.db
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.md
 ┃ ┃ ┃ ┃ ┣ moonshine_corpus_report.qwen.qwen_20260226_063147.md
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.json
 ┃ ┃ ┃ ┃ ┣ moonshine_distillation_manifest.qwen.qwen_20260226_063147.json
 ┃ ┃ ┃ ┃ ┣ moonshine_qwen_qwen_20260226_063147.db
 ┃ ┃ ┃ ┃ ┣ provider_input_manifest.qwen.qwen_20260226_063147.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.json
 ┃ ┃ ┃ ┃ ┣ token_ledger.qwen.qwen_20260226_063147.json
 ┃ ┃ ┃ ┃ ┗ visualizations_manifest.qwen.qwen_20260226_063147.json
 ┃ ┣ schema_forge
 ┃ ┃ ┣ chatgpt_export
 ┃ ┃ ┃ ┣ schema.anomalies.json
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┣ claude_export_1
 ┃ ┃ ┃ ┣ schema.anomalies.json
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┣ claude_export_2_conversations
 ┃ ┃ ┃ ┣ schema.anomalies.json
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┣ claude_export_2_memories
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┣ claude_export_2_projects
 ┃ ┃ ┃ ┣ schema.anomalies.json
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┣ claude_export_2_projects_rerun
 ┃ ┃ ┃ ┣ schema.anomalies.json
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┣ claude_merged
 ┃ ┃ ┃ ┗ conversations.json
 ┃ ┃ ┣ claude_merged_schema
 ┃ ┃ ┃ ┣ schema.anomalies.json
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┣ claude_merged_schema_rerun
 ┃ ┃ ┃ ┣ schema.anomalies.json
 ┃ ┃ ┃ ┣ schema.report.md
 ┃ ┃ ┃ ┣ schema.schema.yaml
 ┃ ┃ ┃ ┣ schema.stats.json
 ┃ ┃ ┃ ┗ schema.template.json
 ┃ ┃ ┗ claude_merged_v2
 ┃ ┃ ┃ ┗ conversations.json
 ┃ ┣ visuals
 ┃ ┃ ┣ atlas_payload.json
 ┃ ┃ ┣ strategic_command_atlas.html
 ┃ ┃ ┗ visual_manifest.json
 ┃ ┣ cost_projection.json
 ┃ ┣ data_profile.json
 ┃ ┣ data_profile.md
 ┃ ┣ dedup_clusters.parquet
 ┃ ┣ dedup_results.json
 ┃ ┣ failure_manifest.json
 ┃ ┣ june_5mb_chunk.md
 ┃ ┣ MIGRATION_DELTA.md
 ┃ ┣ pii_safety_report.json
 ┃ ┣ pii_safety_results.json
 ┃ ┣ quality_risk_report.json
 ┃ ┣ quality_scores.json
 ┃ ┣ raw_only_gate_manifest.json
 ┃ ┣ real_export_forensics_run.log
 ┃ ┣ reports-tree.md
 ┃ ┣ repro_manifest.json
 ┃ ┣ tokenization_results.json
 ┃ ┣ tokenizer_benchmark.csv
 ┃ ┣ token_forensics.json
 ┃ ┣ token_forensics.md
 ┃ ┣ token_ledger.json
 ┃ ┣ token_reconciliation.md
 ┃ ┣ validation_manifest.json
 ┃ ┣ validation_report.md
 ┃ ┣ visuals_validation_manifest.json
 ┃ ┗ visuals_validation_report.md
 ┣ scripts
 ┃ ┣ backfill_phase2_data.py
 ┃ ┣ build_clean_release_repo.py
 ┃ ┣ check_schema.py
 ┃ ┣ fix_migration_indexes.py
 ┃ ┣ generate_extended_visuals.py
 ┃ ┣ generate_final_db_pass.py
 ┃ ┣ generate_real_export_forensics.py
 ┃ ┣ merge_provider_to_main.py
 ┃ ┣ migrate_main_db_to_phase2.py
 ┃ ┣ moonshine_export_deep_dive.py
 ┃ ┣ normalize_claude_export.py
 ┃ ┣ reconcile_tokens_and_extract_june.py
 ┃ ┣ recount_main_tokens.py
 ┃ ┣ run_provider_analysis.py
 ┃ ┣ run_validation.py
 ┃ ┣ run_visual_intelligence.py
 ┃ ┣ update_analyzer_schema.py
 ┃ ┣ validate_visual_intelligence.py
 ┃ ┣ verify_release_authority.py
 ┃ ┗ _patch_moonshine_schema.py
 ┣ tools
 ┃ ┣ config.yaml
 ┃ ┣ json_tool.py
 ┃ ┣ schema_forge_README.md
 ┃ ┣ _patch_schema_forge.py
 ┃ ┣ _patch_schema_forge_v2.py
 ┃ ┣ _patch_schema_forge_v3.py
 ┃ ┣ _patch_schema_forge_v4.py
 ┃ ┗ _patch_schema_forge_v5.py
 ┣ visualizations
 ┃ ┣ expansion_20260218
 ┃ ┃ ┣ corpus_overview.png
 ┃ ┃ ┣ corrections_scatter.png
 ┃ ┃ ┣ distilled_corpus_dashboard.png
 ┃ ┃ ┣ period_comparison.png
 ┃ ┃ ┣ quality_metrics_distilled_timeseries.png
 ┃ ┃ ┣ quality_metrics_timeseries.png
 ┃ ┃ ┣ token_ratio_distribution.png
 ┃ ┃ ┗ topic_distribution.png
 ┃ ┣ extended
 ┃ ┃ ┣ cumulative_token_growth.png
 ┃ ┃ ┣ extended_visual_manifest.json
 ┃ ┃ ┣ monthly_message_volume.png
 ┃ ┃ ┣ monthly_tokens_millions.png
 ┃ ┃ ┣ monthly_tokens_vs_messages_scatter.png
 ┃ ┃ ┣ period_quality_signals.png
 ┃ ┃ ┣ role_distribution_pie.png
 ┃ ┃ ┣ tone_cluster_distribution.png
 ┃ ┃ ┣ top15_conversations_tokens.png
 ┃ ┃ ┗ top_topics_barh.png
 ┃ ┣ providers
 ┃ ┃ ┗ claude
 ┃ ┃ ┃ ┗ claude_20260226_065717
 ┃ ┃ ┃ ┃ ┣ corpus_overview.png
 ┃ ┃ ┃ ┃ ┣ corrections_scatter.png
 ┃ ┃ ┃ ┃ ┣ distilled_corpus_dashboard.png
 ┃ ┃ ┃ ┃ ┣ period_comparison.png
 ┃ ┃ ┃ ┃ ┣ quality_metrics_distilled_timeseries.png
 ┃ ┃ ┃ ┃ ┣ quality_metrics_timeseries.png
 ┃ ┃ ┃ ┃ ┣ token_ratio_distribution.png
 ┃ ┃ ┃ ┃ ┗ topic_distribution.png
 ┃ ┣ corpus_overview.png
 ┃ ┣ corpus_overview_dashboard.png
 ┃ ┣ corrections_scatter.png
 ┃ ┣ distilled_corpus_dashboard.png
 ┃ ┣ period_comparison.png
 ┃ ┣ quality_metrics_distilled_timeseries.png
 ┃ ┣ quality_metrics_timeseries.png
 ┃ ┣ token_ratio_distribution.png
 ┃ ┣ topic_distribution.png
 ┃ ┗ visualizations-2-27-2026.zip
 ┣ visuals
 ┃ ┗ logo.png
 ┣ visual_intelligence
 ┃ ┣ command_atlas.py
 ┃ ┗ __init__.py
 ┣ file-tree.md
 ┣ LICENSE
 ┣ moonshine_corpus_analyzer.py
 ┣ moonshine_visualizer.py
 ┣ PROJECT_DATABASE_DOCUMENTATION.md
 ┣ README.md
 ┣ requirements.txt
 ┣ run_token_forensics.py
 ┣ token_forensics_agents.py
 ┣ token_forensics_orchestrator.py
 ┗ WIKI.md