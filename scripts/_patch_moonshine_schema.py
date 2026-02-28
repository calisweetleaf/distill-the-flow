from pathlib import Path

path = Path("C:/Users/treyr/Desktop/Dev-Drive/distill_the_flow/moonshine_corpus_analyzer.py")
text = path.read_text(encoding="utf-8")

old_init = '''    def __init__(
        self,
        conversations_path: Path,
        output_dir: Path = Path("reports"),
        policy: Optional[DistillationPolicy] = None,
        source_sha256: Optional[str] = None,
    ):
        self.conversations_path = Path(conversations_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "moonshine_corpus.db"
        self.policy = policy or DistillationPolicy()
        self.source_sha256 = source_sha256 or self._hash_source_file()
        self.run_id = datetime.now(timezone.utc).strftime("moonshine_%Y%m%d_%H%M%S")
'''

new_init = '''    def __init__(
        self,
        conversations_path: Path,
        output_dir: Path = Path("reports"),
        policy: Optional[DistillationPolicy] = None,
        source_sha256: Optional[str] = None,
        provider: str = "chatgpt",
        provider_run_id: Optional[str] = None,
    ):
        self.conversations_path = Path(conversations_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "moonshine_corpus.db"
        self.policy = policy or DistillationPolicy()
        self.source_sha256 = source_sha256 or self._hash_source_file()
        self.run_id = datetime.now(timezone.utc).strftime("moonshine_%Y%m%d_%H%M%S")

        normalized_provider = (provider or "chatgpt").strip().lower()
        if not normalized_provider:
            raise ValueError("provider must be non-empty")

        self.provider = normalized_provider
        self.provider_run_id = provider_run_id or self.run_id
        self.ingested_at = datetime.now(timezone.utc).isoformat()
        self.source_path = str(self.conversations_path).replace("\\\\", "/")
'''

if old_init not in text:
    raise SystemExit("old_init block not found")
text = text.replace(old_init, new_init, 1)

old_distill_call = '''        self._write_distillation_manifest(distillation_manifest)
        self._update_token_ledger(distillation_manifest)
        print(f"   [OK] Distilled: {len(distilled_metrics)} conversations, "
              f"{distillation_manifest['distilled_tokens_selected']:,} tokens")
'''

new_distill_call = '''        self._write_distillation_manifest(distillation_manifest)
        self._update_token_ledger(distillation_manifest)
        self._enforce_phase2_schema_contract()
        print(f"   [OK] Distilled: {len(distilled_metrics)} conversations, "
              f"{distillation_manifest['distilled_tokens_selected']:,} tokens")
'''

if old_distill_call not in text:
    raise SystemExit("distillation call block not found")
text = text.replace(old_distill_call, new_distill_call, 1)

insert_marker = '    # ------------------------------------------------------------------\n\n    def _generate_report(self, metrics_list: List[ConversationMetrics]) -> Path:\n'

new_schema_method = '''    def _enforce_phase2_schema_contract(self):
        """Backfill provider-aware schema fields required by Phase 2 multi-provider merge."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        def _ensure_column(table: str, column: str, ddl_type: str):
            cols = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")

        for table in ["conversations", "messages", "distilled_conversations"]:
            _ensure_column(table, "provider", "TEXT")
            _ensure_column(table, "provider_run_id", "TEXT")
            _ensure_column(table, "source_file_sha256", "TEXT")
            _ensure_column(table, "source_path", "TEXT")
            _ensure_column(table, "ingested_at", "TEXT")
            _ensure_column(table, "record_uid", "TEXT")

        _ensure_column("messages", "conversation_record_uid", "TEXT")

        params = (
            self.provider,
            self.provider_run_id,
            self.source_sha256,
            self.source_path,
            self.ingested_at,
            self.provider,
            self.provider_run_id,
            self.provider,
            self.provider_run_id,
            self.source_sha256,
            self.source_path,
            self.ingested_at,
        )
        cursor.execute(
            """
            UPDATE conversations
            SET provider = COALESCE(provider, ?),
                provider_run_id = COALESCE(provider_run_id, ?),
                source_file_sha256 = COALESCE(source_file_sha256, ?),
                source_path = COALESCE(source_path, ?),
                ingested_at = COALESCE(ingested_at, ?),
                record_uid = COALESCE(record_uid, ? || ':' || ? || ':' || conversation_id)
            """,
            params,
        )

        cursor.execute(
            """
            UPDATE messages
            SET provider = COALESCE(provider, ?),
                provider_run_id = COALESCE(provider_run_id, ?),
                source_file_sha256 = COALESCE(source_file_sha256, ?),
                source_path = COALESCE(source_path, ?),
                ingested_at = COALESCE(ingested_at, ?),
                conversation_record_uid = COALESCE(conversation_record_uid, ? || ':' || ? || ':' || conversation_id),
                record_uid = COALESCE(record_uid, ? || ':' || ? || ':' || conversation_id || ':' || message_id)
            """,
            (
                self.provider,
                self.provider_run_id,
                self.source_sha256,
                self.source_path,
                self.ingested_at,
                self.provider,
                self.provider_run_id,
                self.provider,
                self.provider_run_id,
            ),
        )

        cursor.execute(
            """
            UPDATE distilled_conversations
            SET provider = COALESCE(provider, ?),
                provider_run_id = COALESCE(provider_run_id, ?),
                source_file_sha256 = COALESCE(source_file_sha256, ?),
                source_path = COALESCE(source_path, ?),
                ingested_at = COALESCE(ingested_at, ?),
                record_uid = COALESCE(record_uid, ? || ':' || ? || ':' || conversation_id)
            """,
            (
                self.provider,
                self.provider_run_id,
                self.source_sha256,
                self.source_path,
                self.ingested_at,
                self.provider,
                self.provider_run_id,
            ),
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_provider ON conversations(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_provider_run ON conversations(provider, provider_run_id)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_record_uid ON conversations(record_uid)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_provider ON messages(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_provider_run ON messages(provider, provider_run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv_record_uid ON messages(conversation_record_uid)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_msg_record_uid ON messages(record_uid)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_distilled_provider ON distilled_conversations(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_distilled_provider_run ON distilled_conversations(provider, provider_run_id)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_distilled_record_uid ON distilled_conversations(record_uid)")

        conn.commit()
        conn.close()

'''

if insert_marker not in text:
    raise SystemExit("insert marker for schema method not found")
text = text.replace(insert_marker, new_schema_method + insert_marker, 1)

old_ledger_tail = '''        ledger["counters"]["content_tokens_cleaned"] = manifest["distilled_tokens_selected"]
        ledger["counters"]["distilled_tokens_selected"] = manifest["distilled_tokens_selected"]
        ledger["distillation_run_id"] = manifest["run_id"]
        ledger["distillation_timestamp"] = manifest["distillation_timestamp"]
        with open(ledger_path, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, indent=2)
'''

new_ledger_tail = '''        ledger["provider"] = self.provider
        ledger["provider_run_id"] = self.provider_run_id
        ledger["source_path"] = self.source_path
        ledger["source_sha256"] = self.source_sha256
        ledger["counters"]["content_tokens_cleaned"] = manifest["distilled_tokens_selected"]
        ledger["counters"]["distilled_tokens_selected"] = manifest["distilled_tokens_selected"]
        ledger["distillation_run_id"] = manifest["run_id"]
        ledger["distillation_timestamp"] = manifest["distillation_timestamp"]
        with open(ledger_path, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, indent=2)
'''

if old_ledger_tail not in text:
    raise SystemExit("ledger tail block not found")
text = text.replace(old_ledger_tail, new_ledger_tail, 1)

old_main = '''def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python moonshine_corpus_analyzer.py <conversations.json>")
        print("Example: python moonshine_corpus_analyzer.py 02-14-26-ChatGPT/conversations.json")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path("reports")
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    policy = DistillationPolicy()
    analyzer = MoonshineCorpusAnalyzer(input_path, output_dir, policy=policy)
    result = analyzer.analyze()
    
    print(f"\\n[RESULTS] Results:")
    print(f"   Conversations: {result['conversations_analyzed']:,}")
    print(f"   Messages: {result['messages_extracted']:,}")
    print(f"   Database: {result['database']}")
    print(f"   Report: {result['report']}")
'''

new_main = '''def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Moonshine corpus analyzer")
    parser.add_argument("input_path", help="Path to provider export conversations JSON")
    parser.add_argument("output_dir", nargs="?", default="reports", help="Output directory")
    parser.add_argument("--provider", default="chatgpt", help="Provider label (chatgpt, claude, etc.)")
    parser.add_argument("--provider-run-id", default=None, help="Optional provider run identifier")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        raise SystemExit(1)

    policy = DistillationPolicy()
    analyzer = MoonshineCorpusAnalyzer(
        input_path,
        output_dir,
        policy=policy,
        provider=args.provider,
        provider_run_id=args.provider_run_id,
    )
    result = analyzer.analyze()

    print(f"\\n[RESULTS] Results:")
    print(f"   Provider: {analyzer.provider}")
    print(f"   Provider Run ID: {analyzer.provider_run_id}")
    print(f"   Conversations: {result['conversations_analyzed']:,}")
    print(f"   Messages: {result['messages_extracted']:,}")
    print(f"   Database: {result['database']}")
    print(f"   Report: {result['report']}")
'''

if old_main not in text:
    raise SystemExit("main function block not found")
text = text.replace(old_main, new_main, 1)

path.write_text(text, encoding="utf-8")
print("patched moonshine_corpus_analyzer.py")
