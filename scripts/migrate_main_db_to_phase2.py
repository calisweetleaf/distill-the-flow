#!/usr/bin/env python3
"""
Migrate existing moonshine_mash_active.db to Phase 2 multi-provider schema.

Adds required fields:
- provider
- provider_run_id
- source_file_sha256
- source_path
- ingested_at
- record_uid (deterministic)
- conversation_record_uid (messages only)

Usage:
    python scripts/migrate_main_db_to_phase2.py [db_path] [--dry-run]
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def migrate_database(db_path: str, dry_run: bool = False) -> dict:
    """
    Migrate database to Phase 2 schema.
    
    Returns migration report dict.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating: {db_path}")
    print("=" * 60)
    
    # Load token ledger for provenance
    ledger_path = db_path.parent / "token_ledger.main.json"
    if ledger_path.exists():
        with open(ledger_path, 'r') as f:
            ledger = json.load(f)
        source_sha256 = ledger.get("source_sha256", "unknown")
        source_path = ledger.get("source_path", "unknown")
        provider_run_id = ledger.get("run_id", "unknown")
    else:
        print(f"Warning: No ledger found at {ledger_path}")
        source_sha256 = "unknown"
        source_path = "unknown"
        provider_run_id = "migration_unknown"
    
    # Deterministic provider values
    provider = "chatgpt"  # Current canonical is ChatGPT-only
    ingested_at = datetime.now(timezone.utc).isoformat()
    
    print(f"Provider: {provider}")
    print(f"Provider Run ID: {provider_run_id}")
    print(f"Source SHA256: {source_sha256}")
    print(f"Source Path: {source_path}")
    print(f"Ingested At: {ingested_at}")
    print()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    migration_stats = {
        "db_path": str(db_path),
        "dry_run": dry_run,
        "provider": provider,
        "provider_run_id": provider_run_id,
        "source_sha256": source_sha256,
        "source_path": source_path,
        "ingested_at": ingested_at,
        "tables_migrated": [],
        "indexes_created": [],
    }
    
    # Helper to check if column exists
    def column_exists(table: str, column: str) -> bool:
        cursor.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cursor.fetchall())
    
    # Helper to add column if missing
    def add_column(table: str, column: str, ddl_type: str):
        if not column_exists(table, column):
            if dry_run:
                print(f"  [DRY RUN] Would add column {column} to {table}")
            else:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                print(f"  Added column {column} to {table}")
            return True
        else:
            print(f"  Column {column} already exists in {table}")
            return False
    
    # Migrate conversations table
    print("[1/3] Migrating conversations table...")
    tables_changed = []
    for col in ["provider", "provider_run_id", "source_file_sha256", "source_path", "ingested_at", "record_uid"]:
        if add_column("conversations", col, "TEXT"):
            tables_changed.append(f"conversations.{col}")
    
    # Migrate messages table
    print("\n[2/3] Migrating messages table...")
    for col in ["provider", "provider_run_id", "source_file_sha256", "source_path", "ingested_at", "record_uid"]:
        if add_column("messages", col, "TEXT"):
            tables_changed.append(f"messages.{col}")
    add_column("messages", "conversation_record_uid", "TEXT")
    
    # Migrate distilled_conversations table
    print("\n[3/3] Migrating distilled_conversations table...")
    for col in ["provider", "provider_run_id", "source_file_sha256", "source_path", "ingested_at", "record_uid"]:
        if add_column("distilled_conversations", col, "TEXT"):
            tables_changed.append(f"distilled_conversations.{col}")
    
    migration_stats["tables_migrated"] = tables_changed
    
    if not dry_run:
        # Backfill data
        print("\n[BACKFILL] Populating provider fields...")
        
        # Update conversations
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE provider IS NULL")
        conv_to_update = cursor.fetchone()[0]
        
        cursor.execute("""
            UPDATE conversations
            SET provider = ?,
                provider_run_id = ?,
                source_file_sha256 = ?,
                source_path = ?,
                ingested_at = ?,
                record_uid = ? || ':' || ? || ':' || conversation_id
            WHERE provider IS NULL
        """, (provider, provider_run_id, source_sha256, source_path, ingested_at,
              provider, provider_run_id))
        
        print(f"  Updated {cursor.rowcount} conversations")
        
        # Update messages
        cursor.execute("SELECT COUNT(*) FROM messages WHERE provider IS NULL")
        msg_to_update = cursor.fetchone()[0]
        
        cursor.execute("""
            UPDATE messages
            SET provider = ?,
                provider_run_id = ?,
                source_file_sha256 = ?,
                source_path = ?,
                ingested_at = ?,
                conversation_record_uid = ? || ':' || ? || ':' || conversation_id,
                record_uid = ? || ':' || ? || ':' || conversation_id || ':' || message_id
            WHERE provider IS NULL
        """, (provider, provider_run_id, source_sha256, source_path, ingested_at,
              provider, provider_run_id, provider, provider_run_id))
        
        print(f"  Updated {cursor.rowcount} messages")
        
        # Update distilled_conversations
        cursor.execute("SELECT COUNT(*) FROM distilled_conversations WHERE provider IS NULL")
        dist_to_update = cursor.fetchone()[0]
        
        cursor.execute("""
            UPDATE distilled_conversations
            SET provider = ?,
                provider_run_id = ?,
                source_file_sha256 = ?,
                source_path = ?,
                ingested_at = ?,
                record_uid = ? || ':' || ? || ':' || conversation_id
            WHERE provider IS NULL
        """, (provider, provider_run_id, source_sha256, source_path, ingested_at,
              provider, provider_run_id))
        
        print(f"  Updated {cursor.rowcount} distilled_conversations")
        
        migration_stats["rows_updated"] = {
            "conversations": conv_to_update,
            "messages": msg_to_update,
            "distilled_conversations": dist_to_update,
        }
        
        # Create indexes
        print("\n[INDEXES] Creating Phase 2 indexes...")
        indexes = [
            ("idx_conv_provider", "conversations", "provider"),
            ("idx_conv_provider_run", "conversations", "provider, provider_run_id"),
            ("idx_conv_record_uid", "conversations", "record_uid"),
            ("idx_msg_provider", "messages", "provider"),
            ("idx_msg_provider_run", "messages", "provider, provider_run_id"),
            ("idx_msg_conv_record_uid", "messages", "conversation_record_uid"),
            ("idx_msg_record_uid", "messages", "record_uid"),
            ("idx_distilled_provider", "distilled_conversations", "provider"),
            ("idx_distilled_provider_run", "distilled_conversations", "provider, provider_run_id"),
            ("idx_distilled_record_uid", "distilled_conversations", "record_uid"),
        ]
        
        created_indexes = []
        for idx_name, table, columns in indexes:
            try:
                if "record_uid" in columns:
                    cursor.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})")
                else:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})")
                created_indexes.append(idx_name)
                print(f"  Created index {idx_name}")
            except sqlite3.OperationalError as e:
                print(f"  Index {idx_name} may already exist: {e}")
        
        migration_stats["indexes_created"] = created_indexes
        
        conn.commit()
    
    conn.close()
    
    print("\n" + "=" * 60)
    print(f"Migration {'simulated' if dry_run else 'completed'} successfully!")
    
    return migration_stats


def verify_migration(db_path: str) -> dict:
    """Verify Phase 2 schema is complete and populated."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    verification = {
        "schema_parity": {},
        "provider_coverage": {},
        "record_uid_uniqueness": {},
    }
    
    required_cols = ["provider", "provider_run_id", "source_file_sha256",
                     "source_path", "ingested_at", "record_uid"]
    
    for table in ["conversations", "messages", "distilled_conversations"]:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        
        missing = [c for c in required_cols if c not in columns]
        if table == "messages":
            if "conversation_record_uid" not in columns:
                missing.append("conversation_record_uid")
        
        verification["schema_parity"][table] = "PASS" if not missing else f"MISSING: {missing}"
        
        # Provider coverage check
        if not missing:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE provider IS NOT NULL")
            with_provider = cursor.fetchone()[0]
            coverage = (with_provider / total * 100) if total > 0 else 0
            verification["provider_coverage"][table] = {
                "total_rows": total,
                "with_provider": with_provider,
                "coverage_pct": round(coverage, 2),
            }
            
            # Record UID uniqueness
            cursor.execute(f"SELECT COUNT(DISTINCT record_uid) FROM {table}")
            unique_uids = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total_rows = cursor.fetchone()[0]
            verification["record_uid_uniqueness"][table] = {
                "unique_uids": unique_uids,
                "total_rows": total_rows,
                "is_unique": unique_uids == total_rows,
            }
    
    conn.close()
    return verification


def main():
    parser = argparse.ArgumentParser(description="Migrate DB to Phase 2 schema")
    parser.add_argument("db_path", nargs="?", default="reports/main/moonshine_mash_active.db",
                        help="Path to database file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without changes")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification")
    parser.add_argument("--output-report", type=str, help="Write verification report to JSON file")
    
    args = parser.parse_args()
    
    if args.verify_only:
        print("Running verification only...")
        verification = verify_migration(args.db_path)
        print("\n=== VERIFICATION RESULTS ===")
        print(json.dumps(verification, indent=2))
        
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(verification, f, indent=2)
            print(f"\nReport written to: {args.output_report}")
        
        # Exit with error if any check failed
        all_pass = all(v == "PASS" for v in verification["schema_parity"].values())
        all_unique = all(v["is_unique"] for v in verification["record_uid_uniqueness"].values())
        all_covered = all(v["coverage_pct"] == 100.0 for v in verification["provider_coverage"].values())
        
        if not (all_pass and all_unique and all_covered):
            print("\n[FAILED] Some verification checks failed!")
            sys.exit(1)
        else:
            print("\n[PASS] All verification checks passed!")
            sys.exit(0)
    
    # Run migration
    try:
        stats = migrate_database(args.db_path, dry_run=args.dry_run)
        
        if not args.dry_run:
            print("\nRunning post-migration verification...")
            verification = verify_migration(args.db_path)
            
            full_report = {
                "migration": stats,
                "verification": verification,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            if args.output_report:
                with open(args.output_report, 'w') as f:
                    json.dump(full_report, f, indent=2)
                print(f"\nFull report written to: {args.output_report}")
            
            # Print summary
            print("\n=== MIGRATION SUMMARY ===")
            print(f"Tables migrated: {len(stats['tables_migrated'])}")
            if 'rows_updated' in stats:
                for table, count in stats['rows_updated'].items():
                    print(f"  {table}: {count} rows updated")
            print(f"Indexes created: {len(stats['indexes_created'])}")
            
            print("\n=== VERIFICATION ===")
            for table, result in verification["schema_parity"].items():
                status = "âœ“" if result == "PASS" else "âœ—"
                print(f"  {status} {table}: {result}")
        
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
