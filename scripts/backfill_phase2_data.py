#!/usr/bin/env python3
"""Backfill Phase 2 provider data after schema migration."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def backfill_data(db_path: str, ledger_path: str = None):
    """Backfill provider fields and record_uids."""
    db_path = Path(db_path)
    
    # Load provenance from ledger
    if ledger_path is None:
        ledger_path = db_path.parent / "token_ledger.main.json"
    else:
        ledger_path = Path(ledger_path)
    
    with open(ledger_path, 'r') as f:
        ledger = json.load(f)
    
    provider = "chatgpt"
    provider_run_id = ledger.get("run_id", "unknown")
    source_sha256 = ledger.get("source_sha256", "unknown")
    source_path = ledger.get("source_path", "unknown")
    ingested_at = datetime.now(timezone.utc).isoformat()
    
    print(f"Backfilling with:")
    print(f"  Provider: {provider}")
    print(f"  Run ID: {provider_run_id}")
    print(f"  Source: {source_path}")
    print()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Update conversations
    print("[1/3] Backfilling conversations...")
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
    conv_updated = cursor.rowcount
    print(f"  Updated {conv_updated} conversations")
    
    # Update messages
    print("[2/3] Backfilling messages...")
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
    msg_updated = cursor.rowcount
    print(f"  Updated {msg_updated} messages")
    
    # Update distilled_conversations
    print("[3/3] Backfilling distilled_conversations...")
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
    dist_updated = cursor.rowcount
    print(f"  Updated {dist_updated} distilled_conversations")
    
    conn.commit()
    conn.close()
    
    print("\nBackfill complete!")
    return {
        "conversations": conv_updated,
        "messages": msg_updated,
        "distilled_conversations": dist_updated,
    }


if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "reports/main/moonshine_mash_active.db"
    ledger_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    stats = backfill_data(db_path, ledger_path)
    print(f"\nSummary: {stats}")
