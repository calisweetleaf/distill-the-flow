#!/usr/bin/env python3
"""Check current DB schema for Phase 2 migration assessment."""
import sqlite3
import sys
from pathlib import Path

def check_schema(db_path: str):
    """Check schema of existing database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"=== DATABASE: {db_path} ===\n")
    
    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"Tables: {tables}\n")
    
    required_phase2_cols = [
        "provider", "provider_run_id", "source_file_sha256",
        "source_path", "ingested_at", "record_uid"
    ]
    
    for table in tables:
        print(f"=== {table.upper()} ===")
        cursor.execute(f"PRAGMA table_info({table})")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        for col, dtype in columns.items():
            print(f"  {col}: {dtype}")
        
        # Check for Phase 2 columns
        missing = [c for c in required_phase2_cols if c not in columns]
        if missing:
            print(f"  [MISSING Phase 2 cols]: {missing}")
        else:
            print(f"  [OK] All Phase 2 cols present")
        
        # Check for conversation_record_uid on messages
        if table == "messages":
            if "conversation_record_uid" not in columns:
                print(f"  [MISSING]: conversation_record_uid")
            else:
                print(f"  [OK] conversation_record_uid present")
        
        print()
    
    # Check indexes
    print("=== INDEXES ===")
    cursor.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'")
    for row in cursor.fetchall():
        print(f"  {row[0]} on {row[1]}")
    
    conn.close()

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "reports/main/moonshine_mash_active.db"
    check_schema(db_path)
