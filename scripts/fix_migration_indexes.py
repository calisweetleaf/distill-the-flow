#!/usr/bin/env python3
"""Fix indexes after partial migration failure."""
import sqlite3
import sys

def fix_indexes(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Fixing indexes in: {db_path}")
    
    # Check current state
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index'")
    existing = {row[0]: row[1] for row in cursor.fetchall()}
    
    print("\nExisting indexes:")
    for name in sorted(existing.keys()):
        print(f"  {name}")
    
    # Drop and recreate conversation_record_uid as non-unique
    if 'idx_msg_conv_record_uid' in existing:
        print("\nDropping idx_msg_conv_record_uid...")
        cursor.execute("DROP INDEX idx_msg_conv_record_uid")
    
    print("Creating idx_msg_conv_record_uid (non-unique)...")
    cursor.execute("CREATE INDEX idx_msg_conv_record_uid ON messages(conversation_record_uid)")
    
    # Create remaining indexes
    indexes = [
        ('idx_msg_record_uid', 'messages', 'record_uid', True),
        ('idx_distilled_provider', 'distilled_conversations', 'provider', False),
        ('idx_distilled_provider_run', 'distilled_conversations', 'provider, provider_run_id', False),
        ('idx_distilled_record_uid', 'distilled_conversations', 'record_uid', True),
    ]
    
    for idx_name, table, columns, unique in indexes:
        if idx_name in existing and idx_name != 'idx_msg_conv_record_uid':
            print(f"Index {idx_name} already exists, skipping")
            continue
        
        try:
            unique_str = "UNIQUE " if unique else ""
            sql = f"CREATE {unique_str}INDEX {idx_name} ON {table}({columns})"
            print(f"Creating {idx_name}...")
            cursor.execute(sql)
        except sqlite3.OperationalError as e:
            print(f"  Error: {e}")
    
    conn.commit()
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "reports/main/moonshine_mash_active.db"
    fix_indexes(db_path)
