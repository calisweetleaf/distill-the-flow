#!/usr/bin/env python3
"""Update moonshine_corpus_analyzer.py to include Phase 2 schema by default."""


def main():
    with open('moonshine_corpus_analyzer.py', 'r') as f:
        lines = f.readlines()
    
    # Find and update the CREATE TABLE statements
    output = []
    i = 0
    changes = []
    
    while i < len(lines):
        line = lines[i]
        
        # Look for conversations table creation
        if 'CREATE TABLE conversations' in line and 'period INTEGER' in lines[i+25:i+35]:
            # Find the end of this CREATE statement
            j = i
            while j < len(lines) and ')' not in lines[j]:
                j += 1
            
            # Replace the entire CREATE TABLE block
            new_table = '''        cursor.execute("""
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                updated_at REAL,
                total_turns INTEGER,
                user_turns INTEGER,
                assistant_turns INTEGER,
                duration_minutes REAL,
                user_tokens INTEGER,
                assistant_tokens INTEGER,
                token_ratio REAL,
                total_tokens INTEGER,
                user_entropy REAL,
                semantic_density REAL,
                information_gain REAL,
                repetition_score REAL,
                tone_shift REAL,
                malicious_compliance REAL,
                topic_primary TEXT,
                topic_secondary TEXT,
                tone_cluster TEXT,
                code_blocks INTEGER,
                terminal_outputs INTEGER,
                tables INTEGER,
                manifests INTEGER,
                correction_events INTEGER,
                period INTEGER,
                provider TEXT,
                provider_run_id TEXT,
                source_file_sha256 TEXT,
                source_path TEXT,
                ingested_at TEXT,
                record_uid TEXT UNIQUE
            )
        """)'''
            output.append(new_table + '\n')
            changes.append("Updated conversations table with Phase 2 columns")
            i = j + 1
            continue
        
        # Look for messages table creation
        if 'CREATE TABLE messages' in line:
            j = i
            while j < len(lines) and ')' not in lines[j]:
                j += 1
            
            new_table = '''        cursor.execute("""
            CREATE TABLE messages (
                message_id TEXT,
                conversation_id TEXT,
                conversation_title TEXT,
                role TEXT,
                text TEXT,
                create_time REAL,
                char_count INTEGER,
                word_count INTEGER,
                provider TEXT,
                provider_run_id TEXT,
                source_file_sha256 TEXT,
                source_path TEXT,
                ingested_at TEXT,
                conversation_record_uid TEXT,
                record_uid TEXT,
                PRIMARY KEY (message_id, conversation_id)
            )
        """)'''
            output.append(new_table + '\n')
            changes.append("Updated messages table with Phase 2 columns")
            i = j + 1
            continue
        
        # Look for distilled_conversations table creation
        if 'CREATE TABLE distilled_conversations' in line and 'inclusion_reason' in lines[i+15:i+25]:
            j = i
            while j < len(lines) and ')' not in lines[j]:
                j += 1
            
            new_table = '''        cursor.execute("""
            CREATE TABLE distilled_conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                updated_at REAL,
                total_turns INT,
                user_turns INT,
                assistant_turns INT,
                total_tokens INT,
                user_tokens INT,
                assistant_tokens INT,
                token_ratio REAL,
                information_gain REAL,
                malicious_compliance REAL,
                user_entropy REAL,
                semantic_density REAL,
                repetition_score REAL,
                correction_events INT,
                topic_primary TEXT,
                tone_cluster TEXT,
                period INT,
                source_hash TEXT NOT NULL,
                distilled_at TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                run_id TEXT NOT NULL,
                quality_tier TEXT NOT NULL,
                inclusion_reason TEXT NOT NULL,
                provider TEXT,
                provider_run_id TEXT,
                source_file_sha256 TEXT,
                source_path TEXT,
                ingested_at TEXT,
                record_uid TEXT UNIQUE,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)'''
            output.append(new_table + '\n')
            changes.append("Updated distilled_conversations table with Phase 2 columns")
            i = j + 1
            continue
        
        output.append(line)
        i += 1
    
    with open('moonshine_corpus_analyzer.py', 'w') as f:
        f.writelines(output)
    
    print("Changes made:")
    for change in changes:
        print(f"  - {change}")
    
    return len(changes)


if __name__ == "__main__":
    count = main()
    print(f"\nTotal changes: {count}")
