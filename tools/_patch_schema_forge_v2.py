from pathlib import Path

path = Path(r"C:/Users/treyr/Desktop/Dev-Drive/distill_the_flow/tools/json_tool.py")
text = path.read_text(encoding="utf-8")

text = text.replace('''    "claude_memories": {\n        "file_patterns": ["memories.json"],\n        "top_level_keys": {"memories", "version"},\n        "description": "Anthropic Claude export - memories.json"\n    },\n''', '''    "claude_memories": {\n        "file_patterns": ["memories.json"],\n        "top_level_keys": {"conversations_memory", "project_memories", "account_uuid"},\n        "description": "Anthropic Claude export - memories.json"\n    },\n''', 1)

text = text.replace('''    "claude_projects": {\n        "file_patterns": ["projects.json"],\n        "top_level_keys": {"projects", "id", "name"},\n        "description": "Anthropic Claude export - projects.json"\n    },\n''', '''    "claude_projects": {\n        "file_patterns": ["projects.json"],\n        "top_level_keys": {"uuid", "name", "description", "is_private", "docs"},\n        "description": "Anthropic Claude export - projects.json"\n    },\n''', 1)

text = text.replace('''        self.stats["record_count"] = count\n        if count > 0:\n            root_schema.list_element = elem_schema\n        return root_schema\n''', '''        self.stats["record_count"] = count\n        if count > 0:\n            root_schema.list_element = elem_schema\n            root_schema.max_depth = elem_schema.max_depth\n        return root_schema\n''', 1)

text = text.replace('''        root.list_element = elem_schema\n        self.stats["record_count"] = len(data)\n        return root\n''', '''        root.list_element = elem_schema\n        root.max_depth = elem_schema.max_depth\n        self.stats["record_count"] = len(data)\n        return root\n''', 1)

old_merge = '''    all_conversations: Dict[str, Any] = {}\n    source_counts: Dict[str, int] = {}\n    errors: List[str] = []\n\n    for path in paths:\n'''
new_merge = '''    all_conversations: Dict[str, Any] = {}\n    source_counts: Dict[str, int] = {}\n    source_input_counts: Dict[str, int] = {}\n    errors: List[str] = []\n\n    for path in paths:\n'''
if old_merge not in text:
    raise SystemExit('merge header block not found')
text = text.replace(old_merge, new_merge, 1)

old_conv_block = '''        source_counts[str(path)] = 0\n        for convo in conversations:\n'''
new_conv_block = '''        source_counts[str(path)] = 0\n        source_input_counts[str(path)] = len(conversations)\n        for convo in conversations:\n'''
if old_conv_block not in text:
    raise SystemExit('merge counts block not found')
text = text.replace(old_conv_block, new_conv_block, 1)

old_summary = '''    summary = {\n        "total_conversations": len(merged_list),\n        "source_new_counts": source_counts,\n        "errors": errors,\n        "output": str(output_path),\n        "duplicates_removed": sum(source_counts.values()) - len(merged_list),\n    }\n'''
new_summary = '''    total_input_records = sum(source_input_counts.values())\n    summary = {\n        "total_conversations": len(merged_list),\n        "source_input_counts": source_input_counts,\n        "source_new_counts": source_counts,\n        "errors": errors,\n        "output": str(output_path),\n        "duplicates_removed": total_input_records - len(merged_list),\n    }\n'''
if old_summary not in text:
    raise SystemExit('merge summary block not found')
text = text.replace(old_summary, new_summary, 1)

path.write_text(text, encoding='utf-8')
print('patched json_tool.py v2')
