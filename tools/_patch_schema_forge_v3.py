from pathlib import Path

path = Path(r"D:/distill_the_flow/tools/json_tool.py")
text = path.read_text(encoding="utf-8")

old_fingerprint = '''def fingerprint_provider(file_path: Path, sample_keys: Set[str]) -> str:\n    \"\"\"Identify provider from filename and discovered top-level keys.\"\"\"\n    fname = file_path.name.lower()\n\n    for provider, meta in PROVIDER_FINGERPRINTS.items():\n        # File name match\n        name_match = any(pat.lower() in fname for pat in meta["file_patterns"])\n        # Key overlap score\n        expected = meta["top_level_keys"]\n        overlap = len(sample_keys & expected)\n        score = overlap / max(len(expected), 1)\n\n        if name_match and score >= 0.4:\n            return provider\n        if not name_match and score >= 0.6:\n            return provider\n\n    return \"unknown\"\n'''

new_fingerprint = '''def fingerprint_provider(file_path: Path, sample_keys: Set[str]) -> str:\n    \"\"\"Identify provider from filename and discovered top-level keys.\"\"\"\n    fname = file_path.name.lower()\n    best_provider = \"unknown\"\n    best_score = -1.0\n    best_name_match = False\n\n    for provider, meta in PROVIDER_FINGERPRINTS.items():\n        name_match = any(pat.lower() in fname for pat in meta["file_patterns"])\n        expected = meta["top_level_keys"]\n        overlap = len(sample_keys & expected)\n        score = overlap / max(len(expected), 1)\n\n        qualifies = (name_match and score >= 0.4) or ((not name_match) and score >= 0.6)\n        if not qualifies:\n            continue\n\n        if name_match and not best_name_match:\n            best_provider = provider\n            best_score = score\n            best_name_match = True\n            continue\n\n        if name_match == best_name_match and score > best_score:\n            best_provider = provider\n            best_score = score\n\n    return best_provider\n'''

old_detect = '''    def _detect_mode(self, file_path: Path) -> str:\n        \"\"\"Detect JSON vs JSONL by extension first, then content sniff.\"\"\"\n        ext = file_path.suffix.lower()\n        if ext in (\".jsonl\", \\".ndjson\\"):\n            return \"jsonl\"\n        if ext == \".json\":\n            # Quick sniff: read first non-whitespace byte\n            with open(file_path, \"r\", encoding=\"utf-8\", errors=\"replace\") as fh:\n                head = fh.read(4096).strip()\n            if not head:\n                return \"json\"\n            # If first char is { or [ and it contains newline-separated objects - could be JSONL\n            # Heuristic: try json.loads on head; if fails, try JSONL\n            try:\n                json.loads(head if len(head) < 4096 else head[:4096])\n                return \"json\"\n            except json.JSONDecodeError:\n                # Check if multiple lines each look like JSON objects\n                lines = [l.strip() for l in head.split(\"\\n\") if l.strip()]\n                valid_obj_lines = sum(\n                    1 for l in lines[:10]\n                    if l.startswith(\"{\") or l.startswith(\"[\")\n                )\n                if valid_obj_lines >= 2:\n                    return \"jsonl\"\n            return \"json\"\n        return \"json\"\n'''

new_detect = '''    def _detect_mode(self, file_path: Path) -> str:\n        \"\"\"Detect JSON vs JSONL by extension first, then lightweight content sniff.\"\"\"\n        ext = file_path.suffix.lower()\n        if ext in (\".jsonl\", \\".ndjson\\"):\n            return \"jsonl\"\n        if ext == \".json\":\n            with open(file_path, \"r\", encoding=\"utf-8\", errors=\"replace\") as fh:\n                head = fh.read(4096).lstrip()\n            if not head:\n                return \"json\"\n            if head.startswith(\"[\") or head.startswith(\"{\"):\n                return \"json\"\n            lines = [l.strip() for l in head.split(\"\\n\") if l.strip()]\n            valid_obj_lines = sum(\n                1 for l in lines[:10]\n                if l.startswith(\"{\") or l.startswith(\"[\")\n            )\n            if valid_obj_lines >= 2:\n                return \"jsonl\"\n            return \"json\"\n        return \"json\"\n'''

if old_fingerprint not in text:
    raise SystemExit('fingerprint function block not found')
text = text.replace(old_fingerprint, new_fingerprint, 1)

if old_detect not in text:
    raise SystemExit('detect_mode block not found')
text = text.replace(old_detect, new_detect, 1)

path.write_text(text, encoding='utf-8')
print('patched json_tool.py v3')
