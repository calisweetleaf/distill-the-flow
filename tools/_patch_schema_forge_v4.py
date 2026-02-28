from pathlib import Path

path = Path(r"C:/Users/treyr/Desktop/Dev-Drive/distill_the_flow/tools/json_tool.py")
text = path.read_text(encoding="utf-8")

old_fingerprint = '''def fingerprint_provider(file_path: Path, sample_keys: Set[str]) -> str:
    """Identify provider from filename and discovered top-level keys."""
    fname = file_path.name.lower()

    for provider, meta in PROVIDER_FINGERPRINTS.items():
        # File name match
        name_match = any(pat.lower() in fname for pat in meta["file_patterns"])
        # Key overlap score
        expected = meta["top_level_keys"]
        overlap = len(sample_keys & expected)
        score = overlap / max(len(expected), 1)

        if name_match and score >= 0.4:
            return provider
        if not name_match and score >= 0.6:
            return provider

    return "unknown"
'''

new_fingerprint = '''def fingerprint_provider(file_path: Path, sample_keys: Set[str]) -> str:
    """Identify provider from filename and discovered top-level keys."""
    fname = file_path.name.lower()
    candidates: List[Tuple[float, int, int, str]] = []

    for provider, meta in PROVIDER_FINGERPRINTS.items():
        name_match = any(pat.lower() in fname for pat in meta["file_patterns"])
        expected = meta["top_level_keys"]
        overlap = len(sample_keys & expected)
        score = overlap / max(len(expected), 1)
        threshold = 0.4 if name_match else 0.6

        if score >= threshold:
            # Prefer filename match first, then stronger key overlap, then more absolute key matches.
            candidates.append((1.0 if name_match else 0.0, score, overlap, provider))

    if not candidates:
        return "unknown"

    candidates.sort(reverse=True)
    return candidates[0][3]
'''

old_detect = '''    def _detect_mode(self, file_path: Path) -> str:
        """Detect JSON vs JSONL by extension first, then content sniff."""
        ext = file_path.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            return "jsonl"
        if ext == ".json":
            # Quick sniff: read first non-whitespace byte
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(4096).strip()
            if not head:
                return "json"
            # If first char is { or [ and it contains newline-separated objects - could be JSONL
            # Heuristic: try json.loads on head; if fails, try JSONL
            try:
                json.loads(head if len(head) < 4096 else head[:4096])
                return "json"
            except json.JSONDecodeError:
                # Check if multiple lines each look like JSON objects
                lines = [l.strip() for l in head.split("\\n") if l.strip()]
                valid_obj_lines = sum(
                    1 for l in lines[:10]
                    if l.startswith("{") or l.startswith("[")
                )
                if valid_obj_lines >= 2:
                    return "jsonl"
            return "json"
        return "json"
'''

new_detect = '''    def _detect_mode(self, file_path: Path) -> str:
        """Detect JSON vs JSONL by extension first, then content sniff."""
        ext = file_path.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            return "jsonl"
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(4096)
            stripped = head.lstrip()
            if not stripped:
                return "json"
            # A .json file that begins with an object/array delimiter should be treated as JSON,
            # even if the first 4KB is incomplete due to pretty-printing or file size.
            if stripped[0] in "[{":
                return "json"
            lines = [line.strip() for line in stripped.splitlines() if line.strip()]
            valid_obj_lines = sum(1 for line in lines[:10] if line.startswith("{") or line.startswith("["))
            if valid_obj_lines >= 2:
                return "jsonl"
            return "json"
        return "json"
'''

if old_fingerprint not in text:
    raise SystemExit('fingerprint block not found')
if old_detect not in text:
    raise SystemExit('detect block not found')

text = text.replace(old_fingerprint, new_fingerprint)
text = text.replace(old_detect, new_detect)
path.write_text(text, encoding='utf-8')
print('patched tools/json_tool.py')
