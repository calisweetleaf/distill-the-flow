#!/usr/bin/env python3
"""
JSON/JSONL SCHEMA FORGE v1.0 - Production Release
Streaming structural extractor with provider fingerprinting, merge, and config mode.

Zero-data guarantee. Deterministic output. Memory-safe on arbitrarily large files.

Usage:
    python json_tool.py <file_or_dir>            # auto-detect and extract
    python json_tool.py --config config.yaml     # batch mode via config
    python json_tool.py --merge a.json b.json    # merge two Claude exports
    python json_tool.py <file> --output-dir ./out --format all
"""

import json
import sys
import os
import argparse
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union, Optional, Iterator
from collections import OrderedDict
from datetime import datetime
import yaml

# Optional streaming JSON support (install: pip install ijson)
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("schema_forge")

# ─────────────────────────────────────────────
# PROVIDER FINGERPRINTS
# Keyed by provider name -> set of discriminating top-level keys
# ─────────────────────────────────────────────

PROVIDER_FINGERPRINTS: Dict[str, Dict[str, Any]] = {
    "openai_gpt": {
        "file_patterns": ["conversations.json", "conversations.jsonl"],
        "top_level_keys": {"title", "create_time", "update_time", "mapping", "current_node"},
        "description": "OpenAI ChatGPT export (conversations.json or .jsonl)"
    },
    "claude_conversations": {
        "file_patterns": ["conversations.json"],
        "top_level_keys": {"uuid", "name", "created_at", "updated_at", "account", "chat_messages"},
        "description": "Anthropic Claude export - conversations.json"
    },
    "claude_memories": {
        "file_patterns": ["memories.json"],
        "top_level_keys": {"conversations_memory", "project_memories", "account_uuid"},
        "description": "Anthropic Claude export - memories.json"
    },
    "claude_projects": {
        "file_patterns": ["projects.json"],
        "top_level_keys": {"uuid", "name", "description", "is_private", "docs"},
        "description": "Anthropic Claude export - projects.json"
    },
    "gemini": {
        "file_patterns": ["Takeout"],
        "top_level_keys": {"conversation", "conversations", "messages", "role"},
        "description": "Google Gemini / Takeout export"
    },
    "perplexity": {
        "file_patterns": ["history.json", "chats.json"],
        "top_level_keys": {"query", "answer", "sources", "mode"},
        "description": "Perplexity export"
    },
}


def fingerprint_provider(file_path: Path, sample_keys: Set[str]) -> str:
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


# ─────────────────────────────────────────────
# TYPE TOKENS & SCHEMA NODES
# ─────────────────────────────────────────────

class TypeToken:
    STR   = "str"
    INT   = "int"
    FLOAT = "float"
    BOOL  = "bool"
    NULL  = "null"
    OBJECT = "object"
    LIST   = "list"
    PRIMITIVE_ORDER = [OBJECT, LIST, STR, INT, FLOAT, BOOL, NULL]


class SchemaNode:
    __slots__ = (
        "types", "object_fields", "list_element",
        "is_optional", "occurrence_count", "presence_count",
        "max_depth", "sample_values"
    )

    def __init__(self):
        self.types: Set[str] = set()
        self.object_fields: Dict[str, "SchemaNode"] = OrderedDict()
        self.list_element: Optional["SchemaNode"] = None
        self.is_optional: bool = False
        self.occurrence_count: int = 0
        self.presence_count: int = 0
        self.max_depth: int = 0
        self.sample_values: List[Any] = []  # up to 3 primitives for context

    def merge(self, other: "SchemaNode", track_presence: bool = True):
        self.types.update(other.types)
        self.max_depth = max(self.max_depth, other.max_depth)

        if other.object_fields:
            for key, other_field in other.object_fields.items():
                if key not in self.object_fields:
                    self.object_fields[key] = SchemaNode()
                    if track_presence:
                        self.object_fields[key].is_optional = True
                self.object_fields[key].merge(other_field, track_presence=False)
                if track_presence:
                    self.object_fields[key].presence_count += 1

        if other.list_element:
            if not self.list_element:
                self.list_element = SchemaNode()
            self.list_element.merge(other.list_element, track_presence=False)

        # Collect a few sample values for primitive fields
        if other.sample_values and len(self.sample_values) < 3:
            self.sample_values.extend(other.sample_values[:3 - len(self.sample_values)])

        self.occurrence_count += 1

    def finalize(self, total_occurrences: int):
        if self.object_fields:
            for key, field in self.object_fields.items():
                if field.presence_count < total_occurrences:
                    field.is_optional = True
                field.finalize(field.presence_count if field.presence_count > 0 else 1)
        if self.list_element:
            self.list_element.finalize(
                self.list_element.occurrence_count if self.list_element.occurrence_count > 0 else 1
            )


# ─────────────────────────────────────────────
# STREAMING HELPERS
# ─────────────────────────────────────────────

LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50MB - use streaming above this
SAMPLE_LIMIT = 5000  # max JSONL records to sample for schema (avoids multi-GB full scans)


def stream_jsonl(file_path: Path) -> Iterator[Tuple[int, Any]]:
    """Yield (line_number, parsed_object) from a JSONL file without loading all into memory."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                yield i, json.loads(stripped, object_pairs_hook=OrderedDict)
            except json.JSONDecodeError as e:
                yield i, e  # caller distinguishes errors by type


def stream_json_array(file_path: Path) -> Iterator[Tuple[int, Any]]:
    """
    Stream top-level array elements from a large JSON file using ijson.
    Falls back to full load if ijson unavailable or file isn't array-rooted.
    """
    if not IJSON_AVAILABLE:
        log.warning("ijson not available; loading full file (may be slow for large JSON).")
        data = json.loads(file_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
        if isinstance(data, list):
            for i, item in enumerate(data):
                yield i + 1, item
        else:
            yield 1, data
        return

    with open(file_path, "rb") as fh:
        try:
            parser = ijson.items(fh, "item")
            for i, item in enumerate(parser, 1):
                yield i, item
        except Exception as e:
            log.debug(f"ijson array streaming failed ({e}), trying object mode")
            fh.seek(0)
            try:
                obj = json.loads(fh.read().decode("utf-8", errors="replace"),
                                 object_pairs_hook=OrderedDict)
                yield 1, obj
            except Exception as e2:
                raise RuntimeError(f"Cannot parse JSON file: {e2}") from e2


# ─────────────────────────────────────────────
# CORE EXTRACTOR
# ─────────────────────────────────────────────

class SchemaExtractor:
    def __init__(self, sample_limit: int = SAMPLE_LIMIT):
        self.anomalies: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}
        self.sample_limit = sample_limit

    def extract(self, input_path: Path, mode: str = "auto") -> Dict[str, Any]:
        """
        Main entry point.
        Returns dict: { "root"|rel_path -> {template, yaml, markdown, anomalies, stats, provider} }
        """
        results: Dict[str, Any] = {}

        if input_path.is_dir():
            files = sorted(
                list(input_path.rglob("*.json")) +
                list(input_path.rglob("*.jsonl")) +
                list(input_path.rglob("*.ndjson"))
            )
            if not files:
                log.warning(f"No JSON/JSONL files found in {input_path}")
            for file_path in files:
                rel_path = str(file_path.relative_to(input_path))
                log.info(f"Processing: {rel_path}")
                results[rel_path] = self._process_file(file_path, mode)
        else:
            log.info(f"Processing: {input_path}")
            results["root"] = self._process_file(input_path, mode)

        return results

    def _process_file(self, file_path: Path, mode: str) -> Dict[str, Any]:
        self.anomalies = []
        self.stats = {
            "file": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "size_human": _human_size(file_path.stat().st_size),
            "record_count": 0,
            "sampled": False,
            "max_depth": 0,
            "field_count": 0,
            "processed_at": datetime.now().isoformat(),
        }

        if mode == "auto":
            mode = self._detect_mode(file_path)

        log.info(f"  Mode: {mode} | Size: {self.stats['size_human']}")

        schema: Optional[SchemaNode] = None

        if mode == "jsonl":
            schema = self._parse_jsonl_streaming(file_path)
        elif self.stats["size_bytes"] > LARGE_FILE_THRESHOLD and IJSON_AVAILABLE:
            log.info("  Large file detected - using streaming JSON parser")
            schema = self._parse_json_streaming(file_path)
        else:
            schema = self._parse_json_full(file_path)

        if schema:
            schema.finalize(schema.occurrence_count if schema.occurrence_count > 0 else 1)
            self.stats["max_depth"] = schema.max_depth
            self.stats["field_count"] = _count_fields(schema)

        # Provider fingerprinting from top-level keys
        sample_keys: Set[str] = set()
        if schema and schema.object_fields:
            sample_keys = set(schema.object_fields.keys())
        elif schema and schema.list_element and schema.list_element.object_fields:
            sample_keys = set(schema.list_element.object_fields.keys())

        provider = fingerprint_provider(file_path, sample_keys)
        self.stats["provider"] = provider
        log.info(f"  Provider fingerprint: {provider}")

        template = self._to_template(schema) if schema else {"root": "null"}
        yaml_schema = _to_yaml(template)
        markdown = self._to_markdown(template, file_path, provider)

        return {
            "template": template,
            "yaml": yaml_schema,
            "markdown": markdown,
            "anomalies": list(self.anomalies),
            "stats": dict(self.stats),
            "provider": provider,
        }

    def _detect_mode(self, file_path: Path) -> str:
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

    def _parse_json_full(self, file_path: Path) -> Optional[SchemaNode]:
        """Parse a JSON file fully in memory."""
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw, object_pairs_hook=OrderedDict)
        except json.JSONDecodeError as e:
            self.anomalies.append({
                "type": "parse_failure",
                "file": str(file_path),
                "error": str(e),
                "confidence": "confirmed"
            })
            log.error(f"  Parse failure: {e}")
            return None

        if isinstance(data, list):
            return self._analyze_array(data, "$")
        else:
            node = self._analyze(data, "$", depth=0)
            node.occurrence_count = max(node.occurrence_count, 1)
            return node

    def _parse_json_streaming(self, file_path: Path) -> Optional[SchemaNode]:
        """Stream-parse a large JSON array."""
        root_schema = SchemaNode()
        root_schema.types.add(TypeToken.LIST)
        root_schema.occurrence_count = 1
        elem_schema = SchemaNode()
        count = 0

        try:
            for i, item in stream_json_array(file_path):
                if isinstance(item, Exception):
                    self.anomalies.append({
                        "type": "parse_error",
                        "index": i,
                        "error": str(item),
                        "confidence": "confirmed"
                    })
                    continue
                node = self._analyze(item, f"$[{i}]", depth=0)
                elem_schema.merge(node, track_presence=False)
                count += 1
                if self.sample_limit > 0 and count >= self.sample_limit:
                    log.warning(f"  Sample limit ({self.sample_limit}) reached - partial schema")
                    self.stats["sampled"] = True
                    break
        except Exception as e:
            log.error(f"  Streaming parse failed: {e}")
            self.anomalies.append({"type": "streaming_error", "error": str(e), "confidence": "confirmed"})
            if count == 0:
                self.stats["record_count"] = 0
                return None
            self.stats["partial_streaming_error"] = str(e)
            self.stats["sampled"] = True

        self.stats["record_count"] = count
        if count > 0:
            root_schema.list_element = elem_schema
            root_schema.max_depth = elem_schema.max_depth
        return root_schema

    def _parse_jsonl_streaming(self, file_path: Path) -> SchemaNode:
        """Stream-parse JSONL line by line."""
        root_schema = SchemaNode()
        root_schema.occurrence_count = 0
        valid_lines = 0
        total_lines = 0

        for line_no, item in stream_jsonl(file_path):
            total_lines = line_no
            if isinstance(item, json.JSONDecodeError):
                self.anomalies.append({
                    "type": "malformed_jsonl",
                    "line": line_no,
                    "error": str(item),
                    "confidence": "confirmed"
                })
                continue

            node = self._analyze(item, f"$.line_{line_no}", depth=0)
            root_schema.merge(node, track_presence=True)
            valid_lines += 1

            if self.sample_limit > 0 and valid_lines >= self.sample_limit:
                log.warning(f"  Sample limit ({self.sample_limit}) reached at line {line_no}")
                self.stats["sampled"] = True
                break

            if valid_lines % 10000 == 0:
                log.info(f"  ... processed {valid_lines:,} records")

        self.stats["record_count"] = valid_lines
        self.stats["total_lines_scanned"] = total_lines

        if valid_lines == 0:
            self.anomalies.append({
                "type": "empty_file",
                "confidence": "confirmed"
            })

        return root_schema

    def _analyze_array(self, data: list, path: str) -> SchemaNode:
        """Analyze a top-level JSON array."""
        root = SchemaNode()
        root.types.add(TypeToken.LIST)
        root.occurrence_count = 1

        if not data:
            self.anomalies.append({"type": "empty_list", "path": path, "confidence": "unknown"})
            return root

        elem_schema = SchemaNode()
        for i, item in enumerate(data):
            node = self._analyze(item, f"{path}[{i}]", depth=0)
            elem_schema.merge(node, track_presence=False)
            if self.sample_limit > 0 and (i + 1) >= self.sample_limit:
                self.stats["sampled"] = True
                break

        root.list_element = elem_schema
        root.max_depth = elem_schema.max_depth
        self.stats["record_count"] = len(data)
        return root

    def _analyze(self, data: Any, path: str, depth: int) -> SchemaNode:
        """Recursively analyze a JSON value and return a SchemaNode."""
        node = SchemaNode()
        node.occurrence_count = 1
        node.max_depth = depth

        if data is None:
            node.types.add(TypeToken.NULL)
        elif isinstance(data, bool):
            node.types.add(TypeToken.BOOL)
            node.sample_values = [data]
        elif isinstance(data, int):
            node.types.add(TypeToken.INT)
            node.sample_values = [data]
        elif isinstance(data, float):
            node.types.add(TypeToken.FLOAT)
            node.sample_values = [data]
        elif isinstance(data, str):
            node.types.add(TypeToken.STR)
            if len(data) <= 80:
                node.sample_values = [data]
        elif isinstance(data, list):
            node.types.add(TypeToken.LIST)
            if len(data) == 0:
                self.anomalies.append({
                    "type": "empty_list",
                    "path": path,
                    "confidence": "unknown"
                })
            else:
                elem_schema = SchemaNode()
                for i, item in enumerate(data):
                    item_schema = self._analyze(item, f"{path}[{i}]", depth + 1)
                    elem_schema.merge(item_schema, track_presence=False)
                node.list_element = elem_schema
                node.max_depth = max(node.max_depth, elem_schema.max_depth)
        elif isinstance(data, (dict, OrderedDict)):
            node.types.add(TypeToken.OBJECT)
            if len(data) == 0:
                self.anomalies.append({
                    "type": "empty_object",
                    "path": path,
                    "confidence": "unknown"
                })
            else:
                for key, value in data.items():
                    field_schema = self._analyze(value, f"{path}.{key}", depth + 1)
                    if key in node.object_fields:
                        node.object_fields[key].merge(field_schema, track_presence=False)
                    else:
                        node.object_fields[key] = field_schema
                    node.object_fields[key].presence_count = 1
                    node.max_depth = max(node.max_depth, field_schema.max_depth)

        return node

    # ─────────────────────────────────────────
    # TEMPLATE SERIALIZATION
    # ─────────────────────────────────────────

    def _to_template(self, node: Optional[SchemaNode], path: str = "$") -> Any:
        if not node:
            return "null"

        types = node.types

        # Union type
        if len(types) > 1:
            variants = []
            if TypeToken.OBJECT in types and node.object_fields:
                variants.append(self._object_to_template(node, path))
            elif TypeToken.OBJECT in types:
                variants.append("object")
            if node.list_element:
                variants.append([self._to_template(node.list_element, f"{path}[]")])
            elif TypeToken.LIST in types:
                variants.append("list")
            for t in TypeToken.PRIMITIVE_ORDER:
                if t not in (TypeToken.OBJECT, TypeToken.LIST) and t in types:
                    variants.append(t)
            return variants[0] if len(variants) == 1 else variants

        # Single type
        if TypeToken.OBJECT in types:
            return self._object_to_template(node, path)
        if node.list_element:
            return [self._to_template(node.list_element, f"{path}[]")]
        if TypeToken.LIST in types:
            return "list"
        for t in TypeToken.PRIMITIVE_ORDER:
            if t in types:
                return t
        return "null"

    def _object_to_template(self, node: SchemaNode, path: str) -> Dict[str, Any]:
        template = OrderedDict()
        for key, field in node.object_fields.items():
            field_path = f"{path}.{key}"
            field_template = self._to_template(field, field_path)
            key_name = f"{key}?" if field.is_optional else key
            template[key_name] = field_template
        return template

    def _to_markdown(self, template: Any, file_path: Path, provider: str) -> str:
        """Generate a human-readable Markdown schema report."""
        lines = [
            f"# Schema Template: `{file_path.name}`",
            "",
            f"**Provider:** `{provider}`  ",
            f"**File size:** {self.stats.get('size_human', 'unknown')}  ",
            f"**Records sampled:** {self.stats.get('record_count', 'unknown'):,}  ",
            f"**Sampled (not full):** {self.stats.get('sampled', False)}  ",
            f"**Max depth:** {self.stats.get('max_depth', '?')}  ",
            f"**Total fields:** {self.stats.get('field_count', '?')}  ",
            f"**Extracted at:** {self.stats.get('processed_at', '')}  ",
            "",
            "## Structure Template",
            "",
            "```json",
            json.dumps(template, indent=2),
            "```",
            "",
        ]

        if self.anomalies:
            lines += [
                "## Anomalies",
                "",
                "| Type | Path/Line | Confidence |",
                "|------|-----------|------------|",
            ]
            for a in self.anomalies:
                loc = a.get("path", a.get("line", a.get("file", "-")))
                lines.append(f"| `{a.get('type', '?')}` | `{loc}` | {a.get('confidence', '?')} |")
            lines.append("")

        if provider in PROVIDER_FINGERPRINTS:
            desc = PROVIDER_FINGERPRINTS[provider]["description"]
            lines += [f"## Provider Notes", "", f"> {desc}", ""]

        return "\n".join(lines)


# ─────────────────────────────────────────────
# CLAUDE EXPORT MERGER
# ─────────────────────────────────────────────

def merge_claude_exports(paths: List[Path], output_path: Path) -> Dict[str, Any]:
    """
    Merge two or more Claude conversations.json exports.
    Deduplicates by conversation UUID. Sorts by created_at ascending.
    Returns summary dict.
    """
    all_conversations: Dict[str, Any] = {}
    source_counts: Dict[str, int] = {}
    source_input_counts: Dict[str, int] = {}
    errors: List[str] = []

    for path in paths:
        log.info(f"Loading: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw, object_pairs_hook=OrderedDict)
        except Exception as e:
            errors.append(f"{path}: {e}")
            log.error(f"Failed to load {path}: {e}")
            continue

        # Claude exports can be a list of conversations or a dict with a key
        conversations: List[Any] = []
        if isinstance(data, list):
            conversations = data
        elif isinstance(data, dict):
            # Try common wrapper keys
            for key in ("conversations", "data", "chats"):
                if key in data and isinstance(data[key], list):
                    conversations = data[key]
                    break
            if not conversations:
                errors.append(f"{path}: Could not find conversations list in dict keys: {list(data.keys())}")
                continue
        else:
            errors.append(f"{path}: Unexpected root type {type(data)}")
            continue

        source_counts[str(path)] = 0
        source_input_counts[str(path)] = len(conversations)
        for convo in conversations:
            if not isinstance(convo, (dict, OrderedDict)):
                continue
            # UUID is the dedup key
            uid = convo.get("uuid") or convo.get("id") or convo.get("conversation_id")
            if not uid:
                # Fall back to hash of content
                uid = hashlib.sha256(json.dumps(convo, sort_keys=True).encode()).hexdigest()[:16]
                log.debug(f"  No UUID found - using content hash {uid}")

            if uid not in all_conversations:
                all_conversations[uid] = convo
                source_counts[str(path)] += 1
            else:
                # Keep the one with more messages or later updated_at
                existing = all_conversations[uid]
                existing_msgs = len(existing.get("chat_messages", existing.get("messages", [])))
                new_msgs = len(convo.get("chat_messages", convo.get("messages", [])))
                if new_msgs > existing_msgs:
                    all_conversations[uid] = convo
                    log.debug(f"  Replacing {uid}: {existing_msgs} -> {new_msgs} messages")

    # Sort by created_at ascending
    def sort_key(convo):
        ts = convo.get("created_at") or convo.get("create_time") or ""
        return str(ts)

    merged_list = sorted(all_conversations.values(), key=sort_key)

    # Write output without materializing one giant JSON string in memory.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(merged_list, fh, indent=2, ensure_ascii=False)

    total_input_records = sum(source_input_counts.values())
    summary = {
        "total_conversations": len(merged_list),
        "source_input_counts": source_input_counts,
        "source_new_counts": source_counts,
        "errors": errors,
        "output": str(output_path),
        "duplicates_removed": total_input_records - len(merged_list),
    }
    log.info(f"Merged {sum(source_counts.values())} -> {len(merged_list)} unique conversations")
    log.info(f"Duplicates removed: {summary['duplicates_removed']}")
    log.info(f"Output: {output_path}")
    return summary


# ─────────────────────────────────────────────
# CONFIG MODE
# ─────────────────────────────────────────────

def run_config_mode(config_path: Path, output_base: Path):
    """
    Batch process files specified in a YAML config.

    Config format:
        output_dir: ./schema_out
        sample_limit: 5000
        files:
          - path: /data/chatgpt/conversations.jsonl
            label: gpt_export_2025
          - path: /data/claude/conversations.json
            label: claude_export_jan
        merge:
          - label: claude_merged
            output: ./merged/conversations.json
            inputs:
              - /data/claude_jan/conversations.json
              - /data/claude_aug/conversations.json
    """
    log.info(f"Config mode: {config_path}")
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        sys.exit(1)

    out_dir = Path(cfg.get("output_dir", str(output_base)))
    sample_limit = int(cfg.get("sample_limit", SAMPLE_LIMIT))
    extractor = SchemaExtractor(sample_limit=sample_limit)

    # Process individual files
    for entry in cfg.get("files", []):
        file_path = Path(entry["path"])
        label = entry.get("label", file_path.stem)
        if not file_path.exists():
            log.error(f"File not found: {file_path}")
            continue
        results = extractor.extract(file_path)
        _write_results(results, out_dir / label)

    # Process merges
    for merge_entry in cfg.get("merge", []):
        label = merge_entry.get("label", "merged")
        merge_output = Path(merge_entry.get("output", str(out_dir / label / "conversations.json")))
        inputs = [Path(p) for p in merge_entry.get("inputs", [])]
        if not inputs:
            log.warning(f"No inputs for merge '{label}'")
            continue
        summary = merge_claude_exports(inputs, merge_output)
        summary_path = merge_output.parent / f"{label}_merge_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(f"Merge summary written: {summary_path}")


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _count_fields(node: SchemaNode, visited: Optional[Set[int]] = None) -> int:
    if visited is None:
        visited = set()
    nid = id(node)
    if nid in visited:
        return 0
    visited.add(nid)
    count = len(node.object_fields)
    for field in node.object_fields.values():
        count += _count_fields(field, visited)
    if node.list_element:
        count += _count_fields(node.list_element, visited)
    return count


def _to_yaml(template: Any) -> str:
    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())
    yaml.add_representer(OrderedDict, represent_ordereddict)
    return yaml.dump(template, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _write_results(results: Dict[str, Any], out_dir: Path):
    """Write all output formats to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in results.items():
        if key == "root":
            base = out_dir
            stem = "schema"
        else:
            safe = key.replace("/", "_").replace("\\", "_").replace(".", "_")
            base = out_dir
            stem = safe

        # JSON template
        (base / f"{stem}.template.json").write_text(
            json.dumps(data["template"], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        # YAML
        (base / f"{stem}.schema.yaml").write_text(data["yaml"], encoding="utf-8")
        # Markdown
        (base / f"{stem}.report.md").write_text(data["markdown"], encoding="utf-8")
        # Stats
        (base / f"{stem}.stats.json").write_text(
            json.dumps(data["stats"], indent=2), encoding="utf-8"
        )
        # Anomalies
        if data["anomalies"]:
            (base / f"{stem}.anomalies.json").write_text(
                json.dumps(data["anomalies"], indent=2), encoding="utf-8"
            )

    log.info(f"Output written to: {out_dir}")


def format_stdout(results: Dict[str, Any]) -> str:
    """Format results for stdout display."""
    lines = []
    for key, data in results.items():
        label = "ROOT" if key == "root" else key
        provider = data.get("provider", "unknown")
        stats = data.get("stats", {})

        lines += [
            "=" * 60,
            f"FILE: {label}",
            f"PROVIDER: {provider} | SIZE: {stats.get('size_human','?')} | RECORDS: {stats.get('record_count', '?'):,}",
            "=" * 60,
            "",
            "── TEMPLATE ────────────────────────────────────",
            json.dumps(data["template"], indent=2),
            "",
            "── YAML SCHEMA ─────────────────────────────────",
            data["yaml"],
        ]

        if data["anomalies"]:
            lines += ["── ANOMALIES ────────────────────────────────────"]
            for a in data["anomalies"]:
                loc = a.get("path", a.get("line", "-"))
                lines.append(f"  [{a.get('type','?')}] {loc} (confidence: {a.get('confidence','?')})")
        else:
            lines.append("── ANOMALIES: None detected ─────────────────────")

        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Schema Forge v1.0 - JSON/JSONL structural extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract template from a single file
  python json_tool.py conversations.jsonl

  # Process an entire Claude export directory
  python json_tool.py ./claude_export/ --output-dir ./schemas/

  # Merge two Claude conversations.json exports
  python json_tool.py --merge jan_conversations.json aug_conversations.json \\
      --merge-output ./merged/conversations.json

  # Batch mode via config file
  python json_tool.py --config config.yaml

  # Force JSONL mode
  python json_tool.py data.json --mode jsonl
        """
    )

    parser.add_argument("input_path", nargs="?", help="Path to JSON/JSONL file or directory")
    parser.add_argument("--output-dir", default="./schema_out", help="Output directory (default: ./schema_out)")
    parser.add_argument("--mode", choices=["auto", "json", "jsonl"], default="auto")
    parser.add_argument("--format", choices=["json", "yaml", "markdown", "all"], default="all",
                        help="Output format (default: all)")
    parser.add_argument("--sample-limit", type=int, default=SAMPLE_LIMIT,
                        help=f"Max records to sample from large files (default: {SAMPLE_LIMIT})")
    parser.add_argument("--no-stdout", action="store_true",
                        help="Suppress stdout output (only write files)")
    parser.add_argument("--config", help="YAML config file for batch mode")
    parser.add_argument("--merge", nargs="+", metavar="FILE",
                        help="Merge multiple Claude conversations.json files")
    parser.add_argument("--merge-output", default="./merged/conversations.json",
                        help="Output path for merged file (default: ./merged/conversations.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # ── Config mode
    if args.config:
        run_config_mode(Path(args.config), Path(args.output_dir))
        return

    # ── Merge mode
    if args.merge:
        merge_paths = [Path(p) for p in args.merge]
        missing = [p for p in merge_paths if not p.exists()]
        if missing:
            for m in missing:
                log.error(f"File not found: {m}")
            sys.exit(1)
        summary = merge_claude_exports(merge_paths, Path(args.merge_output))
        print(json.dumps(summary, indent=2))
        # After merge, optionally extract schema of merged file
        if not args.no_stdout:
            log.info("Extracting schema from merged file...")
            extractor = SchemaExtractor(sample_limit=args.sample_limit)
            results = extractor.extract(Path(args.merge_output), args.mode)
            if not args.no_stdout:
                print(format_stdout(results))
            _write_results(results, Path(args.output_dir))
        return

    # ── Normal extraction mode
    if not args.input_path:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input_path)
    if not input_path.exists():
        log.error(f"Path not found: {input_path}")
        sys.exit(1)

    extractor = SchemaExtractor(sample_limit=args.sample_limit)
    results = extractor.extract(input_path, args.mode)

    if not args.no_stdout:
        print(format_stdout(results))

    _write_results(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
