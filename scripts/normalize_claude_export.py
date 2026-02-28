#!/usr/bin/env python3
"""
Normalize Claude export format into ChatGPT-style conversation mapping.

Supports two inputs:
1) Single export file (JSON or JSONL)
2) Claude export directory containing conversations.json (+ optional sidecars)

Output:
- Normalized conversations list (ChatGPT-style mapping)
- Optional metadata sidecar JSON with source/sidecar stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


def parse_iso_timestamp(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_message_text(message: Dict[str, Any]) -> str:
    direct = str(message.get("text") or "").strip()
    if direct:
        return direct

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            txt = block.strip()
            if txt:
                parts.append(txt)
            continue

        if not isinstance(block, dict):
            continue

        for key in ("text", "content", "value", "result", "output"):
            value = block.get(key)
            if isinstance(value, str):
                txt = value.strip()
                if txt:
                    parts.append(txt)
                    break

    return "\n\n".join(parts).strip()


def normalize_conversation(conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conv_id = conv.get("uuid") or conv.get("id")
    if not conv_id:
        return None

    title = str(conv.get("name") or conv.get("title") or "Untitled").strip() or "Untitled"

    conv_created = parse_iso_timestamp(conv.get("created_at"))
    conv_updated = parse_iso_timestamp(conv.get("updated_at"))

    raw_messages = conv.get("chat_messages") or conv.get("messages") or []
    if not isinstance(raw_messages, list) or not raw_messages:
        return None

    def sort_key(msg: Dict[str, Any]) -> Tuple[float, str]:
        ts = parse_iso_timestamp(msg.get("created_at")) or 0.0
        mid = str(msg.get("uuid") or "")
        return (ts, mid)

    raw_messages_sorted = sorted(raw_messages, key=sort_key)

    mapping: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []

    for idx, message in enumerate(raw_messages_sorted):
        sender = str(message.get("sender") or "").lower().strip()
        role = ROLE_MAP.get(sender)
        if role is None:
            continue

        text = extract_message_text(message)
        if not text:
            continue

        node_id = str(message.get("uuid") or f"{conv_id}_m{idx}")
        create_time = parse_iso_timestamp(message.get("created_at"))

        parent_id = ordered_ids[-1] if ordered_ids else None
        mapping[node_id] = {
            "id": node_id,
            "parent": parent_id,
            "children": [],
            "message": {
                "id": node_id,
                "author": {"role": role},
                "content": {"parts": [text]},
                "create_time": create_time,
            },
        }

        if parent_id and parent_id in mapping:
            mapping[parent_id]["children"].append(node_id)

        ordered_ids.append(node_id)

    if not mapping:
        return None

    first_msg_ts = mapping[ordered_ids[0]]["message"].get("create_time")
    last_msg_ts = mapping[ordered_ids[-1]]["message"].get("create_time")

    out = {
        "id": conv_id,
        "title": title,
        "create_time": conv_created or first_msg_ts,
        "update_time": conv_updated or last_msg_ts,
        "mapping": mapping,
    }

    account = conv.get("account")
    summary = conv.get("summary")
    if isinstance(account, dict) or summary:
        out["claude_metadata"] = {
            "account_uuid": account.get("uuid") if isinstance(account, dict) else None,
            "summary": summary,
        }

    return out


def normalize_export(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Claude export must be a list of conversations")

    out: List[Dict[str, Any]] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        normalized = normalize_conversation(conv)
        if normalized:
            out.append(normalized)
    return out


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_claude_input(input_path: Path) -> Tuple[Any, Dict[str, Any]]:
    input_path = input_path.resolve()

    if input_path.is_dir():
        conv_path = input_path / "conversations.json"
        if not conv_path.exists():
            raise FileNotFoundError(f"Directory input must contain conversations.json: {input_path}")

        raw = _load_json_or_jsonl(conv_path)
        metadata: Dict[str, Any] = {
            "input_kind": "directory",
            "input_path": str(input_path).replace("\\", "/"),
            "conversations_source": str(conv_path).replace("\\", "/"),
            "conversations_source_sha256": sha256_file(conv_path),
            "sidecars": {},
        }

        for sidecar_name in ("projects", "memories", "users"):
            sidecar_path = input_path / f"{sidecar_name}.json"
            if not sidecar_path.exists():
                continue

            try:
                sidecar_data = _load_json_or_jsonl(sidecar_path)
                sidecar_count = len(sidecar_data) if isinstance(sidecar_data, list) else None
            except Exception as exc:  # noqa: BLE001
                sidecar_count = None
                metadata["sidecars"][sidecar_name] = {
                    "path": str(sidecar_path).replace("\\", "/"),
                    "sha256": sha256_file(sidecar_path),
                    "count": sidecar_count,
                    "load_error": str(exc),
                }
                continue

            metadata["sidecars"][sidecar_name] = {
                "path": str(sidecar_path).replace("\\", "/"),
                "sha256": sha256_file(sidecar_path),
                "count": sidecar_count,
            }

        if isinstance(raw, list):
            metadata["raw_conversation_count"] = len(raw)
        return raw, metadata

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    raw = _load_json_or_jsonl(input_path)
    metadata = {
        "input_kind": "file",
        "input_path": str(input_path).replace("\\", "/"),
        "conversations_source": str(input_path).replace("\\", "/"),
        "conversations_source_sha256": sha256_file(input_path),
        "sidecars": {},
        "raw_conversation_count": len(raw) if isinstance(raw, list) else None,
    }
    return raw, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Claude export to ChatGPT-style mapping")
    parser.add_argument("--input", required=True, help="Path to Claude export JSON/JSONL file or export directory")
    parser.add_argument("--output", required=True, help="Path for normalized JSON")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional metadata sidecar path (default: <output>.meta.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    raw, metadata = load_claude_input(input_path)
    normalized = normalize_export(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f)

    metadata_output = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".meta.json")
    metadata["normalized_count"] = len(normalized)
    metadata["normalized_output"] = str(output_path).replace("\\", "/")
    with metadata_output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Normalized conversations: {len(normalized)}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
