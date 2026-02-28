#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ijson
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "02-14-26-ChatGPT" / "conversations.json"
REPORTS_DIR = ROOT / "reports"
OUT_JSON = REPORTS_DIR / "token_forensics.json"
OUT_MD = REPORTS_DIR / "token_forensics.md"


def month_key(ts) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return f"{dt.year:04d}-{dt.month:02d}"
    except Exception:
        return "unknown"


def iter_conversations(path: Path) -> Iterable[dict]:
    with path.open("rb") as f:
        for conv in ijson.items(f, "item"):
            if isinstance(conv, dict):
                yield conv


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("o200k_base")

    total_conversations = 0
    total_messages = 0
    total_tokens = 0
    total_chars = 0
    total_words = 0

    role_counts = Counter()
    conv_token_totals: Dict[str, int] = defaultdict(int)
    month_tokens: Dict[str, int] = defaultdict(int)
    month_messages: Dict[str, int] = defaultdict(int)

    for conv in iter_conversations(INPUT_PATH):
        total_conversations += 1
        conv_id = conv.get("id", f"conv_{total_conversations}")
        conv_create_time = conv.get("create_time")

        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            continue

        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue

            author = msg.get("author") or {}
            role = author.get("role", "unknown")
            if role == "system":
                continue

            content = msg.get("content") or {}
            text = ""
            if isinstance(content.get("parts"), list):
                text = "\n".join(str(p) for p in content.get("parts") if p)
            elif isinstance(content, str):
                text = content
            text = text.strip()
            if not text:
                continue

            msg_time = msg.get("create_time") or conv_create_time
            mk = month_key(msg_time)

            tok = len(enc.encode(text, disallowed_special=()))
            ch = len(text)
            wd = len(text.split())

            total_messages += 1
            total_tokens += tok
            total_chars += ch
            total_words += wd

            role_counts[role] += 1
            conv_token_totals[conv_id] += tok
            month_tokens[mk] += tok
            month_messages[mk] += 1

    avg_tokens_per_message = (total_tokens / total_messages) if total_messages else 0.0
    avg_tokens_per_conversation = (total_tokens / total_conversations) if total_conversations else 0.0

    top_conversations = sorted(conv_token_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    monthly = sorted(month_tokens.keys())

    report = {
        "report_version": "2.1.0-real-export-streaming",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source_file": str(INPUT_PATH),
            "source_type": "ChatGPT export conversations list",
            "tokenizer": "o200k_base",
            "parse_mode": "streaming_ijson",
            "includes_json_structure_tokens": False,
            "scope": "message content only (non-system)"
        },
        "dataset_summary": {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_tokens_per_message": round(avg_tokens_per_message, 2),
            "avg_tokens_per_conversation": round(avg_tokens_per_conversation, 2)
        },
        "role_distribution": dict(role_counts),
        "monthly_distribution": [
            {
                "month_utc": m,
                "tokens": month_tokens[m],
                "messages": month_messages[m]
            }
            for m in monthly
        ],
        "top_conversations_by_tokens": [
            {"conversation_id": cid, "tokens": tok}
            for cid, tok in top_conversations
        ]
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: List[str] = [
        "# Token Forensics Report (Real ChatGPT Export)",
        "",
        f"**Generated:** {report['generated_at']}",
        "**Report Version:** 2.1.0-real-export-streaming",
        "",
        "## Provenance",
        "",
        f"- Source file: `{report['provenance']['source_file']}`",
        "- Scope: message content only (non-system)",
        "- Tokenizer: `o200k_base`",
        "- Parse mode: streaming (`ijson`)",
        "- This report was computed only from your real ChatGPT export.",
        "",
        "## Executive Summary",
        "",
        f"- Total Conversations: **{total_conversations:,}**",
        f"- Total Messages: **{total_messages:,}**",
        f"- Total Tokens: **{total_tokens:,}**",
        f"- Avg Tokens/Message: **{avg_tokens_per_message:.2f}**",
        f"- Avg Tokens/Conversation: **{avg_tokens_per_conversation:.2f}**",
        "",
        "## Role Distribution",
        "",
        "| Role | Messages |",
        "|---|---:|",
    ]

    for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {role} | {count:,} |")

    lines.extend([
        "",
        "## Monthly Token Distribution (UTC)",
        "",
        "| Month | Tokens | Messages |",
        "|---|---:|---:|",
    ])

    for m in monthly:
        lines.append(f"| {m} | {month_tokens[m]:,} | {month_messages[m]:,} |")

    lines.extend([
        "",
        "## Top Conversations by Tokens",
        "",
        "| Conversation ID | Tokens |",
        "|---|---:|",
    ])

    for cid, tok in top_conversations:
        lines.append(f"| `{cid}` | {tok:,} |")

    lines.extend([
        "",
        "---",
        "",
        "This file intentionally replaces mixed synthetic-source numbers.",
    ])

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {OUT_JSON}")
    print(f"[OK] Wrote {OUT_MD}")
    print(f"[OK] total_tokens={total_tokens:,} total_messages={total_messages:,} total_conversations={total_conversations:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
