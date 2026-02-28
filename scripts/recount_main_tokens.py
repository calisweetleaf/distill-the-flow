#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def _get_encoder(name: str):
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding(name)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tiktoken/o200k_base unavailable in current .venv; install tiktoken first"
        ) from exc


def _tok_len(text: str, encoder, cache: Dict[str, int]) -> int:
    hit = cache.get(text)
    if hit is not None:
        return hit
    n = len(encoder.encode(text, disallowed_special=()))
    cache[text] = n
    return n


def _count_all_non_system_tokens(conn: sqlite3.Connection, encoder, cache: Dict[str, int]) -> Dict[str, int]:
    token_by_provider: Dict[str, int] = defaultdict(int)
    msg_by_provider: Dict[str, int] = defaultdict(int)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT provider, role, text
        FROM messages
        WHERE text IS NOT NULL AND text != ''
        """
    )

    for provider, role, text in cur:
        provider = provider or "unknown"
        if (role or "").lower() == "system":
            continue
        token_by_provider[provider] += _tok_len(text, encoder, cache)
        msg_by_provider[provider] += 1

    return {
        "token_by_provider": dict(sorted(token_by_provider.items())),
        "message_count_by_provider": dict(sorted(msg_by_provider.items())),
        "token_total_non_system": int(sum(token_by_provider.values())),
        "message_total_non_system": int(sum(msg_by_provider.values())),
    }


def _count_distilled_non_system_tokens(conn: sqlite3.Connection, encoder, cache: Dict[str, int]) -> Dict[str, int]:
    token_by_provider: Dict[str, int] = defaultdict(int)
    msg_by_provider: Dict[str, int] = defaultdict(int)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.provider, m.role, m.text
        FROM messages m
        JOIN distilled_conversations d
          ON d.conversation_id = m.conversation_id
         AND d.provider = m.provider
        WHERE m.text IS NOT NULL AND m.text != ''
        """
    )

    for provider, role, text in cur:
        provider = provider or "unknown"
        if (role or "").lower() == "system":
            continue
        token_by_provider[provider] += _tok_len(text, encoder, cache)
        msg_by_provider[provider] += 1

    return {
        "token_by_provider": dict(sorted(token_by_provider.items())),
        "message_count_by_provider": dict(sorted(msg_by_provider.items())),
        "token_total_non_system": int(sum(token_by_provider.values())),
        "message_total_non_system": int(sum(msg_by_provider.values())),
    }


def _table_counts(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    cur = conn.cursor()
    out: Dict[str, Dict[str, int]] = {}

    for table in ("conversations", "messages", "distilled_conversations"):
        cur.execute(f"SELECT provider, COUNT(*) FROM {table} GROUP BY provider ORDER BY provider")
        out[table] = {provider: int(count) for provider, count in cur.fetchall()}

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Recount exact tokens from main moonshine DB")
    parser.add_argument(
        "--db",
        default="reports/main/moonshine_mash_active.db",
        help="Path to main moonshine DB",
    )
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help="tiktoken encoding name",
    )
    parser.add_argument(
        "--out",
        default="reports/main/token_recount.main.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    encoder = _get_encoder(args.encoding)
    cache: Dict[str, int] = {}

    conn = sqlite3.connect(str(db_path))
    try:
        tables = _table_counts(conn)
        all_non_system = _count_all_non_system_tokens(conn, encoder, cache)
        distilled_non_system = _count_distilled_non_system_tokens(conn, encoder, cache)
    finally:
        conn.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path).replace("\\", "/"),
        "encoding": args.encoding,
        "cache_size": len(cache),
        "table_counts_by_provider": tables,
        "all_non_system_exact": all_non_system,
        "distilled_non_system_exact": distilled_non_system,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\n[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
