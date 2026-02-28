#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TARGET_BYTES = 5 * 1024 * 1024


def parse_token_count_guess(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"final count:\s*([0-9,]+)",
        r"([0-9]{3}(?:,[0-9]{3})+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    raise ValueError("Could not parse token count from token-count-guess.md")


def parse_kimi_total_tokens(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\|\s*Total Tokens\s*\|\s*([^|]+)\|", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else "unknown"


def read_validation_stats(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_monthly_distribution(conn: sqlite3.Connection) -> List[Tuple[str, int, int]]:
    cur = conn.cursor()
    cur.execute("SELECT created_at, total_tokens FROM conversations WHERE created_at IS NOT NULL")
    rows = cur.fetchall()

    buckets: Dict[str, Dict[str, int]] = {}
    for ts, total_tokens in rows:
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            continue
        key = f"{dt.year:04d}-{dt.month:02d}"
        if key not in buckets:
            buckets[key] = {"count": 0, "tokens": 0}
        buckets[key]["count"] += 1
        buckets[key]["tokens"] += int(total_tokens or 0)

    return [(k, v["count"], v["tokens"]) for k, v in sorted(buckets.items())]


def extract_june_chunk(conn: sqlite3.Connection, out_path: Path, target_bytes: int) -> Dict[str, int]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT conversation_id, title, created_at, total_tokens
        FROM conversations
        WHERE created_at IS NOT NULL
        ORDER BY created_at ASC
        """
    )

    june_conversations = []
    for conversation_id, title, created_at, total_tokens in cur.fetchall():
        try:
            dt = datetime.fromtimestamp(float(created_at), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            continue
        if dt.month == 6:
            june_conversations.append(
                {
                    "conversation_id": conversation_id,
                    "title": title or "Untitled",
                    "created_at": dt,
                    "total_tokens": int(total_tokens or 0),
                }
            )

    lines: List[str] = [
        "# June-Centered Corpus Chunk (~5 MB)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Target size (bytes): {target_bytes}",
        f"June conversations available: {len(june_conversations)}",
        "",
        "This chunk is assembled from SQLite `reports/moonshine_corpus.db` conversations/messages where created month is June (UTC).",
        "",
        "---",
        "",
    ]

    current_size = len(("\n".join(lines) + "\n").encode("utf-8"))
    used_conversations = 0
    used_messages = 0

    for conv in june_conversations:
        section_header = [
            f"## Conversation: {conv['title']}",
            "",
            f"- Conversation ID: `{conv['conversation_id']}`",
            f"- Created (UTC): {conv['created_at'].isoformat()}",
            f"- Conversation token estimate: {conv['total_tokens']:,}",
            "",
        ]
        section_text = "\n".join(section_header) + "\n"
        section_bytes = len(section_text.encode("utf-8"))
        if current_size + section_bytes > target_bytes:
            break

        lines.extend(section_header)
        current_size += section_bytes

        cur.execute(
            """
            SELECT role, create_time, text
            FROM messages
            WHERE conversation_id = ?
            ORDER BY create_time ASC
            """,
            (conv["conversation_id"],),
        )

        any_message_added = False
        for role, create_time, text in cur.fetchall():
            role = role or "unknown"
            try:
                ts = datetime.fromtimestamp(float(create_time), tz=timezone.utc).isoformat() if create_time else "unknown"
            except (TypeError, ValueError, OSError):
                ts = "unknown"

            body = (text or "").strip()
            if not body:
                continue

            body = body.replace("\r\n", "\n").replace("\r", "\n")
            body = re.sub(r"\n{3,}", "\n\n", body)
            body = body[:2500]

            block = [
                f"### {role.title()} @ {ts}",
                "",
                body,
                "",
            ]
            block_text = "\n".join(block) + "\n"
            block_bytes = len(block_text.encode("utf-8"))
            if current_size + block_bytes > target_bytes:
                break

            lines.extend(block)
            current_size += block_bytes
            used_messages += 1
            any_message_added = True

        if any_message_added:
            divider = ["---", ""]
            divider_text = "\n".join(divider) + "\n"
            divider_bytes = len(divider_text.encode("utf-8"))
            if current_size + divider_bytes <= target_bytes:
                lines.extend(divider)
                current_size += divider_bytes
            used_conversations += 1

        if current_size >= target_bytes:
            break

    output_text = "\n".join(lines).rstrip() + "\n"
    out_path.write_text(output_text, encoding="utf-8")

    return {
        "size_bytes": len(output_text.encode("utf-8")),
        "conversation_count": used_conversations,
        "message_count": used_messages,
        "june_total_conversations": len(june_conversations),
    }


def write_reconciliation_report(
    out_path: Path,
    validation: Dict,
    token_guess: int,
    kimi_total_tokens: str,
    monthly: List[Tuple[str, int, int]],
    june_chunk_stats: Dict[str, int],
) -> None:
    stats = validation.get("statistics", {})
    val_total_samples = int(stats.get("total_samples", 0))
    val_total_tokens = int(stats.get("total_tokens", 0))
    ratio_guess_vs_validation = (token_guess / val_total_tokens) if val_total_tokens else 0.0

    lines = [
        "# Token Reconciliation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Primary Numbers",
        "",
        f"- `token-count-guess.md` full-file token count (`o200k_base` on raw JSON text): **{token_guess:,}**",
        f"- `reports/validation_manifest.json` token count (validated dataset): **{val_total_tokens:,}**",
        f"- `reports/validation_manifest.json` sample count: **{val_total_samples:,}**",
        f"- `kimi-updates.md` stated corpus tokens: **{kimi_total_tokens}**",
        "",
        "## Why The Numbers Differ",
        "",
        "1. Different unit of analysis:",
        "`token-count-guess.md` counts tokens in the entire raw JSON file text, including keys, punctuation, metadata, and all serialized content.",
        "2. Different dataset scope:",
        "`reports/token_forensics.md` is generated from `token_row_metrics.parquet` validated in `validation_manifest.json`, which currently contains 10,000 rows.",
        "3. Different tokenizers/estimators:",
        "`token-count-guess.md` uses `o200k_base`; validated reports use forensics pipeline token columns and benchmark stats.",
        "4. Different method in Kimi update:",
        "`kimi-updates.md` references Moonshine analyzer output that uses heuristic token estimation (word-based), not raw BPE tokenization.",
        "",
        "## Ratio Check",
        "",
        f"- Full-file count / validated forensics count: **{ratio_guess_vs_validation:.2f}x**",
        "",
        "## Monthly Distribution From `moonshine_corpus.db`",
        "",
        "| Month (UTC) | Conversations | Estimated Tokens |",
        "|---|---:|---:|",
    ]

    for month, conv_count, token_sum in monthly:
        lines.append(f"| {month} | {conv_count:,} | {token_sum:,} |")

    lines.extend(
        [
            "",
            "## June Chunk Export",
            "",
            "- Output: `reports/june_5mb_chunk.md`",
            f"- Target size: {TARGET_BYTES:,} bytes",
            f"- Actual size: {june_chunk_stats['size_bytes']:,} bytes",
            f"- June conversations included: {june_chunk_stats['conversation_count']:,} / {june_chunk_stats['june_total_conversations']:,}",
            f"- Messages included: {june_chunk_stats['message_count']:,}",
            "",
            "## Bottom Line",
            "",
            "Tokens did not disappear. The artifacts are measuring different things (full raw export text vs validated subset vs heuristic conversation estimates).",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    token_guess = parse_token_count_guess(ROOT / "token-count-guess.md")
    kimi_total_tokens = parse_kimi_total_tokens(ROOT / "kimi-updates.md")
    validation = read_validation_stats(REPORTS / "validation_manifest.json")

    conn = sqlite3.connect(str(REPORTS / "moonshine_corpus.db"))
    try:
        monthly = read_monthly_distribution(conn)
        june_chunk_stats = extract_june_chunk(conn, REPORTS / "june_5mb_chunk.md", TARGET_BYTES)
    finally:
        conn.close()

    write_reconciliation_report(
        REPORTS / "token_reconciliation.md",
        validation,
        token_guess,
        kimi_total_tokens,
        monthly,
        june_chunk_stats,
    )

    print("[OK] Wrote reports/token_reconciliation.md")
    print("[OK] Wrote reports/june_5mb_chunk.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
