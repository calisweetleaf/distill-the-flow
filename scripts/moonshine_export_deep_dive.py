#!/usr/bin/env python3
"""Generate deeper analytics from a Moonshine SQLite corpus database.

This script is non-invasive: it only reads from the provided SQLite DB and
writes JSON/Markdown artifacts to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class DeepDiveConfig:
    db_path: Path
    out_dir: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _columns(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows]


def _fetch_dicts(con: sqlite3.Connection, query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    cur = con.execute(query, params)
    cols = [str(d[0]) for d in cur.description or []]
    out: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        out.append({k: row[i] for i, k in enumerate(cols)})
    return out


def _safe_avg_expr(existing_cols: Iterable[str], col: str, alias: str) -> str:
    if col in existing_cols:
        return f"ROUND(AVG({col}), 6) AS {alias}"
    return f"NULL AS {alias}"


def _safe_sum_expr(existing_cols: Iterable[str], col: str, alias: str) -> str:
    if col in existing_cols:
        return f"COALESCE(SUM({col}), 0) AS {alias}"
    return f"0 AS {alias}"


def build_deep_dive(config: DeepDiveConfig) -> Dict[str, Any]:
    if not config.db_path.exists():
        raise FileNotFoundError(f"Database not found: {config.db_path}")

    con = sqlite3.connect(str(config.db_path))
    con.row_factory = sqlite3.Row

    try:
        table_names = [
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        ]

        result: Dict[str, Any] = {
            "report_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "database_path": str(config.db_path),
            "database_sha256": _sha256(config.db_path),
            "tables": table_names,
            "table_counts": {},
            "period_rollup": [],
            "topic_rollup": [],
            "tone_rollup": [],
            "topic_tone_matrix": [],
            "high_correction_density": [],
            "high_signal_candidates": [],
            "dpo_potential_by_topic": [],
            "distilled_quality_tiers": [],
            "monthly_top_conversations": {},
        }

        for table in table_names:
            try:
                cnt = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                result["table_counts"][table] = int(cnt)
            except sqlite3.Error:
                result["table_counts"][table] = None

        if not _table_exists(con, "conversations"):
            return result

        conv_cols = _columns(con, "conversations")

        # Period rollup
        period_query = f"""
        SELECT
            period,
            COUNT(*) AS conversations,
            {_safe_sum_expr(conv_cols, 'total_tokens', 'total_tokens')},
            {_safe_avg_expr(conv_cols, 'information_gain', 'avg_information_gain')},
            {_safe_avg_expr(conv_cols, 'malicious_compliance', 'avg_malicious_compliance')},
            {_safe_avg_expr(conv_cols, 'user_entropy', 'avg_user_entropy')},
            {_safe_avg_expr(conv_cols, 'token_ratio', 'avg_token_ratio')}
        FROM conversations
        GROUP BY period
        ORDER BY period
        """
        result["period_rollup"] = _fetch_dicts(con, period_query)

        # Topic rollup
        topic_query = f"""
        SELECT
            topic_primary,
            COUNT(*) AS conversations,
            {_safe_sum_expr(conv_cols, 'total_tokens', 'total_tokens')},
            {_safe_avg_expr(conv_cols, 'information_gain', 'avg_information_gain')},
            {_safe_avg_expr(conv_cols, 'correction_events', 'avg_correction_events')}
        FROM conversations
        GROUP BY topic_primary
        ORDER BY conversations DESC
        LIMIT 30
        """
        result["topic_rollup"] = _fetch_dicts(con, topic_query)

        # Tone rollup
        tone_query = f"""
        SELECT
            tone_cluster,
            COUNT(*) AS conversations,
            {_safe_avg_expr(conv_cols, 'information_gain', 'avg_information_gain')},
            {_safe_avg_expr(conv_cols, 'malicious_compliance', 'avg_malicious_compliance')}
        FROM conversations
        GROUP BY tone_cluster
        ORDER BY conversations DESC
        """
        result["tone_rollup"] = _fetch_dicts(con, tone_query)

        # Topic x Tone matrix
        result["topic_tone_matrix"] = _fetch_dicts(
            con,
            """
            SELECT
                topic_primary,
                tone_cluster,
                COUNT(*) AS conversations
            FROM conversations
            GROUP BY topic_primary, tone_cluster
            ORDER BY conversations DESC
            LIMIT 100
            """,
        )

        # High correction density
        if "correction_events" in conv_cols and "total_tokens" in conv_cols:
            result["high_correction_density"] = _fetch_dicts(
                con,
                """
                SELECT
                    conversation_id,
                    title,
                    topic_primary,
                    correction_events,
                    total_tokens,
                    ROUND((correction_events * 1000.0) / NULLIF(total_tokens, 0), 4) AS corrections_per_1k_tokens
                FROM conversations
                WHERE correction_events > 0 AND total_tokens > 0
                ORDER BY corrections_per_1k_tokens DESC, correction_events DESC
                LIMIT 30
                """,
            )

        # High-signal candidates
        if "information_gain" in conv_cols and "malicious_compliance" in conv_cols:
            result["high_signal_candidates"] = _fetch_dicts(
                con,
                """
                SELECT
                    conversation_id,
                    title,
                    topic_primary,
                    information_gain,
                    malicious_compliance,
                    correction_events,
                    total_tokens
                FROM conversations
                WHERE information_gain >= 0.58
                  AND malicious_compliance <= 0.25
                ORDER BY information_gain DESC, malicious_compliance ASC
                LIMIT 50
                """,
            )

        # DPO potential by topic
        if "correction_events" in conv_cols:
            result["dpo_potential_by_topic"] = _fetch_dicts(
                con,
                """
                SELECT
                    topic_primary,
                    COUNT(*) AS conversations_with_corrections,
                    SUM(correction_events) AS total_correction_events,
                    ROUND(AVG(correction_events), 4) AS avg_correction_events
                FROM conversations
                WHERE correction_events > 0
                GROUP BY topic_primary
                ORDER BY total_correction_events DESC
                """,
            )

        # Distilled quality tiers
        if _table_exists(con, "distilled_conversations"):
            dist_cols = _columns(con, "distilled_conversations")
            if "quality_tier" in dist_cols:
                result["distilled_quality_tiers"] = _fetch_dicts(
                    con,
                    """
                    SELECT quality_tier, COUNT(*) AS conversations
                    FROM distilled_conversations
                    GROUP BY quality_tier
                    ORDER BY conversations DESC
                    """,
                )

        # Monthly top conversations by token volume
        if "period" in conv_cols and "total_tokens" in conv_cols:
            periods = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT period FROM conversations WHERE period IS NOT NULL ORDER BY period"
                ).fetchall()
            ]
            for p in periods:
                result["monthly_top_conversations"][str(p)] = _fetch_dicts(
                    con,
                    """
                    SELECT
                        conversation_id,
                        title,
                        topic_primary,
                        total_tokens,
                        information_gain,
                        malicious_compliance,
                        correction_events
                    FROM conversations
                    WHERE period = ?
                    ORDER BY total_tokens DESC
                    LIMIT 10
                    """,
                    (p,),
                )

        return result
    finally:
        con.close()


def to_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Moonshine Deep Dive")
    lines.append("")
    lines.append(f"Generated: {payload.get('generated_at', 'unknown')}")
    lines.append(f"Database: `{payload.get('database_path', '')}`")
    lines.append("")

    lines.append("## Table Counts")
    lines.append("")
    lines.append("| Table | Rows |")
    lines.append("|---|---:|")
    for table, cnt in payload.get("table_counts", {}).items():
        lines.append(f"| {table} | {cnt} |")
    lines.append("")

    def _emit_rows(title: str, rows: List[Dict[str, Any]], limit: int = 10) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("No data.")
            lines.append("")
            return
        keys = list(rows[0].keys())
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in rows[:limit]:
            vals = [str(row.get(k, "")) for k in keys]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    _emit_rows("Period Rollup", payload.get("period_rollup", []), limit=50)
    _emit_rows("Topic Rollup", payload.get("topic_rollup", []), limit=30)
    _emit_rows("Tone Rollup", payload.get("tone_rollup", []), limit=20)
    _emit_rows("Top Correction Density", payload.get("high_correction_density", []), limit=20)
    _emit_rows("High-Signal Candidates", payload.get("high_signal_candidates", []), limit=20)
    _emit_rows("DPO Potential By Topic", payload.get("dpo_potential_by_topic", []), limit=20)
    _emit_rows("Distilled Quality Tiers", payload.get("distilled_quality_tiers", []), limit=20)

    lines.append("## Monthly Top Conversations")
    lines.append("")
    monthly = payload.get("monthly_top_conversations", {})
    if not monthly:
        lines.append("No monthly data.")
    else:
        for period, rows in monthly.items():
            lines.append(f"### {period}")
            lines.append("")
            if not rows:
                lines.append("No rows.")
                lines.append("")
                continue
            keys = list(rows[0].keys())
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("|" + "|".join(["---"] * len(keys)) + "|")
            for row in rows[:10]:
                vals = [str(row.get(k, "")) for k in keys]
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")

    lines.append("---")
    lines.append("Generated by `scripts/moonshine_export_deep_dive.py`.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "moonshine_deep_dive.json"
    md_path = out_dir / "moonshine_deep_dive.md"
    manifest_path = out_dir / "moonshine_deep_dive_manifest.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(to_markdown(payload))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "moonshine_deep_dive.json": {
                "path": str(json_path),
                "sha256": _sha256(json_path),
                "bytes": json_path.stat().st_size,
            },
            "moonshine_deep_dive.md": {
                "path": str(md_path),
                "sha256": _sha256(md_path),
                "bytes": md_path.stat().st_size,
            },
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "manifest": str(manifest_path),
    }


def parse_args() -> DeepDiveConfig:
    parser = argparse.ArgumentParser(description="Moonshine deep dive analytics")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports") / "moonshine_corpus.db",
        help="Path to moonshine SQLite DB",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "moonshine_deep_dive",
        help="Output directory for deep dive artifacts",
    )
    args = parser.parse_args()
    return DeepDiveConfig(db_path=args.db, out_dir=args.out_dir)


def main() -> int:
    cfg = parse_args()
    try:
        payload = build_deep_dive(cfg)
        out = write_outputs(payload, cfg.out_dir)
    except Exception as exc:
        print(f"[ERROR] deep dive generation failed: {exc}")
        return 1

    print("[OK] moonshine deep dive artifacts generated")
    print(f"  - json: {out['json']}")
    print(f"  - markdown: {out['markdown']}")
    print(f"  - manifest: {out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
