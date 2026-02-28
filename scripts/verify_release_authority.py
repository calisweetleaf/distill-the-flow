from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Moonshine release authority artifacts.")
    parser.add_argument("--db", required=True, help="Path to moonshine_mash_active.db")
    parser.add_argument("--recount", required=True, help="Path to token_recount.main.postdeps.json")
    parser.add_argument("--parquet", required=True, help="Path to canonical token_row_metrics.raw.parquet")
    return parser.parse_args()


def query_provider_counts(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT provider, COUNT(*) FROM {table} GROUP BY provider ORDER BY provider"
    ).fetchall()
    return {provider: count for provider, count in rows}


def query_provider_run_counts(conn: sqlite3.Connection, table: str, provider: str) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT provider_run_id, COUNT(*) FROM {table} WHERE provider = ? GROUP BY provider_run_id ORDER BY provider_run_id",
        (provider,),
    ).fetchall()
    return {provider_run_id: count for provider_run_id, count in rows}


def inspect_parquet(parquet_path: Path) -> dict:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - operational script
        return {"status": "unavailable", "error": f"pandas import failed: {exc}"}

    df = pd.read_parquet(parquet_path)
    return {
        "status": "ok",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "column_count": int(len(df.columns)),
    }


def main() -> int:
    args = parse_args()
    db_path = Path(args.db).resolve()
    recount_path = Path(args.recount).resolve()
    parquet_path = Path(args.parquet).resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not recount_path.exists():
        raise FileNotFoundError(f"Recount not found: {recount_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    recount = json.loads(recount_path.read_text(encoding="utf-8"))

    with sqlite3.connect(db_path) as conn:
        db_counts = {
            "conversations": query_provider_counts(conn, "conversations"),
            "messages": query_provider_counts(conn, "messages"),
            "distilled_conversations": query_provider_counts(conn, "distilled_conversations"),
        }
        claude_run_counts = {
            "conversations": query_provider_run_counts(conn, "conversations", "claude"),
            "messages": query_provider_run_counts(conn, "messages", "claude"),
            "distilled_conversations": query_provider_run_counts(conn, "distilled_conversations", "claude"),
        }
        quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]

    recount_counts = recount.get("table_counts_by_provider", {})
    counts_match = db_counts == recount_counts

    output = {
        "db_path": db_path.as_posix(),
        "recount_path": recount_path.as_posix(),
        "parquet_path": parquet_path.as_posix(),
        "db_counts": db_counts,
        "recount_counts": recount_counts,
        "counts_match": counts_match,
        "claude_run_counts": claude_run_counts,
        "quick_check": quick_check,
        "all_non_system_exact": recount.get("all_non_system_exact", {}),
        "distilled_non_system_exact": recount.get("distilled_non_system_exact", {}),
        "parquet": inspect_parquet(parquet_path),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - operational script
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        raise
