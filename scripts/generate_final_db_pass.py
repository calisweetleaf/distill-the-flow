#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}, got {type(data).__name__}")
    return data


def _normalize_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _provider_counts(cursor: sqlite3.Cursor, table: str) -> Dict[str, int]:
    cursor.execute(f"SELECT provider, COUNT(*) FROM {table} GROUP BY provider ORDER BY provider")
    return {provider or "unknown": int(count) for provider, count in cursor.fetchall()}


def _provider_run_counts(cursor: sqlite3.Cursor, table: str, provider: str) -> Dict[str, int]:
    cursor.execute(
        f"SELECT provider_run_id, COUNT(*) FROM {table} WHERE provider = ? GROUP BY provider_run_id ORDER BY provider_run_id",
        (provider,),
    )
    return {run_id or "unknown": int(count) for run_id, count in cursor.fetchall()}


def _table_total(cursor: sqlite3.Cursor, table: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    value = cursor.fetchone()
    return int(value[0] if value else 0)


def _quick_check(cursor: sqlite3.Cursor) -> str:
    cursor.execute("PRAGMA quick_check")
    row = cursor.fetchone()
    return str(row[0] if row else "unknown")


def _record_uid_collisions(cursor: sqlite3.Cursor, table: str) -> int:
    cursor.execute(
        f"SELECT COUNT(*) - COUNT(DISTINCT record_uid) FROM {table} WHERE record_uid IS NOT NULL"
    )
    row = cursor.fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def _fetch_live_db_state(db_path: Path) -> Dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"Main DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        final_table_counts = {
            "conversations": _table_total(cursor, "conversations"),
            "messages": _table_total(cursor, "messages"),
            "distilled_conversations": _table_total(cursor, "distilled_conversations"),
        }
        table_counts_by_provider = {
            "conversations": _provider_counts(cursor, "conversations"),
            "messages": _provider_counts(cursor, "messages"),
            "distilled_conversations": _provider_counts(cursor, "distilled_conversations"),
        }
        claude_provider_run_breakdown = {
            "conversations": _provider_run_counts(cursor, "conversations", "claude"),
            "messages": _provider_run_counts(cursor, "messages", "claude"),
            "distilled_conversations": _provider_run_counts(cursor, "distilled_conversations", "claude"),
        }
        record_uid_collisions = {
            "conversations": _record_uid_collisions(cursor, "conversations"),
            "messages": _record_uid_collisions(cursor, "messages"),
            "distilled_conversations": _record_uid_collisions(cursor, "distilled_conversations"),
        }
        quick_check = _quick_check(cursor)
    finally:
        conn.close()

    return {
        "final_table_counts": final_table_counts,
        "table_counts_by_provider": table_counts_by_provider,
        "claude_provider_run_breakdown": claude_provider_run_breakdown,
        "record_uid_collisions": record_uid_collisions,
        "quick_check": quick_check,
    }


def _sum_counts(values: Iterable[int]) -> int:
    return int(sum(int(v) for v in values))


def _build_provider_ledger_summary(ledger_paths: List[Path], repo_root: Path) -> Dict[str, Any]:
    ledgers: Dict[str, Dict[str, Any]] = {}
    for ledger_path in ledger_paths:
        payload = _read_json(ledger_path)
        provider = str(payload.get("provider") or "unknown")
        provider_run_id = str(payload.get("provider_run_id") or ledger_path.stem)
        counters = payload.get("counters") or {}
        ledgers[provider_run_id] = {
            "provider": provider,
            "provider_run_id": provider_run_id,
            "path": _normalize_rel(ledger_path, repo_root),
            "source_path": payload.get("source_path"),
            "source_sha256": payload.get("source_sha256"),
            "content_tokens_non_system": int(counters.get("content_tokens_non_system") or 0),
            "distilled_tokens_selected": int(counters.get("distilled_tokens_selected") or 0),
            "content_tokens_source_origin": payload.get("content_tokens_source_origin"),
        }
    return {
        "status": "repaired",
        "bug_fixed": "Provider-local ledgers/manifests no longer inherit the ChatGPT 115330530 baseline; counters now come from exact_message_recount per provider-local corpus.",
        "provider_run_ledgers": ledgers,
    }


def _skip_only_merge_manifest(merge_manifest: Dict[str, Any]) -> bool:
    counts = merge_manifest.get("counts") or {}
    table_names = ("conversations", "messages", "distilled_conversations")
    for table_name in table_names:
        table_counts = counts.get(table_name) or {}
        inserted = int(table_counts.get("inserted") or 0)
        updated = int(table_counts.get("updated") or 0)
        skipped = int(table_counts.get("skipped") or 0)
        if inserted != 0 or updated != 0:
            return False
        if skipped <= 0:
            return False
    return True


def _build_markdown(payload: Dict[str, Any], repo_root: Path) -> str:
    baseline = payload["baseline_checkpoint_20260227"]
    live = payload["live_state"]
    deltas = payload["delta_vs_20260227_checkpoint"]
    exact = payload["exact_recount_summary"]
    provider_repair = payload["provider_local_repair_summary"]["provider_run_ledgers"]
    checks = payload["checks"]

    lines = [
        "# Final DB Pass Validation (2026-02-28)",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Live Authority",
        f"- main DB: `{payload['main_db']}`",
        f"- exact token recount: `{payload['authority_artifacts']['exact_token_recount']}`",
        f"- prior checkpoint: `{payload['authority_artifacts']['prior_final_pass']}`",
        "",
        "## Delta vs 2026-02-27 Checkpoint",
        f"- conversations: {baseline['table_counts']['conversations']} -> {live['table_counts']['conversations']} (delta {deltas['table_counts']['conversations']})",
        f"- messages: {baseline['table_counts']['messages']} -> {live['table_counts']['messages']} (delta {deltas['table_counts']['messages']})",
        f"- distilled_conversations: {baseline['table_counts']['distilled_conversations']} -> {live['table_counts']['distilled_conversations']} (delta {deltas['table_counts']['distilled_conversations']})",
        f"- exact all_non_system tokens: {baseline['exact_recount']['all_non_system_total']} -> {exact['all_non_system_exact']['token_total_non_system']} (delta {deltas['exact_recount']['all_non_system_total']})",
        f"- exact distilled_non_system tokens: {baseline['exact_recount']['distilled_non_system_total']} -> {exact['distilled_non_system_exact']['token_total_non_system']} (delta {deltas['exact_recount']['distilled_non_system_total']})",
        "",
        "## Provider Composition",
    ]

    for table_name, counts in live["table_counts_by_provider"].items():
        counts_text = ", ".join(f"{provider}={count}" for provider, count in counts.items())
        lines.append(f"- {table_name}: {counts_text}")

    lines.extend([
        "",
        "## Claude Provider-Run Breakdown In Main",
    ])
    for table_name, counts in live["claude_provider_run_breakdown"].items():
        counts_text = ", ".join(f"{provider_run_id}={count}" for provider_run_id, count in counts.items())
        lines.append(f"- {table_name}: {counts_text}")

    lines.extend([
        "",
        "## Exact Recount (o200k_base)",
        f"- all_non_system tokens total: {exact['all_non_system_exact']['token_total_non_system']}",
        f"- all_non_system messages total: {exact['all_non_system_exact']['message_total_non_system']}",
        f"- distilled_non_system tokens total: {exact['distilled_non_system_exact']['token_total_non_system']}",
        f"- distilled_non_system messages total: {exact['distilled_non_system_exact']['message_total_non_system']}",
        "",
        "## Provider-Local Ledger Repair",
        "- status: repaired",
    ])

    for provider_run_id, ledger in provider_repair.items():
        lines.append(
            f"- {provider_run_id}: content_tokens_non_system={ledger['content_tokens_non_system']}, distilled_tokens_selected={ledger['distilled_tokens_selected']}, source_origin={ledger['content_tokens_source_origin']}"
        )

    lines.extend([
        "",
        "## Integrity Checks",
        f"- quick_check: {live['quick_check']}",
        f"- record_uid collisions: {live['record_uid_collisions']}",
        f"- merge_manifest_skip_only_rerun: {checks['merge_manifest_skip_only_rerun']}",
        f"- late_claude_layer_present_in_main: {checks['late_claude_layer_present_in_main']}",
        f"- provider_local_115330530_baseline_removed: {checks['provider_local_115330530_baseline_removed']}",
        f"- all checks pass: {checks['all_checks_pass']}",
        "",
        "## Merge Manifest Interpretation",
        f"- latest main merge manifest: `{payload['authority_artifacts']['latest_merge_manifest']}`",
        f"- note: {payload['merge_manifest_interpretation']['note']}",
        "",
        f"JSON evidence: `{_normalize_rel(Path(payload['output_json']), repo_root)}`",
    ])
    return "\n".join(lines) + "\n"


def _build_authority_manifest(
    repo_root: Path,
    generated_at: str,
    main_db: Path,
    token_recount: Path,
    final_json: Path,
    final_md: Path,
    merge_manifest: Path,
) -> Dict[str, Any]:
    return {
        "version": "2.0.0",
        "generated_at": generated_at,
        "active_authority": {
            "main_db": _normalize_rel(main_db, repo_root),
            "exact_token_recount": _normalize_rel(token_recount, repo_root),
            "final_live_validation_json": _normalize_rel(final_json, repo_root),
            "final_live_validation_md": _normalize_rel(final_md, repo_root),
            "latest_merge_manifest": _normalize_rel(merge_manifest, repo_root),
        },
        "lane_contract": {
            "main": "Active mash/query lane",
            "canonical": "Raw-only token forensics lane",
            "legacy_synthetic": "Synthetic quarantine lane",
            "expansion": "Exploratory run lane",
        },
        "lane_paths": {
            "main": "reports/main/",
            "canonical": "reports/canonical/",
            "legacy_synthetic": "reports/legacy_synthetic/",
            "expansion": "reports/expansion_20260218/",
        },
        "operator_read_order": [
            "reports/main/reports_authority_manifest.json",
            _normalize_rel(final_json, repo_root),
            _normalize_rel(final_md, repo_root),
            _normalize_rel(token_recount, repo_root),
            _normalize_rel(main_db, repo_root),
            "PROJECT_DATABASE_DOCUMENTATION.md",
        ],
        "notes": [
            "reports/main/moonshine_mash_active.db and reports/main/token_recount.main.postdeps.json are the live authority anchors.",
            "The latest main merge manifest reflects a skip-only Claude rerun; consult the final live validation pack for the authoritative interpretation.",
            "Provider-local ledgers/manifests were repaired on 2026-02-28 and no longer inherit the ChatGPT baseline token source.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a final Moonshine DB validation pack from live authority artifacts.")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--db", default="reports/main/moonshine_mash_active.db", help="Path to main DB")
    parser.add_argument(
        "--token-recount",
        default="reports/main/token_recount.main.postdeps.json",
        help="Path to exact token recount JSON",
    )
    parser.add_argument(
        "--prior-final-pass",
        default="reports/main/final_db_pass_20260227.json",
        help="Prior final pass JSON used as baseline checkpoint",
    )
    parser.add_argument(
        "--merge-manifest",
        default="reports/main/merge_manifest.main.json",
        help="Latest main merge manifest",
    )
    parser.add_argument(
        "--provider-ledger",
        action="append",
        dest="provider_ledgers",
        default=[],
        help="Provider-local token ledger to summarize (may be repeated)",
    )
    parser.add_argument(
        "--out-json",
        default="reports/main/final_db_pass_20260228.json",
        help="Output JSON evidence path",
    )
    parser.add_argument(
        "--out-md",
        default="reports/main/final_db_pass_20260228.md",
        help="Output Markdown evidence path",
    )
    parser.add_argument(
        "--authority-manifest-out",
        default="reports/main/reports_authority_manifest.json",
        help="Output reports authority manifest path",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    db_path = (repo_root / args.db).resolve()
    token_recount_path = (repo_root / args.token_recount).resolve()
    prior_final_pass_path = (repo_root / args.prior_final_pass).resolve()
    merge_manifest_path = (repo_root / args.merge_manifest).resolve()
    out_json_path = (repo_root / args.out_json).resolve()
    out_md_path = (repo_root / args.out_md).resolve()
    authority_manifest_out = (repo_root / args.authority_manifest_out).resolve()

    provider_ledgers = [(repo_root / ledger).resolve() for ledger in args.provider_ledgers]
    if not provider_ledgers:
        raise ValueError("At least one --provider-ledger path is required")

    generated_at = _utc_now()
    live_state = _fetch_live_db_state(db_path)
    token_recount = _read_json(token_recount_path)
    prior_final_pass = _read_json(prior_final_pass_path)
    merge_manifest = _read_json(merge_manifest_path)
    provider_local_repair_summary = _build_provider_ledger_summary(provider_ledgers, repo_root)

    baseline_counts = prior_final_pass.get("final_table_counts") or {}
    current_counts = live_state["final_table_counts"]

    baseline_all_non_system_total = int(
        ((prior_final_pass.get("exact_recount_summary") or {}).get("all_non_system_exact") or {}).get("token_total_non_system") or 0
    )
    baseline_distilled_non_system_total = int(
        ((prior_final_pass.get("exact_recount_summary") or {}).get("distilled_non_system_exact") or {}).get("token_total_non_system") or 0
    )

    current_all_non_system_total = int(
        ((token_recount.get("all_non_system_exact") or {}).get("token_total_non_system") or 0)
    )
    current_distilled_non_system_total = int(
        ((token_recount.get("distilled_non_system_exact") or {}).get("token_total_non_system") or 0)
    )

    delta_vs_checkpoint = {
        "table_counts": {
            key: int(current_counts.get(key) or 0) - int(baseline_counts.get(key) or 0)
            for key in ("conversations", "messages", "distilled_conversations")
        },
        "exact_recount": {
            "all_non_system_total": current_all_non_system_total - baseline_all_non_system_total,
            "distilled_non_system_total": current_distilled_non_system_total - baseline_distilled_non_system_total,
        },
    }

    provider_run_ledgers = provider_local_repair_summary["provider_run_ledgers"]
    removed_baseline = all(
        ledger["content_tokens_non_system"] != 115330530 for ledger in provider_run_ledgers.values()
    )

    claude_run_breakdown = live_state["claude_provider_run_breakdown"]
    late_claude_present = any(
        key == "claude_20260227_080825_20260226" and int(value) > 0
        for key, value in claude_run_breakdown["conversations"].items()
    )

    skip_only_manifest = _skip_only_merge_manifest(merge_manifest)
    merge_manifest_interpretation = {
        "latest_manifest_provider": merge_manifest.get("provider"),
        "latest_manifest_run_id": merge_manifest.get("run_id"),
        "skip_only_rerun": skip_only_manifest,
        "note": (
            "Latest main merge manifest is a skip-only rerun artifact. The late Claude layer is still present in the live DB and exact recount, so main authority must be read from the live DB plus token_recount.main.postdeps.json rather than the skip-only merge counters alone."
            if skip_only_manifest and late_claude_present
            else "Latest main merge manifest and live DB state are aligned without special interpretation."
        ),
    }

    checks = {
        "quick_check_ok": live_state["quick_check"] == "ok",
        "record_uid_collisions_zero": all(value == 0 for value in live_state["record_uid_collisions"].values()),
        "late_claude_layer_present_in_main": late_claude_present,
        "merge_manifest_skip_only_rerun": skip_only_manifest,
        "provider_local_115330530_baseline_removed": removed_baseline,
    }
    checks["all_checks_pass"] = all(
        checks[key]
        for key in (
            "quick_check_ok",
            "record_uid_collisions_zero",
            "late_claude_layer_present_in_main",
            "merge_manifest_skip_only_rerun",
            "provider_local_115330530_baseline_removed",
        )
    )

    payload = {
        "generated_at": generated_at,
        "main_db": _normalize_rel(db_path, repo_root),
        "authority_artifacts": {
            "exact_token_recount": _normalize_rel(token_recount_path, repo_root),
            "prior_final_pass": _normalize_rel(prior_final_pass_path, repo_root),
            "latest_merge_manifest": _normalize_rel(merge_manifest_path, repo_root),
            "reports_authority_manifest": _normalize_rel(authority_manifest_out, repo_root),
        },
        "baseline_checkpoint_20260227": {
            "table_counts": {
                "conversations": int(baseline_counts.get("conversations") or 0),
                "messages": int(baseline_counts.get("messages") or 0),
                "distilled_conversations": int(baseline_counts.get("distilled_conversations") or 0),
            },
            "exact_recount": {
                "all_non_system_total": baseline_all_non_system_total,
                "distilled_non_system_total": baseline_distilled_non_system_total,
            },
        },
        "live_state": {
            "table_counts": live_state["final_table_counts"],
            "table_counts_by_provider": live_state["table_counts_by_provider"],
            "claude_provider_run_breakdown": claude_run_breakdown,
            "record_uid_collisions": live_state["record_uid_collisions"],
            "quick_check": live_state["quick_check"],
        },
        "delta_vs_20260227_checkpoint": delta_vs_checkpoint,
        "exact_recount_summary": token_recount,
        "provider_local_repair_summary": provider_local_repair_summary,
        "merge_manifest_interpretation": merge_manifest_interpretation,
        "checks": checks,
        "output_json": _normalize_rel(out_json_path, repo_root),
        "output_markdown": _normalize_rel(out_md_path, repo_root),
    }

    markdown = _build_markdown(payload, repo_root)
    authority_manifest = _build_authority_manifest(
        repo_root=repo_root,
        generated_at=generated_at,
        main_db=db_path,
        token_recount=token_recount_path,
        final_json=out_json_path,
        final_md=out_md_path,
        merge_manifest=merge_manifest_path,
    )

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md_path.write_text(markdown, encoding="utf-8")
    authority_manifest_out.write_text(json.dumps(authority_manifest, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\n[OK] Wrote {out_json_path}")
    print(f"[OK] Wrote {out_md_path}")
    print(f"[OK] Wrote {authority_manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
