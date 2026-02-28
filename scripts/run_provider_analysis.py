#!/usr/bin/env python3
"""
run_provider_analysis.py â€” Generic provider analysis runner
============================================================

Normalizes provider-specific export formats into the standard
MoonshineCorpusAnalyzer input format and runs full analysis.

Supported providers:
  deepseek  â€” exports/deepseek/conversations.json
  qwen      â€” exports/qwen/qwen-chat-export-*.json
  claude    â€” exports/claude/claude_conversations.json OR
              exports/claude/<export_folder>/conversations.json (+ sidecars)

Output contract (P1):
  reports/providers/<provider>/<run_id>/
    moonshine_<provider>_<run_id>.db
    token_ledger.<provider>.<run_id>.json
    moonshine_corpus_report.<provider>.<run_id>.md
    moonshine_distillation_manifest.<provider>.<run_id>.json
    visualizations_manifest.<provider>.<run_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path so we can import moonshine_corpus_analyzer
sys.path.insert(0, str(Path(__file__).parent.parent))

from moonshine_corpus_analyzer import MoonshineCorpusAnalyzer


# ---------------------------------------------------------------------------
# Provider normalizers
# ---------------------------------------------------------------------------

class DeepSeekNormalizer:
    """
    Normalizes DeepSeek export (conversations.json) to ChatGPT-compatible format.
    """

    def normalize(self, raw: list) -> list:
        result = []
        for conv in raw:
            normalized = self._normalize_conv(conv)
            if normalized:
                result.append(normalized)
        return result

    def _iso_to_timestamp(self, iso_str: Optional[str]) -> Optional[float]:
        if not iso_str:
            return None
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return None

    def _normalize_conv(self, conv: dict) -> Optional[dict]:
        conv_id = conv.get("id", "")
        title = conv.get("title", "Untitled")
        create_time = self._iso_to_timestamp(conv.get("inserted_at"))
        update_time = self._iso_to_timestamp(conv.get("updated_at"))

        mapping = conv.get("mapping", {})
        if not mapping:
            return None

        normalized_mapping = {}
        for node_id, node_data in mapping.items():
            msg = node_data.get("message")
            if not msg or not isinstance(msg, dict):
                continue

            fragments = msg.get("fragments") or []
            if not fragments:
                continue

            types = {f.get("type", "").upper() for f in fragments}
            if "REQUEST" in types:
                role = "user"
                content_parts = [
                    f["content"]
                    for f in fragments
                    if f.get("type", "").upper() == "REQUEST" and f.get("content")
                ]
            else:
                role = "assistant"
                content_parts = [
                    f["content"]
                    for f in fragments
                    if f.get("type", "").upper() not in ("THINK",) and f.get("content")
                ]
                if not content_parts:
                    content_parts = [f["content"] for f in fragments if f.get("content")]

            text = "\n\n".join(content_parts).strip()
            if not text:
                continue

            node_create_time = self._iso_to_timestamp(msg.get("inserted_at"))

            normalized_mapping[node_id] = {
                "id": node_id,
                "parent": node_data.get("parent"),
                "children": node_data.get("children", []),
                "message": {
                    "id": node_id,
                    "author": {"role": role},
                    "content": {"parts": [text]},
                    "create_time": node_create_time or create_time,
                },
            }

        if not normalized_mapping:
            return None

        return {
            "id": conv_id,
            "title": title,
            "create_time": create_time,
            "update_time": update_time,
            "mapping": normalized_mapping,
        }


class QwenNormalizer:
    """
    Normalizes Qwen export (qwen-chat-export-*.json) to ChatGPT-compatible format.
    """

    def normalize(self, raw: dict) -> list:
        convs = raw.get("data", [])
        result = []
        for conv in convs:
            normalized = self._normalize_conv(conv)
            if normalized:
                result.append(normalized)
        return result

    def _normalize_conv(self, conv: dict) -> Optional[dict]:
        conv_id = conv.get("id", "")
        title = conv.get("title", "Untitled")
        create_time = float(conv.get("created_at") or 0) or None
        update_time = float(conv.get("updated_at") or 0) or None

        chat = conv.get("chat") or {}
        history = chat.get("history") or {}
        messages_dict = history.get("messages") or {}

        if not messages_dict:
            return None

        normalized_mapping = {}
        for msg_id, msg in messages_dict.items():
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            if role == "user":
                text = str(msg.get("content", "")).strip()
            else:
                content_list = msg.get("content_list") or []
                answer_parts = [
                    item.get("content", "")
                    for item in content_list
                    if item.get("phase") == "answer" and item.get("content")
                ]
                if answer_parts:
                    text = "\n\n".join(answer_parts).strip()
                else:
                    text = str(msg.get("content", "")).strip()

            if not text:
                continue

            msg_timestamp = msg.get("timestamp")
            create_msg_time = float(msg_timestamp) if msg_timestamp is not None else create_time

            normalized_mapping[msg_id] = {
                "id": msg_id,
                "parent": msg.get("parentId"),
                "children": msg.get("childrenIds", []),
                "message": {
                    "id": msg_id,
                    "author": {"role": role},
                    "content": {"parts": [text]},
                    "create_time": create_msg_time,
                },
            }

        if not normalized_mapping:
            return None

        return {
            "id": conv_id,
            "title": title,
            "create_time": create_time,
            "update_time": update_time,
            "mapping": normalized_mapping,
        }


# ---------------------------------------------------------------------------
# Claude loader/normalizer bridge
# ---------------------------------------------------------------------------

def _load_claude_module():
    module_path = Path(__file__).parent / "normalize_claude_export.py"
    spec = importlib.util.spec_from_file_location("normalize_claude_export", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Claude normalizer module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_claude_export(export_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Path]:
    module = _load_claude_module()
    raw, metadata = module.load_claude_input(export_path)
    normalized = module.normalize_export(raw)

    source_path = Path(metadata.get("conversations_source") or export_path)
    if not source_path.is_absolute():
        source_path = source_path.resolve()

    metadata = dict(metadata)
    metadata["normalized_count"] = len(normalized)
    return normalized, metadata, source_path


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

def discover_export_path(provider: str) -> Path:
    """Auto-discover export input path for a provider."""
    if provider == "deepseek":
        p = Path("exports/deepseek/conversations.json")
        if p.exists():
            return p

    elif provider == "qwen":
        qwen_dir = Path("exports/qwen")
        if qwen_dir.exists():
            matches = sorted(qwen_dir.glob("qwen-chat-export-*.json"))
            if matches:
                return matches[0]

    elif provider == "claude":
        claude_dir = Path("exports/claude")
        if claude_dir.exists():
            folder_candidates = [p for p in claude_dir.iterdir() if p.is_dir() and (p / "conversations.json").exists()]
            if folder_candidates:
                folder_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return folder_candidates[0]

            legacy = claude_dir / "claude_conversations.json"
            if legacy.exists():
                return legacy

    raise FileNotFoundError(
        f"Could not auto-discover export for provider '{provider}'. "
        "Use --export-path to specify explicitly."
    )


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_analysis(
    provider: str,
    export_path: Path,
    run_id: str,
    output_dir: Path,
    dry_run: bool = False,
):
    """Normalize export, invoke analyzer, produce P1 contract artifacts."""
    print("=" * 60)
    print(f"PROVIDER ANALYSIS â€” {provider.upper()}")
    print("=" * 60)
    print(f"Export  : {export_path}")
    print(f"Run ID  : {run_id}")
    print(f"Output  : {output_dir}")
    print("-" * 60)

    # Step 1: Load + normalize
    print(f"\n[NORMALIZE] Loading {export_path} ...")
    provider_input_manifest: Dict[str, Any] = {
        "provider": provider,
        "run_id": run_id,
        "input_path": str(export_path).replace("\\", "/"),
        "input_kind": "directory" if export_path.is_dir() else "file",
        "normalized_at": datetime.now(timezone.utc).isoformat(),
    }

    source_payload_path: Path = export_path

    if provider == "deepseek":
        with export_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        normalizer = DeepSeekNormalizer()
        normalized = normalizer.normalize(raw)

    elif provider == "qwen":
        with export_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        normalizer = QwenNormalizer()
        normalized = normalizer.normalize(raw)

    elif provider == "claude":
        normalized, claude_meta, source_payload_path = normalize_claude_export(export_path)
        provider_input_manifest["claude_metadata"] = claude_meta

    else:
        with export_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        normalized = raw if isinstance(raw, list) else raw.get("data", [])

    provider_input_manifest["normalized_conversations"] = len(normalized)

    source_hash = None
    if source_payload_path.exists() and source_payload_path.is_file():
        source_hash = sha256_file(source_payload_path)
        provider_input_manifest["source_payload_path"] = str(source_payload_path).replace("\\", "/")
        provider_input_manifest["source_payload_sha256"] = source_hash

    print(f"  Normalized: {len(normalized)} conversations")

    if not normalized:
        print("[WARN] No conversations normalized. Exiting.")
        return None

    if dry_run:
        print("[DRY-RUN] Would run analysis â€” exiting without writing.")
        print(json.dumps(provider_input_manifest, indent=2))
        return None

    # Step 2: write normalized JSON temp + input manifest
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_normalized = output_dir / f"_normalized_{provider}_{run_id}.json"
    with temp_normalized.open("w", encoding="utf-8") as f:
        json.dump(normalized, f)
    print(f"  Normalized JSON written to temp: {temp_normalized}")

    input_manifest_path = output_dir / f"provider_input_manifest.{provider}.{run_id}.json"
    with input_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(provider_input_manifest, f, indent=2)
    print(f"  Input manifest written: {input_manifest_path.name}")

    # Step 3: Run MoonshineCorpusAnalyzer
    print(f"\n[ANALYZE] Running MoonshineCorpusAnalyzer...")
    analyzer = MoonshineCorpusAnalyzer(
        conversations_path=temp_normalized,
        output_dir=output_dir,
        provider=provider,
        provider_run_id=run_id,
        source_sha256=source_hash,
    )
    results = analyzer.analyze()

    def _refresh_contract_artifact(source: Path, target: Path, label: str):
        if not source.exists():
            return
        existed = target.exists()
        shutil.copy2(str(source), str(target))
        action = "refreshed" if existed else "copied"
        print(f"  [P1] {label} {action}: {target.name}")

    # Step 4: Refresh P1 contract artifacts so reruns repair stale provider-local outputs
    generic_db = output_dir / "moonshine_corpus.db"
    target_db = output_dir / f"moonshine_{provider}_{run_id}.db"
    _refresh_contract_artifact(generic_db, target_db, "DB")

    # Step 5: Refresh token ledger contract artifact
    generic_ledger = output_dir / "token_ledger.json"
    target_ledger = output_dir / f"token_ledger.{provider}.{run_id}.json"
    _refresh_contract_artifact(generic_ledger, target_ledger, "Ledger")

    # Step 6: Refresh report contract artifact
    generic_report = output_dir / "moonshine_corpus_report.md"
    target_report = output_dir / f"moonshine_corpus_report.{provider}.{run_id}.md"
    _refresh_contract_artifact(generic_report, target_report, "Report")

    # Step 7: Refresh distillation manifest contract artifact
    generic_manifest = output_dir / "moonshine_distillation_manifest.json"
    target_manifest = output_dir / f"moonshine_distillation_manifest.{provider}.{run_id}.json"
    _refresh_contract_artifact(generic_manifest, target_manifest, "Distillation manifest")

    # Step 8: Write visualizations manifest (stub)
    viz_manifest = output_dir / f"visualizations_manifest.{provider}.{run_id}.json"
    if not viz_manifest.exists():
        viz_data = {
            "version": "1.0.0",
            "provider": provider,
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "visualizations": [],
            "note": "Visualizations not generated in this analysis pass.",
        }
        with viz_manifest.open("w", encoding="utf-8") as f:
            json.dump(viz_data, f, indent=2)
        print(f"  [P1] Visualizations manifest: {viz_manifest.name}")

    # Step 9: Remove temp normalized file
    if temp_normalized.exists():
        temp_normalized.unlink()
        print(f"  [CLEAN] Removed temp: {temp_normalized.name}")

    print(f"\n[OK] Provider analysis complete for {provider}/{run_id}")
    print(f"  Conversations: {results['conversations_analyzed']}")
    print(f"  Messages     : {results['messages_extracted']}")
    print(f"  Distilled    : {results['distilled_conversations']}")
    print(f"  Tokens       : {results['distilled_tokens']:,}")
    print(f"  Output dir   : {output_dir}")

    return {
        "provider": provider,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "db_path": str(target_db),
        "conversations_analyzed": results["conversations_analyzed"],
        "messages_extracted": results["messages_extracted"],
        "distilled_conversations": results["distilled_conversations"],
        "distilled_tokens": results["distilled_tokens"],
        "source_payload_path": str(source_payload_path),
        "source_payload_sha256": source_hash,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Moonshine provider analysis (deepseek, qwen, or claude)."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["deepseek", "qwen", "claude"],
        help="Provider to analyze",
    )
    parser.add_argument(
        "--export-path",
        dest="export_path",
        default=None,
        help="Path to provider export file/folder (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Run ID (auto-generated if omitted: <provider>_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory (default: reports/providers/<provider>/<run_id>/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Normalize only, don't run analysis",
    )

    args = parser.parse_args()
    provider = args.provider.strip().lower()

    if args.export_path:
        export_path = Path(args.export_path)
    else:
        export_path = discover_export_path(provider)

    if not export_path.exists():
        sys.exit(f"[FATAL] Export path not found: {export_path}")

    run_id = args.run_id or f"{provider}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("reports") / "providers" / provider / run_id

    result = run_analysis(provider, export_path, run_id, output_dir, args.dry_run)

    if result is None and not args.dry_run:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
