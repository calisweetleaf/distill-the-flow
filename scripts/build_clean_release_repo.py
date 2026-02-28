from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT_FILES = [
    Path(".gitattributes"),
    Path("README.md"),
    Path("WIKI.md"),
    Path("PROJECT_DATABASE_DOCUMENTATION.md"),
    Path("PROJECT_MOONSHINE_UPDATE_1.md"),
    Path("distill-the-flow-filetree.md"),
]

ROOT_REPORT_FILES = [
    Path("reports/CURRENT_REPORTS_FILETREE.md"),
    Path("reports/dedup_clusters.parquet"),
    Path("reports/raw_only_gate_manifest.json"),
    Path("reports/token_forensics.json"),
    Path("reports/token_ledger.json"),
    Path("reports/validation_manifest.json"),
    Path("reports/validation_report.md"),
]

DOCS_ALLOWED_SUFFIXES = {".md"}
FILE_TREE_ALLOWED_SUFFIXES = {".md"}
REPORTS_ALLOWED_SUFFIXES = {".json", ".md", ".db", ".parquet"}
SKIP_SUFFIXES = {".zip"}


@dataclass
class CopyRecord:
    relative_path: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a curated clean release repository for Project Moonshine "
            "from the validated working repo surface."
        )
    )
    parser.add_argument("--source-root", required=True, help="Path to the working repository root")
    parser.add_argument("--target-root", required=True, help="Path to the clean release repository root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def relative_str(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def should_copy_docs(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in DOCS_ALLOWED_SUFFIXES and path.suffix.lower() not in SKIP_SUFFIXES


def should_copy_file_trees(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in FILE_TREE_ALLOWED_SUFFIXES


def should_copy_reports(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in REPORTS_ALLOWED_SUFFIXES and path.suffix.lower() not in SKIP_SUFFIXES


def should_copy_visualizations(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() not in SKIP_SUFFIXES


def copy_file(src: Path, dst: Path, dry_run: bool) -> CopyRecord:
    ensure_exists(src, "source file")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return CopyRecord(relative_path=dst.as_posix(), size_bytes=src.stat().st_size)


def sync_directory(
    source_root: Path,
    target_root: Path,
    relative_dir: Path,
    predicate: Callable[[Path], bool],
    dry_run: bool,
) -> list[CopyRecord]:
    src_dir = source_root / relative_dir
    ensure_exists(src_dir, f"directory {relative_dir.as_posix()}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {src_dir}")

    copied: list[CopyRecord] = []
    for src in sorted(src_dir.rglob("*")):
        if not predicate(src):
            continue
        rel = src.relative_to(source_root)
        dst = target_root / rel
        copied.append(copy_file(src, dst, dry_run))
    return copied


def load_authority_snapshot(source_root: Path) -> dict:
    recount_path = source_root / "reports/main/token_recount.main.postdeps.json"
    ensure_exists(recount_path, "authority recount artifact")
    data = json.loads(recount_path.read_text(encoding="utf-8"))

    provider_counts = data.get("table_counts_by_provider", {})
    required_providers = {"chatgpt", "claude", "deepseek", "qwen"}
    seen_providers = set(provider_counts.get("conversations", {}).keys())
    missing = sorted(required_providers - seen_providers)
    if missing:
        raise ValueError(f"Authority recount is missing providers: {missing}")

    return {
        "recount_path": recount_path.as_posix(),
        "conversations_by_provider": provider_counts.get("conversations", {}),
        "messages_by_provider": provider_counts.get("messages", {}),
        "distilled_by_provider": provider_counts.get("distilled_conversations", {}),
        "all_non_system_exact": data.get("all_non_system_exact", {}),
        "distilled_non_system_exact": data.get("distilled_non_system_exact", {}),
    }


def load_db_metadata(source_root: Path) -> dict:
    db_path = source_root / "reports/main/moonshine_mash_active.db"
    parquet_path = source_root / "reports/canonical/token_row_metrics.raw.parquet"
    ensure_exists(db_path, "main authority db")
    ensure_exists(parquet_path, "canonical parquet artifact")
    return {
        "db_path": db_path.as_posix(),
        "db_size_bytes": db_path.stat().st_size,
        "parquet_path": parquet_path.as_posix(),
        "parquet_size_bytes": parquet_path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()

    ensure_exists(source_root, "source root")
    ensure_exists(target_root, "target root")
    ensure_exists(target_root / ".git", "target git metadata")

    authority = load_authority_snapshot(source_root)
    db_metadata = load_db_metadata(source_root)

    copied: list[CopyRecord] = []

    for rel_path in ROOT_FILES:
        copied.append(copy_file(source_root / rel_path, target_root / rel_path, args.dry_run))

    for rel_path in ROOT_REPORT_FILES:
        copied.append(copy_file(source_root / rel_path, target_root / rel_path, args.dry_run))

    copied.extend(sync_directory(source_root, target_root, Path("docs"), should_copy_docs, args.dry_run))
    copied.extend(sync_directory(source_root, target_root, Path("file-trees"), should_copy_file_trees, args.dry_run))
    copied.extend(sync_directory(source_root, target_root, Path("reports/main"), should_copy_reports, args.dry_run))
    copied.extend(sync_directory(source_root, target_root, Path("reports/canonical"), should_copy_reports, args.dry_run))
    copied.extend(sync_directory(source_root, target_root, Path("visualizations"), should_copy_visualizations, args.dry_run))

    copied.append(
        copy_file(
            source_root / "visuals/logo.png",
            target_root / "visuals/logo.png",
            args.dry_run,
        )
    )

    summary = {
        "status": "dry_run" if args.dry_run else "copied",
        "source_root": source_root.as_posix(),
        "target_root": target_root.as_posix(),
        "copied_file_count": len(copied),
        "copied_total_bytes": sum(item.size_bytes for item in copied),
        "authority": authority,
        "artifacts": db_metadata,
        "copied_paths": [item.relative_path for item in copied],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - operational script
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        raise
