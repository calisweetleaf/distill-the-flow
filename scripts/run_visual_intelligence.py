from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_intelligence.command_atlas import VisualAtlasError, generate_command_atlas


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Strategic Command Atlas visual generator.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "visuals")
    args = parser.parse_args()

    try:
        outputs = generate_command_atlas(args.reports_dir, args.out_dir)
    except VisualAtlasError as exc:
        print(f"[ERROR] visual generation failed: {exc}")
        return 1

    print("[OK] visual intelligence artifacts generated")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
