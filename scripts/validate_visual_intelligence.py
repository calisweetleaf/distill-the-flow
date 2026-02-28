from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_intelligence.command_atlas import VisualAtlasError, generate_command_atlas


def _log(lines: List[str], message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"[{stamp}] {message}"
    lines.append(line)
    print(line)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_validation(root: Path) -> Dict[str, Any]:
    reports_dir = root / "reports"
    out_dir = reports_dir / "visuals"

    terminal_log: List[str] = []
    checks: List[Dict[str, Any]] = []

    _log(terminal_log, "Starting visual intelligence validation run")
    _log(terminal_log, f"Root directory: {root}")

    try:
        outputs = generate_command_atlas(reports_dir=reports_dir, out_dir=out_dir)
        _log(terminal_log, "Visual artifact generation succeeded")

        required_paths = [
            outputs["payload"],
            outputs["dashboard"],
            outputs["manifest"],
        ]
        for path in required_paths:
            exists = path.exists()
            checks.append({"check": f"exists::{path.name}", "passed": exists})
            _log(terminal_log, f"Check exists {path.name}: {exists}")
            _assert(exists, f"Missing expected artifact: {path}")

        payload = _read_json(outputs["payload"])
        manifest = _read_json(outputs["manifest"])

        required_payload_keys = [
            "meta",
            "network",
            "phase_space",
            "truncation_surface",
            "cost_matrix",
            "interventions",
        ]
        for key in required_payload_keys:
            passed = key in payload
            checks.append({"check": f"payload_key::{key}", "passed": passed})
            _log(terminal_log, f"Check payload key '{key}': {passed}")
            _assert(passed, f"Payload missing key: {key}")

        node_count = len(payload.get("network", {}).get("nodes", []))
        link_count = len(payload.get("network", {}).get("links", []))
        checks.append({"check": "network_nonempty", "passed": node_count > 0 and link_count > 0})
        _log(terminal_log, f"Check network density nodes={node_count}, links={link_count}")
        _assert(node_count > 0 and link_count > 0, "Network graph is empty")

        phase_points = payload.get("phase_space", [])
        checks.append({"check": "phase_space_nonempty", "passed": len(phase_points) > 0})
        _log(terminal_log, f"Check phase-space points count={len(phase_points)}")
        _assert(len(phase_points) > 0, "Phase-space data is empty")

        html_text = outputs["dashboard"].read_text(encoding="utf-8")
        required_markers = [
            "Capability Constellation",
            "Quality-Risk Phase Space",
            "Cost Regime Matrix",
            "Intervention Flow Geometry",
        ]
        for marker in required_markers:
            passed = marker in html_text
            checks.append({"check": f"html_marker::{marker}", "passed": passed})
            _log(terminal_log, f"Check html marker '{marker}': {passed}")
            _assert(passed, f"Dashboard HTML missing marker: {marker}")

        sections = manifest.get("sections", [])
        checks.append({"check": "manifest_sections_count", "passed": len(sections) >= 6})
        _log(terminal_log, f"Check manifest section count={len(sections)}")
        _assert(len(sections) >= 6, "Manifest is missing section registrations")

        status = "PASS"
        _log(terminal_log, "Validation completed with PASS")

    except (VisualAtlasError, AssertionError, json.JSONDecodeError) as exc:
        status = "FAIL"
        _log(terminal_log, f"Validation failed: {exc}")
        outputs = {
            "payload": out_dir / "atlas_payload.json",
            "dashboard": out_dir / "strategic_command_atlas.html",
            "manifest": out_dir / "visual_manifest.json",
        }

    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "artifacts": {k: str(v) for k, v in outputs.items()},
        "checks": checks,
        "terminal_log": terminal_log,
    }

    validation_manifest_path = reports_dir / "visuals_validation_manifest.json"
    validation_manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Visual Intelligence Validation Report",
        "",
        f"- Status: **{status}**",
        f"- Generated At: `{summary['generated_at']}`",
        f"- Root: `{root}`",
        "",
        "## Artifacts",
    ]
    for key, value in summary["artifacts"].items():
        report_lines.append(f"- `{key}`: `{value}`")

    report_lines.extend(["", "## Checks"])
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        report_lines.append(f"- `{mark}` {check['check']}")

    report_lines.extend(["", "## Terminal Log"])
    for line in terminal_log:
        report_lines.append(f"- `{line}`")

    report_path = reports_dir / "visuals_validation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return summary


def main() -> int:
    summary = run_validation(ROOT)
    print(f"Validation status: {summary['status']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
