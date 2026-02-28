from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class VisualAtlasError(RuntimeError):
    """Raised when required report artifacts are missing or malformed."""


def _read_json(path: Path, required: bool = False) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise VisualAtlasError(f"Required report missing: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VisualAtlasError(f"Invalid JSON in {path}: {exc}") from exc


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_range_key(key: str) -> Tuple[float, float]:
    if "-" not in key:
        midpoint = _to_float(key, 0.0)
        return midpoint, midpoint
    left, right = key.split("-", 1)
    return _to_float(left, 0.0), _to_float(right, 0.0)


def _normalize_histogram(histogram: Dict[str, Any]) -> List[Dict[str, Any]]:
    bins: List[Dict[str, Any]] = []
    total = sum(max(_to_int(count, 0), 0) for count in histogram.values()) or 1
    for label, count in histogram.items():
        low, high = _parse_range_key(label)
        safe_count = max(_to_int(count, 0), 0)
        bins.append(
            {
                "label": label,
                "low": low,
                "high": high,
                "mid": (low + high) / 2.0,
                "count": safe_count,
                "density": safe_count / total,
            }
        )
    bins.sort(key=lambda row: row["mid"])
    return bins


def _build_network(
    token_forensics: Dict[str, Any],
    quality_report: Dict[str, Any],
    pii_report: Dict[str, Any],
    cost_report: Dict[str, Any],
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []

    def add_node(node_id: str, label: str, group: str, magnitude: float) -> None:
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "group": group,
                "magnitude": max(float(magnitude), 0.0001),
            }
        )

    total_samples = _to_int(token_forensics.get("dataset_summary", {}).get("total_samples"), 1)
    total_tokens = _to_int(token_forensics.get("dataset_summary", {}).get("total_tokens"), 1)

    add_node("dataset", "Mission Dataset", "core", float(total_samples))

    for source, stats in quality_report.get("quality_by_source", {}).items():
        sample_count = _to_int(stats.get("sample_count"), 0)
        quality_mean = _to_float(stats.get("avg_quality"), 0.0)
        node_id = f"source::{source}"
        add_node(node_id, source, "source", float(sample_count))
        links.append(
            {
                "source": "dataset",
                "target": node_id,
                "weight": sample_count / max(total_samples, 1),
                "meta": {"quality_mean": quality_mean},
            }
        )

    for tokenizer, stats in token_forensics.get("tokenizer_comparison", {}).get("by_tokenizer", {}).items():
        token_total = _to_int(stats.get("total_tokens"), 0)
        node_id = f"tokenizer::{tokenizer}"
        add_node(node_id, tokenizer, "tokenizer", float(token_total))
        links.append(
            {
                "source": "dataset",
                "target": node_id,
                "weight": token_total / max(total_tokens, 1),
                "meta": {"avg_tokens_per_sample": _to_float(stats.get("avg_tokens_per_sample"), 0.0)},
            }
        )

    for context_window, stats in token_forensics.get("context_window_analysis", {}).items():
        truncation_rate = _to_float(stats.get("truncation_rate_pct"), 0.0)
        node_id = f"context::{context_window}"
        add_node(node_id, f"{context_window} window", "context", truncation_rate + 1.0)
        links.append(
            {
                "source": "dataset",
                "target": node_id,
                "weight": truncation_rate / 100.0,
                "meta": {"fit_pct": _to_float(stats.get("samples_fitting_pct"), 0.0)},
            }
        )

    risk_levels = pii_report.get("safety_risk_distribution", {}).get("risk_levels", {})
    for risk_label, count in risk_levels.items():
        node_id = f"risk::{risk_label}"
        safe_count = _to_int(count, 0)
        add_node(node_id, risk_label.replace("_", " "), "risk", float(safe_count))
        links.append(
            {
                "source": "dataset",
                "target": node_id,
                "weight": safe_count / max(total_samples, 1),
                "meta": {},
            }
        )

    for model_size, scenario in cost_report.get("cost_scenarios", {}).items():
        est = scenario.get("cost_estimates", {})
        projected = _to_float(est.get("a100_cost_1epoch_usd"), 0.0)
        node_id = f"model::{model_size}"
        add_node(node_id, f"Model {model_size}", "model", projected + 1.0)
        links.append(
            {
                "source": "dataset",
                "target": node_id,
                "weight": min(projected / 1_000_000.0, 1.0),
                "meta": {
                    "a100_cost_3epoch_usd": _to_float(est.get("a100_cost_3epoch_usd"), 0.0),
                    "h100_cost_3epoch_usd": _to_float(est.get("h100_cost_3epoch_usd"), 0.0),
                },
            }
        )

    return {"nodes": nodes, "links": links}


def _build_phase_space(
    token_forensics: Dict[str, Any],
    quality_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    quality_bins = _normalize_histogram(quality_report.get("quality_distribution", {}).get("histogram", {}))
    repetition_mean = _to_float(
        token_forensics.get("quality_distribution", {})
        .get("repetition_score", {})
        .get("statistics", {})
        .get("mean"),
        0.0,
    )
    entropy_mean = _to_float(
        token_forensics.get("quality_distribution", {})
        .get("entropy_score", {})
        .get("statistics", {})
        .get("mean"),
        0.0,
    )

    points: List[Dict[str, Any]] = []
    for idx, row in enumerate(quality_bins):
        quality_mid = row["mid"]
        density = row["density"]
        inferred_risk = max(0.0, min(1.0, (1.0 - quality_mid) * (0.65 + repetition_mean)))
        stability = max(0.0, min(1.0, quality_mid * (0.55 + entropy_mean)))
        points.append(
            {
                "id": f"qbin-{idx}",
                "quality": round(quality_mid, 4),
                "density": round(density, 6),
                "risk": round(inferred_risk, 6),
                "stability": round(stability, 6),
                "count": row["count"],
                "label": row["label"],
            }
        )
    return points


def _build_truncation_surface(token_forensics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for context_window, stats in token_forensics.get("context_window_analysis", {}).items():
        fit_pct = _to_float(stats.get("samples_fitting_pct"), 0.0)
        truncation_pct = _to_float(stats.get("truncation_rate_pct"), 0.0)
        rows.append(
            {
                "window": context_window,
                "fit_pct": fit_pct,
                "truncation_pct": truncation_pct,
                "pressure_index": round((truncation_pct / 100.0) * (1.0 + (1.0 - fit_pct / 100.0)), 6),
            }
        )
    rows.sort(key=lambda item: _to_float(item["window"].replace("k", ""), 9999.0))
    return rows


def _build_cost_matrix(cost_report: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    maxima = 0.0

    for model_size, scenario in cost_report.get("cost_scenarios", {}).items():
        estimates = scenario.get("cost_estimates", {})
        row = {
            "model": model_size,
            "a100_1epoch": _to_float(estimates.get("a100_cost_1epoch_usd"), 0.0),
            "a100_3epoch": _to_float(estimates.get("a100_cost_3epoch_usd"), 0.0),
            "h100_1epoch": _to_float(estimates.get("h100_cost_1epoch_usd"), 0.0),
            "h100_3epoch": _to_float(estimates.get("h100_cost_3epoch_usd"), 0.0),
        }
        numeric_values = [row["a100_1epoch"], row["a100_3epoch"], row["h100_1epoch"], row["h100_3epoch"]]
        maxima = max(maxima, max(numeric_values) if numeric_values else 0.0)
        rows.append(row)

    rows.sort(key=lambda item: _to_float(item["model"].replace("B", ""), 0.0))
    return {"rows": rows, "max_value": maxima or 1.0}


def _build_interventions(quality_report: Dict[str, Any], token_forensics: Dict[str, Any]) -> Dict[str, Any]:
    criteria = quality_report.get("recommended_exclusions", {}).get("criteria", [])
    total_samples = _to_int(token_forensics.get("dataset_summary", {}).get("total_samples"), 0)

    flows: List[Dict[str, Any]] = []
    aggregate_excluded = 0
    for item in criteria:
        excluded = _to_int(item.get("estimated_excluded"), 0)
        aggregate_excluded += excluded
        flows.append(
            {
                "criterion": str(item.get("criterion", "unknown")),
                "description": str(item.get("description", "")),
                "excluded": excluded,
                "rationale": str(item.get("rationale", "")),
            }
        )

    retained = max(total_samples - min(aggregate_excluded, total_samples), 0)
    return {
        "flows": flows,
        "total_samples": total_samples,
        "aggregate_excluded": aggregate_excluded,
        "retained_estimate": retained,
    }


def _build_payload(
    token_forensics: Dict[str, Any],
    quality_report: Dict[str, Any],
    pii_report: Dict[str, Any],
    cost_report: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "meta": {
            "title": "Strategic Command Atlas",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_total_samples": _to_int(token_forensics.get("dataset_summary", {}).get("total_samples"), 0),
            "dataset_total_tokens": _to_int(token_forensics.get("dataset_summary", {}).get("total_tokens"), 0),
        },
        "network": _build_network(token_forensics, quality_report, pii_report, cost_report),
        "phase_space": _build_phase_space(token_forensics, quality_report),
        "truncation_surface": _build_truncation_surface(token_forensics),
        "cost_matrix": _build_cost_matrix(cost_report),
        "interventions": _build_interventions(quality_report, token_forensics),
        "safety_language": pii_report.get("high_risk_content", {}).get("by_language", {}),
        "risk_stats": pii_report.get("safety_risk_distribution", {}).get("statistics", {}),
    }


def _html_template(payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Strategic Command Atlas</title>
  <style>
    :root {{
      --bg0: #061016;
      --bg1: #0a1d27;
      --bg2: #112b38;
      --ink: #d7f0ff;
      --muted: #7ea7be;
      --accent: #44f0d1;
      --warn: #ff9f4d;
      --risk: #ff5f6d;
      --line: rgba(126, 167, 190, 0.28);
      --panel: rgba(9, 22, 31, 0.82);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: \"IBM Plex Sans\", \"Space Grotesk\", Bahnschrift, \"Segoe UI\", sans-serif;
      background:
        radial-gradient(circle at 12% 20%, rgba(68, 240, 209, 0.12), transparent 34%),
        radial-gradient(circle at 88% 5%, rgba(255, 95, 109, 0.12), transparent 30%),
        linear-gradient(135deg, var(--bg0), var(--bg1) 45%, var(--bg2));
      min-height: 100vh;
    }}

    .grid-bg {{
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(126, 167, 190, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 167, 190, 0.06) 1px, transparent 1px);
      background-size: 28px 28px;
      pointer-events: none;
      z-index: 0;
    }}

    .shell {{
      position: relative;
      z-index: 1;
      max-width: 1620px;
      margin: 0 auto;
      padding: 22px 24px 40px;
      display: grid;
      gap: 18px;
    }}

    .header {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(17, 43, 56, 0.65), rgba(9, 22, 31, 0.82));
      padding: 16px 18px;
      border-radius: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 12px;
    }}

    .title {{
      margin: 0;
      font-size: clamp(1.2rem, 2.3vw, 2rem);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 800;
    }}

    .subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
      letter-spacing: 0.04em;
    }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      min-width: 560px;
    }}

    .kpi {{
      border: 1px solid var(--line);
      border-radius: 9px;
      background: rgba(4, 15, 22, 0.68);
      padding: 8px 10px;
    }}

    .kpi .label {{ font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .kpi .value {{ font-size: 1rem; font-weight: 700; margin-top: 4px; }}

    .panel-grid {{
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 14px;
    }}

    .panel {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 12px;
      min-height: 300px;
      position: relative;
      overflow: hidden;
    }}

    .panel h3 {{
      margin: 2px 0 12px;
      font-size: 0.94rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
    }}

    .stack {{
      display: grid;
      gap: 14px;
    }}

    svg {{ width: 100%; height: auto; display: block; }}

    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.78rem;
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(17, 43, 56, 0.45);
    }}

    .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

    .tooltip {{
      position: fixed;
      pointer-events: none;
      opacity: 0;
      transform: translate(-50%, -110%);
      background: rgba(6, 16, 22, 0.95);
      border: 1px solid var(--line);
      color: var(--ink);
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 0.78rem;
      line-height: 1.3;
      max-width: 320px;
      z-index: 3;
      transition: opacity 120ms ease;
    }}

    .axis-label {{ fill: var(--muted); font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; }}
    .axis-line {{ stroke: rgba(126, 167, 190, 0.35); stroke-width: 1; }}

    @media (max-width: 1220px) {{
      .kpi-grid {{ min-width: 0; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .panel-grid {{ grid-template-columns: 1fr; }}
      .header {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"grid-bg\"></div>
  <div class=\"shell\">
    <section class=\"header\">
      <div>
        <h1 class=\"title\">Strategic Command Atlas</h1>
        <p class=\"subtitle\" id=\"subtitle\"></p>
      </div>
      <div class=\"kpi-grid\">
        <div class=\"kpi\"><div class=\"label\">Samples</div><div class=\"value\" id=\"kpi-samples\">-</div></div>
        <div class=\"kpi\"><div class=\"label\">Tokens</div><div class=\"value\" id=\"kpi-tokens\">-</div></div>
        <div class=\"kpi\"><div class=\"label\">Mean Risk</div><div class=\"value\" id=\"kpi-risk\">-</div></div>
        <div class=\"kpi\"><div class=\"label\">Retained Estimate</div><div class=\"value\" id=\"kpi-retained\">-</div></div>
      </div>
    </section>

    <section class=\"panel-grid\">
      <article class=\"panel\">
        <h3>Capability Constellation</h3>
        <svg id=\"network\" viewBox=\"0 0 980 520\" preserveAspectRatio=\"xMidYMid meet\"></svg>
        <div class=\"legend\" id=\"network-legend\"></div>
      </article>

      <div class=\"stack\">
        <article class=\"panel\">
          <h3>Context Truncation Stress Surface</h3>
          <svg id=\"truncation\" viewBox=\"0 0 560 240\" preserveAspectRatio=\"xMidYMid meet\"></svg>
        </article>

        <article class=\"panel\">
          <h3>Risk Language Topology</h3>
          <svg id=\"lang-risk\" viewBox=\"0 0 560 250\" preserveAspectRatio=\"xMidYMid meet\"></svg>
        </article>
      </div>
    </section>

    <section class=\"panel-grid\">
      <article class=\"panel\">
        <h3>Quality-Risk Phase Space</h3>
        <svg id=\"phase-space\" viewBox=\"0 0 980 360\" preserveAspectRatio=\"xMidYMid meet\"></svg>
      </article>

      <article class=\"panel\">
        <h3>Cost Regime Matrix (USD)</h3>
        <svg id=\"cost-matrix\" viewBox=\"0 0 560 360\" preserveAspectRatio=\"xMidYMid meet\"></svg>
      </article>
    </section>

    <section class=\"panel\">
      <h3>Intervention Flow Geometry</h3>
      <svg id=\"interventions\" viewBox=\"0 0 1440 320\" preserveAspectRatio=\"xMidYMid meet\"></svg>
    </section>
  </div>

  <div class=\"tooltip\" id=\"tooltip\"></div>

  <script>
    const payload = {payload_json};

    const palette = {{
      core: '#44f0d1',
      source: '#78c4ff',
      tokenizer: '#f6c177',
      context: '#9d7cff',
      risk: '#ff5f6d',
      model: '#6ce08f'
    }};

    function fmtNumber(value) {{
      if (value >= 1_000_000_000) return (value / 1_000_000_000).toFixed(2) + 'B';
      if (value >= 1_000_000) return (value / 1_000_000).toFixed(2) + 'M';
      if (value >= 1_000) return (value / 1_000).toFixed(1) + 'K';
      return String(Math.round(value));
    }}

    function fmtUsd(value) {{
      return '$' + Math.round(value).toLocaleString();
    }}

    function setKpis() {{
      document.getElementById('subtitle').textContent = `Generated ${{payload.meta.generated_at}} | Internal Strategic View`; 
      document.getElementById('kpi-samples').textContent = fmtNumber(payload.meta.dataset_total_samples);
      document.getElementById('kpi-tokens').textContent = fmtNumber(payload.meta.dataset_total_tokens);
      document.getElementById('kpi-risk').textContent = Number(payload.risk_stats.mean || 0).toFixed(4);
      document.getElementById('kpi-retained').textContent = fmtNumber(payload.interventions.retained_estimate || 0);
    }}

    function tooltip() {{
      const el = document.getElementById('tooltip');
      return {{
        show(x, y, html) {{
          el.innerHTML = html;
          el.style.left = `${{x}}px`;
          el.style.top = `${{y}}px`;
          el.style.opacity = '1';
        }},
        hide() {{
          el.style.opacity = '0';
        }}
      }};
    }}

    function drawNetwork() {{
      const svg = document.getElementById('network');
      const tt = tooltip();
      const nodes = payload.network.nodes.map((d, i) => ({{
        ...d,
        x: 120 + (i % 9) * 90 + Math.random() * 20,
        y: 80 + Math.floor(i / 9) * 88 + Math.random() * 20,
        vx: 0,
        vy: 0
      }}));
      const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
      const links = payload.network.links
        .map(l => ({{ ...l, a: nodeMap[l.source], b: nodeMap[l.target] }}))
        .filter(l => l.a && l.b);

      const gLinks = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      const gNodes = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      svg.appendChild(gLinks);
      svg.appendChild(gNodes);

      const linkEls = links.map(link => {{
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('stroke', 'rgba(126,167,190,0.35)');
        line.setAttribute('stroke-width', String(1 + link.weight * 8));
        line.setAttribute('stroke-linecap', 'round');
        gLinks.appendChild(line);
        return {{ line, link }};
      }});

      const nodeEls = nodes.map(node => {{
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        const r = 5 + Math.sqrt(node.magnitude) * 0.15;

        circle.setAttribute('r', String(Math.min(r, 26)));
        circle.setAttribute('fill', palette[node.group] || '#c2d8e6');
        circle.setAttribute('fill-opacity', '0.82');
        circle.setAttribute('stroke', 'rgba(215,240,255,0.7)');
        circle.setAttribute('stroke-width', '1');

        label.textContent = node.label;
        label.setAttribute('x', '10');
        label.setAttribute('y', '-10');
        label.setAttribute('fill', '#b9d5e8');
        label.setAttribute('font-size', '10');
        label.setAttribute('letter-spacing', '0.03em');

        group.appendChild(circle);
        group.appendChild(label);
        gNodes.appendChild(group);

        group.addEventListener('mousemove', e => {{
          tt.show(e.clientX, e.clientY, `<strong>${{node.label}}</strong><br/>Group: ${{node.group}}<br/>Magnitude: ${{fmtNumber(node.magnitude)}}`);
        }});
        group.addEventListener('mouseleave', () => tt.hide());

        return {{ group, node }};
      }});

      for (let i = 0; i < 180; i += 1) {{
        for (const {{link}} of linkEls) {{
          const dx = link.b.x - link.a.x;
          const dy = link.b.y - link.a.y;
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
          const target = 120;
          const force = (dist - target) * 0.0008;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          link.a.vx += fx;
          link.a.vy += fy;
          link.b.vx -= fx;
          link.b.vy -= fy;
        }}
        for (const node of nodes) {{
          node.vx *= 0.89;
          node.vy *= 0.89;
          node.x += node.vx;
          node.y += node.vy;
          node.x = Math.max(35, Math.min(945, node.x));
          node.y = Math.max(30, Math.min(495, node.y));
        }}
        for (let a = 0; a < nodes.length; a += 1) {{
          for (let b = a + 1; b < nodes.length; b += 1) {{
            const na = nodes[a];
            const nb = nodes[b];
            const dx = nb.x - na.x;
            const dy = nb.y - na.y;
            const dist2 = dx * dx + dy * dy;
            if (dist2 < 1800) {{
              const push = 0.02 / Math.max(dist2, 12);
              na.vx -= dx * push;
              na.vy -= dy * push;
              nb.vx += dx * push;
              nb.vy += dy * push;
            }}
          }}
        }}
      }}

      for (const {{line, link}} of linkEls) {{
        line.setAttribute('x1', link.a.x);
        line.setAttribute('y1', link.a.y);
        line.setAttribute('x2', link.b.x);
        line.setAttribute('y2', link.b.y);
      }}
      for (const {{group, node}} of nodeEls) {{
        group.setAttribute('transform', `translate(${{node.x}},${{node.y}})`);
      }}

      const legend = document.getElementById('network-legend');
      const groups = [...new Set(nodes.map(n => n.group))];
      groups.forEach(group => {{
        const item = document.createElement('span');
        item.className = 'tag';
        const dot = document.createElement('span');
        dot.className = 'dot';
        dot.style.background = palette[group] || '#c2d8e6';
        item.appendChild(dot);
        item.appendChild(document.createTextNode(group));
        legend.appendChild(item);
      }});
    }}

    function drawTruncation() {{
      const svg = document.getElementById('truncation');
      const tt = tooltip();
      const data = payload.truncation_surface;
      const width = 560;
      const baseY = 210;
      const barW = 110;
      const gap = 34;

      const maxTrunc = Math.max(...data.map(d => d.truncation_pct), 1);

      data.forEach((d, i) => {{
        const x = 44 + i * (barW + gap);
        const h = (d.truncation_pct / maxTrunc) * 150;

        const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bar.setAttribute('x', x);
        bar.setAttribute('y', baseY - h);
        bar.setAttribute('width', barW);
        bar.setAttribute('height', h);
        bar.setAttribute('rx', 8);
        bar.setAttribute('fill', 'rgba(255,159,77,0.78)');
        bar.setAttribute('stroke', 'rgba(255,255,255,0.35)');

        const cap = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        cap.textContent = `${{d.window}}`;
        cap.setAttribute('x', x + barW / 2);
        cap.setAttribute('y', 228);
        cap.setAttribute('text-anchor', 'middle');
        cap.setAttribute('class', 'axis-label');

        const val = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        val.textContent = `${{d.truncation_pct.toFixed(2)}}%`;
        val.setAttribute('x', x + barW / 2);
        val.setAttribute('y', baseY - h - 8);
        val.setAttribute('text-anchor', 'middle');
        val.setAttribute('fill', '#ffd6ad');
        val.setAttribute('font-size', '11');

        bar.addEventListener('mousemove', e => {{
          tt.show(e.clientX, e.clientY, `<strong>${{d.window}} context</strong><br/>Truncation: ${{d.truncation_pct.toFixed(2)}}%<br/>Fit: ${{d.fit_pct.toFixed(2)}}%<br/>Pressure: ${{d.pressure_index.toFixed(4)}}`);
        }});
        bar.addEventListener('mouseleave', () => tt.hide());

        svg.appendChild(bar);
        svg.appendChild(cap);
        svg.appendChild(val);
      }});

      const axis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      axis.setAttribute('x1', 30);
      axis.setAttribute('y1', baseY);
      axis.setAttribute('x2', width - 20);
      axis.setAttribute('y2', baseY);
      axis.setAttribute('class', 'axis-line');
      svg.appendChild(axis);
    }}

    function drawLanguageRisk() {{
      const svg = document.getElementById('lang-risk');
      const tt = tooltip();
      const entries = Object.entries(payload.safety_language || {{}})
        .map(([lang, count]) => ({{ lang, count: Number(count || 0) }}))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

      const maxCount = Math.max(...entries.map(e => e.count), 1);
      entries.forEach((entry, i) => {{
        const y = 24 + i * 21;
        const w = (entry.count / maxCount) * 350;

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.textContent = entry.lang;
        label.setAttribute('x', 10);
        label.setAttribute('y', y + 12);
        label.setAttribute('class', 'axis-label');

        const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bar.setAttribute('x', 70);
        bar.setAttribute('y', y);
        bar.setAttribute('width', w);
        bar.setAttribute('height', 14);
        bar.setAttribute('rx', 4);
        bar.setAttribute('fill', 'rgba(255,95,109,0.72)');

        const value = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        value.textContent = String(entry.count);
        value.setAttribute('x', 430 + 8);
        value.setAttribute('y', y + 12);
        value.setAttribute('fill', '#f4c3ca');
        value.setAttribute('font-size', '10');

        bar.addEventListener('mousemove', e => {{
          tt.show(e.clientX, e.clientY, `<strong>${{entry.lang}}</strong><br/>High-risk samples: ${{entry.count}}`);
        }});
        bar.addEventListener('mouseleave', () => tt.hide());

        svg.appendChild(label);
        svg.appendChild(bar);
        svg.appendChild(value);
      }});
    }}

    function drawPhaseSpace() {{
      const svg = document.getElementById('phase-space');
      const tt = tooltip();
      const points = payload.phase_space;

      const left = 58;
      const right = 960;
      const top = 24;
      const bottom = 320;

      const xScale = q => left + q * (right - left);
      const yScale = d => bottom - d * (bottom - top);

      const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      xAxis.setAttribute('x1', left);
      xAxis.setAttribute('y1', bottom);
      xAxis.setAttribute('x2', right);
      xAxis.setAttribute('y2', bottom);
      xAxis.setAttribute('class', 'axis-line');

      const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      yAxis.setAttribute('x1', left);
      yAxis.setAttribute('y1', top);
      yAxis.setAttribute('x2', left);
      yAxis.setAttribute('y2', bottom);
      yAxis.setAttribute('class', 'axis-line');

      svg.appendChild(xAxis);
      svg.appendChild(yAxis);

      const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      xLabel.textContent = 'Quality Coordinate';
      xLabel.setAttribute('x', (left + right) / 2);
      xLabel.setAttribute('y', 350);
      xLabel.setAttribute('text-anchor', 'middle');
      xLabel.setAttribute('class', 'axis-label');
      svg.appendChild(xLabel);

      const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      yLabel.textContent = 'Density';
      yLabel.setAttribute('x', 16);
      yLabel.setAttribute('y', (top + bottom) / 2);
      yLabel.setAttribute('text-anchor', 'middle');
      yLabel.setAttribute('class', 'axis-label');
      yLabel.setAttribute('transform', `rotate(-90 16 ${{(top + bottom) / 2}})`);
      svg.appendChild(yLabel);

      points.forEach(point => {{
        const cx = xScale(point.quality);
        const cy = yScale(point.density * 2.3);
        const radius = 5 + Math.sqrt(point.count) * 0.1;
        const heat = Math.round(255 * point.risk);
        const cool = Math.round(255 * (1 - point.risk));
        const color = `rgba(${{heat}},${{70 + cool / 3}},${{cool}},0.78)`;

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', cx);
        circle.setAttribute('cy', cy);
        circle.setAttribute('r', Math.min(radius, 28));
        circle.setAttribute('fill', color);
        circle.setAttribute('stroke', 'rgba(255,255,255,0.44)');
        circle.setAttribute('stroke-width', '1');

        circle.addEventListener('mousemove', e => {{
          tt.show(
            e.clientX,
            e.clientY,
            `<strong>Bin ${{point.label}}</strong><br/>Quality: ${{point.quality}}<br/>Density: ${{point.density.toFixed(4)}}<br/>Risk: ${{point.risk.toFixed(4)}}<br/>Count: ${{point.count}}`
          );
        }});
        circle.addEventListener('mouseleave', () => tt.hide());

        svg.appendChild(circle);
      }});
    }}

    function drawCostMatrix() {{
      const svg = document.getElementById('cost-matrix');
      const tt = tooltip();
      const rows = payload.cost_matrix.rows;
      const maxValue = payload.cost_matrix.max_value || 1;
      const cols = ['a100_1epoch', 'a100_3epoch', 'h100_1epoch', 'h100_3epoch'];
      const labels = ['A100 / 1ep', 'A100 / 3ep', 'H100 / 1ep', 'H100 / 3ep'];

      const x0 = 140;
      const y0 = 56;
      const cellW = 92;
      const cellH = 72;

      labels.forEach((label, i) => {{
        const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        t.textContent = label;
        t.setAttribute('x', x0 + i * cellW + cellW / 2);
        t.setAttribute('y', 24);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('class', 'axis-label');
        svg.appendChild(t);
      }});

      rows.forEach((row, rIdx) => {{
        const yl = y0 + rIdx * cellH;
        const modelLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        modelLabel.textContent = row.model;
        modelLabel.setAttribute('x', 24);
        modelLabel.setAttribute('y', yl + cellH / 2 + 4);
        modelLabel.setAttribute('class', 'axis-label');
        svg.appendChild(modelLabel);

        cols.forEach((key, cIdx) => {{
          const value = Number(row[key] || 0);
          const norm = Math.max(0, Math.min(1, value / maxValue));
          const red = Math.round(255 * norm);
          const green = Math.round(205 * (1 - norm));
          const blue = Math.round(130 * (1 - norm));

          const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          rect.setAttribute('x', x0 + cIdx * cellW);
          rect.setAttribute('y', yl);
          rect.setAttribute('width', cellW - 8);
          rect.setAttribute('height', cellH - 8);
          rect.setAttribute('rx', 7);
          rect.setAttribute('fill', `rgba(${{red}},${{green}},${{blue}},0.82)`);
          rect.setAttribute('stroke', 'rgba(255,255,255,0.32)');

          const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          txt.textContent = fmtUsd(value);
          txt.setAttribute('x', x0 + cIdx * cellW + (cellW - 8) / 2);
          txt.setAttribute('y', yl + (cellH - 8) / 2 + 4);
          txt.setAttribute('text-anchor', 'middle');
          txt.setAttribute('fill', '#f4f8ff');
          txt.setAttribute('font-size', '10');

          rect.addEventListener('mousemove', e => {{
            tt.show(e.clientX, e.clientY, `<strong>${{row.model}} | ${{labels[cIdx]}}</strong><br/>Projected cost: ${{fmtUsd(value)}}`);
          }});
          rect.addEventListener('mouseleave', () => tt.hide());

          svg.appendChild(rect);
          svg.appendChild(txt);
        }});
      }});
    }}

    function drawInterventions() {{
      const svg = document.getElementById('interventions');
      const tt = tooltip();
      const total = Number(payload.interventions.total_samples || 1);
      const flows = payload.interventions.flows || [];

      const leftX = 40;
      const rightX = 1320;
      const centerY = 160;

      const source = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      source.setAttribute('x', leftX);
      source.setAttribute('y', 116);
      source.setAttribute('width', 140);
      source.setAttribute('height', 88);
      source.setAttribute('rx', 10);
      source.setAttribute('fill', 'rgba(68,240,209,0.22)');
      source.setAttribute('stroke', 'rgba(68,240,209,0.7)');

      const sourceText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      sourceText.textContent = 'Candidate Set';
      sourceText.setAttribute('x', leftX + 70);
      sourceText.setAttribute('y', 154);
      sourceText.setAttribute('text-anchor', 'middle');
      sourceText.setAttribute('fill', '#9deee0');
      sourceText.setAttribute('font-size', '12');

      const sourceCount = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      sourceCount.textContent = fmtNumber(total);
      sourceCount.setAttribute('x', leftX + 70);
      sourceCount.setAttribute('y', 177);
      sourceCount.setAttribute('text-anchor', 'middle');
      sourceCount.setAttribute('fill', '#d9fff7');
      sourceCount.setAttribute('font-size', '14');
      sourceCount.setAttribute('font-weight', '700');

      svg.appendChild(source);
      svg.appendChild(sourceText);
      svg.appendChild(sourceCount);

      const sink = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      sink.setAttribute('x', rightX - 180);
      sink.setAttribute('y', 116);
      sink.setAttribute('width', 160);
      sink.setAttribute('height', 88);
      sink.setAttribute('rx', 10);
      sink.setAttribute('fill', 'rgba(108,224,143,0.2)');
      sink.setAttribute('stroke', 'rgba(108,224,143,0.65)');

      const sinkText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      sinkText.textContent = 'Retained Set';
      sinkText.setAttribute('x', rightX - 100);
      sinkText.setAttribute('y', 154);
      sinkText.setAttribute('text-anchor', 'middle');
      sinkText.setAttribute('fill', '#c7ffd6');
      sinkText.setAttribute('font-size', '12');

      const retainedCount = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      retainedCount.textContent = fmtNumber(payload.interventions.retained_estimate || 0);
      retainedCount.setAttribute('x', rightX - 100);
      retainedCount.setAttribute('y', 177);
      retainedCount.setAttribute('text-anchor', 'middle');
      retainedCount.setAttribute('fill', '#e6ffee');
      retainedCount.setAttribute('font-size', '14');
      retainedCount.setAttribute('font-weight', '700');

      svg.appendChild(sink);
      svg.appendChild(sinkText);
      svg.appendChild(retainedCount);

      flows.forEach((flow, idx) => {{
        const y = 40 + idx * 46;
        const ratio = Math.min(1, Number(flow.excluded || 0) / Math.max(total, 1));
        const strokeW = 2 + ratio * 18;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const cx1 = 390;
        const cx2 = 910;
        const d = `M ${{leftX + 140}} ${{centerY}} C ${{cx1}} ${{centerY}}, ${{cx2}} ${{y}}, ${{rightX - 180}} ${{y}}`;
        path.setAttribute('d', d);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', 'rgba(255,95,109,0.5)');
        path.setAttribute('stroke-width', strokeW);
        path.setAttribute('stroke-linecap', 'round');

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.textContent = `${{flow.criterion}} (${{fmtNumber(flow.excluded)}})`;
        label.setAttribute('x', rightX - 170);
        label.setAttribute('y', y - 8);
        label.setAttribute('fill', '#ffced4');
        label.setAttribute('font-size', '10');

        path.addEventListener('mousemove', e => {{
          tt.show(
            e.clientX,
            e.clientY,
            `<strong>${{flow.criterion}}</strong><br/>Excluded: ${{fmtNumber(flow.excluded)}}<br/>${{flow.description}}<br/><em>${{flow.rationale}}</em>`
          );
        }});
        path.addEventListener('mouseleave', () => tt.hide());

        svg.appendChild(path);
        svg.appendChild(label);
      }});
    }}

    function main() {{
      setKpis();
      drawNetwork();
      drawTruncation();
      drawLanguageRisk();
      drawPhaseSpace();
      drawCostMatrix();
      drawInterventions();
    }}

    main();
  </script>
</body>
</html>
"""


def generate_command_atlas(reports_dir: Path, out_dir: Path) -> Dict[str, Path]:
    token_forensics = _read_json(reports_dir / "token_forensics.json", required=True)
    quality_report = _read_json(reports_dir / "quality_risk_report.json", required=True)
    pii_report = _read_json(reports_dir / "pii_safety_report.json", required=True)
    cost_report = _read_json(reports_dir / "cost_projection.json", required=True)

    payload = _build_payload(token_forensics, quality_report, pii_report, cost_report)
    html = _html_template(payload)

    out_dir.mkdir(parents=True, exist_ok=True)

    payload_path = out_dir / "atlas_payload.json"
    html_path = out_dir / "strategic_command_atlas.html"
    manifest_path = out_dir / "visual_manifest.json"

    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_reports": [
            str(reports_dir / "token_forensics.json"),
            str(reports_dir / "quality_risk_report.json"),
            str(reports_dir / "pii_safety_report.json"),
            str(reports_dir / "cost_projection.json"),
        ],
        "artifacts": {
            "payload": str(payload_path),
            "dashboard": str(html_path),
        },
        "meta": payload["meta"],
        "sections": [
            "capability_constellation",
            "truncation_stress_surface",
            "risk_language_topology",
            "quality_risk_phase_space",
            "cost_regime_matrix",
            "intervention_flow_geometry",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "payload": payload_path,
        "dashboard": html_path,
        "manifest": manifest_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Strategic Command Atlas visuals from existing report artifacts."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing token_forensics.json and companion reports.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "visuals",
        help="Output directory for generated visual artifacts.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        outputs = generate_command_atlas(args.reports_dir, args.out_dir)
    except VisualAtlasError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("[OK] Strategic Command Atlas generated")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
