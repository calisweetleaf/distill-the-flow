#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "token_forensics.json"
DB_PATH = ROOT / "reports" / "moonshine_corpus.db"
OUT_DIR = ROOT / "visualizations" / "extended"


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _bar_monthly_tokens(data: dict) -> None:
    rows = [r for r in data.get("monthly_distribution", []) if r.get("month_utc") != "unknown"]
    months = [r["month_utc"] for r in rows]
    tokens_m = [r.get("tokens", 0) / 1_000_000 for r in rows]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(months, tokens_m, color="#2a9d8f")
    ax.set_title("Monthly Tokens (Millions)")
    ax.set_ylabel("Tokens (M)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, OUT_DIR / "monthly_tokens_millions.png")


def _line_monthly_messages(data: dict) -> None:
    rows = [r for r in data.get("monthly_distribution", []) if r.get("month_utc") != "unknown"]
    months = [r["month_utc"] for r in rows]
    msgs = [r.get("messages", 0) for r in rows]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(months, msgs, marker="o", color="#1d3557")
    ax.set_title("Monthly Message Volume")
    ax.set_ylabel("Messages")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, OUT_DIR / "monthly_message_volume.png")


def _role_pie(data: dict) -> None:
    role = data.get("role_distribution", {})
    labels = list(role.keys())
    vals = list(role.values())
    if not vals:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=120)
    ax.set_title("Role Distribution")
    _save(fig, OUT_DIR / "role_distribution_pie.png")


def _top_conv_bar(data: dict) -> None:
    top = data.get("top_conversations_by_tokens", [])[:15]
    ids = [r["conversation_id"][:8] for r in top][::-1]
    vals = [r.get("tokens", 0) / 1000 for r in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ids, vals, color="#e76f51")
    ax.set_title("Top 15 Conversations by Tokens")
    ax.set_xlabel("Tokens (K)")
    _save(fig, OUT_DIR / "top15_conversations_tokens.png")


def _monthly_scatter(data: dict) -> None:
    rows = [r for r in data.get("monthly_distribution", []) if r.get("month_utc") != "unknown"]
    tokens_m = np.array([r.get("tokens", 0) / 1_000_000 for r in rows])
    msgs_k = np.array([r.get("messages", 0) / 1000 for r in rows])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(msgs_k, tokens_m, s=70, c=np.linspace(0, 1, len(rows)), cmap="viridis")
    for r, x, y in zip(rows, msgs_k, tokens_m):
        ax.annotate(r["month_utc"], (x, y), fontsize=8)
    ax.set_title("Monthly Tokens vs Messages")
    ax.set_xlabel("Messages (K)")
    ax.set_ylabel("Tokens (M)")
    _save(fig, OUT_DIR / "monthly_tokens_vs_messages_scatter.png")


def _rolling_growth(data: dict) -> None:
    rows = [r for r in data.get("monthly_distribution", []) if r.get("month_utc") != "unknown"]
    months = [r["month_utc"] for r in rows]
    vals = np.array([r.get("tokens", 0) for r in rows], dtype=float)
    csum = np.cumsum(vals) / 1_000_000

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(months, csum, color="#6a4c93", linewidth=2)
    ax.fill_between(months, csum, color="#b8a1d9", alpha=0.35)
    ax.set_title("Cumulative Token Growth")
    ax.set_ylabel("Cumulative Tokens (M)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, OUT_DIR / "cumulative_token_growth.png")


def _topic_tone_from_db() -> None:
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT topic_primary, COUNT(*) FROM conversations GROUP BY topic_primary ORDER BY COUNT(*) DESC LIMIT 12")
    topics = cur.fetchall()

    cur.execute("SELECT tone_cluster, COUNT(*) FROM conversations GROUP BY tone_cluster ORDER BY COUNT(*) DESC")
    tones = cur.fetchall()

    cur.execute("SELECT period, AVG(information_gain), AVG(malicious_compliance), COUNT(*) FROM conversations GROUP BY period ORDER BY period")
    periods = cur.fetchall()

    conn.close()

    if topics:
        labels = [t[0] for t in topics][::-1]
        vals = [t[1] for t in topics][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels, vals, color="#457b9d")
        ax.set_title("Top Topics (Conversation Count)")
        _save(fig, OUT_DIR / "top_topics_barh.png")

    if tones:
        labels = [t[0] for t in tones]
        vals = [t[1] for t in tones]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(labels, vals, color="#ffb703")
        ax.set_title("Tone Cluster Distribution")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, OUT_DIR / "tone_cluster_distribution.png")

    if periods:
        p = [f"P{int(x[0])}" for x in periods]
        info = [float(x[1]) for x in periods]
        syco = [float(x[2]) for x in periods]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(p, info, marker="o", label="Info Gain", color="#2a9d8f")
        ax.plot(p, syco, marker="s", label="Malicious Compliance", color="#d62828")
        ax.set_title("Period Quality Signals")
        ax.legend()
        _save(fig, OUT_DIR / "period_quality_signals.png")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    _bar_monthly_tokens(data)
    _line_monthly_messages(data)
    _role_pie(data)
    _top_conv_bar(data)
    _monthly_scatter(data)
    _rolling_growth(data)
    _topic_tone_from_db()

    created = sorted([p.name for p in OUT_DIR.glob("*.png")])
    manifest = {
        "generated_from": [str(REPORT_PATH), str(DB_PATH)],
        "output_dir": str(OUT_DIR),
        "count": len(created),
        "files": created,
    }
    (OUT_DIR / "extended_visual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Extended visuals created: {len(created)}")
    for name in created:
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
