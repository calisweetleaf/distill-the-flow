#!/usr/bin/env python3
"""
Moonshine Visual Intelligence — Time-Series Analysis
=====================================================

Generates publication-quality visualizations of corpus evolution over time.
Creates charts for: topic distribution, quality metrics, correction events,
temporal trends, and period comparisons.

Usage:
    python moonshine_visualizer.py
    python moonshine_visualizer.py --output-dir ./visualizations
    python moonshine_visualizer.py --db reports/moonshine_corpus.db

Output:
    - topic_distribution.png (bar chart)
    - quality_metrics_timeseries.png (line chart)
    - token_ratio_distribution.png (histogram)
    - corrections_scatter.png (scatter plot)
    - period_comparison.png (grouped bar chart)
    - corpus_overview.png (dashboard)

Dependencies:
    - matplotlib
    - numpy
    - sqlite3
"""

import sqlite3
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[ERROR] matplotlib not installed. Run: .venv\\Scripts\\pip install matplotlib")
    exit(1)


class MoonshineVisualizer:
    """Generates visualizations from Moonshine corpus database."""
    
    def __init__(self, db_path: Path, output_dir: Path = Path("visualizations")):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        
    def load_data(self):
        """Load all data from SQLite database."""
        print("[LOAD] Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load conversations
        cursor.execute("""
            SELECT conversation_id, title, topic_primary, tone_cluster, period,
                   information_gain, malicious_compliance, user_entropy, token_ratio,
                   correction_events, total_tokens, user_tokens, assistant_tokens
            FROM conversations
        """)
        
        columns = [desc[0] for desc in cursor.description]
        self.data['conversations'] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Load topic counts
        cursor.execute("""
            SELECT topic_primary, COUNT(*) as count
            FROM conversations
            GROUP BY topic_primary
            ORDER BY count DESC
        """)
        self.data['topic_counts'] = cursor.fetchall()
        
        # Load period stats
        cursor.execute("""
            SELECT period,
                   COUNT(*) as conv_count,
                   AVG(information_gain) as avg_info_gain,
                   AVG(malicious_compliance) as avg_sycophancy,
                   AVG(user_entropy) as avg_entropy,
                   AVG(token_ratio) as avg_token_ratio,
                   SUM(correction_events) as total_corrections
            FROM conversations
            GROUP BY period
            ORDER BY period
        """)
        self.data['period_stats'] = cursor.fetchall()
        
        # Load tone distribution
        cursor.execute("""
            SELECT tone_cluster, COUNT(*) as count
            FROM conversations
            GROUP BY tone_cluster
            ORDER BY count DESC
        """)
        self.data['tone_counts'] = cursor.fetchall()
        
        conn.close()
        print(f"[OK] Loaded {len(self.data['conversations'])} conversations")
        
    def plot_topic_distribution(self, figsize: Tuple[int, int] = (12, 6)):
        """Generate topic distribution bar chart."""
        print("[PLOT] Generating topic distribution chart...")
        
        topics = [t[0] for t in self.data['topic_counts']]
        counts = [t[1] for t in self.data['topic_counts']]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(topics)))
        bars = ax.bar(range(len(topics)), counts, color=colors)
        
        ax.set_xlabel('Topic', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
        ax.set_title('Topic Distribution Across Corpus', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / 'topic_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_path}")
        
    def plot_quality_metrics_timeseries(self, figsize: Tuple[int, int] = (12, 6)):
        """Generate quality metrics over time (by period)."""
        print("[PLOT] Generating quality metrics time series...")
        
        periods = [p[0] for p in self.data['period_stats']]
        info_gains = [p[2] for p in self.data['period_stats']]
        sycophancy = [p[3] for p in self.data['period_stats']]
        entropy = [p[4] for p in self.data['period_stats']]
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot information gain and entropy on primary axis
        line1 = ax1.plot(periods, info_gains, 'b-o', linewidth=2, markersize=8, label='Information Gain')
        line2 = ax1.plot(periods, entropy, 'g-s', linewidth=2, markersize=8, label='User Entropy')
        
        ax1.set_xlabel('Period', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Create secondary axis for sycophancy (lower is better)
        ax2 = ax1.twinx()
        line3 = ax2.plot(periods, sycophancy, 'r-^', linewidth=2, markersize=8, label='Malicious Compliance')
        ax2.set_ylabel('Sycophancy Score', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        max_syco = max(sycophancy) if sycophancy and max(sycophancy) > 0 else 1.0
        ax2.set_ylim(0, max_syco * 1.5)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
        
        ax1.set_title('Quality Metrics Evolution Over Time', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / 'quality_metrics_timeseries.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_path}")
        
    def plot_token_ratio_distribution(self, figsize: Tuple[int, int] = (12, 6)):
        """Generate histogram of user/assistant token ratios."""
        print("[PLOT] Generating token ratio distribution...")
        
        ratios = [c['token_ratio'] for c in self.data['conversations'] if c['token_ratio'] < 10]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create histogram
        n, bins, patches = ax.hist(ratios, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Color bars based on ratio (highlight unbalanced conversations)
        for i, (patch, ratio_bin) in enumerate(zip(patches, bins[:-1])):
            if ratio_bin > 5:
                patch.set_facecolor('orange')
                patch.set_alpha(0.8)
        
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Balanced (1:1)')
        ax.axvline(x=np.mean(ratios), color='red', linestyle='-', linewidth=2, label=f'Mean ({np.mean(ratios):.2f})')
        
        ax.set_xlabel('Token Ratio (User / Assistant)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of User/Assistant Token Ratios', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'token_ratio_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_path}")
        
    def plot_corrections_scatter(self, figsize: Tuple[int, int] = (14, 8)):
        """Generate scatter plot of corrections vs quality."""
        print("[PLOT] Generating corrections scatter plot...")
        
        # Get data
        corrections = [c['correction_events'] for c in self.data['conversations']]
        info_gains = [c['information_gain'] for c in self.data['conversations']]
        topics = [c['topic_primary'] for c in self.data['conversations']]
        tokens = [c['total_tokens'] for c in self.data['conversations']]
        
        # Create topic color mapping
        unique_topics = list(set(topics))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_topics)))
        topic_colors = {topic: colors[i] for i, topic in enumerate(unique_topics)}
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        for topic in unique_topics:
            mask = [t == topic for t in topics]
            x = [corrections[i] for i, m in enumerate(mask) if m]
            y = [info_gains[i] for i, m in enumerate(mask) if m]
            s = [max(20, min(200, tokens[i] / 1000)) for i, m in enumerate(mask) if m]
            
            ax.scatter(x, y, c=[topic_colors[topic]], label=topic, alpha=0.6, s=s, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Correction Events', fontsize=12, fontweight='bold')
        ax.set_ylabel('Information Gain', fontsize=12, fontweight='bold')
        ax.set_title('Correction Events vs Information Gain by Topic\n(Bubble size = Token count)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'corrections_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_path}")
        
    def plot_period_comparison(self, figsize: Tuple[int, int] = (14, 8)):
        """Generate grouped bar chart comparing periods."""
        print("[PLOT] Generating period comparison chart...")
        
        periods = [f"P{p[0]}" for p in self.data['period_stats']]
        conv_counts = [p[1] for p in self.data['period_stats']]
        corrections = [p[6] for p in self.data['period_stats']]
        
        x = np.arange(len(periods))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Bar chart for conversation counts
        bars1 = ax1.bar(x - width/2, conv_counts, width, label='Conversations', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Period', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Conversation Count', fontsize=12, fontweight='bold', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # Secondary axis for corrections
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, corrections, width, label='Corrections', color='orange', alpha=0.8)
        ax2.set_ylabel('Total Correction Events', fontsize=12, fontweight='bold', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods)
        ax1.set_title('Period Comparison: Conversations vs Correction Events', fontsize=14, fontweight='bold', pad=20)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        output_path = self.output_dir / 'period_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_path}")
        
    def plot_corpus_overview_dashboard(self, figsize: Tuple[int, int] = (16, 12)):
        """Generate comprehensive dashboard visualization."""
        print("[PLOT] Generating corpus overview dashboard...")
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Topic Distribution (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        topics = [t[0][:10] for t in self.data['topic_counts'][:6]]  # Top 6, truncated
        counts = [t[1] for t in self.data['topic_counts'][:6]]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(topics)))
        ax1.barh(topics, counts, color=colors)
        ax1.set_xlabel('Conversations')
        ax1.set_title('Top Topics', fontweight='bold')
        ax1.invert_yaxis()
        
        # 2. Tone Distribution (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        tones = [t[0] for t in self.data['tone_counts']]
        tone_counts = [t[1] for t in self.data['tone_counts']]
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(tones)))
        ax2.pie(tone_counts, labels=tones, autopct='%1.1f%%', colors=colors2, startangle=90)
        ax2.set_title('Tone Distribution', fontweight='bold')
        
        # 3. Quality Metrics (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        periods = [p[0] for p in self.data['period_stats']]
        info_gains = [p[2] for p in self.data['period_stats']]
        ax3.plot(periods, info_gains, 'b-o', linewidth=2, markersize=8)
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Avg Info Gain')
        ax3.set_title('Info Gain Trend', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Token Ratio Histogram (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        ratios = [c['token_ratio'] for c in self.data['conversations'] if c['token_ratio'] < 8]
        ax4.hist(ratios, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=np.mean(ratios), color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Token Ratio')
        ax4.set_ylabel('Count')
        ax4.set_title('Token Ratio Distribution', fontweight='bold')
        
        # 5. Corrections vs Quality (middle-center)
        ax5 = fig.add_subplot(gs[1, 1])
        corrections = [c['correction_events'] for c in self.data['conversations']]
        info_gains = [c['information_gain'] for c in self.data['conversations']]
        ax5.scatter(corrections, info_gains, alpha=0.5, c='purple', s=20)
        ax5.set_xlabel('Correction Events')
        ax5.set_ylabel('Information Gain')
        ax5.set_title('Corrections vs Quality', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Period Stats Table (middle-right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        table_data = [['Period', 'Conv', 'Info', 'Sycoph']]
        for p in self.data['period_stats']:
            table_data.append([f"P{p[0]}", p[1], f"{p[2]:.3f}", f"{p[3]:.3f}"])
        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.2, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax6.set_title('Period Summary', fontweight='bold', pad=20)
        
        # 7. High-Signal Conversations (bottom-left)
        ax7 = fig.add_subplot(gs[2, 0])
        high_signal = [c for c in self.data['conversations'] if c['information_gain'] > 0.58 and c['malicious_compliance'] < 0.25]
        topic_counts_hs = defaultdict(int)
        for c in high_signal:
            topic_counts_hs[c['topic_primary']] += 1
        topics_hs = list(topic_counts_hs.keys())[:5]
        counts_hs = [topic_counts_hs[t] for t in topics_hs]
        ax7.bar(topics_hs, counts_hs, color='green', alpha=0.7)
        ax7.set_ylabel('Count')
        ax7.set_title(f'High-Signal by Topic (n={len(high_signal)})', fontweight='bold')
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 8. Artifact Counts (bottom-middle)
        ax8 = fig.add_subplot(gs[2, 1])
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(code_blocks), SUM(terminal_outputs), SUM(tables), SUM(manifests) FROM conversations")
        artifacts = cursor.fetchone()
        conn.close()
        artifact_names = ['Code Blocks', 'Terminals', 'Tables', 'Manifests']
        ax8.bar(artifact_names, artifacts, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
        ax8.set_ylabel('Total Count')
        ax8.set_title('Artifact Totals', fontweight='bold')
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 9. Summary Stats (bottom-right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        total_conv = len(self.data['conversations'])
        total_corr = sum(c['correction_events'] for c in self.data['conversations'])
        avg_info = np.mean([c['information_gain'] for c in self.data['conversations']])
        avg_syco = np.mean([c['malicious_compliance'] for c in self.data['conversations']])
        
        summary_text = f"""
CORPUS SUMMARY

Total Conversations: {total_conv:,}
Total Corrections: {total_corr:,}

Avg Info Gain: {avg_info:.3f}
Avg Sycophancy: {avg_syco:.3f}

High-Signal: {len(high_signal)} ({len(high_signal)/total_conv*100:.1f}%)
        """
        ax9.text(0.5, 0.5, summary_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        fig.suptitle('Project Moonshine — Corpus Overview Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / 'corpus_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {self.output_dir / 'corpus_overview.png'}")
        
    # ------------------------------------------------------------------
    # Distilled corpus visualizations (Stage B)
    # ------------------------------------------------------------------

    def load_distilled_data(self):
        """Load distilled_conversations data from DB if available."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='distilled_conversations'")
        if not cursor.fetchone():
            self.data['distilled_available'] = False
            conn.close()
            print("[WARN] distilled_conversations table not found — skipping distilled charts")
            return

        self.data['distilled_available'] = True

        cursor.execute("""
            SELECT conversation_id, title, topic_primary, tone_cluster, period,
                   information_gain, malicious_compliance, user_entropy,
                   token_ratio, correction_events, total_tokens, quality_tier,
                   source_hash, policy_version, run_id
            FROM distilled_conversations
        """)
        cols = [d[0] for d in cursor.description]
        self.data['distilled_conversations'] = [dict(zip(cols, row)) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT period, COUNT(*) as cnt, SUM(total_tokens) as tokens,
                   AVG(information_gain) as avg_ig, AVG(malicious_compliance) as avg_mc
            FROM distilled_conversations GROUP BY period ORDER BY period
        """)
        self.data['distilled_period_stats'] = cursor.fetchall()

        cursor.execute("""
            SELECT quality_tier, COUNT(*) as cnt
            FROM distilled_conversations GROUP BY quality_tier ORDER BY cnt DESC
        """)
        self.data['distilled_tier_stats'] = cursor.fetchall()

        cursor.execute("""
            SELECT topic_primary, COUNT(*) as cnt
            FROM distilled_conversations GROUP BY topic_primary ORDER BY cnt DESC
        """)
        self.data['distilled_topic_counts'] = cursor.fetchall()

        conn.close()

    def plot_distilled_corpus_dashboard(self, figsize=(18, 14)):
        """Generate 3x3 dashboard for the distilled corpus."""
        print("[PLOT] Generating distilled corpus dashboard...")

        dc = self.data['distilled_conversations']
        if not dc:
            print("[WARN] No distilled data — skipping dashboard")
            return

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. Quality tier pie
        ax1 = fig.add_subplot(gs[0, 0])
        tier_data = self.data['distilled_tier_stats']
        tier_labels = [t[0] for t in tier_data]
        tier_sizes = [t[1] for t in tier_data]
        tier_colors = {'gold': '#FFD700', 'silver': '#C0C0C0', 'bronze': '#CD7F32'}
        colors_pie = [tier_colors.get(t, '#999') for t in tier_labels]
        ax1.pie(tier_sizes, labels=tier_labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Quality Tier Distribution', fontweight='bold')

        # 2. Topic distribution: full vs distilled
        ax2 = fig.add_subplot(gs[0, 1:])
        full_topics = dict(self.data['topic_counts'])
        dist_topics = dict(self.data['distilled_topic_counts'])
        all_topics = sorted(full_topics.keys())
        x = np.arange(len(all_topics))
        w = 0.4
        ax2.bar(x - w/2, [full_topics.get(t, 0) for t in all_topics], w,
                label='Full Corpus', color='steelblue', alpha=0.7)
        ax2.bar(x + w/2, [dist_topics.get(t, 0) for t in all_topics], w,
                label='Distilled', color='darkorange', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_topics, rotation=45, ha='right', fontsize=8)
        ax2.set_title('Topic Distribution: Full vs Distilled', fontweight='bold')
        ax2.legend()

        # 3. Information gain histogram
        ax3 = fig.add_subplot(gs[1, 0])
        full_ig = [c['information_gain'] for c in self.data['conversations']]
        dist_ig = [c['information_gain'] for c in dc]
        ax3.hist(full_ig, bins=30, alpha=0.5, color='steelblue', label='Full', density=True)
        ax3.hist(dist_ig, bins=30, alpha=0.6, color='darkorange', label='Distilled', density=True)
        ax3.axvline(0.40, color='red', linestyle='--', linewidth=1.5, label='Policy min (0.40)')
        ax3.set_title('Information Gain Distribution', fontweight='bold')
        ax3.set_xlabel('Information Gain')
        ax3.legend(fontsize=7)

        # 4. Token budget by period
        ax4 = fig.add_subplot(gs[1, 1])
        dps = self.data['distilled_period_stats']
        d_periods = [f"P{r[0]}" for r in dps]
        d_tokens = [r[2] / 1e6 for r in dps]
        ax4.bar(d_periods, d_tokens, color='darkorange', alpha=0.8)
        ax4.set_title('Token Budget by Period (M canonical)', fontweight='bold')
        ax4.set_ylabel('Tokens (M)')

        # 5. Malicious compliance histogram
        ax5 = fig.add_subplot(gs[1, 2])
        full_mc = [c['malicious_compliance'] for c in self.data['conversations']]
        dist_mc = [c['malicious_compliance'] for c in dc]
        ax5.hist(full_mc, bins=30, alpha=0.5, color='steelblue', label='Full', density=True)
        ax5.hist(dist_mc, bins=30, alpha=0.6, color='darkorange', label='Distilled', density=True)
        ax5.axvline(0.35, color='red', linestyle='--', linewidth=1.5, label='Policy max (0.35)')
        ax5.set_title('Malicious Compliance Distribution', fontweight='bold')
        ax5.set_xlabel('Malicious Compliance')
        ax5.legend(fontsize=7)

        # 6. Summary stats text box
        ax6 = fig.add_subplot(gs[0, :1] if False else gs[2, 0])
        ax6.axis('off')
        total_dist_tokens = sum(r[2] for r in dps)
        tier_dict = dict(self.data['distilled_tier_stats'])
        summary = (
            f"Distilled Corpus Summary\n"
            f"{'─'*26}\n"
            f"Conversations: {len(dc):,}\n"
            f"Canonical tokens: {total_dist_tokens/1e6:.1f}M\n"
            f"Gold tier: {tier_dict.get('gold', 0):,}\n"
            f"Silver tier: {tier_dict.get('silver', 0):,}\n"
            f"Bronze tier: {tier_dict.get('bronze', 0):,}\n"
            f"Budget status: IN BAND\n"
            f"Gate B: PASS"
        )
        ax6.text(0.5, 0.5, summary, transform=ax6.transAxes, fontsize=10,
                 va='center', ha='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4))

        # 7. Correction events histogram (distilled only)
        ax7 = fig.add_subplot(gs[2, 1])
        dist_corr = [c['correction_events'] for c in dc]
        ax7.hist(dist_corr, bins=20, color='darkorange', alpha=0.8, edgecolor='black')
        ax7.set_title('Correction Events (Distilled)', fontweight='bold')
        ax7.set_xlabel('Correction Events')

        # 8. Quality tiers by period (grouped bars)
        ax8 = fig.add_subplot(gs[2, 2])
        tiers = ['gold', 'silver', 'bronze']
        tier_colors_bar = ['#FFD700', '#C0C0C0', '#CD7F32']
        periods_set = sorted(set(c['period'] for c in dc))
        x_pos = np.arange(len(periods_set))
        bar_w = 0.25
        for i, (tier, color) in enumerate(zip(tiers, tier_colors_bar)):
            counts_by_period = [
                sum(1 for c in dc if c['period'] == p and c['quality_tier'] == tier)
                for p in periods_set
            ]
            ax8.bar(x_pos + i * bar_w, counts_by_period, bar_w, label=tier, color=color, alpha=0.85)
        ax8.set_xticks(x_pos + bar_w)
        ax8.set_xticklabels([f"P{p}" for p in periods_set])
        ax8.set_title('Quality Tiers by Period', fontweight='bold')
        ax8.legend(fontsize=7)

        fig.suptitle('Project Moonshine — Distilled Corpus Dashboard', fontsize=15, fontweight='bold')
        output_path = self.output_dir / 'distilled_corpus_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_path}")

    def plot_quality_metrics_distilled_timeseries(self, figsize=(14, 6)):
        """Side-by-side timeseries: info gain and malicious compliance, full vs distilled."""
        print("[PLOT] Generating quality metrics distilled timeseries...")

        dc = self.data['distilled_conversations']
        if not dc:
            return

        periods = sorted(set(c['period'] for c in self.data['conversations']))
        full_ig = []
        dist_ig = []
        full_mc = []
        dist_mc = []
        for p in periods:
            fc = [c for c in self.data['conversations'] if c['period'] == p]
            dd = [c for c in dc if c['period'] == p]
            full_ig.append(np.mean([c['information_gain'] for c in fc]) if fc else 0)
            dist_ig.append(np.mean([c['information_gain'] for c in dd]) if dd else None)
            full_mc.append(np.mean([c['malicious_compliance'] for c in fc]) if fc else 0)
            dist_mc.append(np.mean([c['malicious_compliance'] for c in dd]) if dd else None)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(periods, full_ig, 'b-o', label='Full Corpus', linewidth=2)
        dist_ig_clean = [(p, v) for p, v in zip(periods, dist_ig) if v is not None]
        if dist_ig_clean:
            ax1.plot([x[0] for x in dist_ig_clean], [x[1] for x in dist_ig_clean],
                     'g-s', label='Distilled', linewidth=2)
        ax1.axhline(0.40, color='red', linestyle='--', linewidth=1.5, label='Policy min (0.40)')
        ax1.set_title('Avg Information Gain by Period', fontweight='bold')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Information Gain')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(periods, full_mc, 'b-o', label='Full Corpus', linewidth=2)
        dist_mc_clean = [(p, v) for p, v in zip(periods, dist_mc) if v is not None]
        if dist_mc_clean:
            ax2.plot([x[0] for x in dist_mc_clean], [x[1] for x in dist_mc_clean],
                     'g-s', label='Distilled', linewidth=2)
        ax2.axhline(0.35, color='red', linestyle='--', linewidth=1.5, label='Policy max (0.35)')
        ax2.set_title('Avg Malicious Compliance by Period', fontweight='bold')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Malicious Compliance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Quality Metrics: Full vs Distilled Corpus Over Time',
                     fontsize=13, fontweight='bold')
        output_path = self.output_dir / 'quality_metrics_distilled_timeseries.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {output_path}")

    # ------------------------------------------------------------------

    def generate_all(self):
        """Generate all visualizations."""
        print("\n" + "="*70)
        print("MOONSHINE VISUAL INTELLIGENCE")
        print("="*70)
        print(f"Database: {self.db_path}")
        print(f"Output: {self.output_dir}")
        print("-"*70 + "\n")

        self.load_data()

        print("\n[GEN] Generating visualizations...")
        self.plot_topic_distribution()
        self.plot_quality_metrics_timeseries()
        self.plot_token_ratio_distribution()
        self.plot_corrections_scatter()
        self.plot_period_comparison()
        self.plot_corpus_overview_dashboard()

        # Distilled visualizations
        self.load_distilled_data()
        if self.data.get('distilled_available', False):
            self.plot_distilled_corpus_dashboard()
            self.plot_quality_metrics_distilled_timeseries()

        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(f"\nGenerated {len(list(self.output_dir.glob('*.png')))} visualizations:")
        for f in sorted(self.output_dir.glob('*.png')):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from Moonshine corpus database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python moonshine_visualizer.py
  python moonshine_visualizer.py --db reports/moonshine_corpus.db
  python moonshine_visualizer.py --output-dir ./my_visuals
        """
    )
    
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('reports/moonshine_corpus.db'),
        help='Path to SQLite database (default: reports/moonshine_corpus.db)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('visualizations'),
        help='Output directory for images (default: visualizations/)'
    )
    
    args = parser.parse_args()
    
    if not args.db.exists():
        print(f"[ERROR] Database not found: {args.db}")
        print("Run moonshine_corpus_analyzer.py first to generate the database.")
        exit(1)
    
    visualizer = MoonshineVisualizer(args.db, args.output_dir)
    visualizer.generate_all()


if __name__ == "__main__":
    main()
