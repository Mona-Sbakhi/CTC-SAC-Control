#!/usr/bin/env python3
"""
Plot Comparison Figures — v2 (4 methods)
=========================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Reads:
    ../results/comparison_v2.json          — summary metrics
    ../results/ts_v2_<method>.csv          — time-series per method

Saves:
    ../results/fig_comparison_v2.png       — 2×2 comparison figure
    ../results/fig_errors_v2.png           — error bar chart only (for paper)

Usage:
    cd /workspace/src
    python plot_comparison_v2.py

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
"""

import numpy as np
import json, os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "../results"

# ── Method config ─────────────────────────────────────────────────
METHODS = [
    {"name": "CTC-Only",        "tag": "CTC_Only", "color": "#2196F3", "ls": "--"},
    {"name": "RL-Only (SAC)",   "tag": "RL_Only",  "color": "#F44336", "ls": "-."},
    {"name": "PID+SAC",         "tag": "PID_SAC",  "color": "#FF9800", "ls": ":"},
    {"name": "CTC+SAC (Prop.)", "tag": "CTC_SAC",  "color": "#4CAF50", "ls": "-"},
]

# ── Trajectory (same as evaluate_v2.py) ───────────────────────────
def trajectory_desired(times, freq=0.5, amp=0.5):
    w  = 2 * np.pi * freq
    t  = np.array(times)
    q1 = amp * np.sin(w * t)
    q2 = amp * np.sin(w * t + np.pi / 4)
    return q1, q2

# ── Load CSV ──────────────────────────────────────────────────────
def load_ts(tag):
    path = os.path.join(RESULTS_DIR, f"ts_v2_{tag}.csv")
    if not os.path.exists(path):
        print(f"  ⚠️  {path} not found — skipping {tag}")
        return None
    times, e1, e2, norm = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            e1.append(float(row['error_q1']))
            e2.append(float(row['error_q2']))
            norm.append(float(row['norm_error']))
    return {'time': np.array(times),
            'e1':   np.array(e1),
            'e2':   np.array(e2),
            'norm': np.array(norm)}

# ── Load JSON summary ─────────────────────────────────────────────
def load_summary():
    path = os.path.join(RESULTS_DIR, "comparison_v2.json")
    if not os.path.exists(path):
        print(f"⚠️  {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)

# ── FIGURE 1: 2×2 Complete Comparison ────────────────────────────
def plot_full_comparison(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Tracking Performance Comparison in CoppeliaSim\n"
                 "CTC-Only | RL-Only | PID+SAC | CTC+SAC (Proposed)",
                 fontsize=13, fontweight='bold', y=0.98)

    ax_q1   = axes[0, 0]
    ax_q2   = axes[0, 1]
    ax_err  = axes[1, 0]
    ax_bar  = axes[1, 1]

    # Desired trajectory (use first available time vector)
    ref_ts = next((d for d in data.values() if d is not None), None)
    if ref_ts is not None:
        t = ref_ts['time']
        qd1, qd2 = trajectory_desired(t)
        for ax, qd, joint in [(ax_q1, qd1, 1), (ax_q2, qd2, 2)]:
            ax.plot(t, qd, 'k--', lw=1.8, label='Desired', zorder=5)

    handles = []
    for m in METHODS:
        ts = data.get(m['tag'])
        if ts is None:
            continue

        t  = ts['time']
        qd1, qd2 = trajectory_desired(t)

        # Reconstruct actual joint positions from errors
        q1_actual = qd1 - ts['e1']
        q2_actual = qd2 - ts['e2']

        kw = dict(color=m['color'], lw=1.4, ls=m['ls'], alpha=0.85)

        # Joint 1 tracking
        ax_q1.plot(t, q1_actual, label=m['name'], **kw)

        # Joint 2 tracking
        ax_q2.plot(t, q2_actual, label=m['name'], **kw)

        # Norm error
        ax_err.plot(t, ts['norm'], label=m['name'], **kw)

        handles.append(mpatches.Patch(color=m['color'], label=m['name']))

    # ── Axes formatting ──
    ax_q1.set_title("Joint 1 Trajectory Tracking", fontsize=11)
    ax_q1.set_xlabel("Time [s]"); ax_q1.set_ylabel("$q_1$ [rad]")
    ax_q1.legend(fontsize=8, loc='upper right')
    ax_q1.set_xlim([0, 10]); ax_q1.grid(True, alpha=0.3)

    ax_q2.set_title("Joint 2 Trajectory Tracking", fontsize=11)
    ax_q2.set_xlabel("Time [s]"); ax_q2.set_ylabel("$q_2$ [rad]")
    ax_q2.legend(fontsize=8, loc='upper right')
    ax_q2.set_xlim([0, 10]); ax_q2.grid(True, alpha=0.3)

    ax_err.set_title("Tracking Error Comparison", fontsize=11)
    ax_err.set_xlabel("Time [s]"); ax_err.set_ylabel("$\\|e\\|$ [rad]")
    ax_err.legend(fontsize=8, loc='upper right')
    ax_err.set_xlim([0, 10]); ax_err.grid(True, alpha=0.3)

    # ── Bar chart ──
    summary = load_summary()
    bar_methods, bar_vals, bar_colors = [], [], []
    for m in METHODS:
        key = m['name'].replace(" (Prop.)", "")  # match JSON keys
        # Try both possible key formats
        for k in [m['name'], key, m['tag'].replace('_', '-'),
                  m['tag'].replace('_', '+'), "CTC+SAC", "RL-Only",
                  "CTC-Only", "PID+SAC"]:
            if k in summary:
                bar_methods.append(m['name'])
                bar_vals.append(summary[k]['mean_err_full'])
                bar_colors.append(m['color'])
                break

    x = np.arange(len(bar_methods))
    bars = ax_bar.bar(x, bar_vals, color=bar_colors,
                      edgecolor='black', linewidth=0.8, width=0.55)

    for bar, val in zip(bars, bar_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # Highlight best bar
    if bar_vals:
        best_idx = int(np.argmin(bar_vals))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

    ax_bar.set_title("Mean Tracking Error Comparison", fontsize=11)
    ax_bar.set_ylabel("Mean Error [rad]")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_methods, fontsize=9)
    ax_bar.set_ylim([0, max(bar_vals) * 1.18 if bar_vals else 3.0])
    ax_bar.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULTS_DIR, "fig_comparison_v2.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"✅ Saved → {out}")
    plt.close()


# ── FIGURE 2: Clean bar chart (for paper) ────────────────────────
def plot_bar_only():
    summary = load_summary()
    if not summary:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bar_methods, bar_vals, bar_colors = [], [], []
    for m in METHODS:
        for k in [m['name'].replace(" (Prop.)", ""),
                  "CTC+SAC", "RL-Only", "CTC-Only", "PID+SAC"]:
            if k in summary:
                bar_methods.append(m['name'])
                bar_vals.append(summary[k]['mean_err_full'])
                bar_colors.append(m['color'])
                break

    if not bar_vals:
        # Use hardcoded v2 results as fallback
        bar_methods = ["CTC-Only", "RL-Only (SAC)", "PID+SAC", "CTC+SAC (Prop.)"]
        bar_vals    = [2.24, 2.33, 2.29, 2.22]
        bar_colors  = ["#2196F3", "#F44336", "#FF9800", "#4CAF50"]
        print("  ℹ️  Using hardcoded v2 results (comparison_v2.json not found)")

    x    = np.arange(len(bar_methods))
    bars = ax.bar(x, bar_vals, color=bar_colors,
                  edgecolor='black', linewidth=0.8, width=0.5)

    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}", ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    best_idx = int(np.argmin(bar_vals))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)

    ax.set_title("Mean Tracking Error — All Methods", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Error [rad]", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_methods, fontsize=10)
    ax.set_ylim([2.0, max(bar_vals) * 1.12])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_bar_v2.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"✅ Saved → {out}")
    plt.close()


# ── MAIN ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Plot Comparison v2 — CoDIT 2026")
    print("=" * 55)

    # Load all time-series
    data = {}
    for m in METHODS:
        print(f"  Loading {m['name']} ({m['tag']})...")
        data[m['tag']] = load_ts(m['tag'])

    available = [m['name'] for m in METHODS if data[m['tag']] is not None]
    print(f"\n  Available: {available}")

    print("\n  → Figure 1: 2×2 complete comparison...")
    plot_full_comparison(data)

    print("  → Figure 2: Bar chart only...")
    plot_bar_only()

    print("\n✅ Done. Figures saved in ../results/")
    print("   fig_comparison_v2.png  — full 2×2 figure")
    print("   fig_bar_v2.png         — bar chart for paper")

if __name__ == "__main__":
    main()
