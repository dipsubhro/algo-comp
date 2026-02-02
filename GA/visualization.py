"""
Visualization for GA Results
───────────────────────────────
format_number() andar hi define hai — menu.py se kuch pass karne ki zaroorat nahi.
Har graph label mein same decimal format use hota hai.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from config import FIGURE_SIZE, DPI
from ga_algorithm import calculate_statistics


# ═══════════════════════════════════════════════════════════════
#  format_number — menu waala hi exact logic
# ═══════════════════════════════════════════════════════════════
def format_number(value):
    """Convert number to decimal format with sufficient precision"""
    if value == 0:
        return "0.00000000"
    elif abs(value) < 1e-10:
        return f"{value:.20f}".rstrip('0').rstrip('.')
    elif abs(value) < 1e-6:
        return f"{value:.16f}".rstrip('0').rstrip('.')
    elif abs(value) < 0.001:
        return f"{value:.12f}".rstrip('0').rstrip('.')
    elif abs(value) < 1:
        return f"{value:.8f}".rstrip('0').rstrip('.')
    elif abs(value) < 100:
        return f"{value:.6f}".rstrip('0').rstrip('.')
    else:
        return f"{value:.4f}".rstrip('0').rstrip('.')


# ═══════════════════════════════════════════════════════════════
#  SINGLE FUNCTION
#  3 graphs: Convergence | Distribution | Stats Bar
# ═══════════════════════════════════════════════════════════════
def plot_individual_function(func_name, results, history, stats):
    fig = plt.figure(figsize=(18, 5.5))
    fig.patch.set_facecolor('#16162a')
    gs = gridspec.GridSpec(1, 3, wspace=0.30)

    C_CONV   = '#2E86AB'
    C_DIST   = '#A23B72'
    C_MEAN   = '#E94F37'
    C_MED    = '#44BBA4'

    # ── 1. Convergence ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1a1a2e')
    ax1.plot(history, linewidth=2.2, color=C_CONV, solid_capstyle='round')
    ax1.fill_between(range(len(history)), history, alpha=0.12, color=C_CONV)
    ax1.set_title(f'{func_name} — Convergence', fontweight='bold', fontsize=13, color='#eee')
    ax1.set_xlabel('Generation', fontsize=10, color='#aaa')
    ax1.set_ylabel('Fitness', fontsize=10, color='#aaa')
    ax1.tick_params(colors='#aaa', labelsize=8)
    ax1.grid(True, alpha=0.15, color='#fff')
    for sp in ax1.spines.values():
        sp.set_color('#333')

    # ── 2. Distribution ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#1a1a2e')
    ax2.hist(results, bins=15, color=C_DIST, alpha=0.75, edgecolor='#16162a', linewidth=1.2)
    ax2.axvline(stats['mean'],   color=C_MEAN, linestyle='--', linewidth=2,
                label=f"Mean: {format_number(stats['mean'])}")
    ax2.axvline(stats['median'], color=C_MED,  linestyle='--', linewidth=2,
                label=f"Median: {format_number(stats['median'])}")
    ax2.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc', fontsize=9)
    ax2.set_title(f'{func_name} — Distribution', fontweight='bold', fontsize=13, color='#eee')
    ax2.set_xlabel('Fitness', fontsize=10, color='#aaa')
    ax2.set_ylabel('Frequency', fontsize=10, color='#aaa')
    ax2.tick_params(colors='#aaa', labelsize=8)
    ax2.grid(True, alpha=0.15, color='#fff')
    for sp in ax2.spines.values():
        sp.set_color('#333')

    # ── 3. Stats Bar (Min | Mean | Median | Std | Max) ──────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#1a1a2e')

    labels = ['Min', 'Mean', 'Median', 'Std Dev', 'Max']
    values = [stats['min'], stats['mean'], stats['median'], stats['std'], stats['max']]
    colors = ['#2E86AB', '#E94F37', '#44BBA4', '#F18F01', '#A23B72']

    bars = ax3.barh(labels, values, color=colors, edgecolor='#16162a', linewidth=1, height=0.55)

    max_val = max((v for v in values if v != 0), default=1)
    for bar, val in zip(bars, values):
        ax3.text(
            bar.get_width() + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            format_number(val),
            va='center', ha='left',
            fontsize=8.5, color='#ccc', fontweight='bold'
        )

    ax3.set_xlim(0, max_val * 1.45)
    ax3.set_title(f'{func_name} — Stats', fontweight='bold', fontsize=13, color='#eee')
    ax3.set_xlabel('Value', fontsize=10, color='#aaa')
    ax3.tick_params(colors='#aaa', labelsize=8.5)
    ax3.grid(True, alpha=0.15, color='#fff', axis='x')
    for sp in ax3.spines.values():
        sp.set_color('#333')
    ax3.invert_yaxis()

    fig.suptitle(f'GA Results — {func_name}', fontsize=16, fontweight='bold',
                 color='#eee', y=1.02)

    filename = f'{func_name}_results.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════
#  ALL FUNCTIONS
#  5 graphs: Min | Mean | Median | Std | Convergence top-5
# ═══════════════════════════════════════════════════════════════
def plot_all_functions_combined(all_results, plot_data):
    # ── pre-compute ─────────────────────────────────────────────
    stats_dict = {name: calculate_statistics(res) for name, res in all_results.items()}
    sorted_funcs = sorted(stats_dict.items(), key=lambda x: x[1]['mean'])
    names   = [f[0] for f in sorted_funcs]
    mins    = [stats_dict[n]['min']    for n in names]
    means   = [stats_dict[n]['mean']   for n in names]
    medians = [stats_dict[n]['median'] for n in names]
    stds    = [stats_dict[n]['std']    for n in names]

    # 3 rows x 2 cols = 6 slots, last slot (row3 col2) khali
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#16162a')
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25)

    C_MIN  = '#2E86AB'
    C_MEAN = '#E94F37'
    C_MED  = '#44BBA4'
    C_STD  = '#F18F01'

    # ── helper: horizontal bar chart + format_number labels ─────
    def _draw_hbar(ax, title, values, color):
        ax.set_facecolor('#1a1a2e')
        bars = ax.barh(names, values, color=color, edgecolor='#16162a', linewidth=0.8, height=0.6)

        max_val = max((v for v in values if v != 0), default=1)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max_val * 0.015,
                bar.get_y() + bar.get_height() / 2,
                format_number(val),
                va='center', ha='left',
                fontsize=7.5, color='#ccc', fontweight='bold'
            )
        ax.set_xlim(0, max_val * 1.38)
        ax.set_title(title, fontweight='bold', fontsize=13, color='#eee')
        ax.set_xlabel('Value', fontsize=10, color='#aaa')
        ax.tick_params(colors='#aaa', labelsize=8)
        ax.grid(True, alpha=0.15, color='#fff', axis='x')
        for sp in ax.spines.values():
            sp.set_color('#333')
        ax.invert_yaxis()

    # ── 1. Min ──────────────────────────────────────────────────
    _draw_hbar(fig.add_subplot(gs[0, 0]), 'Best Fitness (Min)',      mins,    C_MIN)

    # ── 2. Mean ─────────────────────────────────────────────────
    _draw_hbar(fig.add_subplot(gs[0, 1]), 'Average Fitness (Mean)', means,   C_MEAN)

    # ── 3. Median ───────────────────────────────────────────────
    _draw_hbar(fig.add_subplot(gs[1, 0]), 'Median Fitness',         medians, C_MED)

    # ── 4. Std Dev ──────────────────────────────────────────────
    _draw_hbar(fig.add_subplot(gs[1, 1]), 'Consistency (Std Dev)',  stds,    C_STD)

    # ── 5. Convergence — top 5 (full width, bottom row) ────────
    ax5 = fig.add_subplot(gs[2, :])   # colspan=2 — pura bottom row
    ax5.set_facecolor('#1a1a2e')
    top5 = names[:5]
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, 5))
    for i, fname in enumerate(top5):
        if fname in plot_data:
            ax5.plot(plot_data[fname], label=fname,
                     color=cmap[i], linewidth=2.5 if i == 0 else 1.8, alpha=0.9)
    ax5.set_title('Convergence — Top 5 Best', fontweight='bold', fontsize=13, color='#eee')
    ax5.set_xlabel('Generation', fontsize=10, color='#aaa')
    ax5.set_ylabel('Fitness', fontsize=10, color='#aaa')
    ax5.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc', fontsize=8.5)
    ax5.tick_params(colors='#aaa', labelsize=8)
    ax5.grid(True, alpha=0.15, color='#fff')
    for sp in ax5.spines.values():
        sp.set_color('#333')

    fig.suptitle('GA Benchmark — All 15 Functions', fontsize=20, fontweight='bold',
                 color='#eee', y=0.995)

    filename = 'all_functions_comparison.png'
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {filename}")