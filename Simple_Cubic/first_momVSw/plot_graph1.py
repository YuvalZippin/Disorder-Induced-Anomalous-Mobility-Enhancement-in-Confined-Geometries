#!/usr/bin/env python3
# plot_graph1.py
# Usage: python3 plot_graph1.py --csv results_geometry_a.csv --alpha 0.3
# python3 plot_graph1.py --csv results_geometry_a.csv --alpha 0.3 --out graph_a.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Argument Parsing
    ap = argparse.ArgumentParser(description="Plot <x> vs w for QTM in Geometry A.")
    ap.add_argument("--csv", required=True, default="results_geometry_a.csv", help="Input CSV from C++ engine")
    ap.add_argument("--alpha", type=float, required=True, default=0.3, help="Anomalous exponent alpha")
    ap.add_argument("--out", default="graph1_geometry_a.pdf", help="Output plot filename")
    args = ap.parse_args()

    # 2. Matplotlib configuration for Physical Review style
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "lines.linewidth": 1.5,
        "lines.markersize": 9,
        "figure.figsize": (7, 5)
    })

    # 3. Data Ingestion
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {args.csv}. Run the C++ engine first.")
        return

    required_cols = {"w", "F", "avg_x"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[ERROR] CSV missing required columns. Found: {df.columns}")

    # 4. Plot Generation
    fig, ax = plt.subplots()
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    unique_forces = np.sort(df['F'].unique())
    theoretical_slope = 2.0 * args.alpha - 2.0  # -1.4 for alpha=0.3
    
    print(f"--- Analysis Report (alpha={args.alpha}) ---")
    print(f"Theoretical scaling: <x> ~ w^{theoretical_slope:.2f}")

    # Plot data series for each force
    for idx, F_val in enumerate(unique_forces):
        df_F = df[df['F'] == F_val].sort_values('w')
        w_vals = df_F['w'].values
        x_vals = df_F['avg_x'].values
        
        ax.plot(w_vals, x_vals, marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                linestyle='-', label=rf'$F = {F_val}$')
        
        # Calculate empirical slope using least squares (log-log space)
        if len(w_vals) >= 2:
            log_w = np.log(w_vals)
            log_x = np.log(x_vals)
            A = np.vstack([log_w, np.ones_like(log_w)]).T
            m, c = np.linalg.lstsq(A, log_x, rcond=None)[0]
            print(f"Empirical slope for F={F_val}: {m:.4f}")

    # 5. Overlay Theoretical Reference Line
    # Position the dashed line above the highest data point series
    max_x = df['avg_x'].max()
    w_min = df['w'].min()
    w_max = df['w'].max()
    
    w_dense = np.logspace(np.log10(w_min), np.log10(w_max), 50)
    # y = A * w^(slope). Solve for A such that line starts slightly above max data point
    A_ref = (max_x * 1.5) / (w_min ** theoretical_slope)
    x_theory = A_ref * (w_dense ** theoretical_slope)
    
    ax.plot(w_dense, x_theory, '--', color='black', linewidth=2.0, 
            label=rf'$\propto w^{{{theoretical_slope:.1f}}}$')

    # 6. Axis Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$\langle x \rangle$')
    
    # Optional: Set discrete ticks for w since they are integers
    w_ticks = np.sort(df['w'].unique())
    ax.set_xticks(w_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.legend(frameon=True, loc='best')
    plt.tight_layout()

    # 7. Export
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\n[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()