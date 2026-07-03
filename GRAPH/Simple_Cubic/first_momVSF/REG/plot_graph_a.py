#!/usr/bin/env python3
# plot_graph_a.py
# Usage: python3 plot_graph_a.py --csv results_graph_a.csv --alpha 0.3
# python3 plot_graph_a.py --csv results_graph_a.csv --alpha 0.3 --out graph_a.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Argument Parsing
    ap = argparse.ArgumentParser(description="Plot <x> vs F for QTM in Geometry A.")
    ap.add_argument("--csv", required=True, default="results_graph_a.csv", help="Input CSV from Engine 3")
    ap.add_argument("--alpha", type=float, required=True, default=0.3, help="Anomalous exponent alpha")
    ap.add_argument("--out", default="graph_a_force_dependence.pdf", help="Output plot filename")
    args = ap.parse_args()

    # 2. Matplotlib configuration for Physical Review style
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "lines.linewidth": 1.5,
        "lines.markersize": 10,
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
    
    # Matching the specific markers and colors from Image (a)
    # w=5 (orange triangle), w=10 (green star), w=25 (red circle)
    markers = ['^', '*', 'o', 's', 'D']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4', '#9467bd']
    
    unique_widths = np.sort(df['w'].unique())
    theoretical_slope = args.alpha
    
    print(f"--- Analysis Report (alpha = {args.alpha}) ---")
    print(f"Theoretical scaling: <x> ~ F^{theoretical_slope:.6f}")

    # Plot data series for each width
    for idx, w_val in enumerate(unique_widths):
        df_w = df[df['w'] == w_val].sort_values('F')
        F_vals = df_w['F'].values
        x_vals = df_w['avg_x'].values
        
        ax.plot(F_vals, x_vals, marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                linestyle='-', label=rf'$w = {int(w_val)}$')
        
        # Calculate empirical slope using least squares in log-log space
        if len(F_vals) >= 2:
            log_F = np.log(F_vals)
            log_x = np.log(x_vals)
            A = np.vstack([log_F, np.ones_like(log_F)]).T
            m, c = np.linalg.lstsq(A, log_x, rcond=None)[0]
            print(f"Empirical slope for w={int(w_val)}: {m:.6f}")

    # 5. Overlay Theoretical Reference Line
    max_x = df['avg_x'].max()
    F_min = df['F'].min()
    F_max = df['F'].max()
    
    F_dense = np.logspace(np.log10(F_min), np.log10(F_max), 50)
    
    # Position the reference line slightly above the highest data point series
    A_ref = (max_x * 1.5) / (F_max ** theoretical_slope)
    x_theory = A_ref * (F_dense ** theoretical_slope)
    
    ax.plot(F_dense, x_theory, '--', color='black', linewidth=2.0, 
            label=rf'$\propto F^{{{theoretical_slope:.1f}}}$')

    # 6. Axis Formatting
    # Using log-log scale to explicitly show the F^alpha scaling law
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$F$')
    ax.set_ylabel(r'$\langle x \rangle$')
    
    # Position the (a) text box in the upper left
    ax.text(0.15, 0.9, r'(a)', transform=ax.transAxes, fontsize=24, verticalalignment='top')

    ax.legend(frameon=True, loc='best')
    plt.tight_layout()

    # 7. Export
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\n[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()