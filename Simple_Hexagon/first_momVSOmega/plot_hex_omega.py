#!/usr/bin/env python3
# plot_hex_omega.py
# Usage: python3 plot_hex_omega.py --csv results_hexagonal_t1e17.csv --alpha 0.3
# python3 plot_hex_omega.py --csv results_hexagonal_t1e17.csv --alpha 0.3 --out graph_hex_omega.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Argument Parsing
    ap = argparse.ArgumentParser(description="Plot <x> vs R (Omega proxy) for QTM in Simple Hexagonal 3D.")
    ap.add_argument("--csv", required=True, default="results_hexagonal_t1e17.csv", help="Input CSV from C++ engine")
    ap.add_argument("--alpha", type=float, required=True, default=0.3, help="Anomalous exponent alpha")
    ap.add_argument("--out", default="graph_hex_omega.pdf", help="Output plot filename")
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

    required_cols = {"R", "F_input", "average_x"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[ERROR] CSV missing required columns. Found: {df.columns}")

    # 4. Plot Generation
    fig, ax = plt.subplots()
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    unique_forces = np.sort(df['F_input'].unique())
    # The theoretical slope for <x> vs R is -2(1 - alpha)
    theoretical_slope = -2.0 * (1.0 - args.alpha)
    
    print(f"--- Analysis Report (alpha={args.alpha}) ---")
    print(f"Theoretical scaling: <x> ~ R^{theoretical_slope:.2f}")

    # Plot data series for each force
    for idx, F_val in enumerate(unique_forces):
        df_F = df[df['F_input'] == F_val].sort_values('R')
        R_vals = df_F['R'].values
        x_vals = df_F['average_x'].values
        
        ax.plot(R_vals, x_vals, marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                linestyle='-', label=rf'$F = {F_val}$')
        
        # Calculate empirical slope using least squares (log-log space)
        if len(R_vals) >= 2:
            log_R = np.log(R_vals)
            log_x = np.log(x_vals)
            A = np.vstack([log_R, np.ones_like(log_R)]).T
            m, c = np.linalg.lstsq(A, log_x, rcond=None)[0]
            print(f"Empirical slope for F={F_val}: {m:.4f}")

    # 5. Overlay Theoretical Reference Line
    # Position the dashed line above the highest data point series
    max_x = df['average_x'].max()
    R_min = df['R'].min()
    R_max = df['R'].max()
    
    R_dense = np.logspace(np.log10(R_min), np.log10(R_max), 50)
    # y = A * R^(slope). Solve for A such that line starts slightly above max data point
    A_ref = (max_x * 1.5) / (R_min ** theoretical_slope)
    x_theory = A_ref * (R_dense ** theoretical_slope)
    
    ax.plot(R_dense, x_theory, '--', color='black', linewidth=2.0, 
            label=rf'$\propto R^{{{theoretical_slope:.1f}}}$')

    # 6. Axis Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$\langle \mathbf{x}(t) \rangle$')
    
    # Set discrete ticks for R since they are integers
    R_ticks = np.sort(df['R'].unique())
    ax.set_xticks(R_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.legend(frameon=True, loc='best')
    plt.tight_layout()

    # 7. Export
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\n[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()