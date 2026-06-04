#!/usr/bin/env python3
# plot_force_scaling.py
# Usage: python3 plot_force_scaling.py --csv results_hexagonal_t1e17.csv --alpha 0.3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Argument Parsing
    ap = argparse.ArgumentParser(description="Plot <x> vs F for QTM in Simple Hexagonal 3D.")
    ap.add_argument("--csv", required=True, default="results_hexagonal_t1e17.csv", help="Input CSV")
    ap.add_argument("--alpha", type=float, required=True, default=0.3, help="Anomalous exponent alpha")
    ap.add_argument("--out", default="graph_force_scaling.pdf", help="Output plot filename")
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
        print(f"[ERROR] Could not find {args.csv}.")
        return

    # 4. Plot Generation
    fig, ax = plt.subplots()
    
    markers = ['^', '*', 'o', 's', 'D']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4', '#9467bd']
    
    # Select specific radii to plot (matching the visual sparsity of the reference image)
    # The reference image uses 3 widths. We will use R = 5, 10, and 30.
    target_radii = [5, 10, 30]
    
    print(f"--- Analysis Report (alpha={args.alpha}) ---")
    print(f"Theoretical scaling: <x> ~ F^{args.alpha}")

    # Plot data series for each chosen radius
    for idx, R_val in enumerate(target_radii):
        if R_val not in df['R'].values:
            continue
            
        df_R = df[df['R'] == R_val].sort_values('F_input')
        F_vals = df_R['F_input'].values
        x_vals = df_R['average_x'].values
        
        ax.plot(F_vals, x_vals, marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                linestyle='-', label=rf'$R = {R_val}$')
        
        # Calculate empirical slope using least squares (log-log space)
        if len(F_vals) >= 2:
            log_F = np.log(F_vals)
            log_x = np.log(x_vals)
            A = np.vstack([log_F, np.ones_like(log_F)]).T
            m, c = np.linalg.lstsq(A, log_x, rcond=None)[0]
            print(f"Empirical slope for R={R_val}: {m:.4f} (Expected: {args.alpha})")

    # 5. Overlay Theoretical Reference Line (dashed line)
    # Position the dashed line above the highest data point series
    max_x = df['average_x'].max()
    F_min = df['F_input'].min()
    F_max = df['F_input'].max()
    
    F_dense = np.logspace(np.log10(F_min), np.log10(F_max), 50)
    
    # y = A * F^(alpha). Solve for A such that the line starts comfortably above the top data point
    A_ref = (max_x * 1.5) / (F_min ** args.alpha)
    x_theory = A_ref * (F_dense ** args.alpha)
    
    ax.plot(F_dense, x_theory, '--', color='black', linewidth=2.0, 
            label=rf'$\propto F^{{{args.alpha}}}$')

    # 6. Axis Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$F$')
    ax.set_ylabel(r'$\langle x \rangle$')
    
    ax.legend(frameon=True, loc='lower right')
    plt.tight_layout()

    # 7. Export
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\n[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()