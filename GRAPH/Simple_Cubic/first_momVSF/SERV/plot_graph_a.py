#!/usr/bin/env python3
# plot_graph_a.py
# Usage: python3 plot_graph_a.py --csv results_graph_a.csv --alpha 0.3 --out graph_a_exact.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def asymptotic_theory(w, F, alpha, C_global):
    """
    Evaluates the explicit asymptotic scaling theory:
    <x> = C * w**(2*alpha - 2) * F**alpha
    """
    return C_global * (w**(2.0*alpha - 2.0)) * (F**alpha)

def main():
    ap = argparse.ArgumentParser(description="Plot exact analytical scaling vs Simulation for Geometry A (Force)")
    ap.add_argument("--csv", required=True, default="results_graph_a.csv", help="Input CSV")
    ap.add_argument("--alpha", type=float, default=0.3, help="Anomalous exponent")
    ap.add_argument("--out", default="graph_a_exact.png", help="Output filename")
    args = ap.parse_args()

    # Academic Physical Review styling
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 10,
        "figure.figsize": (7, 5) # Single column format
    })

    try:
        df = pd.read_csv(args.csv)
        print(f"[SYSTEM] Loaded data from {args.csv}")
    except FileNotFoundError:
        print(f"[ERROR] CSV not found. Run the C++ engine first.")
        return

    # --- Anchor the Theory ---
    # Extract the prefactor C_global from the first data point to anchor the analytical curves
    df_sorted = df.sort_values(['w', 'F'])
    w0, F0, x0 = df_sorted['w'].iloc[0], df_sorted['F'].iloc[0], df_sorted['avg_x'].iloc[0]
    C_global = x0 / ( (w0**(2.0*args.alpha - 2.0)) * (F0**args.alpha) )
    print(f"\n[SYSTEM] Calibrated global prefactor C_global = {C_global:.4e}")
    print("--- Theory vs Simulation: Error Analysis ---")

    fig, ax = plt.subplots()
    
    markers = ['^', '*', 'o', 's', 'D']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4', '#9467bd']
    
    unique_w = np.sort(df['w'].unique())
    F_dense = np.logspace(np.log10(df['F'].min()), np.log10(df['F'].max()), 50)

    for idx, w_val in enumerate(unique_w):
        df_w = df[df['w'] == w_val].sort_values('F')
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        F_sim = df_w['F'].values
        x_sim = df_w['avg_x'].values
        
        # Calculate error for terminal output
        x_th_sim = asymptotic_theory(w_val, F_sim, args.alpha, C_global)
        mape = np.mean(np.abs(x_sim - x_th_sim) / x_th_sim) * 100
        print(f"w = {int(w_val):<4} | Mean Relative Error (MAPE): {mape:.4f}%")
        
        # Plot Simulation (Points)
        ax.plot(F_sim, x_sim, marker=m, color=c, linestyle='none', 
                label=rf'Sim: $w = {int(w_val)}$', zorder=3)
        
        # Plot Specific Theory matching the points (Colored Dashed Line)
        x_th_dense = asymptotic_theory(w_val, F_dense, args.alpha, C_global)
        ax.plot(F_dense, x_th_dense, color=c, linestyle='--', alpha=0.7, zorder=2)

    # Black Dashed Global Reference Line (shifted up for visibility)
    ref_y = (df['avg_x'].max() * 1.5) * (F_dense / F_dense[-1])**(args.alpha)
    ax.plot(F_dense, ref_y, color='black', linestyle='--', linewidth=2.5, 
            label=rf'$\propto F^{{{args.alpha}}}$')

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$F$')
    ax.set_ylabel(r'$\langle x \rangle$')
    
    # Custom Legend handling
    theory_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Explicit Theory')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [theory_line], frameon=True, loc='best')

    print("-------------------------------------------\n")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()