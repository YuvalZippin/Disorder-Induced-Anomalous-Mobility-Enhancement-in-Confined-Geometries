#!/usr/bin/env python3
# plot_graph_b_hex_exact.py
# Usage: python3 plot_graph_b_hex_exact.py --csv results_geometry_b_hex.csv --alpha 0.3 --out graph_b_hex_width_dependence.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def asymptotic_theory(w, F, alpha, C_global):
    """
    Evaluates the explicit asymptotic scaling theory for Hexagonal Geometry B:
    <x> = C * ( (3*sqrt(3)/2) * w**2 )**(alpha - 1) * F**alpha
    """
    omega = 1.5 * np.sqrt(3.0) * (w**2)
    return C_global * (omega**(alpha - 1.0)) * (F**alpha)

def main():
    ap = argparse.ArgumentParser(description="Plot exact analytical scaling vs Simulation for Geometry B (Width)")
    ap.add_argument("--csv", required=True, default="results_geometry_b_hex.csv", help="Input CSV")
    ap.add_argument("--alpha", type=float, default=0.3, help="Anomalous exponent")
    ap.add_argument("--out", default="graph_b_hex_width_dependence.png", help="Output filename")
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
    
    # Calculate the exact hexagonal area for the anchor point
    omega0 = 1.5 * np.sqrt(3.0) * (w0**2)
    C_global = x0 / ( (omega0**(args.alpha - 1.0)) * (F0**args.alpha) )
    
    theoretical_power = 2.0 * args.alpha - 2.0
    
    print(f"\n[SYSTEM] Calibrated global prefactor C_global = {C_global:.4e}")
    print(f"[SYSTEM] Expected theoretical spatial scaling: <x> ~ w^({theoretical_power:.2f})")
    print("--- Theory vs Simulation: Error Analysis (Hexagonal) ---")

    fig, ax = plt.subplots()
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    unique_F = np.sort(df['F'].unique())
    w_dense = np.logspace(np.log10(df['w'].min()), np.log10(df['w'].max()), 50)

    for idx, F_val in enumerate(unique_F):
        # Group by Force (F) and plot against width (w)
        df_F = df[df['F'] == F_val].sort_values('w')
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        w_sim = df_F['w'].values
        x_sim = df_F['avg_x'].values
        
        # Calculate error for terminal output
        x_th_sim = asymptotic_theory(w_sim, F_val, args.alpha, C_global)
        mape = np.mean(np.abs(x_sim - x_th_sim) / x_th_sim) * 100
        print(f"F = {F_val:<5} | Mean Relative Error (MAPE): {mape:.4f}%")
        
        # Plot Simulation (Points)
        ax.plot(w_sim, x_sim, marker=m, color=c, linestyle='none', 
                label=rf'Sim: $F = {F_val}$', zorder=3)
        
        # Plot Specific Theory matching the points (Colored Dashed Line)
        x_th_dense = asymptotic_theory(w_dense, F_val, args.alpha, C_global)
        ax.plot(w_dense, x_th_dense, color=c, linestyle='--', alpha=0.7, zorder=2)

    # Black Dashed Global Reference Line
    # Positioned above the highest curve for clear visibility of the slope
    highest_y_at_w_min = asymptotic_theory(w_dense[0], unique_F[-1], args.alpha, C_global)
    ref_y = (highest_y_at_w_min * 1.5) * (w_dense / w_dense[0])**(theoretical_power)
    
    ax.plot(w_dense, ref_y, color='black', linestyle='--', linewidth=2.5, 
            label=rf'$\propto w^{{{theoretical_power:.1f}}}$', zorder=1)

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$\langle x \rangle$')
    
    # Explicit integer ticks for the width axis
    w_ticks = np.sort(df['w'].unique())
    ax.set_xticks(w_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Custom Legend handling
    theory_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Exact Theory (Hex)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [theory_line], frameon=True, loc='best')

    print("-------------------------------------------\n")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()