#!/usr/bin/env python3
# plot_exact_theory.py
# Usage: python3 plot_exact_theory.py
# python3 plot_exact_theory.py --out exact_theory_comparison.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import io

# Embedded fallback data (Geometry A)
FALLBACK_DATA = """w,F,N_traj,avg_x
1,0.01,100000,7.3081003700000001e+03
1,0.02,100000,8.8950835299999999e+03
1,0.05,100000,1.1243090340000001e+04
3,0.01,100000,1.5834140900000000e+03
3,0.02,100000,1.9505578399999999e+03
3,0.05,100000,2.5805616000000000e+03
5,0.01,100000,7.7498512000000005e+02
5,0.02,100000,9.5388879999999995e+02
5,0.05,100000,1.2614635900000001e+03
7,0.01,100000,4.8534320000000002e+02
7,0.02,100000,5.9740129999999999e+02
7,0.05,100000,7.8846779000000004e+02
9,0.01,100000,3.4208747000000000e+02
9,0.02,100000,4.2005139000000003e+02
9,0.05,100000,5.5728536999999994e+02
"""

def asymptotic_theory(w, F, alpha, C_global):
    """
    Evaluates the explicit user formula:
    <x> = C * w**(2*alpha - 2) * F**alpha
    """
    return C_global * (w**(2.0*alpha - 2.0)) * (F**alpha)

def main():
    ap = argparse.ArgumentParser(description="Plot exact analytical scaling vs Simulation")
    ap.add_argument("--csv", default="results_geometry_a.csv", help="Input CSV")
    ap.add_argument("--alpha", type=float, default=0.3, help="Anomalous exponent")
    ap.add_argument("--out", default="exact_theory_comparison.pdf", help="Output filename")
    args = ap.parse_args()

    # Academic Physical Review styling
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 13,
        "lines.linewidth": 2.0,
        "lines.markersize": 10,
        "figure.figsize": (14, 6)
    })

    try:
        df = pd.read_csv(args.csv)
        print(f"[SYSTEM] Loaded data from {args.csv}")
    except FileNotFoundError:
        print(f"[SYSTEM] CSV not found. Using embedded fallback data.")
        df = pd.read_csv(io.StringIO(FALLBACK_DATA))

    # --- Anchor the Theory ---
    # Extract the prefactor C_global from the first data point (w=1, F=0.01)
    w0, F0, x0 = df['w'].iloc[0], df['F'].iloc[0], df['avg_x'].iloc[0]
    C_global = x0 / ( (w0**(2.0*args.alpha - 2.0)) * (F0**args.alpha) )
    print(f"[SYSTEM] Calibrated global prefactor C_global = {C_global:.4e}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    print("--- Theory vs Simulation: Error Analysis ---")

    # ==========================================
    # GRAPH (a): Force Scaling (<x> vs F) -> Now on the LEFT (ax1)
    # ==========================================
    target_widths = [1, 5, 9]
    df_filtered = df[df['w'].isin(target_widths)]
    unique_w = np.sort(df_filtered['w'].unique())
    F_dense = np.logspace(np.log10(df['F'].min()), np.log10(df['F'].max()), 50)

    for idx, w_val in enumerate(unique_w):
        df_w = df_filtered[df_filtered['w'] == w_val].sort_values('F')
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        F_sim = df_w['F'].values
        x_sim = df_w['avg_x'].values
        
        # Calculate error for terminal output
        x_th_sim = asymptotic_theory(w_val, F_sim, args.alpha, C_global)
        mape = np.mean(np.abs(x_sim - x_th_sim) / x_th_sim) * 100
        print(f"Graph (a) [Force] | w = {int(w_val):<4} | Mean Relative Error: {mape:.4f}%")
        
        # Simulation (Points)
        ax1.plot(F_sim, x_sim, marker=m, color=c, linestyle='none', 
                 label=rf'Sim: $w = {int(w_val)}$', zorder=3)
        
        # Specific Theory matching the points (Colored Dashed Line)
        x_th_dense = asymptotic_theory(w_val, F_dense, args.alpha, C_global)
        ax1.plot(F_dense, x_th_dense, color=c, linestyle='--', alpha=0.7, zorder=2)

    # Black Dashed Reference Line
    ref_y1 = (df_filtered['avg_x'].max() * 2.0) * (F_dense / F_dense[-1])**(args.alpha)
    ax1.plot(F_dense, ref_y1, color='black', linestyle='--', linewidth=2.5, 
             label=rf'$\propto F^{{{args.alpha}}}$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$F$')
    ax1.set_ylabel(r'$\langle x \rangle$')
    
    # Place label (a) cleanly outside the plotting area
    ax1.set_title(r'(a)', loc='left', fontsize=22)
    
    theory_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Explicit Theory')
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles1 + [theory_line], frameon=True, loc='lower right')


    # ==========================================
    # GRAPH (b): Spatial Scaling (<x> vs w) -> Now on the RIGHT (ax2)
    # ==========================================
    unique_F = np.sort(df['F'].unique())
    w_dense = np.logspace(np.log10(df['w'].min()), np.log10(df['w'].max()), 50)

    for idx, F_val in enumerate(unique_F):
        df_F = df[df['F'] == F_val].sort_values('w')
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        w_sim = df_F['w'].values
        x_sim = df_F['avg_x'].values
        
        # Calculate error for terminal output
        x_th_sim = asymptotic_theory(w_sim, F_val, args.alpha, C_global)
        mape = np.mean(np.abs(x_sim - x_th_sim) / x_th_sim) * 100
        print(f"Graph (b) [Space] | F = {F_val:<4} | Mean Relative Error: {mape:.4f}%")
        
        # Simulation (Points)
        ax2.plot(w_sim, x_sim, marker=m, color=c, linestyle='none', 
                 label=rf'Sim: $F = {F_val}$', zorder=3)
        
        # Specific Theory matching the points (Colored Dashed Line)
        x_th_dense = asymptotic_theory(w_dense, F_val, args.alpha, C_global)
        ax2.plot(w_dense, x_th_dense, color=c, linestyle='--', alpha=0.7, zorder=2)

    # Black Dashed Reference Line
    ref_y2 = (df['avg_x'].max() * 1.5) * (w_dense / w_dense[0])**(2.0*args.alpha - 2.0)
    ax2.plot(w_dense, ref_y2, color='black', linestyle='--', linewidth=2.5, 
             label=rf'$\propto w^{{{2.0*args.alpha - 2.0:.1f}}}$')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$w$')
    ax2.set_ylabel(r'$\langle x \rangle$')
    ax2.set_xticks(np.sort(df['w'].unique()))
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Place label (b) cleanly outside the plotting area
    ax2.set_title(r'(b)', loc='left', fontsize=22)
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles2 + [theory_line], frameon=True, loc='lower left')

    print("-------------------------------------------\n")

    # --- Export ---
    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()