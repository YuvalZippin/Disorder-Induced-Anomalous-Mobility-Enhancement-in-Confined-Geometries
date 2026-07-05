#!/usr/bin/env python3
# plot_finale_force_combined.py
# Usage: python3 plot_finale_force_combined.py --cubic CUBIC.csv --hex HEX.csv --widths_cubic 5 15 --widths_hex 10 20 --out graph_force_combined.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def exact_theory_cubic(F, w, alpha, C_0, kappa=10.0):
    """ Exact non-linear theory for Geometry A (Cubic, NN=6): Omega = w^2 """
    omega = w**2
    F_eff = kappa * (F**alpha) * (omega**(alpha - 1.0))
    bias = np.sinh(F_eff / 2.0) / (np.cosh(F_eff / 2.0) + 2.0)
    return C_0 * bias

def exact_theory_hex(F, w, alpha, C_0, kappa=10.0):
    """ Exact non-linear theory for Geometry B (Hexagonal, NN=8): Omega = (3*sqrt(3)/2) * w^2 """
    omega = 1.5 * np.sqrt(3.0) * (w**2)
    F_eff = kappa * (F**alpha) * (omega**(alpha - 1.0))
    bias = np.sinh(F_eff / 2.0) / (np.cosh(F_eff / 2.0) + 3.0)
    return C_0 * bias

def main():
    ap = argparse.ArgumentParser(description="Grand Finale: Combined Exact Scaling vs Force (F) with Alternating Widths")
    ap.add_argument("--cubic", required=True, default="CUBIC.csv", help="Input CSV for Cubic (Geometry A)")
    ap.add_argument("--hex", required=True, default="HEX.csv", help="Input CSV for Hexagonal (Geometry B)")
    ap.add_argument("--alpha", type=float, default=0.3, help="Anomalous exponent")
    
    # Split the widths argument into two independent lists to prevent visual overlap
    ap.add_argument("--widths_cubic", type=int, nargs='+', default=[5, 15], help="List of widths (w) to plot for Cubic")
    ap.add_argument("--widths_hex", type=int, nargs='+', default=[10, 20], help="List of widths (w) to plot for Hexagonal")
    
    ap.add_argument("--out", default="graph_force_combined.png", help="Output filename")
    args = ap.parse_args()

    # Academic Physical Review styling
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 16,
        "axes.labelsize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 13,
        "lines.linewidth": 2.0,
        "lines.markersize": 10,
        "figure.figsize": (9, 6) # Wider to accommodate dual legend cleanly
    })

    try:
        df_cubic_full = pd.read_csv(args.cubic)
        df_hex_full = pd.read_csv(args.hex)
        print(f"[SYSTEM] Loaded {args.cubic} and {args.hex} successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find CUBIC.csv or HEX.csv. Ensure files exist.")
        return

    # --- Robust Global Calibration (C_0) BEFORE filtering ---
    # Calibrate C_0 across the full Cubic dataset to ensure theoretical accuracy
    kappa = 10.0
    omega_cub = df_cubic_full['w']**2
    F_eff_cub = kappa * (df_cubic_full['F']**args.alpha) * (omega_cub**(args.alpha - 1.0))
    bias_cub_exact = np.sinh(F_eff_cub / 2.0) / (np.cosh(F_eff_cub / 2.0) + 2.0)
    
    C_0 = np.mean(df_cubic_full['avg_x'] / bias_cub_exact)
    
    # --- DYNAMIC CLUTTER REDUCTION FILTER ---
    df_cubic = df_cubic_full[df_cubic_full['w'].isin(args.widths_cubic) & (df_cubic_full['F'] > 0.0)]
    df_hex = df_hex_full[df_hex_full['w'].isin(args.widths_hex) & (df_hex_full['F'] > 0.0)]

    print(f"\n[SYSTEM] Calibrated Universal Prefactor C_0 = {C_0:.4e}")
    print(f"[SYSTEM] Plotting Cubic for w={args.widths_cubic}")
    print(f"[SYSTEM] Plotting Hexagonal for w={args.widths_hex}")
    print("--- Rendering Combined Geometries (Force Dependence) ---")

    fig, ax = plt.subplots()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Extract all unique widths to be plotted to maintain a consistent color mapping
    plotted_widths = np.sort(list(set(df_cubic['w'].unique()).union(set(df_hex['w'].unique()))))
    
    # Create a dense force array for smooth theoretical curves
    f_min = min(df_cubic_full['F'].min(), df_hex_full['F'].min())
    f_max = max(df_cubic_full['F'].max(), df_hex_full['F'].max())
    F_dense = np.logspace(np.log10(f_min), np.log10(f_max), 50)

    for idx, w_val in enumerate(plotted_widths):
        c = colors[idx % len(colors)]
        
        # ==========================================
        # GEOMETRY A (CUBIC) -> Squares ('s'), Dashed ('--')
        # ==========================================
        if w_val in args.widths_cubic:
            df_w_cub = df_cubic[df_cubic['w'] == w_val].sort_values('F')
            if not df_w_cub.empty:
                ax.plot(df_w_cub['F'], df_w_cub['avg_x'], marker='s', color=c, 
                        linestyle='none', zorder=3)
                x_th_cub = exact_theory_cubic(F_dense, w_val, args.alpha, C_0, kappa)
                ax.plot(F_dense, x_th_cub, color=c, linestyle='--', alpha=0.8, zorder=2)
            
        # ==========================================
        # GEOMETRY B (HEXAGONAL) -> Hexagons ('h'), Dotted (':')
        # ==========================================
        if w_val in args.widths_hex:
            df_w_hex = df_hex[df_hex['w'] == w_val].sort_values('F')
            if not df_w_hex.empty:
                ax.plot(df_w_hex['F'], df_w_hex['avg_x'], marker='h', color=c, 
                        linestyle='none', zorder=3)
                x_th_hex = exact_theory_hex(F_dense, w_val, args.alpha, C_0, kappa)
                ax.plot(F_dense, x_th_hex, color=c, linestyle=':', alpha=0.8, zorder=2)

    # Black Dashed Global Reference Line (Asymptotic Power Law)
    # Anchor to the top curve
    min_w_plotted = plotted_widths[0]
    highest_theory = exact_theory_cubic(F_dense[-1], min_w_plotted, args.alpha, C_0, kappa)
    ref_y = (highest_theory * 1.6) * (F_dense / F_dense[-1])**(args.alpha)
    
    ax.plot(F_dense, ref_y, color='black', linestyle='-.', linewidth=2.0, 
            label=rf'$\propto F^{{{args.alpha:.1f}}}$')

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$F$')
    ax.set_ylabel(r'$\langle x(t) \rangle$') # Updated label to match your image exactly
    
    # ---> THE FIX: HUGE dynamic headroom for BOTH legends <---
    ymin, ymax = ax.get_ylim()
    # Pushing the floor down significantly (divided by 15.0)
    ax.set_ylim(ymin / 15.0, ymax * 5.0) 

    # --- Custom Dual Legend Structure ---
    color_handles = [Line2D([0], [0], color=colors[i % len(colors)], linewidth=3, 
                            label=rf'$w = {int(w)}$') for i, w in enumerate(plotted_widths)]
    leg1 = ax.legend(handles=color_handles, title="Transverse Width", loc='lower right', frameon=True, framealpha=1.0)
    ax.add_artist(leg1)
    
    geom_handles = [
        Line2D([0], [0], color='gray', marker='s', linestyle='--', markersize=10, 
               label='Cubic (Sim & Theory)'),
        Line2D([0], [0], color='gray', marker='h', linestyle=':', markersize=10, 
               label='Hexagonal (Sim & Theory)'),
        Line2D([0], [0], color='black', linestyle='-.', linewidth=2.0, 
               label=rf'$\propto F^{{{args.alpha:.1f}}}$')
    ]
    ax.legend(handles=geom_handles, loc='upper left', frameon=True, framealpha=1.0)

    print("-------------------------------------------\n")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"[SYSTEM] Grand Finale Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()