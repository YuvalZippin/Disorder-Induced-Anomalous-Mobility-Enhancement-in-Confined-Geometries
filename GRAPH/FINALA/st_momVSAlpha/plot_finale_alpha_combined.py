#!/usr/bin/env python3
# plot_finale_alpha_combined.py
# Usage: python3 plot_finale_alpha_combined.py --cubic CUBIC.csv --hex HEX.csv --widths_cubic 5 15 --widths_hex 10 20 --out graph_alpha_combined.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def exact_theory_cubic(alpha, w, kappa, F_phys):
    """ Exact non-linear theory for Geometry A (Cubic, NN=6): Omega = w^2 """
    omega = w**2
    F_eff = kappa * (F_phys**alpha) * (omega**(alpha - 1.0))
    bias = np.sinh(F_eff / 2.0) / (np.cosh(F_eff / 2.0) + 2.0)
    return bias / (F_phys**alpha)

def exact_theory_hex(alpha, w, kappa, F_phys):
    """ Exact non-linear theory for Geometry B (Hexagonal, NN=8): Omega = (3*sqrt(3)/2) * w^2 """
    omega = 1.5 * np.sqrt(3.0) * (w**2)
    F_eff = kappa * (F_phys**alpha) * (omega**(alpha - 1.0))
    bias = np.sinh(F_eff / 2.0) / (np.cosh(F_eff / 2.0) + 3.0)
    return bias / (F_phys**alpha)

def main():
    ap = argparse.ArgumentParser(description="Grand Finale: Combined Exact Normalized Scaling vs Alpha")
    ap.add_argument("--cubic", required=True, default="CUBIC.csv", help="Input CSV for Cubic (Geometry A)")
    ap.add_argument("--hex", required=True, default="HEX.csv", help="Input CSV for Hexagonal (Geometry B)")
    
    # Split the widths argument into two independent lists to prevent visual overlap
    ap.add_argument("--widths_cubic", type=int, nargs='+', default=[5, 20], help="List of widths (w) to plot for Cubic")
    ap.add_argument("--widths_hex", type=int, nargs='+', default=[10], help="List of widths (w) to plot for Hexagonal")
    
    ap.add_argument("--out", default="graph_alpha_combined.png", help="Output filename")
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
        "figure.figsize": (9, 6) # Slightly wider to accommodate the dual legend comfortably
    })

    try:
        df_cubic_full = pd.read_csv(args.cubic)
        df_hex_full = pd.read_csv(args.hex)
        print(f"[SYSTEM] Loaded {args.cubic} and {args.hex} successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find the CSV files. Please ensure CUBIC.csv and HEX.csv exist.")
        return

    # --- DYNAMIC CLUTTER REDUCTION FILTER ---
    df_cubic = df_cubic_full[df_cubic_full['w'].isin(args.widths_cubic) & (df_cubic_full['alpha'] > 0.0)]
    df_hex = df_hex_full[df_hex_full['w'].isin(args.widths_hex) & (df_hex_full['alpha'] > 0.0)]

    # Engine parameters used in the C++ simulations
    kappa = 10.0
    F_phys = 0.05

    fig, ax = plt.subplots()
    
    # Extract all unique widths to be plotted to maintain a consistent color mapping
    plotted_widths = np.sort(list(set(df_cubic['w'].unique()).union(set(df_hex['w'].unique()))))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    print("--- Rendering Combined Geometries (Alpha Dependence) ---")
    print(f"[SYSTEM] Plotting Cubic for w={args.widths_cubic}")
    print(f"[SYSTEM] Plotting Hexagonal for w={args.widths_hex}")

    for idx, w_val in enumerate(plotted_widths):
        c = colors[idx % len(colors)]
        alpha_dense = np.linspace(0.2, 0.8, 100)
        
        # ==========================================
        # GEOMETRY A (CUBIC) -> Squares ('s'), Dashed ('--')
        # ==========================================
        if w_val in args.widths_cubic:
            df_w_cub = df_cubic[df_cubic['w'] == w_val].sort_values('alpha')
            if not df_w_cub.empty:
                alpha_cub = df_w_cub['alpha'].values
                norm_Y_cub = df_w_cub['norm_Y'].values
                
                ax.plot(alpha_cub, norm_Y_cub, marker='s', color=c, linestyle='none', zorder=3)
                Y_th_cub = exact_theory_cubic(alpha_dense, w_val, kappa, F_phys)
                ax.plot(alpha_dense, Y_th_cub, color=c, linestyle='--', alpha=0.8, zorder=2)
        
        # ==========================================
        # GEOMETRY B (HEXAGONAL) -> Hexagons ('h'), Dotted (':')
        # ==========================================
        if w_val in args.widths_hex:
            df_w_hex = df_hex[df_hex['w'] == w_val].sort_values('alpha')
            if not df_w_hex.empty:
                alpha_hex = df_w_hex['alpha'].values
                norm_Y_hex = df_w_hex['norm_Y'].values
                
                ax.plot(alpha_hex, norm_Y_hex, marker='h', color=c, linestyle='none', zorder=3)
                Y_th_hex = exact_theory_hex(alpha_dense, w_val, kappa, F_phys)
                ax.plot(alpha_dense, Y_th_hex, color=c, linestyle=':', alpha=0.8, zorder=2)

    # --- Formatting the Log-Scale Equation ---
    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\frac{\langle x(t) \rangle A_\alpha}{F^\alpha t^\alpha}$')
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    
    # ---> THE FIX: Add dynamic headroom for BOTH legends <---
    ymin, ymax = ax.get_ylim()
    # Multiplying ymax by 10 pushes the ceiling up. Dividing ymin by 5 pushes the floor down.
    ax.set_ylim(ymin / 5.0, ymax * 10.0) 

    # --- Custom Dual Legend Structure ---
    # Legend 1: Colors mapped to Transverse Width (w)
    color_handles = [Line2D([0], [0], color=colors[i % len(colors)], linewidth=3, 
                            label=rf'$w = {int(w)}$') for i, w in enumerate(plotted_widths)]
    leg1 = ax.legend(handles=color_handles, title="Transverse Width", loc='upper left', frameon=True, framealpha=1.0)
    ax.add_artist(leg1)
    
    # Legend 2: Markers/Lines mapped to Geometry Type
    geom_handles = [
        Line2D([0], [0], color='gray', marker='s', linestyle='--', markersize=10, 
               label='Cubic (Sim & Theory)'),
        Line2D([0], [0], color='gray', marker='h', linestyle=':', markersize=10, 
               label='Hexagonal (Sim & Theory)')
    ]
    ax.legend(handles=geom_handles, loc='lower right', frameon=True, framealpha=1.0)

    print("-------------------------------------------\n")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"[SYSTEM] Grand Finale Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()