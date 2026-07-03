#!/usr/bin/env python3
# plot_graph_c_hex_log.py
# Usage: python3 plot_graph_c_hex_log.py --csv results_graph2_hex_norm_server.csv --out graph_c_hex_log.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    ap = argparse.ArgumentParser(description="Plot exact normalized scaling for Hexagonal Geometry B")
    ap.add_argument("--csv", required=True, default="results_graph2_hex_norm_server.csv")
    ap.add_argument("--out", default="graph_c_hex_log.png")
    args = ap.parse_args()

    # Physical Review standard formatting parameters
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
        "figure.figsize": (7, 5)
    })

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {args.csv}.")
        return

    # Clean non-physical geometries and boundary conditions from the dataset
    df = df[(df['w'] > 0) & (df['alpha'] > 0.0)]

    fig, ax = plt.subplots()
    
    markers = ['^', '*', 'o', 'x']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    unique_widths = np.sort(df['w'].unique())
    
    # Engine parameters used in the C++ simulation
    kappa = 10.0
    F_phys = 0.05

    print("--- Exact Theory vs Hexagonal Simulation ---")

    for idx, w_val in enumerate(unique_widths):
        df_w = df[df['w'] == w_val].sort_values('alpha')
        alpha_vals = df_w['alpha'].values
        norm_Y_vals = df_w['norm_Y'].values
        
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        # Plot Simulation Data (Points)
        ax.plot(alpha_vals, norm_Y_vals, marker=m, color=c, linestyle='none', 
                label=rf'Sim: $w = {int(w_val)}$', zorder=3)
        
        # --- EXACT NON-LINEAR THEORY (HEXAGONAL) ---
        alpha_dense = np.linspace(alpha_vals.min(), alpha_vals.max(), 50)
        
        # 1. Cross-sectional area of a regular hexagon
        omega = 1.5 * np.sqrt(3.0) * (w_val**2)
        
        # 2. Effective force using the hexagonal area mapping
        F_eff = kappa * (F_phys**alpha_dense) * (omega**(alpha_dense - 1.0))
        
        # 3. Exact discrete bias for NN=8 (Hexagonal Lattice)
        bias = np.sinh(F_eff / 2.0) / (np.cosh(F_eff / 2.0) + 3.0)
        Y_th = bias / (F_phys**alpha_dense)
        
        ax.plot(alpha_dense, Y_th, color=c, linestyle='--', alpha=0.7, zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\frac{\langle x \rangle A_\alpha}{F^\alpha t^\alpha}$')
    
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    
    theory_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Exact Theory (Hex)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [theory_line], frameon=True, loc='best')
    
    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\n[SYSTEM] Log-scaled exact normalized plot saved to: {args.out}")

if __name__ == "__main__":
    main()