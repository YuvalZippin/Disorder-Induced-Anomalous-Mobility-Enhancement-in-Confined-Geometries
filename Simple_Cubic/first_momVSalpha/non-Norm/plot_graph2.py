#!/usr/bin/env python3
# plot_graph2.py
# Usage: python3 plot_graph2.py --csv results_graph2.csv

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Argument Parsing
    ap = argparse.ArgumentParser(description="Plot <x> vs alpha for QTM Subordination.")
    ap.add_argument("--csv", required=True, default="results_graph2.csv", help="Input CSV from Engine 2")
    ap.add_argument("--out", default="graph2_alpha_dependence.pdf", help="Output plot filename")
    args = ap.parse_args()

    # 2. Matplotlib configuration for PRL/PRE style
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
        print(f"[ERROR] Could not find {args.csv}.")
        return

    required_cols = {"alpha", "w", "avg_x"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[ERROR] CSV missing required columns. Found: {df.columns}")

    # 4. Plot Generation
    fig, ax = plt.subplots()
    
    # Matching the exact marker styles and colors from the provided image
    markers = ['^', '*', 'o', 'x']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    # Sort widths dynamically to ensure correct legend assignment
    unique_widths = np.sort(df['w'].unique())
    
    print("--- Analysis Report ---")

    for idx, w_val in enumerate(unique_widths):
        # Extract data for current width, sorted by alpha
        df_w = df[df['w'] == w_val].sort_values('alpha')
        alpha_vals = df_w['alpha'].values
        x_vals = df_w['avg_x'].values
        
        # In the target graph (c), the y-axis shows a normalized or scaled <x> 
        # that converges to ~1.0 at alpha = 1.0. 
        # If your C++ engine outputs raw <x>, you may need to apply a normalization 
        # factor here based on the theoretical analytical prefactor. 
        # For now, we plot the raw/effective value straight from the CSV.
        y_vals = x_vals 
        
        ax.plot(alpha_vals, y_vals, marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                linestyle='-', label=rf'$w = {int(w_val)}$')
        
        print(f"Width w={w_val}: Alpha range [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}] plotted.")

    # 5. Axis Formatting
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\langle x \rangle$')
    
    # Set discrete ticks for alpha based on common ranges
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Add the (c) text box in the upper center/left as seen in the reference image
    ax.text(0.3, 0.9, r'(c)', transform=ax.transAxes, fontsize=24, verticalalignment='top')

    # Legend inside the plot, top right
    ax.legend(frameon=True, loc='upper right')
    
    plt.tight_layout()
    #! ax.set_xscale('log')
    #! ax.set_yscale('log')

    # 6. Export
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\n[SYSTEM] Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()