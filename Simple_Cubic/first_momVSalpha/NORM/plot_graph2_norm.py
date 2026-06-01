#!/usr/bin/env python3
# plot_graph2_norm.py
# Usage: python3 plot_graph2_norm.py --csv results_graph2_norm.csv

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, default="results_graph2_norm.csv")
    ap.add_argument("--out", default="graph2_normalized.pdf")
    args = ap.parse_args()

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

    df = pd.read_csv(args.csv)
    fig, ax = plt.subplots()
    
    markers = ['^', '*', 'o', 'x']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    unique_widths = np.sort(df['w'].unique())
    
    for idx, w_val in enumerate(unique_widths):
        df_w = df[df['w'] == w_val].sort_values('alpha')
        alpha_vals = df_w['alpha'].values
        
        # Pulling the newly calculated normalized column
        norm_x_vals = df_w['norm_x'].values
        
        ax.plot(alpha_vals, norm_x_vals, marker=markers[idx % len(markers)], 
                color=colors[idx % len(colors)], linestyle='-', label=rf'$w = {int(w_val)}$')

    ax.set_xlabel(r'$\alpha$')
    # Updated label to reflect the normalization
    ax.set_ylabel(r'$\langle x \rangle / (F^\alpha t^\alpha)$')
    
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 0.9])
    ax.text(0.3, 0.9, r'(c)', transform=ax.transAxes, fontsize=24, verticalalignment='top')
    ax.legend(frameon=True, loc='upper right')
    
    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"[SYSTEM] Normalized plot saved to: {args.out}")

if __name__ == "__main__":
    main()