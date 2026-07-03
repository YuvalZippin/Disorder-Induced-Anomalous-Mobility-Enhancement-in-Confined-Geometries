#!/usr/bin/env python3
# plot_finale_omega_combined.py
# Usage: python3 plot_finale_omega_combined.py --cubic CUBIC.csv --hex HEX.csv --forces 0.01 0.05 --out graph_omega_combined.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def exact_theory_cubic(omega, F, alpha, C_0, kappa=10.0):
    """ Exact non-linear theory for Geometry A (Cubic, NN=6) """
    F_eff = kappa * (F**alpha) * (omega**(alpha - 1.0))
    bias = np.sinh(F_eff / 2.0) / (np.cosh(F_eff / 2.0) + 2.0)
    return C_0 * bias

def exact_theory_hex(omega, F, alpha, C_0, kappa=10.0):
    """ Exact non-linear theory for Geometry B (Hexagonal, NN=8) """
    F_eff = kappa * (F**alpha) * (omega**(alpha - 1.0))
    bias = np.sinh(F_eff / 2.0) / (np.cosh(F_eff / 2.0) + 3.0)
    return C_0 * bias

def main():
    ap = argparse.ArgumentParser(description="Grand Finale: Combined Exact Scaling vs Omega (Cross-Section)")
    ap.add_argument("--cubic", required=True, default="CUBIC.csv", help="Input CSV for Cubic (Geometry A)")
    ap.add_argument("--hex", required=True, default="HEX.csv", help="Input CSV for Hexagonal (Geometry B)")
    ap.add_argument("--alpha", type=float, default=0.3, help="Anomalous exponent")
    ap.add_argument("--forces", type=float, nargs='+', default=[0.01, 0.02, 0.05], help="List of Forces (F) to plot")
    ap.add_argument("--out", default="graph_omega_combined.png", help="Output filename")
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
        "figure.figsize": (9, 6) # Wider for dual legend
    })

    try:
        df_cubic = pd.read_csv(args.cubic)
        df_hex = pd.read_csv(args.hex)
        print(f"[SYSTEM] Loaded {args.cubic} and {args.hex} successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find CUBIC.csv or HEX.csv. Run C++ engines first.")
        return

    # --- Translate Width (w) to Cross-Sectional Area (Omega) ---
    df_cubic['omega'] = df_cubic['w']**2
    df_hex['omega'] = 1.5 * np.sqrt(3.0) * (df_hex['w']**2)

    # --- Robust Global Calibration (C_0) ---
    # Instead of anchoring to one point, we use the mean of the exact bias ratio 
    # across the entire Cubic dataset to find the perfect global prefactor.
    kappa = 10.0
    F_eff_cub = kappa * (df_cubic['F']**args.alpha) * (df_cubic['omega']**(args.alpha - 1.0))
    bias_cub_exact = np.sinh(F_eff_cub / 2.0) / (np.cosh(F_eff_cub / 2.0) + 2.0)
    
    C_0 = np.mean(df_cubic['avg_x'] / bias_cub_exact)
    
    theoretical_power = args.alpha - 1.0
    print(f"\n[SYSTEM] Calibrated Universal Prefactor C_0 = {C_0:.4e}")
    print(f"[SYSTEM] Expected asymptotic spatial scaling: <x> ~ \Omega^({theoretical_power:.2f})")
    print("--- Plotting Combined Geometries (Omega Dependence) ---")

    fig, ax = plt.subplots()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create a dense array of Omega for smooth theoretical lines
    omega_min = min(df_cubic['omega'].min(), df_hex['omega'].min())
    omega_max = max(df_cubic['omega'].max(), df_hex['omega'].max())
    omega_dense = np.logspace(np.log10(omega_min), np.log10(omega_max), 50)

    for idx, f_val in enumerate(args.forces):
        c = colors[idx % len(colors)]
        
        # Robust filtering for floating point F values
        df_f_cub = df_cubic[np.isclose(df_cubic['F'], f_val)].sort_values('omega')
        df_f_hex = df_hex[np.isclose(df_hex['F'], f_val)].sort_values('omega')
        
        if df_f_cub.empty and df_f_hex.empty:
            continue

        # ==========================================
        # GEOMETRY A (CUBIC) -> Squares ('s'), Dashed ('--')
        # ==========================================
        if not df_f_cub.empty:
            ax.plot(df_f_cub['omega'], df_f_cub['avg_x'], marker='s', color=c, 
                    linestyle='none', zorder=3)
            # Use Exact Non-Linear Theory
            x_th_cub = exact_theory_cubic(omega_dense, f_val, args.alpha, C_0, kappa)
            ax.plot(omega_dense, x_th_cub, color=c, linestyle='--', alpha=0.8, zorder=2)
            
        # ==========================================
        # GEOMETRY B (HEXAGONAL) -> Hexagons ('h'), Dotted (':')
        # ==========================================
        if not df_f_hex.empty:
            ax.plot(df_f_hex['omega'], df_f_hex['avg_x'], marker='h', color=c, 
                    linestyle='none', zorder=3)
            # Use Exact Non-Linear Theory
            x_th_hex = exact_theory_hex(omega_dense, f_val, args.alpha, C_0, kappa)
            ax.plot(omega_dense, x_th_hex, color=c, linestyle=':', alpha=0.8, zorder=2)

    # Black Dashed Global Reference Line (Asymptotic Power Law)
    # Positioned above the curves to show the theoretical asymptotic slope
    highest_theory = exact_theory_cubic(omega_dense[0], max(args.forces), args.alpha, C_0, kappa)
    ref_y = (highest_theory * 1.8) * (omega_dense / omega_dense[0])**(theoretical_power)
    
    ax.plot(omega_dense, ref_y, color='black', linestyle='-.', linewidth=2.0, 
            label=rf'$\propto \Omega^{{{theoretical_power:.1f}}}$')

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Transverse Area $\Omega$')
    ax.set_ylabel(r'$\langle x \rangle$')
    
    # --- Custom Dual Legend Structure ---
    valid_forces = [f for f in args.forces if np.any(np.isclose(df_cubic['F'], f)) or np.any(np.isclose(df_hex['F'], f))]
    color_handles = [Line2D([0], [0], color=colors[i % len(colors)], linewidth=3, 
                            label=rf'$F = {f}$') for i, f in enumerate(valid_forces)]
    leg1 = ax.legend(handles=color_handles, title="Constant Force", loc='lower left', frameon=True)
    ax.add_artist(leg1)
    
    geom_handles = [
        Line2D([0], [0], color='gray', marker='s', linestyle='--', markersize=10, 
               label='Cubic (Sim & Theory)'),
        Line2D([0], [0], color='gray', marker='h', linestyle=':', markersize=10, 
               label='Hexagonal (Sim & Theory)'),
        Line2D([0], [0], color='black', linestyle='-.', linewidth=2.0, 
               label=rf'$\propto \Omega^{{{theoretical_power:.1f}}}$')
    ]
    ax.legend(handles=geom_handles, loc='upper right', frameon=True)

    print("-------------------------------------------\n")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"[SYSTEM] Grand Finale Plot successfully rendered to: {args.out}")

if __name__ == "__main__":
    main()