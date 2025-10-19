#!/usr/bin/env python3
# plot_mu_vs_time_convergence.py

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp

def theoretical_mu(alpha, w, A):
    # Leading-order 3D prefactor
    mu_th = (6.0 ** (-alpha)) / (A * (sp.gamma(1 + alpha) ** 2)) * (w ** (-2 * (1 - alpha)))
    return mu_th

def main():
    ap = argparse.ArgumentParser(description="Compare and analyze μ between simulation and theory with convergence view.")
    ap.add_argument("--csv", required=True, help="CSV: t, average_x, mu_num, mu_theory, ...")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--A", type=float, default=1.0)
    ap.add_argument("--out", default="mu_vs_time_convergence.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = df["t"].values
    mu_sim = df["mu_num"].values

    # Calculate theory
    mu_th = theoretical_mu(args.alpha, args.w, args.A)

    # Main plot: μ (simulation) and theory
    plt.figure(figsize=(7, 4.6))
    plt.plot(t, mu_sim, "o-", label="simulation μ")
    plt.axhline(mu_th, color="k", ls="--", label="theory μ (3D, leading)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"time $t$")
    plt.ylabel(r"$\mu(t) = \langle x(t) \rangle / (F^{\alpha} t^{\alpha})$")
    plt.title(r"$\mu$ vs time: 3D periodic channel (simulation vs theory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out.replace(".png", "_main.png"), dpi=220)
    if args.show:
        plt.show()

    # Ratio plot (convergence view)
    ratio = mu_sim / mu_th
    plt.figure(figsize=(7, 4.6))
    plt.plot(t, ratio, 'o-', label=r'$ \mu_{\mathrm{sim}} / \mu_{\mathrm{theory}} $')
    plt.xscale('log')
    plt.xlabel(r'time $t$')
    plt.ylabel(r'ratio $\mu_{\mathrm{sim}} / \mu_{\mathrm{theory}}$')
    plt.title(r'Convergence of simulation and theory')
    plt.axhline(1, color='gray', linestyle='--', lw=1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out.replace(".png","_ratio.png"), dpi=220)
    if args.show:
        plt.show()

    # Find closest approach to 1 (maximum convergence)
    closest_idx = np.argmin(np.abs(ratio - 1))
    print(f"Best convergence: t={t[closest_idx]:.3g}, simulation/theory={ratio[closest_idx]:.3g}")

if __name__ == "__main__":
    main()
