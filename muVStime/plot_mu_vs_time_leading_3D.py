#!/usr/bin/env python3
# plot_mu_vs_time_leading_3D.py
# Usage:
#   python3 plot_mu_vs_time_leading_3D.py --csv results_mu_vs_time_leading.csv --alpha 0.3 --w 5 --F 0.01 --A 1.0 --out mu_vs_time_leading_3D.png --show

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp


def theoretical_mu(alpha, w, A):
    # Leading-order 3D prefactor
    mu_th = (6.0 ** (-alpha)) / (A * (sp.gamma(1 + alpha) ** 2)) * (w ** (-2 * (1 - alpha)))
    return mu_th

def avg_displacement_theory(t, F, alpha, w, A):
    mu = theoretical_mu(alpha, w, A)
    return mu * (F ** alpha) * (t ** alpha)

def main():
    ap = argparse.ArgumentParser(description="Analyze <x(t)> and mu(t) for 3D channel random walk simulation.")
    ap.add_argument("--csv", required=True, help="CSV: t, average_x, mu_num, mu_theory, ...")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--w", type=float, required=True, help="Channel width")
    ap.add_argument("--F", type=float, required=True)
    ap.add_argument("--A", type=float, default=1.0, help="Amplitude factor")
    ap.add_argument("--out", default="mu_vs_time_leading_3D.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = df["t"].values
    x_sim = df["average_x"].values
    mu_sim = df["mu_num"].values

    # Theory
    mu_th = theoretical_mu(args.alpha, args.w, args.A)
    x_th = avg_displacement_theory(t, args.F, args.alpha, args.w, args.A)

    plt.figure(figsize=(7, 4.6))
    plt.plot(t, mu_sim, "o-", label="simulation μ_num")
    plt.axhline(mu_th, color="k", ls="--", label="theory μ_theory (3D)")

    # Overlay <x(t)> theory and simulation
    plt.figure(figsize=(7, 4.6))
    plt.plot(t, x_sim, "o-", label="simulation <x>")
    plt.plot(t, x_th, "--", label="theory <x> (leading)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time t")
    plt.ylabel("μ = <x> / (F^α t^α)")
    plt.title("μ vs time: 3D periodic channel (simulation vs theory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
