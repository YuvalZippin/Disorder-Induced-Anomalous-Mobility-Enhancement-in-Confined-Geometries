#!/usr/bin/env python3
# plot_first_moment_vs_time.py
# Usage:
#   python3 plot_first_moment_vs_time.py --csv results_moment_vs_time.csv --alpha 0.3 --F 0.005 --w 5 --A 1.0 --show
#
# Input CSV columns (from moment_vs_time.cpp):
#   w,t,F_used,average_x,mu_num,mu_theory,ratio,rel_err,slope_local,steps_max,steps_mean
#
# Output:
#   first_moment_vs_time.png     (main plot: <x>(t) and theory)
#   ratio_mu_vs_time.png         (diagnostic: mu_num/mu_theory vs time)
#   Prints: estimated best convergence point based on |ratio-1| minimum

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma

def mu_theory(alpha, w, A):
    # μ_th = 6^{-α} / (A Γ(1+α)^2) · w^{-2(1−α)}
    return (6.0 ** (-alpha)) / (A * (gamma(1 + alpha) ** 2)) * (w ** (-2.0 * (1.0 - alpha)))

def x_theory(t, F, alpha, w, A):
    # <x>(t) = μ_th · F^α · t^α
    return mu_theory(alpha, w, A) * (F ** alpha) * (t ** alpha)

def main():
    ap = argparse.ArgumentParser(description="Plot first moment <x(t)> vs time with leading-order theory overlay.")
    ap.add_argument("--csv", required=True, help="CSV from moment_vs_time.cpp")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1)")
    ap.add_argument("--F", type=float, required=True, help="Force used (if omitted in CSV)")
    ap.add_argument("--w", type=float, required=True, help="Channel width")
    ap.add_argument("--A", type=float, default=1.0, help="Amplitude A")
    ap.add_argument("--out", default="first_moment_vs_time.png", help="Output PNG")
    ap.add_argument("--show", action="store_true", help="Show figure")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"t","average_x"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")

    t = df["t"].to_numpy()
    x_sim = df["average_x"].to_numpy()

    # Prefer F_used in CSV if present; otherwise use --F
    if "F_used" in df.columns and np.isfinite(df["F_used"].iloc[0]):
        F_used = float(df["F_used"].iloc[0])
    else:
        F_used = float(args.F)

    # Theory curve
    x_th = x_theory(t, F_used, args.alpha, args.w, args.A)

    # Main plot: <x>(t) and theory
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(t, x_sim, "o-", ms=5, lw=1.4, label="simulation ⟨x(t)⟩")
    plt.plot(t, x_th, "--", lw=1.6, label="theory ⟨x⟩ (leading)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time t")
    plt.ylabel("⟨x(t)⟩")
    plt.title("First moment vs time: 3D channel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")

    # Convergence diagnostics via μ ratio if columns are present
    if {"mu_num","mu_theory"}.issubset(df.columns):
        mu_ratio = (df["mu_num"] / df["mu_theory"]).to_numpy()
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(t, mu_ratio, "o-", ms=5, lw=1.4, label=r"$\mu_{\mathrm{sim}} / \mu_{\mathrm{th}}$")
        plt.axhline(1.0, color="gray", ls="--", lw=1.2)
        plt.xscale("log")
        plt.xlabel("time t")
        plt.ylabel(r"ratio $\mu_{\mathrm{sim}}/\mu_{\mathrm{th}}$")
        plt.title("Convergence of μ: simulation/theory")
        plt.legend()
        plt.tight_layout()
        ratio_out = args.out.replace(".png", "_ratio.png")
        plt.savefig(ratio_out, dpi=220)
        print(f"Saved: {ratio_out}")

        # Report best convergence (closest to 1)
        idx = int(np.argmin(np.abs(mu_ratio - 1.0)))
        print(f"Best convergence: t ≈ {t[idx]:.6g}, mu_sim/mu_th ≈ {mu_ratio[idx]:.6g}")

        # Optional: mark the closest point on main figure
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(t, x_sim, "o-", ms=5, lw=1.4, label="simulation ⟨x(t)⟩")
        plt.plot(t, x_th, "--", lw=1.6, label="theory ⟨x⟩ (leading)")
        plt.scatter([t[idx]], [x_sim[idx]], c="red", zorder=5, label="closest μ ratio")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("time t")
        plt.ylabel("⟨x(t)⟩")
        plt.title("First moment vs time: 3D channel (closest ratio marked)")
        plt.legend()
        plt.tight_layout()
        out2 = args.out.replace(".png", "_marked.png")
        plt.savefig(out2, dpi=220)
        print(f"Saved: {out2}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
