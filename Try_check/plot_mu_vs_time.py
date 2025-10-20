#!/usr/bin/env python3
# plot_mu_vs_time.py
# Usage:
#   python3 plot_mu_vs_time.py --csv mu_vs_time.csv --alpha 0.3 --K 0.19
import argparse, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot mu(t)=<x>/(F^alpha t^alpha) vs t for fixed w with theory overlay.")
    ap.add_argument("--csv", required=True, help="CSV with columns: t,mu,average_x,steps_max,steps_mean")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1)")
    ap.add_argument("--K", type=float, default=None, help="Theoretical plateau value for mu (optional)")
    ap.add_argument("--out", default="mu_vs_time.png", help="Output PNG filename")
    ap.add_argument("--show", action="store_true", help="Show figure")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"t", "mu"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")

    # Sort by time
    df = df.sort_values("t")
    tvals = df["t"].values
    muvals = df["mu"].values

    # Plot mu(t) vs t
    plt.figure(figsize=(6.8, 4.6))
    plt.plot(tvals, muvals, "o-", ms=5, lw=1.2, label="simulation")

    # Overlay theory plateau if provided
    if args.K is not None and np.isfinite(args.K) and args.K > 0:
        plt.axhline(args.K, color="r", ls="--", lw=1.6, label=f"theory plateau K={args.K:.3g}")

    # Optional: report plateau estimate from tail
    tail_points = min(5, len(muvals))
    tail_mu = np.mean(muvals[-tail_points:])
    print(f"[tail] Estimated plateau mu ≈ {tail_mu:.6g} (average of last {tail_points} points)")

    plt.xscale("log")
    plt.xlabel("time t")
    plt.ylabel("mu(t) = <x> / (F^alpha t^alpha)")
    plt.title("mu vs time (plateau = asymptotic regime)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
