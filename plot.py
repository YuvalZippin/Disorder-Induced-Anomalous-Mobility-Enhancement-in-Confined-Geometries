#!/usr/bin/env python3
# plot_leading_law.py
# Usage examples:
#   python plot_leading_law.py --csv results_wsalpha.csv --alpha 0.3 --auto-A
#   python plot_leading_law.py --csv results_wsalpha.csv --alpha 0.3 --A 3.9
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def gamma1p(a):
    return math.gamma(1.0 + a)

def klead(alpha, A):
    # K_lead(α) = 6^{-α}/(A Γ(1+α)^2)
    return (6.0 ** (-alpha)) / (A * (gamma1p(alpha) ** 2))

def estimate_A_from_plateau(alpha, C_vals):
    # A_est = 6^{-α}/(C*Γ(1+α)^2)
    C = np.asarray(C_vals)
    C = C[np.isfinite(C) & (C > 0)]
    C_star = np.median(C) if C.size else np.nan
    if not np.isfinite(C_star) or C_star <= 0:
        return np.nan
    return (6.0 ** (-alpha)) / (C_star * (gamma1p(alpha) ** 2))

def main():
    ap = argparse.ArgumentParser(description="Plot C_sim vs t with theoretical line and <x> vs t with theoretical t^alpha curves.")
    ap.add_argument("--csv", required=True, help="Input CSV with columns: w,t,F_used,average_x,C_sim,steps_max,steps_mean")
    ap.add_argument("--alpha", type=float, required=True, help="Alpha in (0,1)")
    ap.add_argument("--A", type=float, default=None, help="Use this A for theory; if omitted and --auto-A is set, estimate from plateau")
    ap.add_argument("--auto-A", action="store_true", help="Estimate A from plateau of C_sim (median of last tail points per w)")
    ap.add_argument("--tail-points", type=int, default=5, help="Tail points at largest t to estimate A (per w)")
    ap.add_argument("--out-prefix", default="plots", help="Prefix for saved figures")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    args = ap.parse_args()

    alpha = args.alpha
    df = pd.read_csv(args.csv)
    required_cols = {"w","t","F_used","average_x","C_sim"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # group by width and sort by time
    groups = {}
    for w, g in df.groupby("w"):
        gg = g.sort_values("t").copy()
        groups[int(w)] = gg

    # If auto-A requested and A not provided, estimate from tail plateau across widths
    A_used = args.A
    if A_used is None and args.auto_A:
        C_tails = []
        for w, g in groups.items():
            tail = g.tail(min(args.tail_points, len(g)))
            C_tails.extend(tail["C_sim"].values.tolist())
        A_est = estimate_A_from_plateau(alpha, C_tails)
        A_used = A_est
        print(f"[auto-A] Estimated A ≈ {A_used:.6g} from tail plateau of C_sim")
    if A_used is None:
        print("[warn] No A provided and --auto-A not set; theory lines will be omitted")
    
    # Plot 1: C_sim vs t (horizontal theoretical line = K_lead if A is known)
    plt.figure(figsize=(7.0, 4.8))
    for w, g in sorted(groups.items()):
        t = g["t"].values
        C = g["C_sim"].values
        plt.semilogx(t, C, marker="o", ms=3, lw=1.2, label=f"w={w}")
    if A_used is not None and np.isfinite(A_used) and A_used > 0:
        K = klead(alpha, A_used)
        plt.axhline(K, color="k", ls="--", lw=1.5, label=f"Theory K_lead={K:.4g}")
    plt.xlabel("t")
    plt.ylabel("C_sim = <x> t^{-α} F^{-α} w^{2(1-α)}")
    plt.title("C_sim vs t (plateau test)")
    plt.legend()
    plt.tight_layout()
    fig1_path = f"{args.out_prefix}_Csim_vs_t.png"
    plt.savefig(fig1_path, dpi=200)

    # Plot 2: <x> vs t (log-log) per w with theoretical t^α curves if A known
    plt.figure(figsize=(7.0, 4.8))
    for w, g in sorted(groups.items()):
        t = g["t"].values
        x = g["average_x"].values
        F = g["F_used"].values
        plt.loglog(t, x, marker="o", ms=3, lw=1.2, label=f"data w={w}")
        if A_used is not None and np.isfinite(A_used) and A_used > 0:
            # Theory: <x>_th = K_lead * w^{-2(1-α)} * F^α * t^α
            K = klead(alpha, A_used)
            wfac = (float(w) ** (-2.0 * (1.0 - alpha)))
            # Use per-row F for robustness (if varied)
            x_th = K * wfac * (F ** alpha) * (t ** alpha)
            plt.loglog(t, x_th, lw=1.5, ls="--", label=f"theory w={w}")
    plt.xlabel("t")
    plt.ylabel("<x>")
    plt.title("<x> vs t with t^α theory")
    plt.legend()
    plt.tight_layout()
    fig2_path = f"{args.out_prefix}_x_vs_t.png"
    plt.savefig(fig2_path, dpi=200)

    # Report simple alpha fits per width and plateau stats
    def linfit_loglog(t, y):
        X = np.log(t); Y = np.log(np.maximum(y, 1e-300))
        n = len(X)
        sx = X.sum(); sy = Y.sum()
        sxx = (X*X).sum(); sxy = (X*Y).sum()
        denom = n*sxx - sx*sx
        slope = (n*sxy - sx*sy)/denom
        intercept = (sy - slope*sx)/n
        return slope, intercept

    print("\nPer-width diagnostics:")
    for w, g in sorted(groups.items()):
        t = g["t"].values
        x = g["average_x"].values
        C = g["C_sim"].values
        slope, _ = linfit_loglog(t, x)
        C_tail = g["C_sim"].tail(min(args.tail_points, len(g))).values
        print(f"  w={w}: alpha_fit≈{slope:.4f}, C_tail_mean≈{np.mean(C_tail):.6g}, C_tail_std≈{np.std(C_tail):.3g}")

    print(f"\nSaved: {fig1_path}\nSaved: {fig2_path}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
