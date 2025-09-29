#!/usr/bin/env python3
# plot_mu_vs_w.py
# Usage:
#   python3 plot_mu_vs_w.py --csv results_wsalpha.csv --alpha 0.3 --auto-K
#   python3 plot_mu_vs_w.py --csv results_wsalpha.csv --alpha 0.3 --K 0.19
import argparse, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot mu(w)=<x>/(F^alpha t^alpha) vs w with theory overlay.")
    ap.add_argument("--csv", required=True, help="CSV with columns: w,t,F_used,average_x,C_sim,steps_max,steps_mean")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1)")
    ap.add_argument("--K", type=float, default=None, help="K_lead amplitude to use; if omitted and --auto-K set, estimate from C_sim plateau")
    ap.add_argument("--auto-K", action="store_true", help="Estimate K_lead from tail of C_sim (median across widths)")
    ap.add_argument("--tail-points", type=int, default=5, help="Number of largest-t rows per width to average")
    ap.add_argument("--out", default="mu_vs_w.png", help="Output PNG filename")
    ap.add_argument("--show", action="store_true", help="Show figure")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"w","t","F_used","average_x","C_sim"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")

    # Compute per-row mu_i = <x> / (F^alpha t^alpha)
    alpha = args.alpha
    df = df.copy()
    df["mu_row"] = df["average_x"] / (np.power(df["F_used"], alpha) * np.power(df["t"], alpha))

    # For each width, take tail in t and average
    mu_w, mu_w_err, ws = [], [], []
    for w, g in df.groupby("w"):
        g2 = g.sort_values("t").tail(min(args.tail_points, len(g)))
        vals = g2["mu_row"].values
        mu_hat = float(np.mean(vals))
        mu_se = float(np.std(vals, ddof=1)/np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ws.append(float(w)); mu_w.append(mu_hat); mu_w_err.append(mu_se)

    ws = np.array(ws); mu_w = np.array(mu_w); mu_w_err = np.array(mu_w_err)
    order = np.argsort(ws); ws, mu_w, mu_w_err = ws[order], mu_w[order], mu_w_err[order]

    # Estimate K_lead from C_sim plateau if requested (C_sim equals K_lead)
    K_used = args.K
    if K_used is None and args.auto_K:
        tails = []
        for _, g in df.groupby("w"):
            g2 = g.sort_values("t").tail(min(args.tail_points, len(g)))
            tails.extend(g2["C_sim"].values.tolist())
        tails = np.array([x for x in tails if np.isfinite(x) and x > 0])
        K_used = float(np.median(tails)) if len(tails) else None
        print(f"[auto-K] Estimated K_lead ≈ {K_used:.6g}" if K_used is not None else "[auto-K] Could not estimate K")

    # Plot μ vs w on log-log axes
    plt.figure(figsize=(6.8, 4.6))
    plt.errorbar(ws, mu_w, yerr=mu_w_err, fmt="o", ms=5, lw=1.2, label="simulation")

    # Overlay theory: mu_th(w) = K * w^{-2(1-α)}
    if K_used is not None and np.isfinite(K_used) and K_used > 0:
        w_dense = np.logspace(np.log10(ws.min()), np.log10(ws.max()), 200)
        slope = -2.0*(1.0 - alpha)
        mu_th = K_used * np.power(w_dense, slope)
        plt.plot(w_dense, mu_th, "--", lw=1.6, label=f"theory slope={slope:.3g}")
    else:
        print("[warn] No K provided; plotting data only")

    # Optional: report fitted slope from data
    if len(ws) >= 2:
        X = np.log(ws); Y = np.log(mu_w)
        A = np.vstack([X, np.ones_like(X)]).T
        m, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        print(f"[fit] slope log(mu) vs log(w) ≈ {m:.4f} (theory {-2*(1-alpha):.4f})")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("width w")
    plt.ylabel("mu(w) = <x> / (F^alpha t^alpha)")
    plt.title("mu vs w (expect slope -2(1-alpha))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
