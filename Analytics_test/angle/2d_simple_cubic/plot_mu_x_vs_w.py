#!/usr/bin/env python3
# plot_mu_x_vs_w.py
# Example usage:
#   python3 plot_mu_x_vs_w.py --csv results_2d_angled.csv --alpha 0.3 --auto-K --show
import argparse, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot mu_x(w) with theory overlay.")
    ap.add_argument("--csv", required=True, help="CSV with results_2d_angled.csv columns: w,t,F_used,theta_deg,average_x,C_sim_x, etc.")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1)")
    ap.add_argument("--K", type=float, default=None, help="theory amplitude; if omitted, use --auto-K")
    ap.add_argument("--auto-K", action="store_true", help="Estimate K_lead from C_sim_x and its late-t plateau")
    ap.add_argument("--tail-points", type=int, default=5, help="Largest-t rows per w to average")
    ap.add_argument("--out", default="mu_x_vs_w.png", help="Output PNG filename")
    ap.add_argument("--show", action="store_true", help="Show figure interactively")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"w","t","F_used","theta_deg","average_x","C_sim_x"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}")

    alpha = args.alpha
    # Compute theta in radians
    theta = np.deg2rad(df["theta_deg"].values)
    df["theta"] = theta

    # Compute per-row normalized mu_x
    df["mu_x"] = df["average_x"] / (np.power(df["F_used"] * np.cos(df["theta"]), alpha) * np.power(df["t"], alpha))

    # For each width, take tail in t and average
    ws, mus_x, mus_x_err = [], [], []
    for w, g in df.groupby("w"):
        g2 = g.sort_values("t").tail(min(args.tail_points, len(g)))
        vals_x = g2["mu_x"].values
        mus_x.append(np.mean(vals_x)); mus_x_err.append(np.std(vals_x, ddof=1)/np.sqrt(len(vals_x)))
        ws.append(float(w))

    ws, mus_x, mus_x_err = map(np.array, [ws, mus_x, mus_x_err])
    order = np.argsort(ws); ws, mus_x, mus_x_err = ws[order], mus_x[order], mus_x_err[order]

    # Estimate K from C_sim_x if requested
    K_used = args.K
    if K_used is None and args.auto_K:
        tails = []
        for _, g in df.groupby("w"):
            g2 = g.sort_values("t").tail(min(args.tail_points, len(g)))
            tails.extend(g2["C_sim_x"].values.tolist())
        tails = np.array([x for x in tails if np.isfinite(x) and x > 0])
        K_used = float(np.median(tails)) if len(tails) else None
        print(f"[auto-K] Estimated K_lead ≈ {K_used:.6g}" if K_used is not None else "[auto-K] Could not estimate K")

    # Plot mu_x vs w on log-log axes
    plt.figure(figsize=(6.8, 4.6))
    plt.errorbar(ws, mus_x, yerr=mus_x_err, fmt="o", ms=5, lw=1.2, label="$\\mu_x$ (sim)")

    # Overlay theory: mu_th(w) = K * w^{-2(1-α)}
    if K_used is not None and np.isfinite(K_used) and K_used > 0:
        w_dense = np.logspace(np.log10(ws.min()), np.log10(ws.max()), 200)
        slope = -1.0*(1.0-alpha)
        mu_th = K_used * np.power(w_dense, slope)
        plt.plot(w_dense, mu_th, "--", lw=1.7, label=f"theory slope={slope:.3g}")
    else:
        print("[warn] No K provided; plotting data only")

    # Slope fit
    if len(ws) >= 2:
        X = np.log(ws)
        Yx = np.log(mus_x)
        m_x, b_x = np.polyfit(X, Yx, 1)
        print(f"[fit] slope log(mu_x) vs log(w): {m_x:.4f} (theory {slope:.4f})")

    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("width $w$")
    plt.ylabel(r"$\mu_x$ (rescaled mobility)")
    plt.title(r"$\mu_x$ vs $w$ for 2D angled drive")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show: plt.show()

if __name__ == "__main__":
    main()
