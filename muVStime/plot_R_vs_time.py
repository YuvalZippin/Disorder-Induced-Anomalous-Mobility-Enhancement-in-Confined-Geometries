#!/usr/bin/env python3
# plot_R_vs_time.py
# Examples:
#   python3 plot_R_vs_time.py --csv results_mu_vs_time.csv --alpha 0.3 --use-csv-muinf --show
#   python3 plot_R_vs_time.py --csv results_mu_vs_time.csv --alpha 0.3 --K 0.19 --w 5 --show
#   python3 plot_R_vs_time.py --csv results_mu_vs_time.csv --alpha 0.3 --auto-tail --tail-points 6 --show

import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def median_finite(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.median(x))

def main():
    ap = argparse.ArgumentParser(description="Plot R(t)=mu(t)/mu_inf with robust mu_inf selection.")
    ap.add_argument("--csv", required=True, help="CSV from simulator")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1) if recomputing mu from average_x,F_used,t")
    # Sources for mu_inf
    ap.add_argument("--use-csv-muinf", action="store_true", help="Try CSV columns mu_infty or mu_theory")
    ap.add_argument("--K", type=float, default=None, help="K_lead to compute mu_inf = K * w^{-2(1-alpha)}")
    ap.add_argument("--w", type=float, default=None, help="Width w for --K")
    ap.add_argument("--auto-tail", action="store_true", help="If set, estimate mu_inf as median of last N mu points")
    ap.add_argument("--tail-points", type=int, default=6, help="N tail points for auto-tail")
    # Plot
    ap.add_argument("--out", default="R_vs_time.png", help="Output PNG")
    ap.add_argument("--show", action="store_true", help="Show figure")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Ensure time-sorted
    if "t" not in df.columns:
        raise ValueError("CSV missing column 't'")
    df = df.sort_values("t").reset_index(drop=True)
    t = df["t"].to_numpy()

    # Determine mu(t)
    if "mu_sim" in df.columns:
        mu = df["mu_sim"].to_numpy()
    elif {"average_x","F_used","t"}.issubset(df.columns):
        mu = df["average_x"].to_numpy() / (
            np.power(df["F_used"].to_numpy(), args.alpha) * np.power(df["t"].to_numpy(), args.alpha)
        )
    else:
        raise ValueError("CSV needs mu_sim or (average_x,F_used,t) to compute mu")

    # Determine mu_inf priority: CSV (mu_infty or mu_theory) -> K,w -> auto-tail
    mu_inf = None
    src = None
    if args.use_csv_muinf:
        if "mu_infty" in df.columns:
            mu_inf = median_finite(df["mu_infty"].to_numpy())
            src = "mu_infty"
        if (mu_inf is None) and ("mu_theory" in df.columns):
            mu_inf = median_finite(df["mu_theory"].to_numpy())
            src = "mu_theory"
    if (mu_inf is None) and (args.K is not None) and (args.w is not None):
        mu_inf = float(args.K) * float(args.w) ** (-2.0 * (1.0 - args.alpha))
        src = "K,w"
    if (mu_inf is None) and args.auto_tail:
        n = max(1, min(args.tail_points, len(df)))
        tail = mu[-n:]
        mu_inf = median_finite(tail)
        src = f"tail({n})"
    if mu_inf is None:
        raise ValueError("Could not determine mu_inf. Provide --use-csv-muinf or --K & --w or --auto-tail")

    print(f"[info] mu_inf = {mu_inf:.8g} (source: {src})")

    # Compute R(t)
    R = mu / mu_inf

    # Plot R(t) with a horizontal baseline at 1
    plt.figure(figsize=(6.8, 4.4))
    plt.semilogx(t, R, "o-", ms=4.5, lw=1.2, label="R(t) = μ/μ∞")
    plt.hlines(1.0, t.min(), t.max(), colors="C1", linestyles="--", lw=1.6, label="theory: 1")
    plt.xlabel("time t")
    plt.ylabel("R(t) = μ(t) / μ∞")
    plt.title("Normalized ratio R(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
