#!/usr/bin/env python3
# plot_mu_vs_time.py
# Usage examples:
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time.csv --alpha 0.3 --auto-theory
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time.csv --alpha 0.3 --K 0.19 --w 5
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time.csv --alpha 0.3 --fit-tail 6 --show

import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot mu(t) on log-log axes with optional theory overlay and tail-slope fit.")
    ap.add_argument("--csv", required=True, help="CSV with columns: t,F_used,average_x,mu_sim,mu_theory,steps_max,steps_mean")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1) for recomputing mu if needed")
    ap.add_argument("--out", default="mu_vs_time.png", help="Output PNG filename")
    ap.add_argument("--show", action="store_true", help="Show figure")

    # Theory overlay options
    ap.add_argument("--K", type=float, default=None, help="K_lead to form mu_th = K * w^{-2(1-alpha)}")
    ap.add_argument("--w", type=float, default=5.0, help="Width w for theory overlay if --K given")
    ap.add_argument("--auto-theory", action="store_true", help="If mu_theory column exists, use its median; else median of tail(mu_sim)")

    # Tail processing / slope fit
    ap.add_argument("--tail-points", type=int, default=6, help="Number of largest-t points to summarize for plateaus")
    ap.add_argument("--fit-tail", type=int, default=0, help="If >0, fit slope d log(mu)/d log(t) on last N points and print")
    args = ap.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    # Validate required columns minimally
    cols = set(df.columns)
    needed_min = {"t"}
    if not needed_min.issubset(cols):
        raise ValueError(f"CSV missing columns: {needed_min - cols}")

    # Choose a mu column: prefer 'mu_sim'; else recompute from average_x,F_used
    mu_col = None
    if "mu_sim" in cols:
        mu_col = "mu_sim"
    elif {"average_x","F_used","t"}.issubset(cols):
        df = df.copy()
        df["mu_recomp"] = df["average_x"] / (np.power(df["F_used"], args.alpha) * np.power(df["t"], args.alpha))
        mu_col = "mu_recomp"
    else:
        raise ValueError("CSV needs 'mu_sim' or ('average_x','F_used','t') to compute mu")

    # Sort by time
    df = df.sort_values("t")
    t = df["t"].values
    mu = df[mu_col].values

    # Determine theory value if requested
    mu_theory = None
    if args.K is not None and np.isfinite(args.K) and args.K > 0:
        mu_theory = float(args.K) * float(args.w) ** (-2.0 * (1.0 - args.alpha))
    elif args.auto_theory:
        if "mu_theory" in cols and np.isfinite(df["mu_theory"]).any():
            mu_theory = float(np.median(df["mu_theory"].replace([np.inf, -np.inf], np.nan).dropna().values))
        else:
            tail_n = max(1, min(args.tail_points, len(df)))
            mu_theory = float(np.median(mu[-tail_n:]))

    # Optional tail slope fit on log-log
    if args.fit_tail and args.fit_tail > 1 and len(df) >= args.fit_tail:
        tail_n = args.fit_tail
        X = np.log(t[-tail_n:])
        Y = np.log(mu[-tail_n:])
        # Guard against nonpositive values for log
        ok = np.isfinite(X) & np.isfinite(Y)
        ok &= (mu[-tail_n:] > 0) & (t[-tail_n:] > 0)
        if ok.sum() >= 2:
            m, b = np.polyfit(X[ok], Y[ok], 1)
            print(f"[fit] tail slope d log(mu)/d log(t) over last {ok.sum()} points ≈ {m:.4f}")
        else:
            print("[fit] insufficient positive finite tail points for slope")

    # Plot
    plt.figure(figsize=(6.8, 4.6))
    plt.loglog(t, mu, "o-", ms=4.5, lw=1.2, label="simulation")
    if mu_theory is not None and np.isfinite(mu_theory) and mu_theory > 0:
        plt.hlines(mu_theory, t.min(), t.max(), colors="C1", linestyles="--", lw=1.6, label="theory plateau")

    plt.xlabel("time t")
    plt.ylabel("mu(t) = <x> / (F^alpha t^alpha)")
    plt.title("mu vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
