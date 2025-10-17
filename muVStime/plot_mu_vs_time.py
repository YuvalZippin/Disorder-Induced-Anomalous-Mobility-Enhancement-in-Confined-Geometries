#!/usr/bin/env python3
# plot_mu_vs_time.py (compatible with upgraded CSV)
# Examples:
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time_up.csv --alpha 0.3 --use-csv-theory --plot-R --show
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time_up.csv --alpha 0.3 --K 0.19 --w 5 --plot-R --show
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time_up.csv --alpha 0.3 --auto-tail --tail-points 6 --plot-R --show

import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def finite_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None

def main():
    ap = argparse.ArgumentParser(description="Plot mu(t) and optionally R(t)=mu/mu_inf using upgraded CSV.")
    ap.add_argument("--csv", required=True, help="CSV from simulator with columns like: t, mu_sim, mu_inf, ...")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1) if recomputing mu from average_x,F_used,t")

    ap.add_argument("--out", default="mu_vs_time.png", help="Output PNG filename")
    ap.add_argument("--show", action="store_true", help="Show figure")

    # Baseline sources (use dest to ensure underscore attribute names)
    ap.add_argument("--use-csv-theory", dest="use_csv_theory", action="store_true",
                    help="Prefer CSV mu_inf; fallback to mu_infty/mu_theory/mu_theory_plateau")
    ap.add_argument("--K", type=float, default=None, help="K_lead to compute mu_inf = K * w^{-2(1-alpha)}")
    ap.add_argument("--w", type=float, default=None, help="Width w for --K")
    ap.add_argument("--auto-tail", action="store_true", help="Estimate mu_inf as median of last N mu points")
    ap.add_argument("--tail-points", type=int, default=6, help="N tail points when using --auto-tail")

    # Optional normalized ratio panel
    ap.add_argument("--plot-R", dest="plot_R", action="store_true", help="Also plot R(t)=mu/mu_inf")

    args = ap.parse_args()

    # Load and sort
    df = pd.read_csv(args.csv)  # pandas CSV read [web:25]
    if "t" not in df.columns:
        raise ValueError("CSV must contain column 't'")  # pandas column check [web:25]
    df = df.sort_values("t").reset_index(drop=True)  # stable sort for plotting [web:25]
    t = df["t"].to_numpy()

    # Pick or compute mu(t)
    if "mu_sim" in df.columns:
        mu = df["mu_sim"].to_numpy()
    elif {"average_x","F_used","t"}.issubset(df.columns):
        mu = df["average_x"].to_numpy() / (
            np.power(df["F_used"].to_numpy(), args.alpha) * np.power(df["t"].to_numpy(), args.alpha)
        )
    else:
        raise ValueError("Need mu_sim or (average_x,F_used,t) to compute mu")  # data requirement [web:25]

    # Determine mu_inf
    mu_inf, src = None, None
    if args.use_csv_theory:
        if "mu_inf" in df.columns:
            mu_inf, src = finite_median(df["mu_inf"].to_numpy()), "mu_inf"
        if (mu_inf is None) and ("mu_infty" in df.columns):
            mu_inf, src = finite_median(df["mu_infty"].to_numpy()), "mu_infty"
        if (mu_inf is None) and ("mu_theory" in df.columns):
            mu_inf, src = finite_median(df["mu_theory"].to_numpy()), "mu_theory"
        if (mu_inf is None) and ("mu_theory_plateau" in df.columns):
            mu_inf, src = finite_median(df["mu_theory_plateau"].to_numpy()), "mu_theory_plateau"

    if (mu_inf is None) and (args.K is not None) and (args.w is not None):
        mu_inf = float(args.K) * float(args.w) ** (-2.0 * (1.0 - args.alpha))  # baseline compute [web:142]
        src = "K,w"

    if (mu_inf is None) and args.auto_tail:
        n = max(1, min(args.tail_points, len(df)))
        mu_inf, src = finite_median(mu[-n:]), f"tail({n})"  # robust tail median [web:150]

    if args.plot_R and mu_inf is None:
        print("[warn] --plot-R requested but mu_inf not found; R(t) panel will be omitted")  # flow guard [web:157]

    # Figure
    if args.plot_R and (mu_inf is not None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.8), sharex=True, gridspec_kw={"height_ratios":[2,1]})
    else:
        fig, ax1 = plt.subplots(figsize=(6.8, 4.6)); ax2 = None

    # μ(t) on log-log
    ax1.loglog(t, mu, "o-", ms=4.5, lw=1.2, label="simulation")  # log-log plot [web:142]
    if (mu_inf is not None) and np.isfinite(mu_inf) and (mu_inf > 0):
        ax1.hlines(mu_inf, t.min(), t.max(), colors="C1", linestyles="--", lw=1.6,
                   label=f"theory plateau ({src})")  # horizontal line [web:145]
    ax1.set_xlabel("time t")
    ax1.set_ylabel("mu(t) = <x> / (F^alpha t^alpha)")
    ax1.set_title("mu vs time")
    ax1.legend()
    fig.tight_layout()

    # Optional R(t)
    if args.plot_R and (mu_inf is not None):
        R = mu / mu_inf
        ax2.semilogx(t, R, "o-", ms=4.5, lw=1.2, label="R(t) = μ/μ∞")  # semilog x for ratio [web:123]
        ax2.hlines(1.0, t.min(), t.max(), colors="C1", linestyles="--", lw=1.6, label="theory: 1")  # baseline [web:145]
        ax2.set_ylabel("R(t)")
        ax2.set_xlabel("time t")
        ax2.legend()
        fig.tight_layout()

    fig.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
