#!/usr/bin/env python3
# plot_mu_vs_time.py
# Examples:
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time_3d.csv --alpha 0.3 --use-csv --plot-R --show
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time_3d.csv --alpha 0.3 --K 0.19 --w 5 --plot-R --show
#   python3 plot_mu_vs_time.py --csv results_mu_vs_time_3d.csv --alpha 0.3 --auto-tail --tail-points 6 --plot-R --show

import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def finite_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None

def main():
    ap = argparse.ArgumentParser(description="Plot μ(t) for 3D and optional R(t)=μ/μ∞.")
    ap.add_argument("--csv", required=True, help="CSV from full_sim_mu_vs_time_3d.cpp")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1) (used only if recomputing μ)")
    ap.add_argument("--use-csv", action="store_true", help="Use mu_inf column from CSV if present")
    ap.add_argument("--K", type=float, default=None, help="K_lead for mu_inf = K * w^{-2(1-alpha)}")
    ap.add_argument("--w", type=float, default=None, help="Width w used with --K")
    ap.add_argument("--auto-tail", action="store_true", help="Estimate mu_inf as median of last N mu points")
    ap.add_argument("--tail-points", type=int, default=6, help="N tail points for auto-tail")
    ap.add_argument("--plot-R", action="store_true", help="Also plot R(t)=μ/μ∞")
    ap.add_argument("--out", default="mu_vs_time_3d.png", help="PNG output")
    ap.add_argument("--show", action="store_true", help="Show figure")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("t").reset_index(drop=True)
    if "t" not in df.columns: raise ValueError("CSV must contain 't'")

    t = df["t"].to_numpy()
    if "mu" in df.columns: mu = df["mu"].to_numpy()
    elif "mu_sim" in df.columns: mu = df["mu_sim"].to_numpy()
    elif {"average_x","F_used","t"}.issubset(df.columns):
        mu = df["average_x"].to_numpy() / (np.power(df["F_used"], args.alpha)*np.power(df["t"], args.alpha))
    else:
        raise ValueError("Need mu or mu_sim or (average_x,F_used,t) to compute μ")

    mu_inf, src = None, None
    if args.use_csv:
        for col in ["mu_inf","mu_infty","mu_theory","mu_theory_plateau"]:
            if col in df.columns:
                mu_inf = finite_median(df[col].to_numpy()); src = col
                if mu_inf is not None: break
    if (mu_inf is None) and (args.K is not None) and (args.w is not None):
        mu_inf = float(args.K) * float(args.w) ** (-2.0*(1.0 - args.alpha)); src = "K,w"
    if (mu_inf is None) and args.auto_tail:
        n = max(1, min(args.tail_points, len(df)))
        mu_inf = finite_median(mu[-n:]); src = f"tail({n})"

    if args.plot_R and mu_inf is None:
        print("[warn] --plot-R requested but mu_inf not set; plotting μ(t) only")

    if args.plot_R and (mu_inf is not None):
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7.0, 6.8), sharex=True, gridspec_kw={"height_ratios":[2,1]})
    else:
        fig, ax1 = plt.subplots(figsize=(6.8, 4.6)); ax2=None

    ax1.loglog(t, mu, "o-", ms=4.5, lw=1.2, label="simulation")
    if mu_inf is not None and np.isfinite(mu_inf) and mu_inf>0:
        ax1.hlines(mu_inf, t.min(), t.max(), colors="C1", linestyles="--", lw=1.6, label=f"theory plateau ({src})")
    ax1.set_xlabel("time t")
    ax1.set_ylabel("mu(t) = <x> / (F^alpha t^alpha)")
    ax1.set_title("3D: μ vs time")
    ax1.legend()
    fig.tight_layout()

    if args.plot_R and (mu_inf is not None):
        R = mu / mu_inf
        ax2.semilogx(t, R, "o-", ms=4.5, lw=1.2, label="R(t)=μ/μ∞")
        ax2.hlines(1.0, t.min(), t.max(), colors="C1", linestyles="--", lw=1.6, label="theory: 1")
        ax2.set_ylabel("R(t)")
        ax2.set_xlabel("time t")
        ax2.legend()
        fig.tight_layout()

    fig.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")
    if args.show: plt.show()

if __name__ == "__main__":
    main()
