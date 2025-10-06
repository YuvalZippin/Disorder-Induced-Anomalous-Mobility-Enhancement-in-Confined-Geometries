#!/usr/bin/env python3
# start_mu/onset_plots.py
# Usage examples:
#   python3 onset_plots.py --csv start_mu_results.csv --alpha 0.3 --bc periodic --bins-per-decade 12 --roll 7 --tail 7 --tolC 0.07 --tolA 0.03
#   python3 onset_plots.py --csv start_mu_results.csv --alpha 0.3 --bc reflecting --bins-per-decade 12 --roll 7 --tail 7 --tolC 0.07 --tolA 0.03
import argparse, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# Optional Savitzky–Golay filter for evenly spaced indices
try:
    from scipy.signal import savgol_filter  # [web:241]
    HAVE_SG = True
except Exception:
    HAVE_SG = False

def log_bin_series(t, y, yse=None, bins_per_decade=12):
    """Return (t_bin_center, y_bin_mean, y_bin_se) using log10(t) binning."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if yse is not None: yse = np.asarray(yse, float)
    if t.size == 0: 
        return t, y, (yse if yse is not None else None)
    lo, hi = np.log10(t.min()), np.log10(t.max())
    ndec = hi - lo
    nbins = max(1, int(ndec * bins_per_decade))
    edges = np.linspace(lo, hi, nbins+1)
    idx = np.digitize(np.log10(t), edges) - 1
    tb, yb, sb = [], [], []
    for b in range(nbins):
        sel = (idx == b)
        if not np.any(sel): 
            continue
        tt = t[sel]; yy = y[sel]
        tb.append(10**((edges[b]+edges[b+1])/2.0))
        yb.append(np.mean(yy))
        if yse is not None:
            se_i = yse[sel]
            sb.append(np.sqrt(np.sum(se_i**2)) / max(1, np.sum(sel)))
        else:
            sb.append(np.std(yy, ddof=1) / np.sqrt(np.sum(sel)) if np.sum(sel)>1 else 0.0)
    return np.array(tb), np.array(yb), np.array(sb)

def rolling_mean(y, w):
    if w<=1: 
        return np.asarray(y, float).copy()
    k = int(w)
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan, dtype=float)
    n = len(y); half = k//2
    for i in range(n):
        L = max(0, i-half)
        R = min(n-1, L+k-1)
        L = max(0, R-k+1)
        out[i] = np.nanmean(y[L:R+1])
    return out

def find_onset_plateau(t, C, Khat, tol=0.07, consec=3):
    """First time entering [Khat*(1±tol)] for 'consec' points and stays thereafter."""
    if not np.isfinite(Khat) or Khat<=0 or len(C)==0:
        return np.nan
    low = Khat*(1.0 - tol); high = Khat*(1.0 + tol)
    inside = (C >= low) & (C <= high)
    n = len(C)
    for i in range(n - consec + 1):
        if np.all(inside[i:i+consec]) and np.all(inside[i+consec-1:]):
            return t[i+consec-1]
    return np.nan

def find_onset_slope(t, alpha_fit, alpha, tol=0.03, consec=3):
    inside = np.isfinite(alpha_fit) & (np.abs(alpha_fit - alpha) <= tol)
    n = len(alpha_fit)
    for i in range(n - consec + 1):
        if np.all(inside[i:i+consec]) and np.all(inside[i+consec-1:]):
            return t[i+consec-1]
    return np.nan

def main():
    ap = argparse.ArgumentParser(description="Onset detection for mobility scaling with binning/smoothing.")
    ap.add_argument("--csv", required=True, help="start_mu_results.csv")
    ap.add_argument("--alpha", type=float, required=True, help="alpha in (0,1)")
    ap.add_argument("--bc", choices=["periodic","reflecting"], required=True, help="boundary condition label to plot")
    ap.add_argument("--tail", type=int, default=7, help="largest-t points used for plateau K̂")
    ap.add_argument("--tolC", type=float, default=0.07, help="relative tolerance for plateau band (e.g., 0.07)")
    ap.add_argument("--tolA", type=float, default=0.03, help="absolute tolerance for alpha_fit band (e.g., 0.03)")
    ap.add_argument("--bins-per-decade", type=int, default=12, help="log10(t) bins per decade for averaging (0=off)")
    ap.add_argument("--roll", type=int, default=7, help="rolling window size on binned series (0=off)")
    ap.add_argument("--smooth", choices=["none","sg"], default="none", help="Savitzky–Golay on binned series (evenly spaced index)")
    ap.add_argument("--sg-window", type=int, default=7, help="SG window length (odd)")
    ap.add_argument("--sg-poly", type=int, default=2, help="SG polynomial order (< window)")
    ap.add_argument("--outdir", default="start_mu_plots", help="output directory")
    ap.add_argument("--show", action="store_true", help="show figures")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = {"bc","w","F_used","t","N_traj","avg_x","C_sim","se_C_sim","alpha_fit"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")

    df = df[df["bc"]==args.bc].copy()
    if df.empty:
        raise ValueError(f"No rows with bc={args.bc}")

    groups = df.groupby(["w","F_used"], sort=True)
    summary = []

    for (w,F), g in groups:
        g = g.sort_values("t").reset_index(drop=True)
        t_raw = g["t"].values
        C_raw = g["C_sim"].values
        se_raw = g["se_C_sim"].values
        A_raw = g["alpha_fit"].values
        alpha = args.alpha

        # Plateau estimate K̂ from tail of raw series (robust to smoothing choices)
        tail = g.tail(min(args.tail, len(g)))
        Khat = float(np.mean(tail["C_sim"].values))
        Kse  = float(np.std(tail["C_sim"].values, ddof=1)/math.sqrt(len(tail))) if len(tail)>1 else 0.0

        # Binning
        tC, Cb, SEb = (t_raw, C_raw, se_raw)
        tA, Ab = (t_raw, A_raw)
        if args.bins_per_decade > 0:
            tC, Cb, SEb = log_bin_series(t_raw, C_raw, se_raw, bins_per_decade=args.bins_per_decade)
            tA, Ab, _   = log_bin_series(t_raw, A_raw, None,        bins_per_decade=args.bins_per_decade)

        # Rolling mean
        if args.roll and args.roll>1:
            Cb = rolling_mean(Cb, args.roll)
            Ab = rolling_mean(Ab, args.roll)

        # Optional Savitzky–Golay on the binned index domain
        if args.smooth == "sg":
            if not HAVE_SG:
                print("[warn] SciPy not installed; skipping SG smoothing")
            else:
                win = max(3, args.sg_window | 1)  # force odd
                poly = min(args.sg_poly, win-1)
                if len(Cb) >= win:
                    Cb = savgol_filter(Cb, window_length=win, polyorder=poly, mode="interp")
                if len(Ab) >= win:
                    Ab = savgol_filter(Ab, window_length=win, polyorder=poly, mode="interp")

        # Onsets on the binned/smoothed series (for display and robust detection)
        t_star_C = find_onset_plateau(tC, Cb, Khat, tol=args.tolC, consec=3)
        t_star_A = find_onset_slope(tA, Ab, alpha, tol=args.tolA, consec=3)
        t_star   = np.nanmax([t_star_C, t_star_A]) if (np.isfinite(t_star_C) or np.isfinite(t_star_A)) else np.nan

        # Plot C_sim
        plt.figure(figsize=(7.2, 4.6))
        if SEb is not None and SEb.size==Cb.size:
            plt.errorbar(tC, Cb, yerr=SEb, fmt="o", ms=3, lw=1.0, capsize=2, label="C_sim (binned/smoothed)")
        else:
            plt.semilogx(tC, Cb, marker="o", ms=3, lw=1.2, label="C_sim (binned/smoothed)")
        band_lo, band_hi = Khat*(1-args.tolC), Khat*(1+args.tolC)
        plt.axhline(Khat, color="k", lw=1.2, ls="--", label=f"K̂={Khat:.4g}±{Kse:.1g}")
        plt.fill_between(tC, band_lo, band_hi, color="k", alpha=0.07, label=f"±{int(args.tolC*100)}% band")
        if np.isfinite(t_star_C):
            plt.axvline(t_star_C, color="tab:green", ls=":", lw=1.5, label=f"t*_C≈{t_star_C:.2e}")
        if np.isfinite(t_star_A):
            plt.axvline(t_star_A, color="tab:orange", ls=":", lw=1.5, label=f"t*_α≈{t_star_A:.2e}")
        if np.isfinite(t_star):
            plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=f"t*≈{t_star:.2e}")
        plt.xlabel("t")
        plt.ylabel("C_sim = <x> t^{-α} F^{-α} w^{2(1-α)}")
        plt.title(f"{args.bc}: C_sim vs t  (w={int(w)}, F={F:g})")
        plt.legend()
        plt.tight_layout()
        p1 = Path(args.outdir) / f"{args.bc}_Csim_w{int(w)}_F{F:g}.png"
        plt.savefig(p1, dpi=220)
        if args.show: plt.show()
        plt.close()

        # Plot alpha_fit
        plt.figure(figsize=(7.2, 4.6))
        plt.semilogx(tA, Ab, marker="o", ms=3, lw=1.2, label="alpha_fit (binned/smoothed)")
        plt.axhline(alpha, color="k", lw=1.2, ls="--", label=f"alpha={alpha}")
        plt.fill_between(tA, alpha-args.tolA, alpha+args.tolA, color="k", alpha=0.07, label=f"±{args.tolA} band")
        if np.isfinite(t_star_A):
            plt.axvline(t_star_A, color="tab:orange", ls=":", lw=1.5, label=f"t*_α≈{t_star_A:.2e}")
        if np.isfinite(t_star):
            plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=f"t*≈{t_star:.2e}")
        plt.xlabel("t")
        plt.ylabel("alpha_fit(t)")
        plt.title(f"{args.bc}: alpha_fit vs t  (w={int(w)}, F={F:g})")
        plt.legend()
        plt.tight_layout()
        p2 = Path(args.outdir) / f"{args.bc}_alpha_w{int(w)}_F{F:g}.png"
        plt.savefig(p2, dpi=220)
        if args.show: plt.show()
        plt.close()

        summary.append({
            "bc": args.bc, "w": int(w), "F_used": F,
            "Khat": Khat, "Khat_se": Kse,
            "t_star_C": t_star_C, "t_star_A": t_star_A, "t_star": t_star,
            "bins_per_decade": args.bins_per_decade, "roll": args.roll, "smooth": args.smooth
        })

    summ = pd.DataFrame(summary).sort_values(["bc","w","F_used"])
    out_csv = Path(args.outdir) / f"onset_summary_{args.bc}.csv"
    summ.to_csv(out_csv, index=False)
    print(f"Saved figures to {args.outdir} and summary to {out_csv}")

if __name__ == "__main__":
    main()
