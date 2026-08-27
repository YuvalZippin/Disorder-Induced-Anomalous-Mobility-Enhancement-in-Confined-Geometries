#!/usr/bin/env python3
# plot_qtm_firstmoment.py
# Usage:
#   python3 plot_qtm_firstmoment.py --csv qtm_firstmoment_periodic_fast.csv --w 3 --F 0.01 --alpha 0.3 --A 1 --tol 0.1 --minrun 3 --bins-per-decade 12 --roll 7 --alpha-fit-tol 0.05 --fit-minlen 6 --show
import argparse, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def klead(alpha, A):
    return (6.0 ** (-alpha)) / (A * (math.gamma(1.0 + alpha) ** 2))  # [web:14][web:15]

def x_theory(t, w, F, alpha, A):
    K = klead(alpha, A)
    return K * (w ** (-2.0 * (1.0 - alpha))) * (F ** alpha) * (t ** alpha)  # [web:14][web:15]

def onset_by_ratio(t, x, x_th, tol=0.1, minrun=3):
    x_th_safe = np.maximum(x_th, 1e-300)
    rel = np.abs(x - x_th) / x_th_safe
    inside = rel <= tol
    n = len(t)
    for i in range(n - minrun + 1):
        if np.all(inside[i:i+minrun]) and np.all(inside[i+minrun-1:]):  # sustained entry [web:14]
            return t[i+minrun-1]
    return None  # [web:14]

# --- Smoothing: log-binning + rolling mean for visual clarity and robust fitting [web:324][web:223]
def log_bin_series(t, y, bins_per_decade=12):
    t = np.asarray(t, float); y = np.asarray(y, float)
    lo, hi = np.log10(t.min()), np.log10(t.max())
    nbins = max(1, int((hi - lo) * bins_per_decade))
    edges = np.linspace(lo, hi, nbins+1)
    idx = np.digitize(np.log10(t), edges) - 1
    tb, yb = [], []
    for b in range(nbins):
        sel = (idx == b)
        if not np.any(sel): continue
        tb.append(10**((edges[b]+edges[b+1])/2.0))
        yb.append(np.mean(y[sel]))
    return np.array(tb), np.array(yb)  # [web:324]

def rolling_mean(y, w=7):
    if w<=1: return y
    k = int(w); n=len(y); out = np.full_like(y, np.nan, float)
    h = k//2
    for i in range(n):
        L = max(0, i-h); R = min(n-1, L+k-1); L = max(0, R-k+1)
        out[i] = np.nanmean(y[L:R+1])
    return out  # [web:324]

# --- Rolling-slope window selection near α to avoid plateau bias [web:223]
def rolling_slope(t, x, k):
    X = np.log(np.maximum(np.asarray(t, float), 1e-300))
    Y = np.log(np.maximum(np.asarray(x, float), 1e-300))
    n = len(X); k = max(3, int(k)); half = k//2
    slopes = np.full(n, np.nan)
    for i in range(n):
        L = max(0, i-half); R = min(n-1, L+k-1); L = max(0, R-k+1)
        m = R-L+1
        if m < 3: continue
        xw, yw = X[L:R+1], Y[L:R+1]
        sx, sy = xw.sum(), yw.sum()
        sxx, sxy = (xw*xw).sum(), (xw*yw).sum()
        denom = m*sxx - sx*sx
        if abs(denom) < 1e-20: continue
        slopes[i] = (m*sxy - sx*sy) / denom
    return slopes  # [web:223]

def find_powerlaw_window(t, x, alpha_target, k=9, tol=0.05, minlen=6):
    s = rolling_slope(t, x, k)
    good = np.isfinite(s) & (np.abs(s - alpha_target) <= tol)
    bestL=bestR=None; bestLen=0
    i=0; n=len(t)
    while i<n:
        if not good[i]: i+=1; continue
        j=i
        while j<n and good[j]: j+=1
        if (j-i)>=minlen and (j-i)>bestLen:
            bestL, bestR, bestLen = i, j-1, (j-i)
        i=j
    if bestL is not None: return bestL, bestR
    # fallback: pick shortest acceptable window near α
    win=minlen
    bestErr=1e9; bestL, bestR = 0, win-1
    for L in range(0, n-win+1):
        R=L+win-1
        a,b,_ = loglog_linfit(t[L:R+1], x[L:R+1])
        err=abs(b-alpha_target)
        if err<bestErr:
            bestErr, bestL, bestR = err, L, R
    return bestL, bestR  # [web:223]

def loglog_linfit(t, y):
    X = np.log(np.maximum(np.asarray(t, float), 1e-300))
    Y = np.log(np.maximum(np.asarray(y, float), 1e-300))
    n = len(X); sx, sy = X.sum(), Y.sum()
    sxx, sxy = (X*X).sum(), (X*Y).sum()
    denom = n*sxx - sx*sx
    b = (n*sxy - sx*sy) / denom
    a = (sy - b*sx) / n
    resid = Y - (a + b*X)
    s2 = (resid @ resid) / max(1, n-2)
    var_b = s2 * n / denom
    return a, b, math.sqrt(max(var_b,0.0))  # [web:223]

def main():
    ap = argparse.ArgumentParser(description="Plot QTM <x> vs t with theory overlay, onset, and robust α-fit window.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--w", type=float, default=3.0)
    ap.add_argument("--F", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--A", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--minrun", type=int, default=3)
    ap.add_argument("--bins-per-decade", type=int, default=12)
    ap.add_argument("--roll", type=int, default=7)
    ap.add_argument("--fit-k", type=int, default=9, help="rolling window for local slope")
    ap.add_argument("--alpha-fit-tol", type=float, default=0.05, help="|slope-alpha| tolerance")
    ap.add_argument("--fit-minlen", type=int, default=6, help="minimum contiguous points to fit")
    ap.add_argument("--out", default="qtm_firstmoment_vs_t.png")
    ap.add_argument("--ratio-out", default="qtm_ratio_vs_t.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"t","average_x"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: t,average_x")  # [web:139]

    # Raw
    t_raw = df["t"].values.astype(float)
    x_raw = df["average_x"].values.astype(float)

    # Binning and smoothing
    if args.bins_per_decade > 0:
        t_b, x_b = log_bin_series(t_raw, x_raw, bins_per_decade=args.bins_per_decade)  # [web:324]
    else:
        t_b, x_b = t_raw, x_raw
    x_s = rolling_mean(x_b, w=args.roll) if args.roll and args.roll>1 else x_b  # [web:324]

    # Use binned/smoothed for plotting/fit
    t, x = t_b, x_s

    # Theory on binned times
    x_th = x_theory(t, args.w, args.F, args.alpha, args.A)  # [web:14][web:15]

    # Onset on binned/smoothed series
    t_star = onset_by_ratio(t, x, x_th, tol=args.tol, minrun=args.minrun)  # [web:14]

    # Auto-select α-window and fit only there
    L, R = find_powerlaw_window(t, x, args.alpha, k=args.fit_k, tol=args.alpha_fit_tol, minlen=args.fit_minlen)  # [web:223]
    a_hat, alpha_fit, alpha_se = loglog_linfit(t[L:R+1], x[L:R+1])  # [web:223]
    C_hat = math.exp(a_hat)
    x_fit_line = C_hat * (t ** alpha_fit)

    # Plot
    plt.figure(figsize=(7.6, 5.0))
    plt.loglog(t, x, "o-", ms=3, lw=1.2, label=r"simulation $\langle x\rangle$ (binned/smoothed)")  # [web:324]
    plt.loglog(t, x_th, "--", lw=1.6, label=f"theory (A={args.A:g})")  # [web:14][web:15]
    plt.loglog(t, x_fit_line, ":", lw=1.7, label=rf"fit (window): $\alpha_{{\rm fit}}={alpha_fit:.3f}\pm{alpha_se:.3f}$")  # [web:223]
    plt.axvspan(t[L], t[R], color="tab:green", alpha=0.08, label="fit window")  # [web:223]
    if t_star is not None:
        plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=rf"onset $t^*\approx {t_star:.2e}$")  # [web:14]
    plt.xlabel("t"); plt.ylabel(r"$\langle x \rangle$")
    plt.title(rf"QTM periodic: $w={args.w}$, $F={args.F}$, $\alpha={args.alpha}$, $A={args.A}$")  # [web:14]
    plt.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=220)  # [web:139]
    if args.show: plt.show()
    plt.close()

    # Ratio panel (binned/smoothed)
    ratio = x / np.maximum(x_th, 1e-300)
    plt.figure(figsize=(7.6, 3.9))
    plt.semilogx(t, ratio, "o-", ms=3, lw=1.2, label=r"$\langle x\rangle / \langle x\rangle_{\rm th}$")  # [web:223]
    plt.axhline(1.0, color="k", ls="--", lw=1.2, label="level = 1")  # [web:223]
    plt.fill_between(t, 1.0-args.tol, 1.0+args.tol, color="k", alpha=0.07, label=f"±{int(args.tol*100)}% band")  # [web:223]
    if t_star is not None:
        plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=rf"onset $t^*\approx {t_star:.2e}$")  # [web:14]
    plt.xlabel("t"); plt.ylabel("ratio"); plt.title("Ratio to theory vs t")  # [web:223]
    plt.legend(); plt.tight_layout(); plt.savefig(args.ratio_out, dpi=220)  # [web:139]
    if args.show: plt.show()
    plt.close()

    # Report
    print(f"Fit window: indices [{L},{R}] -> times [{t[L]:.2e}, {t[R]:.2e}]")  # [web:223]
    print(f"alpha_target={args.alpha:.3f}  alpha_fit={alpha_fit:.3f} ± {alpha_se:.3f}")  # [web:223]
    print(f"K_lead(alpha={args.alpha}, A={args.A}) = {klead(args.alpha, args.A):.6g}")  # [web:14][web:15]
    if t_star is None:
        print("No onset detected within tolerance and minrun on this grid.")  # [web:14]
    else:
        print(f"Estimated onset time t* ≈ {t_star:.2e}")  # [web:14]
    print(f"Saved: {args.out}"); print(f"Saved: {args.ratio_out}")  # [web:139]

if __name__ == "__main__":
    main()
