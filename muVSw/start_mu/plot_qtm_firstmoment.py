#!/usr/bin/env python3
# plot_qtm_firstmoment.py
# Usage:
#   python3 plot_qtm_firstmoment.py --csv qtm_firstmoment_periodic_fast.csv --w 3 --F 0.01 --alpha 0.3 --A 1 --tol 0.1 --minrun 3 --show

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def klead(alpha, A):
    # K_lead(α) = 6^{-α}/(A Γ(1+α)^2)
    return (6.0 ** (-alpha)) / (A * (math.gamma(1.0 + alpha) ** 2))

def x_theory(t, w, F, alpha, A):
    K = klead(alpha, A)
    return K * (w ** (-2.0 * (1.0 - alpha))) * (F ** alpha) * (t ** alpha)

def onset_by_ratio(t, x, x_th, tol=0.1, minrun=3):
    # First time where |x - x_th| <= tol * x_th for at least 'minrun' consecutive points,
    # and remains within thereafter (conservative pick).
    x_th_safe = np.maximum(x_th, 1e-300)
    rel = np.abs(x - x_th) / x_th_safe
    inside = rel <= tol
    n = len(t)
    for i in range(n - minrun + 1):
        if np.all(inside[i:i+minrun]) and np.all(inside[i+minrun-1:]):
            return t[i+minrun-1]
    return None

def main():
    ap = argparse.ArgumentParser(description="Plot QTM <x> vs t with theory overlay and onset detection.")
    ap.add_argument("--csv", required=True, help="CSV with columns: t,average_x")
    ap.add_argument("--w", type=float, default=3.0, help="Width w (default 3)")
    ap.add_argument("--F", type=float, default=0.01, help="Force F (default 0.01)")
    ap.add_argument("--alpha", type=float, default=0.3, help="Alpha (default 0.3)")
    ap.add_argument("--A", type=float, default=1.0, help="Prefactor A (default 1)")
    ap.add_argument("--tol", type=float, default=0.10, help="Relative tolerance for onset band (default 0.10)")
    ap.add_argument("--minrun", type=int, default=3, help="Consecutive points required in band (default 3)")
    ap.add_argument("--out", default="qtm_firstmoment_vs_t.png", help="Output PNG filename")
    ap.add_argument("--ratio-out", default="qtm_ratio_vs_t.png", help="Optional ratio plot PNG filename")
    ap.add_argument("--show", action="store_true", help="Show plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"t","average_x"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: t,average_x")

    t = df["t"].values.astype(float)
    x = df["average_x"].values.astype(float)

    # Theory
    x_th = x_theory(t, args.w, args.F, args.alpha, args.A)

    # Onset detection
    t_star = onset_by_ratio(t, x, x_th, tol=args.tol, minrun=args.minrun)

    # Plot <x> vs t (log-log)
    plt.figure(figsize=(7.2, 4.8))
    plt.loglog(t, x, "o-", ms=3, lw=1.2, label=r"simulation $\langle x\rangle$")
    plt.loglog(t, x_th, "--", lw=1.6, label="theory (A=%.3g)" % args.A)
    if t_star is not None:
        plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=r"onset $t^*\approx %.2e$" % t_star)
    plt.xlabel("t")
    plt.ylabel(r"$\langle x \rangle$")
    plt.title(rf"QTM periodic: $w={args.w}$, $F={args.F}$, $\alpha={args.alpha}$, $A={args.A}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    if args.show:
        plt.show()
    plt.close()

    # Optional ratio plot: x/x_th vs t
    ratio = x / np.maximum(x_th, 1e-300)
    plt.figure(figsize=(7.2, 3.8))
    plt.semilogx(t, ratio, "o-", ms=3, lw=1.2, label=r"$\langle x\rangle / \langle x\rangle_{\rm th}$")
    plt.axhline(1.0, color="k", ls="--", lw=1.2, label="level = 1")
    band = args.tol
    plt.fill_between(t, 1.0-band, 1.0+band, color="k", alpha=0.07, label=f"±{int(band*100)}% band")
    if t_star is not None:
        plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=r"onset $t^*\approx %.2e$" % t_star)
    plt.xlabel("t")
    plt.ylabel("ratio")
    plt.title("Ratio to theory vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.ratio_out, dpi=220)
    if args.show:
        plt.show()
    plt.close()

    # Print report
    print(f"K_lead(alpha={args.alpha}, A={args.A}) = {klead(args.alpha, args.A):.6g}")
    if t_star is None:
        print("No onset detected within tolerance and minrun on this grid.")
    else:
        print(f"Estimated onset time t* ≈ {t_star:.2e}")
    print(f"Saved: {args.out}")
    print(f"Saved: {args.ratio_out}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# plot_qtm_firstmoment.py
# Usage:
#   python3 plot_qtm_firstmoment.py --csv qtm_firstmoment_periodic_fast.csv --w 3 --F 0.01 --alpha 0.3 --A 1 --tol 0.1 --minrun 3 --show

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def klead(alpha, A):
    # K_lead(α) = 6^{-α}/(A Γ(1+α)^2)
    return (6.0 ** (-alpha)) / (A * (math.gamma(1.0 + alpha) ** 2))

def x_theory(t, w, F, alpha, A):
    K = klead(alpha, A)
    return K * (w ** (-2.0 * (1.0 - alpha))) * (F ** alpha) * (t ** alpha)

def onset_by_ratio(t, x, x_th, tol=0.1, minrun=3):
    # First time where |x - x_th| <= tol * x_th for at least 'minrun' consecutive points,
    # and remains within thereafter (conservative pick).
    x_th_safe = np.maximum(x_th, 1e-300)
    rel = np.abs(x - x_th) / x_th_safe
    inside = rel <= tol
    n = len(t)
    for i in range(n - minrun + 1):
        if np.all(inside[i:i+minrun]) and np.all(inside[i+minrun-1:]):
            return t[i+minrun-1]
    return None

def main():
    ap = argparse.ArgumentParser(description="Plot QTM <x> vs t with theory overlay and onset detection.")
    ap.add_argument("--csv", required=True, help="CSV with columns: t,average_x")
    ap.add_argument("--w", type=float, default=3.0, help="Width w (default 3)")
    ap.add_argument("--F", type=float, default=0.01, help="Force F (default 0.01)")
    ap.add_argument("--alpha", type=float, default=0.3, help="Alpha (default 0.3)")
    ap.add_argument("--A", type=float, default=1.0, help="Prefactor A (default 1)")
    ap.add_argument("--tol", type=float, default=0.10, help="Relative tolerance for onset band (default 0.10)")
    ap.add_argument("--minrun", type=int, default=3, help="Consecutive points required in band (default 3)")
    ap.add_argument("--out", default="qtm_firstmoment_vs_t.png", help="Output PNG filename")
    ap.add_argument("--ratio-out", default="qtm_ratio_vs_t.png", help="Optional ratio plot PNG filename")
    ap.add_argument("--show", action="store_true", help="Show plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"t","average_x"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: t,average_x")

    t = df["t"].values.astype(float)
    x = df["average_x"].values.astype(float)

    # Theory
    x_th = x_theory(t, args.w, args.F, args.alpha, args.A)

    # Onset detection
    t_star = onset_by_ratio(t, x, x_th, tol=args.tol, minrun=args.minrun)

    # Plot <x> vs t (log-log)
    plt.figure(figsize=(7.2, 4.8))
    plt.loglog(t, x, "o-", ms=3, lw=1.2, label=r"simulation $\langle x\rangle$")
    plt.loglog(t, x_th, "--", lw=1.6, label="theory (A=%.3g)" % args.A)
    if t_star is not None:
        plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=r"onset $t^*\approx %.2e$" % t_star)
    plt.xlabel("t")
    plt.ylabel(r"$\langle x \rangle$")
    plt.title(rf"QTM periodic: $w={args.w}$, $F={args.F}$, $\alpha={args.alpha}$, $A={args.A}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    if args.show:
        plt.show()
    plt.close()

    # Optional ratio plot: x/x_th vs t
    ratio = x / np.maximum(x_th, 1e-300)
    plt.figure(figsize=(7.2, 3.8))
    plt.semilogx(t, ratio, "o-", ms=3, lw=1.2, label=r"$\langle x\rangle / \langle x\rangle_{\rm th}$")
    plt.axhline(1.0, color="k", ls="--", lw=1.2, label="level = 1")
    band = args.tol
    plt.fill_between(t, 1.0-band, 1.0+band, color="k", alpha=0.07, label=f"±{int(band*100)}% band")
    if t_star is not None:
        plt.axvline(t_star, color="tab:red", ls="-.", lw=1.6, label=r"onset $t^*\approx %.2e$" % t_star)
    plt.xlabel("t")
    plt.ylabel("ratio")
    plt.title("Ratio to theory vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.ratio_out, dpi=220)
    if args.show:
        plt.show()
    plt.close()

    # Print report
    print(f"K_lead(alpha={args.alpha}, A={args.A}) = {klead(args.alpha, args.A):.6g}")
    if t_star is None:
        print("No onset detected within tolerance and minrun on this grid.")
    else:
        print(f"Estimated onset time t* ≈ {t_star:.2e}")
    print(f"Saved: {args.out}")
    print(f"Saved: {args.ratio_out}")

if __name__ == "__main__":
    main()
