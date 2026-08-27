#!/usr/bin/env python3
"""
analyze_omega.py -- first moment <x_par(t)> versus cross-section Omega.

Reads the CSV written by qtm_omega_scan.cpp and produces the figure plus a
text report.

Design rule: NOTHING IS FITTED. Eq. (10) and Eq. (35) of the paper are
absolute predictions once alpha, A, T, and the lattice are fixed, so they are
drawn as-is. A free prefactor calibrated against the same data it is being
compared to would make the agreement a tautology rather than a test.

Two independent things are checked:
  (i)  the exponent   -- fitted slope on log-log vs the predicted -(1-alpha)
  (ii) the prefactor  -- absolute ratio <x>_sim / <x>_theory, expected 1

The theory is recomputed here from (a, b, w, F, model) independently of the
C++ and cross-checked against the CSV columns, so a bug in either side shows up
as a mismatch instead of hiding.

Usage:
  python3 analyze_omega.py --csv omega_a1_b2.csv
  python3 analyze_omega.py --csv iso.csv invd2.csv --out compare.png
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
#  Theory, mirroring qtm_omega_scan.cpp exactly
# ----------------------------------------------------------------------

def jump_weights(a, b, F, model):
    """Unnormalised jump weights. Force acts along +x only."""
    e = math.exp(F * a / 2.0)                       # F . e_x = F * a
    if model == "iso":
        return e, 1.0 / e, 1.0
    return e / (a * a), 1.0 / (e * a * a), 1.0 / (b * b)


def transport(a, b, w, F, model):
    """v_par, D_par, D_perp, eps -- paper Eqs. (19) and (33)."""
    wxp, wxm, wy = jump_weights(a, b, F, model)
    Z = wxp + wxm + 2.0 * wy
    pxp, pxm, py = wxp / Z, wxm / Z, wy / Z
    v = a * (pxp - pxm)
    return v, 0.5 * a * a * (pxp + pxm), b * b * py, w * v / a


def polylog_neg(alpha, z, tol=1e-16, jmax=200_000_000):
    """Li_{-alpha}(z) = sum_j j^alpha z^j, converging for 0 < z < 1."""
    if z <= 0.0:
        return 0.0
    s, zj = 0.0, z
    for j in range(1, jmax):
        term = (j ** alpha) * zj
        s += term
        zj *= z
        if j > 32 and term < tol * s:
            break
    return s


def g_star(a, b, w, Dpar, Dperp):
    """Geometric constant of Eqs. (29)-(30): G* = G - a^2/(pi^2 Omega D_par).

    In d = 2 the transverse modes are k_m = 2 pi m / (w b), m = 1 .. w-1.
    G* is what the paper drops when it writes 1-Q0 = Omega v/a, and it is a
    TIME-INDEPENDENT systematic -- no amount of extra t removes it.
    """
    G = 0.0
    w = int(round(w))
    for m in range(1, w):
        q = math.sqrt((2.0 * math.pi * m / (w * b)) ** 2 * Dperp)
        G += (1.0 / q) * math.atan(math.pi * math.sqrt(Dpar) / (a * q))
    G *= a / (math.pi * w * math.sqrt(Dpar))
    return G - a * a / (math.pi ** 2 * w * Dpar)


def predict(a, b, w, F, model, alpha, A, T, gstar=False):
    """Absolute predictions. Returns (eq10, eq35, eps, Lambda)."""
    v, Dpar, Dperp, eps = transport(a, b, w, F, model)
    if gstar and w > 1:
        # 1-Q0 = 1/(a/(Omega v) + G*) = eps0/(1 + G* eps0)   -- Eqs. (30),(12)
        eps = eps / (1.0 + g_star(a, b, w, Dpar, Dperp) * eps)
    G = math.gamma(1.0 + alpha)
    Ta = T ** alpha
    eq35 = v * eps ** (alpha - 1.0) * Ta / (A * G * G)          # Eq. (35)
    if 0.0 < eps < 1.0:
        Q0 = 1.0 - eps
        lam = (1.0 - Q0) ** 2 * polylog_neg(alpha, Q0) / Q0     # Eq. (6)
        eq10 = v * Ta / (A * G * lam)                           # Eq. (10)
    else:
        lam, eq10 = float("nan"), float("nan")
    return eq10, eq35, eps, lam


def D0_of(a, b, model):
    return a * a / 4.0 if model == "iso" else a * a * b * b / (2.0 * (a * a + b * b))


# ----------------------------------------------------------------------
#  Weighted straight-line fit in log-log space
# ----------------------------------------------------------------------

def wls_loglog(om, x, sx):
    """Returns slope, intercept, sigma_slope, sigma_intercept, chi2/dof."""
    X, Y = np.log(om), np.log(x)
    W = 1.0 / (sx / x) ** 2                       # sigma(ln x) = sigma(x)/x
    Sw, Sx = W.sum(), (W * X).sum()
    Sy, Sxx = (W * Y).sum(), (W * X * X).sum()
    Sxy = (W * X * Y).sum()
    det = Sw * Sxx - Sx * Sx
    m = (Sw * Sxy - Sx * Sy) / det
    c = (Sxx * Sy - Sx * Sxy) / det
    sm, sc = math.sqrt(Sw / det), math.sqrt(Sxx / det)
    dof = max(1, len(X) - 2)
    red = float((W * (Y - m * X - c) ** 2).sum() / dof)
    infl = math.sqrt(max(1.0, red))               # inflate if scatter exceeds errors
    return m, c, sm * infl, sc * infl, red


# ----------------------------------------------------------------------

def tex_t(T):
    """2.16e+17 -> $t = 2.2\\times10^{17}$"""
    e = int(math.floor(math.log10(T)))
    m = T / 10.0 ** e
    head = f"{m:.1f}\\times" if abs(m - 1.0) > 0.05 else ""
    return rf"$t = {head}10^{{{e}}}$"


def load(path):
    d = pd.read_csv(path)
    need = {"Omega", "T", "x_mean", "x_stderr", "alpha", "A", "a", "b", "F",
            "model", "valid"}
    missing = need - set(d.columns)
    if missing:
        sys.exit(f"[error] {path} is missing columns: {sorted(missing)}")
    for c in ("x_ci_lo", "x_ci_hi"):
        if c not in d.columns:
            d[c] = d["x_mean"] + (1.96 * d["x_stderr"]) * (1 if c == "x_ci_hi" else -1)
    return d


GSTAR = [False]


def cross_check(d, path):
    """Recompute the theory here and compare with what the C++ wrote."""
    if "x_theory_eq10" not in d.columns:
        return
    worst, where = 0.0, None
    for _, r in d.iterrows():
        e10, _, _, _ = predict(r.a, r.b, int(r.Omega), r.F, r.model,
                               r.alpha, r.A, r["T"], GSTAR[0])
        if np.isfinite(e10) and np.isfinite(r.x_theory_eq10) and r.x_theory_eq10 > 0:
            rel = abs(e10 / r.x_theory_eq10 - 1.0)
            if rel > worst:
                worst, where = rel, (int(r.Omega), r["T"])
    if GSTAR[0]:
        print(f"[note] {path}: theory includes G* (second order in the drive); "
              f"the C++ columns do not, so a mismatch below is expected")
        return
    if worst > 1e-3:
        print(f"[warn] {path}: python and C++ theory differ by {worst:.2%} "
              f"(worst at Omega={where[0]}, T={where[1]:.1e}). Check both.")
    else:
        print(f"[ok]   {path}: python theory matches the C++ columns "
              f"to {worst:.1e}")


def report(d, path):
    a, b = d.a.iloc[0], d.b.iloc[0]
    model, alpha, A = d.model.iloc[0], d.alpha.iloc[0], d.A.iloc[0]
    F, NW = d.F.iloc[0], int(d.N_walkers.iloc[0]) if "N_walkers" in d else -1
    D0 = D0_of(a, b, model)

    print(f"\n=== {path} ===")
    print(f"  a = {a:g}   b = {b:g}   a/b = {a/b:g}   model = {model}")
    print(f"  D0 = {D0:.6f}   F = {F:g}   alpha = {alpha:g}   A = {A:g}   "
          f"walkers = {NW}")
    print(f"  predicted Omega exponent: -(1-alpha) = {alpha-1:+.4f}")

    for T in sorted(d["T"].unique()):
        sub = d[np.isclose(d["T"], T)].sort_values("Omega")
        good = sub[sub.valid == 1]
        tag = f"  T = {T:.3e}  ({len(good)}/{len(sub)} rows valid)"
        if len(good) < 3:
            print(tag + "   -- too few valid points to fit")
            continue
        err = 0.5 * (good.x_ci_hi.values - good.x_ci_lo.values)
        err = np.where(err > 0, err, good.x_stderr.values)
        m, c, sm, sc, red = wls_loglog(good.Omega.values, good.x_mean.values, err)
        dev = (m - (alpha - 1.0)) / sm if sm > 0 else float("nan")
        print(tag + f"   slope = {m:+.4f} +- {sm:.4f}"
                    f"   ({dev:+.1f} sigma from prediction)   chi2/dof = {red:.2f}")

    T = d["T"].max()
    sub = d[np.isclose(d["T"], T)].sort_values("Omega")
    print(f"\n  at the largest T = {T:.3e}:")
    hdr = f"  {'Omega':>6} {'<x>_sim':>11} {'Eq.10':>11} {'Eq.35':>11} " \
          f"{'ratio':>7} {'eps':>8} {'<N>/N*':>8} {'top1%':>7} {'ok':>4}"
    print(hdr)
    for _, r in sub.iterrows():
        if GSTAR[0]:
            e10, e35, _, _ = predict(r.a, r.b, int(r.Omega), r.F, r.model,
                                     r.alpha, r.A, r["T"], True)
        else:
            e10 = r.get("x_theory_eq10", np.nan)
            e35 = r.get("x_theory_eq34", np.nan)
        rat = r.x_mean / e10 if (np.isfinite(e10) and e10 > 0) else np.nan
        print(f"  {int(r.Omega):>6d} {r.x_mean:>11.4g} {e10:>11.4g} {e35:>11.4g} "
              f"{rat:>7.3f} {r.eps_theory:>8.2e} {r.N_over_Nstar:>8.1f} "
              f"{r.get('top1pct_share', np.nan):>7.3f} "
              f"{'yes' if r.valid else 'NO':>4}")

    # honest warnings rather than silent averaging over bad points
    if "capped" in d and d.capped.max() > 0:
        print(f"  [warn] {int(d.capped.max())} walkers hit the step cap -- "
              f"<x> is biased low; raise --ncap.")
    if "top1pct_share" in sub and sub.top1pct_share.max() > 0.2:
        print(f"  [warn] top 1% of walkers carry "
              f"{sub.top1pct_share.max():.0%} of <x> -- heavy-tail dominated, "
              f"the mean is poorly resolved; raise N.")
    if (sub.valid == 0).any():
        bad = sorted(sub[sub.valid == 0].Omega.astype(int))
        print(f"  [note] excluded from the fit (outside the regime): Omega = {bad}")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="<x> vs Omega for the anisotropic QTM channel")
    ap.add_argument("--csv", nargs="+", required=True, help="one or more CSVs from qtm_omega_scan")
    ap.add_argument("--out", default="omega_scan.png")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--theory", default="eq10", choices=["eq10", "eq35", "both"],
                    help="which absolute prediction to draw")
    ap.add_argument("--hide-invalid", action="store_true",
                    help="drop points outside the validity regime instead of drawing them hollow")
    ap.add_argument("--gstar", action="store_true",
                    help="include the second-order geometric constant G* in the "
                         "theory (Eqs. 29-31) instead of stopping at 1-Q0 = Omega v/a")
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--single", action="store_true",
                    help="left panel only, paper style (use a .pdf extension for vector output)")
    ap.add_argument("--last-T", action="store_true",
                    help="keep only the largest t, the converged one")
    ap.add_argument("--figsize", default=None, help="W,H in inches, e.g. 7,5.5")
    ap.add_argument("--legend", default="full", choices=["full", "guide", "none"],
                    help="single-panel legend: everything, the guide line only, or nothing")
    ap.add_argument("--convergence", default=None, metavar="FILE",
                    help="also write the deficit-vs-t figure that shows the "
                         "offset is finite-time and decays as a power of t")
    ap.add_argument("--annotate", action="store_true",
                    help="stamp a,b,model,F,t,N on the figure so the run is unambiguous")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 13, "axes.labelsize": 17,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 11, "lines.linewidth": 1.6,
        "lines.markersize": 7, "figure.figsize": (12.5, 5.2),
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })

    frames = [load(p) for p in args.csv]
    if args.last_T:
        frames = [d[np.isclose(d["T"], d["T"].max())].copy() for d in frames]
    if args.single:
        plt.rcParams["figure.figsize"] = (7.0, 5.5)
    if args.figsize:
        plt.rcParams["figure.figsize"] = tuple(float(v) for v in args.figsize.split(","))
    for d, p in zip(frames, args.csv):
        GSTAR[0] = args.gstar
        if not args.no_crosscheck:
            cross_check(d, p)
        report(d, p)

    alpha = frames[0].alpha.iloc[0]
    slope = alpha - 1.0
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    markers = ["s", "o", "^", "D", "v"]

    if args.single:
        fig, axL = plt.subplots()
        axR = None
    else:
        fig, (axL, axR) = plt.subplots(1, 2)

    all_T = sorted(set(np.concatenate([d["T"].unique() for d in frames])))
    cmap = {T: colors[i % len(colors)] for i, T in enumerate(all_T)}

    for fi, (d, path) in enumerate(zip(frames, args.csv)):
        mk = markers[fi % len(markers)]
        a, b = d.a.iloc[0], d.b.iloc[0]
        model, A = d.model.iloc[0], d.A.iloc[0]
        F = d.F.iloc[0]

        for T in sorted(d["T"].unique()):
            sub = d[np.isclose(d["T"], T)].sort_values("Omega")
            if args.hide_invalid:
                sub = sub[sub.valid == 1]
            if sub.empty:
                continue
            col = cmap[min(all_T, key=lambda z: abs(z - T))]

            th = np.array([predict(a, b, o, F, model, alpha, A, T, args.gstar)[0]
                           for o in sub.Omega.values])

            for keep, fill in ((sub.valid == 1, col), (sub.valid == 0, "none")):
                s_ = sub[keep]
                if s_.empty:
                    continue
                idx = np.isin(sub.Omega.values, s_.Omega.values)
                lo = np.clip(s_.x_mean - s_.x_ci_lo, 0, None)
                hi = np.clip(s_.x_ci_hi - s_.x_mean, 0, None)
                axL.errorbar(s_.Omega, s_.x_mean, yerr=[lo, hi], fmt=mk,
                             color=col, mfc=fill, mec=col, ls="none",
                             capsize=2, elinewidth=1, zorder=4)
                if axR is not None:
                    t_ = th[idx]
                    axR.errorbar(s_.Omega, s_.x_mean / t_,
                                 yerr=[lo / t_, hi / t_], fmt=mk,
                                 color=col, mfc=fill, mec=col, ls="none",
                                 capsize=2, elinewidth=1, zorder=4)

            og = np.logspace(np.log10(sub.Omega.min()), np.log10(sub.Omega.max()), 200)
            for name, ls in (("eq10", "-"), ("eq35", "--")):
                if args.theory not in (name, "both"):
                    continue
                yy = np.array([predict(a, b, o, F, model, alpha, A, T, args.gstar)[
                                   0 if name == "eq10" else 1] for o in og])
                ok = np.isfinite(yy) & (yy > 0)
                axL.plot(og[ok], yy[ok], ls, color=col, lw=1.4, alpha=.85, zorder=2)

    # guide line of the predicted power, anchored above the data
    d0 = frames[0]
    og = np.logspace(np.log10(d0.Omega.min()), np.log10(d0.Omega.max()), 50)
    top = max(f.x_mean.max() for f in frames)
    axL.plot(og, 2.2 * top * (og / og[0]) ** slope, "-.", color="k", lw=1.5)

    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"$\Omega$")
    axL.set_ylabel(r"$\langle x_\parallel(t)\rangle$")

    if axR is not None:
        axR.axhspan(0.95, 1.05, color="0.85", zorder=0)
        axR.axhline(1.0, color="k", lw=1.2, ls="-", zorder=1)
        axR.set_xscale("log")
        axR.set_xlabel(r"$\Omega$")
        axR.set_ylabel(r"$\langle x_\parallel\rangle\,/\,"
                       r"\langle x_\parallel\rangle_{\rm Eq.(10)}$")
        axR.set_title(r"absolute test: on the line $\Rightarrow$ exponent "
                      r"and prefactor both correct", fontsize=11, pad=8)

    om_all = np.concatenate([f.Omega.values for f in frames])
    for ax in [a for a in (axL, axR) if a is not None]:
        if om_all.max() / om_all.min() < 30:
            ticks = sorted(set(om_all.astype(int)))
            ax.set_xticks(ticks, minor=False); ax.set_xticks([], minor=True)
            ax.set_xticklabels([str(t) for t in ticks])

    handles = []
    if len(all_T) > 1:
        handles += [Line2D([], [], color=cmap[T], marker="s", ls="none",
                           label=tex_t(T)) for T in all_T]
    if len(frames) > 1:
        handles += [Line2D([], [], color="gray", marker=markers[i % len(markers)],
                           ls="none", label=p.rsplit("/", 1)[-1].replace(".csv", ""))
                    for i, p in enumerate(args.csv)]
    _c = cmap[all_T[0]] if (len(all_T) == 1 and len(frames) == 1) else "gray"
    handles += [Line2D([], [], color=_c, ls="-",
                       label=("Eq. (10) + $G^*$" if args.gstar
                              else r"Eq. (10), exact $\Lambda$"))]
    if args.theory in ("eq35", "both"):
        handles += [Line2D([], [], color="gray", ls="--", label="Eq. (35), asymptotic")]
    if not args.hide_invalid and any((f.valid == 0).any() for f in frames):
        handles += [Line2D([], [], color="gray", marker="s", mfc="none", ls="none",
                           label=r"open: $\langle N\rangle/N^*<$ threshold")]
    guide = Line2D([], [], color="k", ls="-.", label=rf"$\propto \Omega^{{{slope:.2f}}}$")

    if axR is None:
        sel = {"full": handles + [guide], "guide": [guide], "none": []}[args.legend]
        if sel:
            axL.legend(handles=sel, loc="lower left", frameon=False, fontsize=11)
    else:
        axL.legend(handles=[guide], loc="lower left", frameon=False)
        lo_, hi_ = axR.get_ylim()
        axR.set_ylim(lo_ - 0.12 * (hi_ - lo_), hi_)
        axR.legend(handles=handles, loc="lower right", fontsize=10, ncol=2,
                   frameon=True, framealpha=0.92, edgecolor="none")

    if args.annotate:
        d0_ = frames[0]
        txt = (rf"$a={d0_.a.iloc[0]:g}$, $b={d0_.b.iloc[0]:g}$, "
               rf"{d0_.model.iloc[0]}, $F={d0_.F.iloc[0]:.4g}$" "\n"
               rf"$\alpha={d0_.alpha.iloc[0]:g}$, $A={d0_.A.iloc[0]:g}$, "
               rf"$t={d0_['T'].max():.1e}$, $N={int(d0_.N_walkers.iloc[0]):,}$")
        axL.text(0.97, 0.97, txt, transform=axL.transAxes, ha="right", va="top",
                 fontsize=10, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\n[saved] {args.out}")

    if args.convergence:
        fc, ac = plt.subplots(figsize=(6.4, 5.0))
        for fi, d in enumerate(frames):
            a, b = d.a.iloc[0], d.b.iloc[0]
            model, A, F = d.model.iloc[0], d.A.iloc[0], d.F.iloc[0]
            for j, Om in enumerate(sorted(d.Omega.unique())):
                g = d[d.Omega == Om].sort_values("T")
                th = np.array([predict(a, b, Om, F, model, alpha, A, T, args.gstar)[0]
                               for T in g["T"].values])
                defic = 1.0 - g.x_mean.values / th
                err = g.x_stderr.values / th
                pos = defic > 0
                ac.errorbar(g["T"].values[pos], defic[pos] * 100, yerr=err[pos] * 100,
                            marker=markers[fi % len(markers)],
                            color=plt.cm.viridis(j / max(1, len(d.Omega.unique()) - 1) * 0.88), ls="none", capsize=2,
                            elinewidth=1, label=rf"$\Omega={int(Om)}$" if fi == 0 else None)
        tt = np.array(sorted(set(np.concatenate([f["T"].values for f in frames]))))
        ref = 12.0 * (tt / tt.min()) ** (-alpha)
        ac.plot(tt, ref, "-.", color="k", lw=1.5,
                label=rf"$\propto t^{{-{alpha:g}}}$")
        ac.axhline(0.29, color="0.5", ls=":", lw=1.2)
        ac.text(tt.min(), 0.31, "statistical noise floor", fontsize=10, color="0.4")
        ac.set_xscale("log"); ac.set_yscale("log")
        ac.set_xlabel(r"$t$")
        ac.set_ylabel(r"$100\,\left[1-\langle x_\parallel\rangle/"
                      r"\langle x_\parallel\rangle_{\rm Eq.(10)}\right]$")
        ac.legend(frameon=False, fontsize=10, ncol=2)
        fc.tight_layout()
        fc.savefig(args.convergence, dpi=args.dpi, bbox_inches="tight")
        print(f"[saved] {args.convergence}")


if __name__ == "__main__":
    main()