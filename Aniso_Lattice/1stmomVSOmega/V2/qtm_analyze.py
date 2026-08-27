#!/usr/bin/env python3
# =====================================================================
#  qtm_analyze.py
#
#  Analysis and plotting for the CSV written by qtm_omega_scan.cpp (V1),
#  qtm_omega_scan_v2.cpp and qtm_omega_scan_v3.cpp. The three share one
#  header, so the same script reads all of them; several files can be
#  passed at once and are concatenated.
#
#  It reproduces Fig. 2 of Zippin & Burov, "Anomalous Mobility Enhancement
#  in Restricted Geometries": the mean longitudinal displacement against
#  the number of sites in the cross-section, at fixed forces, on log-log
#  axes, with a guide line of slope -(1-alpha).
#
#  NOTHING IS FITTED. Every line drawn is an absolute prediction evaluated
#  from the parameters recorded in the CSV:
#
#    Eq. (35)   <x_par> = (D0 F)^a / [A G(1+a)^2] * (a_lat/Omega)^(1-a) * t^a
#    Eq. (10)   <x_par> = v_par t^a / [A G(1+a) Lambda]
#    Eq. (34)   <x_par> = v_par eps^(a-1) t^a / [A G(1+a)^2]
#
#  Eq. (35) is the paper's figure line. It uses the Einstein relation
#  v_par ~ D0 F and the near-recurrent limit 1-Q0 ~ Omega v_par / a_lat,
#  so it is the fully closed-form prediction. Eq. (10) keeps the exact
#  v_par and the exact Lambda = (1-Q0)^2 Li_{-a}(Q0)/Q0 and is therefore
#  the sharper statement of the same theory; the gap between the two is
#  the error of the asymptotic reduction, not a discrepancy with the data.
#
#  The only number chosen by hand anywhere is the vertical placement of
#  the dash-dot guide line, which carries no physics: it is a slope marker
#  and is offset upward purely so that it does not overlap the data, in
#  the same way as in the paper's figures. Its SLOPE is the prediction
#  -(1-alpha) and is never adjusted.
#
#  The measured slopes that get printed are diagnostics, obtained by
#  regressing the simulation points alone. They are never used to place
#  any curve in the figure.
#
#  Usage:
#    python3 qtm_analyze.py omega_scan_v2.csv
#    python3 qtm_analyze.py run_a.csv run_b.csv --t 1e17 --out fig
#    python3 qtm_analyze.py omega.csv --only-valid
# =====================================================================

import argparse
import sys
from math import gamma

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter

# ---------------------------------------------------------------------
REQUIRED = ["alpha", "A", "a", "b", "D0", "Omega", "F", "T",
            "x_mean", "x_stderr"]

# marker / linestyle per (model, bc), mirroring the paper's two lattices
STYLE = {
    ("invd2", "pbc"):     dict(marker="s", ls="--", label="1/d^2 rates, PBC"),
    ("invd2", "reflect"): dict(marker="h", ls=":",  label="1/d^2 rates, reflecting"),
    ("iso", "pbc"):       dict(marker="o", ls="-.", label="isotropic rates, PBC"),
    ("iso", "reflect"):   dict(marker="D", ls=":",  label="isotropic rates, reflecting"),
}
FALLBACK = dict(marker="^", ls="--", label="")

# tab10 blue / orange first, as in the paper
FORCE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            sys.exit(f"{p}: missing column(s) {missing} -- is this a scan CSV?")
        df["source"] = p
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in ("model", "bc"):
        if c not in df.columns:
            df[c] = "invd2" if c == "model" else "pbc"
    if "valid" not in df.columns:
        df["valid"] = 1
    # de-duplicate identical (source-independent) rows from repeated runs
    keys = ["model", "bc", "alpha", "A", "a", "b", "Omega", "F", "T"]
    df = df.drop_duplicates(subset=keys + ["x_mean"], keep="last")
    return df


def pick_time(df, want):
    """Keep one observation time: the requested one, else the largest."""
    times = np.sort(df["T"].unique())
    if want is None:
        T = times[-1]
    else:
        T = times[np.argmin(np.abs(np.log(times) - np.log(want)))]
        if not np.isclose(T, want, rtol=0.01):
            print(f"# requested t = {want:.3e} not in file; using t = {T:.3e}")
    return T, df[np.isclose(df["T"], T)].copy()


# ---------------------------------------------------------------------
#  Parameter-free predictions
# ---------------------------------------------------------------------
def eq35(d):
    """(D0 F)^a (a_lat/Omega)^(1-a) t^a / [A Gamma^2(1+a)] -- paper Eq. (35)."""
    al = d["alpha"].to_numpy()
    G1 = np.array([gamma(1.0 + x) for x in al])
    return ((d["D0"] * d["F"]) ** al
            * (d["a"] / d["Omega"]) ** (1.0 - al)
            * d["T"] ** al / (d["A"] * G1 ** 2))


def eq10(d):
    """v_par t^a / [A Gamma(1+a) Lambda] -- paper Eq. (10), exact Lambda."""
    if "x_theory_eq10" in d.columns and d["x_theory_eq10"].notna().all():
        return d["x_theory_eq10"].to_numpy()
    if not {"v_par", "Lambda_theory"}.issubset(d.columns):
        return None
    al = d["alpha"].to_numpy()
    G1 = np.array([gamma(1.0 + x) for x in al])
    return (d["v_par"] * d["T"] ** al
            / (d["A"] * G1 * d["Lambda_theory"])).to_numpy()


def wls_slope(x, y, sy):
    """Weighted least squares of log y on log x. Diagnostic only."""
    m = (x > 0) & (y > 0) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(x[m]), np.log(y[m])
    sl = np.where(sy[m] > 0, sy[m] / y[m], 1e-3)
    w = 1.0 / sl ** 2
    Sw, Sx, Sy = w.sum(), (w * lx).sum(), (w * ly).sum()
    Sxx, Sxy = (w * lx * lx).sum(), (w * lx * ly).sum()
    det = Sw * Sxx - Sx * Sx
    slope = (Sw * Sxy - Sx * Sy) / det
    inter = (Sxx * Sy - Sx * Sxy) / det
    se = np.sqrt(Sw / det)
    return slope, se, inter


def omega_ticks(ax, om):
    """Plain integer ticks on a log Omega axis when the range is narrow."""
    om = np.unique(np.asarray(om, dtype=float))
    if len(om) and om.max() / om.min() < 30:
        ax.set_xticks(om)
        ax.set_xticklabels([f"{v:g}" for v in om])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())


def yerr_of(d):
    """Asymmetric bootstrap CI if present, else the standard error."""
    y = d["x_mean"].to_numpy()
    if {"x_ci_lo", "x_ci_hi"}.issubset(d.columns) and d["x_ci_lo"].notna().all():
        lo = np.maximum(y - d["x_ci_lo"].to_numpy(), 0.0)
        hi = np.maximum(d["x_ci_hi"].to_numpy() - y, 0.0)
        return np.vstack([lo, hi])
    e = d["x_stderr"].to_numpy()
    return np.vstack([e, e])


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyse a QTM Omega scan (V1/V2/V3 CSV).")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--t", type=float, default=None,
                    help="observation time to plot (default: largest in the file)")
    ap.add_argument("--out", default="qtm_omega",
                    help="output stem; writes <stem>.png/.pdf and <stem>_diag.png")
    ap.add_argument("--only-valid", action="store_true",
                    help="drop rows the simulation flagged as outside the asymptotic regime")
    ap.add_argument("--guide-offset", type=float, default=1.8,
                    help="cosmetic vertical offset of the slope guide line (no physics)")
    ap.add_argument("--no-eq10", action="store_true",
                    help="draw only Eq. (35), omit the exact Eq. (10) line")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    df = load(args.csv)
    T, d = pick_time(df, args.t)

    if args.only_valid:
        n0 = len(d)
        d = d[d["valid"] == 1]
        print(f"# --only-valid: kept {len(d)}/{n0} rows")
    if d.empty:
        sys.exit("no rows left to plot")

    d = d.sort_values(["model", "bc", "F", "Omega"])
    alpha = d["alpha"].iloc[0]
    if d["alpha"].nunique() > 1:
        print("# warning: several alpha values in the data; the guide line uses "
              f"alpha = {alpha}")

    forces = np.sort(d["F"].unique())
    fcol = {f: FORCE_COLORS[i % len(FORCE_COLORS)] for i, f in enumerate(forces)}
    groups = list(d.groupby(["model", "bc"], sort=False))

    # -----------------------------------------------------------------
    #  Figure 1: the paper's Fig. 2
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for (model, bc), g in groups:
        st = STYLE.get((model, bc), FALLBACK)
        for f in np.sort(g["F"].unique()):
            s = g[g["F"] == f].sort_values("Omega")
            om = s["Omega"].to_numpy(float)
            th35 = eq35(s)
            ax.plot(om, th35, ls=st["ls"], color=fcol[f], lw=1.3, zorder=2)
            if not args.no_eq10:
                t10 = eq10(s)
                if t10 is not None:
                    ax.plot(om, t10, ls="-", color=fcol[f], lw=0.8, alpha=0.55, zorder=2)
            ax.errorbar(om, s["x_mean"].to_numpy(), yerr=yerr_of(s),
                        fmt=st["marker"], ms=7, color=fcol[f],
                        mec="white", mew=0.8, ecolor=fcol[f], elinewidth=1.2,
                        capsize=2.5, ls="none", zorder=3)

    # slope guide: exponent is the prediction, height is cosmetic only
    om_all = d["Omega"].to_numpy(float)
    og = np.array([om_all.min() * 0.9, om_all.max() * 1.1])
    ref = d[d["F"] == forces[-1]].sort_values("Omega")
    amp = eq35(ref).to_numpy()[0] * args.guide_offset * (ref["Omega"].to_numpy(float)[0]) ** (1 - alpha)
    ax.plot(og, amp * og ** (-(1 - alpha)), ls="-.", color="black", lw=1.4,
            label=rf"$\propto \Omega^{{-{1-alpha:.2g}}}$", zorder=1)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\Omega$", fontsize=13)
    ax.set_ylabel(r"$\langle x(t)\rangle$", fontsize=13)
    om_u = np.unique(d["Omega"].to_numpy(float))
    if om_u.max() / om_u.min() < 30:
        omega_ticks(ax, om_u)
    else:
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.tick_params(which="both", direction="in", top=True, right=True)

    # legend 1: forces
    fh = [plt.Line2D([], [], color=fcol[f], marker=STYLE.get(groups[0][0], FALLBACK)["marker"],
                     ls="none", ms=7, mec="white", label=rf"$F = {f:g}$") for f in forces]
    leg1 = ax.legend(handles=fh, title="Constant Force", loc="lower left",
                     fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)

    # legend 2: geometry / lines
    gh = []
    for (model, bc), _ in groups:
        st = STYLE.get((model, bc), FALLBACK)
        gh.append(plt.Line2D([], [], color="0.35", marker=st["marker"], ls=st["ls"],
                             ms=6, label=f"{model}/{bc} (Sim & Eq. 35)"))
    if not args.no_eq10:
        gh.append(plt.Line2D([], [], color="0.35", ls="-", lw=0.8, alpha=0.7,
                             label="Eq. (10), exact $\\Lambda$"))
    gh.append(plt.Line2D([], [], color="black", ls="-.", lw=1.4,
                         label=rf"$\propto \Omega^{{-{1-alpha:.2g}}}$"))
    ax.legend(handles=gh, loc="upper right", fontsize=8.5, framealpha=0.95)

    ax.set_title(rf"$t = {T:.0e}$,  $\alpha = {alpha:g}$,  $A = {d['A'].iloc[0]:g}$,  "
                 rf"$N = {int(d['N_walkers'].iloc[0]):,}$ walkers  (nothing fitted)",
                 fontsize=9.5, pad=8)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=args.dpi)
    fig.savefig(args.out + ".pdf")

    # -----------------------------------------------------------------
    #  Figure 2: diagnostics
    # -----------------------------------------------------------------
    fig2, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) simulation / Eq. (35)
    for (model, bc), g in groups:
        st = STYLE.get((model, bc), FALLBACK)
        for f in np.sort(g["F"].unique()):
            s = g[g["F"] == f].sort_values("Omega")
            r = s["x_mean"].to_numpy() / eq35(s).to_numpy()
            e = s["x_stderr"].to_numpy() / eq35(s).to_numpy()
            axs[0].errorbar(s["Omega"], r, yerr=e, fmt=st["marker"], ms=6,
                            color=fcol[f], mec="white", capsize=2.5, ls=st["ls"], lw=0.9)
    axs[0].axhline(1.0, color="k", lw=1.0, ls="-")
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"$\Omega$"); axs[0].set_ylabel(r"simulation / Eq. (35)")
    axs[0].set_title("(a) absolute test, no free parameters", fontsize=10)

    # (b) scaled mobility: A_alpha <x> / (F^a t^a) collapses onto D0^a (a/Omega)^(1-a)
    for (model, bc), g in groups:
        st = STYLE.get((model, bc), FALLBACK)
        for f in np.sort(g["F"].unique()):
            s = g[g["F"] == f].sort_values("Omega")
            al = s["alpha"].to_numpy()
            Aa = s["A"].to_numpy() * np.array([gamma(1 + x) for x in al]) ** 2
            mu = Aa * s["x_mean"].to_numpy() / (s["F"].to_numpy() ** al * s["T"].to_numpy() ** al)
            axs[1].plot(s["Omega"], mu, st["marker"], ms=6, color=fcol[f],
                        mec="white", ls="none")
            pred = s["D0"].to_numpy() ** al * (s["a"].to_numpy() / s["Omega"].to_numpy()) ** (1 - al)
            axs[1].plot(s["Omega"], pred, ls=st["ls"], color=fcol[f], lw=1.1)
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel(r"$\Omega$")
    axs[1].set_ylabel(r"$A_\alpha \langle x\rangle / (F^\alpha t^\alpha)$")
    axs[1].set_title(r"(b) Eq. (36): the two forces must collapse", fontsize=10)

    # (c) local slope between neighbouring Omega points
    for (model, bc), g in groups:
        st = STYLE.get((model, bc), FALLBACK)
        for f in np.sort(g["F"].unique()):
            s = g[g["F"] == f].sort_values("Omega")
            om = s["Omega"].to_numpy(float); y = s["x_mean"].to_numpy()
            if len(om) < 2:
                continue
            loc = np.diff(np.log(y)) / np.diff(np.log(om))
            mid = np.sqrt(om[:-1] * om[1:])
            axs[2].plot(mid, loc, st["marker"], ls=st["ls"], ms=6, color=fcol[f],
                        mec="white", lw=0.9)
    axs[2].axhline(-(1 - alpha), color="k", ls="-.", lw=1.4)
    axs[2].set_xscale("log"); axs[2].set_xlabel(r"$\Omega$ (geometric mean of the pair)")
    axs[2].set_ylabel("local slope  d ln<x> / d ln Omega")
    axs[2].set_title(rf"(c) local exponent vs. $-(1-\alpha) = {-(1-alpha):.2f}$", fontsize=10)

    for a_ in axs:
        a_.tick_params(which="both", direction="in", top=True, right=True)
        omega_ticks(a_, d["Omega"].to_numpy(float))
    axs[0].legend(handles=[plt.Line2D([], [], color=fcol[f], marker="s", ls="none",
                                      ms=6, mec="white", label=rf"$F = {f:g}$")
                           for f in forces], fontsize=9, loc="best")
    axs[1].text(0.03, 0.06, "lines: Eq. (36), one per force\n(they coincide by construction)",
                transform=axs[1].transAxes, fontsize=7.5, color="0.35")
    fig2.tight_layout()
    fig2.savefig(args.out + "_diag.png", dpi=args.dpi)

    # -----------------------------------------------------------------
    #  Printed report
    # -----------------------------------------------------------------
    print(f"\n# ===== QTM Omega scan, t = {T:.3e}, alpha = {alpha:g} =====")
    print("# prediction: <x> ~ Omega^-(1-alpha), slope = "
          f"{-(1-alpha):.3f};  no parameter is fitted anywhere.\n")

    hdr = (f"{'model/bc':<16}{'F':>9}{'Omega':>7}{'<x>_sim':>12}{'+-':>10}"
           f"{'Eq.35':>12}{'ratio':>8}{'Eq.10':>12}{'ratio':>8}{'eps':>10}{'N/N*':>8}{'ok':>4}")
    print(hdr); print("#" + "-" * (len(hdr) - 1))
    for (model, bc), g in groups:
        for f in np.sort(g["F"].unique()):
            s = g[g["F"] == f].sort_values("Omega")
            t35 = eq35(s).to_numpy(); t10 = eq10(s)
            for i, (_, row) in enumerate(s.iterrows()):
                r10 = f"{row['x_mean']/t10[i]:8.3f}" if t10 is not None else "     n/a"
                v10 = f"{t10[i]:12.4e}" if t10 is not None else "         n/a"
                print(f"{model+'/'+bc:<16}{f:>9g}{int(row['Omega']):>7}"
                      f"{row['x_mean']:>12.4e}{row['x_stderr']:>10.2e}"
                      f"{t35[i]:>12.4e}{row['x_mean']/t35[i]:>8.3f}{v10}{r10}"
                      f"{row.get('eps_theory', np.nan):>10.2e}"
                      f"{row.get('N_over_Nstar', np.nan):>8.1f}"
                      f"{'yes' if row.get('valid', 1) == 1 else 'NO':>4}")

    print("\n# measured Omega-exponent (diagnostic regression on the points alone):")
    for (model, bc), g in groups:
        for f in np.sort(g["F"].unique()):
            s = g[g["F"] == f].sort_values("Omega")
            sl, se, _ = wls_slope(s["Omega"].to_numpy(float), s["x_mean"].to_numpy(),
                                  s["x_stderr"].to_numpy())
            ok = s[s["valid"] == 1] if "valid" in s else s
            extra = ""
            if len(ok) >= 2 and len(ok) != len(s):
                slo, seo, _ = wls_slope(ok["Omega"].to_numpy(float), ok["x_mean"].to_numpy(),
                                        ok["x_stderr"].to_numpy())
                extra = f"   [valid rows only: {slo:+.4f} +- {seo:.4f}]"
            if not np.isfinite(sl):
                print(f"  {model}/{bc}  F={f:<8g} only {len(s)} point(s) -- no slope")
                continue
            dev = (sl - (alpha - 1.0)) / se if se > 0 else np.nan
            print(f"  {model}/{bc}  F={f:<8g} slope = {sl:+.4f} +- {se:.4f}"
                  f"   (predicted {alpha-1.0:+.4f}, {dev:+.1f} sigma){extra}")

    if len(forces) >= 2:
        f1, f2 = forces[0], forces[-1]
        pred = (f2 / f1) ** alpha
        print(f"\n# force scaling at fixed Omega:  <x>(F={f2:g})/<x>(F={f1:g}) "
              f"should be (F2/F1)^alpha = {pred:.4f}")
        for (model, bc), g in groups:
            for om in np.sort(g["Omega"].unique()):
                a1 = g[(g["F"] == f1) & (g["Omega"] == om)]["x_mean"]
                a2 = g[(g["F"] == f2) & (g["Omega"] == om)]["x_mean"]
                if len(a1) and len(a2):
                    r = a2.iloc[0] / a1.iloc[0]
                    print(f"  {model}/{bc}  Omega = {int(om):<5} ratio = {r:6.3f}"
                          f"   ({100*(r/pred-1):+.1f}% vs prediction)")

    if "valid" in d.columns and (d["valid"] == 0).any():
        n = int((d["valid"] == 0).sum())
        print(f"\n# NOTE: {n} of {len(d)} rows are flagged outside the asymptotic regime "
              "(eps too large, or <N> not yet far above N*). Eq. (35) is not expected to "
              "hold there; rerun with --only-valid to see the effect on the exponent.")
    if "capped" in d.columns and (d["capped"] > 0).any():
        print(f"# NOTE: {int(d['capped'].max())} walker(s) hit the step cap in at least one "
              "run; <x> is biased low for those rows.")
    if "top1pct_share" in d.columns and (d["top1pct_share"] > 0.25).any():
        print("# NOTE: in some rows the top 1% of walkers carry more than a quarter of <x>; "
              "the mean is tail-dominated there and the error bar is optimistic.")

    print(f"\n# wrote {args.out}.png, {args.out}.pdf, {args.out}_diag.png")


if __name__ == "__main__":
    main()