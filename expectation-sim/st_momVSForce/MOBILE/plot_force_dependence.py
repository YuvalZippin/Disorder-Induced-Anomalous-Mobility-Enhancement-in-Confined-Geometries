#!/usr/bin/env python3
# =====================================================================================
#  plot_force_dependence.py
#
#  Mean longitudinal displacement versus applied force in a QTM channel.
#  Reads the CSV produced by qtm_channel_force.cpp (one file, both lattices).
#
#      python3 plot_force_dependence.py --csv qtm_force.csv --out fig_force
#
#  Two things differ from the previous plotting script, and both matter:
#
#    * NOTHING IS FITTED. The old script calibrated a global prefactor,
#          C_0 = mean( avg_x / bias ),
#      taken from the simulation itself, and also carried the ad-hoc kappa = 10 of the
#      old engines. The caption of the paper claims the opposite. Here the curves are
#      Eq. (19) evaluated with q and Omega and nothing else.
#
#    * The window is reported, not assumed. Every point is printed with its epsilon
#      (theoretical and measured), its Lambda (theoretical and measured), and the size
#      of the correction that the asymptotic Li_{-alpha}(1-eps) ~ Gamma(1+alpha)
#      eps^{-alpha-1} throws away. That table is the answer to "were you inside the
#      nearly recurrent limit".
#
#  ---------------------------------------------------------------------------------
#  THE STYLE BLOCK BELOW IS THE HOUSE STYLE FOR EVERY FIGURE IN THE PAPER.
#  Import it (from plot_force_dependence import apply_house_style, CHANNEL_COLORS, ...)
#  rather than re-typing it, so that all figures stay identical.
#  ---------------------------------------------------------------------------------
# =====================================================================================

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.special import gamma as Gamma, zeta as hurwitz_zeta


# =====================================================================================
#  HOUSE STYLE  --  keep identical across every figure of the paper
# =====================================================================================

FIGSIZE = (7.4, 5.6)          # rendered at \columnwidth, so labels land at ~8-9 pt

#  Colours encode the cross-section Omega, ordered from the narrowest channel to the
#  widest. Okabe-Ito palette: colour-blind safe and it survives greyscale printing.
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#8B4A9C", "#56B4E9", "#CC79A7"]

#  Lattice identity is carried by the marker and the line style, exactly as the
#  caption of the paper states: squares + dashed = simple cubic,
#                               hexagons + dotted = simple hexagonal.
LATTICE_MARKER = {"cubic": "s", "hex": "h"}
LATTICE_LINE = {"cubic": (0, (5.5, 2.2)), "hex": (0, (1.2, 1.8))}
LATTICE_NAME = {"cubic": "Cubic", "hex": "Hex."}

GUIDE_COLOR = "0.15"
GUIDE_STYLE = (0, (7, 2.5, 1.5, 2.5))


def apply_house_style():
    """Global rcParams. Call once, before creating any figure."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "cm",
        "font.size": 15,
        "axes.labelsize": 21,
        "axes.titlesize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12.5,
        "legend.title_fontsize": 12.5,
        "lines.linewidth": 1.9,
        "lines.markersize": 8.5,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6.5,
        "ytick.major.size": 6.5,
        "xtick.minor.size": 3.4,
        "ytick.minor.size": 3.4,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.handletextpad": 0.7,
        "legend.labelspacing": 0.45,
        "legend.borderaxespad": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.figsize": FIGSIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _minor_label(x, _pos):
    """Label only the 2 x 10^k and 5 x 10^k minor ticks; a bare pair of decade labels
    looks empty on a two-decade axis, and labelling every minor looks noisy."""
    if x <= 0:
        return ""
    k = np.floor(np.log10(x))
    m = x / 10.0 ** k
    if abs(m - 2.0) < 0.05 or abs(m - 5.0) < 0.05:
        return r"$%d{\times}10^{%d}$" % (int(round(m)), int(k))
    return ""


def style_log_axis(axis):
    axis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,), numticks=20))
    axis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    axis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1 * 10,
                                              numticks=200))
    axis.set_minor_formatter(mticker.FuncFormatter(_minor_label))


def log_slope_angle(ax, x0, y0, x1, y1):
    """Screen angle of the segment (x0,y0)-(x1,y1), for rotating an inline label."""
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    return np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


# =====================================================================================
#  Theory  --  no free parameters
# =====================================================================================

def zeta_any(s):
    """Riemann zeta for arbitrary real s, via reflection when s < 1."""
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    hi = s > 1.0
    out[hi] = hurwitz_zeta(s[hi], 1.0)
    lo = ~hi
    if np.any(lo):
        sl = s[lo]
        out[lo] = (2.0 ** sl * np.pi ** (sl - 1.0) * np.sin(np.pi * sl / 2.0)
                   * Gamma(1.0 - sl) * hurwitz_zeta(1.0 - sl, 1.0))
    return out


def polylog_neg(alpha, z, kmax=6):
    """Li_{-alpha}(z) for 0 < z < 1, through the Wood expansion

           Li_s(e^{-mu}) = Gamma(1-s) mu^{s-1} + sum_k zeta(s-k) (-mu)^k / k!

    which converges for |mu| < 2 pi and is therefore essentially exact here, where
    mu = -ln(1-eps) <= 0.05. The first term is the singular one whose leading part,
    Gamma(1+alpha) eps^{-alpha-1}, is the approximation used in Eq. (17)."""
    z = np.asarray(z, dtype=float)
    mu = -np.log(z)
    out = Gamma(1.0 + alpha) * mu ** (-alpha - 1.0)
    fact = 1.0
    for k in range(kmax + 1):
        if k > 0:
            fact *= k
        out = out + zeta_any(np.full_like(mu, -alpha - k)) * (-mu) ** k / fact
    return out


def v_par(F, q):
    """Exact single-step longitudinal drift, v = p(+) - p(-) with p ~ N exp(F.e/2)."""
    return 2.0 * np.sinh(0.5 * F) / (2.0 * np.cosh(0.5 * F) + (q - 2.0))


def x_theory_asymptotic(F, Omega, q, alpha, T, A=1.0):
    """Eq. (19):  <x> = (D0 F)^alpha (a/Omega)^(1-alpha) t^alpha / [A Gamma^2(1+alpha)].
    v_par is used instead of D0 F; the two differ by O(F^2/24), i.e. below 1e-6 here."""
    v = v_par(F, q)
    return v ** alpha * Omega ** (alpha - 1.0) * T ** alpha / (A * Gamma(1.0 + alpha) ** 2)


def x_theory_exact_lambda(F, Omega, q, alpha, T, A=1.0):
    """Eq. (14) with the full Lambda = (1-Q0)^2 Li_{-alpha}(Q0)/Q0, still using
    1 - Q0 = Omega v_par. Its ratio to the asymptotic form measures how much the
    eps -> 0 expansion costs at the eps actually simulated."""
    v = v_par(F, q)
    eps = Omega * v
    Q0 = 1.0 - eps
    lam = eps ** 2 * polylog_neg(alpha, Q0) / Q0
    return v * T ** alpha / (A * Gamma(1.0 + alpha) * lam)


def lambda_asymptotic(eps, alpha):
    return Gamma(1.0 + alpha) * eps ** (1.0 - alpha)


# =====================================================================================
#  Diagnostics
# =====================================================================================

def report(df, alpha, T, A):
    print("\n" + "=" * 96)
    print(" VALIDITY WINDOW AND CONSISTENCY  (eps = 1 - Q0 = Omega v_par)")
    print("=" * 96)
    hdr = ("  lattice   w  Omega        F      eps_th     eps_sim  sim/th     rho"
           "    exact/asym    sim/thy")
    print(hdr)
    print("  " + "-" * 94)

    for (lat, w), g in df.groupby(["lattice", "w"], sort=False):
        g = g.sort_values("F")
        for _, r in g.iterrows():
            lam_th = lambda_asymptotic(r["eps_theory"], alpha)
            ratio_exact = (x_theory_exact_lambda(r["F"], r["Omega"], r["q"], alpha, T, A)
                           / x_theory_asymptotic(r["F"], r["Omega"], r["q"], alpha, T, A))
            ratio_sim = r["mean_x"] / x_theory_asymptotic(r["F"], r["Omega"], r["q"],
                                                          alpha, T, A)
            em = r.get("esc_meas", np.nan)
            print("  %-7s %3d %6d  %9.2e  %9.2e  %9.2e  %6.3f  %7.0f   %8.4f   %8.4f"
                  % (lat, w, r["Omega"], r["F"], r["eps_theory"], em,
                     em / r["eps_theory"], r.get("rho", np.nan),
                     ratio_exact, ratio_sim))
        print("  " + "-" * 94)

    eps_max = df["eps_theory"].max()
    print("  max eps in the data set : %.4f   (eps = 1-Q0 = Omega v_par)" % eps_max)
    if eps_max > 0.10:
        print("  ** eps > 0.10: the nearly recurrent expansion costs more than 2 per cent.")
    elif eps_max > 0.06:
        print("  note: eps above 0.06; the prefactor error approaches a few per cent.")
    else:
        print("  eps is small enough throughout.")

    # The second condition, and in practice the binding one. eps = Omega v is the escape
    # probability of the infinite walk; it is only realised once the drift has overtaken
    # the longitudinal diffusion. Truncating at N leaves D/N above Omega v by roughly
    # 0.7/sqrt(rho), measured on a plain biased walk on this same channel:
    #     rho     3      10     33     98     330
    #     D/N     1.61   1.23   1.12   1.05   1.06     (in units of Omega v)
    # so rho of order a hundred is needed, not of order ten.
    if "rho" in df:
        rho_min = df["rho"].min()
        print("  min rho in the data set : %.0f   (rho = <N> v^2 / D0)" % rho_min)
        if rho_min < 60:
            print("  ** rho < 60: 1-Q0 has NOT converged to Omega v. The asymptotic law does")
            print("     not apply at these points; raise t or the force, or narrow the channel.")
        elif rho_min < 200:
            print("  note: rho below 200; expect 1-Q0 a few per cent above Omega v.")
        else:
            print("  rho is large enough throughout.")
    if "esc_meas" in df:
        bad = df[df["esc_meas"] / df["eps_theory"] > 1.05]
        if len(bad):
            print("  ** %d of %d points have a measured 1-Q0 more than 5%% above Omega v."
                  % (len(bad), len(df)))
        else:
            print("  the measured 1-Q0 agrees with Omega v to better than 5%% everywhere.")

    # slope in F, channel by channel
    print("\n" + "=" * 96)
    print(" MEASURED EXPONENTS   (expected: d ln<x> / d ln F = alpha = %.3f)" % alpha)
    print("=" * 96)
    for (lat, w), g in df.groupby(["lattice", "w"], sort=False):
        g = g.sort_values("F")
        s, _ = np.polyfit(np.log(g["F"]), np.log(g["mean_x"]), 1)
        dev = g["mean_x"] / x_theory_asymptotic(g["F"], g["Omega"], g["q"], alpha, T, A)
        print("  %-7s w=%-3d Omega=%-5d  slope = %.4f   sim/theory = %.4f +/- %.4f"
              % (lat, w, g["Omega"].iloc[0], s, dev.mean(), dev.std()))

    # exponent of the cross-section, at fixed force, across all channels
    print("\n" + "=" * 96)
    print(" CROSS-SECTION EXPONENT  (expected: d ln[<x>/v^alpha] / d ln Omega ="
          " alpha - 1 = %.3f)" % (alpha - 1.0))
    print("=" * 96)
    for F0, g in df.groupby("F", sort=True):
        if len(g) < 3:
            continue
        y = np.log(g["mean_x"] / v_par(g["F"], g["q"]) ** alpha)
        s, _ = np.polyfit(np.log(g["Omega"]), y, 1)
        print("  F = %.3e   exponent = %+.4f" % (F0, s))
    print()


# =====================================================================================
#  Figure
# =====================================================================================

def make_figure(df, alpha, T, A, out_stem, show_errorbars=True, show_guide=True):
    apply_house_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    channels = (df.groupby(["lattice", "w"], sort=False)["Omega"]
                  .first().reset_index().sort_values("Omega"))

    F_lo, F_hi = df["F"].min(), df["F"].max()
    F_dense = np.logspace(np.log10(F_lo), np.log10(F_hi), 300)

    y_lo, y_hi = np.inf, -np.inf
    handles = []

    for i, (_, ch) in enumerate(channels.iterrows()):
        lat, w, Om = ch["lattice"], int(ch["w"]), int(ch["Omega"])
        g = df[(df["lattice"] == lat) & (df["w"] == w)].sort_values("F")
        q = g["q"].iloc[0]
        col = PALETTE[i % len(PALETTE)]

        y_th = x_theory_asymptotic(F_dense, Om, q, alpha, T, A)
        ax.plot(F_dense, y_th, color=col, linestyle=LATTICE_LINE[lat],
                linewidth=1.9, zorder=2, solid_capstyle="round")

        if show_errorbars and "sem_x" in g:
            ax.errorbar(g["F"], g["mean_x"], yerr=g["sem_x"],
                        fmt=LATTICE_MARKER[lat], color=col, markerfacecolor=col,
                        markeredgecolor="white", markeredgewidth=0.8,
                        elinewidth=1.0, capsize=0.0, linestyle="none", zorder=4)
        else:
            ax.plot(g["F"], g["mean_x"], LATTICE_MARKER[lat], color=col,
                    markerfacecolor=col, markeredgecolor="white",
                    markeredgewidth=0.8, linestyle="none", zorder=4)

        y_lo = min(y_lo, g["mean_x"].min(), y_th.min())
        y_hi = max(y_hi, g["mean_x"].max(), y_th.max())

        handles.append(Line2D([0], [0], color=col, marker=LATTICE_MARKER[lat],
                              linestyle=LATTICE_LINE[lat], linewidth=1.9,
                              markerfacecolor=col, markeredgecolor="white",
                              markeredgewidth=0.8, markersize=8.5,
                              label=r"%s, $w=%d$, $\Omega=%d$"
                                    % (LATTICE_NAME[lat], w, Om)))

    # ---- limits: headroom above for the guide line, headroom below for the legend ----
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(F_lo / 1.45, F_hi * 1.45)
    ax.set_ylim(y_lo / 4.0, y_hi * 2.3)

    # ---- guide line, parallel to the data and clear of it ---------------------------
    if show_guide:
        y_guide = y_hi * 1.55 * (F_dense / F_hi) ** alpha
        ax.plot(F_dense, y_guide, color=GUIDE_COLOR, linestyle=GUIDE_STYLE,
                linewidth=1.6, zorder=3)
        fig.canvas.draw()
        xa = np.exp(0.36 * np.log(F_lo) + 0.64 * np.log(F_hi))
        ya = y_hi * 1.55 * (xa / F_hi) ** alpha
        ang = log_slope_angle(ax, F_lo, y_hi * 1.55 * (F_lo / F_hi) ** alpha,
                              F_hi, y_hi * 1.55)
        ax.annotate(r"$\propto F_\parallel^{\,%.1f}$" % alpha,
                    xy=(xa, ya), xytext=(0, 9), textcoords="offset points",
                    rotation=ang, rotation_mode="anchor",
                    ha="center", va="bottom", color=GUIDE_COLOR, fontsize=15)

    # ---- axes -----------------------------------------------------------------------
    ax.set_xlabel(r"$F_\parallel$", labelpad=2)
    ax.set_ylabel(r"$\langle x_\parallel(t)\rangle$", labelpad=4)
    style_log_axis(ax.xaxis)
    style_log_axis(ax.yaxis)
    ax.tick_params(which="both", top=True, right=True)
    ax.tick_params(which="minor", labelsize=12.5)   # minor labels a touch
    ax.tick_params(which="major", labelsize=15.0)   # smaller than the decades

    # ---- legend: lower right, inside the band of empty space under the curves -------
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    borderaxespad=0.9, labelspacing=0.5, handlelength=2.6,
                    handletextpad=0.75)
    leg.set_zorder(6)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out_stem, ext), dpi=600)
        print("  wrote %s.%s" % (out_stem, ext))
    plt.close(fig)


# =====================================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Mean displacement versus force for the QTM channel.")
    ap.add_argument("--csv", nargs="+", default=["qtm_force.csv"],
                    help="one or more CSV files from qtm_channel_force.cpp")
    ap.add_argument("--out", default="fig_force_dependence",
                    help="output stem; .pdf and .png are written")
    ap.add_argument("--channels", nargs="*", default=None,
                    help="subset, as lattice:w pairs, e.g. cubic:3 hex:5")
    ap.add_argument("--no-errorbars", action="store_true")
    ap.add_argument("--no-guide", action="store_true")
    args = ap.parse_args()

    frames = []
    for path in args.csv:
        try:
            frames.append(pd.read_csv(path))
        except FileNotFoundError:
            sys.exit("could not open %s" % path)
    df = pd.concat(frames, ignore_index=True)

    if args.channels:
        keep = {(c.split(":")[0], int(c.split(":")[1])) for c in args.channels}
        df = df[[(l, w) in keep for l, w in zip(df["lattice"], df["w"])]]
        if df.empty:
            sys.exit("no rows left after --channels filtering")

    alpha = float(df["alpha"].iloc[0])
    T = float(df["T"].iloc[0])
    A = float(df["A"].iloc[0]) if "A" in df else 1.0
    if df["alpha"].nunique() > 1 or df["T"].nunique() > 1:
        sys.exit("this figure expects a single alpha and a single T in the input")

    print("  alpha = %.3f    t = %.3e    A = %.3f    walkers = %d"
          % (alpha, T, A, int(df["M"].iloc[0])))

    report(df, alpha, T, A)
    make_figure(df, alpha, T, A, args.out,
                show_errorbars=not args.no_errorbars,
                show_guide=not args.no_guide)


if __name__ == "__main__":
    main()