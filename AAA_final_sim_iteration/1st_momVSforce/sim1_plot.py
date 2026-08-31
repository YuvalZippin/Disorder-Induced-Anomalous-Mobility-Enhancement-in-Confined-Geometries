#!/usr/bin/env python3
# =============================================================================
#  SIM 1 : <x_par(t)> vs applied force F_par   ->  sim1_graph.pdf
#  Symbols = simulation (C++ core), lines = theory Eq. (35), nothing fitted.
#    squares + dashed  : simple cubic      (periodic transverse BC)
#    circles + dotted  : simple hexagonal  (reflecting transverse BC)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------- settings ------------------------------------
CSV_MAIN   = "sim1_results.csv"
CSV_META   = "sim1_channels.csv"
OUT_PDF    = "sim1_graph.pdf"
STYLE_FILE = None            # e.g. "paper.mplstyle" -> applied verbatim
ALPHA      = 0.30            # disorder index used in the run
SHOW_ERR   = True            # statistical error bars on the simulation points
GUIDE_UP   = 3.0             # guide line placed this factor above the top curve
FIGSIZE    = (5.6, 4.4)

COLORS = {5: "#1f77b4", 10: "#ff7f0e", 15: "#2ca02c", 20: "#d62728"}
LSTYLE = {"cubic": "--", "hex": ":"}
MARKER = {"cubic": "s",  "hex": "o"}

# ------------------------------- rc params -----------------------------------
mpl.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size":          12,
    "axes.labelsize":     14,
    "axes.linewidth":     0.9,
    "legend.fontsize":    10,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,
    "legend.borderpad":   0.45,
    "legend.handlelength": 2.0,
    "legend.labelspacing": 0.35,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   5.0,
    "ytick.major.size":   5.0,
    "xtick.minor.size":   2.8,
    "ytick.minor.size":   2.8,
    "axes.grid":          False,
    "savefig.dpi":        600,
})
if STYLE_FILE:
    plt.style.use(STYLE_FILE)

# --------------------------------- data --------------------------------------
df   = pd.read_csv(CSV_MAIN)
meta = pd.read_csv(CSV_META)
F    = df["F"].to_numpy()

channels = [(r.lattice, int(r.w), int(r.Omega)) for r in meta.itertuples()]
widths   = sorted({w for _, w, _ in channels})

fig, ax = plt.subplots(figsize=FIGSIZE)

ymin, ymax = np.inf, -np.inf
for lat, w, Om in channels:
    tag = f"{lat}_w{w}"
    x   = df[f"x_{tag}"].to_numpy()
    se  = df[f"se_{tag}"].to_numpy()
    th  = df[f"xth_{tag}"].to_numpy()
    ok  = x > 0.0                                   # log axis: drop non-positive means
    col = COLORS.get(w, "k")

    ax.plot(F, th, ls=LSTYLE[lat], color=col, lw=1.3, zorder=2)
    if SHOW_ERR:
        ax.errorbar(F[ok], x[ok], yerr=se[ok], fmt=MARKER[lat], color=col,
                    ms=5.0, mfc="none", mew=1.2, elinewidth=0.8,
                    capsize=2.0, ls="none", zorder=3)
    else:
        ax.plot(F[ok], x[ok], MARKER[lat], color=col, ms=5.0, mfc="none",
                mew=1.2, ls="none", zorder=3)

    ymin = min(ymin, np.min(th), np.min(x[ok]) if ok.any() else np.inf)
    ymax = max(ymax, np.max(th), np.max(x[ok]) if ok.any() else -np.inf)

# ------------------------- guide line  x ~ F^alpha ---------------------------
Fg    = np.array([F.min(), F.max()])
Cg    = GUIDE_UP * ymax / F.max() ** ALPHA
ax.plot(Fg, Cg * Fg ** ALPHA, ls="-.", color="k", lw=1.2, zorder=1)

# --------------------------------- axes --------------------------------------
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$F$")
ax.set_ylabel(r"$\langle x(t) \rangle$")
ax.set_xlim(F.min() / 1.6, F.max() * 1.6)
ax.set_ylim(ymin / 15.0, Cg * Fg[-1] ** ALPHA * 20.0)  # headroom for the legends
ax.tick_params(which="both", direction="in", top=True, right=True)

# -------------------------------- legends ------------------------------------
h_geom = [
    Line2D([], [], color="k", ls=LSTYLE["cubic"], marker=MARKER["cubic"],
           mfc="none", mew=1.2, ms=5.0, lw=1.3, label=r"cubic"),
    Line2D([], [], color="k", ls=LSTYLE["hex"], marker=MARKER["hex"],
           mfc="none", mew=1.2, ms=5.0, lw=1.3, label=r"hexagonal"),
    Line2D([], [], color="k", ls="-.", lw=1.2,
           label=r"$\propto F^{%.1f}$" % ALPHA),
]
leg1 = ax.legend(handles=h_geom, loc="upper left", handletextpad=0.7,
                 borderaxespad=0.8)
leg1.get_frame().set_linewidth(0.8)
ax.add_artist(leg1)

h_w = [Line2D([], [], color=COLORS[w], ls="-", lw=1.8, label=r"$w=%d$" % w)
       for w in widths]
leg2 = ax.legend(handles=h_w, loc="lower right", ncol=1, handletextpad=0.7,
                 borderaxespad=0.8)
leg2.get_frame().set_linewidth(0.8)

fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {os.path.abspath(OUT_PDF)}")