#!/usr/bin/env python3
# =====================================================================================
#  SIM 2 : <x_par(t)>  vs  cross-section area Omega, at several fixed forces.
#  Reads  sim2_results.csv  (+ optional sim2_params.json)  ->  sim2_graph.pdf
# =====================================================================================

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------- settings ----------------------------------------
CSV_FILE   = sys.argv[1] if len(sys.argv) > 1 else "sim2_results.csv"
JSON_FILE  = "sim2_params.json"
OUT_PDF    = "sim2_graph.pdf"
ALPHA_DEF  = 0.3                      # used only if sim2_params.json is absent
USE_TEX    = True
SHOW_ERR   = True                     # statistical error bars on the symbols
FIGSIZE    = (6.0, 4.6)
FORCE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
GUIDE_OFFSET = 0.30                   # guide line sits this factor below the data

# ----------------------------------- style -------------------------------------------
mpl.rcParams.update({
    "text.usetex":        USE_TEX,
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "DejaVu Serif"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.size":          11,
    "axes.labelsize":     14,
    "axes.titlesize":     13,
    "legend.fontsize":    10,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "axes.linewidth":     0.9,
    "axes.grid":          False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":   5.5,
    "ytick.major.size":   5.5,
    "xtick.minor.size":   3.0,
    "ytick.minor.size":   3.0,
    "xtick.major.width":  0.9,
    "ytick.major.width":  0.9,
    "lines.linewidth":    1.2,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,
    "savefig.bbox":       "tight",
})

# ----------------------------------- inputs ------------------------------------------
if not os.path.exists(CSV_FILE):
    sys.exit(f"missing {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

alpha = ALPHA_DEF
if os.path.exists(JSON_FILE):
    with open(JSON_FILE) as fh:
        alpha = json.load(fh).get("alpha", ALPHA_DEF)

GEOM = [  # (key, label, marker, linestyle)
    ("cubic", "Cubic", "s", "--"),
    ("hex",   "Hexagonal", "o", ":"),
]

forces = np.unique(df["Force"].values)
colors = {F: FORCE_COLORS[i % len(FORCE_COLORS)] for i, F in enumerate(forces)}


def fmt_force(F):
    e = int(np.floor(np.log10(F)))
    m = F / 10.0 ** e
    if abs(m - 1.0) < 1e-9:
        return r"$F=10^{%d}$" % e
    return r"$F=%g\times 10^{%d}$" % (m, e)


# ----------------------------------- figure ------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

all_x, all_y = [], []
for F in forces:
    sub = df[df["Force"] == F]
    for key, _lab, mk, ls in GEOM:
        om = sub[f"Omega_{key}"].to_numpy(dtype=float)
        y = sub[f"x_{key}"].to_numpy(dtype=float)
        e = sub[f"x_{key}_sem"].to_numpy(dtype=float) if SHOW_ERR else None
        good = np.isfinite(om) & np.isfinite(y) & (om > 0) & (y > 0)
        if not good.any():
            continue
        om, y = om[good], y[good]
        e = e[good] if e is not None else None
        order = np.argsort(om)
        om, y = om[order], y[order]
        e = e[order] if e is not None else None
        ax.errorbar(om, y, yerr=e, color=colors[F], marker=mk, linestyle=ls,
                    markersize=5.0, markerfacecolor="none", markeredgewidth=1.0,
                    elinewidth=0.8, capsize=0.0, zorder=3)
        all_x.append(om)
        all_y.append(y)

X = np.concatenate(all_x)
Y = np.concatenate(all_y)

# ------------------------- reference power law  Omega^-(1-alpha) ----------------------
slope = -(1.0 - alpha)
xg = np.array([X.min(), X.max()])
amp = GUIDE_OFFSET * np.min(Y * X ** (-slope))
ax.plot(xg, amp * xg ** slope, color="k", linestyle="-.", linewidth=1.3, zorder=2)

# ----------------------------------- axes --------------------------------------------
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel(r"$\langle x(t) \rangle$")

ylo = min(Y.min(), (amp * xg ** slope).min())
yhi = Y.max()
ax.set_xlim(X.min() / 1.8, X.max() * 1.8)
ax.set_ylim(ylo / 1.8, yhi * 9.0)

# ---------------------------------- legends ------------------------------------------
h_geom = [Line2D([], [], color="0.25", marker=mk, linestyle=ls, markersize=5.5,
                 markerfacecolor="none", markeredgewidth=1.0, label=lab)
          for _k, lab, mk, ls in GEOM]
h_geom.append(Line2D([], [], color="k", linestyle="-.", linewidth=1.3,
                     label=r"$\propto\Omega^{-(1-\alpha)}$"))

h_force = [Line2D([], [], color=colors[F], linestyle="-", linewidth=2.0,
                  label=fmt_force(F)) for F in forces]

leg1 = ax.legend(handles=h_geom, loc="upper right", bbox_to_anchor=(0.985, 0.985),
                 handlelength=2.2, borderpad=0.45, labelspacing=0.35)
leg1.get_frame().set_linewidth(0.8)
ax.add_artist(leg1)

fig.canvas.draw()
bb = leg1.get_window_extent().transformed(ax.transAxes.inverted())
leg2 = ax.legend(handles=h_force, loc="upper right",
                 bbox_to_anchor=(0.985, bb.y0 - 0.035),
                 handlelength=2.2, borderpad=0.45, labelspacing=0.35)
leg2.get_frame().set_linewidth(0.8)

fig.savefig(OUT_PDF, bbox_inches="tight", dpi=600)
print(f"wrote {OUT_PDF}  (alpha={alpha}, forces={list(forces)})")