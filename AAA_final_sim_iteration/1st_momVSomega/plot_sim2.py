#!/usr/bin/env python3
# =====================================================================================
#  SIM 2 : <x_par(t)> vs cross-section area Omega, at several fixed forces.
#  Parses the WIDE csv written by sim2_cross_section.cpp:
#      lattice,w,q,Omega,D0, x_F<val>,se_F<val>,eps_F<val>,njumps_F<val>,xth_F<val>, ...
#  Forces are detected dynamically from the "x_F..." column headers.
#  Output: sim2_graph.pdf

# run: python3 sim2_plot.py [sim2_results.csv]
# =====================================================================================

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------- settings ----------------------------------------
CSV_FILE  = sys.argv[1] if len(sys.argv) > 1 else "sim2_results.csv"
OUT_PDF   = "sim2_graph.pdf"
USE_TEX   = True
SHOW_ERR  = True                       # statistical error bars on the symbols
SHOW_THEO = True                       # xth_F... columns, black dash-dot
FIGSIZE   = (6.0, 4.6)
FORCE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

# lattice tag in the csv -> (legend label, marker, linestyle)
GEOM = {
    "cubic": ("Cubic",     "s", "--"),
    "hex":   ("Hexagonal", "o", ":"),
}

# ----------------------------------- style -------------------------------------------
mpl.rcParams.update({
    "text.usetex":         USE_TEX,
    "font.family":         "serif",
    "font.serif":          ["Computer Modern Roman", "DejaVu Serif"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.size":           11,
    "axes.labelsize":      14,
    "legend.fontsize":     10,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.linewidth":      0.9,
    "axes.grid":           False,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":    5.5,
    "ytick.major.size":    5.5,
    "xtick.minor.size":    3.0,
    "ytick.minor.size":    3.0,
    "xtick.major.width":   0.9,
    "ytick.major.width":   0.9,
    "lines.linewidth":     1.2,
    "legend.frameon":      True,
    "legend.framealpha":   1.0,
    "legend.edgecolor":    "black",
    "legend.fancybox":     False,
    "savefig.bbox":        "tight",
})

# ----------------------------------- parsing -----------------------------------------
if not os.path.exists(CSV_FILE):
    sys.exit(f"missing {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

# dynamic force detection from the wide column blocks
forces, tags = [], {}
for col in df.columns:
    if col.startswith("x_F"):
        tag = col[3:]                       # e.g. "0.0005"
        forces.append(float(tag))
        tags[float(tag)] = tag
forces = sorted(forces)
if not forces:
    sys.exit("no x_F* columns found")

colors = {F: FORCE_COLORS[i % len(FORCE_COLORS)] for i, F in enumerate(forces)}


def fmt_force(F):
    e = int(np.floor(np.log10(F)))
    m = F / 10.0 ** e
    if abs(m - 1.0) < 1e-9:
        return r"$F=10^{%d}$" % e
    return r"$F=%g\times 10^{%d}$" % (m, e)


# ----------------------------------- figure ------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)
all_y = []

for F in forces:
    tg = tags[F]
    for lat, (_lab, mk, ls) in GEOM.items():
        sub = df[df["lattice"] == lat].sort_values("Omega")
        if sub.empty:
            continue
        om = sub["Omega"].to_numpy(dtype=float)
        y = sub[f"x_F{tg}"].to_numpy(dtype=float)
        se = sub[f"se_F{tg}"].to_numpy(dtype=float) if SHOW_ERR else None

        # theory first, so the symbols sit on top of it
        if SHOW_THEO and f"xth_F{tg}" in sub:
            th = sub[f"xth_F{tg}"].to_numpy(dtype=float)
            g = np.isfinite(th) & (th > 0) & (om > 0)
            if g.any():
                ax.plot(om[g], th[g], color="k", linestyle="-.", linewidth=1.1, zorder=2)
                all_y.append(th[g])

        g = np.isfinite(y) & (y > 0) & (om > 0)
        if not g.any():
            continue
        ax.errorbar(om[g], y[g], yerr=(se[g] if se is not None else None),
                    color=colors[F], marker=mk, linestyle=ls, markersize=5.0,
                    markerfacecolor="none", markeredgewidth=1.0,
                    elinewidth=0.8, capsize=0.0, zorder=3)
        all_y.append(y[g])

Y = np.concatenate(all_y)
X = df["Omega"].to_numpy(dtype=float)

# ----------------------------------- axes --------------------------------------------
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel(r"$\langle x(t) \rangle$")
ax.set_xlim(X.min() / 1.8, X.max() * 1.8)
ax.set_ylim(Y.min() / 1.8, Y.max() * 9.0)

# ---------------------------------- legends ------------------------------------------
h_geom = [Line2D([], [], color="0.25", marker=mk, linestyle=ls, markersize=5.5,
                 markerfacecolor="none", markeredgewidth=1.0, label=lab)
          for lab, mk, ls in GEOM.values()]
if SHOW_THEO:
    h_geom.append(Line2D([], [], color="k", linestyle="-.", linewidth=1.1,
                         label=r"Theory"))

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
print(f"wrote {OUT_PDF}  (forces={forces})")