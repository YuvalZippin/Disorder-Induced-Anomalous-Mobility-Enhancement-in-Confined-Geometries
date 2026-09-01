#!/usr/bin/env python3
# =============================================================================
#  SIM 3 : <x_par(t)> vs cross-section area Omega   ->  sim3_graph.pdf
#  Colour  = geometry (Omega is the abscissa here, so it cannot encode colour)
#  Squares + dashed : 2D rectangular   (a,b)      periodic transverse BC
#  Circles + dotted : 3D orthorhombic  (a,b,c)    periodic transverse BC
#  Black dash-dot   : theory, Eq. (35), taken straight from the CSV
#  Geometries are discovered dynamically from the CSV headers.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# ------------------------------- settings ------------------------------------
CSV_MAIN   = "sim3_results.csv"
OUT_PDF    = "sim3_graph.pdf"
STYLE_FILE = None                     # e.g. "paper.mplstyle" -> applied verbatim
SHOW_ERR   = True
FIGSIZE    = (5.6, 4.4)

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

#   prefix        label                        marker  ls    palette index
GEOMS = [
    ("rect2D_",  r"2D rectangular $(a,b)$",     "s",   "--",  0),
    ("orth3D_",  r"3D orthorhombic $(a,b,c)$",  "o",   ":",   3),
]

# ------------------------------- rc params -----------------------------------
mpl.rcParams.update({
    "text.usetex":         True,
    "font.family":         "serif",
    "font.serif":          ["Computer Modern Roman", "DejaVu Serif"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size":           12,
    "axes.labelsize":      14,
    "axes.linewidth":      0.9,
    "legend.fontsize":     10,
    "legend.frameon":      True,
    "legend.framealpha":   1.0,
    "legend.edgecolor":    "black",
    "legend.fancybox":     False,
    "legend.borderpad":    0.45,
    "legend.handlelength": 2.1,
    "legend.labelspacing": 0.35,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.major.size":    5.0,
    "ytick.major.size":    5.0,
    "xtick.minor.size":    2.8,
    "ytick.minor.size":    2.8,
    "axes.grid":           False,
    "savefig.dpi":         600,
})
if STYLE_FILE:
    plt.style.use(STYLE_FILE)

# usetex is requested above; verify the TeX pipeline instead of dying at savefig
try:
    from matplotlib.texmanager import TexManager
    TexManager().make_dvi(r"$\Omega\ \langle x(t)\rangle$", 12)
except Exception as exc:
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"
    print(f"WARNING: LaTeX unavailable ({type(exc).__name__}); using mathtext. "
          f"Install texlive-latex-extra texlive-fonts-recommended cm-super dvipng.",
          file=sys.stderr)

# ------------------------ dynamic geometry discovery -------------------------
df = pd.read_csv(CSV_MAIN, comment="#")

found = []
for prefix, label, marker, ls, ci in GEOMS:
    cols = [prefix + c for c in ("Omega", "x_mean", "x_sem", "x_theory")]
    if not all(c in df.columns for c in cols):
        continue
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[(sub[cols[0]] > 0.0) & (sub[cols[1]] > 0.0)]   # log axis
    if sub.empty:
        continue
    sub = sub.sort_values(cols[0])
    found.append(dict(label=label, marker=marker, ls=ls, colour=PALETTE[ci],
                      om=sub[cols[0]].to_numpy(), x=sub[cols[1]].to_numpy(),
                      se=sub[cols[2]].to_numpy(), th=sub[cols[3]].to_numpy()))

if not found:
    raise SystemExit("no usable geometry columns in " + CSV_MAIN)


# --------------------------- helpers -----------------------------------------
def log_ticks(axis):
    """Readable labels inside a narrow decade span."""
    fmt = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    axis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20))
    axis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0),
                                             numticks=20))
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(fmt)


def clear_legend(fig, ax, leg, pad_pt=5.0, max_iter=40):
    """Grow the ordinate until no drawn curve enters the legend box."""
    for _ in range(max_iter):
        fig.canvas.draw()
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        bb = bb.padded(pad_pt * fig.dpi / 72.0)
        hit = False
        for ln in ax.get_lines():
            xy = ln.get_xydata()
            if len(xy) < 1:
                continue
            if len(xy) > 1:                       # densify: catch chords, not just nodes
                s = np.linspace(0.0, 1.0, 40)[None, :, None]
                a, b = xy[:-1][:, None, :], xy[1:][:, None, :]
                pts = (a + s * (b - a)).reshape(-1, 2)
            else:
                pts = xy
            d = ax.transData.transform(pts)
            if np.any((d[:, 0] >= bb.x0) & (d[:, 0] <= bb.x1) &
                      (d[:, 1] >= bb.y0) & (d[:, 1] <= bb.y1)):
                hit = True
                break
        if not hit:
            return
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ * 10 ** 0.05)
    print("WARNING: could not fully clear the legend; place it manually.",
          file=sys.stderr)


# --------------------------------- plot --------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

lo, hi = np.inf, -np.inf
for g in found:
    ax.plot(g["om"], g["th"], ls="-.", color="k", lw=1.1, zorder=2)   # Eq. (35)
    ax.plot(g["om"], g["x"], ls=g["ls"], color=g["colour"], lw=1.3, zorder=3)
    if SHOW_ERR:
        ax.errorbar(g["om"], g["x"], yerr=g["se"], fmt=g["marker"],
                    color=g["colour"], ms=5.0, mfc="none", mew=1.2,
                    elinewidth=0.8, capsize=2.0, ls="none", zorder=4)
    else:
        ax.plot(g["om"], g["x"], g["marker"], color=g["colour"], ms=5.0,
                mfc="none", mew=1.2, ls="none", zorder=4)
    lo = min(lo, g["th"].min(), g["x"].min())
    hi = max(hi, g["th"].max(), g["x"].max())

# --------------------------------- axes --------------------------------------
allom = np.concatenate([g["om"] for g in found])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel(r"$\langle x(t) \rangle$")
ax.set_xlim(allom.min() / 1.7, allom.max() * 1.7)
ax.set_ylim(lo / 1.9, hi * 2.2)                  # clear corners for the legend box
ax.tick_params(which="both", direction="in", top=True, right=True)
log_ticks(ax.xaxis)
log_ticks(ax.yaxis)

# -------------------------------- legend -------------------------------------
h_key = [Line2D([], [], color=g["colour"], ls=g["ls"], marker=g["marker"],
                mfc="none", mew=1.2, ms=5.0, lw=1.3, label=g["label"])
         for g in found]
h_key.append(Line2D([], [], color="k", ls="-.", lw=1.1, label=r"theory"))

leg = ax.legend(handles=h_key, loc="upper right", handletextpad=0.7,
                borderaxespad=0.8)
leg.get_frame().set_linewidth(0.8)
leg.set_zorder(10)
clear_legend(fig, ax, leg)

fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {os.path.abspath(OUT_PDF)}")