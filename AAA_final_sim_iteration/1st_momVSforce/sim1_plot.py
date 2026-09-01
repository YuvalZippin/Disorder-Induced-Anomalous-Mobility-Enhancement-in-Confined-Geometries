#!/usr/bin/env python3
# =============================================================================
#  SIM 1 : <x_par(t)> vs applied force F_par   ->  sim1_graph.pdf
#  Colour  = cross-section group (paired across lattices by similar Omega)
#  Squares + dashed : simple cubic      (periodic transverse BC)
#  Circles + dotted : simple hexagonal  (reflecting transverse BC)
#  Black dash-dot   : theory, Eq. (35), taken straight from the CSV
#  Channels are discovered dynamically from the CSV headers.
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------- settings ------------------------------------
CSV_MAIN   = "sim1_results.csv"
CSV_META   = "sim1_channels.csv"      # optional; supplies exact Omega
OUT_PDF    = "sim1_graph.pdf"
STYLE_FILE = None                     # e.g. "paper.mplstyle" -> applied verbatim
SHOW_ERR   = True
FIGSIZE    = (5.6, 4.4)

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
LSTYLE  = {"cubic": "--", "hex": ":"}
MARKER  = {"cubic": "s",  "hex": "o"}
LATNAME = {"cubic": r"cubic", "hex": r"hexagonal"}

# ------------------------------- rc params -----------------------------------
mpl.rcParams.update({
    "text.usetex":         False,         # בוטל השימוש במנוע חיצוני
    "mathtext.fontset":    "cm",          # שימוש בפונט Computer Modern הפנימי
    "font.family":         "serif",
    "font.serif":          ["cmr10"],     # שם הפונט שהמערכת מזהה
    # "text.latex.preamble": r"\usepackage{amsmath}", # לא נתמך במנוע הפנימי
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

# ------------------------- dynamic channel discovery -------------------------
df = pd.read_csv(CSV_MAIN)
F  = df["F"].to_numpy()

pat   = re.compile(r"^x_(cubic|hex)_w(\d+)$")
found = [(m.group(1), int(m.group(2))) for m in map(pat.match, df.columns) if m]
if not found:
    raise SystemExit("no channel columns of the form x_<lattice>_w<width> in " + CSV_MAIN)

def omega_of(lat, w):
    return w * w if lat == "cubic" else 3 * w * w - 3 * w + 1

omega = {(lat, w): omega_of(lat, w) for lat, w in found}
if os.path.exists(CSV_META):                     # prefer the exact count from the run
    meta = pd.read_csv(CSV_META)
    for r in meta.itertuples():
        omega[(r.lattice, int(r.w))] = int(r.Omega)

# pair the lattices by cross-section: the k-th narrowest cubic channel shares a
# colour with the k-th narrowest hexagonal channel
rank, colour = {}, {}
for lat in ("cubic", "hex"):
    for k, c in enumerate(sorted([c for c in found if c[0] == lat],
                                 key=lambda c: omega[c])):
        rank[c] = k
for c in found:
    colour[c] = PALETTE[rank[c] % len(PALETTE)]

channels = sorted(found, key=lambda c: (c[0] != "cubic", omega[c]))

# --------------------------------- plot --------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

lo, hi = np.inf, -np.inf
for lat, w in channels:
    tag = f"{lat}_w{w}"
    x   = df[f"x_{tag}"].to_numpy()
    se  = df[f"se_{tag}"].to_numpy()
    th  = df[f"xth_{tag}"].to_numpy()
    ok  = x > 0.0                                # log axis: drop non-positive means
    col = colour[(lat, w)]

    ax.plot(F, th, ls="-.", color="k", lw=1.1, zorder=2)           # theory, Eq. (35)
    ax.plot(F[ok], x[ok], ls=LSTYLE[lat], color=col, lw=1.3, zorder=3)
    if SHOW_ERR:
        ax.errorbar(F[ok], x[ok], yerr=se[ok], fmt=MARKER[lat], color=col,
                    ms=5.0, mfc="none", mew=1.2, elinewidth=0.8, capsize=2.0,
                    ls="none", zorder=4)
    else:
        ax.plot(F[ok], x[ok], MARKER[lat], color=col, ms=5.0, mfc="none",
                mew=1.2, ls="none", zorder=4)

    lo = min(lo, th.min(), x[ok].min() if ok.any() else np.inf)
    hi = max(hi, th.max(), x[ok].max() if ok.any() else -np.inf)

# --------------------------------- axes --------------------------------------
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$F$")
ax.set_ylabel(r"$\langle x(t) \rangle$")
ax.set_xlim(F.min() / 1.7, F.max() * 1.7)
ax.set_ylim(lo / 18.0, hi * 25.0)                # clear corners for the legend boxes
ax.tick_params(which="both", direction="in", top=True, right=True)

# -------------------------------- legends ------------------------------------
h_key = [Line2D([], [], color="k", ls=LSTYLE[lat], marker=MARKER[lat], mfc="none",
                mew=1.2, ms=5.0, lw=1.3, label=LATNAME[lat])
         for lat in ("cubic", "hex") if any(c[0] == lat for c in found)]
h_key.append(Line2D([], [], color="k", ls="-.", lw=1.1, label=r"theory"))

leg1 = ax.legend(handles=h_key, loc="upper left", handletextpad=0.7,
                 borderaxespad=0.8)
leg1.get_frame().set_linewidth(0.8)
ax.add_artist(leg1)

h_om = [Line2D([], [], color=colour[c], ls=LSTYLE[c[0]], marker=MARKER[c[0]],
               mfc="none", mew=1.2, ms=5.0, lw=1.3, label=r"$\Omega=%d$" % omega[c])
        for c in channels]
leg2 = ax.legend(handles=h_om, loc="lower right", ncol=2 if len(h_om) >= 4 else 1,
                 columnspacing=1.1, handletextpad=0.7, borderaxespad=0.8)
leg2.get_frame().set_linewidth(0.8)

fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {os.path.abspath(OUT_PDF)}")