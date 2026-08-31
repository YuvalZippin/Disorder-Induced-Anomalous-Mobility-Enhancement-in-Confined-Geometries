#!/usr/bin/env python3
# =====================================================================================
#  check_escape_window.py
#
#  Calibrates the validity window of the QTM channel figures, WITHOUT any trapping.
#
#      python3 check_escape_window.py                 # both lattices, default channels
#      python3 check_escape_window.py --alpha 0.3 --t 1e24
#
#  WHY THIS EXISTS
#  ---------------
#  The transport law rests on 1 - Q_0 = Omega v_par. That is the escape probability of
#  the INFINITE walk. A walk of finite length N has not yet realised it: the drift only
#  overtakes the longitudinal diffusion after N* = D0/v^2 steps, and until then the
#  walker keeps finding fresh sites at a rate above Omega v. The control parameter is
#
#        rho = N v^2 / D0 .
#
#  This script measures the ratio (D/N)/(Omega v) directly, by running a plain biased
#  nearest-neighbour walk on the same channel and counting distinct sites. No traps, no
#  subordination, nothing to get wrong. Reading the ratio against rho tells you, for
#  your own lattice and your own tolerance, how large rho has to be -- and therefore
#  which forces are usable at a given measurement time.
#
#  Reference numbers measured on the simple cubic channel with w = 3 at eps = 0.021:
#
#        rho        3      10      33      98     330
#        ratio    1.61    1.23    1.12    1.05    1.06
#
#  so rho of order a hundred is needed for five per cent, not of order ten. Note that
#  the residual at large rho is set by eps instead: the non-uniform transverse modes
#  (the G* term of the paper) pull 1 - Q_0 below Omega v by a term of order eps.
#
#  HOW TO USE THE OUTPUT
#  ---------------------
#  Pick the largest ratio you are willing to carry as a systematic (it propagates to
#  <x> as ratio^(alpha-1), i.e. a 5% error in 1-Q_0 becomes 3.5% in the displacement).
#  Read off the rho that achieves it. The script then prints, for each channel, the
#  force window that satisfies both
#        rho >= rho_min          (drift regime reached)
#        eps  = Omega v <= eps_max   (nearly recurrent expansion accurate)
#  at the measurement time you supply, together with the smallest t for which that
#  window is non-empty. Omega_max scales only as t^(alpha/2), so widening the channel
#  is expensive: doubling it costs a factor of a hundred in t.
# =====================================================================================

import argparse
import numpy as np
from scipy.special import gamma as Gamma


# -------------------------------------------------------------------------------------
def jump_probs(F, q):
    """p(+), p(-) and the exact single-step drift for a channel of coordination q."""
    nrm = 1.0 / (2.0 * np.cosh(0.5 * F) + (q - 2.0))
    pp, pm = nrm * np.exp(0.5 * F), nrm * np.exp(-0.5 * F)
    return pp, pm, pp - pm


def hex_patch(w):
    """Sites of a hexagonal patch of w shells on the triangular lattice, and an index."""
    pts = [(p, r) for p in range(-w, w + 1) for r in range(-w, w + 1)
           if abs(p + r) <= w]
    idx = {pt: i for i, pt in enumerate(pts)}
    return pts, idx


def walk_cubic(w, F, N, rng):
    """Plain biased walk on a square-section channel with periodic transverse walls."""
    pp, pm, v = jump_probs(F, 6)
    r = rng.random(N)
    dx = np.where(r < pp, 1, np.where(r < pp + pm, -1, 0))
    tr = rng.integers(0, 4, N)
    dy = np.where(dx != 0, 0, np.where(tr == 0, 1, np.where(tr == 1, -1, 0)))
    dz = np.where(dx != 0, 0, np.where(tr == 2, 1, np.where(tr == 3, -1, 0)))
    x = np.cumsum(dx).astype(np.int64)
    y = np.cumsum(dy) % w
    z = np.cumsum(dz) % w
    key = x * (w * w) + y * w + z
    return len(np.unique(key)) / N, v


def walk_hex(w, F, N, rng):
    """Plain biased walk along the stacking axis of a simple hexagonal channel, with
    reflecting walls: a jump leaving the patch is rejected and the walker stays put,
    the step still counting."""
    pp, pm, v = jump_probs(F, 8)
    pts, idx = hex_patch(w)
    Om = len(pts)
    dpq = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    r = rng.random(N)
    dz = np.where(r < pp, 1, np.where(r < pp + pm, -1, 0))
    tr = rng.integers(0, 6, N)
    z = np.cumsum(dz).astype(np.int64)

    # the transverse walk has to be stepped, because of the rejections
    p = q = 0
    cell = np.empty(N, dtype=np.int64)
    trl = tr.tolist()
    dzl = dz.tolist()
    for n in range(N):
        if dzl[n] == 0:
            a, b = dpq[trl[n]]
            np_, nq_ = p + a, q + b
            if abs(np_) <= w and abs(nq_) <= w and abs(np_ + nq_) <= w:
                p, q = np_, nq_
        cell[n] = idx[(p, q)]
    key = z * Om + cell
    return len(np.unique(key)) / N, v


# -------------------------------------------------------------------------------------
def sweep(lat, w, F, Ns, reps, rng):
    q = 6 if lat == "cubic" else 8
    D0 = 1.0 / q
    Om = w * w if lat == "cubic" else 3 * w * w + 3 * w + 1
    print("\n  %s  w = %d   Omega = %d   F = %.3e" % (lat, w, Om, F))
    print("        N        rho     (D/N)/(Omega v)")
    for N in Ns:
        vals = []
        for _ in range(reps):
            dn, v = (walk_cubic(w, F, N, rng) if lat == "cubic"
                     else walk_hex(w, F, N, rng))
            vals.append(dn / (Om * v))
        _, _, v = jump_probs(F, q)
        rho = N * v * v / D0
        m, s = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
        print("   %.1e   %8.1f      %.4f +/- %.4f" % (N, rho, m, s))


def window(t, alpha, Om, q, rho_min, eps_max, A=1.0):
    """Force window satisfying both conditions, and the smallest t that leaves one."""
    D0, G2 = 1.0 / q, Gamma(1.0 + alpha) ** 2
    F_lo = ((rho_min * A * G2 * D0 * Om ** (1 - alpha) / t ** alpha)
            ** (1.0 / (1.0 + alpha))) * q
    F_hi = (eps_max / Om) * q
    t_min = (rho_min * A * G2 * D0 * Om ** 2 / eps_max ** (1 + alpha)) ** (1.0 / alpha)
    return F_lo, F_hi, t_min


# -------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--t", type=float, default=1e24)
    ap.add_argument("--rho-min", type=float, default=150.0)
    ap.add_argument("--eps-max", type=float, default=0.06)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--nmax", type=float, default=3e6,
                    help="largest walk length in the sweep; raise for a cleaner curve")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20250831)

    channels = [("hex", 1), ("cubic", 4), ("cubic", 6), ("hex", 4)]

    if not args.skip_sweep:
        print("=" * 72)
        print(" CONVERGENCE OF D/N TO Omega v   (plain biased walk, no traps)")
        print("=" * 72)
        Ns = [int(args.nmax / 30), int(args.nmax / 10), int(args.nmax / 3), int(args.nmax)]
        # the hexagonal sweep is stepped in Python, so keep it short
        sweep("cubic", 4, 5.0e-3, Ns, args.reps, rng)
        sweep("hex", 1, 5.0e-3, [n // 20 for n in Ns], max(2, args.reps // 2), rng)

    print("\n" + "=" * 72)
    print(" USABLE FORCE WINDOW   (rho >= %.0f,  eps <= %.3f,  alpha = %.2f)"
          % (args.rho_min, args.eps_max, args.alpha))
    print("=" * 72)
    print("  lattice   w  Omega      F_lo       F_hi    decades    t needed")
    lo_all, hi_all = 0.0, np.inf
    for lat, w in channels:
        q = 6 if lat == "cubic" else 8
        Om = w * w if lat == "cubic" else 3 * w * w + 3 * w + 1
        lo, hi, tmin = window(args.t, args.alpha, Om, q, args.rho_min, args.eps_max)
        lo_all, hi_all = max(lo_all, lo), min(hi_all, hi)
        d = np.log10(hi / lo) if hi > lo else float("nan")
        print("  %-7s %3d %6d  %9.2e  %9.2e   %6.2f    %.1e"
              % (lat, w, Om, lo, hi, d, tmin))
    if hi_all > lo_all:
        print("\n  common window: F in [%.3e, %.3e]  (%.2f decades)"
              % (lo_all, hi_all, np.log10(hi_all / lo_all)))
    else:
        print("\n  ** no common window at t = %.1e. Raise t, or drop the widest channel."
              % args.t)
    print("\n  Omega_max ~ t^(alpha/2), so doubling the widest channel costs a factor of"
          "\n  %.0f in measurement time." % (2 ** (2.0 / args.alpha)))


if __name__ == "__main__":
    main()