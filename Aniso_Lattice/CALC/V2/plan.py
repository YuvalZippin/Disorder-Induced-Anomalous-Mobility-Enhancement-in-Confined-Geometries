#!/usr/bin/env python3
"""
plan.py -- pick the cheapest operating point that is still inside the
regime where Eq. (33) and Eq. (34) of the paper actually hold.

Two constraints must hold simultaneously:
  (A) near-recurrent :  eps = Omega * v_par / a  <=  eps_max      (paper's Omega v/a << 1)
  (B) drift-dominated:  <N> >= S * N*,  N* = 2 D_par / v_par^2    (measured: S ~ 10)

Given (A) as an equality, F is fixed, and then
      N*        = 2 w^2 D_par / (eps^2 a^2)
      cost/walker = S * N*                      <-- INDEPENDENT OF T
      T         = [ A G^2 eps^(1-a) * S * N* ]^(1/a),  G = Gamma(1+alpha)

So T is free; what costs you is large w and small eps, both quadratically.

Usage:  python3 plan.py --a 1 --b 0.5,1,2,4 --w 5 --alpha 0.3 --eps 0.03
"""
import argparse, math

def parse(s): return [float(t) for t in s.split(',') if t]

def transport(a, b, w, F, model):
    """Exact v_par, D_par, D_perp from the jump weights -- same code path as the C++."""
    e = math.exp(F * a / 2.0)
    if model == 'iso':
        wxp, wxm, wy = e, 1.0/e, 1.0
    else:                                    # rate ~ 1/d^2
        wxp, wxm, wy = e/(a*a), 1.0/(e*a*a), 1.0/(b*b)
    Z = wxp + wxm + 2.0*wy
    pxp, pxm, py = wxp/Z, wxm/Z, wy/Z
    return a*(pxp-pxm), 0.5*a*a*(pxp+pxm), b*b*py

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', default='1');      p.add_argument('--b', default='1')
    p.add_argument('--w', default='5');      p.add_argument('--alpha', type=float, default=0.3)
    p.add_argument('--amp', type=float, default=1.0)
    p.add_argument('--eps', type=float, default=0.03, help='target escape probability (<<1)')
    p.add_argument('--safety', type=float, default=10.0, help='required <N>/N*')
    p.add_argument('--model', default='invd2', choices=['iso','invd2'])
    p.add_argument('--ntraj', type=int, default=100000)
    p.add_argument('--nsps', type=float, default=6e7, help='steps/sec/core (measure yours)')
    p.add_argument('--cores', type=int, default=64)
    p.add_argument('--F', type=float, default=None,
                   help='FIXED force (required for a b-scan: holding eps fixed instead '
                        'holds D0*F fixed and the b-dependence cancels by construction)')
    args = p.parse_args()

    al, A = args.alpha, args.amp
    G = math.gamma(1.0 + al)

    print(f"# model={args.model} alpha={al} A={A} eps_target={args.eps} safety={args.safety}")
    print(f"# {'a':>5} {'b':>5} {'w':>4} | {'F':>10} {'T':>10} {'<N>':>10} {'D_par':>9} "
          f"{'v_par':>10} | {'core-sec':>9}")

    total = 0.0; Tneed = 0.0; rows = []
    for a in parse(args.a):
        for b in parse(args.b):
            for w in parse(args.w):
                w = int(round(w))
                # D_par at zero force (F only shifts it at O(F^2))
                _, D, _ = transport(a, b, w, 0.0, args.model)
                F = args.F if args.F is not None else args.eps * a / (w * D)
                v, D, Dp = transport(a, b, w, F, args.model)
                eps  = w * v / a
                Nst  = 2.0 * D / (v*v)
                Nreq = args.safety * Nst
                T    = (A * G*G * eps**(1.0-al) * Nreq) ** (1.0/al)
                sec  = args.ntraj * Nreq / args.nsps
                total += sec; Tneed = max(Tneed, T)
                flag = '  << eps too big' if eps > args.eps*1.001 else ''
                print(f"  {a:>5.3g} {b:>5.3g} {w:>4d} | {F:>10.4g} {T:>10.3e} {Nreq:>10.2e} "
                      f"{D:>9.4f} {v:>10.4g} | {sec:>9.1f}   eps={eps:.4f}{flag}")

    if args.F is not None:
        print(f"\n# use ONE common T = {Tneed:.3e} for every point (set by the slowest one)")
    print(f"# total {total:.0f} core-sec  ->  {total/args.cores/60:.1f} min on {args.cores} cores")
    print(f"# feed the F and T columns straight to qtm2 (--T accepts a comma list;")
    print(f"#  adding smaller T values is free and gives you the convergence check).")

if __name__ == '__main__':
    main()