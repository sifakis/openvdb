#!/usr/bin/env python3
"""Exact closed form for a circular Sigma -- no expansion, no eps-scanning.

Derivation.  Target lattice point v1 at depth eps, witness at offset w, q = w.n,
|w|^2 = q^2 + p^2 with p the tangential offset.  Planar quantities:
    g    = (|w| - q)/2      certification cost
    d0p  = (|w| + q)/2      witness depth at its own threshold      (g + d0p = |w|)
Note the identity  g * d0p = (|w|^2 - q^2)/4 = p^2/4.

Circle of radius R, exterior = inside the disk (CONCAVE).  Centre O = (R-eps)*n,
witness depth d0 = R - |w - O|.  Setting eps + d0 = |w| and solving exactly:

    eps_thr = R(|w| - q) / (2R - |w| - q) = g / (1 - d0p/R)

Exterior = outside the disk (CONVEX): O = -(R+eps)*n, d0 = |w - O| - R, giving

    eps_thr = R(|w| - q) / (2R + |w| + q) = g / (1 + d0p/R)

Both reduce to g as R -> infinity, and to first order give g + g*d0p/R = g + p^2/(4R).
"""
import math

W, RD = 3.0, math.sqrt(2) / 2
EPS_C = 0.019202630763796344
K_FIRST_ORDER = 0.028315226153961
CAND = [(a, b) for a in range(-7, 8) for b in range(-7, 8)
        if (a, b) != (0, 0) and math.hypot(a, b) <= 7.5]


def thr(w, n, R, concave):
    """Exact depth at which witness w starts certifying; None if inadmissible."""
    wl = math.hypot(*w)
    q = w[0] * n[0] + w[1] * n[1]
    g, d0p = (wl - q) / 2, (wl + q) / 2
    if R is None:
        e = g
    elif concave:
        if d0p >= R:                      # witness cannot lie inside the disk
            return None
        e = g / (1.0 - d0p / R)
    else:
        e = g / (1.0 + d0p / R)
    d0 = wl - e                           # depth at threshold, since eps + d0 = |w|
    if not (RD < d0 <= W):
        return None
    return e


def eps_star_dir(alpha, R, concave):
    n = (math.cos(alpha), math.sin(alpha))
    vals = [t for t in (thr(w, n, R, concave) for w in CAND) if t is not None]
    return min(vals) if vals else None


def eps_star(R, concave, coarse=4000):
    best = (-1.0, 0.0)
    for i in range(coarse + 1):
        a = (math.pi / 2) * i / coarse
        v = eps_star_dir(a, R, concave)
        if v is not None and v > best[0]:
            best = (v, a)
    a0, h = best[1], (math.pi / 2) / coarse
    for _ in range(120):
        c = [(eps_star_dir(x, R, concave), x) for x in (a0 - h, a0, a0 + h)]
        c = [(v, x) for v, x in c if v is not None]
        if c:
            best = max(c)
            a0 = best[1]
        h *= 0.62
    return best


print("identity check  g*d0p == p^2/4:")
n = (math.cos(0.278), math.sin(0.278))
for w in [(1, 0), (2, 1), (3, 1)]:
    wl = math.hypot(*w); q = w[0] * n[0] + w[1] * n[1]
    g, d0p = (wl - q) / 2, (wl + q) / 2
    print(f"  w={str(w):>7}  g*d0p = {g*d0p:.15f}   p^2/4 = {(wl*wl-q*q)/4:.15f}"
          f"   equal: {abs(g*d0p-(wl*wl-q*q)/4) < 1e-15}")

vp, ap = eps_star(None, True)
print(f"\nplanar control: {vp:.12f} at {math.degrees(ap):.6f} deg  "
      f"(eps_c = {EPS_C:.12f}, ratio {vp/EPS_C:.6f})\n")

print(f"{'R':>9} {'concave EXACT':>15} {'eps_c+K/R':>13} {'exact/1st':>10} "
      f"{'convex EXACT':>14} {'eps_c-K/R':>13} {'alpha*(deg)':>12}")
for R in [1e6, 1000, 200, 100, 50, 21, 12, 8, 5, 4, 3, 2.5, 2.0, 1.8, 1.6]:
    vc, ac = eps_star(R, True)
    vx, _ = eps_star(R, False)
    pc, px = EPS_C + K_FIRST_ORDER / R, EPS_C - K_FIRST_ORDER / R
    sc = f"{vc:15.9f}" if vc > 0 else f"{'NO WITNESS':>15}"
    print(f"{R:9.1f} {sc} {pc:13.9f} {vc/pc:10.4f} {vx:14.9f} {px:13.9f} "
          f"{math.degrees(ac):12.4f}")
