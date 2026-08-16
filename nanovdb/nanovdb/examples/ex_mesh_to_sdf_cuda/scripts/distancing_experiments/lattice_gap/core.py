#!/usr/bin/env python3
"""Independent re-implementation of the 2D lattice certification-gap problem.

Definitions (from CONTEXT.md, re-derived here):
  n = (cos a, sin a) unit normal.  w integer vector != 0.
  g(w,a)  = (|w| - w.n)/2       cost: v1 certified by w iff eps > g
  d0(w,a) = (|w| + w.n)/2       witness depth at that threshold
  admissible iff  BAR < d0 <= W,  BAR = sqrt(2)/2
  eps*(a,W) = min over admissible w of g(w,a)
  eps*(W)   = sup over a of eps*(a,W)
"""
import math

BAR = math.sqrt(2.0) / 2.0


def cand_set(Rmax):
    """All primitive+nonprimitive integer w != 0 with |w| <= Rmax."""
    R = int(math.floor(Rmax + 1e-12))
    out = []
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            if x == 0 and y == 0:
                continue
            L = math.hypot(x, y)
            if L <= Rmax + 1e-12:
                out.append((x, y, L))
    return out


def cand_primitive(Rmax):
    """Primitive only (gcd=1). g(k*w) = k*g(w) so non-primitive never wins the min,
    BUT its admissibility differs, so keep both unless proven.  Provided for tests."""
    out = []
    for (x, y, L) in cand_set(Rmax):
        if math.gcd(abs(x), abs(y)) == 1:
            out.append((x, y, L))
    return out


def eps_dir(a, W, C, bar=BAR, ret_all=False):
    """min-cost admissible witness at angle a.  Returns (g, w, d0, |w|) or None."""
    ca, sa = math.cos(a), math.sin(a)
    best = None
    rows = []
    for (x, y, L) in C:
        wn = x * ca + y * sa
        g = 0.5 * (L - wn)
        d0 = 0.5 * (L + wn)
        ok = (d0 > bar) and (d0 <= W)
        if ret_all:
            rows.append((g, (x, y), d0, L, ok))
        if not ok:
            continue
        if best is None or g < best[0]:
            best = (g, (x, y), d0, L)
    if ret_all:
        rows.sort()
        return best, rows
    return best


def eps_val(a, W, C, bar=BAR):
    r = eps_dir(a, W, C, bar)
    return r[0] if r is not None else float('inf')
