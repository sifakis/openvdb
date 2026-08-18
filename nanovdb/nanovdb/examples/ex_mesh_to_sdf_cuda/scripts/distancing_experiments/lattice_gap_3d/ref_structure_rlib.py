"""Referee re-implementation, from scratch. 3D lattice certification gap.

Conventions (from CONTEXT.md + CONTEXT_3D.md):
  target lattice point at origin, depth eps>0, unit normal n into exterior.
  witness at integer offset w, depth d0_actual = eps + w.n
  pairwise rule: eps + d0_actual > |w|   <=>   eps > g(w,n) := (|w| - w.n)/2
  threshold depth d0_thr(w,n) = (|w| + w.n)/2  (= depth at eps = g)
  barrier: UDF <= r_d = sqrt(3)/2 ; in band: UDF <= W

  THRESHOLD rule: w admissible iff r_d < d0_thr <= W ; F_thr(n) = min over admissible of g
  EXACT rule: J_w = ( max(g, r_d - w.n), min(r_d, W - w.n) ];
              F_exa(n) = sup( (0, r_d] \\ union J_w )   (0 if empty)
Both capped at r_d (deeper targets are non-barrier -> certified by CC).
"""
import math
from fractions import Fraction

RD = math.sqrt(3.0) / 2.0


def enum_w(rmax):
    """All nonzero integer vectors with |w| <= rmax."""
    R = int(math.floor(rmax + 1e-12))
    r2 = rmax * rmax + 1e-12
    out = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            for c in range(-R, R + 1):
                if a == b == c == 0:
                    continue
                s = a * a + b * b + c * c
                if s <= r2:
                    out.append((a, b, c, math.sqrt(s)))
    return out


def gcd3(a, b, c):
    from math import gcd
    return gcd(gcd(abs(a), abs(b)), abs(c))


def primitives(ws):
    return [w for w in ws if gcd3(w[0], w[1], w[2]) == 1]


def g_of(w, n):
    return (w[3] - (w[0] * n[0] + w[1] * n[1] + w[2] * n[2])) / 2.0


def F_thr(n, ws, W, rd=RD, use_barrier=True, band_strict=False, barrier_strict=True):
    """min g over threshold-admissible w, capped at rd."""
    best = rd
    nx, ny, nz = n
    for (a, b, c, L) in ws:
        d = a * nx + b * ny + c * nz
        d0 = (L + d) / 2.0
        if use_barrier:
            ok = (d0 > rd) if barrier_strict else (d0 >= rd)
            if not ok:
                continue
        if band_strict:
            if not (d0 < W):
                continue
        else:
            if not (d0 <= W):
                continue
        gg = (L - d) / 2.0
        if gg < best:
            best = gg
    return best


def windows(n, ws, W, rd=RD):
    """List of (lo, hi] usable windows intersected with (0, rd]."""
    nx, ny, nz = n
    out = []
    for (a, b, c, L) in ws:
        d = a * nx + b * ny + c * nz
        lo = (L - d) / 2.0            # g
        blo = rd - d                  # barrier lower bound
        if blo > lo:
            lo = blo
        hi = W - d
        if hi > rd:
            hi = rd
        if hi > lo and hi > 0.0:
            if lo < 0.0:
                lo = 0.0
            out.append((lo, hi))
    return out


def F_exa(n, ws, W, rd=RD):
    """sup of (0,rd] minus union of windows."""
    iv = windows(n, ws, W, rd)
    if not iv:
        return rd
    iv.sort(key=lambda t: t[0])
    # sweep from the top: find largest x in (0,rd] not covered.
    # merge intervals (they are (lo,hi], half-open; union of such)
    merged = []
    for lo, hi in iv:
        if merged and lo <= merged[-1][1]:
            # (a,b] u (lo,hi] with lo<=b -> (a, max(b,hi)]
            if hi > merged[-1][1]:
                merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    # now merged is a disjoint increasing list of (lo,hi]
    # uncovered sup: walk from top
    top = rd
    for lo, hi in reversed(merged):
        if hi >= top:
            top = lo          # everything in (lo, top] covered; sup of uncovered <= lo
            if top <= 0.0:
                return 0.0
        elif hi < top:
            return top        # gap (hi, top] uncovered, sup = top
    return top


# ---------- exact-ish high precision helpers ----------
def isqrt_dec(x, prec=60):
    from decimal import Decimal, getcontext
    getcontext().prec = prec
    return Decimal(x).sqrt()
