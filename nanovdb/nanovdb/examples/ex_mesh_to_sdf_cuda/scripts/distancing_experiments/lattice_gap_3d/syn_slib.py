"""Synthesist's independent re-verification library.  Pure Python 3, no numpy.

Conventions (dimension-free):
    g(w,n)   = (|w| - w.n)/2
    d0(w,n)  = (|w| + w.n)/2          (witness depth at its own threshold)
    r_d      = sqrt(3)/2  in 3D
Threshold rule : w admissible iff r_d < d0 <= W ;  F_thr(n) = min g over admissible w.
Exact rule     : J_w = ( max(g, r_d - w.n), min(r_d, W - w.n) ] ;
                 F_exa(n) = sup( (0, r_d] \\ union J_w ).
"""
import math, heapq
from itertools import product

RD = math.sqrt(3.0) / 2.0
W_DEFAULT = 3.0


def witnesses(n2max):
    """All nonzero integer vectors with |w|^2 <= n2max, with |w| precomputed."""
    m = int(math.isqrt(n2max))
    out = []
    for a in range(-m, m + 1):
        for b in range(-m, m + 1):
            for c in range(-m, m + 1):
                s = a * a + b * b + c * c
                if 0 < s <= n2max:
                    out.append((a, b, c, math.sqrt(s)))
    return out


def primitive(w):
    return math.gcd(math.gcd(abs(w[0]), abs(w[1])), abs(w[2])) == 1


def F_thr(n, ws, W=W_DEFAULT, rd=RD, strict_band=False):
    best = rd
    nx, ny, nz = n
    for (a, b, c, L) in ws:
        d = a * nx + b * ny + c * nz
        d0 = (L + d) / 2.0
        if d0 <= rd:
            continue
        if (d0 >= W) if strict_band else (d0 > W):
            continue
        g = (L - d) / 2.0
        if g < best:
            best = g
    return best


def windows(n, ws, W=W_DEFAULT, rd=RD):
    nx, ny, nz = n
    iv = []
    for (a, b, c, L) in ws:
        d = a * nx + b * ny + c * nz
        g = (L - d) / 2.0
        lo = g if g > rd - d else rd - d
        hi = rd if rd < W - d else W - d
        if hi > lo:
            iv.append((lo, hi))
    return iv


def uncovered_sup(iv, rd=RD):
    """sup of (0, rd] minus the union of half-open intervals (lo,hi]."""
    cur = rd
    changed = True
    while changed:
        changed = False
        for (lo, hi) in iv:
            if lo < cur <= hi:
                cur = lo
                changed = True
    return cur if cur > 0 else 0.0


def F_exa(n, ws, W=W_DEFAULT, rd=RD):
    return uncovered_sup(windows(n, ws, W, rd), rd)


# ---------------------------------------------------------------- branch & bound

def _norm(v):
    s = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / s, v[1] / s, v[2] / s)


def _plane_dist(v1, v2, v3):
    """distance from origin to the plane through three unit vectors (lower bound on |u|)."""
    ax = v2[0] - v1[0]; ay = v2[1] - v1[1]; az = v2[2] - v1[2]
    bx = v3[0] - v1[0]; by = v3[1] - v1[1]; bz = v3[2] - v1[2]
    cx = ay * bz - az * by; cy = az * bx - ax * bz; cz = ax * by - ay * bx
    nn = math.sqrt(cx * cx + cy * cy + cz * cz)
    if nn == 0.0:
        return 0.0
    return abs(cx * v1[0] + cy * v1[1] + cz * v1[2]) / nn


def cell_bound_thr(tri, ws, W, rd, strict_band=False):
    v1, v2, v3 = tri
    h = _plane_dist(v1, v2, v3)
    if h <= 0.0:
        return rd
    best = rd
    for (a, b, c, L) in ws:
        d1 = a * v1[0] + b * v1[1] + c * v1[2]
        d2 = a * v2[0] + b * v2[1] + c * v2[2]
        d3 = a * v3[0] + b * v3[1] + c * v3[2]
        lo = min(d1, d2, d3); hi = max(d1, d2, d3)
        if lo >= 0.0:
            dmin, dmax = lo, hi / h
        elif hi <= 0.0:
            dmin, dmax = lo / h, hi
        else:
            dmin, dmax = lo / h, hi / h
        d0lo = (L + dmin) / 2.0
        d0hi = (L + dmax) / 2.0
        if d0lo <= rd:
            continue                      # not guaranteed non-barrier
        if (d0hi >= W) if strict_band else (d0hi > W):
            continue                      # not guaranteed in band
        ghi = (L - dmin) / 2.0            # sup of g over the cell
        if ghi < best:
            best = ghi
    return best


def cell_bound_exa(tri, ws, W, rd):
    v1, v2, v3 = tri
    h = _plane_dist(v1, v2, v3)
    if h <= 0.0:
        return rd
    iv = []
    for (a, b, c, L) in ws:
        d1 = a * v1[0] + b * v1[1] + c * v1[2]
        d2 = a * v2[0] + b * v2[1] + c * v2[2]
        d3 = a * v3[0] + b * v3[1] + c * v3[2]
        lo = min(d1, d2, d3); hi = max(d1, d2, d3)
        if lo >= 0.0:
            dmin, dmax = lo, hi / h
        elif hi <= 0.0:
            dmin, dmax = lo / h, hi
        else:
            dmin, dmax = lo / h, hi / h
        ghi = (L - dmin) / 2.0
        wlo = max(ghi, rd - dmin)          # guaranteed window bottom
        whi = min(rd, W - dmax)            # guaranteed window top
        if whi > wlo:
            iv.append((wlo, whi))
    return uncovered_sup(iv, rd)


FUND = (_norm((1.0, 0.0, 0.0)), _norm((1.0, 1.0, 0.0)), _norm((1.0, 1.0, 1.0)))


def bnb(rule, ws, W=W_DEFAULT, rd=RD, tol=1e-12, maxcells=400000, strict_band=False):
    """Certified bracket [lo, up] for max over the fundamental spherical triangle."""
    bound = (lambda t: cell_bound_thr(t, ws, W, rd, strict_band)) if rule == 'thr' \
        else (lambda t: cell_bound_exa(t, ws, W, rd))
    val = (lambda n: F_thr(n, ws, W, rd, strict_band)) if rule == 'thr' \
        else (lambda n: F_exa(n, ws, W, rd))
    v1, v2, v3 = FUND
    cen = _norm(((v1[0] + v2[0] + v3[0]) / 3, (v1[1] + v2[1] + v3[1]) / 3,
                 (v1[2] + v2[2] + v3[2]) / 3))
    lo = val(cen)
    for v in (v1, v2, v3):
        lo = max(lo, val(v))
    heap = [(-bound(FUND), 0, FUND)]
    cnt = 0
    ncells = 1
    while heap:
        negU, _, tri = heapq.heappop(heap)
        U = -negU
        if U <= lo + tol:
            return lo, max(U, lo + tol), ncells
        if ncells > maxcells:
            return lo, max(U, lo + tol), ncells
        a, b, c = tri
        m1 = _norm(((b[0] + c[0]) / 2, (b[1] + c[1]) / 2, (b[2] + c[2]) / 2))
        m2 = _norm(((a[0] + c[0]) / 2, (a[1] + c[1]) / 2, (a[2] + c[2]) / 2))
        m3 = _norm(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2))
        for t in ((a, m3, m2), (m3, b, m1), (m2, m1, c), (m1, m2, m3)):
            ncells += 1
            cc = _norm(((t[0][0] + t[1][0] + t[2][0]) / 3, (t[0][1] + t[1][1] + t[2][1]) / 3,
                        (t[0][2] + t[1][2] + t[2][2]) / 3))
            v = val(cc)
            if v > lo:
                lo = v
            u = bound(t)
            if u > lo + tol:
                cnt += 1
                heapq.heappush(heap, (-u, cnt, t))
    return lo, lo + tol, ncells
