#!/usr/bin/env python3
"""Shared machinery for the 2D lattice certification-gap problem.

cost of witness w at unit normal n:   g(w,n) = (|w| - w.n)/2 = |w| sin^2(theta/2)
depth of w at its own threshold:      d0(w,n) = (|w| + w.n)/2
admissible:                           sqrt(2)/2 < d0 <= W
eps*(n,W) = min over admissible w of g ;  eps*(W) = sup over n of eps*(n,W)
"""
import math

BAR = math.sqrt(2) / 2.0


def candidates(W, slack=0.75):
    """All lattice vectors that can possibly be the minimiser.

    d0 = |w| - g <= W  and we only care about w whose g is below the running
    best (always < slack here), hence |w| < W + slack.
    """
    R = int(math.floor(W + slack)) + 1
    out = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if (a, b) == (0, 0):
                continue
            L = math.hypot(a, b)
            if L <= W + slack:
                out.append((a, b, L))
    return out


def cost(w, alpha):
    a, b, L = w
    return (L - (a * math.cos(alpha) + b * math.sin(alpha))) / 2.0


def depth(w, alpha):
    a, b, L = w
    return (L + (a * math.cos(alpha) + b * math.sin(alpha))) / 2.0


def F(alpha, W, C):
    """min over admissible w of g(w,n).  Returns (cost, w) or (inf, None)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    best = (float('inf'), None)
    for (a, b, L) in C:
        wn = a * ca + b * sa
        d0 = (L + wn) / 2.0
        if d0 <= BAR or d0 > W:
            continue
        c = (L - wn) / 2.0
        if c < best[0]:
            best = (c, (a, b))
    return best


def _solve_lin(a, b, c):
    """all alpha in [0, pi/2] with a cos(alpha) + b sin(alpha) = c."""
    R = math.hypot(a, b)
    if R == 0:
        return []
    if abs(c) > R:
        return []
    psi = math.atan2(b, a)             # a cos + b sin = R cos(alpha - psi)
    d = math.acos(max(-1.0, min(1.0, c / R)))
    out = []
    for al in (psi + d, psi - d):
        for k in (-2, -1, 0, 1, 2):
            x = al + 2 * math.pi * k
            if -1e-12 <= x <= math.pi / 2 + 1e-12:
                out.append(min(max(x, 0.0), math.pi / 2))
    return out


def events(W, C):
    """Candidate alphas: pairwise cost crossovers, band edges, barrier edges."""
    ev = [0.0, math.pi / 2]
    n = len(C)
    for i in range(n):
        a1, b1, L1 = C[i]
        # band edge  w.n = 2W - |w| ; barrier edge  w.n = sqrt(2) - |w|
        ev += _solve_lin(a1, b1, 2 * W - L1)
        ev += _solve_lin(a1, b1, math.sqrt(2) - L1)
        for j in range(i + 1, n):
            a2, b2, L2 = C[j]
            ev += _solve_lin(a2 - a1, b2 - b1, L2 - L1)
    return sorted(set(ev))


def sup_eps(W, C=None, deltas=(0.0, 1e-11, 1e-8, 1e-6)):
    """sup over alpha in [0,pi/2] of F(.,W).  Event-based, so slivers are found."""
    if C is None:
        C = candidates(W)
    best = (-1.0, None, None)
    for e in events(W, C):
        for d in deltas:
            for al in ({e + d, e - d} if d else {e}):
                if al < 0 or al > math.pi / 2:
                    continue
                c, w = F(al, W, C)
                if c != float('inf') and c > best[0]:
                    best = (c, al, w)
    return best


def scan_eps(W, C=None, N=200000):
    """dense-grid cross check (will miss slivers narrower than the grid)."""
    if C is None:
        C = candidates(W)
    best = (-1.0, None, None)
    for i in range(N + 1):
        al = math.pi / 2 * i / N
        c, w = F(al, W, C)
        if c != float('inf') and c > best[0]:
            best = (c, al, w)
    return best
