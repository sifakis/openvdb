#!/usr/bin/env python3
"""Exact minimal polynomials of eps* and W_hi.

Work in K = Q(sqrt5, q, sqrt2) with q = sqrt(2 sqrt5 - 4).
Degree argument:
  * [Q(sqrt5):Q] = 2.
  * 2 sqrt5-4 is not a square in Q(sqrt5): its norm (2 sqrt5-4)(-2 sqrt5-4) = -4 < 0,
    while the norm of a square is a square in Q (hence >= 0).  So [Q(sqrt5,q):Q] = 4.
  * Q(sqrt5,q) is NOT totally real: under sqrt5 -> -sqrt5, q^2 = 2 sqrt5-4 goes to
    -2 sqrt5-4 < 0.  Q(sqrt2,sqrt5) IS totally real (and degree 4), so
    sqrt2 not in Q(sqrt5,q); hence [K:Q] = 8 and
      {1, r5, q, r5 q, r2, r10, r2 q, r10 q}
    is a Q-basis of K.  Linear algebra on powers therefore yields *the* minimal
    polynomial.
"""
from fractions import Fraction as Fr
from decimal import Decimal as D, getcontext
import itertools

getcontext().prec = 70

# monomial key = (i,j,k) meaning r5^i q^j r2^k, i,j,k in {0,1}
KEYS = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]


def mul_mono(m1, m2):
    i, j, k = m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]
    coef = Fr(1)
    if i == 2:
        coef *= 5
        i = 0
    if k == 2:
        coef *= 2
        k = 0
    if j == 2:                       # q^2 = 2 r5 - 4
        j = 0
        out = []
        i2, c2 = i + 1, coef * 2
        if i2 == 2:
            c2 *= 5
            i2 = 0
        out.append(((i2, j, k), c2))
        out.append(((i, j, k), -4 * coef))
        return out
    return [((i, j, k), coef)]


def mul(x, y):
    r = {}
    for m1, c1 in x.items():
        for m2, c2 in y.items():
            for m, c in mul_mono(m1, m2):
                r[m] = r.get(m, Fr(0)) + c1 * c2 * c
    return {m: c for m, c in r.items() if c != 0}


def add(x, y):
    r = dict(x)
    for m, c in y.items():
        r[m] = r.get(m, Fr(0)) + c
    return {m: c for m, c in r.items() if c != 0}


def smul(s, x):
    return {m: Fr(s) * c for m, c in x.items() if Fr(s) * c != 0}


ONE = {(0, 0, 0): Fr(1)}
R5 = {(1, 0, 0): Fr(1)}
Q = {(0, 1, 0): Fr(1)}
R2 = {(0, 0, 1): Fr(1)}
R10 = {(1, 0, 1): Fr(1)}

r5d, r2d, r10d = D(5).sqrt(), D(2).sqrt(), D(10).sqrt()
qd = (2 * r5d - 4).sqrt()
NUM = {(0, 0, 0): D(1), (1, 0, 0): r5d, (0, 1, 0): qd, (1, 1, 0): r5d * qd,
       (0, 0, 1): r2d, (1, 0, 1): r10d, (0, 1, 1): r2d * qd, (1, 1, 1): r10d * qd}


def val(x):
    return sum(D(c.numerator) / D(c.denominator) * NUM[m] for m, c in x.items())


def minpoly(x, maxdeg=8):
    pows = [ONE]
    for _ in range(maxdeg):
        pows.append(mul(pows[-1], x))
    for deg in range(1, maxdeg + 1):
        # solve sum_{i=0..deg} c_i x^i = 0 with c_deg = 1
        rows = []   # 8 equations (one per basis monomial), unknowns c_0..c_{deg-1}
        for m in KEYS:
            rows.append([pows[i].get(m, Fr(0)) for i in range(deg)]
                        + [-pows[deg].get(m, Fr(0))])
        sol = solve(rows, deg)
        if sol is not None:
            return [Fr(1)] + [sol[deg - 1 - i] for i in range(deg)][::-1][::-1] \
                if False else ([Fr(1)] + list(reversed(sol)))[::-1]
    return None


def solve(rows, n):
    """least-structure exact Gaussian elimination; returns solution list or None"""
    M = [r[:] for r in rows]
    piv = []
    r = 0
    for c in range(n):
        p = None
        for i in range(r, len(M)):
            if M[i][c] != 0:
                p = i
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        f = M[r][c]
        M[r] = [v / f for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                g = M[i][c]
                M[i] = [a - g * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    # consistency + uniqueness
    for i in range(r, len(M)):
        if M[i][n] != 0:
            return None
    if len(piv) < n:
        return None
    sol = [Fr(0)] * n
    for i, c in enumerate(piv):
        sol[c] = M[i][n]
    return sol


def show(name, x, poly):
    # poly given low->high after normalisation below
    den = 1
    for c in poly:
        den = den * c.denominator // __import__('math').gcd(den, c.denominator)
    ipoly = [int(c * den) for c in poly]
    g = 0
    for c in ipoly:
        g = __import__('math').gcd(g, abs(c))
    ipoly = [c // g for c in ipoly]
    if ipoly[-1] < 0:
        ipoly = [-c for c in ipoly]
    s = " + ".join(f"{c}x^{i}" for i, c in enumerate(ipoly) if c) \
        .replace("x^0", "").replace("x^1 ", "x ").replace("+ -", "- ")
    print(f"  {name}: degree {len(ipoly)-1}")
    print(f"    {s} = 0")
    print(f"    integer coefficients (low->high): {ipoly}")
    v = val(x)
    acc = D(0)
    for c in reversed(ipoly):
        acc = acc * v + D(c)
    print(f"    value = {v:.40f}")
    print(f"    P(value) = {acc:.3e}  (Decimal prec 70)")
    return ipoly


print("=" * 78)
print("sanity: the ring arithmetic reproduces the numeric values")
print("=" * 78)
import random
random.seed(7)
worst = D(0)
for _ in range(200):
    x = {m: Fr(random.randint(-5, 5), random.randint(1, 4)) for m in KEYS}
    y = {m: Fr(random.randint(-5, 5), random.randint(1, 4)) for m in KEYS}
    worst = max(worst, abs(val(mul(x, y)) - val(x) * val(y)))
print(f"  max |val(x*y) - val(x)val(y)| over 200 random pairs: {worst:.3e}")

print("=" * 78)
print("eps* = (3 - sqrt5 - q)/4")
print("=" * 78)
EPS = smul(Fr(1, 4), add(add(smul(3, ONE), smul(-1, R5)), smul(-1, Q)))
p = minpoly(EPS)
show("min poly of eps*", EPS, p)

print("=" * 78)
print("W_hi = sqrt10/2 + (sqrt5 - 1) + q/2")
print("=" * 78)
WHI = add(add(smul(Fr(1, 2), R10), add(R5, smul(-1, ONE))), smul(Fr(1, 2), Q))
p2 = minpoly(WHI)
ip2 = show("min poly of W_hi", WHI, p2)

print("=" * 78)
print("W_lo = sqrt5  (degree 2, trivially)")
print("=" * 78)
p3 = minpoly(R5)
show("min poly of W_lo", R5, p3)
