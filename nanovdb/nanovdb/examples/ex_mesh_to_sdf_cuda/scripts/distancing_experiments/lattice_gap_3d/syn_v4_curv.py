"""Raw-geometry check of the concave-sphere curvature values.  No use of g, c, or any series.

Exterior = interior of a ball of radius R centred at C = origin + (R - eps) n.
UDF(x) = | R - |x - C| |.  Witness usable iff it is exterior (|x-C| < R), non-barrier,
in band, and eps + UDF(w) > |w|.
"""
import math
RD = math.sqrt(3) / 2
W = 3.0


def wits(n2max):
    m = int(math.isqrt(n2max))
    return [(a, b, c, math.sqrt(a * a + b * b + c * c))
            for a in range(-m, m + 1) for b in range(-m, m + 1) for c in range(-m, m + 1)
            if 0 < a * a + b * b + c * c <= n2max]


WS = wits(25)


def certified(n, eps, R, concave=True):
    sgn = 1.0 if concave else -1.0
    # concave: C = origin + (R-eps) n ; convex: C = origin - (R+eps) n
    t = (R - eps) if concave else -(R + eps)
    C = (t * n[0], t * n[1], t * n[2])
    for (a, b, c, L) in WS:
        dx = a - C[0]; dy = b - C[1]; dz = c - C[2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        if concave:
            if r >= R:
                continue          # not exterior
            d0 = R - r
        else:
            if r <= R:
                continue
            d0 = r - R
        if d0 <= RD or d0 > W:
            continue
        if eps + d0 > L + 1e-13:
            return True
    return False


def sup_uncovered(n, R, concave=True, hi=None, step=2e-5):
    """largest eps in (0, r_d] with the origin uncertified (descending scan + bisection)."""
    if hi is None:
        hi = RD
    e = hi
    while e > 0:
        if not certified(n, e, R, concave):
            break
        e -= step
    if e <= 0:
        return 0.0
    # refine upward: find the top of the uncovered run
    a, b = e, min(e + step, hi)
    for _ in range(60):
        m = (a + b) / 2
        if certified(n, m, R, concave):
            b = m
        else:
            a = m
    return a


def unit(v):
    s = math.sqrt(sum(c * c for c in v))
    return tuple(c / s for c in v)


cases = [
    (100.0, (0.7183787310, 0.6193477580, 0.3176119960), 0.039089435425096, 0.039078258),
    (20.0, None, 0.041916417955, None),
    (12.0, None, 0.044621250541646, 0.043726855),
    (10.0, None, 0.046116224664, None),
    (5.0, (0.7342646959277773, 0.5958530896373325, 0.32529133385894815),
     0.057951775942861, 0.051099734),
    (3.0, (0.8632544645733072, 0.44253449005694867, 0.24280641363911454),
     0.099168924645437, 0.059499359),
]
print("CONCAVE sphere, EXACT rule, raw geometry (never uses g or c)")
print(" R   direction                              raw sup_uncovered   'curvature' TABLE2   "
      "'certificate' 2nd-order")
for R, n, tab2, tabc in cases:
    if n is None:
        continue
    n = unit(n)
    v = sup_uncovered(n, R, True)
    print(f"{R:6.1f}  ({n[0]:.9f},{n[1]:.9f},{n[2]:.9f})  {v:.12f}   {tab2:.12f}   "
          f"{'-' if tabc is None else format(tabc, '.9f')}")

# coarse grid search over the fundamental triangle at R = 5 and R = 3: does anything exceed?
print("\ncoarse grid search (fundamental domain) - does anything exceed TABLE 2?")
for R, tab2 in ((5.0, 0.057951775942861), (3.0, 0.099168924645437)):
    best = 0.0; bn = None
    K = 26
    for i in range(K + 1):
        for j in range(i + 1):
            for k in range(j + 1):
                if i == 0:
                    continue
                n = unit((i, j, k))
                v = sup_uncovered(n, R, True, step=5e-4)
                if v > best:
                    best, bn = v, n
    print(f"  R={R}: grid best {best:.9f} (vs {tab2:.9f}); argmax {bn}")
