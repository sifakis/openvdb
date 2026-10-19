#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""How far must the ball-certification sweep look?  Reproduces the numbers in the
"Stencil reach of the closure sweep" section of NewDistancingApproach.md.

1. Bubble simulations.  The exterior near the configuration is U = B(n, rn) u B(q, rq), with n at
   the origin and q a lattice point; Sigma is the boundary of U (a tunnel joining B(n, rn) to the
   rest of the exterior is assumed on the far side, so n is a valid seed).  All UDF values are true
   distances to Sigma.  The closure is run to its fixpoint with stencils |w|^2 <= k for increasing
   k, and the smallest k that certifies q is reported.

2. Ellipse enumeration.  A hop n -> q can be replaced by two shorter hops through any active
   lattice point x in E = { x : |x-n| + |x-q| < rn + rq }.  For each hop class longer than W, the
   largest rn + rq that keeps E free of lattice points (other than n, q) is m(w) =
   min_x |x| + |x-w|; the script reports it, and the largest lens thickness and window radius
   achievable with rn <= W, rq < D.

3. Stencil tap counts.

Pure Python, no numpy.  Runs in a few minutes (the 3D UDF evaluation dominates).
"""
import itertools
import math

W = 3.0


def fib_sphere(N):
    g = math.pi * (3 - math.sqrt(5))
    for i in range(N):
        z = 1 - 2 * (i + 0.5) / N
        r = math.sqrt(1 - z * z)
        yield (r * math.cos(g * i), r * math.sin(g * i), z)


def circle(N):
    for i in range(N):
        t = 2 * math.pi * i / N
        yield (math.cos(t), math.sin(t))


def bubble(q, rn, rq):
    """Smallest k such that the closure with stencil |w|^2 <= k certifies q (None if > 20)."""
    dim = len(q)
    n = (0.0,) * dim
    dirs = list(fib_sphere(400000) if dim == 3 else circle(400000))
    # Boundary of U, sampled densely: each sphere minus the part swallowed by the other ball.
    S = [tuple(rn * u[k] for k in range(dim)) for u in dirs]
    S = [p for p in S if math.dist(p, q) >= rq]
    S += [p for p in (tuple(q[k] + rq * u[k] for k in range(dim)) for u in dirs[::20])
          if math.dist(p, n) >= rn]
    h = 0.5
    buckets = {}
    for p in S:
        buckets.setdefault(tuple(math.floor(c / h) for c in p), []).append(p)

    def udf(x):
        best, c = float("inf"), tuple(math.floor(v / h) for v in x)
        for rad in range(40):
            for o in itertools.product(range(-rad, rad + 1), repeat=dim):
                if max(abs(v) for v in o) != rad:
                    continue
                for p in buckets.get(tuple(c[k] + o[k] for k in range(dim)), ()):
                    best = min(best, math.dist(x, p))
            if best < rad * h:
                return best
        return best

    pts = {}
    for x in itertools.product(range(-4, 8), repeat=dim):
        if math.dist(x, n) < rn or math.dist(x, q) < rq:  # exterior lattice points only
            u = udf(x)
            if u <= W:
                pts[x] = u
    Q = tuple(int(v) for v in q)
    for k in range(1, 21):
        cert, changed = {tuple(int(v) for v in n)}, True
        while changed:
            changed = False
            for x, ux in pts.items():
                if x in cert:
                    continue
                for y in cert:
                    L2 = sum((x[i] - y[i]) ** 2 for i in range(dim))
                    if L2 <= k and ux + pts[y] > math.sqrt(L2) + 1e-12:
                        cert.add(x)
                        changed = True
                        break
        if Q in cert:
            return k, len(pts)
    return None, len(pts)


def ellipse_table(D, dim):
    vecs = {2: [(3, 1), (3, 2)], 3: [(3, 1, 0), (3, 1, 1), (2, 2, 2), (3, 2, 0), (3, 2, 1)]}[dim]
    for w in vecs:
        L = math.hypot(*w)
        if L >= W + D:
            continue
        zero = (0,) * dim
        m = min(math.dist(x, zero) + math.dist(x, w)
                for x in itertools.product(range(-1, 5), repeat=dim) if x not in (zero, w))
        best = (0.0, 0.0)
        for i in range(1, 1001):
            for j in range(1, 1001):
                rn, rq = W * i / 1000, D * j / 1000 * 0.99999
                if not (L < rn + rq <= m):
                    continue
                a = (L * L + rn * rn - rq * rq) / (2 * L)
                best = max(best, (math.sqrt(max(0.0, rn * rn - a * a)), rn + rq - L))
        status = "never (E always holds a lattice point)" if best == (0.0, 0.0) else \
            "lattice-free iff rn+rq <= %.4f; max window radius %.3f, max lens %.3f" % (m, *best)
        print("    %-10s |w| = %.4f : %s" % (w, L, status))


def taps(k, dim):
    return sum(1 for w in itertools.product(range(-4, 5), repeat=dim) if 0 < sum(v * v for v in w) <= k)


if __name__ == "__main__":
    print("Bubble simulations (smallest |w|^2 whose stencil certifies q):")
    for q, rn, rq in [((3.0, 1.0), 2.55, 0.65), ((3.0, 1.0), 2.75, 0.45), ((3.0, 2.0), 2.95, 0.68),
                      ((3.0, 1.0, 1.0), 2.95, 0.45), ((3.0, 2.0, 1.0), 2.98, 0.82)]:
        k, npts = bubble(q, rn, rq)
        print("  %dD q=%-15s udf(n)=%.2f udf(q)=%.2f : needs |w|^2 >= %s  (%d exterior band points)"
              % (len(q), q, rn, rq, k, npts))
    for D, name in [(0.5, "1/2"), (math.sqrt(2) / 2, "sqrt2/2"), (math.sqrt(3) / 2, "sqrt3/2")]:
        for dim in (2, 3):
            if (dim == 2 and name == "sqrt3/2") or (dim == 3 and name == "sqrt2/2"):
                continue
            print("Ellipse enumeration, %dD, D = %s (W + D = %.4f):" % (dim, name, W + D))
            ellipse_table(D, dim)
    print("Taps: 2D |w|^2<=10: %d, <=13: %d, box: %d;  3D |w|^2<=9: %d, <=11: %d, <=14: %d, box: %d"
          % (taps(10, 2), taps(13, 2), 7 ** 2 - 1, taps(9, 3), taps(11, 3), taps(14, 3), 7 ** 3 - 1))
