#!/usr/bin/env python3
"""
Corrected targeted validation. The theoretical eps*(W) is about capturing an arbitrary
CONTINUOUS query point p (not restricted to lattice locations) inside O = union of balls
around the CERTIFIED LATTICE points. Testing only lattice points (as the first version of
this script did) is close to vacuous, since almost every non-barrier lattice point is
already certified by plain 6-connectivity CC alone, well before enrichment even runs.

Fix n=(1,0,0), origin=(c, 0.5, 0.5) (the tangential worst-case position -- cube-center,
offset 0.5 in both y and z, matching the axis-aligned covering-radius argument), scan c
over [0,1). For each c: build the certified lattice set exactly as before, then binary
search on epsilon for the query point p=(c+eps, 0.5, 0.5) to find the smallest eps at
which p enters O (is within the ball of some certified lattice point). Compare against

    eps*_aligned(d) = d - sqrt(d^2 - 0.5),  d = deepest available integer depth
"""

import math
from planar_experiment import GRID, N, BARRIER_R, flood_fill_non_barrier, enrich_closure

W = 3.0


def certified_set_for(c, W):
    udf = [0.0] * N
    is_ext_gt = [False] * N
    for i, (x, y, z) in enumerate(GRID):
        s = x - c
        udf[i] = abs(s)
        is_ext_gt[i] = s > 0

    in_band = [udf[i] <= W for i in range(N)]
    is_barrier = [udf[i] <= BARRIER_R for i in range(N)]
    non_barrier_in_band = [in_band[i] and not is_barrier[i] for i in range(N)]

    seed = None
    for i in range(N):
        if non_barrier_in_band[i] and is_ext_gt[i]:
            seed = i
            break
    if seed is None:
        return None, None

    certified = flood_fill_non_barrier(seed, non_barrier_in_band)
    certified = enrich_closure(certified, in_band, udf)

    for i in range(N):
        if certified[i] and not is_ext_gt[i]:
            raise AssertionError(f"INCLUSION VIOLATION at {GRID[i]}, c={c}")

    return certified, udf


def in_union_of_balls(p, certified, udf):
    for i, v in enumerate(GRID):
        if certified[i]:
            d2 = (p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2 + (p[2] - v[2]) ** 2
            if d2 < udf[i] * udf[i]:
                return True
    return False


def empirical_eps(c, certified, udf, lo=0.0, hi=1.0, iters=40):
    # find smallest eps in [lo,hi] such that p=(c+eps,0.5,0.5) is captured.
    # hi must be captured; if not, expand.
    while not in_union_of_balls((c + hi, 0.5, 0.5), certified, udf):
        hi *= 2
        if hi > 10:
            return None  # not captured even far away -- shouldn't happen for W=3
    for _ in range(iters):
        mid = (lo + hi) / 2
        if in_union_of_balls((c + mid, 0.5, 0.5), certified, udf):
            hi = mid
        else:
            lo = mid
    return hi


def predicted_eps(c, W):
    k = math.floor(W + c)
    d = k - c
    return d - math.sqrt(d * d - 0.5)


print(f"{'c':>8} {'empirical eps':>15} {'predicted eps':>15} {'deepest d':>10}")
worst_ratio = 0.0
for i in range(0, 100):
    c = i / 100.0 + 0.0013
    certified, udf = certified_set_for(c, W)
    if certified is None:
        continue
    emp = empirical_eps(c, certified, udf)
    pred = predicted_eps(c, W)
    k = math.floor(W + c)
    d = k - c
    if i % 10 == 0:
        print(f"{c:8.4f} {emp:15.6f} {pred:15.6f} {d:10.4f}")
    if emp is not None and pred > 0:
        worst_ratio = max(worst_ratio, emp / pred)

print(f"\nmax(empirical/predicted) over scan: {worst_ratio:.4f} "
      "(should be <=1 if the closed-form is a correct exact match)")
