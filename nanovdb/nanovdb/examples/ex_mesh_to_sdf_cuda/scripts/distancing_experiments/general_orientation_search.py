#!/usr/bin/env python3
"""
Monte Carlo search over general orientations for the worst-case (largest) capture
threshold eps of a CONTINUOUS query point, to compare against the theoretical bounds:
    rigorous universal bound   : eps*_refined(3) = 0.19325
    suggestive further bound   : ~0.1307
    axis-aligned worst offset  : 0.12915  (already exactly confirmed by direct scan)

For each trial: random n (uniform on sphere), random origin in [0,1)^3, build the
certified lattice set (CC + enrichment closure, W=3). Then for many random tangential
offsets near the origin, binary-search the smallest eps along the normal ray at which
the continuous point becomes captured (enters the union of balls around certified
points). Track the running max across all trials/samples.
"""

import math
import random
from planar_experiment import GRID, N, BARRIER_R, flood_fill_non_barrier, enrich_closure

W = 3.0


def tangent_basis(n):
    # any vector not parallel to n
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    # t1 = a - (a.n) n, normalized
    dot = a[0] * n[0] + a[1] * n[1] + a[2] * n[2]
    t1 = (a[0] - dot * n[0], a[1] - dot * n[1], a[2] - dot * n[2])
    norm1 = math.sqrt(sum(c * c for c in t1))
    t1 = (t1[0] / norm1, t1[1] / norm1, t1[2] / norm1)
    # t2 = n x t1
    t2 = (
        n[1] * t1[2] - n[2] * t1[1],
        n[2] * t1[0] - n[0] * t1[2],
        n[0] * t1[1] - n[1] * t1[0],
    )
    return t1, t2


def build_trial(rng, W):
    origin = (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))
    nx, ny, nz = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    n = (nx / nn, ny / nn, nz / nn)

    udf = [0.0] * N
    is_ext_gt = [False] * N
    for i, (x, y, z) in enumerate(GRID):
        s = (x - origin[0]) * n[0] + (y - origin[1]) * n[1] + (z - origin[2]) * n[2]
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
        return None

    certified = flood_fill_non_barrier(seed, non_barrier_in_band)
    certified = enrich_closure(certified, in_band, udf)

    for i in range(N):
        if certified[i] and not is_ext_gt[i]:
            raise AssertionError(f"INCLUSION VIOLATION, origin={origin}, n={n}")

    cert_pts = [GRID[i] for i in range(N) if certified[i]]
    cert_udf = [udf[i] for i in range(N) if certified[i]]
    return origin, n, cert_pts, cert_udf


def in_union_of_balls(p, cert_pts, cert_udf):
    for v, d in zip(cert_pts, cert_udf):
        d2 = (p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2 + (p[2] - v[2]) ** 2
        if d2 < d * d:
            return True
    return False


def capture_eps(origin, n, tang_offset, cert_pts, cert_udf, hi0=1.0, iters=35):
    base = (origin[0] + tang_offset[0], origin[1] + tang_offset[1], origin[2] + tang_offset[2])
    lo, hi = 0.0, hi0
    p_hi = (base[0] + hi * n[0], base[1] + hi * n[1], base[2] + hi * n[2])
    while not in_union_of_balls(p_hi, cert_pts, cert_udf):
        hi *= 2
        if hi > 8:
            return None
        p_hi = (base[0] + hi * n[0], base[1] + hi * n[1], base[2] + hi * n[2])
    for _ in range(iters):
        mid = (lo + hi) / 2
        p_mid = (base[0] + mid * n[0], base[1] + mid * n[1], base[2] + mid * n[2])
        if in_union_of_balls(p_mid, cert_pts, cert_udf):
            hi = mid
        else:
            lo = mid
    return hi


def main(num_trials=150, samples_per_trial=25, seed=7):
    rng = random.Random(seed)
    worst_overall = 0.0
    worst_config = None
    skipped = 0
    for t in range(num_trials):
        r = build_trial(rng, W)
        if r is None:
            skipped += 1
            continue
        origin, n, cert_pts, cert_udf = r
        t1, t2 = tangent_basis(n)
        for _ in range(samples_per_trial):
            # random tangential offset within a disk of radius ~1.5 around origin
            radius = rng.uniform(0.0, 1.5)
            theta = rng.uniform(0.0, 2 * math.pi)
            off = (
                radius * math.cos(theta) * t1[0] + radius * math.sin(theta) * t2[0],
                radius * math.cos(theta) * t1[1] + radius * math.sin(theta) * t2[1],
                radius * math.cos(theta) * t1[2] + radius * math.sin(theta) * t2[2],
            )
            eps = capture_eps(origin, n, off, cert_pts, cert_udf)
            if eps is not None and eps > worst_overall:
                worst_overall = eps
                worst_config = (origin, n, off)
        if (t + 1) % 25 == 0:
            print(f"  ...{t+1}/{num_trials} trials, running max eps={worst_overall:.5f}")

    print(f"\nfinal max empirical eps over {num_trials} trials x {samples_per_trial} "
          f"tangential samples (skipped {skipped}): {worst_overall:.5f}")
    print("compare: rigorous universal bound = 0.19325, "
          "suggestive further bound ~0.1307, axis-aligned worst = 0.12915")
    if worst_config is not None:
        o, n, off = worst_config
        print(f"worst config: origin={o}, n={n}, tangential offset={off}")


if __name__ == "__main__":
    main()
