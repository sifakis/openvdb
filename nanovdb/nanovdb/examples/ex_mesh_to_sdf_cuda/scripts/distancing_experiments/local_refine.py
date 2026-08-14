#!/usr/bin/env python3
"""
Local hill-climbing refinement starting from the best config found by the Monte Carlo
search, to see how close general orientations can get to (or past) the axis-aligned
worst-case value of 0.12915.
"""

import math
import random
from general_orientation_search import (
    build_trial, capture_eps, tangent_basis, W,
)

rng = random.Random(99)


def eval_config(origin, n, off):
    r = build_trial_fixed(origin, n)
    if r is None:
        return None
    cert_pts, cert_udf = r
    return capture_eps(origin, n, off, cert_pts, cert_udf)


def build_trial_fixed(origin, n):
    from planar_experiment import GRID, N, BARRIER_R, flood_fill_non_barrier, enrich_closure
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
    cert_pts = [GRID[i] for i in range(N) if certified[i]]
    cert_udf = [udf[i] for i in range(N) if certified[i]]
    return cert_pts, cert_udf


def normalize(v):
    nn = math.sqrt(sum(c * c for c in v))
    return tuple(c / nn for c in v)


# start from the best config the Monte Carlo search found
origin = [0.7770406273978518, 0.4567543112447, 0.40670900875736216]
n = [-0.9405013777804657, -0.01691840266614113, 0.33936842228509123]
off = [0.1080462542559194, 1.071038823743452, 0.3528257469467358]

best_eps = eval_config(tuple(origin), tuple(n), tuple(off))
print(f"start: eps={best_eps:.6f}")

MAX_OFF_MAG = 1.5  # matches the original Monte Carlo search's sampling radius


def clamp01(v):
    return [c % 1.0 for c in v]


def clamp_off(v):
    mag = math.sqrt(sum(c * c for c in v))
    if mag <= MAX_OFF_MAG or mag == 0.0:
        return v
    scale = MAX_OFF_MAG / mag
    return [c * scale for c in v]


def project_tangential(v, n):
    dot = sum(a * b for a, b in zip(v, n))
    return [a - dot * b for a, b in zip(v, n)]


step = 0.15
no_improve = 0
for it in range(1, 601):
    cand_origin = clamp01([c + rng.uniform(-step, step) for c in origin])
    cand_n = normalize([c + rng.uniform(-step, step) for c in n])
    cand_off = clamp_off(project_tangential(
        [c + rng.uniform(-step, step) for c in off], cand_n))
    eps = eval_config(tuple(cand_origin), tuple(cand_n), tuple(cand_off))
    if eps is not None and eps > best_eps:
        best_eps = eps
        origin, n, off = cand_origin, list(cand_n), cand_off
        no_improve = 0
    else:
        no_improve += 1
        if no_improve > 40:
            step *= 0.7
            no_improve = 0
    if it % 100 == 0:
        print(f"  iter {it}: best eps={best_eps:.6f}, step={step:.4f}")

print(f"\nfinal refined eps: {best_eps:.6f}")
print(f"final origin={origin}")
print(f"final n={n}  (|n|={math.sqrt(sum(c*c for c in n)):.4f})")
print(f"final tangential offset={off}")
print("\ncompare: axis-aligned worst = 0.12915, rigorous universal bound = 0.19325")
