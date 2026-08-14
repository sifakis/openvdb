#!/usr/bin/env python3
"""
Quantify how many BARRIER lattice points (the only ones enrichment can affect --
non-barrier points are always certified by CC alone) actually get captured by the
sphere-union enrichment rule, across many random planar configurations.
"""

import random
from planar_experiment import (
    GRID, N, BARRIER_R, flood_fill_non_barrier, enrich_closure,
)


def run_trial(rng, W):
    origin = (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))
    nx, ny, nz = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
    import math
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    n = (nx / nn, ny / nn, nz / nn)

    udf = [0.0] * N
    is_ext_gt = [False] * N
    for i, (x, y, z) in enumerate(GRID):
        s = (x - origin[0]) * n[0] + (y - origin[1]) * n[1] + (z - origin[2]) * n[2]
        udf[i] = abs(s)
        is_ext_gt[i] = s > 0

    in_band = [udf[i] <= W for i in range(N)] if W is not None else [True] * N
    is_barrier = [udf[i] <= BARRIER_R for i in range(N)]
    non_barrier_in_band = [in_band[i] and not is_barrier[i] for i in range(N)]

    seed = None
    for i in range(N):
        if non_barrier_in_band[i] and is_ext_gt[i]:
            seed = i
            break
    if seed is None:
        return None

    certified_cc = flood_fill_non_barrier(seed, non_barrier_in_band)
    certified_final = enrich_closure(certified_cc[:], in_band, udf)

    # count exterior, in-band barrier lattice points, and how many got certified
    total_barrier_ext = 0
    captured_barrier_ext = 0
    for i in range(N):
        if in_band[i] and is_barrier[i] and is_ext_gt[i]:
            total_barrier_ext += 1
            if certified_final[i]:
                captured_barrier_ext += 1
    return total_barrier_ext, captured_barrier_ext


def main(num_trials, seed):
    rng = random.Random(seed)
    for label, W in [("unrestricted", None), ("W=3", 3.0)]:
        total = 0
        captured = 0
        trials_with_miss = 0
        for _ in range(num_trials):
            r = run_trial(rng, W)
            if r is None:
                continue
            t, c = r
            total += t
            captured += c
            if c < t:
                trials_with_miss += 1
        rate = 100.0 * captured / total if total else float("nan")
        print(f"=== {label} ===")
        print(f"  barrier exterior lattice points seen : {total}")
        print(f"  captured by enrichment                : {captured} ({rate:.3f}%)")
        print(f"  missed                                 : {total - captured}")
        print(f"  trials with >=1 miss                   : {trials_with_miss}/{num_trials}")
        print()


if __name__ == "__main__":
    main(num_trials=500, seed=321)
