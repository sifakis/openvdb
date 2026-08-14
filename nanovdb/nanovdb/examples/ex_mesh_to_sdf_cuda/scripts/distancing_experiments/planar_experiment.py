#!/usr/bin/env python3
"""
Empirical validation of the union-of-balls exterior-classification construction on a
random planar interface, as discussed in NewDistancingApproach.md. Pure Python (no
numpy available in this environment) -- grid is only 11^3=1331 points so this is fine.

Grid: integer lattice points in [-BOX,BOX]^3, dx=1.
Per trial: pick a random origin (continuous, in [0,1]^3) and a random unit normal n;
the interface is Sigma = {x : (x-origin).n = 0}, ground-truth signed distance
sdf(x) = (x-origin).n  (positive = exterior).

Construction under test:
  1. barrier voxels: |udf| <= sqrt(3)/2
  2. seed: one non-barrier voxel, verified (via ground truth, since we generated Sigma)
     to be on the exterior side -- flood-filled via 6-connectivity through non-barrier
     voxels only, to get the CC-certified set (this step does NOT otherwise use ground
     truth: connectivity is purely topological).
  3. enrichment closure: repeatedly certify any point V2 (barrier or not, in-band) for
     which some already-certified V1 satisfies d(V1) + UDF(V2) > dist(V1, V2) (strict).
  4. everything else stays "uncertified" (labeled interior by default).

Two measurements per trial:
  - soundness: assert every certified point is truly exterior (ground truth) --
    this is the inclusion property; a violation would be a serious bug/theory error.
  - tightness: among points within TEST_RADIUS of the origin, the largest UDF among
    points that are truly exterior but did NOT get certified. This is the empirical
    analogue of the theoretical eps*(W) bound.

Run both with no narrowband cutoff (W=None, uses the whole +-5 box) and with an
explicit W=3 cutoff (only points with UDF<=3 are eligible to seed/grow at all), to
separate "is the mechanism correct" from "does it match the W=3 table."
"""

import math
import random

BOX = 5
BARRIER_R = math.sqrt(3) / 2.0
TEST_RADIUS = 1.0

NEIGH_DELTAS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def build_grid():
    pts = []
    for x in range(-BOX, BOX + 1):
        for y in range(-BOX, BOX + 1):
            for z in range(-BOX, BOX + 1):
                pts.append((x, y, z))
    return pts


GRID = build_grid()
N = len(GRID)
IDX_OF = {c: i for i, c in enumerate(GRID)}


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def flood_fill_non_barrier(seed, non_barrier):
    certified = [False] * N
    certified[seed] = True
    frontier = [seed]
    while frontier:
        new_frontier = []
        for idx in frontier:
            cx, cy, cz = GRID[idx]
            for dx_, dy_, dz_ in NEIGH_DELTAS:
                ncoord = (cx + dx_, cy + dy_, cz + dz_)
                nidx = IDX_OF.get(ncoord)
                if nidx is not None and non_barrier[nidx] and not certified[nidx]:
                    certified[nidx] = True
                    new_frontier.append(nidx)
        frontier = new_frontier
    return certified


def enrich_closure(certified, in_band, udf):
    eligible = [in_band[i] and not certified[i] for i in range(N)]
    frontier = [i for i in range(N) if certified[i]]
    while frontier:
        eligible_idx = [i for i in range(N) if eligible[i]]
        if not eligible_idx:
            break
        captured = []
        for j in eligible_idx:
            pj = GRID[j]
            dj = udf[j]
            hit = False
            for i in frontier:
                if udf[i] + dj > dist(GRID[i], pj):
                    hit = True
                    break
            if hit:
                captured.append(j)
        if not captured:
            break
        for j in captured:
            certified[j] = True
            eligible[j] = False
        frontier = captured
    return certified


def run_trial(rng, W):
    origin = (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))
    nx, ny, nz = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    n = (nx / nn, ny / nn, nz / nn)

    signed = [0.0] * N
    udf = [0.0] * N
    is_ext_gt = [False] * N
    for i, (x, y, z) in enumerate(GRID):
        s = (x - origin[0]) * n[0] + (y - origin[1]) * n[1] + (z - origin[2]) * n[2]
        signed[i] = s
        udf[i] = abs(s)
        is_ext_gt[i] = s > 0

    if W is not None:
        in_band = [udf[i] <= W for i in range(N)]
    else:
        in_band = [True] * N
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

    for i in range(N):
        if certified[i] and is_ext_gt[i] != is_ext_gt[seed]:
            raise AssertionError("CC step mixed signs!")

    certified = enrich_closure(certified, in_band, udf)

    for i in range(N):
        if certified[i] and not is_ext_gt[i]:
            raise AssertionError(
                f"INCLUSION VIOLATION at {GRID[i]}, udf={udf[i]:.4f}, "
                f"origin={origin}, n={n}"
            )

    worst = 0.0
    for i, p in enumerate(GRID):
        if (
            in_band[i]
            and dist(p, origin) <= TEST_RADIUS
            and is_ext_gt[i]
            and not certified[i]
        ):
            if udf[i] > worst:
                worst = udf[i]
    return worst, origin, n


def main(num_trials=200, seed=12345):
    rng = random.Random(seed)
    for label, W in [("unrestricted (full +-5 box)", None), ("W=3 narrowband", 3.0)]:
        worst_overall = 0.0
        worst_config = None
        results = []
        skipped = 0
        for t in range(num_trials):
            r = run_trial(rng, W)
            if r is None:
                skipped += 1
                continue
            worst, origin, n = r
            results.append(worst)
            if worst > worst_overall:
                worst_overall = worst
                worst_config = (origin, n)

        results.sort()
        m = len(results)
        mean = sum(results) / m if m else float("nan")
        p95 = results[int(0.95 * (m - 1))] if m else float("nan")
        p99 = results[int(0.99 * (m - 1))] if m else float("nan")
        print(f"=== {label} ===")
        print(f"  trials: {num_trials} (skipped {skipped} degenerate)")
        print(f"  max empirical eps found : {worst_overall:.4f}")
        print(f"  mean                    : {mean:.4f}")
        print(f"  95th percentile         : {p95:.4f}")
        print(f"  99th percentile         : {p99:.4f}")
        if worst_config is not None:
            o, n = worst_config
            print(f"  worst-case origin={o}, normal={n}")
        print()


if __name__ == "__main__":
    main()
