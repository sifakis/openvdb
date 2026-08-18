#!/usr/bin/env python3
import math
import random
from wedge_core import Wedge, run_construction

random.seed(20260817)

N_TRIALS = 25
results = []
for i in range(N_TRIALS):
    vertex = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
    theta1 = random.uniform(0, 2 * math.pi)
    alpha_deg = random.uniform(20, 160)
    alpha = math.radians(alpha_deg)
    wedge = Wedge(vertex, theta1, alpha)

    res = run_construction(wedge)
    if res["no_seed_at_all"]:
        results.append((i, alpha_deg, vertex, theta1, None, None, "NO SEED AT ALL"))
        continue

    grid, d, is_ext, in_band, certified = (
        res["grid"], res["d"], res["is_ext"], res["in_band"], res["certified"])
    misses = [p for p in grid if in_band[p] and is_ext[p] and p not in certified]

    # distance from vertex for each miss, to see if they cluster near the tip
    miss_info = sorted(
        [(dist(p, wedge.vertex), p, d[p]) for p in misses],
        key=lambda t: t[0]
    ) if misses else []

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    closest_miss_to_vertex = miss_info[0][0] if miss_info else None
    max_udf_among_misses = max((d[p] for p in misses), default=0.0)

    results.append((i, alpha_deg, vertex, theta1, len(misses), closest_miss_to_vertex,
                     max_udf_among_misses))

print(f"{'trial':>5} {'alpha_deg':>10} {'#misses':>8} {'closest_miss_to_vertex':>22} "
      f"{'max_udf_of_miss':>16}")
for r in results:
    i, alpha_deg, vertex, theta1, n_or_flag, closest, maxudf = r
    if n_or_flag == "NO SEED AT ALL":
        print(f"{i:5d} {alpha_deg:10.2f}   *** NO ELIGIBLE SEED AT ALL ***")
        continue
    closest_s = f"{closest:.4f}" if closest is not None else "-"
    maxudf_s = f"{maxudf:.4f}" if n_or_flag else "-"
    print(f"{i:5d} {alpha_deg:10.2f} {n_or_flag:8d} {closest_s:>22} {maxudf_s:>16}")

total_misses = sum(r[4] for r in results if r[4] not in (None,) and r[4] != "NO SEED AT ALL")
trials_with_misses = sum(1 for r in results if isinstance(r[4], int) and r[4] > 0)
no_seed_trials = sum(1 for r in results if r[4] is None)
print(f"\n{trials_with_misses}/{N_TRIALS} trials had at least one miss; "
      f"{no_seed_trials} had no eligible seed at all; total misses across all trials: {total_misses}")
