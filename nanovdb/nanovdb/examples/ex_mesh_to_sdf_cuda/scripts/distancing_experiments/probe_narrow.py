#!/usr/bin/env python3
import math
import random
from wedge_core import Wedge, run_construction, dist

random.seed(7)

for alpha_deg in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
    alpha = math.radians(alpha_deg)
    no_seed_count = 0
    trial_info = []
    for trial in range(8):
        vertex = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        theta1 = random.uniform(0, 2 * math.pi)
        wedge = Wedge(vertex, theta1, alpha)
        res = run_construction(wedge, box=10, w=3.0)
        if res["no_seed_at_all"]:
            no_seed_count += 1
        else:
            grid, d, is_ext, in_band, certified = (
                res["grid"], res["d"], res["is_ext"], res["in_band"], res["certified"])
            misses = [p for p in grid if in_band[p] and is_ext[p] and p not in certified]
            trial_info.append((len(misses), max((d[p] for p in misses), default=0.0)))
    print(f"alpha={alpha_deg:3d} deg: no_seed_at_all in {no_seed_count}/8 trials; "
          f"miss counts (of trials WITH a seed): {trial_info}")
