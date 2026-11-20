# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Compare grouped pool-height queries with the original full-scan definition."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from verify_pool import height_tiles


def reference(bed, center, spacing):
    surface = np.full((20, 15), np.nan)
    for x in range(20):
        for z in range(15):
            cx, cz = (x + 0.5) * 0.16, (z + 0.5) * 0.16
            if np.hypot(cx - center[0], cz - center[2]) < 0.65:
                continue
            q = bed[(bed[:, 0] >= x * 0.16) & (bed[:, 0] < (x + 1) * 0.16)
                    & (bed[:, 2] >= z * 0.16) & (bed[:, 2] < (z + 1) * 0.16), 1]
            if len(q) >= 4:
                surface[x, z] = np.quantile(q, 0.9) + 0.5 * spacing
    return surface


def compare(name, bed, center, spacing):
    start = time.perf_counter()
    expected = reference(bed, center, spacing)
    old_seconds = time.perf_counter() - start
    start = time.perf_counter()
    actual = height_tiles(bed, center, spacing)
    new_seconds = time.perf_counter() - start
    assert np.array_equal(expected, actual, equal_nan=True), name
    return dict(case=name, particles=len(bed), exactly_equal=True,
                original_seconds=old_seconds, grouped_seconds=new_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 90, 180])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads((args.cache / "settings.json").read_text())
    center = settings["emitter_center"]
    spacing = settings["particle_volume"] ** (1 / 3)
    # Exact tile edges and their immediate float neighbors, including the box edges.
    points = []
    for x in np.arange(21) * .16:
        for z in np.arange(16) * .16:
            for px in [np.nextafter(x, -np.inf), x, np.nextafter(x, np.inf)]:
                for pz in [np.nextafter(z, -np.inf), z, np.nextafter(z, np.inf)]:
                    for y in [.01, .03, .05, .07]:
                        points.append([px, y, pz])
    results = [compare("float_tile_boundaries", np.array(points), center, spacing),
               compare("empty", np.empty((0, 3)), center, spacing)]
    for frame in args.frames:
        particles = np.loadtxt(args.cache / f"frame_{frame:06d}.csv", delimiter=",",
                               skiprows=1, usecols=(1, 2, 3), ndmin=2)
        results.append(compare(f"frame_{frame:06d}", particles[particles[:, 1] < .25], center, spacing))
    result = dict(pass_all=True, cases=results)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
