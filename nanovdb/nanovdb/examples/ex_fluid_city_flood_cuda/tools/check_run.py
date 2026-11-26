#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""check_run.py <output/name> [frame ...]: the numbers of a city flood run to compare with the paper's run.

From out.log: the number of steps, the mean time per step, the peak device memory and the range of the water's volume
(level set + drops, in percent of the start). From each given frame (default: the last one), with the voxel size of
voxel.txt: the wet cells below street level under the city blocks (x 520..1000 m, z < 812 m, more than 1.5 m below the
street; west and east of that the river itself is there), which must be 0 if the ground holds the water; the wet
cells more than 4 m below the street anywhere (the basin's floor is 3 m below it); how far the water on the streets
reaches; and in four places the share of columns that are wet and of those the share at least three cells deep, which
is the least the solver carries well.
A frame file holds the level set within 2 voxels of the surface: int32 count, nx, ny, nz, then per cell int16 x, y, z
and float32 value (negative in the water). Cell (i, j, k) is at world (i h, j h - 21, k h) meters.
"""
import re, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
h = float((run / "voxel.txt").read_text())
log = (run / "out.log").read_text()
volume = [float(v) for v in re.findall(r"level set [0-9.]+ \+ drops = ([0-9.]+)% of the start", log)]
steps = re.findall(r"time per step so far \((\d+) steps, [0-9.]+ pressure solves per step\): ([0-9.]+) ms", log)
memory = re.findall(r"peak device memory so far, MiB, in the same parts: ([0-9, ]+);", log)
print(f"{run}: voxel {h} m, {steps[-1][0]} steps, {steps[-1][1]} ms per step, peak device memory {max(int(m) for m in memory[-1].split(','))} MiB, "
      f"volume {min(volume):.3f}%..{max(volume):.3f}% of the start, {volume[-1]:.3f}% at the end; {log.strip().splitlines()[-1]}")

places = {"garden east of the park": (822, 860, 740, 785), "road west, far end": (360, 480, 790, 812),
          "road west": (480, 740, 790, 812), "road east": (810, 1080, 790, 812)}  # x0, x1, z0, z1 in meters
frames = sys.argv[2:] or [str(len(list((run / "out").glob("f_*.band"))) - 1)]
for frame in frames:
    path = run / "out" / f"f_{int(frame):03d}.band"
    count = int(np.fromfile(path, "<i4", 1)[0])
    band = np.fromfile(path, np.dtype([("cell", "<i2", 3), ("value", "<f4")]), count, offset=16)
    wet = band["cell"][band["value"] < 0].astype(np.int64)
    x, y, z = wet[:, 0] * h, wet[:, 1] * h - 21, wet[:, 2] * h
    under = (x > 520) & (x < 1000) & (z < 812) & (y < -1.5)
    street = (y > -1.0) & (z < 812)
    print(f"frame {int(frame)}: {count} band cells, {len(wet)} wet, {under.sum()} wet under the blocks, {(y < -4.0).sum()} wet more than 4 m below the street; "
          f"street water x {x[street].min():.0f}..{x[street].max():.0f} m, z {z[street].min():.0f}..{z[street].max():.0f} m")
    for name, (x0, x1, z0, z1) in places.items():
        inside = (x >= x0) & (x < x1) & (z >= z0) & (z < z1) & (y > -1.0)
        depth = np.zeros((round((x1 - x0) / h), round((z1 - z0) / h)), np.int32)
        np.add.at(depth, (wet[inside, 0] - round(x0 / h), wet[inside, 2] - round(z0 / h)), 1)
        deep = depth[depth > 0]
        print(f"    {name}: {100 * len(deep) / depth.size:.0f}% of the columns wet, {100 * np.mean(deep >= 3) if len(deep) else 0:.0f}% of those 3+ cells deep")
