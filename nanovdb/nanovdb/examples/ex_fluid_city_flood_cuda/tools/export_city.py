#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""export_city.py <World_CityDemopack.usd> <out.obj>: the City Demo pack's city (buildings, streets, the river's bed and
banks) as one triangle mesh in the fluid tank's frame.

Tank frame, in meters: x = the stage's x + 760 (0..1911), y up from the street level (the river bed is at -20, the
tallest building reaches 123), z = the stage's z + 773 (0..1764); a translation only. Left out: the river's water body
(the prim below g_WaterFront whose top is at -4 m), since the simulation brings its own water.
Prints, for a look at how closed the ground is, the share of the city's area (2 m cells, outside the river) that has
a horizontal surface at street level.
Run with python -I.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder only (python -I drops it)
import numpy as np
from pxr import Usd, UsdGeom
from inspect_usd import triangles_of, edge_counts

stage = Usd.Stage.Open(sys.argv[1])
scale = UsdGeom.GetStageMetersPerUnit(stage)
water = set()  # the prims of the river's water: below g_WaterFront, neither reaching the bed nor the street level
for prim in stage.Traverse():
    if "g_WaterFront" in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh):
        box = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
        up = 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1
        if box.GetMin()[up] * scale > -19.0 and box.GetMax()[up] * scale < -1.0:
            water.add(str(prim.GetPath()))
points, faces, prims = triangles_of(stage, lambda path: path not in water)
tank = points + np.array([760.0, 0.0, 773.0])
edges, boundary, bad = edge_counts(tank, faces)
print(f"left out as the river's water: {sorted(water)}")
print(f"prims {prims}, triangles {len(faces)}, points {len(tank)}; x {tank[:, 0].min():.2f}..{tank[:, 0].max():.2f}, "
      f"y {tank[:, 1].min():.2f}..{tank[:, 1].max():.2f}, z {tank[:, 2].min():.2f}..{tank[:, 2].max():.2f}; "
      f"edges {edges}, boundary {boundary} ({100.0 * boundary / edges:.1f}%), non-manifold {bad}")

# How much of the plan has ground: 2 m cells covered by a nearly horizontal triangle within 0.5 m of the street level,
# and those covered by the river bed (below -10 m). Each triangle marks the cells of sample points spread over it.
cell, nx, nz = 2.0, 956, 882
ground, river = np.zeros((nx, nz), bool), np.zeros((nx, nz), bool)
t = tank[faces]
normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
area = 0.5 * np.linalg.norm(normal, axis=1)
flat = np.abs(normal[:, 1]) > 0.9 * np.maximum(2.0 * area, 1e-12)
rng = np.random.default_rng(0)
for mask, level in ((ground, lambda y: np.abs(y) < 0.5), (river, lambda y: y < -10.0)):
    pick = flat & level(t[:, :, 1].mean(axis=1))
    count = np.maximum(1, np.ceil(area[pick] / (0.25 * cell * cell)).astype(np.int64))  # about 4 samples per cell
    owner = np.repeat(np.flatnonzero(pick), count)
    u, v = rng.random(len(owner)), rng.random(len(owner))
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    p = t[owner, 0] + u[:, None] * (t[owner, 1] - t[owner, 0]) + v[:, None] * (t[owner, 2] - t[owner, 0])
    i, k = (p[:, 0] // cell).astype(np.int64), (p[:, 2] // cell).astype(np.int64)
    ok = (i >= 0) & (i < nx) & (k >= 0) & (k < nz)
    mask[i[ok], k[ok]] = True
print(f"plan in 2 m cells: river bed {100.0 * river.mean():.1f}%, ground at street level {100.0 * (ground & ~river).mean():.1f}%, "
      f"neither {100.0 * (~ground & ~river).mean():.1f}%")

with open(sys.argv[2], "w") as f:
    np.savetxt(f, tank, fmt="v %.4f %.4f %.4f")
    np.savetxt(f, faces + 1, fmt="f %d %d %d")
