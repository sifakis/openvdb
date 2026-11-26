#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""build_city.py <out.obj> <voxel in m> [options]: the solids of the city flood scene as one triangle mesh for MeshToSDF
(gen_sdf), in the fluid tank's cell coordinates times the voxel size.

World frame (meters), as written by tools/export_city.py: x 0..1911, y up from the street level (the
river bed is at -20), z 0..1764. Between x = 500 and x = 900 the river's north bank runs along x at z = 815, with the
river on the +z side. The scene: a basin in the river against that bank, x in [x0, x1], z in [815, 815 + width], closed
by three walls and a raised floor, and filled with water up to `head` above the street. A flood wall stands on the bank
along the basin; its openings (--gate x0 x1 [x0 x1 ...], from west to east) let the water into the city.
--parapet x0 x1 height: a lower wall on the bank on either side of the basin, from x0 to x1, which keeps the water on
the street along the bank from running back into the river.
The city's own surfaces are sheets (streets, lots, the river's banks) and shells (buildings), and the solver needs
three cells of solid behind any water (its level set continues into the solid; through a thinner wall the water
comes out on the other side). So the scene adds what is behind them: a block under the ground on either side of the
river, up to just below the streets, and a prism from every roof triangle down to the ground.
Tank frame: cell (i, j, k) covers world [i h, (i + 1) h] x [j h - base, (j + 1) h - base] x [k h, (k + 1) h], where
base (21 m) puts the river bed above the tank's floor; the mesh is written as (world + (0, base, 0)) - h / 2, so its
coordinate divided by h is the cell-center index.
--bankwall height x z x z ...: a wall 2 m thick along the polyline (x, z), from 1 m below the street to `height`
above it; it can be given several times. For walls that follow the bank where it is not parallel to x. Its mesh is
also written to <out>.bankwalls.obj in world meters, for a renderer.
--region x0 x1 z0 z1: only the city's triangles with their centroid in this rectangle, and the ground blocks cut to it,
are written (the solids the water can reach in the run; the tank stays the whole city).
--clip <m>: every triangle is cut at this height above the street (0: not cut). The water never gets higher, and the
band around the tall buildings' upper parts is most of what MeshToSDF would store.
Prints the tank size in cells, the basin's water box in cell coordinates and the boxes a renderer draws for the basin
and the walls (RENDER_BOXES, world meters, six numbers each). Run with python -I.
"""
import argparse, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("voxel", type=float)
p.add_argument("--city", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/city.obj"))
p.add_argument("--basin", type=float, nargs=3, default=[690.0, 860.0, 80.0])  # x0, x1, width into the river
p.add_argument("--floor", type=float, default=-3.0)                           # the basin's floor
p.add_argument("--head", type=float, default=4.0)                             # the water's level above the street
p.add_argument("--gate", type=float, nargs="+", default=[760.0, 790.0])       # the openings in the flood wall: x0 x1 each
p.add_argument("--parapet", type=float, nargs=3)                              # x0 x1 height: a low wall along the bank
p.add_argument("--wall", type=float, default=2.0)                             # thickness of the walls
p.add_argument("--clip", type=float, default=0.0)
p.add_argument("--bankwall", type=float, nargs="+", action="append", default=[])
p.add_argument("--region", type=float, nargs=4)                               # x0 x1 z0 z1: only the solids in this rectangle
a = p.parse_args()
h = a.voxel
x0, x1, width = a.basin
gates = list(zip(a.gate[0::2], a.gate[1::2]))
T = a.wall
bank, base, top = 815.0, 21.0, a.head + 1.5  # the walls stand 1.5 m over the water
L, W, H = 1911.0, 1764.0, 124.0

with open(a.city) as f:
    lines = f.read().split("\n")
nv = sum(1 for line in lines if line.startswith("v "))
v = np.loadtxt(lines[:nv], usecols=(1, 2, 3), dtype=np.float64)
t = np.loadtxt(lines[nv:nv + sum(1 for line in lines[nv:] if line.startswith("f "))], usecols=(1, 2, 3), dtype=np.int64) - 1
del lines
print(f"city: {len(v)} points, {len(t)} triangles")
if a.region:
    rx0, rx1, rz0, rz1 = a.region
    c = v[t].mean(axis=1)
    t = t[(c[:, 0] > rx0) & (c[:, 0] < rx1) & (c[:, 2] > rz0) & (c[:, 2] < rz1)]
    print(f"region x {rx0:g}..{rx1:g}, z {rz0:g}..{rz1:g}: {len(t)} triangles")

# The roofs: every level triangle more than 2 m over the street (found before the cut, which removes the high ones).
tri = v[t]
normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
size = np.linalg.norm(normal, axis=1)
roofs = tri[(np.abs(normal[:, 1]) > 0.999 * np.maximum(size, 1e-12)) & (tri[:, :, 1].min(axis=1) > 2.0) & (size > 1e-6)].copy()
if a.clip > 0:
    roofs[:, :, 1] = np.minimum(roofs[:, :, 1], a.clip)
del tri, normal, size

if a.clip > 0:  # keep the part of every triangle with y <= clip: a triangle, or a quad as two triangles
    y = v[t][:, :, 1]
    below = y <= a.clip
    count = below.sum(axis=1)
    keep = [t[count == 3]]
    extra = []
    for tri, inside in zip(t[(count > 0) & (count < 3)], below[(count > 0) & (count < 3)]):
        pts = v[tri]
        poly = []
        for i in range(3):  # walk the edges; cross the plane where the two ends are on different sides
            j = (i + 1) % 3
            if inside[i]:
                poly.append(pts[i])
            if inside[i] != inside[j]:
                s = (a.clip - pts[i][1]) / (pts[j][1] - pts[i][1])
                poly.append(pts[i] + s * (pts[j] - pts[i]))
        first = len(v) + len(extra)
        extra += poly
        keep.append(np.array([(first, first + k, first + k + 1) for k in range(1, len(poly) - 1)], np.int64))
    if extra:
        v = np.vstack([v, np.array(extra)])
    t = np.vstack(keep)
    print(f"cut at {a.clip:g} m: {len(t)} triangles")

points, tris = [v], [t]
count = len(v)

def add(pts, tri):
    global count
    points.append(np.asarray(pts, np.float64))
    tris.append(np.asarray(tri, np.int64) + count)
    count += len(pts)

def prisms(top, bottom):
    """Closed prisms from the triangles `top` (n x 3 x 3, any orientation) straight down to the height `bottom`."""
    n = len(top)
    low = top.copy()
    low[:, :, 1] = bottom
    pts = np.concatenate([top, low], axis=1).reshape(-1, 3)  # per prism: top 0 1 2, bottom 3 4 5
    one = np.array([(0, 1, 2), (3, 5, 4), (0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)], np.int64)
    add(pts, (one[None] + 6 * np.arange(n)[:, None, None]).reshape(-1, 3))

tri = v[t]
normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
size = np.linalg.norm(normal, axis=1)
level = np.abs(normal[:, 1]) > 0.999 * np.maximum(size, 1e-12)
# The buildings: a prism under every roof triangle.
prisms(roofs, -0.35)
# The ground: the river's banks are the upright triangles that reach from the bed to the street; their upper corners
# give the two bank lines z(x), the north one (smaller z) and the south one.
upright = (np.abs(normal[:, 1]) < 0.05 * np.maximum(size, 1e-12)) & (tri[:, :, 1].min(axis=1) < -19.0) & (tri[:, :, 1].max(axis=1) > -0.5)
corners = tri[upright].reshape(-1, 3)
corners = corners[corners[:, 1] > -0.5]
order = np.argsort(corners[:, 0])
cx, cz = corners[order, 0], corners[order, 2]
# A corner is on the north bank if the river's bed (the level triangles at -20 m) lies 3 m to its +z side.
bed = tri[level & (tri[:, :, 1].max(axis=1) < -19.5)][:, :, [0, 2]]
def over_bed(px, pz):
    a, b, c = bed[None, :, 0], bed[None, :, 1], bed[None, :, 2]
    q = np.stack([px, pz], axis=1)[:, None, :]
    cross = lambda u, w: u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]
    d1, d2, d3 = cross(b - a, q - a), cross(c - b, q - b), cross(a - c, q - c)
    return (((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))).any(axis=1)
north = over_bed(cx, cz + 3.0)
south = over_bed(cx, cz - 3.0) & ~north
cx, cz, north = cx[north | south], cz[north | south], north[north | south]
stations = np.unique(np.concatenate([np.arange(-2.0, L + 4.0, 2.0), cx]))
far_north, far_south = -2.0, W + 2.0
if a.region:
    stations = np.unique(np.concatenate([stations[(stations > rx0) & (stations < rx1)], [rx0, rx1]]))
    far_north, far_south = rz0, rz1
def line(mask, pick):
    xs = np.round(cx[mask] * 2) / 2
    keys = np.unique(xs)
    return np.interp(stations, keys, np.array([pick(cz[mask][xs == k]) for k in keys]))
zn, zs = line(north, np.max) + 0.3, line(~north, np.min) - 0.3   # 0.3 m into the river: the block backs the bank's sheet
def strip(za, zb, y):
    a = np.c_[stations, np.full(len(stations), y), za]
    b = np.c_[stations, np.full(len(stations), y), zb]
    return np.concatenate([np.stack([a[:-1], b[:-1], b[1:]], axis=1), np.stack([a[:-1], b[1:], a[1:]], axis=1)])
def block(za, zb, top, bottom):
    """One closed surface between the lines za(x) and zb(x), from `bottom` to `top`: no faces inside it. Its top and
    bottom are triangles 2 m wide and hundreds of meters long; MeshToSDF signs them correctly only with the
    cancellation-free closest point on a triangle (nanovdb/math/Proximity.h of this branch)."""
    upper, lower = strip(za, zb, top), strip(za, zb, bottom)
    add(upper.reshape(-1, 3), np.arange(3 * len(upper)).reshape(-1, 3))
    add(lower.reshape(-1, 3), np.arange(3 * len(lower)).reshape(-1, 3)[:, ::-1])
    ring = np.concatenate([np.c_[stations, za], np.c_[stations, zb][::-1]])   # the outline, in (x, z)
    walls = [[(xa, top, z0), (xb, top, z1), (xb, bottom, z1), (xa, bottom, z0)] for (xa, z0), (xb, z1) in zip(ring, np.roll(ring, -1, axis=0))]
    add(np.asarray(walls).reshape(-1, 3), (np.array([(0, 1, 2), (0, 2, 3)])[None] + 4 * np.arange(len(walls))[:, None, None]).reshape(-1, 3))
block(np.full(len(stations), far_north), np.maximum(zn, far_north + 0.1), -0.35, -base + 0.3)
block(np.minimum(zs, far_south - 0.1), np.full(len(stations), far_south), -0.35, -base + 0.3)
print(f"added: {len(roofs)} roof prisms, ground blocks along {len(stations)} stations (north bank z {zn.min():.0f}..{zn.max():.0f}, south bank {zs.min():.0f}..{zs.max():.0f})")
del tri, normal

drawn = []  # the boxes a renderer draws

def box(lo, hi, draw=True):
    global count
    if draw:
        drawn.append(tuple(lo) + tuple(hi))
    (xa, ya, za), (xb, yb, zb) = lo, hi
    pts = [(x, y, z) for x in (xa, xb) for y in (ya, yb) for z in (za, zb)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    points.append(np.asarray(pts, np.float64))
    tris.append(np.asarray([q[:3] for q in quads] + [(q[0], q[2], q[3]) for q in quads], np.int64) + count)
    count += 8

far = bank + width
box((x0 - T, -base + 0.5, bank - 1.0), (x1 + T, a.floor, far + T))   # the raised floor, a solid block down to the bed
box((x0 - T, a.floor - 0.5, bank - 1.0), (x0, top, far + T))         # west wall
box((x1, a.floor - 0.5, bank - 1.0), (x1 + T, top, far + T))         # east wall
box((x0 - T, a.floor - 0.5, far), (x1 + T, top, far + T))            # the wall on the river's side
ends = [x0 - T] + [x for gate in gates for x in gate] + [x1 + T]
for west, east in zip(ends[0::2], ends[1::2]):                       # the flood wall on the bank, between the openings
    box((west, -1.0, bank - T), (east, top, bank + 0.5))
if a.parapet:
    box((a.parapet[0], -1.0, bank - T), (x0 - T, a.parapet[2], bank + 0.5))
    box((x1 + T, -1.0, bank - T), (a.parapet[1], a.parapet[2], bank + 0.5))
wall_points, wall_tris = [], []
for spec in a.bankwall:
    height, line = spec[0], np.asarray(spec[1:], np.float64).reshape(-1, 2)
    for (xa, za), (xb, zb) in zip(line[:-1], line[1:]):
        along = np.array([xb - xa, zb - za])
        length = np.linalg.norm(along)
        along /= length
        side = np.array([-along[1], along[0]])                       # (along, up, side) is right-handed, as (x, y, z)
        ends = [np.array([xa, za]) - 0.5 * T * along, np.array([xb, zb]) + 0.5 * T * along]   # the pieces overlap at the corners
        pts = [(e[0] + w * side[0], y, e[1] + w * side[1]) for e in ends for y in (-1.0, height) for w in (-0.5 * T, 0.5 * T)]
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        piece = np.asarray([q[:3] for q in quads] + [(q[0], q[2], q[3]) for q in quads], np.int64)
        wall_tris.append(piece + 8 * len(wall_points))
        wall_points.append(np.asarray(pts, np.float64))
        add(pts, piece)
if wall_points:
    with open(a.out + ".bankwalls.obj", "w") as f:
        np.savetxt(f, np.vstack(wall_points), fmt="v %.5f %.5f %.5f")
        np.savetxt(f, np.vstack(wall_tris) + 1, fmt="f %d %d %d")
    print(f"bank walls: {len(wall_points)} pieces")

# gen_sdf takes the mesh's largest extent divided by its resolution as the voxel size: two specks at the tank's floor make
# the extent along x a whole number of voxels, so that the voxel size is h exactly.
low = np.vstack(points).min(axis=0)
cells = int(np.ceil((np.vstack(points)[:, 0].max() - np.floor(low[0] / h) * h) / h))
for x in (np.floor(low[0] / h) * h, np.floor(low[0] / h) * h + cells * h):
    points.append(np.array([(x, -base + 0.2, 1.0), (x, -base + 0.21, 1.0), (x, -base + 0.2, 1.01)]))
    tris.append(np.array([(0, 1, 2)], np.int64) + count)
    count += 3

points = np.vstack(points) + np.array([0.0, base, 0.0]) - h / 2
tris = np.vstack(tris)
with open(a.out, "w") as f:
    np.savetxt(f, points, fmt="v %.5f %.5f %.5f")
    np.savetxt(f, tris + 1, fmt="f %d %d %d")
extent = points.max(axis=0) - points.min(axis=0)
nx, ny, nz = int(np.ceil(L / h)), int(np.ceil((H + base) / h)), int(np.ceil(W / h))
print(f"mesh: {len(points)} points, {len(tris)} triangles, extent {extent[0]:.3f} {extent[1]:.3f} {extent[2]:.3f} m; "
      f"gen_sdf resolution for this voxel: {extent.max() / h:.2f}")
print(f"tank {nx} {ny} {nz} cells")
# The basin's water as a box in cell-center coordinates (lo, hi), one cell inside the walls around it and starting
# below the floor's surface (the part inside the floor is solid and holds no water).
lo = (x0 / h + 1.0, (a.floor + base) / h - 2.0, (bank + 0.5) / h + 1.0)
hi = (x1 / h - 2.0, (a.head + base) / h - 0.5, far / h - 2.0)
print("BOX=" + ",".join(f"{q:.2f}" for q in lo + hi))
print("RENDER_BOXES=" + "  ".join(" ".join(f"{q:g}" for q in b) for b in drawn))
volume = (x1 - x0) * width * (a.head - a.floor)
print(f"water {volume:.0f} m^3 = {volume / h ** 3:.0f} cells = {100.0 * volume / h ** 3 / (nx * ny * nz):.4f}% of the tank's {nx * ny * nz} cells; "
      f"openings {sum(e - w for w, e in gates):g} m wide in all, head {a.head:g} m over the street")
