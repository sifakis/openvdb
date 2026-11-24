#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""plan_hydra.py <mesh dir> <out dir>: where the heads of the hydra go.

Reads the nine snake meshes named in MODELS from <mesh dir> and writes into <out dir> what bin/hydra takes:
    scene.txt          the body, and per head its mesh, its cutter and the matrix that moves both onto the body
    cutter_<name>.obj  the cutter of a head, in the coordinates of that head's mesh file
    plan.json          the numbers behind every head (cut, scale, root), for the record
    distance_<name>.npy, body_220.band   what the plan was computed from; a later plan in the same directory reuses the distances

A head is found on its mesh as follows. The two ends of the snake are the two vertices farthest apart along the
surface; the head end is the one that is higher and further forward (the meshes stand with y up and face +z). With
g the distance along the surface from the head end, the head with its neck is the part g <= NECK of the mesh, and
the cut is the ring g = NECK. The cutter is the convex hull of that part, scaled by 1.04 about its centroid.

A head is placed on the body as follows. A head keeps the orientation it has in its own mesh file, upright and
facing +z like the body, except for a roll about the facing direction, which leans it to one side, and for some a
turn about the vertical. MODELS gives, per head, these angles and where the middle of its cut goes:
  - a place on the body (three heads sit on the body's own neck behind its head, four on its chest, where the neck
    leaves the coil). The cut is put at the point nearest to that place among the points deep inside the body (from
    the body's level set at 220 voxels, written by bin/hydra --band; a point that is THICK less deep counts as that
    much further away). The room for the cut is the depth of that point.
  - or the neck of a head placed before: the middle of that neck's ring at a given distance along the surface from
    that head's end. The room for the cut is the distance from that middle to the nearest tenth of the ring.
The head is scaled so that the radius of its cut (the mean distance of the cut's ring from its middle) is the room
minus MARGIN, times the size given in MODELS, and at most SCALE_MAX. A sphere of that
radius about the middle of the cut is then inside the body or the other neck, so a round cut is hidden however
it is turned. A cut that is not round can still show a little; bin/hydra counts the voxels where one does.

The numbers in MODELS were chosen by hand for these nine meshes, so that the nine heads face the front and show
from there.
Needs numpy and scipy, and bin/hydra (./build.sh).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull

BODY = "basilisk_var_5.obj"
# mesh file -> (name,
#               NECK: length of the head with its neck along the surface, in units of the mesh file,
#               where the middle of the cut goes: a place (x, y, z) in the coordinates of the body's mesh file, or
#                   ("on", name of a head listed before, distance along that head's surface from its head end),
#               roll in degrees about the facing direction +z (positive: the head leans toward -x),
#               turn in degrees about the vertical y (positive: the face turns toward +x),
#               size: the fraction of the largest scale that hides the cut)
MODELS = {
    "basilisk_var_1.obj": ("1", 1.8, (-0.20, 0.45, 0.18), 55, 0, 1.0),
    "basilisk_var_2.obj": ("2", 2.0, (0.10, -0.40, 0.68), 35, 0, 1.0),
    "basilisk_var_3.obj": ("3", 1.7, (-0.18, 0.56, 0.20), 35, 0, 1.0),
    "basilisk_var_4.obj": ("4", 1.8, (0.30, -0.15, 0.50), -45, 0, 1.0),
    "basilisk_var_6.obj": ("6", 2.0, (-0.20, -0.40, 0.66), 60, 0, 1.0),
    "basilisk_var_8.obj": ("8", 1.3, (-0.17, 0.58, 0.20), -12, 0, 1.0),
    "basilisk_var_9.obj": ("9", 2.0, (0.28, -0.10, 0.47), -25, 0, 1.0),
    "basilisk_var_7.obj": ("7", 1.4, ("on", "9", 1.5), -5, -12, 1.0),
}
SCALE_MAX = 1.25     # largest scale of a head
MARGIN = 0.025       # the radius of a scaled cut is this much less than the room for it
THICK = 3.0          # in the choice of the place of a cut: a point less deep than 0.18 counts as THICK times the difference further away
HEAD = 0.5           # the part g < HEAD of a mesh counts as its head
PLAN_EXTENT = 220    # voxels along the body's longest extent in the level set the cuts are placed in


def load_obj(path):
    points, faces = [], []
    with open(path) as stream:
        for line in stream:
            if line.startswith("v "):
                points.append(line.split()[1:4])
            elif line.startswith("f "):
                corners = [int(token.split("/")[0]) for token in line.split()[1:]]
                corners = [c + len(points) if c < 0 else c - 1 for c in corners]
                faces.extend([corners[0], corners[k - 1], corners[k]] for k in range(2, len(corners)))
    return np.array(points, np.float64), np.array(faces, np.int64)


def surface_distance(points, faces):
    """g: for every vertex of the largest piece of the mesh, the distance along the edges from the head end
    (infinite on the other pieces); and the index of the head end."""
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0)
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    n = len(points)
    graph = coo_matrix((np.r_[length, length], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    piece = connected_components(graph, directed=False)[1]
    largest = piece == np.bincount(piece).argmax()

    def distance(source):
        d = dijkstra(graph, directed=False, indices=source)
        d[~largest] = np.inf
        return d

    def farthest(d):
        return int(np.where(np.isfinite(d), d, -1).argmax())

    a = farthest(distance(int(np.flatnonzero(largest)[0])))
    from_a = distance(a)
    b = farthest(from_a)
    if points[a, 1] + points[a, 2] > points[b, 1] + points[b, 2]:
        return from_a, a
    return distance(b), b


def body_depth(band_path, points):
    """For the voxels of the body's level set: how far inside the body they are, in units of the mesh file; with the
    grid's origin and voxel size, such that voxel index (i, j, k) is at origin + (i, j, k) * voxel."""
    raw = band_path.read_bytes()
    record = np.frombuffer(raw, np.dtype([("c", "<i4", 3), ("v", "<f4")]), int(np.frombuffer(raw, "<u8", 1)[0]), 8)
    low = record["c"].min(0) - 2
    shape = record["c"].max(0) - low + 3
    band, negative = np.zeros(shape, bool), np.zeros(shape, bool)
    index = tuple((record["c"] - low).T)
    band[index] = True
    negative[index] = record["v"] < 0
    label = ndimage.label(~band)[0]          # the voxels outside the band: one piece is the outside, the rest lies inside
    inside = negative | (~band & (label != label[0, 0, 0]))
    extent = (points.max(0) - points.min(0)).max()
    voxel = extent / PLAN_EXTENT
    return ndimage.distance_transform_edt(inside) * voxel, 0.5 * (points.max(0) + points.min(0)) + low * voxel, voxel


def main():
    meshes, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    hydra = Path(__file__).resolve().parent / "bin" / "hydra"

    body_points, _ = load_obj(meshes / BODY)
    band = out / f"body_{PLAN_EXTENT}.band"
    subprocess.run([str(hydra), "--band", str(meshes / BODY), str(PLAN_EXTENT), str(band)], check=True, stdout=subprocess.DEVNULL)
    depth, origin, voxel = body_depth(band, body_points)
    cells = np.argwhere(depth >= 0.09)
    places, deep = cells * voxel + origin, depth[tuple(cells.T)]

    scene, plan, placed = [f"body {(meshes / BODY).resolve()}"], {}, {}
    for file, (name, neck, wanted, roll, turn, size) in MODELS.items():
        points, faces = load_obj(meshes / file)
        cache = out / f"distance_{name}.npy"     # the distances along the surface take minutes; they are kept for the next plan
        if cache.exists():
            g = np.load(cache)
        else:
            g, _ = surface_distance(points, faces)
            np.save(cache, g)
        part = g <= neck
        ring = np.abs(g - neck) < 0.02
        cut = points[ring].mean(0)
        cut_radius = np.linalg.norm(points[ring] - cut, axis=1).mean()
        head = points[g < HEAD].mean(0)

        hull = ConvexHull(points[part])
        centroid = points[part].mean(0)
        corners = centroid + 1.04 * (hull.points[hull.vertices] - centroid)
        renumber = -np.ones(len(hull.points), np.int64)
        renumber[hull.vertices] = np.arange(len(hull.vertices))
        triangles = renumber[hull.simplices]
        normal = np.cross(corners[triangles[:, 1]] - corners[triangles[:, 0]], corners[triangles[:, 2]] - corners[triangles[:, 0]])
        flip = np.einsum("ij,ij->i", normal, hull.equations[:, :3]) < 0     # the triangles face outward
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        with open(out / f"cutter_{name}.obj", "w") as stream:
            np.savetxt(stream, corners, fmt="v %.9g %.9g %.9g")
            np.savetxt(stream, triangles + 1, fmt="f %d %d %d")
        # What the cutter holds of the mesh beyond the cut (0 when it holds the head and the neck only).
        beyond = centroid + (points[np.isfinite(g) & (g > neck + 0.15)] - centroid) / 1.04
        captured = sum(int((chunk @ hull.equations[:, :3].T + hull.equations[:, 3] <= 0).all(1).sum()) for chunk in np.array_split(beyond, 1 + len(beyond) // 20000))

        if wanted[0] == "on":
            parent = placed[wanted[1]]
            parent_ring = parent["points"][np.abs(parent["g"] - wanted[2]) < 0.02]
            middle = parent_ring.mean(0)
            root = parent["matrix"][:, :3] @ middle + parent["matrix"][:, 3]
            room = parent["scale"] * np.percentile(np.linalg.norm(parent_ring - middle, axis=1), 10)
        else:
            at = (np.linalg.norm(places - np.array(wanted), axis=1) + THICK * (0.18 - deep)).argmin()
            root, room = places[at], deep[at]
        scale = min(SCALE_MAX, (room - MARGIN) / cut_radius) * size
        a, b = np.radians(roll), np.radians(turn)
        about_z = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        about_y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        rotation = about_y @ about_z
        matrix = np.c_[scale * rotation, root - scale * rotation @ cut]
        placed_head = matrix[:, :3] @ head + matrix[:, 3]
        placed[name] = dict(points=points, g=g, matrix=matrix, scale=scale)

        scene.append(f"head {name} {(meshes / file).resolve()} cutter_{name}.obj " + " ".join(f"{v:.17g}" for v in matrix.reshape(-1)))
        plan[name] = dict(mesh=file, vertices=len(points), triangles=len(faces), neck=neck, wanted=list(wanted), roll=roll, turn=turn, size=size, head=placed_head.tolist(), cut=cut.tolist(), room=float(room),
                          cut_radius=float(cut_radius), scale=float(scale), root=root.tolist(), cutter_triangles=len(triangles), captured_vertices=captured, matrix=matrix.tolist())
        print(f"head {name}: {int(part.sum())} of {len(points)} vertices, cut radius {cut_radius:.3f}, scale {scale:.2f}, "
              f"cut at ({root[0]:.2f}, {root[1]:.2f}, {root[2]:.2f}), room {room:.3f}, head at ({placed_head[0]:.2f}, {placed_head[1]:.2f}, {placed_head[2]:.2f}), cutter {len(triangles)} triangles holding {captured} vertices beyond the cut", flush=True)
    (out / "scene.txt").write_text("\n".join(scene) + "\n")
    (out / "plan.json").write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
