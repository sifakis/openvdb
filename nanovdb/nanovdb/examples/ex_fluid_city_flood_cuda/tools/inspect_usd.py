# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""inspect_usd.py <file.usd>... [--obj <folder>]: what a USD asset holds as triangle meshes.

For every file: the number of mesh prims, the triangles after fan triangulation, the bounding box in meters (stage units
times metersPerUnit, all prims in world space), and how closed the surface is. Closedness is counted on the merged
triangles with points welded by position: an edge used by exactly two triangles is closed, by one is a boundary edge
(a hole or an open sheet), by more than two is non-manifold. A mesh for MeshToSDF's sign should have no boundary edges.
--obj <folder>: also writes each file's merged triangles as <folder>/<name>.obj, in meters, y up.

Run with python -I (the assets are downloaded files).
"""
import sys, os
import numpy as np
from pxr import Usd, UsdGeom


def triangles_of(stage, keep=None, time=None):
    """World-space points (in meters, y up) and triangle indices of every mesh prim that is rendered.
    keep: optional function of the prim path (a string); prims for which it returns False are left out.
    time: the time code of an animated stage (default: the values without a time)."""
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    up_z = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    when = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
    cache = UsdGeom.XformCache(when)
    points, faces, prims, offset = [], [], 0, 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):  # instanced prims too
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if keep and not keep(str(prim.GetPath())):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue  # guides and proxies (e.g. collision shapes) are not what is seen
        mesh = UsdGeom.Mesh(prim)
        p = mesh.GetPointsAttr().Get(when)
        counts = mesh.GetFaceVertexCountsAttr().Get(when)
        indices = mesh.GetFaceVertexIndicesAttr().Get(when)
        if not p or not counts or not indices:
            continue
        p = np.asarray(p, np.float64)
        m = np.asarray(cache.GetLocalToWorldTransform(prim), np.float64)  # row vectors: x' = x M
        p = (np.c_[p, np.ones(len(p))] @ m)[:, :3] * scale
        if up_z:
            p = np.c_[p[:, 0], p[:, 2], -p[:, 1]]
        counts = np.asarray(counts, np.int64)
        indices = np.asarray(indices, np.int64)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        for k in range(1, int(counts.max()) - 1):  # fan: (0, k, k + 1) of every face with more than k + 1 corners
            f = start[counts > k + 1]
            faces.append(np.c_[indices[f], indices[f + k], indices[f + k + 1]] + offset)
        points.append(p)
        offset += len(p)
        prims += 1
    if not prims:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), 0
    return np.vstack(points), np.vstack(faces), prims


def edge_counts(points, faces):
    """Welds points by position, then counts the triangles on every edge: (edges, boundary, non-manifold)."""
    _, weld = np.unique(np.round(points, 6), axis=0, return_inverse=True)
    f = weld.reshape(-1)[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]  # degenerate after the weld
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    _, n = np.unique(e, axis=0, return_counts=True)
    return len(n), int((n == 1).sum()), int((n > 2).sum())


def main():
    args = sys.argv[1:]
    out = None
    if "--obj" in args:
        i = args.index("--obj")
        out = args[i + 1]
        del args[i:i + 2]
        os.makedirs(out, exist_ok=True)
    print("meshes  triangles   size x, y, z (m)        edges  boundary  non-manifold  file")
    for path in args:
        stage = Usd.Stage.Open(path)
        if not stage:
            print("cannot open", path)
            continue
        points, faces, prims = triangles_of(stage)
        if not len(faces):
            print(f"{prims:6d}  no triangles  {path}")
            continue
        size = points.max(axis=0) - points.min(axis=0)
        edges, boundary, bad = edge_counts(points, faces)
        print(f"{prims:6d}  {len(faces):9d}   {size[0]:6.2f} {size[1]:6.2f} {size[2]:6.2f}   {edges:9d}  {boundary:8d} "
              f"({100.0 * boundary / edges:4.1f}%)  {bad:6d}  {path}")
        if out:
            name = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(out, name + ".obj"), "w") as f:
                for p in points:
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                for t in faces + 1:
                    f.write(f"f {t[0]} {t[1]} {t[2]}\n")


if __name__ == "__main__":
    main()
