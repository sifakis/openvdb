# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Resume render reconstruction without publishing a cold temporal filter.

The only temporal state is covariance = .7 * current + .3 * previous.
Replay 32 preceding frames, then independently compare the overlapping mesh and
rendered whitewater before publishing any new frame. Simulation data is read-only.
"""

import csv
import io
import json
import os
from pathlib import Path
import struct
import tempfile


def read_report(path):
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    # A producer can be between buffered writes in the middle of a CSV record.
    text = text[:text.rfind("\n") + 1]
    rows = list(csv.DictReader(io.StringIO(text)))
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError(f"Malformed committed meshing report: {path}")
    return rows


def read_mesh(path):
    import numpy as np
    with path.open("rb") as stream:
        magic, nv, nf = struct.unpack("<3Q", stream.read(24))
        if magic != 0x4D45534830303031 or path.stat().st_size != 24 + 12 * (nv + nf):
            raise RuntimeError(f"Incomplete mesh: {path}")
        vertices = np.fromfile(stream, dtype="<f4", count=3 * nv)
        faces = np.fromfile(stream, dtype="<i4", count=3 * nf)
    return vertices, faces


def compare_overlap(reference, candidate, frame):
    import numpy as np
    name = f"water_{frame:06d}.meshbin"
    av, af = read_mesh(reference / name)
    bv, bf = read_mesh(candidate / name)
    if av.shape != bv.shape or not np.array_equal(af, bf):
        raise RuntimeError("Warm reconstruction changed overlap mesh connectivity")
    error = float(np.max(np.abs(av - bv), initial=0))
    if not np.isfinite(error) or error > 1e-6:
        raise RuntimeError(f"Warm reconstruction coordinate error {error} exceeds 1 micrometre")
    name = f"whitewater_{frame:06d}.csv"
    with (reference / name).open() as stream:
        a = list(csv.DictReader(stream))
    with (candidate / name).open() as stream:
        b = list(csv.DictReader(stream))
    if len(a) != len(b):
        raise RuntimeError("Warm reconstruction changed whitewater count")
    foam_error = 0.0
    for x, y in zip(a, b):
        if x.keys() != y.keys() or any(x[k] != y[k] for k in x if k not in ("x", "y", "z")):
            raise RuntimeError("Warm reconstruction changed whitewater identity or state")
        differences = [abs(float(x[k]) - float(y[k])) for k in ("x", "y", "z")]
        if not all(np.isfinite(delta) for delta in differences):
            raise RuntimeError("Nonfinite warm whitewater position")
        foam_error = max(foam_error, *differences)
    if foam_error > 1e-6:
        raise RuntimeError(f"Warm whitewater error {foam_error} exceeds 1 micrometre")
    return {"frame": frame, "mesh_coordinate_error_m": error,
            "whitewater_coordinate_error_m": foam_error, "threshold_m": 1e-6,
            "mesh_connectivity_equal": True, "whitewater_identity_equal": True}


class SurfaceContinuation:
    def __init__(self, folder, next_simulation_frame):
        self.surface = folder / "surface"
        self.rows = [row for row in read_report(self.surface / "meshing.csv")
                     if int(row["frame"]) < next_simulation_frame]
        if not self.rows or [int(row["frame"]) for row in self.rows] != list(range(len(self.rows))):
            raise RuntimeError("Surface resume requires a contiguous committed prefix starting at frame 0")
        self.last = len(self.rows) - 1
        self.first = max(0, self.last - 32)
        self.output = Path(tempfile.mkdtemp(prefix="surface_resume_", dir=folder))
        self.verified = False
        self.last_published = self.last
        self._write_report(self.output / "prefix.csv")
        self._write_report(self.surface / "meshing.csv")

    def _write_report(self, path):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(path)

    def poll(self):
        rows = read_report(self.output / "meshing.csv")
        if [int(row["frame"]) for row in rows] != list(range(self.first, self.first + len(rows))):
            raise RuntimeError("Noncontiguous warm reconstruction output")
        if not self.verified:
            overlap = next((row for row in rows if int(row["frame"]) == self.last), None)
            if overlap is None:
                return
            result = compare_overlap(self.surface, self.output, self.last)
            for key in ("material_volume", "kernel_integral", "mesh_volume"):
                a, b = float(self.rows[-1][key]), float(overlap[key])
                if not abs(a - b) <= 1e-8 * max(abs(a), 1e-12):
                    raise RuntimeError(f"Warm reconstruction changed {key}")
            (self.output / "overlap_validation.json").write_text(json.dumps(result, indent=2) + "\n")
            self.verified = True
        pending = [row for row in rows if int(row["frame"]) > self.last_published]
        for row in pending:
            frame = int(row["frame"])
            if frame != self.last_published + 1 or row.keys() != self.rows[0].keys():
                raise RuntimeError("Invalid surface continuation frame")
            for name in (f"water_{frame:06d}.obj", f"water_{frame:06d}.meshbin",
                         f"whitewater_{frame:06d}.csv"):
                target = self.surface / name
                temporary = self.surface / (name + "." + self.output.name + ".publish")
                os.link(self.output / name, temporary)
                temporary.replace(target)
            self.rows.append(row)
            self.last_published = frame
        if pending:
            self._write_report(self.surface / "meshing.csv")

    def warmup_seconds(self):
        return sum(sum(float(row[key]) for key in
            ("read_ms", "pca_ms", "grid_ms", "clip_ms", "filter_ms", "contour_ms", "write_ms"))
            for row in read_report(self.output / "meshing.csv") if int(row["frame"]) <= self.last) / 1000
