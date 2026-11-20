# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Check packed edge incidence against the original row-wise mesh audit."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from render_scene import read_obj
from validate_scene import edge_incidence


def compare(name, faces):
    start = time.perf_counter()
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    expected = np.unique(edges, axis=0, return_counts=True)[1]
    old_seconds = time.perf_counter() - start
    start = time.perf_counter()
    actual = edge_incidence(faces)
    new_seconds = time.perf_counter() - start
    assert np.array_equal(expected, actual), name
    return dict(case=name, triangles=len(faces), exactly_equal=True,
                non_two_face_edges=int(np.sum(actual != 2)),
                original_seconds=old_seconds, packed_seconds=new_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 90, 180])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "empty": np.empty((0, 3), dtype=np.int64),
        "open_triangle": np.array([[0, 1, 2]], dtype=np.int64),
        "non_manifold_shared_edge": np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64),
        "wide_indices": np.array([[0, 2**31, 2**32-1], [2**32-1, 2**31, 1]], dtype=np.int64),
    }
    results = [compare(name, faces) for name, faces in cases.items()]
    for frame in args.frames:
        _, faces = read_obj(args.cache / "surface" / f"water_{frame:06d}.obj", prefer_binary=True)
        results.append(compare(f"frame_{frame:06d}", faces))
    result = dict(pass_all=True, cases=results)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
