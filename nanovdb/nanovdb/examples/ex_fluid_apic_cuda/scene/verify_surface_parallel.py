# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Compare serial and parallel PCA reconstruction of the same frame interval."""

import argparse
import json
from pathlib import Path

from surface_resume import compare_overlap, read_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = read_report(args.reference / "meshing.csv")
    candidate = read_report(args.candidate / "meshing.csv")
    if not reference or [r["frame"] for r in reference] != [r["frame"] for r in candidate]:
        raise RuntimeError("Both completed reconstructions must cover the same nonempty interval")
    checks = []
    totals = {"reference_seconds": 0.0, "candidate_seconds": 0.0}
    for a, b in zip(reference, candidate):
        if a.keys() != b.keys():
            raise RuntimeError("Meshing report columns differ")
        result = compare_overlap(args.reference, args.candidate, int(a["frame"]))
        # Independent-particle scheduling must not alter scalar accumulation,
        # volume, clipping, smoothing or temporal diagnostics.
        for key in a:
            if not key.endswith("_ms") and a[key] != b[key]:
                raise RuntimeError(f"Frame {a['frame']} changed {key}: {a[key]} -> {b[key]}")
        for row, key in ((a, "reference_seconds"), (b, "candidate_seconds")):
            totals[key] += sum(float(v) for k, v in row.items() if k.endswith("_ms")) / 1000
        checks.append(result)
    result = {"passed": True, "frames": len(checks), **totals,
              "full_stage_speedup": totals["reference_seconds"] / totals["candidate_seconds"],
              "checks": checks}
    (args.candidate / "parallel_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
