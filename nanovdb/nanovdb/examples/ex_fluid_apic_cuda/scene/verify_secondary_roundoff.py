# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Regression for a recorded float32 basin-boundary false positive."""
import argparse
import json
from pathlib import Path

import numpy as np
from verify_pool import secondary_arithmetic_bound_error


def main(folder):
    settings = json.loads((folder / "settings.json").read_text())
    evidence = json.loads((folder / "secondary_roundoff_diagnosis.json").read_text())
    row = evidence["worst_physical_plane_excess"]["row"]
    q = np.array([[row[1], row[2], row[3], row[7]]])
    checks = {"recorded_float_wall_position": secondary_arithmetic_bound_error(q, settings) <= 0}
    for name, axis, position in (("upper_side_1um", 0, q[0, 0] + 1e-6),
                                 ("lower_side_1um", 0, q[0, 3] - 1e-6),
                                 ("floor_1um", 1, -1e-6)):
        corrupt = q.copy()
        corrupt[0, axis] = position
        checks["reject_" + name] = secondary_arithmetic_bound_error(corrupt, settings) > 0
    double = settings | {"precision": "double"}
    checks["double_physical_plane_criterion_unchanged"] = secondary_arithmetic_bound_error(q, double) > 1e-7
    invalid = q.copy()
    invalid[0, 0] = np.nan
    try:
        secondary_arithmetic_bound_error(invalid, settings)
        checks["reject_nonfinite"] = False
    except AssertionError:
        checks["reject_nonfinite"] = True
    result = {"pass": all(checks.values()), "checks": checks,
              "recorded_frame": evidence["worst_physical_plane_excess"]["frame"],
              "recorded_particle_id": int(row[0]),
              "extra_float_tolerance_m": 0}
    (folder / "secondary_roundoff_regression.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["pass"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    main(p.parse_args().cache)
