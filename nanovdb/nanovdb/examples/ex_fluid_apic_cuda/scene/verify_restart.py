# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Compare a continuous run with restart, including raw APIC affine matrices."""

import argparse, csv, hashlib, json, struct, subprocess
from pathlib import Path
import numpy as np
from pipeline import BINS, HERE

p = argparse.ArgumentParser()
p.add_argument("--character", type=Path, default=HERE.parent / "output/character")
p.add_argument("--output", type=Path, default=HERE.parent / "output/scene_restart_test")
p.add_argument("--reuse", action="store_true")
p.add_argument("--drain-enabled", type=int, choices=(0, 1), default=0)
p.add_argument("--initial-depth", type=float, default=0)
p.add_argument("--foam-radius-scale", type=float, default=1)
p.add_argument("--foam-life-scale", type=float, default=1)
args = p.parse_args()
scene_args = [
    "--initial-depth",
    str(args.initial_depth),
    "--foam-radius-scale",
    str(args.foam_radius_scale),
    "--foam-life-scale",
    str(args.foam_life_scale),
]
if not args.reuse:
    for name, duration, resume in [
        ("continuous", 0.6, False),
        ("resumed", 0.4, False),
        ("resumed", 0.6, True),
        ("repeat", 0.6, False),
    ]:
        out = args.output / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(BINS / "ex_fluid_apic_scene_cuda"),
            "--character",
            str(args.character),
            "--output",
            str(out),
            "--h",
            ".08",
            "--duration",
            str(duration),
            "--drain-enabled",
            str(args.drain_enabled),
        ]
        cmd += scene_args
        if resume:
            cmd += ["--resume", "1"]
        with (out / ("resume.log" if resume else "run.log")).open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def state(folder):
    data = (folder / "checkpoint.bin").read_bytes()
    magic = struct.unpack_from("<Q", data)[0]
    format = {0x464C554944303034: "<10Q9d2iQ4d2Q", 0x464C554944303033: "<10Q9d2iQ"}.get(
        magic, "<10Q9d2i"
    )
    header = struct.unpack_from(format, data)
    assert (
        header[0] in (0x464C554944303032, 0x464C554944303033, 0x464C554944303034)
        and header[1] == 120
        and header[2] == 88
    )
    offset = struct.calcsize(format)
    count = header[3]
    particles = np.frombuffer(data, dtype="<f8", offset=offset, count=count * 15).reshape(-1, 15)
    ids = np.frombuffer(data, dtype="<u8", offset=offset + count * 120, count=count)
    return particles, ids, header


a, idsA, ha = state(args.output / "continuous")
b, idsB, hb = state(args.output / "resumed")
assert a.shape == b.shape and np.array_equal(idsA, idsB)
error = float(abs(a - b).max())
assert error < 2e-8
x = list(csv.DictReader(open(args.output / "continuous/frame_000018_whitewater.csv")))
y = list(csv.DictReader(open(args.output / "resumed/frame_000018_whitewater.csv")))
# Atomic birth-buffer append order is not part of the physical state.
x.sort(key=lambda q: int(q["id"]))
y.sort(key=lambda q: int(q["id"]))
assert [q["id"] for q in x] == [q["id"] for q in y]
wwerror = max(
    [
        abs(float(a[k]) - float(b[k]))
        for a, b in zip(x, y)
        for k in ["x", "y", "z", "vx", "vy", "vz", "radius", "age", "life", "kind"]
    ]
    + [0]
)
repeat_rows = list(csv.DictReader(open(args.output / "repeat/frame_000018_whitewater.csv")))
repeat_rows.sort(key=lambda q: int(q["id"]))
assert [q["id"] for q in x] == [q["id"] for q in repeat_rows]
repeat_error = max(
    [
        abs(float(a[k]) - float(b[k]))
        for a, b in zip(x, repeat_rows)
        for k in ["x", "y", "z", "vx", "vy", "vz", "radius", "age", "life", "kind"]
    ]
    + [0]
)
position_error = max(
    [abs(float(a[k]) - float(b[k])) for a, b in zip(x, y) for k in ["x", "y", "z"]] + [0]
)
velocity_error = max(
    [abs(float(a[k]) - float(b[k])) for a, b in zip(x, y) for k in ["vx", "vy", "vz"]] + [0]
)
lifetime_error = max(
    [
        abs(float(a[k]) - float(b[k]))
        for a, b in zip(x, y)
        for k in ["radius", "age", "life", "kind"]
    ]
    + [0]
)
# P2G floating-point atomics plus nonlinear surface projection are not bitwise
# deterministic. Compare against a continuous repeat, with independent physical
# caps: 1e-4 voxel position error and 1e-5 m/s velocity error. IDs/lifetimes must agree.
assert position_error < 0.08e-4 and velocity_error < 1e-5 and lifetime_error < 2e-8
assert wwerror <= max(2e-8, 2 * repeat_error)
out = args.output / "resumed"
paths = [out / "checkpoint.bin", out / "metrics.csv", out / "settings.json"]
before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
rejected = subprocess.run(
    [
        str(BINS / "ex_fluid_apic_scene_cuda"),
        "--character",
        str(args.character),
        "--output",
        str(out),
        "--h",
        ".08",
        "--duration",
        ".7",
        "--resume",
        "1",
        "--drain-enabled",
        str(1 - args.drain_enabled),
    ]
    + scene_args,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
assert rejected.returncode != 0 and "checkpoint" in rejected.stdout.lower()
assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
(args.output / "rejected_mode.log").write_text(rejected.stdout)
result = {
    "pass": True,
    "restart_time": 0.4,
    "end_time": 0.6,
    "raw_particle_x_v_C_max_difference": error,
    "whitewater_max_difference": wwerror,
    "whitewater_position_error_m": position_error,
    "whitewater_velocity_error_m_s": velocity_error,
    "whitewater_lifetime_radius_error": lifetime_error,
    "continuous_repeat_whitewater_difference": repeat_error,
    "whitewater_position_threshold_m": 0.08e-4,
    "whitewater_velocity_threshold_m_s": 1e-5,
    "threshold": 2e-8,
    "fluid_particles": len(a),
    "secondary_particles": len(x),
    "changed_drain_mode_rejected_without_cache_mutation": True,
    "note": "Secondary append ordering is ignored; persistent IDs and states are compared.",
}
(args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
