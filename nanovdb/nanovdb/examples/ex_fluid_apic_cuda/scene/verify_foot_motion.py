# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Compare foot-driven motion before any emitted particle reaches the pool."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/foot_diagnosis"
rows = []
for name, character in [
    ("scene_foot_original_test", "character"),
    ("scene_foot_grounded_test_limited", "character_grounded"),
]:
    cache = ROOT / "output" / name
    assert json.loads((cache / "validation.json").read_text())["pass"]
    settings = json.loads((cache / "settings.json").read_text())
    vertices = np.load(ROOT / "output" / character / "vertices.npy", mmap_mode="r")
    feet = vertices[0, :, 1] < 0.17
    for f in csv.DictReader((cache / "frames.csv").open()):
        time = float(f["time"])
        if time > 0.65:
            continue
        frame = int(f["frame"])
        q = np.loadtxt(cache / f"frame_{frame:06d}.csv", delimiter=",", skiprows=1)
        # This control has one nozzle particle and no subsequent births.
        assert len(q[q[:, 0] == 0]) == 1 and q[q[:, 0] == 0, 2].min() > 0.25
        q = q[(q[:, 0] >= 1) & (q[:, 0] < 1 + settings["initial_pool_particles"])]
        center = vertices[frame, feet, :][:, [0, 2]].mean(axis=0)
        local = q[(q[:, 1] - center[0]) ** 2 + (q[:, 3] - center[1]) ** 2 < 0.45**2]
        speed2 = np.sum(local[:, 4:7] ** 2, axis=1)
        rows.append(
            dict(
                scene=name,
                frame=frame,
                time=time,
                near_foot_rms_m_s=float(np.sqrt(speed2.mean())),
                specific_kinetic_energy_J_kg=float(0.5 * speed2.mean()),
                height_p999_m=float(np.quantile(local[:, 2], 0.999)),
                max_height_m=float(local[:, 2].max()),
            )
        )
with (OUT / "foot_motion.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
old = [r for r in rows if r["scene"] == "scene_foot_original_test"]
new = [r for r in rows if r["scene"] == "scene_foot_grounded_test_limited"]
ratio = sum(r["near_foot_rms_m_s"] for r in new[1:]) / sum(r["near_foot_rms_m_s"] for r in old[1:])
result = dict(
    pass_=ratio > 1.25,
    mean_rms_ratio=ratio,
    old_mean_rms_m_s=float(np.mean([r["near_foot_rms_m_s"] for r in old[1:]])),
    new_mean_rms_m_s=float(np.mean([r["near_foot_rms_m_s"] for r in new[1:]])),
    grounded_peak_height_m=max(r["max_height_m"] for r in new),
    window_s=[0, 0.6333333333],
    note="Same h, pool depth and near-zero emitter. Moving feet supply work: kinetic energy is a response measure, not a dissipation/conservation proof.",
)
result["pass"] = result.pop("pass_")
(OUT / "foot_motion.json").write_text(json.dumps(result, indent=2) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for r, label in [(old, "Before: 35 mm floor gap"), (new, "After: 1 mm floor gap")]:
    t = [q["time"] for q in r]
    ax[0].plot(t, [q["near_foot_rms_m_s"] * 100 for q in r], label=label)
    ax[1].plot(t, [q["height_p999_m"] * 100 for q in r], label=label)
ax[0].set(xlabel="Time (s)", ylabel="Near-foot RMS velocity (cm/s)")
ax[1].set(xlabel="Time (s)", ylabel="Particle height p99.9 (cm)")
for a in ax:
    a.grid(alpha=0.25)
    a.legend(fontsize=8)
fig.savefig(OUT / "foot_motion.png", dpi=180)
print(json.dumps(result, indent=2))
assert result["pass"]
