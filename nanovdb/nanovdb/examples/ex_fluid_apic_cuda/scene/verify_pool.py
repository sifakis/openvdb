# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Audit closed-basin mass, bounds and actual near-floor particle height separately."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def height_tiles(bed, emitter_center, particle_spacing):
    """The same 16 cm half-open tiles, grouped once instead of 300 full scans."""
    surface = np.full((20, 15), np.nan)
    ix = np.searchsorted(np.arange(21) * 0.16, bed[:, 0], side="right") - 1
    iz = np.searchsorted(np.arange(16) * 0.16, bed[:, 2], side="right") - 1
    inside = (ix >= 0) & (ix < 20) & (iz >= 0) & (iz < 15)
    code = ix[inside] * 15 + iz[inside]
    order = np.argsort(code, kind="stable")
    heights = bed[inside, 1][order]
    starts = np.searchsorted(code[order], np.arange(301))
    for x in range(20):
        for z in range(15):
            cx, cz = (x + 0.5) * 0.16, (z + 0.5) * 0.16
            if np.hypot(cx - emitter_center[0], cz - emitter_center[2]) < 0.65:
                continue
            tile = x * 15 + z
            q = heights[starts[tile] : starts[tile + 1]]
            if len(q) >= 4:
                surface[x, z] = np.quantile(q, 0.9) + 0.5 * particle_spacing
    return surface



def secondary_arithmetic_bound_error(q, settings):
    """Excess beyond the representable wall clamp, without an empirical epsilon.

    Whitewater.cuh evaluates n*h-radius in Real and stores positions in Real.
    CUDA may fuse the multiplication/subtraction. Nine significant CSV digits
    round-trip float32, so compare the reconstructed floats against both valid
    IEEE evaluation orders. This does not replace reporting physical-plane error.
    """
    assert np.isfinite(q).all()
    if settings.get("precision", "double") != "float":
        return float(np.maximum.reduce((q[:, 3] - q[:, 0], q[:, 0] + q[:, 3] - 3.2,
                                        q[:, 3] - q[:, 2], q[:, 2] + q[:, 3] - 2.4,
                                        -q[:, 1])).max())
    stored = q.astype(np.float32)
    radius = stored[:, 3]
    exact = stored.astype(np.float64)
    errors = [exact[:, 3] - exact[:, 0], exact[:, 3] - exact[:, 2], -exact[:, 1]]
    h = np.float32(settings["h"])
    for axis, extent in ((0, 3.2), (2, 2.4)):
        count = round(extent / float(h))
        separate = (np.float32(count) * h - radius).astype(np.float32)
        fused = (count * float(h) - radius.astype(np.float64)).astype(np.float32)
        upper = np.maximum(separate, fused).astype(np.float64)
        errors.append(exact[:, axis] - upper)
    return float(np.maximum.reduce(errors).max())


def audit(folder):
    settings = json.loads((folder / "settings.json").read_text())
    assert not settings["drain_enabled"]
    metrics = rows(folder / "metrics.csv")
    manifest = rows(folder / "frames.csv")
    vp, h = settings["particle_volume"], settings["h"]
    area = 3.2 * 2.4
    emitter_center = settings.get("emitter_center", [1.62, 2.55, 1.0])
    frames, max_bound, max_secondary_bound = [], 0.0, 0.0
    max_secondary_arithmetic_bound = 0.0
    max_side_height, max_secondary_side_height = 0.0, 0.0
    surfaces = []
    initial_pool = settings.get("initial_pool_particles", 0)
    initial_total = int(manifest[0]["particles"])
    for row in manifest:
        i, t = int(row["frame"]), float(row["time"])
        raw = np.loadtxt(
            folder / f"frame_{i:06d}.csv",
            delimiter=",",
            skiprows=1,
            usecols=(0, 1, 2, 3, 5),
            ndmin=2,
        )
        p = raw[:, 1:4]
        pool_ids = (raw[:, 0] >= initial_total - initial_pool) & (raw[:, 0] < initial_total)
        rising = pool_ids & (p[:, 1] > settings.get("initial_depth", 0) + h) & (raw[:, 4] > 0.5)
        max_bound = max(max_bound, float(np.maximum(-p, p - [3.2, 3.2, 2.4]).max()))
        contact = np.minimum.reduce((p[:, 0], 3.2 - p[:, 0], p[:, 2], 2.4 - p[:, 2])) <= h * 0.10001
        if np.any(contact):
            max_side_height = max(max_side_height, float(p[contact, 1].max()))
        bed = p[p[:, 1] < 0.25]
        # Height proxy, not a level set: p90 within 16 cm tiles + half nominal
        # particle spacing. Restrict height measurement away from the impact jet.
        surface = height_tiles(bed, emitter_center, vp ** (1 / 3))
        valid = surface[np.isfinite(surface)]
        frames.append(
            {
                "frame": i,
                "time": t,
                "particles": len(p),
                "total_volume_m3": len(p) * vp,
                "bed_volume_m3": len(bed) * vp,
                "bed_equivalent_depth_m": len(bed) * vp / area,
                "surface_height_median_m": float(np.median(valid)) if len(valid) else 0,
                "wet_probe_tiles": len(valid),
                "rising_initial_pool_particles": int(np.count_nonzero(rising)),
                "initial_pool_max_height_m": float(p[pool_ids, 1].max()) if np.any(pool_ids) else 0,
            }
        )
        surfaces.append(surface)
        if int(row["foam"]) + int(row["spray"]):
            q = np.loadtxt(
                folder / f"frame_{i:06d}_whitewater.csv",
                delimiter=",",
                skiprows=1,
                usecols=(1, 2, 3, 7),
                ndmin=2,
            )
            side = np.maximum.reduce(
                (
                    q[:, 3] - q[:, 0],
                    q[:, 0] + q[:, 3] - 3.2,
                    q[:, 3] - q[:, 2],
                    q[:, 2] + q[:, 3] - 2.4,
                    -q[:, 1],
                )
            )
            max_secondary_bound = max(max_secondary_bound, float(side.max()))
            max_secondary_arithmetic_bound = max(
                max_secondary_arithmetic_bound, secondary_arithmetic_bound_error(q, settings))
            contact = (
                np.minimum.reduce((q[:, 0], 3.2 - q[:, 0], q[:, 2], 2.4 - q[:, 2]))
                <= q[:, 3] + 1e-7
            )
            if np.any(contact):
                max_secondary_side_height = max(
                    max_secondary_side_height, float(q[contact, 1].max())
                )
    initial = frames[0]["total_volume_m3"]
    mass_error = max(
        abs(r["total_volume_m3"] - initial - settings["flow_m3_s"] * r["time"]) for r in frames
    )
    start = min(frames, key=lambda x: abs(x["time"] - 1.2))
    end = frames[-1]
    checks = {
        "no_drain": all(int(r["drained"]) == 0 for r in metrics),
        "mass_ledger": all(int(r["ledger_error"]) == 0 for r in metrics),
        "mass_matches_inflow": mass_error <= vp * 1.000001,
        "fluid_inside_basin": max_bound <= 1e-7,
        "secondary_inside_basin": max_secondary_arithmetic_bound <= (
            0.0 if settings.get("precision", "double") == "float" else 1e-7),
    }
    if end["time"] >= 5.9:
        if settings.get("initial_depth", 0) > 0:
            checks["floor_volume_grows"] = end["bed_volume_m3"] > frames[0][
                "bed_volume_m3"
            ] + 0.8 * settings["flow_m3_s"] * max(0, end["time"] - 2)
        else:
            checks["floor_volume_grows"] = end["bed_volume_m3"] > 3 * start["bed_volume_m3"]
        checks["floor_height_rises"] = (
            end["surface_height_median_m"] > start["surface_height_median_m"] + 0.25 * h
        )
        checks["broad_pool"] = end["wet_probe_tiles"] > 180
    if settings.get("initial_depth", 0) > 0:
        checks["initial_pool_present"] = settings["initial_pool_particles"] > 0
        checks["initial_pool_volume"] = (
            abs(frames[0]["bed_volume_m3"] - settings["initial_pool_volume"]) < vp * 1e-6
        )
        checks["initial_pool_coverage"] = frames[0]["wet_probe_tiles"] > 180
    result = {
        "pass": all(checks.values()),
        "checks": checks,
        "initial_volume_m3": initial,
        "inflow_m3_s": settings["flow_m3_s"],
        "mass_error_m3": mass_error,
        "mass_tolerance_m3": vp,
        "max_fluid_bound_error_m": max_bound,
        "max_secondary_bound_error_m": max_secondary_bound,
        "max_secondary_arithmetic_bound_error_m": max_secondary_arithmetic_bound,
        "secondary_bound_reference": ("float32 wall arithmetic, fused/unfused envelope; zero extra tolerance"
                                      if settings.get("precision", "double") == "float"
                                      else "physical plane, 1e-7 m tolerance"),
        "sampled_max_side_contact_height_m": max_side_height,
        "sampled_max_secondary_side_contact_height_m": max_secondary_side_height,
        "height_probe_exclusion_center_xz": [emitter_center[0], emitter_center[2]],
        "height_reference": start,
        "final": end,
        "note": "Bed is y<25 cm. Equivalent depth is volume/area, not a measured interface. The independent height proxy is a tilewise particle p90 away from impact; waves and particle sampling bias it.",
    }
    (folder / "pool_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    with (folder / "pool_levels.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=frames[0])
        writer.writeheader()
        writer.writerows(frames)
    np.save(folder / "pool_height_probes.npy", np.array(surfaces))
    print(json.dumps(result, indent=2))
    assert result["pass"], "closed basin validation failed"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    audit(p.parse_args().cache)
