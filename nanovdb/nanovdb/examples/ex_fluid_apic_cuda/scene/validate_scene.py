# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Audit simulation and reconstruction separately; images are not pass criteria."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from render_scene import read_obj


def load(path):
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def edge_incidence(faces):
    """Count undirected edges using an exact pair of 32-bit vertex indices."""
    assert np.all(faces >= 0) and np.all(faces < 2**32)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    # Lexicographic pairs map bijectively to uint64. Numeric unique avoids the
    # structured-row comparisons of np.unique(..., axis=0), with identical counts.
    packed = (edges[:, 0].astype(np.uint64) << np.uint64(32)) | edges[:, 1].astype(np.uint64)
    return np.unique(packed, return_counts=True)[1]


def audit_mesh(task):
    folder, row = task
    worstEdges = 0
    minY = 1e9
    worstMismatch = 0
    maxSideLeak = 0
    binaryFrames, binaryError, binaryFacesMatch = 0, 0.0, True
    i = int(row["frame"])
    v, f = read_obj(folder / "surface" / f"water_{i:06d}.obj")
    if (folder / "surface" / f"water_{i:06d}.meshbin").exists():
        bv, bf = read_obj(folder / "surface" / f"water_{i:06d}.obj", prefer_binary=True)
        assert bv.shape == v.shape and bf.shape == f.shape
        binaryError = max(binaryError, float(np.max(abs(bv - v))))
        binaryFacesMatch &= np.array_equal(f, bf)
        binaryFrames += 1
    assert np.isfinite(v).all()
    assert len(v) == int(row["vertices"]) and len(f) == int(row["triangles"])
    assert np.all(f < len(v))
    counts = edge_incidence(f)
    worstEdges = max(worstEdges, int(np.sum(counts != 2)))
    minY = min(minY, float(v[:, 1].min()))
    maxSideLeak = max(
        maxSideLeak,
        float(-v[:, 0].min()),
        float(v[:, 0].max() - 3.2),
        float(-v[:, 2].min()),
        float(v[:, 2].max() - 2.4),
    )
    signed = (
        np.einsum(
            "ij,ij->i",
            v[f[:, 0]].astype(float),
            np.cross(v[f[:, 1]].astype(float), v[f[:, 2]].astype(float)),
        ).sum()
        / 6
    )
    worstMismatch = max(worstMismatch, abs(signed - row["mesh_volume"]) / row["material_volume"])
    return worstEdges, minY, worstMismatch, maxSideLeak, binaryFrames, binaryError, binaryFacesMatch


def main(folder, workers=1):
    assert workers >= 1
    settings = json.loads((folder / "settings.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    rows = load(folder / "metrics.csv")
    frames = load(folder / "frames.csv")
    meshes = load(folder / "surface/meshing.csv")
    assert rows and frames and meshes
    col = lambda k: np.array([r[k] for r in rows])
    single=settings.get("precision","double")=="float"
    pressure_tolerance=1e-4 if single else 2e-8
    contact_tolerance=2e-5 if single else 1e-7
    checks = {}
    details = {}
    checks["completed_duration"] = abs(rows[-1]["time"] - settings["duration"]) < 1e-9
    checks["pressure_absolute_or_relative"] = bool(
        np.all(col("residual_absolute") <= np.maximum(1e-12, pressure_tolerance * col("rhs_norm")))
    )
    for name, column, threshold in [
        ("weighted_flux_divergence", "flux_div_after", 2e-3 if single else 1e-6),
        ("moving_wall_flux", "wall_flux_error", 2e-5 if single else 1e-9),
        ("contact_inward_relative_normal", "contact_normal_error", 2e-5 if single else 1e-8),
        ("remaining_penetration", "penetrations", 0),
        ("penetration_depth", "penetration_depth", contact_tolerance),
        ("whitewater_penetration", "ww_depth", contact_tolerance),
        ("missing_stencils", "missing", 0),
        ("mass_ledger", "ledger_error", 0),
        ("cfl", "cfl", 0.5),
    ]:
        value = float(np.max(np.abs(col(column))))
        checks[name] = value <= threshold
        details[name] = {"maximum": value, "threshold": threshold}
    vp = settings["particle_volume"]
    quantization = np.max(abs(col("emitted") * vp - settings["flow_m3_s"] * col("time")))
    checks["emission_quantization"] = bool(quantization <= vp * (1 + 1e-6))
    details["emission_quantization_m3"] = {"maximum": float(quantization), "threshold": vp}
    ledger = col("particles") - (complete["initial_particles"] + col("emitted") - col("drained"))
    checks["independent_mass_ledger"] = bool(np.all(ledger == 0))
    if not settings.get("drain_enabled", True):
        checks["closed_basin_no_drain"] = bool(np.all(col("drained") == 0))
        checks["closed_basin_accumulation"] = bool(np.all(np.diff(col("particles")) >= 0))
    expected = rows[-1]["ww_expected"]
    births = rows[-1]["ww_spawned"]
    birthDeviation = abs(births - expected)
    checks["whitewater_births"] = birthDeviation <= 6 * np.sqrt(max(expected, 1)) + 5
    details["whitewater_births"] = {
        "actual": births,
        "expected": expected,
        "six_sigma_bound": 6 * np.sqrt(max(expected, 1)) + 5,
    }
    secondaryLedger = col("foam") + col("spray") - col("ww_spawned") + col("ww_expired")
    checks["secondary_lifetime_ledger"] = bool(np.all(secondaryLedger == 0))
    if "unresolved_contact_droplets" in rows[0]:
        unresolved = float(np.max(col("unresolved_contact_droplets") / col("particles")))
        checks["unresolved_contact_fraction"] = unresolved <= 0.005
        details["unresolved_contact_fraction"] = {
            "maximum": unresolved,
            "threshold": 0.005,
            "note": "Ballistic fallback only in solid-classified cells or a component with both MAC faces blocked; represented component stencil failures remain fatal.",
        }
    checks["surface_frame_count"] = len(meshes) == len(frames)
    ratios = np.array([r["volume_ratio"] for r in meshes])
    change = np.array([r["relative_volume_change"] for r in meshes])
    integral = np.array([abs(r["kernel_integral"] / r["material_volume"] - 1) for r in meshes])
    checks["mesh_volume_range"] = bool(np.all((ratios >= 0.80) & (ratios <= 1.20)))
    checks["normalized_volume_flicker"] = bool(np.max(change) <= 0.08)
    checks["kernel_mass_integral"] = bool(np.max(integral) <= 0.03)
    if "filter_volume_ratio" in meshes[0]:
        filter_ratios = np.array([r["filter_volume_ratio"] for r in meshes])
        checks["filter_volume_change"] = bool(np.all(np.isfinite(filter_ratios)) and np.max(np.abs(filter_ratios - 1)) <= .03)
        details["filter_volume_change"] = {"ratio_range": [float(filter_ratios.min()), float(filter_ratios.max())],
                                           "relative_change_threshold": .03}
    if "floor_halfspace_integral" in meshes[0]:
        reflection_error = max(
            abs(r["floor_halfspace_integral"] / r["kernel_integral"] - 1) for r in meshes
        )
        checks["floor_kernel_conservation"] = reflection_error <= 1e-6
        details["floor_kernel_conservation"] = {"maximum": reflection_error, "threshold": 1e-6}
    details["reconstruction"] = {
        "mesh_volume_ratio_range": [float(ratios.min()), float(ratios.max())],
        "threshold": [0.8, 1.2],
        "max_adjacent_normalized_volume_change": float(change.max()),
        "change_threshold": 0.08,
        "max_kernel_integral_relative_error": float(integral.max()),
        "integral_threshold": 0.03,
        "note": "Volume stability is only one flicker proxy; it does not prove local surface coherence.",
    }
    # Audit actual exported topology, not just the mesher counters. All frames.
    worstEdges = 0
    minY = 1e9
    worstMismatch = 0
    maxSideLeak = 0
    binaryFrames, binaryError, binaryFacesMatch = 0, 0.0, True
    from concurrent.futures import ProcessPoolExecutor

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    tasks = ((folder, row) for row in meshes)
    results = executor.map(audit_mesh, tasks) if executor else map(audit_mesh, tasks)
    try:
        for edges, y, mismatch, side, binaries, error, facesMatch in results:
            worstEdges = max(worstEdges, edges)
            minY = min(minY, y)
            worstMismatch = max(worstMismatch, mismatch)
            maxSideLeak = max(maxSideLeak, side)
            binaryFrames += binaries
            binaryError = max(binaryError, error)
            binaryFacesMatch &= facesMatch
    finally:
        if executor:
            executor.shutdown()
    checks["watertight_export"] = worstEdges == 0
    checks["outward_winding_and_volume"] = worstMismatch < 1e-4
    checks["mesh_floor"] = minY >= -settings["h"] * 0.05
    if binaryFrames:
        checks["renderer_binary_matches_obj"] = (
            binaryFrames == len(meshes) and binaryFacesMatch and binaryError <= 5e-7
        )
        details["renderer_binary_matches_obj"] = {
            "frames": binaryFrames,
            "max_vertex_error_m": binaryError,
            "same_faces": bool(binaryFacesMatch),
            "threshold_m": 5e-7,
        }
    if not settings.get("drain_enabled", True):
        checks["mesh_closed_sides"] = maxSideLeak <= 1e-5
    details["mesh_export"] = {
        "max_non_two_face_edges": worstEdges,
        "min_y": minY,
        "max_side_extension_m": maxSideLeak,
        "max_volume_roundtrip_error": worstMismatch,
    }
    performance = {}
    for name in (
        "topology_ms",
        "mesh_sdf_ms",
        "boundary_ms",
        "p2g_ms",
        "pressure_ms",
        "g2p_ms",
        "contact_ms",
        "whitewater_ms",
        "transfer_ms",
        "io_ms",
        "gpu_used_mb",
    ):
        x = col(name)
        performance[name] = {
            "mean": float(x.mean()),
            "p95": float(np.quantile(x, 0.95)),
            "max": float(x.max()),
            "sum": float(x.sum()),
        }
    meshPerformance = {
        k: {
            "mean": float(np.mean([r[k] for r in meshes])),
            "max": float(max(r[k] for r in meshes)),
            "sum": float(sum(r[k] for r in meshes)),
        }
        for k in ("read_ms", "pca_ms", "grid_ms", "clip_ms", "contour_ms", "write_ms", "filter_ms")
        if k in meshes[0]
    }
    checks = {k: bool(v) for k, v in checks.items()}
    result = {
        "pass": all(checks.values()),
        "checks": checks,
        "details": details,
        "steps": len(rows),
        "frames": len(frames),
        "final_particles": rows[-1]["particles"],
        "max_particles": float(col("particles").max()),
        "final_material_volume": rows[-1]["mass_volume"],
        "emitted_volume": rows[-1]["emitted"] * vp,
        "drained_volume": rows[-1]["drained"] * vp,
        "max_whitewater": float(np.max(col("foam") + col("spray"))),
        "performance": performance,
        "meshing_performance": meshPerformance,
        "character_velocity_fd_error": complete["character_velocity_fd_error"],
    }
    (folder / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "checks": checks, "details": details}, indent=2))
    assert result["pass"], "scene validation failed; see validation.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    main(args.cache, args.workers)
