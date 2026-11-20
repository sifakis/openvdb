# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Measured drainage/retention comparison; preserves both simulation caches."""

import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "output/scene_water"
NEW = ROOT / "output/scene_pool_complete"


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    val = json.loads((NEW / "validation.json").read_text())
    pool = json.loads((NEW / "pool_validation.json").read_text())
    restart = json.loads((ROOT / "output/pool_diagnosis/restart/result.json").read_text())
    assert val["pass"] and pool["pass"] and restart["pass"]
    settings = json.loads((NEW / "settings.json").read_text())
    oldsettings = json.loads((OLD / "settings.json").read_text())
    oldm, newm = table(OLD / "metrics.csv"), table(NEW / "metrics.csv")
    meshes = table(NEW / "surface/meshing.csv")
    levels = table(NEW / "pool_levels.csv")
    vp = settings["particle_volume"]
    olddrained = int(oldm[-1]["drained"]) * oldsettings["particle_volume"]
    video = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(NEW / "dance_water.mp4"),
            ]
        )
    )["streams"][0]
    assert int(video["nb_read_frames"]) == val["frames"] and video["r_frame_rate"] == "30/1"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    montage = Image.new("RGB", (1600, 3 * 630), (18, 22, 28))
    draw = ImageDraw.Draw(montage)
    for row, frame in enumerate((36, 90, 180)):
        for col, (folder, label) in enumerate(
            ((OLD, "Before: perimeter drain"), (NEW, "After: closed basin"))
        ):
            im = Image.open(folder / "render" / f"frame_{frame:06d}.png").convert("RGB")
            assert np.asarray(im).std() > 5
            im.thumbnail((800, 600))
            montage.paste(im, (800 * col, 630 * row + 30))
            draw.text(
                (800 * col + 12, 630 * row + 2),
                f"{label} | t={frame/30:.1f}s",
                font=font,
                fill="white",
            )
    montage.save(NEW / "before_after.jpg", quality=95)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(OLD / "dance_water.mp4"),
            "-i",
            str(NEW / "dance_water.mp4"),
            "-filter_complex",
            "[0:v]scale=960:720,drawtext=text='Before - perimeter drain':x=20:y=20:fontsize=26:fontcolor=white[a];"
            "[1:v]scale=960:720,drawtext=text='After - closed basin':x=20:y=20:fontsize=26:fontcolor=white[b];"
            "[a][b]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(NEW / "before_after.mp4"),
        ],
        check=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for rows, label in ((oldm, "Perimeter drain"), (newm, "Closed basin")):
        axes[0].plot(
            [float(r["time"]) for r in rows],
            [float(r["mass_volume"]) * 1000 for r in rows],
            label=label,
        )
    axes[0].set(xlabel="Time (s)", ylabel="Liquid material volume (L)")
    axes[0].legend()
    times = [float(r["time"]) for r in levels]
    axes[1].plot(
        times,
        [float(r["surface_height_median_m"]) * 100 for r in levels],
        label="Tile particle height (p90 median)",
    )
    axes[1].plot(
        times,
        [float(r["bed_equivalent_depth_m"]) * 100 for r in levels],
        label="Bed volume / basin area",
    )
    axes[1].set(xlabel="Time (s)", ylabel="Height / equivalent depth (cm)", xlim=(1.2, 6))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(NEW / "accumulation.png", dpi=180)
    plt.close(fig)
    summary = {
        "video": video,
        "old_drained_m3": olddrained,
        "new_drained_m3": int(newm[-1]["drained"]) * vp,
        "pool": pool,
        "restart": restart,
    }
    (NEW / "pool_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    d, perf = val["details"], val["performance"]
    start, end = pool["height_reference"], pool["final"]
    timings = json.loads((NEW / "pipeline_times.json").read_text())
    timing_rows = "\n".join(
        f"| {k} | {v['mean']:.3f} | {v['sum']/1000:.2f} |"
        for k, v in perf.items()
        if k.endswith("_ms")
    )
    mesh_rows = "\n".join(
        f"| {k} | {v['mean']:.3f} | {v['sum']/1000:.2f} |"
        for k, v in val["meshing_performance"].items()
    )
    report = f"""# Drain condition changed so that water accumulates on the floor

## Cause and changes

The previous scene used an absorbing drain region that deleted particles with `y < 0.18 m` that were also within
14 cm of the floor edge. After the artificial tangential damping was removed, the water spread quickly to the edges,
and **{olddrained*1000:.3f} L** was drained in 6 seconds. Leaving this scene setting as it was is
the reason water did not accumulate on the floor. It was not floor penetration or mass loss in the pressure solver.

- `scene/pool.json`: `drain_enabled: false`. The existing inflow rate, APIC, free-slip/contact,
  character animation, resolution and whitewater settings are kept. No artificial viscosity/floor friction was added.
- The existing closed pressure/G2P domain boundary is reused. Secondary particles are moved inward from the sides by their radius,
  and only the outward normal velocity is removed. The foam surface search does not treat walls
  as air, and the CPU meshing field and the render foam positions are also limited at the same boundary.
- A transparent rim was added to the renderer input. Its displayed height is the bottom 30 cm, and
  the sides of the actual computational domain are closed up to the full height of 3.2 m. It is not a model that overflows at a finite height.
- `mesh_floor_boundary: true`: near the floor, corrects the PCA kernel so that its normal-direction variance
  does not become excessively smaller than the tangential variance. The correction is reduced with distance from the floor and
  becomes 0 at twice the kernel support distance. The tangential anisotropy and the kernels in the air are kept.
  Next, the part of the kernel inside the solid is reflected to the opposite side of the floor. The integral over the floor half-space
  equals the original normalized kernel integral, and it is validated using a trapezoid weight of 1/2 on the boundary plane.
  Particle positions/velocities/masses, the isovalue and the actual fluid motion are not changed.
- The drain setting is stored in the `FLUID003` checkpoint, so a restart between different drain modes is rejected.
- At a narrow character contact region at 4.65 seconds, a case was found where both the upper and lower MAC faces of a fluid cell are blocked
  and the vertical G2P velocity cannot be extended. Only this unrepresented component
  is advanced with the previous velocity + gravity, its C row is reset, and it is then limited by the actual SDF/triangle contact.
  The projected G2P values of the horizontal components are preserved. A missing value in a component that has an open face is still an error.
  Positive/negative regression checks that distinguish this behavior from an actual missing value were added.
- The changes are confined to this example folder. The existing NanoVDB operators and the ovrtx code were not modified.
  The previous results in `output/scene_water` are preserved.

## Measured results

6 seconds, {val['frames']} frames, {val['steps']} substeps. Flow rate {pool['inflow_m3_s']*1000:.6f} L/s,
initial water {pool['initial_volume_m3']*1000:.6f} L, volume per particle {vp*1e6:.3f} mL.

| Item | Result |
|---|---:|
| Final APIC particle count | {end['particles']} |
| Material volume of the final water | {end['total_volume_m3']*1000:.3f} L |
| Drained amount of the new scene | {int(newm[-1]['drained'])*vp*1000:.6f} L |
| Max ledger error relative to the inflow | {pool['mass_error_m3']*1e6:.3f} mL (within one particle) |
| Material volume in the bottom 25 cm: 1.2 → 6 s | {start['bed_volume_m3']*1000:.3f} → {end['bed_volume_m3']*1000:.3f} L |
| Volume/floor area of the same region | {start['bed_equivalent_depth_m']*100:.3f} → {end['bed_equivalent_depth_m']*100:.3f} cm |
| Median per-tile height measured from particle positions | {start['surface_height_median_m']*100:.3f} → {end['surface_height_median_m']*100:.3f} cm |
| Height-measurement tiles that have water at the end | {end['wet_probe_tiles']} |
| Final foam / spray | {newm[-1]['foam']} / {newm[-1]['spray']} |

The height measurement divides the floor into 16 cm tiles and uses `particle y p90 + particle spacing/2` of the tiles,
excluding those within a 65 cm radius of the impact center. It is a proxy affected by waves and particle sampling.
`volume/area` is the mean equivalent depth and is distinguished from the actual surface height. The rise of the water level was not judged
by the liquid mass increase alone; these two values, the particle video and the surface mesh were checked together.

| Validation | Maximum | Criterion |
|---|---:|---:|
| Flux divergence including wall motion/blocked fraction | {d['weighted_flux_divergence']['maximum']:.3e} s⁻¹ | 1e−6 |
| Moving wall face flux error | {d['moving_wall_flux']['maximum']:.3e} | 1e−9 |
| Particle inward normal velocity error | {d['contact_inward_relative_normal']['maximum']:.3e} m/s | 1e−8 |
| Remaining fluid penetration | {d['remaining_penetration']['maximum']:.0f} | 0 |
| Secondary penetration depth | {d['whitewater_penetration']['maximum']:.3e} m | 1e−7 |
| Missing stencils | {d['missing_stencils']['maximum']:.0f} | 0 |
| Position error outside the APIC domain | {pool['max_fluid_bound_error_m']:.3e} m | 1e−7 |
| Error outside the domain including the secondary radius | {pool['max_secondary_bound_error_m']:.3e} m | 1e−7 |
| Extension of the extracted mesh beyond the sides | {d['mesh_export']['max_side_extension_m']:.3e} m | 1e−5 |

The pressure residual passed the `max(1e−12, 2e−8 ||b||)` criterion at every step.
Meshing volume ratio range {d['reconstruction']['mesh_volume_ratio_range']}, max normalized volume change between
adjacent frames {d['reconstruction']['max_adjacent_normalized_volume_change']:.3%}.
The closed edge/winding check of every mesh also passed. This is a volume stability metric, not a proof that excludes
every local flicker. The physical particle ledger and the surface reconstruction volume are recorded separately.
Before the floor reconstruction fix, the last mesh volume ratio failed at 0.694695. After the fix, the last volume ratio
of the same cache is {float(meshes[-1]['volume_ratio']):.6f}. The maximum relative difference between the whole kernel integral before reflection and
the floor half-space integral after reflection is
{d['floor_kernel_conservation']['maximum']:.3e} (criterion 1e−6).
Neither the basin validation criteria nor the existing volume tolerance range [0.8, 1.2] was relaxed.
The unresolved contact fallback was at most {max(int(r['unresolved_contact_droplets']) for r in newm)} per substep,
at most {d['unresolved_contact_fraction']['maximum']:.4%} of all particles.
The maximum height of side-contact particles measured in the 30 fps cache is, for water,
{pool['sampled_max_side_contact_height_m']:.3f} m, secondary
{pool['sampled_max_secondary_side_contact_height_m']:.3f} m. It is not a measurement of the peak between substeps.

The zero-gravity floor transfer of `verify_water` kept 1 m/s for both 1 and 100 repetitions.
The added secondary side collision removed the outward velocity with an analytic tangential velocity error of 0.
The check of mixing the ballistic motion of the blocked component with the G2P of the other components was within an error of 1e−12, and
the check that deliberately erased the value of a represented component correctly detected the missing value.
The maximum APIC x/v/C difference between the result restarted from the 0.4-second checkpoint up to 0.6 seconds and the continuous run was
{restart['raw_particle_x_v_C_max_difference']:.3e}, and the secondary difference was
{restart['whitewater_max_difference']:.3e} (criterion 2e−8).

## Run and cost

Settings: fluid h=32 mm, obstacle h=16 mm, mesh h=10 mm, ppc=8,
max dt=4 ms + adaptive CFL, whitewater 120,000 /(m³·s), 1600×1200 / 96 spp / 30 fps.

```bash
EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
cmake --build build/release --target ex_fluid_apic_scene_cuda ex_fluid_apic_surface ex_fluid_apic_water_verify -j3
build/release/nanovdb/nanovdb/examples/ex_fluid_apic_water_verify "$EX/output/character"
/tmp/fluid-render-venv/bin/python "$EX/scene/pipeline.py" "$EX/scene/pool_profile.json" --stages simulate
/tmp/fluid-render-venv/bin/python "$EX/scene/verify_pool.py" "$EX/output/scene_pool_profile"
/tmp/fluid-render-venv/bin/python "$EX/scene/verify_restart.py" --output "$EX/output/pool_diagnosis/restart"
/tmp/fluid-render-venv/bin/python "$EX/scene/pipeline.py" "$EX/scene/pool.json" --output "$EX/output/scene_pool_complete"
/tmp/fluid-render-venv/bin/python "$EX/scene/report_pool.py"
```

A short restart regression check was run separately during the measurement, so the early simulation timing may include
GPU contention. The following values are the actual wall-clock measurements of this run, not a benchmark of a standalone run.
The first run stopped at 4.65 seconds with a validation failure. The original partial cache was preserved, and the run was resumed after the fix
from the 4.63333-second checkpoint of a copy. The new fallback condition did not occur in the interval that succeeded before the fix,
so the existing state was reused as it was. The per-stage totals include all successful substeps,
and the simulate wall time in `pipeline_times.json` includes only the interval after the resume.

| Stage | substep mean ms | total s |
|---|---:|---:|
{timing_rows}

| CPU mesh stage | frame mean ms | total s |
|---|---:|---:|
{mesh_rows}

Total process times (seconds): `{json.dumps(timings)}`.
`transfer_ms`, `io_ms`, CPU meshing and render times were separated from the solver cost.
Observed maximum GPU usage {perf['gpu_used_mb']['max']:.1f} MB.

## Result files and limitations

- [Final video](output/scene_pool_complete/dance_water.mp4), [comparison with the previous draining scene](output/scene_pool_complete/before_after.mp4)
- [Comparison images at 1.2/3/6 seconds](output/scene_pool_complete/before_after.jpg), [mass and height graphs](output/scene_pool_complete/accumulation.png)
- [Particle/obstacle debug video](output/scene_pool_complete/debug.mp4)
- [Numerical validation](output/scene_pool_complete/validation.json), [basin/height validation](output/scene_pool_complete/pool_validation.json)
- `output/scene_pool_complete/`: APIC/secondary/SDF cache, `surface/`: OBJ,
  `render/`: 181 PNG, `scene.usdc`: 3D animation.

This is a finite-time validation over 6 seconds. Neither the equilibrium nor the overflow of a scene that keeps adding water to a closed region
indefinitely is implemented. At the current resolution the floor water of a few cm depth is only a few voxels, so it is not interpreted as a
precise convergence result for the water surface/thin films. The existing subcell contact ballistic fallback is kept and
monitored with `unresolved_contact_fraction`. No surface tension or actual viscosity model was added.
"""
    (ROOT / "POOL_RESULTS.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
