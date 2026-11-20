# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Report the twelve-second prefilled basin and measured visible white foam."""

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
OUT = ROOT / "output/scene_wet_pool"
OLD = ROOT / "output/scene_pool_complete"


def table(path):
    return list(csv.DictReader(path.open()))


def main():
    v = json.loads((OUT / "validation.json").read_text())
    pool = json.loads((OUT / "pool_validation.json").read_text())
    settings = json.loads((OUT / "settings.json").read_text())
    restart = json.loads((ROOT / "output/wet_pool_diagnosis/restart/result.json").read_text())
    visibility = [
        json.loads((OUT / "foam_visibility" / f"frame_{i:06d}.json").read_text())
        for i in (180, 360)
    ]
    assert v["pass"] and pool["pass"] and restart["pass"] and all(q["pass"] for q in visibility)
    m, frames, levels = (
        table(OUT / "metrics.csv"),
        table(OUT / "frames.csv"),
        table(OUT / "pool_levels.csv"),
    )
    mesh = table(OUT / "surface/meshing.csv")
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
                str(OUT / "dance_water.mp4"),
            ]
        )
    )["streams"][0]
    assert int(video["nb_read_frames"]) == 361 and video["r_frame_rate"] == "30/1"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    montage = Image.new("RGB", (1600, 1260), (20, 25, 30))
    draw = ImageDraw.Draw(montage)
    for n, frame in enumerate((0, 90, 180, 360)):
        im = Image.open(OUT / "render" / f"frame_{frame:06d}.png").convert("RGB")
        assert np.asarray(im).std() > 5
        im.thumbnail((800, 600))
        x, y = (n % 2) * 800, (n // 2) * 630
        montage.paste(im, (x, y + 30))
        draw.text((x + 12, y + 3), f"Prefilled basin | t={frame/30:.1f}s", font=font, fill="white")
    montage.save(OUT / "representative_frames.jpg", quality=95)
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
            str(OUT / "dance_water.mp4"),
            "-filter_complex",
            "[0:v]scale=960:720,drawtext=text='Before - dry start':x=20:y=20:fontsize=26:fontcolor=white[a];"
            "[1:v]scale=960:720,drawtext=text='After - shallow water and foam':x=20:y=20:fontsize=26:fontcolor=white[b];"
            "[a][b]hstack=inputs=2:shortest=1[v]",
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
            str(OUT / "comparison_first_6s.mp4"),
        ],
        check=True,
    )
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    t = [float(q["time"]) for q in frames]
    ax[0].plot(t, [float(q["foam"]) for q in frames], label="Foam")
    ax[0].plot(t, [float(q["spray"]) for q in frames], label="Spray")
    ax[0].set(xlabel="Time (s)", ylabel="Secondary particles")
    ax[0].legend()
    ax[1].plot(
        [float(q["time"]) for q in levels],
        [float(q["bed_equivalent_depth_m"]) * 100 for q in levels],
        label="Bed volume / area",
    )
    ax[1].plot(
        [float(q["time"]) for q in levels],
        [float(q["surface_height_median_m"]) * 100 for q in levels],
        label="Particle height proxy",
    )
    ax[1].set(xlabel="Time (s)", ylabel="Depth / height (cm)")
    ax[1].legend(fontsize=8)
    ax[2].plot(
        [float(q["time"]) for q in levels],
        [
            int(q["rising_initial_pool_particles"]) * settings["particle_volume"] * 1000
            for q in levels
        ],
    )
    ax[2].set(xlabel="Time (s)", ylabel="Rising prefill water above initial surface (L)")
    for x in ax:
        x.grid(alpha=0.25)
    fig.savefig(OUT / "water_and_foam.png", dpi=180)
    plt.close(fig)
    summary = {
        "video": video,
        "visibility": visibility,
        "validation": v["details"],
        "pool": pool,
        "rising_prefill_volume_peak_l": max(int(q["rising_initial_pool_particles"]) for q in levels)
        * settings["particle_volume"]
        * 1000,
        "prefill_particle_max_height_m": max(float(q["initial_pool_max_height_m"]) for q in levels),
    }
    (OUT / "scene_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    perf = "\n".join(
        f"| {k} | {q['mean']:.3f} | {q['sum']/1000:.2f} |"
        for k, q in v["performance"].items()
        if k.endswith("_ms")
    )
    mp = "\n".join(
        f"| {k} | {q['mean']:.3f} | {q['sum']/1000:.2f} |"
        for k, q in v["meshing_performance"].items()
    )
    visible = "\n".join(
        f"| {q['frame']/30:.0f} s | {q['white_contribution_pixels']} | {q['floor_white_pixels']} |"
        for q in visibility
    )
    text = f"""# 12 seconds, a shallow initial water layer and white foam

A new scene was made separately in `scene/wet_pool.json` and `output/scene_wet_pool`.
The existing 6-second cache/video were preserved. The changes are confined to the example folder, and the library and the ovrtx SDK were not modified.

## Changes

- Simulation 0–12 seconds, 30 fps, 361 frames including both ends. The video length is 12.033 seconds.
- The floor was first filled with water **6.4 cm** deep. The particle volume is kept at h³/ppc, and
  the initial velocity and affine C are 0. Only particles that pass the SDF clearance and triangle interior checks are kept.
  The initial pool particles are **{settings['initial_pool_particles']:,} / {settings['initial_pool_volume']*1000:.3f} L**.
  At h=32 mm, ppc=8 the particle spacing is 16 mm, and the initial water layer is four particle layers.
- The emitter position, radius and speed and the closed domain were kept. The flow rate is {settings['flow_m3_s']*1000:.6f} L/s and
  the drain is 0. No physical viscosity/friction was added.
- Previously, a spray that turned into foam still kept the short spray lifetime. Now the foam size and
  lifetime are assigned at the conversion. This behavior was confirmed with an independent regression check. Foam lasts about
  **1.98–5.04 seconds** from creation and after conversion, and the radius of directly created foam is **6–10.5 mm**.
- The whitewater generation rate is 120,000 → **360,000 /(m³·s)**. The existing near-surface acceleration/collision/vorticity
  sources are used as they are, and foam is not sprinkled unconditionally over all the static water.
  Secondary does not enter P2G, pressure or fluid mass. The same SDF/wall collision is applied to converted foam as well.
- Foam is rendered with a neutral white diffuse material (albedo 0.98, roughness 0.65) and as 32-triangle spheres.
  This represents the projected area of a sphere better than the existing small 8-triangle octahedron. The size is not inflated with camera distance.
- The meshing isovalue of the new pool scene is **0.2**. With the particle distribution of the deeper water layer, the existing 0.4
  was found to tend to reconstruct a surface volume that is too small, and it was adjusted after a comparison on the same particles.
  The volume ratio of the 11.67-second representative frame failed at 0.796 with isovalue 0.3 and improved to 0.841 with 0.2. The kernel integral/particle mass are unchanged, and
  every frame must pass the original volume and flicker criteria. It does not affect the simulation.
  The coarse profile (h=80 mm, mesh h=25 mm) failed with isovalue 0.3 at a volume ratio of 1.222–1.232, so
  it keeps 0.4, which passed validation. The two surface reconstruction settings of different resolutions were validated separately.
- The `FLUID004` checkpoint stores the initial water layer count/depth and the foam size, lifetime and generation rate settings.
  Restarting with different initial conditions is rejected.

Having initial water does not necessarily produce white foam. The secondary particles generated by actual fluid collisions
are kept near the water surface, and it was checked separately that the final water render has sufficient white contrast.

## Physical and visual validation

Pressure/flux, penetration, mass and stencil validation of {v['steps']:,} substeps and surface validation of 361 frames: **PASS**.

| Item | Result |
|---|---:|
| Initial total liquid | {pool['initial_volume_m3']*1000:.3f} L |
| Final total liquid | {v['final_material_volume']*1000:.3f} L / {int(v['final_particles']):,} particles |
| Final liquid in the bottom 25 cm | {pool['final']['bed_volume_m3']*1000:.3f} L |
| Final volume/floor-area equivalent depth | {pool['final']['bed_equivalent_depth_m']*100:.3f} cm |
| Final per-tile particle height metric | {pool['final']['surface_height_median_m']*100:.3f} cm |
| Peak rising water volume of particles from the initial water layer | {summary['rising_prefill_volume_peak_l']:.3f} L |
| Maximum height reached by those particles | {summary['prefill_particle_max_height_m']:.3f} m |
| Final foam / spray | {m[-1]['foam']} / {m[-1]['spray']} |
| Max concurrent secondary count | {int(v['max_whitewater']):,} |
| Total secondary particles generated | {m[-1]['ww_spawned']} |
| Max pressure relative residual | {max(float(q['residual']) for q in m):.3e} |
| Max moving-wall flux error | {v['details']['moving_wall_flux']['maximum']:.3e} |
| Max contact relative normal velocity error | {v['details']['contact_inward_relative_normal']['maximum']:.3e} m/s |
| Max flux divergence | {v['details']['weighted_flux_divergence']['maximum']:.3e} s⁻¹ (criterion 1e−6) |
| Remaining fluid penetration | {v['details']['remaining_penetration']['maximum']:.0f} |
| Secondary penetration depth | {v['details']['whitewater_penetration']['maximum']:.3e} m |
| Missing stencils | {v['details']['missing_stencils']['maximum']:.0f} |
| Max unresolved contact fallback fraction | {v['details']['unresolved_contact_fraction']['maximum']:.4%} |
| Mesh/material volume range | {v['details']['reconstruction']['mesh_volume_ratio_range']} |
| Max adjacent normalized mesh volume change | {v['details']['reconstruction']['max_adjacent_normalized_volume_change']:.3%} |

The rising water volume tracks the persistent IDs of the initial pool particles and counts the volume of the particles with
`y > initial depth+h`, `vy > 0.5 m/s`. It is a rise metric of waves/water columns and does not mean that all of it is detached droplets.
The height is measured as the particle y p90 + half spacing of the existing 16 cm tiles and is distinguished from an actual level set.
The accumulation check of a scene with an initial water layer confirms that, in addition to the existing water, at least `0.8 Q max(t−2,0)` has gathered on the floor.
The pressure residual criterion, the penetration criterion and the [0.8,1.2] mesh volume criterion were kept.

With the final water, character, spray and lighting left identical, **the video with only the foam hidden and the video with it visible** were compared.
Pixels with RGB minimum >120, channel difference <65 and luminance increase >30 are counted. The criterion was set to at least 0.2% of the whole frame,
and in the floor ROI at least 0.1% of the whole frame. It is a visual contribution check that also includes the effect of
indirect reflection and does not replace a physical accuracy check. The images were also inspected directly.

| Time | White contribution pixels | Floor ROI contribution pixels |
|---|---:|---:|
{visible}

In the restart comparison the maximum APIC x/v/C difference was {restart['raw_particle_x_v_C_max_difference']:.3e}.
The secondary position difference was {restart['whitewater_position_error_m']:.3e} m and the velocity difference
{restart['whitewater_velocity_error_m_s']:.3e} m/s. The initial 2e−8 whole-state criterion was not passed.
A control that repeated the same continuous run without a restart also showed a difference of {restart['continuous_repeat_whitewater_difference']:.3e}.
To distinguish the effects of the P2G floating-point atomic order and the nonlinear surface correction,
it was checked that the difference is within twice the continuous-repeat error and at the same time within 8e−6 m (1e−4 voxel) in position and 1e−5 m/s in velocity.
ID, kind, radius and lifetime matched. The existing 2e−8 criterion of APIC was kept.

## Reproduction

```bash
EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
PY=/tmp/fluid-render-venv/bin/python
cmake --build build/release --target ex_fluid_apic_scene_cuda ex_fluid_apic_surface ex_fluid_apic_water_verify -j3
build/release/nanovdb/nanovdb/examples/ex_fluid_apic_water_verify "$EX/output/character"
$PY "$EX/scene/pipeline.py" "$EX/scene/wet_pool_profile.json" --stages simulate,mesh,render,validate
$PY "$EX/scene/verify_restart.py" --output "$EX/output/wet_pool_diagnosis/restart" --initial-depth .064 --foam-radius-scale 1.5 --foam-life-scale 1.8
$PY "$EX/scene/pipeline.py" "$EX/scene/wet_pool.json"
$PY "$EX/scene/verify_white_foam.py" "$EX/output/scene_wet_pool" --frame 180
$PY "$EX/scene/verify_white_foam.py" "$EX/output/scene_wet_pool" --frame 360
$PY "$EX/scene/validate_scene.py" "$EX/output/scene_wet_pool" --workers 4
$PY "$EX/scene/report_wet_pool.py"
```

The diagnostic surface was made by running simulation and CPU meshing concurrently. The final isovalue 0.2 reconstruction
processed independent frame segments in parallel on the CPU and merged them. The normalized volume change at the segment boundaries
was recomputed with the volume ratio of the preceding frame. The PCA covariance uses 30% of the previous frame, so
it was recomputed from 20 frames before each split boundary, and the 20 frames after the boundary were replaced.
The decay factor of the history error is 0.3²⁰, and at frames 111/201/291, where it returns to the original segments, the triangle connectivity is
identical and the vertex difference was at most 3.73e−9 m (`wet_pool_diagnosis/temporal_seams.json`).
The pipeline command above applies the temporal filter continuously, without splitting.
The last mesher argument `WAIT_FOR_FRAMES=1`
checks nextFrame of the atomic checkpoint and reads only frames whose outputs are all closed. That waiting time
is not included in the meshing measurement of each frame. The command is as follows (used while the simulation is running separately).

```bash
build/release/nanovdb/nanovdb/examples/ex_fluid_apic_surface "$EX/output/scene_wet_pool" "$EX/output/scene_wet_pool/surface" .01 .000004096 .016 .2 2.4 .12 0 360 1 1 1
```

The OBJs are stored as they are, and for the renderer connection a `.meshbin` cache of the same float32 vertices/int32 triangles was added.
This is the format read by the example's USD converter. The input format/code of ovrtx itself was not changed.
A watertight/winding check of all OBJs and a binary/OBJ value comparison are performed.

## Timings and limitations

| Stage | substep mean ms | total s |
|---|---:|---:|
{perf}

| CPU meshing | frame mean ms | total s |
|---|---:|---:|
{mp}

A short diagnostic GPU run was run concurrently during the initial interval, and meshing ran concurrently on the CPU.
The measured values are the actual wall-clock measurements of this work, not a benchmark of a standalone run.
The observed maximum GPU usage is {v['performance']['gpu_used_mb']['max']:.1f} MiB.
File transfer, output, meshing and render costs are recorded separately from the solver.
Process times are in `pipeline_times.json`, and renderer frame times are in `render/timings.json`.

The foam size and generation rate are settings of a secondary effect that can be seen by eye, not a microscopic physical model of bubbles.
The fluid grid is 32 mm, so very thin water films/small splashes hit the resolution limit.
The sides of the actual domain are closed up to the full height, and the render shows only the bottom 30 cm. There is no overflow model.
The volume and frame-change validation does not prove the absence of every local flicker or convergence to real water.

## Results

- [12-second final video](output/scene_wet_pool/dance_water.mp4)
- [Comparison with the existing scene over the first 6 seconds](output/scene_wet_pool/comparison_first_6s.mp4)
- [Representative images at 0/3/6/12 seconds](output/scene_wet_pool/representative_frames.jpg)
- [Graphs of foam, water level and rising water volume](output/scene_wet_pool/water_and_foam.png)
- [Particle/obstacle debug video](output/scene_wet_pool/debug.mp4)
- [6-second foam visibility comparison](output/scene_wet_pool/foam_visibility/comparison_000180.jpg)
- [12-second foam visibility comparison](output/scene_wet_pool/foam_visibility/comparison_000360.jpg)
- `output/scene_wet_pool/`: the full simulation cache, `surface/`: OBJ+binary+foam cache,
  `render/`: 361 PNGs, `scene.usdc`: 3D animation, `validation.json`, `pool_validation.json`: measured values.
"""
    (ROOT / "WET_POOL_RESULTS.md").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
