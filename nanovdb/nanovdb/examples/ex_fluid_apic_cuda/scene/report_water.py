# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Reproducible measurements and before/after presentation of the water fix."""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "output/scene_final"
NEW = ROOT / "output/scene_water"
DIAG = ROOT / "output/water_diagnosis"


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def particle_stats(folder, frame):
    p = np.genfromtxt(folder / f"frame_{frame:06d}.csv", delimiter=",", names=True)
    floor = p["y"] < 0.05
    speed = np.hypot(p["vx"][floor], p["vz"][floor])
    r = np.hypot(p["x"][floor] - 1.62, p["z"][floor] - 1)
    return {
        "floor_particles": int(floor.sum()),
        "floor_speed_mean": float(speed.mean()),
        "floor_speed_p90": float(np.quantile(speed, 0.9)),
        "floor_radius_p95": float(np.quantile(r, 0.95)),
    }


def main():
    validation = json.loads((NEW / "validation.json").read_text())
    assert validation["pass"]
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
    assert int(video["nb_read_frames"]) == validation["frames"]
    assert video["r_frame_rate"] == "30/1"
    images = [
        np.asarray(Image.open(NEW / "render" / f"frame_{f:06d}.png"), dtype=float)
        for f in (0, 90, 180)
    ]
    assert all(a.std() > 5 for a in images)
    assert np.abs(images[0] - images[-1]).mean() > 1
    comparison = {
        str(frame): {"before": particle_stats(OLD, frame), "after": particle_stats(NEW, frame)}
        for frame in (36, 90, 180)
    }
    m = table(NEW / "metrics.csv")
    oldm = table(OLD / "metrics.csv")
    meshing = table(NEW / "surface/meshing.csv")
    result = {
        "video": video,
        "unresolved_contact_droplets_max": max(int(r["unresolved_contact_droplets"]) for r in m),
        "unresolved_contact_fraction_max": max(
            int(r["unresolved_contact_droplets"]) / int(r["particles"]) for r in m
        ),
        "floor_comparison": comparison,
        "validation": validation["details"],
        "whitewater": {
            "before_final_foam": int(oldm[-1]["foam"]),
            "before_final_spray": int(oldm[-1]["spray"]),
            "after_final_foam": int(m[-1]["foam"]),
            "after_final_spray": int(m[-1]["spray"]),
            "after_total_births": int(m[-1]["ww_spawned"]),
            "render_buried_before": int(meshing[-1]["foam_buried_before"]),
            "render_buried_after": int(meshing[-1]["foam_buried_after"]),
        },
    }
    (NEW / "water_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    montage = Image.new("RGB", (1600, 3 * 630), (18, 22, 28))
    draw = ImageDraw.Draw(montage)
    for row, frame in enumerate((36, 90, 180)):
        for col, (folder, label) in enumerate(((OLD, "Before"), (NEW, "Free-slip + whitewater"))):
            image = Image.open(folder / "render" / f"frame_{frame:06d}.png").convert("RGB")
            image.thumbnail((800, 600))
            montage.paste(image, (col * 800, row * 630 + 30))
            draw.text(
                (col * 800 + 12, row * 630 + 2),
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
            "[0:v]scale=960:720,drawtext=text='Before':x=20:y=20:fontsize=28:fontcolor=white[a];"
            "[1:v]scale=960:720,drawtext=text='Free-slip + whitewater':x=20:y=20:fontsize=28:fontcolor=white[b];"
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
    rows = []
    for frame, value in comparison.items():
        a, b = value["before"], value["after"]
        rows.append(
            f"| {int(frame)/30:.1f} s | {a['floor_speed_mean']:.3f} → {b['floor_speed_mean']:.3f} m/s | "
            f"{a['floor_radius_p95']:.3f} → {b['floor_radius_p95']:.3f} m |"
        )
    d = validation["details"]
    ww = result["whitewater"]
    timing_path = NEW / "pipeline_times.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    visibility = json.loads((NEW / "render_visibility.json").read_text())
    assert visibility["pass"]
    performance = "\n".join(
        f"| {k} | {v['mean']:.3f} | {v['sum']/1000:.3f} |"
        for k, v in validation["performance"].items()
        if k.endswith("_ms")
    )
    text = f"""# Water behavior fix: causes and validation

Only the example code of the existing `mesh-to-sdf-fluid` was modified. The NanoVDB operators and the ovrtx library were not modified.
The original 6-second result `output/scene_final` was preserved, and the new result is in `output/scene_water`.

## Confirmed causes

1. There was no physical viscosity term to begin with. However, `normalize/project` set every MAC velocity component inside the solid
   to the wall velocity, and G2P read it as is, which produced an artificial no-slip damping.
   In a zero-gravity, dt=0 transfer test with a particle of horizontal velocity 1 m/s placed 0.1h above the floor,
   the old velocity was **0.595 m/s** after one transfer and **0.5901639344 m/s** after 100. In the bulk it was 1 m/s.
2. The contact correction removed both the velocity going into the wall and the velocity moving away from the wall.
   A test that gave 1 m/s outward on the actual character SDF lost about **0.999996 m/s** of velocity.
   On contact the whole APIC C was also reset to 0.
3. Spray turned into foam immediately if the liquid density at its birth position was high. The floor contact of secondary particles
   multiplied the tangential velocity by 0.8 every substep, applying a very strong friction that depended on the timestep.
4. The foam density surface and the rendering surface differed, and foam followed the solid-side density boundary or was buried inside the actual water mesh.
   At 1.2 seconds of the fixed scene, **549/772** centers were inside the water before the render correction. After the correction it was **7/772**.
   The presence of foam/spray was also confirmed in a diagnostic render of the existing scene with the water hidden.

## Implementation

- `Cell::u` is kept for pressure/wall flux only. `transport` starts from the open fluid faces and
  fills air and solid ghosts with 6 passes of face-neighbor average extension. G2P and secondary velocity sampling
  use `transport`. The artificial tangential friction that came from filling the solid ghosts with the wall velocity was removed.
  With 4 passes some diagonal stencils at curved contacts were not covered, and it was changed to 6 after confirming the validation failure.
  If the velocity extension does not reach a subcell droplet in a cell classified as solid, only that particle is moved ballistically, its C is reset and then
  the SDF contact is applied. It is measured separately as `unresolved_contact_droplets`, and a missing value in a represented fluid cell is still an error.
  Keeping every unprojected low-mass air face as a seed amplifies affine feedback and is not used.
- Particle contact removes the normal component only when `v_rel·n < 0`. Separation is allowed and the tangential C is kept.
  On collision `C ← (I − nnᵀ)C`. The closed face velocity of the pressure is still the wall velocity.
  Therefore `contact_normal_error` now means **max(0, −v_rel·n)**. A positive velocity during separation is not an error.
- Spray reabsorption is allowed only after 80 ms, when the density is high and the relative velocity is re-entering the liquid.
  The 20% friction per substep at the floor was removed. The existing time-based air drag `exp(−0.35 dt)` was kept.
  Foam and newly born secondary particles follow a density surface that does not mistake the solid interior/floor for air.
  The whitewater acceleration source also includes the moving-obstacle pre-contact impulse.
- The foam radius is 4–7 mm, and spray is the existing 2.5–5 mm. The generation rate was adjusted from 40,000 → 120,000 /(m³·s).
  This generation rate is a visual effect setting and does not affect the fluid mass or the pressure.
- The anisotropic support goes from 2.8 → 2.4 times the particle spacing, the render-only smoothing from 0.20 → 0.12,
  and the meshing h from 12.5 → 10 mm. After a comparison on the same cache, isovalue 0.5 → 0.4 was used.
  The mesh/material volume ratio of the 1.2-second frame improved from 0.786 → 0.955. The physical particles were not changed.
- The foam render copy, when the Newton surface correction selects the solid-side boundary, searches for a nearby liquid–air boundary
  in 6 directions. It checks the 6 cm movement limit, the obstacle clearance and the floor height. Foam still buried after the correction is also recorded in the CSV.
- The checkpoint magic of the new physical model is distinguished as `FLUID002`, so a restart from the old `FLUID001` is rejected.
- In concave regions the sampled SDF can be negative even outside the triangle mesh. The secondary fallback
  checks both conditions. At the reproduction position the SDF went from −3.408 mm → +0.360 mm, and the triangle interior test was false both before and after.
  This correction resolves only the negative SDF that remains after the existing iterative normal movement has finished.

## Numerical validation

`scene/verify_water.cu`: the horizontal velocity at the floor is **1 m/s** for both 1 and 100 repetitions. The velocity/C change of a separating contact is **0**.
At secondary floor contact the analytic air drag error is **0**, and the spray state is kept 2 ms after birth.
Bulk constant/affine APIC maximum velocity error ≤1.67e−15, C error ≤4.17e−14.
In the CSV of the 30-step fixed/dynamic topology comparison the maximum position difference is at most 1.1e−11 m and the velocity difference at most 1.1e−9 m/s.
0 sign errors at the 784,930 points of the independent SDF check (including 239 inactive interior points),
expected free-fall whitewater generation 0, ppc 8/16 normalization identical, error against the analytic solution for spray outside the sparse grid 0.

Full validation of the final {validation['steps']} substeps, {validation['frames']} frames: **PASS**.
Fallback for contact droplets unresolved by the grid: at most {result['unresolved_contact_droplets_max']} per substep,
at most {result['unresolved_contact_fraction_max']:.4%} of all particles. It is counted separately as a limitation of the pressure cells not resolving them.

| Item | Maximum | Criterion |
|---|---:|---:|
| Flux divergence including wall motion/blocked fraction | {d['weighted_flux_divergence']['maximum']:.3e} s⁻¹ | 1e−6 |
| Moving wall face flux error | {d['moving_wall_flux']['maximum']:.3e} | 1e−9 |
| Particle inward relative normal error | {d['contact_inward_relative_normal']['maximum']:.3e} m/s | 1e−8 |
| Remaining fluid particle penetration | {d['remaining_penetration']['maximum']:.0f} | 0 |
| Secondary penetration depth | {d['whitewater_penetration']['maximum']:.3e} m | 1e−7 |
| Missing stencils | {d['missing_stencils']['maximum']:.0f} | 0 |
| Mass ledger error | {d['mass_ledger']['maximum']:.0f} | 0 |

The pressure residual passed the `max(1e−12, 2e−8 × ||b||)` criterion at every step.
The free-surface/pressure classification and the particle mass are kept separate as before.

The comparison used the same h=0.032 m, ppc=8 and emitter position/flow rate/animation.
The floor metrics are measured on the APIC particles with y<0.05 m and are not a comparison of the same particle set.

| Time | Mean horizontal flow speed at the floor: previous → fixed | Floor radius p95: previous → fixed |
|---|---|---|
{chr(10).join(rows)}

Foam/spray of the last frame: previous **{ww['before_final_foam']}/{ww['before_final_spray']}** →
fixed **{ww['after_final_foam']}/{ww['after_final_spray']}**. Total births of new secondary particles: {ww['after_total_births']}.
Buried final foam centers: {ww['render_buried_before']} before the render correction → {ww['render_buried_after']} after.
An animation started from an empty whitewater mesh was also compared with loading the last frame directly.
The mean 8-bit RGB difference was {visibility['mean_absolute_8bit_pixel_difference']:.5f}/255, within the criterion of 0.5/255.
The subsequent geometry updates of the empty mesh work correctly, and the problem of particles missing in the renderer was not reproduced.
Meshing volume ratio range {d['reconstruction']['mesh_volume_ratio_range']}, max adjacent normalized volume change
{d['reconstruction']['max_adjacent_normalized_volume_change']:.4%}. The detailed verdicts are in `output/scene_water/validation.json`.

## Reproduction

Run from the repository root.

```bash
cmake --build build/release --target ex_fluid_apic_scene_cuda ex_fluid_apic_surface ex_fluid_apic_water_verify ex_fluid_apic_scene_verify -j3
EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
build/release/nanovdb/nanovdb/examples/ex_fluid_apic_water_verify "$EX/output/character"
/tmp/fluid-render-venv/bin/python "$EX/scene/pipeline.py" "$EX/scene/water.json"
/tmp/fluid-render-venv/bin/python "$EX/scene/render_scene.py" "$EX/output/scene_water" --character "$EX/output/character" --only-frame 180 --width 1600 --height 1200 --samples 96 --whitewater-mode diagnostic
/tmp/fluid-render-venv/bin/python "$EX/scene/verify_render_visibility.py" "$EX/output/scene_water"
/tmp/fluid-render-venv/bin/python "$EX/scene/report_water.py"
```

## Outputs and cost

- [Fixed video](output/scene_water/dance_water.mp4), [side-by-side comparison video](output/scene_water/before_after.mp4)
- [Early/middle/late comparison images](output/scene_water/before_after.jpg), [APIC/boundary debug video](output/scene_water/debug.mp4)
- [Whitewater diagnostic image](output/scene_water/preview_diagnostic/frame_000180.png)
- Cache `output/scene_water/frame_*.csv`, surface `surface/water_*.obj`, all images `render/`
- Reproduction settings `scene/water.json`, numerical comparison `output/scene_water/water_comparison.json`
- Regression test logs `output/water_diagnosis/`

Stage wall time (seconds): `{json.dumps(timings)}`.
This cache was resumed, after the secondary contact fix, from the validated checkpoint at 2.233333 seconds.
The `simulate` wall time is the cost of this resumed interval. The substep measurement totals below include the whole 0–6 seconds.
Below are the simulation measurement totals, with IO shown separately. The early regression checks partly overlapped with GPU use, so
they are not to be interpreted as a dedicated performance benchmark.

| Task | substep mean ms | cumulative s |
|---|---:|---:|
{performance}

## Remaining limitations

This fix removes the artificial adhesion/friction. The numerical dissipation of pure APIC, the cell-level fluid mask and
the thin water films that cannot be represented at the 3.2 cm simulation resolution remain. No claim is made that the viscosity coefficient,
surface tension, contact angle or fine-scale turbulence of real water are reproduced quantitatively.
Conservation of the mass ledger differs from geometric volume conservation, and the meshing volume ratio was also reported separately.
Foam/spray is a visual effect that does not feed back into the fluid, and the 6 cm render correction is not a physical movement.
The volume-based flicker check does not prove the absence of every temporal discontinuity of the local surface.
"""
    (ROOT / "WATER_FIX_REPORT.md").write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
