# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Final artifacts and numerical report for grounded feet and a wider shower."""

import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/scene_grounded_shower"
DIAG = ROOT / "output/foot_diagnosis"
OLD = ROOT / "output/scene_wet_pool"


def load(p):
    return json.loads(p.read_text())


def movie_pair(first, second, destination, slow=False):
    suffix = ",setpts=2*PTS" if slow else ""
    filters = (
        "[0:v]scale=960:720,drawtext=text='Before':x=20:y=20:fontsize=28:fontcolor=white[a];[1:v]scale=960:720,drawtext=text='After':x=20:y=20:fontsize=28:fontcolor=white[b];[a][b]hstack=inputs=2:shortest=1"
        + suffix
        + "[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(first),
            "-i",
            str(second),
            "-filter_complex",
            filters,
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
            str(destination),
        ],
        check=True,
    )


def main():
    v = load(OUT / "validation.json")
    pool = load(OUT / "pool_validation.json")
    foot = load(DIAG / "foot_motion.json")
    energy = [load(DIAG / f"{name}_energy.json") for name in ["foot", "pre_spike", "final"]]
    visible = load(OUT / "foam_visibility/frame_000360.json")
    assert v["pass"] and pool["pass"] and foot["pass"] and visible["pass"]
    metrics = list(csv.DictReader((OUT / "metrics.csv").open()))
    settings = load(OUT / "settings.json")
    config = load(ROOT / "scene/grounded_shower.json")
    video = load(OUT / "video_metadata.json")
    times = load(OUT / "pipeline_times.json")
    assert int(video["nb_read_frames"]) == 361 and video["r_frame_rate"] == "30/1"
    montage = Image.new("RGB", (1600, 1260), (20, 25, 30))
    draw = ImageDraw.Draw(montage)
    for k, frame in enumerate([0, 90, 330, 360]):
        im = Image.open(OUT / "render" / f"frame_{frame:06d}.png").convert("RGB")
        im.thumbnail((800, 600))
        x, y = k % 2 * 800, k // 2 * 630
        montage.paste(im, (x, y + 30))
        draw.text(
            (x + 12, y + 8), f"t={frame/30:.1f} s | grounded feet, 52 cm emitter", fill="white"
        )
    montage.save(OUT / "representative_frames.jpg", quality=95)
    feet = Image.new("RGB", (1920, 1500), (20, 25, 30))
    draw = ImageDraw.Draw(feet)
    for j, frame in enumerate([5, 15]):
        for i, name in enumerate(["scene_foot_original_test", "scene_foot_grounded_test_limited"]):
            im = (
                Image.open(ROOT / "output" / name / "render_feet" / f"frame_{frame:06d}.png")
                .convert("RGB")
                .resize((960, 720))
            )
            feet.paste(im, (i * 960, j * 750 + 30))
            draw.text(
                (i * 960 + 12, j * 750 + 8),
                f'{"Before" if i==0 else "After"} | t={frame/30:.3f}s | isolated foot-driven waves',
                fill="white",
            )
    feet.save(OUT / "feet_comparison.jpg", quality=95)
    movie_pair(
        ROOT / "output/scene_foot_original_test/dance_water_feet.mp4",
        ROOT / "output/scene_foot_grounded_test_limited/dance_water_feet.mp4",
        OUT / "feet_comparison_half_speed.mp4",
        True,
    )
    movie_pair(OLD / "dance_water.mp4", OUT / "dance_water.mp4", OUT / "before_after.mp4")
    vertices = np.load(ROOT / "output/character_grounded/vertices.npy", mmap_mode="r")
    old_vertices = np.load(ROOT / "output/character/vertices.npy", mmap_mode="r")
    head_ids = np.flatnonzero(old_vertices[0, :, 1] > 1.65)
    head = vertices[:361, head_ids, :][:, :, [0, 2]].mean(1)
    distances = np.linalg.norm(head - np.array([config["emitter_x"], config["emitter_z"]]), axis=1)
    assert distances.max() < config["emitter_radius"]
    peak = max(float(q["cfl"]) * settings["h"] / float(q["dt"]) for q in metrics)
    limited_fraction = max(
        int(q["limited_affine_rows"]) / (3 * int(q["particles"])) for q in metrics
    )
    perf = "\n".join(
        f"| {k} | {q['mean']:.3f} | {q['sum']/1000:.2f} |"
        for k, q in v["performance"].items()
        if k.endswith("_ms")
    )
    meshing = "\n".join(
        f"| {k} | {q['mean']:.3f} | {q['sum']/1000:.2f} |"
        for k, q in v["meshing_performance"].items()
    )
    energy_rows = "\n".join(
        f"| {q['time']:.1f}s | {q['apic_energy_before_J']:.6g} | {q['apic_energy_after_J']:.6g} | {100*q['removed_energy_fraction']:.3f}% | {q['mean_velocity_max_difference']:.3e} |"
        for q in energy
    )
    text = f"""# Foot response at the floor and the widened water stream — 12-second results

## What the isovalue is

`mesh_surface.cc` stores the dimensionless render density that sums the anisotropic kernels,
`rho_render(x) = sum(Vp * W_A(x - xp))`, in an OpenVDB FloatGrid.
`volumeToMesh` extracts the surface where `rho_render(x) = isovalue`. The interior is mostly
at or above that value. The value this time is **{config['mesh_isovalue']}**, and the previous video used 0.2.
Lowering the value usually widens the surface. It is **a different value from the zero isosurface of the solid SDF, the APIC particle mass,
the fluid/air mask, the pressure solve and a physical viscosity coefficient**.
The volume reduction of the mesh and the kinetic energy loss of the simulation were recorded separately.
Particle positions, masses and velocities were not changed in order to match the volume.

## Foot response

The floor clearance of the existing FBX export option was 35 mm. The lowest point of the feet early on was about 37.4 mm, so
the depth submerged in the 64 mm initial water layer was smaller than one simulation voxel (32 mm).
Re-exporting with `--floor-clearance 0.001` lowered the early lowest point of the feet to about 3.4 mm.
The previous mesh/cache and the original FBX were preserved, and the triangle connectivity/FPS are the same.

The mesh-only contact against the lowered feet had an error that pushed water particles below the floor.
`resolveFloor` finds a nearby position outside the floor and the character, and `contactVelocity`
projects onto the allowed-velocity half-spaces of the two boundaries with the minimum change. It is applied to both APIC and secondary.
No arbitrary splash impulse or tangential friction was added. The error of the regression check of the
two-surface contact/separation velocity is 0, and the existing free-slip check also passed.

In a separate 0.8-second test in which the falling-water particles do not reach the floor, only the initial pool IDs were tracked.
Over 0–0.633 seconds, the mean velocity RMS within a 45 cm radius around the feet was **{foot['old_mean_rms_m_s']*100:.3f} →
{foot['new_mean_rms_m_s']*100:.3f} cm/s ({foot['mean_rms_ratio']:.3f} times)**.
The maximum particle height was {foot['grounded_peak_height_m']*100:.3f} cm. Small ripples were confirmed in the enlarged images and the
0.5x-speed comparison video. The two renders of the foot comparison both used isovalue 0.2,
which is distinct from the 0.3 of the final falling-water video. The feet themselves do external work, so this energy increase
is not used as a proof of energy conservation/no damping.

## Water stream

| Item | Previous | Current |
|---|---:|---:|
| Emission diameter | 30 cm | 52 cm |
| x/z center | (1.62,1.0) m | (1.67,1.145) m |
| Outlet speed | 0.9 m/s | 0.36 m/s |
| Flow rate | 63.617 L/s | {settings['flow_m3_s']*1000:.3f} L/s |

The flow rate is computed as `pi*r²*speed`. Instead of only enlarging the diameter, which would make the flow rate about three times as large,
the increase was limited to about 20%. The center of the fixed set of head vertices is, over the 12 seconds, at most
{distances.max()*100:.2f} cm away from the center of the emission circle, within the 26 cm radius. This is a placement check, and
the actual water impact was also confirmed in the 11/12-second renders.

## Free-surface affine amplification and dissipation

In the widened water stream, a very small number of particles became abnormally fast at about 1.13 seconds. The pressure residual and
CFL passed, but over the whole record the maximum velocity component rose to about 1051 m/s.
That result is preserved for diagnosis in `scene_grounded_shower_unlimited` and was not used in the final video.
In a reproduction from the 1.1-second checkpoint the P2G peak grew to 137.65 m/s and the values before and after pressure were the same, which
narrowed the cause down to extrapolation/affine re-transfer amplification before the pressure solve.

Currently, only in **stencils that include an extrapolated face**, the C row is shrunk when the affine reconstruction leaves the
min/max of the input velocity components. The mean particle velocity v is not changed, and it is not applied to fully projected
bulk stencils or to an exact affine field. It is neither a velocity clamp nor a viscosity term.
However, there is numerical damping of unresolved affine motion, and this is not hidden.
The peak of the same reproduction interval was within 6.37 m/s.

- Constant velocity transfer error: v 9.99e−16, C 2.43e−14.
- Affine field transfer error: v 1.67e−15, C 4.16e−14.
- Fixed/sparse topology, CSV difference of the same 0.15-second scene: at most 1.0e−10.
- The RMS of the foot test is about 14% smaller than for the same lowered character before the stabilization. Even so it is
  {foot['mean_rms_ratio']:.2f} times that of the previous scene, and the ripple improvement was validated again in the final limiter state.
- The result of a diagnostic that ran G2P on the same projected grid, without force/advection, at one checkpoint is as follows.
  `E = sum(0.5*m*(|v|² + h²/4*||C||²))`. This is the effect of a single limiter application, not a cumulative damping rate.

| Snapshot | APIC E before limiting (J) | After limiting (J) | Removed fraction | Mean velocity difference |
|---|---:|---:|---:|---:|
{energy_rows}

The overall maximum fraction of `limited_affine_rows` is {limited_fraction:.2%}. In the 2-cell-deep initial water layer
most stencils include free-surface/floor ghosts, and small velocity components are also included in this count.
Therefore the application count alone is not interpreted as an energy loss rate. This does not prove convergence to real water over a long time or
the damping rate of every wavelength.

## Full validation

{v['steps']:,} substeps, 361 frames, 30 fps, 12 seconds of simulation: pressure/contact/mass/sparse/surface checks PASS.

| Item | Result |
|---|---:|
| Final APIC particles | {int(v['final_particles']):,} |
| Final water volume (particle mass) | {v['final_material_volume']*1000:.3f} L |
| Water within 25 cm height above the floor | {pool['final']['bed_volume_m3']*1000:.3f} L |
| Max pressure relative residual | {max(float(q['residual']) for q in metrics):.3e} |
| Max moving-wall flux error | {v['details']['moving_wall_flux']['maximum']:.3e} |
| Max flux divergence | {v['details']['weighted_flux_divergence']['maximum']:.3e} s⁻¹ |
| Max contact relative normal error | {v['details']['contact_inward_relative_normal']['maximum']:.3e} m/s |
| Remaining fluid/secondary penetration | {v['details']['remaining_penetration']['maximum']:.0f} / {v['details']['whitewater_penetration']['maximum']:.1e} m |
| Missing stencils / mass ledger error | {v['details']['missing_stencils']['maximum']:.0f} / {v['details']['mass_ledger']['maximum']:.0f} |
| Max velocity component over every substep | {peak:.3f} m/s |
| Mesh / material volume | {v['details']['reconstruction']['mesh_volume_ratio_range']} |
| Max normalized adjacent mesh volume change | {v['details']['reconstruction']['max_adjacent_normalized_volume_change']:.3%} |
| Last foam / spray | {metrics[-1]['foam']} / {metrics[-1]['spray']} |
| White foam contribution pixels at 12 seconds | {visible['white_contribution_pixels']} |

`--max-speed 50` is **a check that aborts the run** right after G2P and mesh contact; it does not limit the velocity.
The exact pass criteria/remaining measured values are in `validation.json` and `pool_validation.json`.

In a separate low-resolution restart test the raw APIC x/v/C passed the 2e−8 criterion, but
the secondary velocity difference was 1.0109e−5 m/s, slightly above the existing 1e−5 criterion. The position difference is
5.88e−6 m, and the velocity difference of the continuous repeat was also 3.39e−6 m/s. The threshold was not relaxed.
This is **an unresolved secondary restart precision limit**, and the final cache was run continuously without resume.
The measurements are in `foot_diagnosis/restart_limited/diagnostic.json`.

## Performance

| Stage | substep mean ms | total s |
|---|---:|---:|
{perf}

| CPU meshing | frame mean ms | total s |
|---|---:|---:|
{meshing}

The whole simulation run (including output) took {times['simulate']:.1f} seconds, and the per-frame compute total of CPU meshing was
{times['mesh_compute_sum_s']:.1f} seconds. The render pipeline (including USD construction/render/encoding) took
{times['render']:.1f} seconds, and the debug video generation {times['debug']:.1f} seconds.
These pipeline times are distinct from the pure solver times above.

CPU meshing ran concurrently with the simulation on sequential frames, so the PCA temporal filter was kept continuous.
The waiting time is excluded from the meshing frame measurement. Short diagnostics/renders ran concurrently, so these times are
not an isolated benchmark. File transfer/output/meshing/render are recorded separately from the solver.

## Reproduction and results

```bash
EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
PY=/tmp/fluid-render-venv/bin/python
blender -b --python "$EX/scene/export_character.py" -- --input "$HOME/Downloads/Arms Hip Hop Dance.fbx" --output "$EX/output/character_grounded" --floor-clearance .001
cmake --build build/release --target ex_fluid_apic_scene_cuda ex_fluid_apic_surface ex_fluid_apic_water_verify ex_fluid_apic_affine_verify -j3
$PY "$EX/scene/pipeline.py" "$EX/scene/grounded_shower.json"
$PY "$EX/scene/validate_scene.py" "$EX/output/scene_grounded_shower" --workers 4
$PY "$EX/scene/verify_white_foam.py" "$EX/output/scene_grounded_shower" --character "$EX/output/character_grounded" --frame 360
build/release/nanovdb/nanovdb/examples/ex_fluid_apic_affine_verify "$EX/output/character_grounded" "$EX/output/scene_grounded_shower"
```

- [Final 12-second video](output/scene_grounded_shower/dance_water.mp4)
- [Previous/current 12-second comparison](output/scene_grounded_shower/before_after.mp4)
- [Foot response comparison, 0.5x speed](output/scene_grounded_shower/feet_comparison_half_speed.mp4)
- [Enlarged foot images](output/scene_grounded_shower/feet_comparison.jpg)
- [Images at 0/3/11/12 seconds](output/scene_grounded_shower/representative_frames.jpg)
- [Foot motion measurement graph](output/foot_diagnosis/foot_motion.png)
- `output/scene_grounded_shower`: simulation cache, `surface/` mesh/foam cache,
  `scene.usdc` 3D animation, `render/` PNG, `debug.mp4`, various validation JSON/CSV.

All changes were confined to the example folder and the example CMake. The existing NanoVDB operators and the ovrtx SDK were not modified.
The resolution is h=32 mm, meshing=10 mm. Small water films/millimeter splashes are unresolved, and
the initial foot response is small ripples of shallow water, not a large splash. The mesh volume metric does not prove the absence of
every local surface flicker. The closed computational sides extend to the full height, and there is no overflow model.
"""
    (ROOT / "GROUNDED_SHOWER_RESULTS.md").write_text(text)
    print("PASS final report and videos")


if __name__ == "__main__":
    main()
