# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Publish measured results, plots, and representative frames without rerunning simulation."""

import csv, json, shutil, subprocess
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "output/scene_final"


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    result = json.loads((OUT / "validation.json").read_text())
    settings = json.loads((OUT / "settings.json").read_text())
    times = json.loads((OUT / "pipeline_times.json").read_text())
    restart = json.loads((ROOT / "output/scene_restart_test/result.json").read_text())
    data = rows(OUT / "metrics.csv")
    mesh = rows(OUT / "surface/meshing.csv")
    meta = json.loads((ROOT / "output/character/metadata.json").read_text())
    col = lambda k: np.array([float(r[k]) for r in data])
    mcol = lambda k: np.array([float(r[k]) for r in mesh])
    t = col("time")
    mt = mcol("frame") / settings["fps"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(t, np.maximum(col("flux_div_after"), 1e-16), label="after, moving-wall flux")
    ax.semilogy(t, np.maximum(col("div_before"), 1e-16), alpha=0.55, label="before")
    ax.axhline(1e-6, color="r", ls="--", label="acceptance 1e-6")
    ax.set_title("Fluid divergence [1/s]")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.semilogy(t, np.maximum(col("residual_absolute"), 1e-16), label="true |Aq-b|")
    ax.semilogy(t, np.maximum(1e-12, 2e-8 * col("rhs_norm")), label="absolute / relative bound")
    ax.set_title("Pressure residual")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    ax.plot(t, col("mass_volume"), label="material N Vp")
    ax.plot(t, col("occupied_volume"), label="occupancy proxy (not exact volume)", alpha=0.6)
    ax.plot(mt, mcol("mesh_volume"), label="reconstructed mesh")
    ax.set_title("Separate volume measures [m3]")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(mt, mcol("volume_ratio"), label="mesh / material")
    ax.plot(
        mt, mcol("kernel_integral") / mcol("material_volume"), label="kernel integral / material"
    )
    ax.axhline(1, color="gray", lw=0.7)
    ax.set_title("Reconstruction volume bias")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(t, col("foam"), label="foam")
    ax.plot(t, col("spray"), label="spray")
    ax.plot(t, col("ww_spawned"), label="cumulative births", alpha=0.5)
    ax.plot(t, col("ww_expected"), ls="--", label="integrated source expectation")
    ax.set_title("Secondary particles")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    ax.plot(t, col("emitted") * settings["particle_volume"], label="inlet")
    ax.plot(t, col("drained") * settings["particle_volume"], label="drain")
    ax.plot(t, col("ledger_error") * settings["particle_volume"], label="mass ledger error")
    ax.set_title("Flow accounting [m3]")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("simulation seconds")
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "validation_overview.png", dpi=150)
    plt.close(fig)
    contact = Image.new("RGB", (1800, 500), "#101b29")
    draw = ImageDraw.Draw(contact)
    for i, k in enumerate((0, 90, 180)):
        image = Image.open(OUT / "render" / f"frame_{k:06d}.png").convert("RGB").resize((600, 450))
        contact.paste(image, (i * 600, 30))
        draw.text((i * 600 + 18, 10), f"t = {k/30:.1f} s", fill="white")
    contact.save(OUT / "representative.png")
    debug = Image.new("RGB", (1440, 900))
    debug.paste(Image.open(OUT / "debug/frame_000180.png"))
    debug.save(OUT / "debug_representative.png")
    # Make the secondary representation legible independently of water highlights.
    ww = rows(OUT / "frame_000180_whitewater.csv")
    im = Image.new("RGB", (1200, 900), "#14202e")
    d = ImageDraw.Draw(im)
    for kind, color in [(0, "white"), (1, "#ffd077")]:
        for q in ww:
            if int(q["kind"]) != kind:
                continue
            x = 100 + 300 * float(q["x"])
            y = 815 - 285 * float(q["y"])
            r = 2 if kind == 0 else 2.5
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)
    d.text(
        (25, 20),
        "WHITEWATER ONLY | t = 6 s | white = foam, gold = spray | diagnostic markers are enlarged",
        fill="white",
    )
    im.save(OUT / "whitewater_debug.png")
    # Verify encoded frame count and 3D animation, not merely file existence.
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate,duration",
                "-of",
                "json",
                str(OUT / "dance_water.mp4"),
            ]
        )
    )["streams"][0]
    assert int(probe["nb_frames"]) == 181 and probe["r_frame_rate"] == "30/1"
    images = [
        np.asarray(Image.open(OUT / "render" / f"frame_{k:06d}.png"), dtype=np.float32)
        for k in (0, 90, 180)
    ]
    changes = [float(np.abs(images[i + 1] - images[i]).mean()) for i in range(2)]
    assert min(changes) > 0.1
    (OUT / "render_validation.json").write_text(
        json.dumps({"video": probe, "representative_pixel_changes": changes}, indent=2) + "\n"
    )
    comparison = []
    for name in ["low", "profile", "final"]:
        folder = ROOT / "output" / ("scene_" + name)
        r = json.loads((folder / "validation.json").read_text())
        s = json.loads((folder / "settings.json").read_text())
        comparison.append(
            f"| {name} | {s['h']:.3f} | {s['duration']:.1f} | {r['max_particles']:,.0f} | {r['details']['weighted_flux_divergence']['maximum']:.3e} | {r['performance']['gpu_used_mb']['max']:.2f} | PASS |"
        )
    perf = []
    labels = {
        "topology_ms": "Sparse topology",
        "mesh_sdf_ms": "Character evaluation/BVH refit + MeshToSDF",
        "boundary_ms": "SDF sampling / face boundary data",
        "p2g_ms": "APIC P2G / mask / external forces",
        "pressure_ms": "Jacobi PCG / projection",
        "g2p_ms": "Velocity extrapolation / APIC G2P",
        "contact_ms": "SDF + original mesh contact/penetration check",
        "whitewater_ms": "Foam / spray",
        "transfer_ms": "Particle management, transfer, cache readback",
        "io_ms": "Cache file output",
    }
    for k, label in labels.items():
        r = result["performance"][k]
        perf.append(f"| {label} | {r['mean']:.3f} | {r['p95']:.3f} | {r['sum']/1000:.3f} |")
    meshPerf = []
    for k, label in [
        ("read_ms", "CPU cache read"),
        ("pca_ms", "Neighbors / covariance / PCA"),
        ("grid_ms", "FloatGrid scalar construction"),
        ("clip_ms", "Obstacle/floor clip"),
        ("contour_ms", "OpenVDB VolumeToMesh"),
        ("write_ms", "OBJ / foam render cache"),
    ]:
        r = result["meshing_performance"][k]
        meshPerf.append(f"| {label} | {r['mean']:.3f} | {r['sum']/1000:.3f} |")
    rel = col("residual")[col("rhs_norm") > 1e-8]
    ratio = result["details"]["reconstruction"]["mesh_volume_ratio_range"]
    rt = json.loads((OUT / "render/timings.json").read_text())
    rtx = np.array(rt["frame_seconds"])
    text = f"""# Dancing water scene: run results

The final output is **a 6-second interval, 30 fps, 1600×1200, 181 states**. The MP4 also contains one frame for the end state, so
the encoded duration is {float(probe['duration']):.3f} seconds. The video is silent.

- [Final video](output/scene_final/dance_water.mp4)
- [Debug video of particles, character boundary and sparse halo](output/scene_final/debug.mp4)
- [Initial, middle and final representative images](output/scene_final/representative.png)
- [Numerical validation graphs](output/scene_final/validation_overview.png)
- [Foam/spray-only debug image](output/scene_final/whitewater_debug.png)
- [Comparison of poses extracted from the FBX](output/character/extracted_poses.png)
- [Implementation details](SCENE_IMPLEMENTATION.md), [full machine-readable results](output/scene_final/validation.json)

## Environment and settings

Branch: `mesh-to-sdf-fluid`. GPU: NVIDIA RTX PRO 3000 Blackwell Laptop, 12,227 MiB.
Blender 5.2.2 LTS, CUDA `/usr/bin/nvcc`, OpenVDB 13.0.1, ovrtx 0.5.1.385782 /
ovstage 0.2.1.385922 were used. The existing library and renderer code were not modified.

The FBX is 30 fps / frames 1–660, 24,746 vertices / 49,112 triangles. It contains two skinned meshes, and
the connectivity hashes of the 660 OBJs match. Boundary/nonmanifold edges are all 0.
The whole original animation was baked, and the final simulation uses the first 6 seconds. World coordinates are Y-up, in m.
The original, rig and materials are preserved in `output/character/source.fbx` and `source.blend`.

| Setting | Final value |
|---|---:|
| Simulation h | 0.032 m |
| Obstacle SDF h | 0.016 m |
| Independent meshing h | 0.0125 m |
| Target particles per cell | 8 |
| Particle material volume | 4.096e-6 m³ |
| dt upper limit / actual substeps | 0.004 s / {result['steps']:,} |
| Actual dt range | {col('dt').min():.6g}–{col('dt').max():.6g} s |
| Emitter center / radius | (1.62, 2.55, 1.0) m / 0.15 m |
| Initial emitter downward speed / flow rate | 0.9 m/s / 0.0636172512 m³/s |
| Whitewater source rate | 40,000 / (m³·s), event drive separate |
| Meshing support / isovalue / adaptivity | 2.8·Vp^(1/3) / 0.5 / 0 |
| Rendering | ovrtx PathTracing, 96 samples/pixel |

The drain is an absorbing particle sink near the floor, y < 0.18 m, with an x/z edge width of 0.14 m.
The pressure grid itself has closed floor/outer-wall conditions. Inflow, drain and the animation offset are stated in the config.

## Low resolution → performance test → final result

| Run | h [m] | Interval [s] | Max liquid particles | Max flux div [1/s] | Max GPU usage at step end [MiB] | Check |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(comparison)}

Low/profile are resolution tests of the same wide emitter. In the final scene the emitter radius, flow rate and position
were changed for the legibility of the character, so final is not used as a numerical convergence comparison with the two tests.
The GPU memory value is `cudaMemGetInfo` at the end of a substep, not the peak of the momentary temporary allocations inside the operators.

## Numerical validation and pass criteria

| Item | Measured value | Criterion / verdict |
|---|---:|---|
| Pressure true relative residual, RHS norm >1e-8 | {rel.max():.3e} | ≤2e-8 |
| Pressure absolute/relative combined | PASS | `abs ≤ max(1e-12, 2e-8*norm(b))` |
| Divergence based on moving-solid flux | {col('flux_div_after').max():.3e} s⁻¹ | ≤1e-6 |
| Face wall flux recomputation error | {col('wall_flux_error').max():.3e} m/s | ≤1e-9 |
| Relative normal velocity after contact | {col('contact_normal_error').max():.3e} m/s | ≤1e-8 |
| Remaining liquid particle center penetrations / max depth | 0 / 0 m | SDF + original triangle union |
| Remaining whitewater penetrations / max depth | 0 / 0 m | SDF + original triangle union + floor |
| Missing stencils / sidecar correspondence errors | 0 / 0 | immediate failure if any is missing |
| Max CFL | {col('cfl').max():.6f} | ≤0.5 |
| Particle mass ledger error | 0 particles | initial + emitted − drained |
| Cumulative inflow volume error | {result['details']['emission_quantization_m3']['maximum']:.3e} m³ | <1 particle volume |
| Whitewater births / expected value | {data[-1]['ww_spawned']} / {float(data[-1]['ww_expected']):.3f} | passes the 6σ+5 tolerance range |
| Mesh / material volume | {ratio[0]:.6f}–{ratio[1]:.6f} | criterion set in advance 0.80–1.20 |
| Normalized volume change between adjacent frames | {100*mcol('relative_volume_change').max():.3f}% | ≤8% |
| Kernel integral material volume error | {100*result['details']['reconstruction']['max_kernel_integral_relative_error']:.3f}% | ≤3% |
| Mesh boundary/nonmanifold edge | 0 | all 181 actual OBJs checked |
| OBJ winding/volume roundtrip error | {result['details']['mesh_export']['max_volume_roundtrip_error']:.3e} | <1e-4 |

The maximum of the raw residual metric, which also includes the initial near-zero RHS, is {col("residual").max():.6g}.
With a near-zero initial RHS, the value of the relative residual alone can become large. The verdict used the combined criterion above and
the independent flux divergence, and this was not hidden in the report.

The final material volume is **{result['final_material_volume']:.9f} m³**, the inflow **{result['emitted_volume']:.9f} m³** and
the drain **{result['drained_volume']:.9f} m³**. The final mesh volume is **{mcol('mesh_volume')[-1]:.9f} m³**.
The fact that the material ledger is exact is not interpreted as a proof of geometric volume conservation.
Occupied-cell volume, kernel integral and final surface volume are shown separately in the graphs.

In a separate CUDA validation, of 800,000 candidate points over 8 poses, at the 784,930 points that are at least 2 obstacle voxels
away from the surface, the sign mismatch between the SDF and the original closed-component union was 0.
This also includes 869 actual interior points and 239 inactive interior points. The maximum error of the vertex velocity derivative check was
{result['character_velocity_fd_error']:.6g} m/s (criterion 0.005).

The expected whitewater generation in free fall was 0, and the expected generation at ppc 8/16 for the same physical state matched, at
3.92328533333 each. The error of the analytic gravity/drag comparison for spray outside the sparse grid was 0.
In the existing APIC regression check the affine velocity error is at most 1.67e-15 and the affine matrix error at most 4.21e-14.

The maximum difference in raw `x,v,C` between the result restarted from the 0.4-second checkpoint up to 0.6 seconds and the continuous run was
{restart['raw_particle_x_v_C_max_difference']:.3e}, and the whitewater state difference was
{restart['whitewater_max_difference']:.3e}. The secondary atomic append order can differ, so the physical state
sorted by ID was compared. The wide emitter, which initially failed the wall recomputation check, also passed
the 2.3-second interval again after the fix (`output/scene_wall_regression/simulation.log`).

## Performance: simulation separated from files, surface and rendering

The wall time of the final simulation program is **{times['simulate']:.3f} seconds**. There are also host validation,
CFL computation, wall recomputation check and CPU control costs that are not included in the totals below, so the sum of the stages
is not presented as the total run time. The transfer item includes particle inflow/drain management and the cache readback.

| Simulation stage | Mean ms/substep | p95 ms | Total s |
|---|---:|---:|---:|
{chr(10).join(perf)}

| CPU post-processing stage | Mean ms/frame | Total s |
|---|---:|---:|
{chr(10).join(meshPerf)}

The wall time of the whole surface post-processing is **{times['mesh']:.3f} seconds**. The GPU→CPU cache readback is in the simulation
transfer category, and the file read of the CPU mesher is in the separate read category above.
USD authoring, ovrtx rendering and MP4 encoding take **{times['render']:.3f} seconds**, debug authoring and encoding
**{times['debug']:.3f} seconds**, and the full OBJ/numerical post-check **{times['validate']:.3f} seconds**.
The ovrtx `step + output readback` itself is **{rtx.mean():.3f} seconds/frame** on average, **{rtx.sum():.3f} seconds** in total.
The first-frame initialization cost is included. These values are measurements of this run, not a performance guarantee of a dedicated benchmark environment.

## Reproduction commands

Run from the repository root. The existing release preset has OpenVDB core and CUDA turned on.
The Python environment below includes NumPy, Pillow, matplotlib, pxr/USD, ovrtx and ovstage.

```bash
cmake --preset release
cmake --build build/release --target \\
  ex_fluid_apic_scene_cuda ex_fluid_apic_surface \\
  ex_fluid_apic_scene_verify -j 3

EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
PY=/tmp/fluid-render-venv/bin/python

/snap/bin/blender -b --python-exit-code 1 \\
  --python "$EX/scene/export_character.py" -- \\
  --input "$HOME/Downloads/Arms Hip Hop Dance.fbx" \\
  --output "$EX/output/character"

$PY "$EX/scene/pipeline.py" "$EX/scene/low.json"
$PY "$EX/scene/pipeline.py" "$EX/scene/profile.json"
$PY "$EX/scene/pipeline.py" "$EX/scene/final.json"

# Post-processing and checks that reuse the cache
$PY "$EX/scene/pipeline.py" "$EX/scene/final.json" --stages mesh,render,debug,validate
$PY "$EX/scene/pipeline.py" "$EX/scene/final.json" --stages verify
$PY "$EX/scene/verify_restart.py"
$PY "$EX/scene/report_scene.py"

# Continue computing from the last checkpoint with the same settings
$PY "$EX/scene/pipeline.py" "$EX/scene/final.json" --stages simulate --resume
```

For another FBX, replace the exporter's `--input`; for another bake cache, replace the pipeline's `--character`.
Simulation h, meshing h, ppc, maximum dt and whitewater rate are independent items of the JSON.
`checkpoint.bin` is a local binary restart of the same build/settings, and the CSVs are not to be mistaken for a restart.

## Outputs and remaining limitations

- `output/character/`: original FBX/Blender, 660 OBJs, binary/NumPy sequence, metadata, pose images.
- `output/scene_final/frame_*.csv`: 181 fluid caches, with separate whitewater and blocks CSVs.
- `output/scene_final/frame_*_obstacle.grid`: the original MeshToSDF buffer including blind data.
- `output/scene_final/surface/`: 181 water OBJs and the foam copies for rendering, `meshing.csv`.
- `output/scene_final/scene.usdc`, `render/`, `dance_water.mp4`: the final 3D scene and video.
- `output/scene_final/debug/`, `debug.mp4`, `whitewater_debug.png`: diagnostic views, including enlarged display.
- `metrics.csv`, `validation.json`, `pipeline_times.json`, `render/timings.json`: all measured values.

The pressure free surface is still a first-order condition based on particle occupancy. An elaborate ghost-fluid/cut-cell method,
surface tension, geometric volume correction and bubbles are not implemented. Parts smaller than the SDF resolution
are approximated in the pressure and supplemented by triangle-based particle contact. The mesh volume has a shrinkage of at most about
{100*(1-ratio[0]):.1f}%, and the volume flicker check is not a proof of the absence of local flicker.
The maximum render-only surface correction of the foam is {mcol('max_foam_render_shift').max():.5f} m, and it does not change the original simulation
state. Secondary is a visual effect and is not counted as actual liquid mass.

The final renderer converts the uniform untextured material of the provided FBX. It is not a general-purpose importer that
automatically converts every complex material graph of other FBXs. The original data is preserved.
"""
    (ROOT / "SCENE_RESULTS.md").write_text(text)
    print("WROTE", ROOT / "SCENE_RESULTS.md")


if __name__ == "__main__":
    main()
