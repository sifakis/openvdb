# NanoVDB GPU APIC fluid example: setup and run guide

## Status on this branch (read first)

This directory was moved here as source only. It has **not** been built or run on this branch.

- **Origin.** `nanovdb/nanovdb/examples/ex_mesh_to_sdf_fluid_cuda` of the branch `mesh-to-sdf-fluid`
  (commit 38833f8e5, 2026-10-18), together with the changes that were uncommitted in that working tree on
  2026-11-20: 8 modified files and 5 new ones (`scene/profile_meshtosdf.cu`, `scene/report_meshtosdf_profile.py`,
  `scene/report_simulation_benchmark.py` and two result documents that were not moved).
- **Renamed.** The directory is `ex_fluid_apic_cuda`, and every `ex_mesh_to_sdf_fluid` in the sources, scripts
  and this guide (directory and target names) is now `ex_fluid_apic`. Nothing else was edited.
- **Not moved.** The `output/` directory, and every document except this guide: the implementation notes
  (`IMPLEMENTATION_SPEC.md`, `SCENE_IMPLEMENTATION.md`, `SCALABILITY.md`) and the result and benchmark reports.
  Links to them below are dead here; they are on the origin branch.
- **Sections 2 and 3 below describe the origin branch** (which branch to clone, how it was built). On this branch
  the example is not registered in `examples/CMakeLists.txt`. The block that registered it, with the new names:

```cmake
nanovdb_example(NAME "ex_fluid_apic_cuda")
if(TARGET ex_fluid_apic_cuda)
  target_compile_definitions(ex_fluid_apic_cuda PRIVATE CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)
  add_executable(ex_fluid_apic_scene_cuda ex_fluid_apic_cuda/scene/scene_main.cu)
  target_link_libraries(ex_fluid_apic_scene_cuda PRIVATE nanovdb)
  target_compile_definitions(ex_fluid_apic_scene_cuda PRIVATE CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)
  add_executable(ex_fluid_apic_profile_meshtosdf ex_fluid_apic_cuda/scene/profile_meshtosdf.cu)
  target_link_libraries(ex_fluid_apic_profile_meshtosdf PRIVATE nanovdb)
  target_compile_definitions(ex_fluid_apic_profile_meshtosdf PRIVATE FLUID_USE_FLOAT CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)
  add_executable(ex_fluid_apic_scene_float_cuda ex_fluid_apic_cuda/scene/scene_main.cu)
  target_link_libraries(ex_fluid_apic_scene_float_cuda PRIVATE nanovdb)
  target_compile_definitions(ex_fluid_apic_scene_float_cuda PRIVATE
    CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING FLUID_USE_FLOAT)
  foreach(precision double float)
    add_executable(ex_fluid_apic_scale_${precision}_verify ex_fluid_apic_cuda/scalable/verify_scalability.cu)
    target_link_libraries(ex_fluid_apic_scale_${precision}_verify PRIVATE nanovdb)
    target_compile_definitions(ex_fluid_apic_scale_${precision}_verify PRIVATE CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)
    if(precision STREQUAL "float")
      target_compile_definitions(ex_fluid_apic_scale_${precision}_verify PRIVATE FLUID_USE_FLOAT)
    endif()
  endforeach()
  add_executable(ex_fluid_apic_scene_verify ex_fluid_apic_cuda/scene/verify_scene.cu)
  add_executable(ex_fluid_apic_affine_verify ex_fluid_apic_cuda/scene/verify_affine_limiter.cu)
  target_link_libraries(ex_fluid_apic_affine_verify PRIVATE nanovdb)
  target_compile_definitions(ex_fluid_apic_affine_verify PRIVATE CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)

  add_executable(ex_fluid_apic_water_verify ex_fluid_apic_cuda/scene/verify_water.cu)
  add_executable(ex_fluid_apic_contact_escape_verify ex_fluid_apic_cuda/scene/verify_contact_escape.cu)
  target_link_libraries(ex_fluid_apic_contact_escape_verify PRIVATE nanovdb)
  target_compile_definitions(ex_fluid_apic_contact_escape_verify PRIVATE FLUID_USE_FLOAT CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)
  target_link_libraries(ex_fluid_apic_water_verify PRIVATE nanovdb)
  target_compile_definitions(ex_fluid_apic_water_verify PRIVATE CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)
  target_link_libraries(ex_fluid_apic_scene_verify PRIVATE nanovdb)
  target_compile_definitions(ex_fluid_apic_scene_verify PRIVATE CCCL_DISABLE_CUB_UMBRELLA_HEADER_WARNING)


endif()
if(TARGET openvdb AND NANOVDB_USE_OPENVDB)
  add_executable(ex_fluid_apic_surface_inspect ex_fluid_apic_cuda/scene/inspect_surface.cc)
  target_link_libraries(ex_fluid_apic_surface_inspect PRIVATE nanovdb)
  add_executable(ex_fluid_apic_surface ex_fluid_apic_cuda/scene/mesh_surface.cc)
  target_link_libraries(ex_fluid_apic_surface PRIVATE openvdb nanovdb)
endif()
```

### What has to change before it builds here

The code was written against the MeshToSDF of the origin branch, which differs from the one on this branch.

1. `Fluid.cuh` uses `nanovdb::cuda::DualDeviceBuffer`, which NanoVDB 14 removed. Device grids are now
   `GridHandle<cuda::Buffer<std::byte>>`.
2. `Obstacle.cuh` and `scene/Character.cuh` call `MeshToSDF::getHandle()` and
   `tools::cuda::sdf_detail::allocate`. On this branch MeshToSDF returns the grid and the signed distances with
   `getHandleAndSDF()`, and the sign of the inactive voxels comes from `tools::cuda::PropagateSign` (invert masks
   per leaf, lower and upper node, plus root origins).
3. `Obstacle.cuh` and `scene/RawSdf.h` decode the old layout of that sign information (a dense root bitmap plus
   `root_extent`). They have to read the invert masks of PropagateSign instead; `levelset::detail::isInverted` in
   `LevelSetTracker.cuh` does that lookup.
4. `scene/profile_meshtosdf.cu` times the internal stages of the origin branch's MeshToSDF and has to follow the
   stages of this branch.

The rest of this guide is unchanged from the origin branch.

---

This document is **the starting point for a person or an agent on a new Linux/NVIDIA computer to run
everything from the source checkout to simulation, surface meshing, debug images and the final video**.
It covers both the basic APIC demo and the dancing-character/continuous-falling-water scene.

## Benchmark units and the current barrier mode

The simulation benchmark uses **ms/simulation step** as its basic unit. Here a step is a
solver substep, not an output video frame. `simulation_steps.csv` stores the time and the
active count of every step. The per-output-frame totals are for reference only. The geometry-only MeshToSDF replay is
reported in **ms/replay step (= one call)** and is not mixed with the fluid simulation measurements.

The current example uses `BarrierSigning::Interior` and records the mode in the `settings.json` of a new scene run. The existing videos and the 2026-10-15 Heuristic
benchmark numbers are preserved as they are and are not treated as results of the new mode. Interior classifies barrier
voxels as interior, so the sign/collision position near the surface can differ from Heuristic.
The existing accuracy check of the cube with a probe 2 cm outside the surface failed in Interior (error about 4 cm, threshold 1.25 cm).
This failure was not hidden, and the validation scope of the new mode is recorded in the last section of MESHTOSDF_BREAKDOWN_BENCHMARK.md.
Resuming from an existing Heuristic checkpoint with the current binary changes the mode, so do not
regard it as a continuation of the same physical settings; validate in a new output instead.

## 1. What to know first

- The branch is `mesh-to-sdf-fluid`, and the example directory is
  `nanovdb/nanovdb/examples/ex_fluid_apic_cuda`.
- The solver, CUDA kernels, obstacle adapter, mesher and renderer connection code are in this directory.
  The only change outside the example is `nanovdb/nanovdb/examples/CMakeLists.txt`, for build registration.
- It reuses the existing NanoVDB `MeshToSDF` and topology operators and OpenVDB `volumeToMesh`.
  The experimental `MeshToSDF_NEW` and `PropagateSign_*` code is not used.
- The FBX/mesh sequence, simulation cache, checkpoints, videos and Python environment under `output/` are
  **not included in Git**. They have to be generated on the new computer or copied separately.
- `CMakePresets.json` is a local file of the previous development machine and is not included in this commit.
  Use the explicit configure command below.
- `scene/pipeline.py`, `scene/overnight_run.py` and the benchmark/some validation scripts look for
  the executables in **`build/release/nanovdb/nanovdb/examples` of the repository**.
  The guide below builds into this path. Binaries in another build directory have to be run directly.
- The current solver is **single GPU**. It does not automatically combine the VRAM of two GPUs or move to another
  GPU on OOM. Select the GPU per run with `CUDA_VISIBLE_DEVICES`.
- Do not modify the existing library or the ovrtx renderer. Do not hide a numerical failure by changing a
  tolerance; keep the log and the settings that reproduce it and investigate the cause.

## 2. Environment check and checkout

```bash
git clone --branch mesh-to-sdf-fluid https://github.com/JaeHyunLee94/openvdb.git
cd openvdb
export EX="$PWD/nanovdb/nanovdb/examples/ex_fluid_apic_cuda"
export BINS="$PWD/build/release/nanovdb/nanovdb/examples"
export PY="$EX/output/render-venv/bin/python"
export CHARACTER="$EX/output/character_grounded"
export CUDA_VISIBLE_DEVICES=0

nvidia-smi
nvcc --version
cmake --version
ninja --version
python3 --version
blender --version
ffmpeg -version
git rev-parse HEAD
```

If you have already checked out, confirm that it is the right branch and preserve existing dirty files/caches.
`EX`, `BINS`, `PY` and `CHARACTER` are used by the later commands of this document. In a new shell,
move to the repository root and set them again. If `blender` is not on PATH, replace it with the actual executable path.

Required components:

| Component | Purpose/what to check |
|---|---|
| NVIDIA driver + CUDA toolkit | GPU kernel execution and `nvcc`; the toolkit must support the target GPU and the host compiler |
| C++17 compiler, CMake, Ninja | C++/CUDA build; check the minimum-version checks of the repository CMake |
| Boost iostreams, TBB, Imath, Blosc, zlib | OpenVDB core and CPU meshing |
| Blender's FBX importer | Export of the mesh sequence with armature/skinning applied |
| Python venv + `scene/requirements-render.txt` | Validation/analysis, USD authoring, ovrtx/ovstage rendering |
| FFmpeg | Encoding the PNG sequence to H.264 MP4 |
| NVML / `nvidia-smi` | Observing GPU memory in long runs and benchmarks |

It was validated on the Linux build/run path. The supervisor uses `fcntl`, `/proc` and Linux NVML,
so do not assume that it runs as is on Windows.

An example of installing the usual development packages on Ubuntu-family systems follows. System installation
follows the permission policy of that computer. Prepare CUDA/the driver separately.

```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build git python3-venv \
  libboost-iostreams-dev libtbb-dev libimath-dev libblosc-dev zlib1g-dev ffmpeg
```

**Even when the package names are the same, you must check that the versions are sufficient.** This branch's
`cmake/config/OpenVDBVersions.cmake` requires Boost >= 1.82, Imath >= 3.2,
TBB >= 2020.3 and Blosc >= 1.17.0. If the distribution packages are old, install suitable versions
into a separate prefix and add `-DCMAKE_PREFIX_PATH=/path/to/dependency-prefix` to the configure
command. Do not solve it by turning off the version checks. Do not assume that the `~/local` or `ext/imath`
installation of the development machine also exists on the new machine.

## 3. CUDA/OpenVDB build

The A6000 has compute capability 8.6, so the example below uses `86`.
On another GPU, change it to the corresponding architecture supported by the installed nvcc.
The measurement binary on the existing Blackwell laptop used `90-virtual` PTX, which differs from the A6000 setting.

```bash
cmake -S . -B build/release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DOPENVDB_BUILD_CORE=ON \
  -DOPENVDB_BUILD_BINARIES=OFF \
  -DOPENVDB_BUILD_UNITTESTS=OFF \
  -DOPENVDB_BUILD_NANOVDB=ON \
  -DNANOVDB_USE_OPENVDB=ON \
  -DNANOVDB_USE_CUDA=ON \
  -DNANOVDB_BUILD_EXAMPLES=ON \
  -DNANOVDB_BUILD_TOOLS=OFF \
  -DNANOVDB_BUILD_UNITTESTS=OFF

cmake --build build/release --target \
  ex_fluid_apic_cuda \
  ex_fluid_apic_scene_cuda \
  ex_fluid_apic_scene_float_cuda \
  ex_fluid_apic_surface \
  ex_fluid_apic_scene_verify \
  ex_fluid_apic_scale_float_verify \
  ex_fluid_apic_scale_double_verify -j 3
```

- CPU surface reconstruction needs **both OpenVDB core and `NANOVDB_USE_OPENVDB=ON`**.
- If build RAM is short, reduce to `-j 1`. To change the compiler/architecture of an old build,
  preserve the existing cache and check with a separate fresh build.
- The NanoVDB CMake fetches CCCL from `main` on GitHub. The first configure needs network
  access, and a reproduction report records the OpenVDB and CCCL commits and the toolkit/compiler versions.
- Consider the configure option below, which was used in the existing development environment, only when an
  AMX intrinsic header error occurs with the Ubuntu CUDA/GCC combination. It is not a generally required option:
  `'-DCMAKE_CUDA_FLAGS=-Xcompiler=-D_AMXTILEINTRIN_H_INCLUDED -Xcompiler=-D_AMXINT8INTRIN_H_INCLUDED -Xcompiler=-D_AMXBF16INTRIN_H_INCLUDED'`.
- The CPU/GPU adapters distinguish and read both the dense `uint8` root bitmap + `root_extent` of the existing
  MeshToSDF and the more recent sparse `int32` root-origin list. It builds without library changes.

## 4. Preparing Python and the renderer

```bash
python3 -m venv "$EX/output/render-venv"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$EX/scene/requirements-render.txt"
"$PY" -c 'import numpy, PIL, matplotlib, scipy, ovrtx, ovstage; from pxr import Usd; print("imports OK")'
"$PY" -m pip freeze > "$EX/output/render-venv-packages.txt"
```

These requirements are the pinned versions of the existing measurement environment. If the Python/platform has no wheel,
record the installation error and prepare a compatible Python environment first. Do not arbitrarily replace ovrtx and
ovstage with different API generations from each other. The README and examples of the separate ovrtx repository on the
existing development machine were used to investigate the connection, but running this example imports **the installed Python SDK**.
There is no need to create the specific path `~/Desktop/Code/NVIDIA/ovrtx` on the new machine.

A successful import does not mean that RTX rendering succeeds. Use the **representative frame rendering** below to
check the GPU/driver/headless environment as well, and then generate the full video. `debug_scene.py` also
uses an example helper that imports `pxr`, so the USD Python package is required.

## 5. Validate the basic solver first, without the character

```bash
"$PY" "$EX/validate.py" "$BINS/ex_fluid_apic_cuda" \
  --output "$EX/output/base_validation_new_machine"
"$BINS/ex_fluid_apic_scale_float_verify" 32 0 mgpcg
"$BINS/ex_fluid_apic_scale_double_verify" 32 0 mgpcg
"$PY" "$EX/scalable/run_verification.py" --max-size 32 --precision all
```

The basic validation runs in the order fixed topology → dynamic topology → static/moving/deforming cube.
It checks the constant/affine velocity APIC transfer, the pressure residual/flux divergence,
the comparison of fixed and dynamic results, and the obstacle SDF and contact. On a nonzero exit, do not
proceed to the next step. Run larger pressure problems after the small validation has passed.
This basic demo needs neither an FBX nor a character cache.

## 6. FBX inspection and grounded character export

The input file is provided separately by the user. The existing scene used `Arms Hip Hop Dance.fbx`.

### Downloading the input FBX from Mixamo

1. Log in to [Mixamo](https://www.mixamo.com/) with an Adobe ID. For the account/service terms, see the
   [Adobe Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html).
2. Select a character, and search for and apply **Arms Hip Hop Dance** in Animations.
   The mesh names of the existing input were `Beta_Joints` and `Beta_Surface`. The character's UI name and the
   motion slider values at the time of the original download were not recorded, so the same animation name alone
   cannot guarantee geometry/motion identical to the existing video. Compare the inspection values below and the preview.
3. Keep the whole animation range and download with **FBX Binary**, **With Skin**,
   **30 fps** and **Keyframe Reduction: None**. These are recommended settings for reproduction and
   are not a record that restores every choice of the original download dialog. If you also changed motion
   adjustment values such as Speed/Trim, record them separately. Downloading without With Skin gives no mesh, and the exporter fails.
4. Save the file to `~/Downloads/Arms Hip Hop Dance.fbx`. Another path can also be used through the
   `FBX` environment variable below. Because the service requires login/character selection, no direct FBX URL or
   automatic download script is provided.
5. Run the `--inspect-only` command below and check `inspection.json`, and after the full export check
   the topology validation result in `metadata.json` and the initial, middle and last
   poses of `source.blend` in Blender. The animation length must
   sufficiently cover simulation duration + animation offset.

The fingerprint of the existing input, for checking, is as follows. The hash of a new download can differ
depending on the FBX exporter's metadata and so on. If the hash differs, do not regard it as a byte-identical input;
compare the geometry, animation timing and preview.

| Property | Original input |
|---|---|
| File name | `Arms Hip Hop Dance.fbx` |
| File size | 3,455,696 bytes (3.296 MiB) |
| SHA-256 | `5b233159115778201f1de250a7d9df69d917b9cbd434d4a996610859a3502823` |
| Imported FPS / frame range | 30 fps / 1–660 |
| Evaluated vertices / triangles | 24,746 / 49,112 |
| Mesh objects | `Beta_Joints`, `Beta_Surface` |
| Import environment | Blender 5.2.2 LTS; evaluated world-space mesh |

```bash
sha256sum "$HOME/Downloads/Arms Hip Hop Dance.fbx"
```

The original FBX is not included in the repository. Adobe Community's
[Mixamo distribution guidance](https://community.adobe.com/questions-696/mixamo-faq-licensing-royalties-ownership-eula-and-tos-589400?lang=en)
explains that public redistribution of the original character/animation files is restricted. Each user downloads it
personally and checks the terms of use that apply. The `source.fbx`, `source.blend` and
OBJ sequence that the exporter creates also stay in the local `output/` and are not added to Git.

### Input inspection and generating the mesh sequence for simulation

```bash
export FBX="$HOME/Downloads/Arms Hip Hop Dance.fbx"
test -f "$FBX"
blender -b --python-exit-code 1 --python "$EX/scene/export_character.py" -- \
  --input "$FBX" --output "$EX/output/character_inspection" --inspect-only

blender -b --python-exit-code 1 --python "$EX/scene/export_character.py" -- \
  --input "$FBX" --output "$CHARACTER" \
  --height 1.75 --center-x 1.6 --center-z 1.2 --floor-clearance 0.001
```

The exporter uses Blender's evaluated mesh and world transform. It includes every mesh object and
fixes the triangulation of the first frame. It checks the vertex count/polygon connectivity every frame
and fails if the topology changes. It converts Blender Z-up to simulation Y-up `(x,z,-y)`
and stores in units of m. Preserve the original FBX/rig separately.

Artifacts to check: `inspection.json`, `metadata.json`, `sequence.txt`,
OBJs such as `frame_000001.obj`, `vertices.bin`, `triangles.bin`, `vertices.npy`,
`triangles.npy`, `source.fbx`, `source.blend` and the pose previews.
Check the first/middle/last poses, and the FPS, frame range, bounds, topology consistency and actual vertex deformation.
The original dance was 30 fps, source frames 1–660. Do not fit the results of another input to these numbers.
`--inspect-only` only performs the inspection and does not generate the full sequence for simulation.

```bash
"$BINS/ex_fluid_apic_scene_verify" "$CHARACTER" 0.01
```

`pipeline.py --character` takes the export **directory**. Do not pass a single FBX or OBJ file
directly. Check that the full bake sufficiently covers simulation duration + animation offset.
Without an FBX the basic cube demo can proceed, but the dance cannot be reproduced.

## 7. Latest scene settings and the low-resolution smoke test

The committed `scene/clear_shower_fine.json` is **the previous fine scene with h=0.02 m**.
The config of the 384 × 384 × 288, 10-second scene that was approved afterwards originally existed only in the ignored cache.
The command below generates that config and a short smoke config using the paths of the new environment.
The emitter/pool/animation/whitewater values of the scene are kept.

```bash
"$PY" - <<'PY'
import json, os
from pathlib import Path
ex = Path(os.environ['EX'])
cfg = json.loads((ex / 'scene/clear_shower_fine.json').read_text())
cfg.update(character=os.environ['CHARACTER'], h=3.2/384,
           obstacle_h=(3.2/384)/2, duration=10, precision='float', pressure='mgpcg',
           mesh_closing_radius=0.25, mesh_pca_threads=4, validation_workers=4)
folder = ex / 'output/configs'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'production_384.json').write_text(json.dumps(cfg, indent=2) + '\n')
smoke = dict(cfg, h=0.04, obstacle_h=0.02, duration=0.25,
             mesh_h=0.02, width=640, height=480, samples=16, validation_workers=1)
(folder / 'smoke.json').write_text(json.dumps(smoke, indent=2) + '\n')
PY
export CONFIG="$EX/output/configs/smoke.json"
export CACHE="$EX/output/smoke_new_machine"
"$PY" "$EX/scene/pipeline.py" "$CONFIG" --character "$CHARACTER" \
  --precision float --pressure mgpcg --output "$CACHE" --stages verify,simulate
"$PY" "$EX/scene/pipeline.py" "$CONFIG" --character "$CHARACTER" \
  --output "$CACHE" --stages mesh,validate
"$PY" "$EX/scene/debug_scene.py" "$CACHE" --character "$CHARACTER" --representative
```

The short smoke test is for checking the connection and the initial pool/foot contact. Do not regard it as
having validated the splash after the falling water reaches the floor or the stability of the 10-second scene.

Meaning of the settings:

| Item | Latest 384 setting / meaning |
|---|---|
| `h`, `obstacle_h` | 0.008333… / 0.004166… m; fluid/obstacle voxel size |
| `duration`, `fps` | 10 seconds / output 30 fps; differs from the number of solver substeps |
| `ppc`, `dt` | 8 / at most 0.004 seconds; the actual dt becomes smaller depending on CFL |
| `emitter_radius`, `emitter_speed` | 0.21 m / 0.36 m/s downward; diameter 0.42 m |
| `emitter_x`, `emitter_z` | 1.67 / 1.1551 m |
| `initial_depth`, `drain_enabled` | initial pool 0.064 m / false (closed basin) |
| `whitewater_rate` | 360000; coefficient of the secondary generation model |
| `foam_radius_scale`, `foam_life_scale` | 1.5 / 1.8; simulation secondary settings |
| `mesh_h` | 0.00625 m; meshing voxel size, independent of the simulation |
| `mesh_isovalue`, `mesh_support`, `mesh_smoothing` | 0.3 / 2.4 / 0.12 |
| `mesh_closing_radius` | 0.25; surface filter setting of the mesher |
| `mesh_pca_threads`, `validation_workers` | 4 / 4; CPU parallelism |
| `width`, `height`, `samples` | 1600 × 1200 / 96; final render |

The domain is X/Y/Z = 3.2/3.2/2.4 m. When raising the resolution, keep the same world size and
set `h=3.2/N`, `obstacle_h=h/2` in a separate config. Choose `mesh_h` independently.
Changing `ppc` affects mass and memory, so record it. The particle count/active cells/VRAM
change with time and sparse occupancy. There is no guarantee that a particular resolution
runs on 48 GB.

## 8. Representative frame rendering and the full run

First render one frame of the smoke cache. The mesh stage must have run before.

```bash
"$PY" "$EX/scene/render_scene.py" "$CACHE" --character "$CHARACTER" \
  --only-frame 0 --width 640 --height 480 --samples 16 --tag smoke_frame
```

Check the image and `render_smoke_frame/timings.json`. The single-frame test does not generate
a video. Resolve black images, non-convergence and SDK/driver errors here.

Run the full production in **a separate output** after the smoke validation has passed.

```bash
export CONFIG="$EX/output/configs/production_384.json"
export CACHE="$EX/output/scene_384_new_machine"
"$PY" "$EX/scene/pipeline.py" "$CONFIG" --character "$CHARACTER" \
  --precision float --pressure mgpcg --output "$CACHE" --stages simulate,mesh,validate
"$PY" "$EX/scene/debug_scene.py" "$CACHE" --character "$CHARACTER" --representative
"$PY" "$EX/scene/render_scene.py" "$CACHE" --character "$CHARACTER" \
  --width 1600 --height 1200 --samples 96 \
  --foam-render-scale 0.7 --spray-render-scale 0.8 --tag smaller_whitewater
```

The last command reproduces the approved small foam/spray look. It is **a radius scale applied only to rendering** and
does not change the simulation mass or the secondary cache. The render stage of `pipeline.py`
uses the default scale=1 and does not pass this look option. Use `--tag` to preserve the original render.
The generated video is `dance_water_smaller_whitewater.mp4`, and the images are in `render_smaller_whitewater/`.

To reuse the cache and rerun only the post-processing:

```bash
"$PY" "$EX/scene/pipeline.py" "$CONFIG" --character "$CHARACTER" \
  --output "$CACHE" --stages mesh,validate
"$PY" "$EX/scene/render_scene.py" "$CACHE" --character "$CHARACTER" \
  --only-frame 150 --camera-view feet --water-style clay --tag feet_clay
```

Use the last command only on a cache that has frame150. `--author-only` only authors the USD, and
`--surface-dir` specifies an alternative surface directory. Meshing has a temporal filter history, so
do not add an arbitrary middle frame in a cold state. Use full regeneration or the history replay of the next section.

## 9. Long runs, stopping and resuming

A normal simulation resume uses the same config, precision, character and cache.

```bash
"$PY" "$EX/scene/pipeline.py" "$CONFIG" --character "$CHARACTER" \
  --output "$CACHE" --stages simulate --resume
```

The checkpoint is saved at every output frame by an atomic rename from a temporary file. It contains positions, velocities,
APIC affine matrices, IDs and secondary state, and the CSVs are not restart input.
Compatibility of moving a binary checkpoint to another precision/build/layout is not guaranteed.
A new simulation uses a new output and does not overwrite an existing cache.

A dedicated supervisor can be used for float long-run management and CPU streaming meshing.

```bash
"$PY" "$EX/scene/overnight_run.py" "$CONFIG" "$CACHE" \
  --hours 12 --stream-mesh
# Continue while also checking the completed surface history:
"$PY" "$EX/scene/overnight_run.py" "$CONFIG" "$CACHE" \
  --resume --stream-mesh --resume-mesh --hours 12
```

The first command is for **a new cache**. Set `character` in `CONFIG` to an absolute path and `precision` to
float. The config generated above satisfies these conditions. `--hours` is the upper limit of the wall-clock run
time and differs from the simulation duration. `--postprocess` also runs the post-processing after completion.
If the GPU driver is reset, preserve the checkpoint and confirm GPU recovery separately.

```bash
touch "$CACHE/STOP"
# After confirming the stop and the checkpoint in status.json:
rm "$CACHE/STOP"
```

STOP is a request to stop near the next checkpoint, not an immediate kill. Remove STOP before
resuming. Do not run multiple supervisors on the same cache. `surface_resume.py` is
a **helper module** used by the supervisor, not a CLI to run directly. It replays the history of the
previous 32 frames and validates the overlapping meshes. **The supervisor does not automatically retry a numerical failure.**
On a nonzero exit, check the log, resolve the cause and then resume safely with the same settings.
It does not automatically relax tolerances or restart without limit.

## 10. Outputs, pass/fail judgment and benchmark

| Location | Contents |
|---|---|
| `pipeline_config.json`, `settings.json` | run settings / the actual settings recorded by the solver |
| `metrics.csv` | per-substep time, pressure/divergence, contact, CFL, region/mass, etc. |
| `frames.csv` | manifest of output frames and simulation time |
| `frame_XXXXXX.csv` | APIC particles |
| `frame_XXXXXX_whitewater.csv` | separately managed foam/spray |
| `frame_XXXXXX_blocks.csv` | debug information of the allocated sparse blocks |
| `checkpoint.bin` | the last complete simulation restart |
| `surface/` | water OBJ/meshbin, whitewater for rendering, meshing timing/report |
| `debug/` | representative images of particles, obstacle and blocks |
| `render[_tag]/` | final PNGs and `timings.json` |
| `dance_water[_tag].mp4` | final video |
| `pipeline_times.json` | wall time per stage |
| `status.json`, supervisor log | supervisor progress/log (when used) |

Judge pass or fail by the exit code and the validation report. Do not pass a run only because the images
look natural. Check the divergence based on `open*u + (1-open)*wallVelocity` of the moving wall, the pressure
residual, primary/secondary penetration, missing stencils and the mass ledger including inflow and outflow.
Distinguish the meshing volume from the particle volume and also check render convergence.
The high-resolution frame CSVs/meshes/full USD are large, so observe disk space and CPU RAM separately from GPU VRAM.

To run a short independent benchmark from an existing **float checkpoint/cache**:

```bash
"$PY" "$EX/scene/benchmark_pipeline.py" \
  --cache "$CACHE" --checkpoint "$CACHE/checkpoint.bin" \
  --output "$EX/output/benchmark_new_machine" --frames 3 --python "$PY"
"$PY" "$EX/scene/report_pipeline_benchmark.py" "$EX/output/benchmark_new_machine"
```

The benchmark output must be a new directory. It generates the 3 frames after the checkpoint, so check the required
animation time and the character path of the original config. The source cache is read-only.
It distinguishes simulation, CPU meshing and USD/render/PNG/encode, and records memory sampling, video decode
and numerical validation. Meshing starts from a cold covariance history, so the conditions are not the same as the
warm measurement of production. The detailed explanation and the existing measurements are in [PIPELINE_BENCHMARK.md](PIPELINE_BENCHMARK.md).

### Benchmark that instruments only the simulation

`benchmark_pipeline.py --simulation-only` passes `--profile 1` to the scene binary and additionally records
the time of `MeshToSDF::getHandle()` alone and the actual fluid/obstacle active voxel counts.
Rebuild the scene target so that the existing binary includes the profiling option.

```bash
python3 "$EX/scene/benchmark_pipeline.py" \
  --cache "$CACHE" --checkpoint "$CACHE/checkpoint.bin" \
  --output "$EX/output/benchmark_simulation_new_machine" --frames 3 --simulation-only
python3 "$EX/scene/report_simulation_benchmark.py" "$EX/output/benchmark_simulation_new_machine"
```

A separately built float executable can be specified with `--binary /absolute/path/to/binary`.
It writes the breakdown and the ratios to `simulation_summary.json`, `simulation_frames.csv` and `RESULTS.md`.
It distinguishes the denominators: simulation compute, substep including cache I/O, and process wall time
including startup. `fluid_active_voxels` also includes air/halo, and `fluid_pressure_cells` and
`leaves × 512` are separate numbers. The default instrumentation format of a normal scene run is kept.

## 11. Full index of scripts

The paths below are relative to this example directory. For the CLIs, check `--help` and the source.
**Some helpers and past experiment runners have no `--help`, so do not run them all in a batch.**

### Main scripts for running the scene

| File | Role and usage |
|---|---|
| `scene/export_character.py` | Blender-only FBX export; section 6 |
| `scene/pipeline.py` | stage-by-stage run with a config and `--stages`; sections 7–9 |
| `scene/overnight_run.py` | float long-run management and stream meshing; section 9 |
| `scene/surface_resume.py` | temporal mesh continuation helper; used by the supervisor |
| `scene/render_scene.py` | cache→USD/PNG/MP4; section 8 |
| `scene/debug_scene.py` | particle/obstacle/block images; `--representative` for 3 representative frames |
| `scene/validate_scene.py` | numerical/geometry validation of the cache; the validate stage of the pipeline |
| `scene/verify_pool.py` | mass, accumulation and boundary validation of the closed basin; also called from the validate stage |
| `scene/benchmark_pipeline.py` | measures the full pipeline by checkpoint continuation; section 10 |
| `scene/profile_meshtosdf.cu` | MeshToSDF geometry replay executable; internal phases and verbose diagnostics |
| `scene/report_meshtosdf_profile.py` | Replay CSV aggregation, graphs and active-count comparison |
| `scene/report_simulation_benchmark.py` | aggregation of simulation profiling results, active counts and numerical validation |
| `scene/report_pipeline_benchmark.py` | benchmark folder→summary/CSV/graphs; section 10 |

### Individual diagnostics and comparisons

| File | Role |
|---|---|
| `scene/inspect_pressure.py` | inspection of the stored pressure diagnostics |
| `scene/render_surface_diagnostic.py` | surface/particle comparison images |
| `scene/verify_foot_motion.py` | comparison of the motion around the feet between the grounded and the original bake |
| `scene/verify_mesh_edges.py` | water mesh edge/connectivity check |
| `scene/verify_pool_tiles.py` | check of the local tile distribution of the pool |
| `scene/verify_render_visibility.py` | foam/spray render visibility check |
| `scene/verify_restart.py` | scene restart reproduction validation |
| `scene/verify_secondary_roundoff.py` | validation of the secondary boundary round-off error |
| `scene/verify_surface_parallel.py` | comparison of the parallel CPU surface computation |
| `scene/verify_white_foam.py` | white foam visibility check |
| `scalable/run_verification.py` | float/double pressure comparison across sizes, shapes and methods |
| `scalable/compare_distribution.py` | reference/candidate cache distribution comparison |
| `scalable/report_scalability.py` | scalability regression aggregation |
| `scalable/verify_resident_restart.py` | GPU resident particle management/restart comparison |

### Basic three-stage demo

| File | Role |
|---|---|
| `validate.py` | runs the fixed/dynamic/cube basic validation in order |
| `validate_long.py` | long basic validation with a specified duration/voxel size |
| `render.py` | basic demo particle/block/SDF images and GIF |
| `render_all.py` | representative images of each mode of the basic demo |
| `animate.py` | renders the basic demo 3D frame sequence |
| `assemble_animation.py` | assembles the basic demo video |
| `report_long.py` | report of the basic long experiment |
| `compare_resolution.py` | basic demo resolution comparison |
| `scene_config.py` | common settings helper of the basic demo |

### Past experiment runners/reports: check the source before use

Some files assume past output names, baseline binaries and several configs.
The standard run path of the new environment is the pipeline/supervisor above. The JSONs/reports were preserved as development history.

| File | Role |
|---|---|
| `scene/finish_clear_shower.py` | driver that completes the past clear shower post-processing |
| `scene/plot_clear_shower.py` | clear shower measurement graphs |
| `scene/preview_clear_shower.py` | clear shower preview |
| `scene/report_grounded_shower.py` | aggregation of the grounded shower results |
| `scene/report_pool.py` | aggregation of the accumulating pool results |
| `scene/report_scene.py` | aggregation of the initial dance scene results |
| `scene/report_water.py` | aggregation of the water improvement experiment results |
| `scene/report_wet_pool.py` | aggregation of the initial pool experiment results |
| `scalable/final_checks.py` | past scalability final check driver |
| `scalable/finish_verification.py` | driver that continues the past verification |
| `scalable/validate_final.py` | validation driver of the past group of final configs |

The correspondence to the C++ implementation/validation sources is in [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md),
[SCENE_IMPLEMENTATION.md](SCENE_IMPLEMENTATION.md) and [SCALABILITY.md](SCALABILITY.md).
`scene/requirements-render.txt` is the Python dependencies, and `.gitignore` is the setting that excludes generated output.

## 12. Troubleshooting

- No binary: check the branch, target, `build/release` and `BINS`.
- CMake dependency/version error: check the prefix/version. The new environment has no local preset.
- CUDA architecture/host compiler error: check the driver/toolkit/compiler combination.
- FBX import/topology error: check the Blender importer, the evaluated modifiers and the connectivity of every mesh.
- RTX initialization/black image: do not judge by the SDK import alone; check a single-frame render and the log.
- OOM: preserve the checkpoint and measure VRAM/CPU RAM/concurrent meshing/render usage separately.
  The normal pipeline runs stage by stage; `--stream-mesh` runs simulation and CPU meshing concurrently.
- Numerical failure: save the first failing substep and the config. Do not arbitrarily change the scene/tolerances/library.
- Resume mismatch: check that the precision, binary layout, character and physical settings are identical.
- Changing only the render: use the cache and render with `--tag`. No simulation rerun is needed.

## 13. Implementation details and past result documents

- [MESHTOSDF_BREAKDOWN_BENCHMARK.md](MESHTOSDF_BREAKDOWN_BENCHMARK.md): the 5 internal phases of MeshToSDF and auxiliary rasterization/sign diagnostics.


- [SIMULATION_BREAKDOWN_BENCHMARK.md](SIMULATION_BREAKDOWN_BENCHMARK.md): MeshToSDF time alone, its ratio of the simulation, and the actual active voxel counts.


- [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md): MAC/APIC, operator reuse, SDF representation.
- [SCALABILITY.md](SCALABILITY.md): precision, GPU resident management, MGPCG, measurements and limitations.
- [SCENE_IMPLEMENTATION.md](SCENE_IMPLEMENTATION.md): character, whitewater, surface, render.
- [SURFACE_FILTER_RESULTS.md](SURFACE_FILTER_RESULTS.md): surface filter and the small whitewater render.
- [PIPELINE_BENCHMARK.md](PIPELINE_BENCHMARK.md): full pipeline measurements and bottlenecks.
- [SCENE_RESULTS.md](SCENE_RESULTS.md), [STREAM_REFINEMENT_RESULTS.md](STREAM_REFINEMENT_RESULTS.md):
  reproduction records of the previous scenes/runs. In a new environment, replace the past `--preset release`, configs and
  output paths with the explicit build/config procedure above.

What follows preserves the detailed explanation and the experiment commands of the previous basic demo.
For the setup/run of the latest 384 scene, use **sections 1–12 of this document**.

---

## Appendix: original demo guide and historical scene notes



This example includes the original three-stage mesh-obstacle liquid demo and a
dancing-character scene with a continuous emitter, foam/spray, anisotropic surfaces,
and an ovrtx animation pipeline. All solver, obstacle adapter, validation,
and output conversion code is local to this directory. The existing library operators
and the ovrtx renderer are not modified.

The current implementation is documented in
[IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md), including operator reuse, MAC/APIC
data layout, the pressure equation and MGPCG, topology lifetime, obstacle SDF
decoding, substep order, validation scope, and current limitations.

GPU resident particle management, float/double selection, compact cells and cooperative
GPU MGPCG are documented in [SCALABILITY.md](SCALABILITY.md). Scene settings remain
unchanged. The pipeline accepts `--precision float|double --pressure mgpcg|jacobi`.

The character extension is documented in [SCENE_IMPLEMENTATION.md](SCENE_IMPLEMENTATION.md).
See [SCENE_RESULTS.md](SCENE_RESULTS.md) for its commands, measurements and output files.
Configurations and reusable scripts are in `scene/`. CPU surface extraction additionally
requires `OPENVDB_BUILD_CORE=ON` and `NANOVDB_USE_OPENVDB=ON`.

The latest twelve-second scene uses `scene/clear_shower_fine.json`: a 42 cm emitter,
20 mm fluid / 10 mm obstacle / 6.25 mm meshing grids, an initial 6.4 cm pool,
and corrected pressure handling of partially blocked cells. See
[STREAM_REFINEMENT_RESULTS.md](STREAM_REFINEMENT_RESULTS.md) for the clumping diagnosis,
matched-resolution comparisons, full validation, timings, and final movies.
The previous 52 cm / 32 mm scene is preserved in `scene/grounded_shower.json` and
[GROUNDED_SHOWER_RESULTS.md](GROUNDED_SHOWER_RESULTS.md), including the isovalue
explanation and foot-motion comparison.

The twelve-second scene starts with 6.4 cm of water and uses `scene/wet_pool.json`.
Its visible white foam, initial pool, measurements and movies are documented in
[WET_POOL_RESULTS.md](WET_POOL_RESULTS.md).
The previous six-second accumulating water scene uses `scene/pool.json` (`drain_enabled: false`).
Its basin boundaries, mass/height measurements, and new movie are described in
[POOL_RESULTS.md](POOL_RESULTS.md). `scene/water.json` preserves the draining scene
for comparison. The free-slip/contact fixes,
whitewater visibility diagnosis, regression measurements and before/after movies
are described in [WATER_FIX_REPORT.md](WATER_FIX_REPORT.md).
`scene/final.json` and `output/scene_final` retain the previous scene baseline.

## Source-only checkout

Simulation caches, character mesh sequences, rendered images/videos, and the local
Python environment under `output/` are intentionally excluded from Git. Supply your
own character FBX and run `scene/export_character.py` before running the dance scene;
see `SCENE_IMPLEMENTATION.md` for the export and pipeline commands.

The obstacle adapters support both MeshToSDF root-interior representations: the
committed library's `uint8` bitmap with `root_extent`, and the newer local `int32`
root-tile origin list. The channel type selects the decoder. No library changes are
required by this example. Timing/results documents describe previously measured
runs; those generated artifacts are not included in a source checkout.

## Build and run

From the repository root, using the existing CUDA-enabled release configuration:

```bash
cmake --build build/release --target ex_fluid_apic_cuda -j 4
python3 nanovdb/nanovdb/examples/ex_fluid_apic_cuda/validate.py \
  build/release/nanovdb/nanovdb/examples/ex_fluid_apic_cuda
```

For a fresh configuration, the relevant options are:

```bash
cmake -S . -B build/fluid -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DOPENVDB_BUILD_CORE=OFF \
  -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_BUILD_NANOVDB=ON \
  -DNANOVDB_USE_OPENVDB=OFF -DNANOVDB_USE_CUDA=ON \
  -DNANOVDB_BUILD_EXAMPLES=ON -DNANOVDB_BUILD_TOOLS=OFF \
  -DNANOVDB_BUILD_UNITTESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=90-virtual \
  '-DCMAKE_CUDA_FLAGS=-Xcompiler=-D_AMXTILEINTRIN_H_INCLUDED -Xcompiler=-D_AMXINT8INTRIN_H_INCLUDED -Xcompiler=-D_AMXBF16INTRIN_H_INCLUDED'
cmake --build build/fluid --target ex_fluid_apic_cuda -j 4
```

The last compiler flags match the installed Ubuntu CUDA/GCC combination. Choose a
CUDA architecture supported by your compiler/GPU. This branch's NanoVDB configuration
fetches CCCL; a fresh build may need network access. The example target disables only
CCCL's umbrella-header advisory, needed because existing operators include `cub.cuh`
and the repository treats CUDA warnings as errors.

Individual runs use `ex_fluid_apic_cuda MODE OUTPUT_DIRECTORY SUBSTEPS`.
Modes are `stage1`, `stage2`, `static`, `moving`, and `deforming`. The validation driver
runs them in that order with 80, 80, 160, 200, and 200 substeps. A failed C++ check returns
a nonzero exit status and stops the driver. Use a new output directory when changing
the number of substeps, to avoid including older frames in a glob.

`output/summary.json` summarizes every metrics column and compares every stored
fixed/sparse particle frame. `output/*.log` records transfer, SDF, manufactured-contact,
and vertex-velocity tests. Output artifacts are intentionally git-ignored.

## Stage 1: fixed topology

- Domain: 32 x 32 x 24 cells, h = 0.05 m; Y is up. The initial dam has eight
  equally weighted particles per occupied cell. No particles are reseeded or deleted
  during a run. Initial particles inside an obstacle/contact margin are omitted.
- `ValueOnIndex` stores topology only. Cell center is `(ijk + 0.5) h`;
  component `a` of the MAC velocity is at that center minus `0.5 h e_a`.
  Slot zero is a permanently empty background sentinel.
- Equal particle masses and quadratic B-spline weights transfer mass and APIC momentum
  independently to each staggered component. G2P reconstructs velocity and
  `C_a = 4/h^2 sum(w u_a (x_face - x_particle))`.
- Fluid/air classification is a **separate particle-position occupancy mask**.
  P2G face mass is neither a fluid SDF nor the pressure-domain mask. The fluid
  pressure domain includes every occupied interior cell, not just a surface band.
- Gravity is followed by a matrix-free MGPCG solve. A cooperative GPU kernel controls
  iterations and accumulates dot products in double; storage is selectable float/double. The solved
  variable is the velocity potential `q = dt p / (rho h)`. Thus `A q = -net_flux`
  and face projection subtracts `q_right - q_left`. Air has zero pressure, with
  full-cell first-order fluid/air coupling and no subcell interface-distance correction;
  no fluid level set is advected.
- Tank boundary faces prescribe zero normal velocity. Projected fluid-adjacent faces
  seed four synchronous velocity-extrapolation layers; unprojected air face values
  are not retained as valid seeds. G2P checks both topology and velocity validity.
- Particle advection is first order. Tank contact clamps position and removes inward
  velocity. The next timestep is capped at 0.005 s and bounded by particle velocity
  plus the APIC affine contribution over the transfer support. Measured particle CFL
  must remain below 0.5; exceeding it fails the run rather than silently continuing.

## Stage 2: dynamic topology

`voxelsToGrid<ValueOnIndex>` (from `PointsToGrid.cuh`) creates a GPU grid from
**all particle cell coordinates**, followed by six 26-neighbor `DilateGrid` passes.
This covers interior pressure cells, both sides of their faces, quadratic transfer
stencils, and velocity extrapolation. The grid is sparse at voxel and 8^3 leaf level;
leaf allocation does not imply that all voxels in the leaf are active.

Every substep creates a new topology and freshly allocates cell sidecars. Cell coordinates
and six neighbor indices are re-enumerated from that new grid. Pressure, fluid
mask, mass, velocity, validity, and obstacle data are regenerated. Only particle
position, velocity, and affine matrices persist. No old voxel index is reused, and
there is no pressure warm start requiring remapping.

`PruneGrid` was inspected: it retains an explicit leaf-indexed voxel mask and returns
new indices. Rebuilding from particles already removes obsolete voxels, so an
additional prune is unnecessary. `MergeGrids` performs an active-mask union and
requires compatible transforms; it neither merges physical channels nor implements
wall boundary conditions. The obstacle grid is sampled separately, so no merge is used.

The tests check missing stencils on GPU and bijective coordinate/neighbor correspondence
on downloaded sidecars. Fixed versus sparse results are compared with a 1e-6 maximum
absolute tolerance for both position and velocity. Double atomic accumulation order
and reduction order can differ between runs/topologies.

## Stage 3: MeshToSDF and walls

The original `MeshToSDF.cuh` is instantiated once per substep, including for the static
case to measure the same path. Experimental `MeshToSDF_NEW` and `PropagateSign_*`
implementations are not included. Mesh topology is eight vertices/twelve triangles:

- Static cube: center `(0.85, 0.38, 0.60)`, half-size 0.18 m.
- Moving cube: `center.x += 0.12 sin(6t)`, velocity `(0.72 cos(6t), 0, 0)`.
- Deforming mesh: half-size `(0.18 s, 0.18/s, 0.18)`, `s = 1 + 0.2 sin(6t)`.
  This is a volume-preserving deformation with unchanged vertex connectivity.
  Velocity is `(s'/s (x-c)_x, -s'/s (x-c)_y, 0)`.

The same continuous motion generates vertices and wall velocity; a central-difference
vertex test checks that correspondence. Geometry and velocity are evaluated at the
substep's end time (first-order implicit wall position). Contact is checked both after
wall motion and after fluid advection.

Obstacle voxel size is 0.025 m with a three-voxel narrow band. Fluid cell/face positions
are converted to world space and then through the obstacle grid's `worldToIndex` map.
Trilinear sampling reads signed float channel `sdf` for active voxels; inactive leaf,
lower, and upper regions read their respective invert masks. Absent root regions read
`root_interior` when present. **The root channel is omitted by MeshToSDF when empty**.
Inactive interior samples return negative band width, never positive background.
The deep-interior test requires both a negative sample and index zero.

Face open area is estimated with 4 x 4 SDF quadrature. A solid-centered neighboring
cell conservatively blocks the full face. Solid cells are excluded from pressure.
Projection and diagnostics use the same discrete physical flux:

```
F_face = open_area * u_fluid + (1 - open_area) * u_wall
D = sum(F_positive_face - F_negative_face) / h
```

The matrix coefficients contain open area; blocked flux enters the right-hand side.
Fully blocked velocities are prescribed, so pressure cannot overwrite their wall
condition. A second CPU flux check reconstructs the analytic wall velocity rather
than trusting the wall sidecar. Normal-velocity diagnostics check the blocked face
trace (including the blocked part of partially open faces), after projection.

Particle collision uses the sampled SDF and its finite-difference normal. Iterative
position correction applies a 0.0125 m contact margin; this accommodates trilinear
corner error. Relative normal velocity is removed at contact, while tangent velocity
is retained. Contact resets the affine matrix. Penetration is measured against both
sampled SDF and the independent analytic mesh geometry. A manufactured inward-moving
particle starts 0.005 m inside the cube and must be recovered. Simulation logs count
pre-correction penetration attempts separately from remaining penetrations.

## Visualization

The existing ovrtx checkout's README, minimal Python example, USD examples, driver
requirements, and `skills/application-flow/SKILL.md` were inspected. No public input
path for an index grid plus arbitrary sidecar was found. `render.py` converts particle
positions to small octahedral marker meshes in local USD, adds the obstacle and tank
wireframe, and uses the existing renderer API without modifying the ovrtx checkout.
This visualizes the actual particles, not a fitted fluid surface.

```bash
python3 -m venv /tmp/fluid-render-venv
/tmp/fluid-render-venv/bin/pip install \
  ovrtx==0.5.1.385782 ovstage==0.2.1.385922 numpy pillow
EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
/tmp/fluid-render-venv/bin/python "$EX/render.py" --rtx \
  "$EX/output/stage1/frame_0040.csv"
python3 "$EX/render.py" --gif "$EX/output/moving/sequence.gif" \
  "$EX"/output/moving/frame_*.csv
```

`frame_NNNN_rtx.png` is the ovrtx render. `frame_NNNN.png` is an X-Y particle slab
(0.5 < z < 0.7) with allocated leaf blocks intersecting the mid-Z plane. Orange blocks
contain fluid cells; cyan blocks are halo-only. These are block-allocation diagnostics,
not claims that every voxel of a leaf is active. Red shows the actual obstacle outline.
Particles are colored by speed. The debug sequence remains available alongside the
3D renders so opaque geometry cannot conceal penetration.

To regenerate all representative images and animations:

```bash
/tmp/fluid-render-venv/bin/python "$EX/render_all.py" --rtx
```

All CUDA stage timers synchronize the default stream and report elapsed wall time,
including allocations and host orchestration. Pressure time includes GPU CG reductions
and projection, but not the independent diagnostics. Collision time includes contact
kernel/readback. CSV/PNG/USD output and rendering occur outside these simulation stage
timers. Do not add output/render wall time to the reported simulation costs.

## Validation thresholds and limitations

- APIC constant/affine field: max velocity error <= 1e-10, max affine error <= 1e-9,
  away from all boundaries.
- True recomputed residual norm <= max(1e-12, 2e-8 * RHS norm); fluid-cell RMS flux
  divergence <= 1e-6 s^-1, including moving wall flux. CG requests 1e-9 relative convergence.
- Fixed/sparse particle differences <= 1e-6; missing topology/velocity stencils = 0.
- Blocked-face and contact relative normal velocity <= 1e-8 m/s.
- Remaining sampled-SDF and analytic-mesh penetration deeper than 1e-7 m = 0.
- Positive contact counts ensure obstacle tests actually exercise wall interaction.

This is a small validation demo, not a production liquid solver. The fluid occupancy
mask is first order, without subcell fluid SDF, surface tension, viscosity, reseeding,
or a volume-conservation correction. Cut-face quadrature plus solid-cell blocking is
not a geometrically exact cut-cell method; blocked-face errors do not establish
pointwise accuracy of an interpolated fluid velocity on arbitrary oblique surfaces.
The basic executable validates axis-aligned translation and volume-preserving box
deformation. The separate scene executable also supports the evaluated character
sequence and closest-triangle wall velocities; see [SCENE_IMPLEMENTATION.md](SCENE_IMPLEMENTATION.md).

The collision margin creates a visible clearance. Deep penetration into the clamped
SDF interior cannot be resolved using its zero gradient; bounded substeps and per-step
contacts avoid it in these scenes. Closed Neumann manufactured pressure systems are
also tested by the scalability verifier; this does not establish arbitrary sealed,
deforming-container compatibility. The thresholds above describe the basic double
tests. Current float thresholds, GPU reductions, persistent MGPCG, measured speedups
and the remaining strict trajectory comparison failure are in
[SCALABILITY.md](SCALABILITY.md). `RESULTS.md` retains the original basic-demo measurements.

## Physical-time runs and 3D animation

Long runs stop at actual simulation time rather than at a fixed substep count:

```bash
EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
build/release/nanovdb/nanovdb/examples/ex_fluid_apic_cuda moving \
  "$EX/output/long/moving" --duration 5 --fps 30
/tmp/fluid-render-venv/bin/python "$EX/validate_long.py" \
  build/release/nanovdb/nanovdb/examples/ex_fluid_apic_cuda --duration 5
```

`--fps` controls output sampling in physical time. The adaptive step is clipped to
reach every frame time exactly. `frames.csv` maps frame number, substep, time, and
particle count; it avoids treating unequal substeps as equal animation intervals.
The original positional `SUBSTEPS` interface still works. Long runs additionally
record maximum/mean speed, finite affine matrices, APIC particle energy, center of
mass, particle count, and tank escape depth. A progress guard rejects collapsed
adaptive timesteps. Tiny frame-boundary clipping steps are allowed and are distinct
from a collapsed proposed timestep.

For animation authoring and diagnostic plots, add these packages to the renderer environment:

```bash
/tmp/fluid-render-venv/bin/pip install usd-core matplotlib
/tmp/fluid-render-venv/bin/python "$EX/animate.py" "$EX/output/long/moving"
/tmp/fluid-render-venv/bin/python "$EX/assemble_animation.py"
/tmp/fluid-render-venv/bin/python "$EX/report_long.py"
```

`animate.py` writes a binary, time-sampled `scene.usdc`, replays it through one ovrtx
renderer using `ovstage.population.update_from_usd_time`, and encodes `animation.mp4`.
The USD time codes are seconds multiplied by the scene's timeCodesPerSecond;
ovstage's update API accepts seconds. Each 5 s obstacle scene has 151 samples at 30 fps,
including both endpoints. The output also includes `debug.mp4` for particle/block
inspection, all rendered PNG frames, and separate authoring/render timings.
Forty initial warmup renders follow the ovrtx warmup skill; these do not advance the
solver. Rendering does not run concurrently with the measured simulation.

The five-second results and 3D video links are in [LONG_RUN_RESULTS.md](LONG_RUN_RESULTS.md).
They include the measured occupied-cell-volume drift. Conserved particle count and
small pressure divergence do not, by themselves, prove conserved material volume.

## Higher simulation resolution

`--voxel-size` scales the fluid discretization while preserving the 1.6 × 1.6 × 1.2 m
tank and the physical dimensions and motion of the dam and obstacle. The cell size
must divide all tank dimensions. A modest increase from 0.05 to 0.04 m changes the
fixed domain from 32 × 32 × 24 to 40 × 40 × 30 cells. Particle spacing scales with
the cell size; the dam keeps its original physical bounds. Obstacle SDF cells are
half the fluid cell size, contact clearance is one quarter, and rendered particle
marker radius is 0.14 times the fluid cell size. `scene.json` records these values
for validation, volume measurements, USD authoring, and block debug views.

```bash
cmake --build build/release --target ex_fluid_apic_cuda -j 4
EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda
OUT="$EX/output/higher_resolution"
/tmp/fluid-render-venv/bin/python "$EX/validate_long.py" \
  build/release/nanovdb/nanovdb/examples/ex_fluid_apic_cuda \
  --duration 5 --voxel-size 0.04 --output "$OUT"
for scene in stage1 stage2; do
  /tmp/fluid-render-venv/bin/python "$EX/animate.py" "$OUT/$scene" --fps 10
done
for scene in static moving deforming; do
  /tmp/fluid-render-venv/bin/python "$EX/animate.py" "$OUT/$scene"
done
/tmp/fluid-render-venv/bin/python "$EX/assemble_animation.py" --output "$OUT"
/tmp/fluid-render-venv/bin/python "$EX/report_long.py" --output "$OUT"
/tmp/fluid-render-venv/bin/python "$EX/compare_resolution.py" \
  --low "$EX/output/long" --high "$OUT"
```

The original 0.05 m output remains under `output/long`; higher-resolution results
are separate. Comparing the two cell sizes is a sensitivity check, not a convergence
proof: there are only two discretizations, the particle populations differ, and
the contact clearance also scales. Within each resolution, fixed/sparse comparisons
still use identical initial particles and compare matching physical-time frames.

Measured higher-resolution results and artifact links are in
[HIGHER_RESOLUTION_RESULTS.md](HIGHER_RESOLUTION_RESULTS.md). The measured loss of
occupied-cell volume is larger at 0.04 m; passing pressure and penetration tests
does not validate geometric volume conservation.
