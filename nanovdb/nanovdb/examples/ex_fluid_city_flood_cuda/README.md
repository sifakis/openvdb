# Level-set free-surface fluid on NanoVDB: a river basin floods a city

A basin of water in the river, 450 m long and 14 m above street level, is held back by a flood wall. The wall has
five openings; the water bursts through them, fills the park behind the wall and spreads through the streets of a
city of 1.9 km x 1.8 km. The city is NVIDIA's Omniverse "AEC City Demo Pack"; its buildings, streets and river banks
are solids for the water, built from the scene's triangles by MeshToSDF.

The top-level `Reproduce.md` lists the exact commands, settings, measured numbers and hardware of the paper's run.

## What it exercises

| Tool | Use here |
|---|---|
| `tools/cuda/LevelSetTracker.cuh` (`track()`) | Every step the band of the water's level set is dilated, renormalized and pruned. |
| `tools/cuda/PropagateSign.cuh` | The tracker's constructor runs it, so the grid knows which side of the water every voxel is on, inside and outside the band. `FillInterior.cuh` builds the grid of the pressure solve (the whole interior of the water) from the band and these masks. The obstacle's inside beyond its band is read the same way. |
| `tools/cuda/MeshToSDF.cuh` | The obstacle: `gen_sdf.cu` turns the scene's mesh (156,791 triangles, among them ground faces 2 m wide and hundreds of meters long) into a signed distance field of 1.08 billion voxels once, before the run. |
| Sparsity | The tank is the whole city, 7644 x 580 x 7056 cells of 0.25 m (31.3 billion). The water occupies 47 million of them at the start, and the grid the solver works on grows from 58 to 138 million voxels (0.19% to 0.44% of the tank). |

## The scene

There is no inflow boundary: the basin is closed, the water drains through the openings under its own head, and
no water is added.

| | |
|---|---|
| World frame | x 0..1911 m (east), y up from street level (the river bed is at -20 m), z 0..1764 m (south); the river's north bank is at z = 815 m along the basin, further north to the west and further south to the east |
| Basin | x 550..1000 m, 100 m into the river, floor 3 m below the street, water 14 m above it: 765,000 m^3 |
| Flood wall | on the bank along the basin, five openings: x 560..600, 655..700, 740..810, 855..900 and 950..990 m (240 m in all) |
| Walls along the bank | 2 m thick, from 1 m below the street to 8 m above it, on either side of the basin: west of it to x = 3 m, east of it to x = 1598 m and from there north across the east end of the ground. They follow the bank, which is not parallel to x, so that water on the bank road does not run back into the river |
| Solids | the city's triangles with their centroid in x 0..1600, z 0..930 m, cut 16 m above the street; a closed block under the ground on either side of the river and a prism under every roof, because the city's own surfaces are sheets and shells and the solver needs three cells of solid behind any water |
| Voxel | 0.25 m (0.5 m for a GPU with 12 GB) |
| Solver | no volume control, semi-Lagrangian advection of the level set, MacCormack advection of the velocity, 16 marker particles per cell within 2 cells of the surface, drops for the water the level set cannot hold, solids offset by 0.6 voxel, velocity components held to 20 m/s |

`build_city.py` writes the solids as one mesh; its options change any of the numbers above.

## Assets

The city is **not in this repository**. `get_assets.sh` downloads the pack (about 200 MB) from the public Omniverse
asset page and exports the city as `output/city.obj`. The pack's license file says it is bound by the NVIDIA Omniverse
terms.

## Steps

```
CCCL=<a checkout of github.com/NVIDIA/cccl> ./build.sh
export RENDER_PYTHON=<python with ovrtx, ovstage, usd-core, numpy, Pillow>
./get_assets.sh
./run_city.sh                 # the paper's run: output/flood/, 0.25 m voxels, 80 seconds, 641 frames
./render_movie.sh             # output/flood/movie/movie.mp4, 2560 x 1440
$RENDER_PYTHON -I tools/check_run.py output/flood 128 288 400 520 640
```

A quick look first, on a GPU with 12 GB: `./run_city.sh probe 0.5 20` and `./render_movie.sh probe`.

`CCCL` is only needed if the CUDA toolkit's own Thrust lacks the `<cuda/std/...>` headers (CUDA 12.0 does); NanoVDB's
CMake build fetches a checkout into `<build>/_deps/cccl-src`. `mesh_band` needs an OpenVDB installation
(`OPENVDB_ROOT`, default `$HOME/local`) and is only used for rendering. `get_assets.sh` and `run_city.sh` only need
`usd-core` and `numpy` of the render Python.

A run under a name that was used before reuses that name's solids and overwrites its frames; frames of a longer
earlier run stay in `out/`.

## Files

| File | Content |
|---|---|
| `eulerian.cu` | The solver's driver: one time step, the tank, the modes and every switch. Its header lists the step. |
| `advect.cuh` | Semi-Lagrangian and MacCormack advection of a sidecar (the level set, the staggered velocity). |
| `FillInterior.cuh` | The band's grid with its interior filled in. |
| `pressure.cuh`, `pressure2.cuh`, `mg.cuh`, `fast.cuh` | Pressure projection: free surface placed by the level set (ghost fluid), multigrid-preconditioned CG on the fluid cells, velocity extrapolation. `fast.cuh` (`PIPE=2`) is the version the run uses. |
| `pls.cuh` | Marker particles that correct the level set (particle level set). |
| `spray.cuh` | The water the level set cannot hold, carried as drops that fall back into the water. |
| `gen_sdf.cu` | Mesh -> obstacle file (MeshToSDF). |
| `mesh_band.cpp` | One frame of the solver -> triangles of the water surface (OpenVDB `volumeToMesh`), for rendering. |
| `build.sh` | Builds the three programs into `bin/`. |
| `get_assets.sh`, `tools/export_city.py`, `tools/inspect_usd.py` | Downloads the city pack; USD -> triangles in the tank's frame. |
| `build_city.py` | The scene's solids as one mesh; prints the tank size, the basin's water box and the boxes a renderer draws, and writes the walls along the bank as a mesh of their own for the renderer. |
| `run_city.sh` | Solids -> MeshToSDF -> solver, with state files to continue from if the solver stops. Its header lists the solver switches. |
| `render_city.py`, `render_movie.sh` | Renders the city with its own materials, the water and the drops; the movie script's header lists the look and the camera. |
| `tools/check_run.py` | The numbers of a run: steps, time per step, memory, volume, wet cells under the ground and far below the street, depth of the water in four places. |

The solver includes `Fluid.cuh` and `scalable/` of `../ex_fluid_apic_cuda` (cell kernels and the multigrid it was
derived from); those files are used as they are. The example is **not registered in `examples/CMakeLists.txt`**:
`build.sh` compiles it directly against this repository's headers.

## What to know about this scene

- **The ground needs the cancellation-free closest point on a triangle** (`nanovdb/math/Proximity.h` of this branch).
  The ground block's faces are 2 m wide and hundreds of meters long. With the former float evaluation the distance of
  the voxels next to such a face came out up to 1.1 voxel too large, the closed block got no inside, and the water fell
  into the ground.
- **The solver is not deterministic.** Two runs of the same binary on the same GPU give reports that agree to all
  printed digits for the first steps and then drift apart in the last digits (0.003% of the volume after 5 seconds at
  0.5 m voxels). A reproduction matches the paper's run in its numbers and its look, not bit for bit.
- **The volume control stays off.** It would put the water lost in thin sheets back as a uniform divergence over all
  of the water, which here is mostly the basin. Without it the drops carry what the level set cannot hold.
- **The volume is not conserved.** In the paper's run the total (level set + drops) is between 99.999% and 102.75%
  of the start for the first 48 seconds, with the peak at 27 seconds; the cause of that gain was not found. From
  about 50 seconds it grows steadily, to 123.8% at 80 seconds. The growth goes with water far below the street: at
  50 seconds almost all of it is between x = 1400 and 1600 m near the bank, on the tank's floor 21 m down, and from
  there it spreads east. Where it gets through the ground was not found. The movie's camera does not see that
  place. `tools/check_run.py` counts this water.
- **Splashes above the cut.** The buildings are cut 16 m above the street. In the first seconds the water's
  splashes reach 28 m; above the cut they meet no building.
- **Thin water.** A level set cannot hold water thinner than about two cells. Where the flood ends in a film, the
  film stalls and flickers; the renderer does not draw layers one cell deep.
- **A stopped run continues with its spray.** The state files hold the marker particles and the drops in the air
  besides the water and its velocity, so a run that `run_city.sh` continues from one goes on with all of them.
- **The water is rendered thin-walled** (`--thin`: no refraction), with a grey-green color that it takes over 4 m of
  depth.
