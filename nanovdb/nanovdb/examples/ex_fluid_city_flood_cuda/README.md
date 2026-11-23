# Level-set free-surface fluid on NanoVDB: a river basin floods a city

A basin of water in the river, 450 m long and 10 m above street level, is held back by a flood wall. The wall has
three openings; the water bursts through them, fills the park behind the wall and spreads through the streets of a
city of 1.9 km x 1.8 km. The city is NVIDIA's Omniverse "AEC City Demo Pack"; its buildings, streets and river banks
are solids for the water, built from the scene's triangles by MeshToSDF.

The top-level `Reproduce.md` lists the exact commands, settings, measured numbers and hardware of the paper's run.

## What it exercises

| Tool | Use here |
|---|---|
| `tools/cuda/LevelSetTracker.cuh` (`track()`) | Every step the band of the water's level set is dilated, renormalized and pruned. |
| `tools/cuda/PropagateSign.cuh` | The tracker's constructor runs it, so the grid knows which side of the water every voxel is on, inside and outside the band. `FillInterior.cuh` builds the grid of the pressure solve (the whole interior of the water) from the band and these masks. The obstacle's inside beyond its band is read the same way. |
| `tools/cuda/MeshToSDF.cuh` | The obstacle: `gen_sdf.cu` turns the scene's mesh (131,428 triangles, among them ground faces 2 m wide and hundreds of meters long) into a signed distance field of 861 million voxels once, before the run. |
| Sparsity | The tank is the whole city, 7644 x 580 x 7056 cells of 0.25 m (31.3 billion). The water occupies 36 million of them, and the grid the solver works on grows from 46 to 76 million voxels (0.15% to 0.24% of the tank). |

## The scene

There is no inflow boundary: the basin is closed, the water drains through the openings under its own head, and the
total volume is fixed.

| | |
|---|---|
| World frame | x 0..1911 m (east), y up from street level (the river bed is at -20 m), z 0..1764 m (south); the river's north bank runs along x at z = 815 m |
| Basin | x 550..1000 m, 100 m into the river, floor 3 m below the street, water 10 m above it: 585,000 m^3 |
| Flood wall | on the bank along the basin, three openings: x 655..700, 740..810 and 855..900 m (160 m in all) |
| Parapet | a 4 m wall on the bank from x 520 to 1150 m, on either side of the basin, so that water on the bank road does not run back into the river |
| Solids | the city's triangles with their centroid in x 150..1500, z 0..930 m, cut 12 m above the street (the water never gets higher); a closed block under the ground on either side of the river and a prism under every roof, because the city's own surfaces are sheets and shells and the solver needs three cells of solid behind any water |
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
| `build_city.py` | The scene's solids as one mesh; prints the tank size, the basin's water box and the boxes a renderer draws. |
| `run_city.sh` | Solids -> MeshToSDF -> solver, with state files to continue from if the solver stops. Its header lists the solver switches. |
| `render_city.py`, `render_movie.sh` | Renders the city with its own materials, the water and the drops; the movie script's header lists the look and the camera. |
| `tools/check_run.py` | The numbers of a run: steps, time per step, memory, volume, wet cells under the ground, depth of the water in four places. |

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
  of the water, which here is mostly the basin. Without it the drops carry what the level set cannot hold, and the
  total (level set + drops) stays between 99.997% and 101.15% of the start over the 80 seconds. It is above 100%
  from 7 to 42 seconds, with the peak at 29 seconds; the cause of that gain was not found.
- **Thin water.** A level set cannot hold water thinner than about two cells. Where the flood ends in a film (the far
  end of the bank road west of the basin, where the water runs back into the river), the film stalls and flickers;
  the renderer does not draw layers one cell deep.
- **The water is rendered thin-walled** (`--thin`: no refraction), with a grey-green color that it takes over 4 m of
  depth.
