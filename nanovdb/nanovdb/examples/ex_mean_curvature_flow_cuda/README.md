# Motion by mean curvature of a mesh's level set

A triangle mesh is turned into a narrow-band signed distance field with `tools::cuda::MeshToSDF`, and its surface
then moves by mean curvature: every point moves along the normal at a speed equal to the mean curvature there. Bumps
go in and dents come out, small features first. In the volume-preserving form the average curvature of the whole
surface is subtracted from the speed; the shape then keeps its volume and ends as a sphere instead of vanishing.

Every step is one curvature update on the GPU (`MeanCurvature.cuh`) followed by
`tools::cuda::LevelSetTracker::track()`, the structure of `openvdb::tools::LevelSetFilter::meanCurvature()`.

## What it exercises

| Tool | Use here |
|---|---|
| `tools/cuda/MeshToSDF.cuh` | The input: the mesh's level set with a band of 5 voxels on either side. |
| `tools/cuda/LevelSetTracker.cuh` (`track()`) | After every update the band is dilated, renormalized and pruned. A run calls it tens of thousands of times in a row. |
| `tools/cuda/PropagateSign.cuh` | The tracker runs it, so every voxel outside the band has a sign. The curvature reads 19 values per voxel; a value outside the band is the background with that sign. |
| Sparsity | The Stanford dragon at 512 voxels along its longest side: 4.5 million active voxels in a bounding box of about 42 million; as the surface shrinks to a sphere, 1.9 million. |

## One step

```
for each active voxel:                        // MeanCurvature.cuh
    m = H |grad phi|,  g = |grad phi|         // 19 taps, the differences of math::CurvatureStencil
lambda = sum(delta(phi) m) / sum(delta(phi) g)     // volume-preserving only; otherwise 0
for each active voxel:
    phi += dt (m - lambda g)                  // dt = dx^2 / 3
track()                                       // dilate, renormalize, prune
```

`H` is the mean curvature `(k1 + k2) / 2`, so a sphere of radius `R` shrinks at `1 / R` and its radius is
`sqrt(R0^2 - 2 t)`. `delta` is the delta function of the surface smeared over 2 voxels on either side, so `lambda` is
the average of `H` over the surface. The time step `dx^2 / 3` is OpenVDB's.

## Steps

```
CCCL=<a checkout of github.com/NVIDIA/cccl> OPENVDB_ROOT=<an OpenVDB installation> ./build.sh
export RENDER_PYTHON=<python with ovrtx, ovstage, usd-core, numpy, Pillow>
./make_movie.sh dragon.obj            # output/flow/movie.mp4: volume-preserving, 60000 steps, 360 frames, 15 s
bin/verify                            # the checks below
```

`make_movie.sh <mesh.obj> [name] [resolution] [steps] [frames] [preserve volume] [schedule exponent]` takes any
closed OBJ mesh; `./make_movie.sh dragon.obj plain 512 8000 240 0 2` is the flow without volume preservation, in
which the dragon breaks into pieces that vanish. The render options (`RENDER_OPTIONS`) default to the dragon's
orientation (y up). The programs can also be run one by one; each source file's header gives its arguments.

The mesh is not part of this repository. The numbers below are for the Stanford dragon as an OBJ file with 435,545
vertices and 871,306 triangles (SHA-256 `aaf8d5b5196a821625f3e6b375366d61983ad66f41ab90c722f4268ced32ca3d`); its
download link was not recorded.

`CCCL` is only needed if the CUDA toolkit's own Thrust lacks the `<cuda/std/...>` headers (CUDA 12.0 does).
`mesh_band` and `verify` need an OpenVDB installation (`OPENVDB_ROOT`, default `$HOME/local`); the flow itself does
not. The example is not registered in `examples/CMakeLists.txt`: `build.sh` compiles it directly against this
repository's headers.

## Files

| File | Content |
|---|---|
| `MeanCurvature.cuh` | The update: the curvature kernel, the average curvature, the new values. |
| `mean_curvature_flow.cu` | Reads a level set, repeats (update, `track()`), writes the frames and one line of numbers per frame. |
| `verify.cu` | The checks against OpenVDB and exact solutions. |
| `render_flow.py` | Renders a frame's mesh with ovrtx. |
| `make_movie.sh` | Mesh -> level set -> flow -> meshes -> images -> movie. |
| `build.sh` | Builds the programs into `bin/`, among them two that are used as they are from other examples: `../ex_level_set_offset_cuda/mesh_to_sdf_file.cu` (MeshToSDF, mesh -> level set file) and `../ex_fluid_city_flood_cuda/mesh_band.cpp` (OpenVDB `volumeToMesh`, frame -> triangles). |

## Checks

`bin/verify` (radius 30 voxels, band of 5 voxels), on an NVIDIA RTX PRO 3000 Blackwell laptop GPU, with OpenVDB 13.0.1:

| Check | Result |
|---|---|
| One update against `openvdb::math::CurvatureStencil::meanCurvatureNormGrad()`, at every active voxel of a sphere and of two merged spheres | largest difference 4.8e-7 voxel |
| A sphere until its radius has halved (1013 steps), against `sqrt(R0^2 - 2 t)` | radius error +0.0038 voxel (OpenVDB's filter: +0.0005); the surface stays within 0.0016 voxel of a sphere |
| The same at radius 15 voxels (254 steps) | radius error +0.0028 voxel |
| Two merged spheres, which have a sharp crease, over 1350 steps against `openvdb::tools::LevelSetFilter::meanCurvature()` | largest difference 0.0067 voxel within one voxel of the surface |
| Volume-preserving, a sphere over 1000 steps | radius 30 -> 29.993 voxels |
| Volume-preserving, two merged spheres over 5400 steps | volume -0.19% (`openvdb::tools::levelSetVolume`); the area falls at every report, from 18045 to 16674 voxels^2 (the sphere of that volume: 16692); the surface's distance from its centroid narrows from 24..48 to 35.8..37.7 voxels (that sphere's radius: 36.4) |
| Voxel size 0.01 instead of 1 | the same numbers in voxels |

OpenVDB has no volume-preserving flow, so that form is checked against its properties only.

The dragon at resolution 512, volume-preserving, 60000 steps (`./make_movie.sh dragon.obj`), on the same GPU:

| | |
|---|---|
| Active voxels | 4,511,290 at the start, 1,858,057 at the end |
| Time | 14.7 minutes for the steps, 23 ms per step at the start and 13 ms at the end |
| Area | 451,238 -> 185,330 voxels^2; the sphere of the same volume has 184,707 |
| Volume | -0.115% over the run |

`bin/mean_curvature_flow` gave the same frame files, byte for byte, in two runs of the first 278 steps.

## What to know

- **The band must be 5 voxels wide, not 3.** The curvature reads the diagonal neighbors. With a band of 3 voxels the
  voxels beyond 1.6 voxels from the surface have neighbors outside the band, where only the background is known;
  their motion is wrong, and the renormalization of `track()` carries that to the surface. `bin/verify 30 1 3` shows
  it: the sphere's radius error is -0.099 voxel instead of +0.0038, and two merged spheres lose 2.4% of their volume
  instead of 0.19%. OpenVDB's filter has the same effect (-0.066 voxel with a band of 3). `track()` itself is not the
  cause: on spheres of 12.5 to 100 voxels that do not move, or move uniformly, it shifted the surface by less than
  5e-6 voxel per call (measured apart from `verify`).
- **The cost grows with the fourth power of the resolution.** The surface has four times the voxels at twice the
  resolution, and the time step `dx^2 / 3` makes the same change of shape take four times the steps.
- **The volume is not corrected.** Subtracting the average curvature keeps it to first order; what is left adds up
  slowly, and faster where the mesh's features are thinner than a few voxels (the dragon at resolution 128 loses 1%
  in 1200 steps).
- **A sphere centered exactly on a voxel leaves that voxel behind when it vanishes**: its value tends to zero from
  below and, by symmetry, has no gradient to move it. Off center, as any mesh is, the surface vanishes completely.
- **Render one frame per process.** With several frames in one run of `render_flow.py` the renderer was seen to draw
  the last frame's mesh for all of them; `make_movie.sh` starts it once per frame.
