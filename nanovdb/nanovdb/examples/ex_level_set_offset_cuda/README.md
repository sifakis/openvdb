# Offsetting a level set with track()

A mesh is turned into a narrow-band signed distance field with `tools::cuda::MeshToSDF`, and the surface is then
grown and shrunk in steps of half a voxel with `tools::cuda::LevelSetTracker`. One step is:

```
for each active voxel:
    phi -= d            // d = 0.5 voxel; phi += d to shrink
track()

track():
    dilate()     // grow the band by one layer of voxels
    normalize()  // restore |grad phi| = 1 (HJ-WENO5, TVD-RK2)
    prune()      // drop the voxels with |phi| >= the band half-width
```

The background is not changed.

## Programs

| target | source | what it does |
|---|---|---|
| `ex_level_set_offset_cuda_mesh_to_sdf_file` | `mesh_to_sdf_file.cu` | OBJ mesh -> level set (grid and signed distances), written to a `.sdf` file. |
| `ex_level_set_offset_cuda_offset_demo` | `offset_demo.cu` | Reads two such files, runs the offset steps, prints the checks of every frame and writes the frames for the viewer. |

`viewer/make_viewer.py` embeds the frames in `viewer/viewer_template.html`, which gives one self-contained HTML
page: a slider over the offset, a 3D view of the surface points, a slice, and the numbers of the frame.

## Build

The targets are part of the NanoVDB examples and need CUDA:

```
cmake --build <build dir> --target ex_level_set_offset_cuda_mesh_to_sdf_file ex_level_set_offset_cuda_offset_demo
```

The executables are written to `<build dir>/nanovdb/nanovdb/examples/`.

## Run

The mesh is not part of the repository; any closed OBJ mesh works. The demo needs the same mesh twice at the same
resolution: once with the band it offsets (half-width 3 voxels), and once with a wide band that serves as the
reference for the error. The wide band must be at least `steps / 2 + 4` voxels.

```
./ex_level_set_offset_cuda_mesh_to_sdf_file dragon.obj 160 3  dragon_160_b3.sdf
./ex_level_set_offset_cuda_mesh_to_sdf_file dragon.obj 160 10 dragon_160_b10.sdf

./ex_level_set_offset_cuda_offset_demo dragon_160_b3.sdf dragon_160_b10.sdf 10 offset_dragon

python3 viewer/make_viewer.py offset_dragon dragon 160 10 "<GPU name>" "<commit>" offset_dragon.html
```

The arguments of `mesh_to_sdf_file` are the mesh, the number of voxels along its longest side, the band half-width
in voxels and the output file. The third argument of `offset_demo` is the number of half-voxel steps in each
direction, so `10` grows and shrinks the surface by up to 5 voxels and writes 21 frames. The arguments of
`make_viewer.py` after the prefix are shown on the page as they are given.

## What the demo prints

One line per frame:

- **active voxels** of the level set after the step.
- **track ms**: the time of the one `track()` call of the step, measured once. A grid of this size is not meant as
  a performance measurement.
- **|phi| >= background**: voxels that should not remain after `prune()`. It must be 0 in every frame.
- **wrong sign**: voxels that are on the other side of the surface than in the reference and farther than 0.25
  voxel from the reference surface.
- **error**: `|phi - (wide - offset)|` in voxels over the voxels within 1 voxel of the offset surface, on the side
  the surface moved to. The reference is exact only on that side: outside when growing, inside when shrinking.
  Median, 99th percentile and maximum.

For the dragon at 160 voxels with 10 steps, `|phi| >= background` is 0 in all 21 frames; at +5 voxels the median
error is 0.020 voxel and the 99th percentile 0.19 voxel, and at -5 voxels 0.019 and 0.21 voxel.
