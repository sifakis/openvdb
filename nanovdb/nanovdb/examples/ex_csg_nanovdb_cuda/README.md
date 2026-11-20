# CSG on NanoVDB index-grid level sets: example programs

Programs that check and time the CSG operations of `nanovdb/tools/cuda/CSG.cuh`: `csgUnion`,
`csgIntersection`, `csgDifference` and `csgComplement`. The design is in [CSG-design.md](CSG-design.md).

A level set here is a `LevelSetTracker::State`: a `ValueOnIndex` grid, a sidecar with the value of every active
voxel (slot 0 holds the background), and the invert masks of `PropagateSign`.

## Programs

| target | source | what it does |
|---|---|---|
| `ex_csg_nanovdb_cuda_csg_operations_test` | `csg_operations_test.cu` | Checks every operation, for every sign pair of the backgrounds, against the per-voxel rule and the analytic distance, and checks 17 set identities byte for byte. Prints `ALL PASS` or `FAIL`. |
| `ex_csg_nanovdb_cuda_csg_union_baselines` | `csg_union_baselines.cu` | Compares `csgUnion` with the two baselines `csgUnionPropagateSign` and `csgUnionLeafResolved` on seven sphere pairs. Prints `ALL PASS` or `FAIL`. |
| `ex_csg_nanovdb_cuda_csg_timing` | `csg_timing.cu` | Wall time of each operation: minimum and median over 41 runs after 2 warm-up runs. |
| `ex_csg_nanovdb_cuda_mesh_to_band` | `mesh_to_band.cu` | Converts an OBJ mesh to a narrow-band level set with `tools::cuda::MeshToSDF` and writes it as a band file, the mesh input of the other programs. |
| `ex_csg_nanovdb_cuda_csg_viewer_slices` | `csg_viewer_slices.cu` | Writes slices of the inputs and of all 12 results while one mesh moves through another. |
| `ex_csg_nanovdb_cuda_csg_viewer_voxels` | `csg_viewer_voxels.cu` | Writes every stored voxel and mask bit of the inputs and the results of the same scene. |

`CsgExampleUtil.cuh` holds what they share: level sets of analytic spheres and of band files, host copies of a
level set, and the timer.

The last two programs write the data of an interactive viewer and of static figures. [viewer/README.md](viewer/README.md)
describes the scene, the commands, every field of the two files, and a Python reader.

## Build

The targets are part of the NanoVDB examples and need CUDA:

```
cmake --build <build dir> --target ex_csg_nanovdb_cuda_csg_operations_test ex_csg_nanovdb_cuda_csg_union_baselines \
    ex_csg_nanovdb_cuda_csg_timing ex_csg_nanovdb_cuda_mesh_to_band \
    ex_csg_nanovdb_cuda_csg_viewer_slices ex_csg_nanovdb_cuda_csg_viewer_voxels
```

The executables are written to `<build dir>/nanovdb/nanovdb/examples/`.

## Run

Sphere inputs need no files:

```
./ex_csg_nanovdb_cuda_csg_operations_test
./ex_csg_nanovdb_cuda_csg_union_baselines
./ex_csg_nanovdb_cuda_csg_timing
```

Mesh inputs are band files. The meshes are not part of the repository; any closed OBJ meshes work. The level sets
must share one index space, so give every mesh the same voxel size by choosing the extents accordingly:

```
./ex_csg_nanovdb_cuda_mesh_to_band dragon.obj    220 dragon.band
./ex_csg_nanovdb_cuda_mesh_to_band armadillo.obj 220 armadillo.band
./ex_csg_nanovdb_cuda_mesh_to_band bunny.obj     220 bunny.band

./ex_csg_nanovdb_cuda_csg_operations_test dragon.band armadillo.band bunny.band
./ex_csg_nanovdb_cuda_csg_timing dragon.band armadillo.band
```

`mesh_to_band` takes the barrier signing of MeshToSDF as an optional fourth argument: `heuristic` (default),
`interior` or `ball`.

## What the operations test checks

For each of seven sphere pairs (overlapping, disjoint, one inside the other, touching bands, equal, in different
root tiles, and large enough for lower- and upper-level invert bits), each of the four sign pairs of the
backgrounds and each of the three operations:

- **Rule.** At every voxel of a box around each sphere, the full-space field of the result equals the rule applied
  to the fields of the inputs: `min` for union, `max` for intersection, `max(f1, -f2)` for difference. The
  full-space field is the stored value of an active voxel, and the background or its negation elsewhere, as the
  invert masks say.
- **Analytic distance.** The same field against the rule applied to the analytic sphere distances: saturated
  values exactly, band values to float rounding.
- **Structure.** No leaf without an active voxel, no lower or upper node without a child, no invert bit where a
  voxel or a child is stored, and the field at a far point equal to the stored background.
- **De Morgan.** `1 - 2` against `1 & ~2`, and `1 | 2` against `~(~1 & ~2)`, byte for byte (grid, values and the
  three levels of invert masks).

Then 17 identities on three inputs, for all eight sign combinations, each side built independently and compared
byte for byte: commutativity, idempotence, double complement, associativity, absorption, distributivity, De
Morgan, and the difference identities. Identities that do not hold for narrow-band fields, such as `A & ~A` being
empty, are left out: inside the band `max(f, -f)` is `|f|`. Empty level sets as operands are checked as well.

With band files, the rule and structure checks run at five translations of the second mesh along +z, from one leaf
beyond the first mesh to no translation, and the identities run on the three meshes. The translations follow from
the bounding boxes of the bands, so any meshes and resolutions work.

## Notes

- The inputs of one operation must share one index space and backgrounds of equal magnitude; the signs of the
  backgrounds may differ.
- Root-level inverted regions (root origins) are not supported yet: the operations throw if an input has any.
- Timings depend on what else is using the GPU. Take them on an idle GPU before quoting them.
