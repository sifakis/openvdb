# Cloth collision against a per-frame SDF: the collision field

The scene this example is for: cloth (a long skirt) on an animated character, with the collisions resolved against
a signed distance field that `tools::cuda::MeshToSDF` rebuilds from the character mesh every frame.

**Status.** Only the collision field and its checks exist. There is no cloth solver here yet. The work was paused
after the first two steps of the plan at the end of this file.

## Files

| file | what it is |
|---|---|
| `SdfField.cuh` | `cloth::SdfField`: the field of a triangle mesh, made by MeshToSDF (nesting rule `Solid`) followed by PropagateSign. `cloth::SdfView::sample()` returns the value and the gradient at any world position, with NanoVDB's `TrilinearSampler` on an accessor that returns the signed background outside the band. |
| `sdf_check.cu` | Checks of `SdfField.cuh` against the exact distance to a sphere, and against OpenVDB (`meshToLevelSet`, `BoxSampler`) on a sphere and on one pose of a character. Needs OpenVDB. |
| `probe_sphere.cu` | Diagnosis: where the values MeshToSDF stores for a finely triangulated sphere differ from the exact distance, per bin of the distance. |

## Build

The targets are part of the NanoVDB examples and need CUDA; `sdf_check` also needs `NANOVDB_USE_OPENVDB`:

```
cmake --build <build dir> --target ex_cloth_collision_cuda_sdf_check ex_cloth_collision_cuda_probe_sphere
```

The executables are written to `<build dir>/nanovdb/nanovdb/examples/`.

## Run

```
./ex_cloth_collision_cuda_sdf_check sphere [band in voxels]
./ex_cloth_collision_cuda_sdf_check character <dir> <frame> <voxel size> [band in voxels]
./ex_cloth_collision_cuda_probe_sphere [radius in voxels]
```

- `sphere` needs no input: a triangulated sphere at a radius of 30, 60 and 120 voxels, band 4 voxels by default.
- `character` reads one pose of an animated mesh from `<dir>/sequence.txt`, `vertices.bin` and `triangles.bin`,
  as written by `ex_fluid_apic_cuda/scene/export_character.py`. The mesh is not part of the repository.
- `probe_sphere` triangulates a sphere into about 16 million triangles and needs several GB of free device memory.

## What `sdf_check sphere` shows

Per radius, against the exact distance `|x - c| - R`:

- the largest error of the stored `|value|`, and the voxels with the wrong sign and how far the farthest of them
  is from the surface;
- the error of the sampled value and the angle of the sampled gradient at random points within `band - 1.75`
  voxels of the surface;
- the sign at 200000 points far from the surface;
- the difference from OpenVDB's `BoxSampler` on a grid with the same values.

On an RTX PRO 3000 Blackwell laptop GPU, band 4 voxels:

| radius in voxels | wrong signs (farthest from the surface) | sampled value: max error | gradient: max angle | far-field wrong signs |
|---|---|---|---|---|
| 30 | 19 (0.0057 voxel) | 0.0094 voxel | 1.50 degrees | 0 of 200000 |
| 60 | 40 (0.0019 voxel) | 0.0052 voxel | 0.81 degrees | 0 of 200000 |
| 120 | 191 (0.0018 voxel) | 0.0040 voxel | 0.40 degrees | 0 of 200000 |

The sampled value and gradient differ from `BoxSampler` by at most 3e-5 voxel.

An accurate value and gradient need all 8 voxels of the trilinear stencil inside the band, which holds only within
about `band - sqrt(3)` voxels of the surface.

## Plan

1. The SDF of the character against OpenVDB. Done (`sdf_check character`).
2. The sampler against a sphere and against OpenVDB. Done (`sdf_check sphere`).
3. An XPBD cloth solver, checked against a hanging-cloth answer.
4. Collision with a static sphere, then with a static character pose.
5. The dancing character with the skirt pinned at the waist and a new SDF every frame.
6. Rendering.

One open point from step 1: MeshToSDF and OpenVDB disagree on the sign at about 0.4% of the voxels of the character
pose, close to the surface. Inside the body near an internal surface (skin under clothes, eyes) the stored value
is the distance to that internal surface, so the gradient there can point inward; the collision response should
keep the cloth on the positive side and not rely on the gradient deep inside.
