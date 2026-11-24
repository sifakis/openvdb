# Hydra

A modeling example for the CSG operations of `nanovdb/tools/cuda/CSG.cuh`: nine triangle meshes of snakes become one
hydra with nine heads. One mesh is the body and keeps its own head; each of the other eight gives its head.

```
hydra = body | (model_1 & cutter_1) | (model_2 & cutter_2) | ... | (model_8 & cutter_8)
```

`|` is `csgUnion` and `&` is `csgIntersection`. Every operand is a narrow-band level set made from a triangle mesh
with `tools::cuda::MeshToSDF`:

| Level set | Mesh it is converted from |
|---|---|
| `body` | the body mesh as it is |
| `model_i` | another snake mesh, scaled, turned and moved so that its head sits on the body |
| `cutter_i` | a convex solid that holds the head and the neck of that snake and nothing else of it (the convex hull of that part of the mesh, scaled by 1.04), moved with its model |

Intersecting a moved model with its cutter leaves a head on a neck. The neck ends in a cut, which lies inside the
body (for one head: inside the neck of another head), so the union hides it. A level set cannot be moved without resampling its values, so each moved solid is
converted from its moved mesh. In total: 17 MeshToSDF conversions, eight intersections, eight unions. The surface of
the result is then extracted as triangles with OpenVDB's `volumeToMesh` and path traced.

## Files

| File | Content |
|---|---|
| `plan_hydra.py` | finds the head and the neck of every snake, makes the cutters and places the heads on the body; writes the scene file that `hydra` takes |
| `hydra.cu` | the conversions and the CSG operations; writes the level sets as band files (the format of `../mesh_to_band.cu`) |
| `band_to_mesh.cpp` | OpenVDB `volumeToMesh`: band file -> triangles (`.meshbin` for `render.py`, or `.obj`) |
| `render.py` | path-traced image of a `.meshbin` file with the ovrtx SDK |
| `build.sh`, `run.sh` | build the two programs; run the whole example |

## How the heads are placed

`plan_hydra.py` describes its steps at its top. In short: on each of the eight meshes, the head with its neck is
the part within a given distance, measured along the surface, of the head end of the snake; the cutter is the convex
hull of that part. A head keeps the orientation it has in its own mesh file, upright and facing +z like the body,
except for a roll about the facing direction, which leans it to one side, and for one a turn about the vertical. So
all nine heads face the front.

Three heads sit on the body's own neck behind its head, four on its chest, where the neck leaves the coil, and one
on the neck of another head. The middle of a cut is put at a point deep inside the body, or at the middle of the
other head's neck. The head is scaled so that the radius of its cut is the room there (the depth of that point, or
the inner radius of that neck) minus 0.025, in units of the body's mesh file; the scales come out between 0.55 and
0.86. A sphere of the cut's radius about the middle of the cut is then inside the solid it sits in, so a round cut
is hidden however it is turned; a cut that is not round can still show a little (see the last section).

The length of each neck, the place of each cut and the angles are numbers in `plan_hydra.py`, chosen by hand for
these nine meshes so that the nine heads show from the front. Another set of meshes needs its own.

## The meshes

The nine meshes were generated with Meshy AI (www.meshy.ai) and are not part of this repository; contact the
authors for them. All are closed and
consistently oriented (every edge has two triangles); meshes 7 and 9 have a second, small piece.

| File | Role | Vertices | Triangles | SHA-256 |
|---|---|---|---|---|
| `basilisk_var_1.obj` | head 1 | 1,606,136 | 3,212,388 | `a0066241fc5e59dfe47b1b1ad43a683921fc646dcec915de8f49dda67491178f` |
| `basilisk_var_2.obj` | head 2 | 1,202,121 | 2,404,294 | `f8eaaedf96988f8af5cc20f7d42cd5acda5536fbfb358b9ecb34ce3d0600066d` |
| `basilisk_var_3.obj` | head 3 | 915,539 | 1,831,126 | `a995e7c563ffcff277592bf729523b5d2fbda5cdeb9dc1dba23458d9828d150a` |
| `basilisk_var_4.obj` | head 4 | 1,042,147 | 2,084,306 | `9982138f65c19955e04f6eeeae7197a768734f1ec3f8eaed6c105e70593b95f8` |
| `basilisk_var_5.obj` | body | 2,547,158 | 5,096,816 | `3a7d9086004cda4e5c43b45d549226d0384b871b5a7e6f218cada1aed3f125f5` |
| `basilisk_var_6.obj` | head 6 | 1,138,137 | 2,276,306 | `49a3d2038b1bd81f46f2a1070d6e4d044215d6a1d16f3b75181f7dfcdaa27213` |
| `basilisk_var_7.obj` | head 7 | 820,511 | 1,641,178 | `b3119774fd3119bd89bb9b7c82afcc89b2241eb2e3e4ad99e62a4c84474d457f` |
| `basilisk_var_8.obj` | head 8 | 898,595 | 1,797,190 | `101a968cd0b780bdbb1f86be868790d74a0bbea9f1685910b785a334b873c967` |
| `basilisk_var_9.obj` | head 9 | 1,498,961 | 3,000,414 | `6fb70ee81e2b54744a927a7db18c192e8d78c8b8c0d350547a2352daee72d6b4` |

Together 23,344,018 triangles. The meshes stand with y up and face +z.

## Requirements

- A CUDA GPU and compiler, as for the other programs of `ex_csg_nanovdb_cuda`.
- An OpenVDB installation, for `band_to_mesh` only.
- For the plan: a Python with `numpy` and `scipy`.
- For the image: a Python with `ovrtx`, `ovstage`, `usd-core`, `numpy` and `Pillow`, and an RTX GPU supported by
  ovrtx.

## Steps

```bash
./build.sh                                   # see the file for NVCC, CCCL, OPENVDB_ROOT, CXX, EXTRA_FLAGS
export PLAN_PYTHON=<python with numpy, scipy>                                # default: python3
export RENDER_PYTHON=<python with ovrtx, ovstage, usd-core, numpy, Pillow>   # optional: without it no image
./run.sh <mesh dir>                          # 1024 voxels along the body's longest extent -> output/1024/
./run.sh <mesh dir> 220                      # a small run -> output/220/
```

`output/plan/` holds the plan: `scene.txt`, the eight `cutter_<i>.obj`, `plan.json` (the numbers behind every head),
the body's level set at 220 voxels in which the cuts were placed, and the distances along the surface of every mesh
(`distance_<i>.npy`). The plan is made once and does not depend on the resolution of the run. Measuring the
distances takes most of its time (about ten minutes here); a later plan in the same directory reuses them.

`output/<extent>/` then holds `body.band`, `head_<i>.band`, `hydra.band`, `hydra.inside`, `hydra.meshbin`,
`scene.json`, `log.txt`
and, with `RENDER_PYTHON`, `hydra.png` (1920 x 1080, 128 samples per pixel, seen from the front). At 1024 voxels the
folder takes 2.4 GB.

`scene.json` records how the body was placed in the grid and the matrix of every head, for programs that draw the
level sets. `bin/hydra ... --operands` also writes the level sets that the intersections take (`model_<i>.band`,
`cutter_<i>.band`). The animation of the paper's supplementary material is made from these files by the paper
repository's `figure factory/CSG/hydra/make_animation.py`, which runs the programs of this folder.

## Numbers

Active voxels printed by `hydra` (band half-width 3 voxels). The extent is that of the body; the hydra reaches
beyond the body's bounding box.

| Body's longest extent | body | the eight heads together | hydra | triangles of the result |
|---|---|---|---|---|
| 220 voxels | 1,201,117 | 758,362 | 1,760,421 | 856,596 |
| 1024 voxels | 29,480,970 | 17,874,439 | 42,967,024 | 21,174,084 |

Two runs at 220 voxels wrote the same `body.band`, `head_7.band` and `hydra.band`, byte for byte.

Wall times on an NVIDIA GeForce RTX 5090 (32 GB, driver 595.112.07, CUDA 12.9, idle before the runs). The program
prints the time of every conversion and of every CSG operation, after one conversion, one intersection, one union
and one sign propagation that are not timed. Each run was made three times; the sums of the three differed by less
than 6%, and the table is the second run. The nine models have 23,344,018 triangles, the eight cutters 22,270.

| Body's longest extent | MeshToSDF, nine models | MeshToSDF, eight cutters | one intersection | one union | 16 CSG operations | sign propagation, 17 level sets |
|---|---|---|---|---|---|---|
| 220 voxels | 1001.6 ms | 20.7 ms | 0.84 to 1.26 ms | 1.32 to 1.43 ms | 19.1 ms | 0.72 ms |
| 1024 voxels | 1042.1 ms | 28.6 ms | 1.26 to 1.63 ms | 2.45 to 2.95 ms | 33.7 ms | 1.45 ms |

A level set converted from a mesh gets its invert masks by one sign propagation (`PropagateSign`); the last column
is the sum of these 17. The CSG operations make the masks of their results without one. For comparison, one sign
propagation on the finished hydra takes 0.08 ms at 220 and 0.34 ms at 1024 voxels; the program measures all of them
by running `PropagateSign` once more on the level set.

For timings of the CSG operations against OpenVDB see `../openvdb_comparison`.

## The inside, level by level

A level set of this example holds its sign away from the active voxels in the tree itself: an inactive voxel of a
leaf, or a slot without a child of a lower or an upper node, is inside or outside as a whole (the invert masks of
`PropagateSign` in a `LevelSetTracker::State`, which the CSG operations combine level by level). `hydra` writes
the inside of the result in that form to `hydra.inside` and prints the counts:

| Body's longest extent | inactive voxels of leaves | slots of lower nodes (8 voxels wide) | slots of upper nodes (128 voxels wide) |
|---|---|---|---|
| 220 voxels | 871,193 | 923 | 0 |
| 1024 voxels | 32,611,628 | 323,945 | 0 |

At 220 voxels these, with the active voxels of negative value, are 2,168,212 voxels: the same voxels that a flood
fill of the space off the band finds inside (no voxel differs, none is counted twice).

## The cuts that show

The program counts the voxels within one voxel of the result's surface whose value is the value of a cutter there:
places where a cut is part of the visible surface instead of being hidden inside the body.

| Body's longest extent | surface voxels | of them on a cut |
|---|---|---|
| 220 voxels | 649,775 | 22 |
| 1024 voxels | 14,707,876 | 175 |

That is 0.001% of the surface at 1024 voxels. The cut of a neck is not a circle, and the scale of a head is set from
the mean radius of its cut; with the radius that holds 95% of the cut, and a margin of 0.03, the count at 1024 voxels
was 0, but the heads were smaller (scales 0.22 to 0.69).
