# CSG: comparison with OpenVDB

Programs and scripts that compare the CSG operations of `nanovdb/tools/cuda/CSG.cuh` with
`openvdb::tools::csgUnion`, `csgIntersection` and `csgDifference` on the same mesh level sets: first that every
voxel of the result is the same, then how long the operations take. `results/` holds the output of the runs whose
numbers are quoted below.

## Files

| file | what it does |
|---|---|
| `build.sh` | Builds everything into `bin/` with nvcc and g++ (its header lists the environment variables). |
| `nanovdb_fields.cu` | Runs the NanoVDB operations on two band files and writes the field value and the active state of every voxel of a box around both inputs. |
| `openvdb_fields.cpp` | Builds the same inputs in OpenVDB, runs OpenVDB's operations in three settings, and compares every voxel with that file. |
| `run_correctness.sh` | The comparison above on 15 pairs of inputs. |
| `nanovdb_timing.cu` | Wall time of the NanoVDB operations and of the union baseline `csgUnionPropagateSign`. |
| `openvdb_timing.cpp` | Wall time of OpenVDB's operations, in place and as copies. |
| `stage_timers.py` | Writes a copy of `CSG.cuh` with timers around its steps; the header of the repository is not changed. `nanovdb_timing.cu` built against the copy (`bin/nanovdb_stage_timing`) prints the time of every step. |
| `device_copy_timing.cu` | Wall time of a plain device copy of N floats, a lower bound for writing N values. |
| `run_timing.sh` | All timings at four resolutions. |
| `results/correctness.txt`, `results/timing.txt` | The output of the two scripts on the machine described below. |

The band file format and `mesh_to_band` are described in [../README.md](../README.md).

## What is compared

The two libraries offer different operations, so both OpenVDB variants are timed:

- **In place** (`csgUnion(a, b)` and the like, OpenVDB's default functions). The result is written into `a`, and
  nodes of `b` are moved into it by pointer. A node that only one input has is never read, so the work is
  proportional to the leaves that both inputs have at the same place. Both inputs are consumed.
- **Copy** (`csgUnionCopy(a, b)` and the like). The inputs stay intact and a new grid is returned.

The NanoVDB operations leave their inputs intact and build a new grid and a new array of values, so the copy
variant is the one that does the same thing. A NanoVDB grid is one contiguous buffer and its active voxels are
numbered in sequence, so a node of another grid cannot be spliced into it; there is no counterpart of the in-place
variant.

The NanoVDB times do not include transfers between host and device: the inputs are on the device before the first
timed run and the results stay there.

## Requirements

| | Needed | Used for the numbers below |
|---|---|---|
| GPU | 12 GB for the default resolutions | NVIDIA RTX PRO 3000 Blackwell laptop GPU, 12 GB, driver 580.178.04 |
| CPU | | Intel Core Ultra 7 265H, 16 threads, 62 GB of memory |
| CUDA toolkit | 12.0 or newer | 12.0 |
| CCCL | a checkout of https://github.com/NVIDIA/cccl, if the toolkit's Thrust has no `<cuda/std/...>` headers | v3.7.0.dev-150-g051a9a7b20, the one that NanoVDB's CMake build fetched |
| OpenVDB | an installation | 13.0.1, with TBB |
| Host compiler | | g++ 12.4.0 |
| Disk | 4.5 GB for the band files of the default resolutions | |

Meshes (not in the repository):

| mesh | vertices | triangles | SHA-256 of the OBJ file | source |
|---|---|---|---|---|
| `armadillo.obj` | 14,587 | 29,170 | `17200138dad1ad0f86ef9a98dd38f9932098b2c9e94d8694bd3f6143601e8c54` | https://github.com/odedstein/meshes, `objects/armadillo/armadillo.obj` |
| `bunny.obj` | 3,301 | 6,598 | `d97f4aa37c2f3bfb14eceb3d06ac1f466062dc45f08e019bfbae9704c28654b6` | https://github.com/odedstein/meshes, `objects/bunny/bunny.obj` |
| `dragon.obj` | 435,545 | 871,306 | `aaf8d5b5196a821625f3e6b375366d61983ad66f41ab90c722f4268ced32ca3d` | the Stanford dragon as a 74 MB OBJ file; its download link was not recorded |

Any closed meshes work; the counts of active voxels below are those of these three.

## Steps

```
CCCL=<cccl checkout> OPENVDB_ROOT=<OpenVDB installation> ./build.sh
./run_correctness.sh dragon.obj armadillo.obj bunny.obj     # -> output/correctness.txt
./run_timing.sh dragon.obj armadillo.obj                    # -> output/timing.txt
```

On the laptop the build also needed `CXX=g++-12` and, to keep nvcc 12.0 from parsing the host compiler's AMX
headers,
`EXTRA_FLAGS="-Xcompiler=-D_AMXTILEINTRIN_H_INCLUDED -Xcompiler=-D_AMXINT8INTRIN_H_INCLUDED -Xcompiler=-D_AMXBF16INTRIN_H_INCLUDED"`.

`run_timing.sh` stops if another compute process is on the GPU. Processes to tolerate are named by a grep pattern,
`ALLOW_GPU_PROCESSES=soffice ./run_timing.sh ...` for the runs below (an office program that holds 154 MB and
computes nothing). The laptop was on mains power with the `performance` power profile; with the `balanced` profile
the platform caps the GPU's power and the NanoVDB times grow.

## Correctness

Inputs: the three meshes at 220 voxels along their longest extent (dragon 496,764 active voxels, armadillo
466,322, bunny 681,590; band half-width and background 3 voxels), and the union of the dragon and the armadillo
moved by 58 voxels along z, which shares a part of its surface with the dragon. The 15 pairs:

| input 1, input 2 | input 2 moved by, in voxels |
|---|---|
| dragon, armadillo | 205, 117, 58, 20, 0 along z |
| bunny, armadillo | 60, 25, 0 along x |
| armadillo, dragon | 30 along y |
| bunny, bunny | 1 along z; 0 |
| armadillo, armadillo | 0 |
| dragon, dragon | 0 |
| dragon, dragon ∪ armadillo | 0 |
| dragon ∪ armadillo, dragon | 0 |

For every pair, at every voxel of the bounding box of both inputs padded by 4 voxels, the field value and the
active state are compared for the two inputs and for union, intersection and difference against OpenVDB with
`prune = false`, with its defaults, and with `pruneCancelledTiles = true`. In the last setting the OpenVDB
difference is compared with the NanoVDB difference made with `pruneCancelledLeaves`. That is 11 comparisons per
pair.

Result (`results/correctness.txt`): all 165 comparisons have no voxel that differs in value, in sign or in active
state. The differences of inputs that coincide:

| difference | active voxels, default | with `pruneCancelledLeaves` | OpenVDB with `pruneCancelledTiles` |
|---|---|---|---|
| dragon − dragon | 496,764 | 0 | 0 |
| bunny − bunny | 681,590 | 0 | 0 |
| armadillo − armadillo | 466,322 | 0 | 0 |
| dragon − (dragon ∪ armadillo) | 441,667 | 47,384 | 47,384 |
| (dragon ∪ armadillo) − dragon | 904,333 | 510,050 | 510,050 |

Not covered: OpenVDB's `pruneCancelledTiles` also acts on union and intersection (a leaf where one input is the
negation of the other), and NanoVDB has no counterpart; the 15 pairs have no such leaf. Fog volumes and inputs with
different band widths were not compared.

## Timing

Median wall time in milliseconds (`results/timing.txt` also has the minimum and the maximum, the two other
OpenVDB settings, and the steps). NanoVDB: 25 seconds of unions first, then 3 warm-up runs and 31, 31, 21 and 15
timed runs at the four resolutions; one run is one call, the destruction of the result included. OpenVDB: 16
threads, 2 warm-up runs and 15, 15, 9 and 7 timed runs; in place, the deep copies of the inputs that each run
needs are not timed. Built with `-O3 -DNDEBUG`.

The dragon and the armadillo, partly overlapping (the armadillo moved along z by 58, 265, 530 and 795 voxels):

| longest extent | active voxels of the inputs | operation | active voxels of the result | NanoVDB | NanoVDB, PropagateSign baseline | OpenVDB in place | OpenVDB copy |
|---|---|---|---|---|---|---|---|
| 220 | 496,764 + 466,322 | union | 849,236 | 1.66 | 2.48 | 0.47 | 2.28 |
| | | intersection | 113,850 | 1.57 | | 0.56 | 1.34 |
| | | difference | 469,892 | 1.61 | | 0.38 | 1.64 |
| 1024 | 10,874,985 + 10,138,209 | union | 18,544,820 | 4.50 | 8.31 | 1.85 | 10.26 |
| | | intersection | 2,468,374 | 1.94 | | 5.43 | 2.70 |
| | | difference | 10,279,575 | 3.01 | | 1.57 | 6.73 |
| 2048 | 43,507,506 + 40,566,676 | union | 74,205,001 | 15.14 | 28.48 | 5.45 | 47.29 |
| | | intersection | 9,869,181 | 3.27 | | 23.57 | 8.94 |
| | | difference | 41,119,229 | 8.18 | | 5.47 | 28.01 |
| 3072 | 97,899,716 + 91,286,911 | union | 166,981,866 | 33.53 | 66.87 | 10.09 | 95.34 |
| | | intersection | 22,204,761 | 6.02 | | 48.13 | 16.75 |
| | | difference | 92,525,341 | 17.73 | | 11.02 | 55.98 |

The dragon and the dragon at the same place, at 1024 (10,874,985 active voxels each):

| operation | active voxels of the result | NanoVDB | NanoVDB, PropagateSign baseline | OpenVDB in place | OpenVDB copy |
|---|---|---|---|---|---|
| union | 10,874,985 | 3.79 | 6.11 | 5.47 | 9.31 |
| intersection | 10,874,985 | 3.68 | | 5.34 | 9.19 |
| difference | 10,874,985 | 3.66 | | 6.81 | 9.51 |
| difference, cancelled leaves pruned | 0 | 6.39 | | 9.25 | |

The baseline `csgUnionPropagateSign` merges the bands, prunes them and reruns `PropagateSign` on the result;
`csgUnion` builds the invert masks from those of the inputs instead.

Reading:

- Against the copy variant, NanoVDB is faster from 1024 on: 2.3 to 3.1 times for union, 1.4 to 2.8 times for
  intersection, 2.2 to 3.4 times for difference. At 220 the two are within 40% of each other.
- Against the in-place variant on partly overlapping inputs, NanoVDB is slower for union (2.4 to 3.5 times) and
  for difference (1.5 to 4.2 times), and faster for intersection from 1024 on (2.8 to 8.0 times). The in-place
  variant moves what does not overlap by pointer; NanoVDB writes every value of the result.
- On inputs that coincide, every leaf is shared and the in-place variant has no node to move: NanoVDB is 1.4 to
  1.9 times faster than it.
- `csgUnion` takes between 50% and 67% of the time of the PropagateSign baseline.

Steps of the NanoVDB union, in milliseconds per call (from `bin/nanovdb_stage_timing`; the timers synchronize
the stream between the steps, so the sum is above the time of the untimed call):

| longest extent | retain masks | topology | values | invert masks | device copy of as many floats as the result has values |
|---|---|---|---|---|---|
| 220 | 0.02 | 1.76 | 0.09 | 0.12 | 0.01 |
| 1024 | 0.07 | 2.02 | 2.21 | 0.24 | 0.38 |
| 2048 | 0.27 | 4.17 | 9.75 | 0.62 | 1.13 |
| 3072 | 0.57 | 7.62 | 22.80 | 0.96 | 2.52 |

The topology build has a part that does not depend on the size of the inputs (the root, which goes through the
host, and the allocation of the internal mask buffers: about 1 ms together), which is most of the time at 220.
The values step grows with the number of values of the result and dominates from 2048 on; it runs one thread per
voxel, and each thread descends the trees of both inputs from the root. It is 6 to 9 times the time of a plain
copy of as many floats.

Notes:

- At 4096 (174,049,428 + 162,298,146 active voxels) an earlier build of the same programs ran on the 12 GB GPU
  while the desktop held 180 MB of its memory; this build ran out of memory while the desktop held 781 MB, and so
  did the earlier one; that resolution is therefore not in the
  default set. `EXTENTS="220 1024 2048 3072 4096" ./run_timing.sh ...` runs it where it fits; an extent that does
  not fit is reported and skipped.
- OpenVDB's in-place times vary more between runs than the others: across three runs the in-place union at 2048
  had medians between 4.5 and 5.4 ms.
