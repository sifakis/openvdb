# ConnectedComponents optimization plan

Status (2026-11-03): experimental and parked. MeshToSDF optimization comes first; this branch records where the
ConnectedComponents work stands so that it can be picked up again.

- `nanovdb/tools/cuda/ConnectedComponentsOpt.cuh`: `tools::cuda::ConnectedComponentsOpt`, same interface and same
  labels as `tools::cuda::ConnectedComponents` (`ConnectedComponents.cuh`, unchanged), whose pieces it reuses.
- `cc_opt_bench.cu`, `build.sh`: benchmark of the two on the same grid, run alternately, labels compared byte for byte.
- `results_rtxpro3000_60W.txt`: its output on an RTX PRO 3000 Blackwell laptop GPU (12 GB) at its 60 W power limit.

## What changes

ConnectedComponents has three stages after reading the grid: leaf-local components, a union across leaf faces, and
the per-voxel labels. ConnectedComponentsOpt changes each, and keeps the labels identical.

1. Leaf stage: union-find over the z-runs of a leaf instead of its voxels. A z-run is a maximal stretch of active
   voxels along z within one row (x, y) of the leaf; two runs are adjacent iff they lie in neighbouring rows
   (x +- 1 or y +- 1) and their bits overlap. The schedule is LeafUnionFind's (Liu and Tarjan: parent-connect, then
   parent-root-connect after 8 rounds, which bounds the rounds by O(log n)), run by one warp per leaf instead of one
   block. Runs are numbered in ascending order of their first voxel offset, so a component's root holds its smallest
   voxel offset, as before, and the global numbering does not change.
2. Cross-leaf stage: the current code counts the touching component pairs of each leaf, scans the counts, reads the
   total back on the host, writes an edge list, and unites the list in a separate pass. The optimized version calls
   the same lock-free `unite` where the pair is found. The union is order-independent (a class's smallest slot stays
   its root), so the representatives are the same, with no count, scan, host synchronization or edge list.
3. Labels: one warp per leaf reads each component's label once and writes it at every voxel of the component, instead
   of one thread per voxel searching the leaf's components. Only slot 0 of the label sidecar is initialized, since
   every other slot is an active voxel (the grid indexes exactly its active voxels, which ConnectedComponents asserts).

## Results (results_rtxpro3000_60W.txt)

Median wall time of a whole run, 3 runs after a warm-up; labels and component count identical in every run.

| input | active voxels | components | ConnectedComponents | ConnectedComponentsOpt | speedup |
|---|---|---|---|---|---|
| bunny, voxel 0.0015, half-width 3 | 14.8M | 1 | 5.80 ms | 1.72 ms | 3.36x |
| armadillo, voxel 0.00097, half-width 3 | 10.1M | 1 | 3.77 ms | 1.21 ms | 3.13x |
| three nested spheres, voxel 0.00195, half-width 3 | 28.7M | 3 | 10.99 ms | 3.36 ms | 3.27x |
| tree, voxel 0.0004, half-width 3 | 171M | 1 | 71.29 ms | 20.60 ms | 3.46x |
| hairball, voxel 0.004, half-width 2 | 561M | 1 | 271.58 ms | 91.84 ms | 2.96x |
| random voxels, 256^3 box, density 0.2 | 3.4M | 1,434,175 | 14.99 ms | 5.06 ms | 2.96x |
| random voxels, density 0.35 | 5.9M | 651,800 | 10.93 ms | 4.76 ms | 2.29x |
| random voxels, density 0.5 | 8.4M | 153,266 | 6.04 ms | 3.14 ms | 1.92x |

On hairball the stages go from: leaf 155.6 ms, cross-leaf count + scatter + unite 64.6 ms, labels 41.5 ms; to:
leaf 36.6 ms, cross-leaf unite 38.3 ms, labels 10.0 ms.

## Why union-find over runs, and not the faster bit flood

Five leaf-stage algorithms were compared (scratch study, 2026-11-03), each with the current and with the optimized
cross-leaf and label stages. Whole ConnectedComponents wall time in ms, labels identical everywhere:

| input | voxel UF (now) | + opt 2, 3 | run UF | + opt 2, 3 | bit scanline | + opt 2, 3 | bit flood (MaskShift) | + opt 2, 3 | x-slice UF | + opt 2, 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| bunny | 6.0 | 4.2 | 3.0 | 1.6 | 2.4 | 1.2 | 2.5 | 1.3 | 2.8 | 1.6 |
| armadillo | 3.6 | 3.2 | 2.1 | 1.2 | 1.7 | 0.9 | 2.1 | 1.0 | 2.0 | 1.2 |
| nested spheres | 11.6 | 9.0 | 5.4 | 3.3 | 4.7 | 2.5 | 4.8 | 2.6 | 6.4 | 3.3 |
| tree | 76.9 | 59.8 | 42.7 | 24.7 | 36.0 | 19.1 | 37.1 | 20.0 | 42.9 | 24.8 |
| hairball | 307.0 | 216.4 | 158.6 | 98.2 | 133.0 | 70.7 | 134.3 | 72.2 | 151.3 | 95.6 |

(bit scanline: flood a seed to the end of every open run along z, y, x in turn, peeling components from the smallest
voxel; bit flood: the same peeling, growing one voxel per step with the warp MaskShift of PropagateSign; x-slice UF:
union-find across x over the 2D pieces of each x-slice, which are found by a 2D flood within the slice.)

- On mesh narrow bands the bit scanline flood is the fastest leaf algorithm: on hairball it saves 27 ms per run over
  run union-find (71 vs 98 ms). On large inputs the difference is small in context: in MeshToSDF, which runs
  ConnectedComponents twice, it is about 55 ms of about 3.3 s on hairball at voxel 0.004, about 1.7%.
- Union-find stays, for two reasons. It is the leaf algorithm of the paper, with its O(log n) bound on the rounds.
  And beyond mesh narrow bands, where a leaf can split into many components (noisy point data, occupancy grids), the
  flood has to be repeated once per component, while union-find solves all of them at once; on synthetic leaves with
  256 components each (checkerboard), union-find over runs had the fastest leaf stage of all (0.074 ms vs 0.291 ms for
  the bit scanline, 2026-09-30).
- Union-find over runs and over x-slice pieces cost the same (98 vs 96 ms on hairball); runs keep the whole leaf
  stage a union-find, while x-slices need a flood inside each slice.

## Open items

- The cross-leaf unite is now the largest stage (38 ms of 92 on hairball). Each leaf looks up its three +axis
  neighbours from the root; 15 of 16 of them lie in the same lower node and could be read from it directly.
- The leaf stage solves each leaf twice (count, then masks and faces), as the current code does. The count kernel's
  runs and roots could be kept instead of recomputed.
- Initializing only slot 0 of the labels relies on every other slot being an active voxel. If ConnectedComponents
  gains a mask that labels only some of the active voxels, the full initialization comes back for that case.
- Before merging into ConnectedComponents.cuh: unit tests against ConnectedComponents on random and mesh grids, and
  review of the run construction (`LeafRuns::build`) and of the warp-level union-find (`RunUnionFind`).

## Reproduce

    ./build.sh                     # CCCL=<path to a CCCL checkout> if not the default
    ./cc_opt_bench mesh 0.004 2 3 hairball.obj
    ./cc_opt_bench random 256 0.35 3

The scratch study behind the leaf-stage comparison (generator of the variant headers, all logs) is kept outside the
repository, in `learning/work/bench/meshtosdf_opt/ccleaf/` on the author's machine.
