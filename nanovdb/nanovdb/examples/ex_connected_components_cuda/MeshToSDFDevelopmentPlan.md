# ex_connected_components_cuda

A NanoVDB / CUDA example that builds toward a **connected-components (CC) labeling**
on NanoVDB index grids. This document is the running design notes for that work.

> **No OpenVDB.** Unlike `ex_mesh_to_grid_cuda`, this example is pure NanoVDB + CUDA.
> The index↔world transform uses `nanovdb::Map::set(scale, translation, taper)`
> directly, the OBJ reader uses NanoVDB types, and the target is registered without
> the `OPENVDB` flag. (NanoVDB itself may still link `libopenvdb` transitively in a
> build configured with `NANOVDB_USE_OPENVDB=ON`, but no example code uses it.)

## Files

| File | Role |
| :--- | :--- |
| `connected_components_cuda.cpp`         | Host driver: arg parsing, OBJ reader, builds the mesh + `nanovdb::Map`, calls the device entry points. No CUDA *code* (only header-only handle/buffer types). |
| `connected_components_cuda_kernels.cu`  | CUDA / NanoVDB side: `computeUDF()` and `printGridDiagnostics()` today; CC kernels + derivative-grid step to follow. |
| `MeshToSDFDevelopmentPlan.md`           | This document. |

The host/device seam is a small set of forward-declared functions (à la
`ex_mesh_to_grid_cuda`'s `mainMeshToGrid`), passing host `std::vector`s + `nanovdb::Map`
in and returning device buffers (`GridHandle<DeviceBuffer>`, `DeviceBuffer`) out.

## Pipeline (incremental)

1. **`computeUDF(points, triangles, map, bandWidth)` → `{ handle, sidecar }`** *(done)*
   Rasterize the mesh into a narrow-band `ValueOnIndex` grid (every active voxel gets a
   dense index in `[1, N]`) plus a UDF sidecar of `N+1` floats:
   `sidecar[0]` = background, `sidecar[leaf.getValue(vi)]` = that voxel's unsigned distance
   to the closest triangle. This is `nanovdb::tools::cuda::MeshToGrid`, shared conceptually
   with `ex_mesh_to_grid_cuda` (which is really *mesh → UDF*).
   **⚠ UNITS:** despite the comment in `MeshToGrid.cuh` saying "voxel units", the sidecar
   is in **WORLD units** — `sidecar[0] = bandWidth · voxelSize` and values are clamped to it.
   (Verified empirically: dragon @ voxelSize 0.0005, bandWidth 3 → background = max = 0.0015.
   Also consistent with `ex_mesh_to_grid_cuda` comparing against OpenVDB's world-space SDF.)

2. **Derive a CC-input topology from `{ handle, sidecar }`** *(DONE — `computeDerivedTopology`)*
   Runs the rasterized active set with the **surface/barrier shell removed** (see "Derivative
   topology" below). On the dragon @ 0.0005: 41,492,695 active voxels → **29,512,911** after
   pruning (~12.0M barrier voxels dropped), exit 0.

3. **Connected-components labeling (CUDA)** *(TODO)*
   Label active voxels of the derived grid so two voxels share a label iff connected
   through a path of adjacent active voxels. Output: a per-active-voxel label buffer
   (a sidecar parallel to the index space).

## Connected-components design (working decisions)

- **Algorithm family:** **union-find**, with **hierarchical insights** that exploit
  NanoVDB's 8³ leaf structure (leaf-local labeling first, then merge across the
  +X/+Y/+Z face neighbors, then a small global union-find over representatives).
  Plain iterative label-propagation is kept only as a correctness oracle (its
  iteration count scales with component diameter — bad for thin voxel structures).
- **Connectivity:** start with **6-connectivity** (face neighbors); leave 18/26 as a
  later parameter. With 6-connectivity each undirected edge is enumerated once by
  visiting only the +X/+Y/+Z neighbors.
- **Label semantics:** compute **root-ids** (label = grid index of the union-find root)
  internally; offer an optional final stream-compaction pass to dense ids `[0, K)` plus
  a component count `K`.
- **NanoVDB hook:** `ValueOnIndex` gives each active voxel a dense index; within a leaf,
  neighbor → index is a value-mask bit test + popcount, and cross-leaf access goes
  through a `ValueAccessor` / the CUDA `NodeManager`. (Confirm the exact `ValueOnIndex`
  index layout before leaning on the leaf-local fast path.)

## Derivative topology (the CC input)

CC runs on the rasterized active set with the **surface/barrier shell removed**. The
barrier is the same one OpenVDB's `MeshToVolume.h` uses to stop its exterior flood-fill.

**The barrier threshold (from OpenVDB).** In `SweepExteriorSign::traceVoxelLine`, a voxel
is a barrier iff its distance value satisfies `dist <= 0.75`, and the sweep only flips
voxels to "exterior" while `dist > 0.75`. The same `0.75` recurs throughout
(`SeedPoints`, the interior-test floodfill — *"all voxels within 0.75 of the zero-crossing"*,
and the voxel/triangle intersection predicate `return !(dist > 0.75)`).

Two subtleties:
- It is the **squared** distance. `traceExteriorBoundaries` runs *before* the sqrt/world
  pass (`TransformValues`), so `dist` there is squared distance in voxel² units. The test
  is `dist² <= 0.75`.
- **Geometric meaning:** `√0.75 = √3/2 ≈ 0.86603` voxels = half the space-diagonal of a
  unit voxel (center-to-corner / circumradius). So "closest triangle within √3/2 of the
  voxel center" ⟺ "the triangle could pass through this voxel's cube" ⟺ it is a
  surface/boundary voxel. That is the principled, conservative *voxel-intersects-surface*
  cutoff. (OpenVDB's linear-distance form is `dist < 0.86602540378443861`.)

**Translation to our sidecar.** `MeshToGrid`'s UDF sidecar is already `sqrt`'d + clamped
and stored in **linear voxel units**. So:

> **barrier voxel ⟺ `UDF <= √3/2 ≈ 0.86603`** (equivalently `UDF² <= 0.75`)
> **keep voxel in CC domain ⟺ `UDF > √3/2`** (equivalently `UDF² > 0.75`)

**Why this works.** For a closed surface, deleting that ~√3/2-thick shell hugging the
surface disconnects the rasterized band into an **exterior shell** and an **interior shell**
(plus one component per internal cavity). Both shells carry the *same* unsigned UDF range
`(√3/2, bandWidth]` — they are distinguishable only by **connectivity**, which is precisely
what CC resolves. Downstream (eventual SDF work): a component touching the grid's exterior
is "outside", the rest "inside". (Robustness caveat: if a feature is thinner than the band,
the shell may fail to separate — this is exactly where CC is expected to beat a directional
flood-fill, and is a topic for later.)

**How to materialize it — DECIDED: (A) a new derived index grid via `PruneGrid`.**
We rebuild a clean, topology-only `ValueOnIndex` grid with the barrier voxels removed and
empty leaves dropped, giving a fresh dense `[1, N']` index space to run CC on. (The rejected
alternative was masking in place and having CC skip barrier voxels — cheaper, but it leaves
CC operating on a sparser-than-necessary index space and complicates neighbor logic.)

Mechanism as built (`computeDerivedTopology`, mirrors `Benchmark.cu`'s `pruneNarrowBand`):

1. `srcLeafCount = DeviceGridTraits<BuildT>::getTreeData(dGrid).mNodeCount[0]`.
2. Allocate a retain-mask sidecar: a device `DeviceBuffer` of one `nanovdb::Mask<3>` (512 bits)
   per source leaf, in leaf order. (During debugging this was a `thrust::universal_vector` so the
   host could popcount it directly; reverted to `DeviceBuffer` once verified.)
3. Fill it with a **custom per-leaf predicate functor** launched via
   `nanovdb::util::cuda::operatorKernel`, **one block per leaf, 512 threads** (one per voxel).
   NOTE: we cannot reuse `InjectGridMaskFunctor` — that derives the mask from a *second grid's*
   topology (`probeLeaf` + `valueMask` intersection). Our keep/drop signal is the UDF sidecar,
   so the functor evaluates the predicate directly:

   ```cpp
   const int leafID = blockIdx.x, threadID = threadIdx.x;   // threadID is the voxel offset
   const auto& leaf = d_grid->tree().template getFirstNode<0>()[leafID];
   auto&       mask = d_retainMask[leafID];
   if (threadID < nanovdb::Mask<3>::WORD_COUNT) mask.words()[threadID] = 0UL; // parallel clear
   __syncthreads();                                          // REQUIRED: clear-before-set
   if (auto n = leaf.data()->getValue(threadID)) {           // n != 0 => active voxel
       const float udf = d_udf[n];
       if (udf * udf >= barrierSqWorld) mask.setOnAtomic(threadID);  // retain non-barrier
   }
   ```
4. `PruneGrid<BuildT>(dGrid, dRetainMask).getHandle()` → the derived topology-only grid.

**⚠ UNITS GOTCHA (cost a debugging session):** the barrier is √3/2 *voxels*, but the UDF
sidecar is in *world* units (see step 1). So the host precomputes
`barrierSqWorld = 0.75f · voxelSize²` and passes it in; comparing `udf*udf >= 0.75f` directly
(voxel-unit assumption) makes the predicate always-false at small voxelSize → an **all-zero
retain mask** → PruneGrid builds an empty grid and then **segfaults** in `processLowerNodes`
(8 root tiles but 0 upper/lower/leaf nodes). Always sanity-check the retain mask is non-empty.

PruneGrid semantics (confirmed): its `d_srcLeafMask` is "a sidecar array of leaf masks for
**voxels to retain**" → set bit = keep. After this the UDF sidecar is no longer needed for CC.

Open follow-ups: (a) `MeshToGrid.cuh`'s doc comment says the sidecar is "voxel units" but it's
world units — worth fixing upstream. (b) `PruneGrid` segfaulting on an all-empty mask instead of
producing an empty grid / throwing is a robustness bug worth reporting.

## Build & run

Built as part of the NanoVDB examples when configured with
`-DNANOVDB_BUILD_EXAMPLES=ON -DNANOVDB_USE_CUDA=ON`:

```bash
./examples/ex_connected_components_cuda input.obj [voxelSize] [bandWidth]
# defaults: voxelSize=0.01, bandWidth=3.0
```

Reference test case (Stanford dragon, ~871k triangles): `voxelSize=0.0005` yields
~41.5M active voxels / ~220k leaves (~37% leaf occupancy), a good "few tens of millions"
working size. Topology diagnostics are read directly off the device grid via
`nanovdb::util::cuda::DeviceGridTraits` (no `deviceDownload`).

## Eventual context (parked, not a near-term goal)

This work is a building block for a future GPU mesh-to-**SDF** pipeline. In OpenVDB's
`MeshToVolume.h`, signing the UDF is done by (a) UDF, (b) a flood-fill of the exterior
that flips sign, and (c) a geometric sign fix on the zero-crossing boundary band. The
long-term intent is to replace OpenVDB's flood-fill (b) — `traceExteriorBoundaries`,
a directional sweep + iterative seed-fill — with a more robust CC-based classification.
That destination is **far off**; the immediate scope is just CC on index grids.
