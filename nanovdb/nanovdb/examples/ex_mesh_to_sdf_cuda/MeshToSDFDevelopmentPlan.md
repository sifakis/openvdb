# ex_mesh_to_sdf_cuda

A NanoVDB / CUDA example that builds toward a **connected-components (CC) labeling**
on NanoVDB index grids. This document is the running design notes for that work.

> **Umbrella plan:** CC is step 3 of the larger mesh-to-SDF pipeline. The overall
> trajectory — UDF + triangle-index sidecar, barrier-voxel signing (replicating
> OpenVDB's `ComputeIntersectingVoxelSign`), and the per-level `invertMask` level-set
> representation extension — lives in [`MeshToSDF_PipelinePlan.md`](./MeshToSDF_PipelinePlan.md).

> **OpenVDB usage.** The core pipeline (steps 1–6) is pure NanoVDB + CUDA — the index↔world
> transform uses `nanovdb::Map` directly and the OBJ reader uses NanoVDB types. `validateMeshToSdf`
> *optionally* cross-checks the signs against OpenVDB's `meshToLevelSet` when the example is built
> with `NANOVDB_USE_OPENVDB=ON` (guarded by `#ifdef NANOVDB_USE_OPENVDB`); without it, the
> CPU-oracle and analytic checks still run.

## Files

| File | Role |
| :--- | :--- |
| `mesh_to_sdf_cuda.cpp`         | Host driver: arg parsing, OBJ reader, synthetic test meshes, `nanovdb::Map`, calls the seam. No CUDA *code* (only header-only handle/buffer types). |
| `mesh_to_sdf_cuda_kernels.cu`  | CUDA / NanoVDB side. Three passes over an opaque `SdfPipeline`: **`buildMeshToSdf`** (steps 1–6: rasterize → prune → CC → sign → fill), **`validateMeshToSdf`** (CPU oracles + OpenVDB / analytic cross-checks), **`exportMeshToSdf`** (Polyscope dump). Plus `printGridDiagnostics()` and the synthetic `testRootInteriorFlood()` unit test. |
| `scripts/mesh_to_sdf_viewer.py`                  | Polyscope viewer for the `exportMeshToSdf` dump (`.ccvis` + `.fill`). Not part of the build. |
| `MeshToSDF_PipelinePlan.md`             | Umbrella pipeline plan (steps 1–6). |
| `MeshToSDFDevelopmentPlan.md`           | This document (CC / step-3 running notes). |
| `standalone/cc_vis.cpp`                 | Standalone 2D CPU visualizer of the SV hook/compress primitives (see `standalone/README.md`). **Not** part of the example build — it lives one directory down so the `nanovdb_example` source glob (non-recursive) skips it; otherwise its `main()` would collide with the driver's. |

The actual per-leaf CC kernel lives in the library, not the example:
`nanovdb/tools/cuda/ConnectedComponents.cuh`; the SDF domain logic lives in
`nanovdb/tools/cuda/MeshToSDF.cuh`.

The host/device seam is a small set of forward-declared functions. `SdfPipeline` is opaque to the
`.cpp` (it embeds CUDA-only types): the driver only holds an `SdfPipeline*` returned by
`buildMeshToSdf`, passes it to `validateMeshToSdf` / `exportMeshToSdf`, then `freeSdfPipeline`.

## Pipeline (incremental)

1. **`computeUDFAndIndex(points, triangles, map, bandWidth)` → `{ handle, udf sidecar, index sidecar }`** *(done)*
   Rasterize the mesh into a narrow-band `ValueOnIndex` grid (every active voxel gets a
   dense index in `[1, N]`) plus a UDF sidecar of `N+1` floats (and a nearest-triangle-index sidecar):
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

3. **Per-leaf connected-component counting (CUDA)** *(IMPLEMENTED + VALIDATED —
   `ConnectedComponents::processLeafConnectedComponents()`)*
   First CC milestone: for every leaf *in isolation*, count the number of distinct
   6-connected components formed by its active voxels, into a device array of one
   `uint16_t` per leaf (`deviceLeafComponentCounts()`). Cross-leaf connectivity is
   ignored at this stage. See "Per-leaf CC kernel" below.

   **Validation (in the example, `validateMeshToSdf()`):** after the kernel runs, the example
   copies the device grid blob into a scratch host buffer (NanoVDB grids are
   position-independent, so a raw byte copy is a valid host grid; the input handle is left
   untouched — no `deviceDownload` residue), runs a host union-find oracle harvested from
   `TEST(TestNanoVDBCUDA, LeafConnectedComponents)`, and compares the two per-leaf count
   arrays elementwise. Prints PASS/FAIL with leaf count, mismatch count, and gpu/cpu totals.

   **✅ STATUS: PASSES (resolved 2026-06-08).** Validates deterministically against the oracle
   across runs and resolutions: dragon @ 0.005 (2,013 leaves), 0.002 (13,442 leaves), and
   0.0005 (220,335 leaves) all report 0 mismatches with gpu total == cpu total on every run.

   **The bug (and how it was localized).** The kernel previously *under*-counted (spuriously
   *merged* components) on ~0.25% of leaves, with the mismatch set/count varying run to run.
   Two observations pinned it down: (a) the derived grid and the CPU oracle are fully
   deterministic — same active-voxel/leaf counts and the same `cpu=` total every run — so the
   drift was purely in the device kernel; and (b) each SV primitive is a deterministic
   function of its input buffer (hooks use a commutative min; reads are from `cur`, writes go
   to `nxt`), so a *correct* execution cannot drift at all. Run-to-run variation therefore
   implied undefined behavior, not merely a logic error in the hook/compress schedule.

   **Root cause — a divergent `__syncthreads()`, NOT a parent-buffer data race.** The
   convergence loop had no barrier between a thread *reading* `changed` (the `break` test) and
   the *next* iteration's `if (n==0) changed = 0` reset. A thread that did not break could race
   ahead and clear `changed` while slower threads were still evaluating their break test; if
   they then disagreed on `break`, the block split across different `__syncthreads()` calls —
   divergent-barrier UB that intermittently corrupted that block's shared memory. That matches
   the signature exactly: rare, nondeterministic, and always toward over-merge. (An earlier
   guess that it was a data race on the shared parent buffers was wrong — swapping the
   block-scoped atomics for device-scope ones changed nothing.)

   **Confirmation tooling note.** `compute-sanitizer --tool racecheck` could *not* be used to
   localize this on the dev box: the only available sanitizer is 2022.4.1 (CUDA 12.0), which
   predates and cannot instrument this Blackwell / `sm_120` GPU (it aborts before the first
   instrumented launch, even on a trivial program). The bug was found by the determinism
   reasoning above instead.

   **Fix.** A single `__syncthreads()` at the end of the loop body, after the `break` test, so
   no thread resets `changed` until every thread has read it. The break is then uniform across
   the block and the loop's barriers can no longer diverge. The device-vs-oracle harness in
   `validateMeshToSdf()` is the standing regression guard.

4. **Global connected-components labeling (CUDA)** *(IMPLEMENTED + VALIDATED — `processCrossLeafEdges`
   + `processComponentLabels`)*
   Merge the leaf-local components across leaf boundaries so two components share a global label iff
   connected through adjacent active voxels — *including across leaves*. Two stages, both validated
   against host oracles in `validateMeshToSdf()` (0 mismatches up to the dragon @ 0.0005): **step 5** ANDs the
   touching per-component face masks over each leaf's +X/+Y/+Z neighbors to emit a cross-leaf edge
   list, and **step 6** runs a lock-free union-find over those edges to assign each component a
   representative (its class's minimum slot). The number of distinct representatives is the true
   global component count. See "Leaf-local component masks and face flags" steps 5–6 below.
   **Remaining follow-up:** the per-active-voxel label scatter (voxel → its component's
   representative) — not yet implemented.

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

## Per-leaf CC kernel (`processLeafConnectedComponents`)

✅ **Implemented and validating against the CPU oracle.** A divergent-`__syncthreads()` bug in
the convergence loop (fixed) previously caused rare, nondeterministic under-counts; see
milestone 3 above for the root cause and fix.

The schedule below relies on a subtle invariant: the loop's `break` must be **uniform** across
the block (all threads read the same `changed`), otherwise the per-iteration `__syncthreads()`
calls diverge. This is enforced by a barrier after the `break` test so no thread resets
`changed` while others are still reading it.

`LeafComponentCountFunctor` in `nanovdb/tools/cuda/ConnectedComponents.cuh`, launched via
`operatorKernel`, **one block per leaf, 512 threads (one per voxel offset `n ∈ [0,512)`)**.
A Shiloach–Vishkin union-find runs entirely in shared memory.

- **Forest representation:** a `parent` array of leaf-local voxel offsets, double-buffered
  (Jacobi: read `cur`, write `nxt`, swap — the swap is done identically by every thread so
  the per-thread register pointers stay in sync). `parent[n] = n` for active roots, a smaller
  active offset for non-roots, `-1` (sentinel) for inactive voxels. Inactive entries are
  carried through every op unchanged, so `parent[n] >= 0` is a stable "is active" test.
  Two `int[512]` buffers = 4 KB shared/block.
- **Three primitives** (`ccHook`, `ccCompress`, plus the neighbor-min helper):
  - **SV-Hook:** each vertex `v` finds the min label `m` over its (≤6) active in-leaf face
    neighbors; if `m < parent[v]` it lowers the *root* slot `parent[parent[v]]` via
    `atomicMin` (shared-memory atomic; deterministic because min is commutative/associative).
  - **Compress:** pointer-jumping `parent[v] ← parent[parent[v]]`, halves tree depth per call,
    no atomics (each thread writes only its own slot).
- **Schedule (as specified):** one **SV-Hook**, then **3 = log₂(8) Compress** steps
  unconditionally (a warm-up sized to the leaf's `DIM`), then **alternate (Hook, Compress)**
  until a full iteration changes nothing. Convergence is detected with a shared `changed`
  flag (`__syncthreads` between phases). A `MaxConvergenceIters = 64` cap guards against a
  non-terminating bug — it is far above the worst case for an 8³ leaf
  (~`log₂(depth) + log₂(#local minima) ≲ 18`), not a limit on any legitimate input.
- **Output:** component count per leaf = number of surviving roots (`parent[n] == n`), summed
  with a shared `atomicAdd` and written as one `uint16_t` per leaf. Worst case is 256
  (3D checkerboard in an 8³ leaf), so `uint16_t` suffices.
- **Offset layout:** x-major, `n = (x<<6)|(y<<3)|z`, so the +X/+Y/+Z in-leaf neighbors are
  `n±64 / n±8 / n±1` (guarded against the leaf faces). Matches the CPU oracle.

For a 2D, CPU, single-step-at-a-time intuition for exactly these primitives and schedule, see
`standalone/cc_vis.cpp`.

## Leaf-local component masks and face flags (`processLeafConnectedComponents`, continued)

Steps 1–6 below are **✅ implemented and validated** (steps 1–4: 2026-06-09/11; steps 5–6: 2026-06-12).

> **✅ Validated against a CPU oracle (2026-06-11).** `validateMeshToSdf()` in the example now runs a host
> oracle (`cpuMasksFaces`) that independently rebuilds each component's `Mask<3>` and its six face
> bitmasks **directly from voxel coordinates** (deliberately *not* mirroring the kernel's
> shift/`0x0101…` extraction), then compares them elementwise against the device buffers.
> This caught and fixed a shared-memory bug: the mask-fill kernel's **anonymous** `__shared__`
> union was compiled to per-thread *local* storage, so each warp-leader's ballot was invisible to
> the threads doing the global write — every component mask came out empty. Fixed by making it a
> **named** `__shared__` union (see step 3). With the fix, the GPU per-component masks **and** the
> six-face extraction match the oracle with **0 mismatches** up to the dragon @ 0.0005
> (337,563 components / 220,335 leaves).

The plan promotes each leaf-local component to a first-class record, detects which records touch
across leaf faces, and then runs the same hook/compress union-find one level up — on a graph
whose vertices are leaf-local components rather than voxels.

1. **Inclusive sum over `leafComponentCounts`.** ✅
   Scans the per-leaf uint16_t counts with CUB `DeviceScan::InclusiveSum` into a
   `[leafCount + 1]` offsets array of `uint64_t` (element `0` = 0 via `cudaMemsetAsync`; elements
   `1..leafCount` upcast from uint16_t via `lambdaKernel` then summed in-place). Element `leafCount`
   is the aggregate `mLeafComponentAggregateCount = K`. The offsets array is `mLeafComponentOffsets`.

2. **Allocate `K` `nanovdb::Mask<3>` objects.** ✅
   `mLeafComponentMasks` (`K × sizeof(Mask<3>)`). No zero-init needed: the mask-fill kernel
   writes all 16 uint32_t words of every mask unconditionally.

3. **Mask-fill kernel (`LeafComponentMaskFunctor`).** ✅
   One block per leaf, 512 threads. Re-runs the SV union-find in shared memory (identical schedule
   to the count kernel). Then enumerates leaf-local components in ascending root-label order using
   a while-true loop:

   - **Block-wide min** via `cub::BlockReduce<uint32_t, LEAF_SIZE>` with `::cuda::minimum<uint32_t>{}`.
     `CC_INACTIVE = -1` recasts to `0xFFFFFFFF` and thus never wins. When the min equals
     `0xFFFFFFFF` all entries are inactive and the loop exits.
   - **`__ballot_sync`** collects the 32-bit membership word for each warp. Written to
     `sMaskWords_u32[warpID]` in shared memory by `laneID == 0`.
   - **Coalesced GMEM write:** after `__syncthreads()`, `tID < 8` writes
     `mask.words()[tID] = sMaskWords[tID]` via the `uint64_t` union view — one 64-byte transaction
     for the whole mask.
   - **Erase:** matched entries set to `CC_INACTIVE` so they don't win a future min. `++localCompIdx`.

   Key shared-memory layout: a **named** `__shared__` union holds both views over the same 64 bytes.
   It must be *named* — an anonymous `__shared__` union was miscompiled to per-thread local storage,
   making each warp-leader's ballot invisible to the threads doing the global write (fixed 2026-06-11):
   ```cpp
   __shared__ union { uint32_t u32[16]; uint64_t u64[8]; } sMaskU;  // u32: ballot/warp; u64: Mask<3> words
   uint32_t* sMaskWords_u32 = sMaskU.u32;
   uint64_t* sMaskWords     = sMaskU.u64;
   ```
   Two `__syncthreads()` per component iteration: SYNC1 after the ballot writes + erase (makes
   `sMaskWords_u32` visible and `cur` erases committed); SYNC2 after the GMEM mask write + face
   extraction (orders those global writes before the next `BlockReduce` call).

4. **Per-component face flags.** ✅
   Allocated as `mLeafComponentFaceMasks` (`K × 6 × sizeof(uint64_t)`), cast as `uint64_t(*)[6]`.
   Face indices are named by `LeafNeighborTap` enum: `{minusX, plusX, minusY, plusY, minusZ, plusZ}`.

   Extracted by `tID == 0` inside the same while-true loop, reading from `sMaskWords[x]`
   immediately after SYNC1 (no global read-back needed):

   - **±X:** `face[minusX] = sMaskWords[0]`, `face[plusX] = sMaskWords[7]` (trivial — whole words).
   - **±Y:** shift-accumulate the bottom / top byte of each word from `x=7` downto `0`; result bit
     index `= x*8 + z` (x major, z minor).
   - **±Z:** shift-accumulate `w & 0x0101…` / `(w >> 7) & 0x0101…` from `x=7` downto `0`; result
     bit index `= y*8 + x` (y major, x minor).

   Cross-leaf connectivity check (step 5) then reduces to a single `(faceA & faceB) != 0` bitwise AND.

5. **Detect cross-leaf connectivity via face flags (`processCrossLeafEdges`).** ✅ *(2026-06-12)*
   One block per leaf. For each `+X`/`+Y`/`+Z` neighbor leaf — located by
   `root().probeLeaf(origin.offsetBy(±8))` and mapped back to a linear leaf index by typed pointer
   difference against `getFirstNode<0>()` — pair every local component of this leaf with every local
   component of the neighbor and test `faceL[+axis] & faceN[-axis] != 0`. Visiting only `+axis`
   neighbors visits each undirected boundary (and each component pair across it) exactly once, so no
   dedup is needed. Materialized with the **count → scan → scatter** idiom; a shared
   `ccForEachCrossLeafEdge` helper drives both passes so their iteration + AND test cannot drift.
   Output: `mCrossLeafEdges` = E × `CrossLeafEdge{uint32_t a,b}` (canonical `a<b`, global slots).
   **Validated:** the example recomputes the edge set from the (already-validated) face masks via the
   same host `probeLeaf` walk and compares as a *sorted set* (GPU emit order is atomic-nondeterministic);
   0 mismatches up to the dragon @ 0.0005 (692,411 edges).

6. **Global union-find on the component graph (`processComponentLabels`).** ✅ *(2026-06-12)*
   Lock-free union-find over the edge list: `cc_detail::ccUnite` finds both roots and CAS-links the
   larger root under the smaller (64-bit `atomicCAS`, retry on contention). Three `lambdaKernel`
   passes: init (`parent[s]=s`), unite (1 thread per edge — a single pass suffices since each
   `ccUnite` self-retries), flatten (`parent[s]=ccFind(s)`). Each class's representative is its
   **minimum slot** (the minimum is never the "larger" root, so it stays put), making the result
   order-independent and deterministic. Output: `deviceComponentParent()` = K × uint64; the number of
   distinct representatives (`parent[s]==s`) is the true **global component count**.
   **Validated:** a host union-find over the same edges (same min-linking) matches `parent[s]`
   elementwise; 0 mismatches. **Remaining follow-up:** per-voxel label scatter (voxel → its
   component's representative) to emit a per-active-voxel label sidecar — not yet implemented.

**Synthetic regression test.** `TEST(TestNanoVDBCUDA, ConnectedComponentsMultiSphere)` (in
`unittest/TestNanoVDB.cu`) runs steps 1–6 on procedurally built **pruned narrow bands** with known
closed-form component counts — no mesh, fully deterministic: one sphere → 2 (inner + outer shell),
N well-separated spheres → 2N, and two heavily-overlapping spheres → 2 (merged into one solid). It
exercises both disjoint components and cross-leaf merging. (The live mesh path is validated in the
example's `validateMeshToSdf()`; e.g. `sphere.obj` reports 2 global labels at voxelSize 0.02/0.01/0.005.)

**Open design questions**

- exact metadata to attach per component-slot if needed downstream (source leaf id, local root id) —
  currently the global representative array suffices;
- *(resolved)* materialize cross-leaf edges explicitly vs. hook directly while scanning — we
  **materialize** an explicit edge list, so each stage is independently CPU-oracle-validated;
- *(resolved)* neighbor-leaf lookup — `root().probeLeaf(origin.offsetBy(±8))` + pointer-diff index
  (no `NodeManager` needed);
- whether to lift the same idea to the lower/upper internal-node levels, or keep the current flat
  representative graph + union-find (current choice).

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
./examples/ex_mesh_to_sdf_cuda input.obj [voxelSize] [bandWidth]
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
