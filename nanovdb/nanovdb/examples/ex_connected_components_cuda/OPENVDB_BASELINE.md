# An OpenVDB baseline for the CUDA connected-components labeling

Status: steps 1-5 of section 8 are implemented and behind `--openvdb-oracle`. Steps 6-7, the sweep
and its write-up, are not. Measured results so far are in section 10, which also records where the
implementation departed from this plan.

## 1. Purpose

`ex_connected_components_cuda` already validates the GPU labeling against a CPU union-find
oracle (`connected_components_cuda_kernels.cu:84`, `validateAgainstOracle`). This note plans a
*second*, independent baseline built on OpenVDB's pre-existing CPU segmentation
(`openvdb/tools/LevelSetUtil.h`), to run alongside the first rather than replace it.

The value of the addition is that OpenVDB's implementation is third-party code. It was not
written by the authors of the CUDA algorithm, does not share its mental model, and is already
relied on by the wider OpenVDB ecosystem. Agreement with it upgrades the claim from "the GPU
matches our own CPU code" to "the GPU matches what OpenVDB already treats as correct."

## 2. Summary

- OpenVDB's `tools::segmentActiveVoxels` / `tools::extractActiveVoxelSegmentMasks` compute the
  same partition as `nanovdb::tools::cuda::ConnectedComponents`, for inputs with no active
  tiles. Same operand set (active voxels), same relation (6-connected face adjacency).
- The two differ in output *form* and *ordering*, never in the partition. Comparisons must
  therefore treat the result as an unordered set-of-sets.
- The single semantic divergence is active tiles, which OpenVDB voxelizes and the CUDA path
  ignores. The example must reject them rather than paper over them.
- The existing flat oracle must stay. It is structurally independent in a way the OpenVDB
  baseline is not: see section 4.

---

## 3. What OpenVDB's segmentation actually computes

### 3.1 Two entry points, only one of which is relevant

`LevelSetUtil.h` exposes three public functions. Only the first two are related to what we do.

| Function | Operand set | Output |
| --- | --- | --- |
| `extractActiveVoxelSegmentMasks(volume, masks)` | active voxels | one bool mask grid per component |
| `segmentActiveVoxels(volume, segments)` | active voxels | one grid per component, values copied |
| `segmentSDF(volume, segments)` | **isosurface-crossing voxels only** | one SDF grid per component |

`segmentSDF` is *not* the same operation and must not be used as a baseline. It first calls
`extractIsosurfaceMask(inputTree, zero)` (`LevelSetUtil.h:2593`) to select only voxels
straddling the zero crossing, segments that thin shell, then re-expands each segment over the
narrow band (`ExpandNarrowbandMask`) and re-signs it (`FloodFillSign`). It is OpenVDB's
equivalent of "partition into closed surfaces first", which is conceptually the same job as
`af631daa9` on the `mesh-to-sdf` branch, but it operates on a different voxel set.

We want `extractActiveVoxelSegmentMasks`. It is the cheapest of the three because it stops at
mask grids and never copies values.

### 3.2 Operand set

`extractActiveVoxelSegmentMasks` preprocesses its input before segmenting
(`LevelSetUtil.h:2388-2395`):

```cpp
BoolTreeType topologyMask(tree, false, TopologyCopy());
tools::pruneInactive(topologyMask);
if (topologyMask.hasActiveTiles()) {
    topologyMask.voxelizeActiveTiles();
}
```

So the operand set is: active voxels, plus active tiles densified into active voxels.

`pruneInactive` is a no-op on the partition. It only drops leaf nodes that contain no active
voxels, and those contribute nothing on either side. It is, however, load-bearing for safety:
line 2430 indexes `nodeSegmentArray[0][0]` unconditionally, which would be out of bounds on an
empty vector if leaf 0 had no segments.

`voxelizeActiveTiles` is the one real divergence. See section 3.4.

### 3.3 Connectivity relation

6-connected face adjacency, identical to the CUDA algorithm.

Intra-leaf, in `nodeMaskSegmentation` (`LevelSetUtil.h:1357-1391`), a serial flood fill over a
`std::deque<Index>` visiting `pos +/- 1` (z), `pos +/- DIM` (y), `pos +/- DIM*DIM` (x), each
guarded against wrapping at the leaf boundary.

Cross-leaf, in `ConnectNodeMaskSegments` (`LevelSetUtil.h:1497-1549`), the same six directions,
mapping e.g. `ijk[2] == 0` to `pos + (DIM-1)` in the -Z neighbour leaf.

Note the difference in *traversal*, which does not affect the relation: OpenVDB probes all six
neighbour leaves (`:1461-1477`), whereas `ccForEachCrossLeafEdge` walks only +X/+Y/+Z and lets
the lower leaf own each undirected boundary. Same edges, OpenVDB just does each one twice.

### 3.4 Active tiles: the one semantic divergence

The CUDA implementation is leaf-only, by construction:

- `leafCount = ...mNodeCount[0]` (`ConnectedComponents.cuh:653`) counts leaf nodes only.
- `leaf.isActive(n)` seeds the union-find (`:306`, `:351`).
- `ccNeighborLeafIndex` resolves neighbours with `root().probeLeaf(no)` and returns `-1` when
  there is no leaf (`:444-445`).

Two consequences for a grid with active tiles, the second being the sharper one:

1. Tile voxels are never labeled.
2. **An active tile acts as a connectivity barrier.** `probeLeaf` returns null over a tile, so
   two leaves joined *through* a tile region are reported as separate components.

Failure mode (2) is silent: it inflates the component count in a way nothing downstream can
distinguish from genuine fragmentation. It is the same class of defect that `8b5a526ce`
added a stderr warning for on the iteration-cap path.

OpenVDB, by contrast, voxelizes tiles and labels them. So on tiled input the two implementations
compute genuinely different answers, and a disagreement would be expected rather than
informative.

**Decision: the example must reject tiled input outright**, before any conversion, with an
always-on check (not an assert -- see section 7.2).

For reference, the cost of the alternative. For the standard `Tree4<T,5,4,3>` configuration:

| Tile level | Voxels covered | Leaves if voxelized |
| --- | --- | --- |
| Level 1 (`InternalNode<Leaf,4>` slot) | 8^3 = 512 | 1 |
| Level 2 (`InternalNode<InternalNode,5>` slot) | 128^3 = 2,097,152 | 4,096 |
| Root slot | 4096^3 = 68,719,476,736 | 134,217,728 |

Voxelizing a root-level tile is not a slow path, it is an out-of-memory condition. If we ever
offer voxelization as an opt-in, `Tree::activeTileCount()` (`Tree.h:402`) should be used to
estimate first.

### 3.5 Verdict on equivalence

For input with no active tiles, the two implementations compute **the same partition**. Same
operand set, same relation, same cross-leaf reach. Everything that differs is output shape or
labeling convention, covered next.

---

## 4. Why this is added alongside the existing oracle, not instead of it

The existing oracle is *structurally* independent. `validateAgainstOracle` builds a flat global
coordinate hash (`coordToId`, keyed on `encodeCoord`) and runs union-find over +X/+Y/+Z
neighbour lookups (`connected_components_cuda_kernels.cu:100-138`). It performs no leaf
decomposition at all.

That matters. The central correctness claim of the CUDA algorithm is not "union-find works", it
is "segmenting each leaf in isolation and then stitching across faces yields the same partition
as labeling globally". The flat oracle tests exactly that premise, because it does not share it.

OpenVDB's implementation uses the *same three-stage decomposition*:

| Stage | `LevelSetUtil.h` | `ConnectedComponents.cuh` |
| --- | --- | --- |
| 1. Per-leaf | `SegmentNodeMask`, TBB over leaves, serial flood fill per leaf | `LeafUnionFind`, 512 threads per leaf |
| 2. Cross-leaf | `ConnectNodeMaskSegments`, pointer adjacency graph | face-mask AND, flat edge array |
| 3. Global | serial BFS over the segment graph | lock-free CAS union-find + scan |

So an OpenVDB baseline would validate the CUDA *implementation* well, but would validate the
*decomposition premise* less strongly than the flat oracle does, because a shared structural
blind spot is at least conceivable.

Keeping all three also makes disagreements localizable. With two implementations a mismatch
tells you something is wrong; with three it usually tells you which one. That matters here
because neither candidate is above suspicion: the CUDA path has the convergence-cap caveat
(section 5.5) and the OpenVDB path has the fragility in section 5.4.

---

## 5. Properties of the baseline worth knowing before trusting it

### 5.1 Component ordering differs

OpenVDB guarantees segments "sorted in descending order based on the active voxel count"
(`LevelSetUtil.h:159`, implemented `:2501-2527`).

The CUDA path assigns dense ids by *minimum global slot*: `RootFlagFunctor` flags self-roots,
an inclusive scan gives each root its id, so component 0 is the one containing the lowest
`leafComponentOffsets[leaf] + localIdx`, i.e. first appearance in breadth-first leaf order.

Same sets, different numbering. **All comparison must be partition-based**, never label-by-label.
See section 8.

### 5.2 Output form differs

OpenVDB returns `std::vector<BoolGrid::Ptr>` -- one materialized grid per component. There is no
per-voxel label output anywhere in the OpenVDB API, and no representative/parent array either.
Internally the equivalence classes are membership lists, `std::deque<std::vector<NodeMaskSegment*>>`
(`LevelSetUtil.h:2428`), built by serial BFS over the `connections` pointer graph (`:2442-2457`).

The CUDA path returns a dense per-voxel label sidecar plus a count. There is no cheap OpenVDB
path to "how many components, and which is each voxel in" -- even asking for the count forces
full materialization of N trees.

### 5.3 Cost

Stage 3 is serial *and* quadratic. `LevelSetUtil.h:2459-2465` rescans every segment to find the
next unvisited one, once per component, with no early break:

```cpp
// find first unvisited segment
for (size_t n = 0, N = leafnodes.size(); n < N; ++n) {
    NodeMaskSegmentPtrVector& nodeSegments = nodeSegmentArray[n];
    for (size_t i = 0, I = nodeSegments.size(); i < I; ++i) {
        if (!nodeSegments[i]->visited) nextSegment = nodeSegments[i].get();
    }
}
```

That is O(components x segments). The merged multi-OBJ scenes added by `c3ee877c4` are exactly
the regime that hurts: many components by construction. The existing flat oracle is O(V) hash
operations by comparison.

Allocation is also heavy: one heap-allocated `NodeMaskSegment` per leaf-local segment, each
carrying a 512-bit mask plus a `std::vector` of connection pointers, then N materialized mask
trees on top.

Implication: the OpenVDB check needs its own timing line and should be gated (section 7.5).

### 5.4 Latent fragility

`findNodeMaskSegmentIndex` returns `Index(-1)` on failure (`LevelSetUtil.h:1590-1593`) and the
caller indexes `connections[idx]` with it unguarded (`:1502-1503`). It cannot fire when
segmentation is complete, but it means malformed input surfaces as memory corruption rather
than as an error. Worth knowing if the baseline ever crashes rather than disagrees.

Separately, `SegmentNodeMask::operator()` `const_cast`s the leaf origin to stash an array index
(`:1418-1420`), which is why `topologyMask.clear()` follows immediately at `:2419`. The input
we hand it is a throwaway topology copy, so this does not affect us, but it does mean the tree
passed in is left corrupted and must not be reused.

### 5.5 Exactness

OpenVDB's flood fill is exact by construction.

The CUDA path is exact unless a leaf exhausts `MaxConvergenceIters`, in which case it
under-labels -- the case `8b5a526ce` added `leavesOverIterationCap` reporting for. Measured
worst case is 6 rounds against a cap of 64 (see `SV_CONVERGENCE.md` on the `mesh-to-sdf`
branch), so this is theoretical, but it is the one way the two can legitimately disagree on
tile-free input. If a disagreement is ever observed, check the cap warning first.

---

## 6. Getting the topology from NanoVDB into OpenVDB

### 6.1 Do not use `nanoToOpenVDB`

`NanoToOpenVDB.h` does support index grids, but both relevant overloads are built around
carrying *data*, not topology:

- `operator()(const NanoGrid<NanoIndexT>&, int blindDataID)` (`:172`) converts a blind-data
  channel and returns `openvdb::GridBase::Ptr`.
- `operator()(const NanoGrid<NanoIndexT>&, const NanoValueT* sideCar, ...)` (`:246`) needs an
  explicit sidecar array.

Neither yields "the active topology as a mask grid", which is all we need. Using them would
also pull type-erased `GridBase` downcasting into the trusted base.

### 6.2 Hand-roll the conversion

Iterate the NanoVDB leaves on the host and copy value masks into an `openvdb::MaskGrid`:

- for each NanoVDB leaf, `acc.touchLeaf(origin)` on the destination
- copy the 512-bit active mask across bit by bit (or word by word, layouts permitting -- both
  use `offset = x*64 + y*8 + z`, so a direct word copy is plausible but must be verified, not
  assumed)

This is roughly 20 lines and, more importantly, keeps the conversion auditable. The converter
must not itself be under test.

### 6.3 Verify the conversion before using it

Cheap assertions that keep the trusted base honest, checked before running any segmentation:

- leaf count matches
- active voxel count matches (`MaskGrid::activeVoxelCount()` vs `Traits::getActiveVoxelCount`)
- per-leaf origins match
- `maskGrid->tree().hasActiveTiles() == false`

If any fail, report and skip the OpenVDB check rather than reporting a spurious FAIL.

---

## 7. Integration into `ex_connected_components_cuda`

### 7.1 Build

**No `CMakeLists.txt` change is required.** `nanovdb/CMakeLists.txt:313-320` attaches both the
OpenVDB link and `-DNANOVDB_USE_OPENVDB` to the `nanovdb` INTERFACE target:

```cmake
if(NANOVDB_USE_OPENVDB)
  if(NOT OPENVDB_BUILD_CORE)
    target_link_libraries(nanovdb INTERFACE OpenVDB::openvdb)
  else()
    target_link_libraries(nanovdb INTERFACE openvdb)
  endif()
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_OPENVDB)
endif()
```

and every example links `nanovdb` (`examples/CMakeLists.txt:79`). The `OPENVDB` option on
`nanovdb_example()` is purely a skip-guard (`:60-63`), not a linkage directive.

So: guard the new code with `#ifdef NANOVDB_USE_OPENVDB`, leave line 118 of
`examples/CMakeLists.txt` alone, and the example keeps building for NanoVDB-only users while
gaining the baseline automatically when `NANOVDB_USE_OPENVDB=ON`.

`openvdb::initialize()` must be called before any OpenVDB use.

### 7.2 Tile rejection

Check on the NanoVDB side if possible, otherwise immediately after conversion, and make it
always-on. An `assert` is the wrong mechanism: a violated tile precondition does not crash, it
silently returns too many components (section 3.4). Report and skip, do not report FAIL.

### 7.3 The baseline call

`tools::extractActiveVoxelSegmentMasks(maskGrid, masks)`, not `segmentActiveVoxels` -- we need
topology only, and it avoids the per-segment value copy.

### 7.4 Comparison

Partition-based, per section 5.1. The existing bijection check at
`connected_components_cuda_kernels.cu:148-161` already has exactly the right shape and is
reusable nearly verbatim:

```cpp
std::unordered_map<uint32_t, uint32_t> gpuToOracle;  // gpu label -> oracle component
uint64_t partitionViolations = 0;
for (uint32_t id = 0; id < parent.size(); ++id) {
    const uint32_t gl = idToGpuLabel[id];
    const uint32_t oc = rootToComp[find(id)];
    auto it = gpuToOracle.find(gl);
    if (it == gpuToOracle.end()) gpuToOracle.emplace(gl, oc);
    else if (it->second != oc)   ++partitionViolations;
}
```

The only substitution is `rootToComp[find(id)]` -> the OpenVDB segment index for that voxel.
Build that by painting a `coord -> segment index` map from the N returned masks, then reusing
the same `coordToId` / `idToCoord` gather the existing oracle already performs, so both checks
share one traversal of the NanoVDB grid.

Pass criteria stay as they are:

- `gpuCount == baselineCount`
- `gpuDistinctLabels == baselineCount`
- `partitionViolations == 0`

### 7.5 Reporting and gating

Emit a separate PASS/FAIL line matching the existing format at `:163-166`, plus its own wall
time so it is visible when the O(components x segments) stage dominates.

Gate behind an opt-in `--openvdb-oracle` flag, added alongside the existing options at
`connected_components_cuda.cpp:106-108` (`--discard-surface-voxels`, `--voxel-size`,
`--band-width`) and to the usage string at `:114-115`.

---

## 8. Implementation plan

Ordered so that each step is independently verifiable.

1. **Scaffold.** Add the `--openvdb-oracle` flag and usage text in
   `connected_components_cuda.cpp`. Thread the boolean through the host/device seam declared at
   `:29` into `connectedComponentsFromMesh` (`connected_components_cuda_kernels.cu:172`).
   Guard everything OpenVDB-specific with `#ifdef NANOVDB_USE_OPENVDB`; when the flag is passed
   in a build without OpenVDB, print a clear "not compiled in" notice.

2. **Conversion + verification.** Implement the hand-rolled NanoVDB -> `MaskGrid` topology copy
   and the section 6.3 assertions. Land this with the assertions printing counts, so the
   conversion can be eyeballed on real meshes before anything depends on it.

3. **Tile rejection.** Always-on check, report and skip.

4. **Baseline invocation.** Call `extractActiveVoxelSegmentMasks`, report the segment count and
   its wall time. At this point the baseline count can already be compared by eye against the
   GPU and the flat oracle on all existing test meshes -- useful signal before the full
   comparison exists.

5. **Partition comparison.** Paint the `coord -> segment index` map, run the bijection check,
   emit PASS/FAIL. Factor the shared comparison logic out of `validateAgainstOracle` so both
   baselines use one implementation and cannot drift.

6. **Sweep.** Run across the existing test meshes at several voxel sizes, with and without
   `--discard-surface-voxels`, and with multiple merged OBJs to exercise the many-component
   regime. Record timings; if the OpenVDB stage 3 dominates badly, add a size threshold above
   which the check auto-skips with a printed notice.

7. **Document the outcome.** Append measured results to this file: agreement across the sweep,
   the cost ratio between the two baselines, and any input class where the OpenVDB path is
   impractical. Follow the pattern of `SV_CONVERGENCE.md` / `PS_COMPARISON.md`.

---

## 9. Risks and open questions

- **Cost may make it impractical on the largest cases.** Section 5.3. Mitigation is the gate in
  step 6, but if the useful many-component cases are exactly the slow ones, the check may only
  be viable on a subset. Worth measuring early (step 4) before building step 5.

- **A disagreement has no a-priori arbiter.** Three-way comparison helps, but the resolution
  procedure should be written down: check the cap warning first (section 5.5), then whether the
  input has tiles, then reduce to a minimal failing leaf pair.

- ~~**Word-level mask copy is unverified.**~~ Resolved by placing voxels by coordinate instead;
  see section 10.2. Note that the bit-by-bit fallback proposed in section 6.2 would not have
  resolved it: writing bit `n` of the destination assumes the same offset convention that a word
  copy does.

- **`openvdb::initialize()` placement.** Needs to happen once, before any OpenVDB call, and
  should not be paid when the flag is off.

---

## 10. Measured results

Steps 1-5 are implemented. The numbers below are single runs on one machine, taken to answer two
questions early: does the baseline agree, and can it be afforded. They are not the sweep of step 6.

### 10.1 Agreement

Every case ran the full partition comparison of section 7.4, not merely a count comparison.

| case | active voxels | leaves | components | violations | unassigned |
| --- | ---: | ---: | ---: | ---: | ---: |
| one mesh, full band | 338,725 | 1,732 | 1 | 0 | 0 |
| one mesh, barrier-pruned | 240,533 | 1,732 | 2 | 0 | 0 |
| three meshes merged, pruned | 834,972 | 5,871 | 50 | 0 | 0 |
| dense mesh, coarse, pruned | 6,874,318 | 29,603 | 12,279 | 0 | 0 |
| dense mesh, medium, pruned | 18,538,056 | 89,738 | 14,550 | 0 | 0 |

The check was also confirmed to fail when it should, which a check that has only ever passed does
not establish. Relabeling a single voxel into another component produced exactly one partition
violation; withholding one segment when painting the slot array produced the expected count of
unassigned voxels and dropped the distinct-label count. Both were injected temporarily and removed.

### 10.2 Departure from section 6.2: place voxels by coordinate

Section 6.2 proposed copying leaf masks word by word, with a bit-by-bit copy as the safe fallback,
and section 9 flagged the word copy as resting on an unverified assumption. **Neither form removes
that assumption**: both write destination offset `n` for source offset `n`, so both are correct only
if the two libraries agree on what `n` means.

The implementation converts through coordinates instead -- NanoVDB's `OffsetToLocalCoord` out,
OpenVDB's `LeafNode::coordToOffset` in -- so each library applies its own convention and the copy is
correct whether or not they agree. Cost is negligible: the conversion is under 0.2 s at 18.5 M
voxels, against tens of seconds for the segmentation it feeds.

### 10.3 Cost

| case | GPU labeling | union-find oracle | conversion | OpenVDB segmentation |
| --- | ---: | ---: | ---: | ---: |
| one mesh, pruned | 0.48 ms | 58 ms | 2.5 ms | 12.6 ms |
| three meshes merged, pruned | 0.94 ms | 217 ms | 8.0 ms | 27.8 ms |
| dense mesh, coarse, pruned | 4.8 ms | 1.88 s | 58 ms | 7.53 s |
| dense mesh, medium, pruned | 12.6 ms | 5.91 s | 163 ms | 34.8 s |
| dense mesh, fine, pruned | -- | -- | -- | **> 30 min, killed** |

The finest case has 79,734,227 active voxels in 525,245 leaves and 333,703 components. Without
`--openvdb-oracle` the same run completes, so the OpenVDB stage is what does not finish, not the
labeling or the union-find oracle.

Two things follow.

**The prediction of section 5.3 holds.** Between the coarse and medium cases the component count
grows only 1.19x while the segmentation cost grows 4.6x. Components x leaves grows 3.6x over the
same pair, so that product tracks the cost far better than any single quantity does, though it still
under-predicts -- treat it as a lower bound on growth rather than a law. Extrapolating it to the
finest case gives 134x the medium case, i.e. over an hour, consistent with what was observed.

**A gate on size would not work.** The medium case has 77x the voxels of the single-mesh case and
takes 2,760x as long. Voxel count, leaf count and component count each fail as predictors on their
own; the product of the last two is the smallest quantity that tracks the cost. Step 6's gate should
therefore be expressed in components x leaves, which is a change to the plan in section 8.

### 10.4 Two axes: resolution scales, component count does not

Section 10.3 varies voxel size and component count together, which hides which one the cost follows.
Holding the component count near-constant and refining a simple closed mesh separates them:

| voxel size | active voxels | components | GPU labeling | OpenVDB | ratio | OpenVDB throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 59,908 | 10 | 0.29 ms | 6.6 ms | 22.6x | 9.0 M voxel/s |
| 0.01 | 240,533 | 2 | 0.43 ms | 11.8 ms | 27.5x | 20.4 M voxel/s |
| 0.005 | 964,846 | 5 | 0.81 ms | 19.0 ms | 23.4x | 50.9 M voxel/s |
| 0.0025 | 3,864,627 | 4 | 2.30 ms | 42.4 ms | 18.4x | 91.0 M voxel/s |

**On the resolution axis alone, OpenVDB scales well.** 64x the voxels cost 6.4x the time; throughput
rises rather than falls, as fixed setup is amortized. The ratio to the GPU stays near 20x across the
whole range, both implementations amortizing their own overhead in the same way.

That 20x is the honest CPU-versus-GPU gap for this operation. The 2,760x seen in section 10.3 is not
that gap: it is the grouping stage of section 5.3 collapsing, and it appears only when the component
count is large.

**The two axes are coupled in practice, though.** Refining a geometrically complex mesh splits thin
features apart, so components multiply faster than voxels do:

| voxel size | active voxels | components |
| ---: | ---: | ---: |
| coarse | 6,874,318 | 12,279 |
| medium | 18,538,056 (2.7x) | 14,550 (1.2x) |
| fine | 79,734,227 (4.3x) | 333,703 (**23x**) |

So "does it scale with resolution" has no answer independent of the input. A simple closed surface
keeps its component count and stays linear; a mesh with thin structure moves onto the quadratic axis
as it is refined, which is why the finest case above does not finish.

### 10.5 Still open

- The **tile rejection path of section 7.2 is untested**. `countActiveTiles` returns zero on every
  input this example can build, because the grids come from a leaf-only rasterization and prune, so
  the reporting branch has never been taken.
- The baseline **downloads the label array a second time**, duplicating the download the union-find
  oracle already performs. Section 7.4's suggestion of sharing one traversal is not implemented.

---

## Appendix A: source references

| Reference | Location |
| --- | --- |
| Existing flat oracle | `connected_components_cuda_kernels.cu:84-168` |
| Bijection check to reuse | `connected_components_cuda_kernels.cu:148-161` |
| Oracle call site | `connected_components_cuda_kernels.cu:231` |
| Example CLI options | `connected_components_cuda.cpp:106-108`, usage `:114-115` |
| Example declaration | `examples/CMakeLists.txt:118` |
| `nanovdb_example()` skip-guard | `examples/CMakeLists.txt:60-63` |
| OpenVDB link + define | `nanovdb/CMakeLists.txt:313-320` |
| CUDA leaf-only iteration | `ConnectedComponents.cuh:653`, `:306`, `:351`, `:444-445` |
| CUDA dense id assignment | `ConnectedComponents.cuh:610-612` |
| OpenVDB public API | `LevelSetUtil.h:148-177` |
| OpenVDB preprocessing | `LevelSetUtil.h:2388-2395` |
| OpenVDB intra-leaf flood fill | `LevelSetUtil.h:1357-1391` |
| OpenVDB cross-leaf connect | `LevelSetUtil.h:1497-1549` |
| OpenVDB quadratic grouping | `LevelSetUtil.h:2459-2465` |
| OpenVDB descending sort | `LevelSetUtil.h:2501-2527` |
| `Index(-1)` fragility | `LevelSetUtil.h:1502-1503`, `:1590-1593` |
| Origin `const_cast` hack | `LevelSetUtil.h:1418-1420` |

## Appendix B: deferred -- a native OpenVDB port

Considered and set aside in favour of validation. Recorded here because the design work is done
and the gap it fills is real: OpenVDB has no label-sidecar API at all (section 5.2).

Sketch:

```cpp
template<typename GridOrTreeT>
typename TreeAdapter<GridOrTreeT>::TreeType::template ValueConverter<uint32_t>::Type::Ptr
connectedComponents(const GridOrTreeT& volume, Index32& componentCount);
```

Key design points established:

- **Output as a label tree, not aux buffers.** `LeafManager` auxiliary buffers cannot hold a
  type other than the tree's own `ValueType` (`LeafManager.h:95`), so labeling a `FloatGrid`
  would require type-punning `float` storage as `uint32_t` -- UB, broken for non-4-byte value
  types, and impossible for `MaskGrid`/`BoolGrid` where `LeafBuffer<bool,Log2Dim>`
  (`LeafBuffer.h:488`) has no value array at all. Requesting aux buffers also triggers an
  immediate full copy of every primary buffer via `syncAllBuffers` (`LeafManager.h:716-721`)
  that would then be overwritten.

  Instead: `TopologyCopy` the input into an integer tree and make the `LeafManager` be over
  *that*. The label channel is then simply the output tree's own primary buffer.

- **Use `ValueConverter`, not a hardcoded `UInt32Tree`.** `UInt32Tree` is specifically
  `Tree4<uint32_t,5,4,3>`, and the `TopologyCopy` constructor throws `TypeError` on a
  configuration mismatch. `typename InTreeT::template ValueConverter<uint32_t>::Type`
  (`Tree.h:219`) preserves the input's configuration, which is the `LevelSetUtil.h` idiom.

- **`Grid<UInt32Tree>` is not a registered type.** `UInt32Tree` exists (`openvdb.h:60`) but
  there is no `UInt32Grid` alias and it is absent from `IntegerGridTypes` (`openvdb.h:95`), so
  it is not registered by `openvdb::initialize()` and will not round-trip through the IO
  registry. Returning a bare *Tree* avoids this; returning a *Grid* argues for `Int32Grid`.

- **The 3-arg `TopologyCopy` is documented as faster** than the 4-arg form (`Tree.h:266-268`),
  so prefer it when active and inactive seed values are the same sentinel.

- **`LeafManager` leaf ordering is deterministic and topology-derived**, so a topology-copied
  tree indexes identically to its source. `initLeafArray` gathers level-1 nodes via
  `getNodes` (`LeafManager.h:594`); `RootNode` children live in a `std::map<Coord, NodeStruct>`
  (`RootNode.h:157`); `InternalNode::beginChildOn()` walks the child mask in ascending order;
  and the parallel path seeks by prefix sum (`LeafManager.h:646-668`) so it produces an array
  identical to the serial one. Nothing depends on allocation order or thread scheduling.

  This correspondence can be designed away entirely: `TopologyCopy` duplicates the value masks,
  so the label tree alone carries everything the algorithm reads, and the input can stay `const`
  and untouched.

- **Caveat: any structural mutation invalidates it.** `LeafManager` caches raw `LeafType*`.
  `pruneInactive`, a leaf-creating `setValue`, `topologyIntersection` or `voxelizeActiveTiles`
  on either tree desynchronizes the indices and dangles the pointers.

- **The output would be dense, not compact.** The NanoVDB sidecar is `activeVoxelCount+1`
  entries indexed by `leaf.getValue(n)`, because `ValueOnIndex` supplies that addressing for
  free. OpenVDB has no equivalent build type -- its nearest neighbour, `MaskTree`, has topology
  but no per-voxel payload -- so an `Int32Tree` carries 512 slots per leaf regardless of
  occupancy. At typical narrow-band occupancy that is roughly 2x the label memory. Closing the
  gap means a flat array plus a per-leaf prefix-sum offset table, i.e. re-deriving
  `ValueOnIndex` on the CPU side.
