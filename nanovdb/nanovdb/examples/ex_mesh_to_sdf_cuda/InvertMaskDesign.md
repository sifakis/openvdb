# Invert-mask construction — notes toward a refactoring

> **Status: partial design discussion, in progress. Nothing implemented.** This records what the
> current implementation does, one structural proposal that was examined and found sound, and the
> observations that motivate a refactoring. It stops where the discussion stopped; §7 lists what is
> still open.
>
> Background on what the masks *mean* is in `MeshToSDF_PipelinePlan.md` §5 (the representation) and
> §6-7 (the fill design). This document is about how the fill is **built**, not what it represents.

---

## 1. What exists today

The sign of the field off the narrow band is carried by four sidecars, produced by three methods on
`sdf_detail::SurfaceSigner`:

| level | sidecar | length | produced by |
|---|---|---|---|
| leaf | `Mask<3>` | `nodeCount[0]` | `fillLeafInvertMask()` |
| lower | `Mask<4>` | `nodeCount[1]` | `fillCoarseInvertMasks()` |
| upper | `Mask<5>` | `nodeCount[2]` | `fillCoarseInvertMasks()` |
| root | `uint8[P·Q·R]` | dense bbox lattice | `fillRootInteriorMask()` |

`sdf_detail::signedSignAt()` composes them into a sign at any coordinate by one tree descent,
stopping at the first level with no child.

Roughly 155 lines of host orchestration driving ~405 lines of device functors
(`MeshToSDF.cuh` 695-1100). The three methods are always called as a fixed sequence of three, at both
call sites — `signSurface()` (per surface) and `fillOnOriginal()` (once on the composed signs).

## 2. The bootstrap chain

Each level is seeded **only** by the level below it:

```
sign ──► leafInvert ──► lowerInvert ──► upperInvert ──► rootInterior
```

Only the leaf level reads the sign sidecar. Everything above reads the completed mask beneath it, and
the root level reads all three. This is a strict ordering dependency, and **nothing currently
enforces it** — it holds because of the order the three methods happen to be called in.

### 2a. The leaf level — how the whole thing bootstraps

`FillLeafInvertMaskFunctor` runs one block per leaf, 512 threads, entirely in shared memory:

| array | domain | role |
|---|---|---|
| `sAct[n]` | all voxels | is it active? → **wall** |
| `sInj[n]` | active voxels only | active **and** `sign == -1` → **source** ("injector") |
| `sInv[n]` | inactive voxels only | marked interior → **output** |

`sInj` and `sInv` are disjoint by construction, and only `sInv` is packed out into the leaf's
`Mask<3>`. `d_sign` is indexed by the `ValueOnIndex` slot `leaf.getValue(n)`, not by the voxel
offset `n`; the `act &&` short-circuit matters, since an inactive voxel's slot is the background 0.

The update rule is a **pull**, not a push:

```cpp
feeds(m) = sInj[m] || (!sAct[m] && sInv[m])
if (!act && !sInv[n] && any of the 6 face neighbours feeds(m)) -> sInv[n] = 1
```

Each thread owns voxel `n` and writes only `sInv[n]`, so the flood needs **no atomics**. The two
terms of `feeds()` are the two kinds of source — an interior band voxel (the original seed) or an
inactive voxel already filled (the propagation) — which collapses the design doc's two conceptual
steps (§6a: "inject onto inactive neighbours, then propagate inactive→inactive") into one uniform
predicate. The push and pull formulations reach the same least fixed point; only the implementation
differs.

Two properties make the pull version correct: read/write separation is achieved with a **barrier
rather than double buffering** (read neighbours into a register, `__syncthreads()`, then write), and
the flood is **monotone** — bits only go 0→1 — so a stale read can only delay convergence, never
corrupt the result.

Three scope facts, each deliberate:

- **6-adjacency only** (`n±64 / n±8 / n±1`), not 18 or 26.
- **Strictly in-leaf.** The bounds guards mean nothing crosses a leaf boundary. This is §6b: each
  materialized leaf self-labels, because monotone distance shells guarantee that a leaf containing an
  interior *inactive* voxel also contains an interior *active* seed of its own. The load-bearing
  argument is the distance field, not connected components.
- **Active voxels are walls.** `feeds()` requires `!sAct[m]` on the propagation term, so the flood
  never passes *through* an active voxel — which is what stops it leaking across the barrier shell.

### 2b. A divergence from the design doc, and why it is harmless

§6a specifies the seeds as set-(3) voxels — *deep interior, non-barrier*. The code uses any active
voxel with `sign == -1`, which after `signBarrier()` includes interior-signed **barrier** voxels.

The two coincide on valid input: a barrier voxel has `d <= sqrt(3)/2`, so any face neighbour has
`d <= 1.87 < bandWidth` and is therefore active — meaning a barrier voxel provably has no inactive
face neighbours and can never actually inject. The code's predicate is simpler and strictly more
robust on degenerate input, so the doc's distinction appears unnecessary.

## 3. The proposed two-phase split

The construction divides into two fundamentally different kinds of work:

1. **Fill sidecars parallel to existing nodes.** Leaf, lower and upper invert masks are sized by
   `nodeCount[0]/[1]/[2]` and are orthogonal to the tree. **No topology change whatsoever.**
2. **Materialize new root tiles**, if the flood finds root-tile-sized deep-interior regions that do
   not yet exist in the root table.

Phase 2 is an outlier: it requires a 4096^3 region entirely inside the level set. `VISUALIZATION.md`
notes that root tiles "essentially never appear for real meshes". So the API should not be shaped
around it.

### 3a. Correction: phase 2 is designed but **not shipped**

Nothing in `MeshToSDF.cuh` creates root tiles — no `addTile`, no `mTableSize` write, no key sort, no
rebuild. `MeshToSDF_PipelinePlan.md` §7e designed exactly that; the shipped code took §8b's
query-time path instead. What `fillRootInteriorMask()` produces is `mRootInterior`, a dense
`P x Q x R` array of `uint8`, which `signedSignAt()` consults when the root probe finds no child:

```cpp
if (!tile || !tile->isChild()) {      // absent root region: consult the sidecar
    const int i = (ijk[0] >> 12) - rootTileMin[0], ...
    return interior ? -1 : +1;
}
```

So today the tree is never touched at **any** level, root included. Phase 2 in the sense of §7e does
not currently exist; the present root stage is a fourth sidecar, not a topology edit.

### 3b. Can phase 2 be deferred? Yes — verified

`RootFaceSeedFunctor<LEVEL>` reads exactly five things: `d_grid`, `d_sign`, and the three completed
invert masks. Every one is a **durable output of phase 1**. It allocates and frees its own
`sawInt`/`sawExt`/`wall` accumulators, and derives `rootTileMin`/`rootDims` from the grid bbox
internally. So the root stage is already a pure function of phase 1's outputs; the current code
merely happens to call it immediately.

**No transient needs retaining.** The coarse evidence buffers (`lowSawInt`/`lowSawExt`/`upSawInt`/
`upSawExt`) are scoped to `fillCoarseInvertMasks()` and freed at its scope exit; nothing downstream
ever sees them.

**One information loss, and it is benign.** The unanimity gate (`sawInt && !sawExt`) collapses three
distinct states — unanimously exterior, mixed, and no-evidence — into the same `bit = 0`, and that
distinction is unrecoverable afterwards. But phase 2 consumes only the collapsed form: it asks
`d_lowerInvert[node].isOn(slot)` and treats everything else as exterior. Deferral is therefore
**semantically exact** — a bit-identical result whenever it runs, not merely an acceptable
approximation.

**The layering survives the materializing variant too.** If phase 2 later becomes real §7e
materialization, it changes *only* the root: node counts `[0]/[1]/[2]` are unchanged, the node arrays
are byte-identical and merely relocated, so all three mask sidecars stay valid and correctly indexed.
That the seam holds under both implementations suggests it is a real boundary rather than an artifact
of the current code.

### 3c. The escalation test looks nearly free

Phase 2 can only do anything if the bbox lattice contains a cell with no root tile in it. Each root
tile is exactly one 4096^3 cell, so `rootTileCount == P·Q·R` means every cell is a wall, the flood has
nowhere to run, and the result is provably all-zero. That is host-side integer arithmetic, no kernels.
Real meshes land there essentially always — a bbox under 4096 gives `P=Q=R=1`, one cell, occupied.

*(Not yet verified: whether the root tile count is conveniently reachable at that point.)*

## 4. Observations motivating a refactoring

### 4a. One template, four hand-written instances

`MeshToSDF_PipelinePlan.md` §7a describes a **uniform template at every level**: seed evidence, gate,
flood. The code does not share it. `LeafFaceSeedFunctor`, `LowerFaceSeedFunctor` and
`RootFaceSeedFunctor<LEVEL>` each independently open-code the same six-face `switch`, the same
`off[6][3]` offset table and the same `sInt`/`sExt` shared reduction, at three different `LOG2DIM`s.
`CoarseInvertFloodFunctor<LEVEL>` *is* parameterized over the level; `RootInteriorFloodFunctor` is a
separate dense-array implementation of the same flood.

### 4b. Three floods, three synchronization disciplines

| stage | model | mechanism |
|---|---|---|
| in-leaf fill | **pull** | Jacobi, own-slot writes, no atomics, barrier-separated |
| coarse / root face seeding | **push** | scattered `setOnAtomic` into another node's evidence masks |
| root cell flood | **pull**, chaotic | writes immediately, no read/write barrier, relying on monotonicity |

The push at face-seeding is *forced* — those writes land in other nodes' buffers, so they must be
scattered and atomic. But the leaf flood and the root flood are both pull over a private array and
differ only in whether they bother with the barrier separation. That is an inconsistency of
discipline rather than of necessity.

### 4c. Two mechanisms where the design describes one

Within a node the fill is **seed-and-flood**; across nodes and levels it is **evidence accumulation
plus a unanimity gate**. §7a presents both as the same template. Whether they should be unified, or
whether the difference is essential, is unresolved.

### 4d. Allocation and lifetime boilerplate

Every buffer is `DeviceBuffer::create(..., nullptr, false)` plus a `cudaMemsetAsync`, and three
separate places end with a `cudaStreamSynchronize` whose only purpose is to stop scope-exit frees
from outrunning kernels — each with a comment saying exactly that. This is the same null-stream
pattern the `connected-components` branch already migrated away from with `cuda::Buffer`, which
retains its allocation stream and frees on it.

The temporaries are also large and short-lived: `fillCoarseInvertMasks()` holds four evidence buffers
totalling `2 x (lowerBytes + upperBytes)` at once, and `fillRootInteriorMask()` holds three more.

### 4e. The root level's asymmetry is visible in the output

Three levels are node-indexed masks; the fourth is a dense coordinate-indexed array that must carry
its own origin and dims. `bakeBlindData()` ships `leaf_invert` / `lower_invert` / `upper_invert`
uniformly through one `addMask` lambda, then needs a special case **plus an extra `root_extent`
channel** to make the root sidecar interpretable — with the comment "unlike the masks above it is not
indexed by a node". Materializing root tiles (§7e) would remove this asymmetry; the query-time
sidecar preserves it.

### 4f. The three entry points are not independently meaningful

They form a pipeline with a strict ordering dependency (§2) that is not expressed in the API. A
single entry point would make the dependency structural rather than conventional.

## 5. What the refactoring must not break

- `signedSignAt()` is the sole consumer contract; its five-way descent must keep working.
- Per-surface *and* composed use: both `signSurface()` and `fillOnOriginal()` call the same three
  methods, the latter with an external sign array (`d_sign` overriding the member).
- `bakeBlindData()` serializes all four sidecars as blind-data channels.
- The per-surface invert masks are **discarded** after inclusion probing (`PIPELINE.md` §4) — they
  exist only so a surface's field can be queried off its own band.

## 6. Where the discussion stopped

Covered so far: the bootstrap chain and the leaf-level mechanism in detail (§2), the two-phase
framing and the verification that deferral is sound and exact (§3), and the structural observations
(§4). No refactoring has been proposed yet.

## 7. Open items

1. **What the refactored API actually looks like.** Not yet discussed. §3b establishes only that a
   phase-1 / phase-2 seam is *available*, not what should sit on either side of it.
2. **Whether to unify the two mechanisms** of §4c, or treat the within-node and cross-node cases as
   genuinely different.
3. **Whether phase 2 should become §7e materialization** rather than the shipped query-time sidecar.
   This is the decision that determines whether §4e's asymmetry disappears or is permanent.
4. **Whether the escalation test of §3c is reachable** where it would be needed.
5. **Whether `cuda::Buffer` adoption belongs in this refactoring** or is a separate migration, given
   that the branch does not yet carry that paradigm (see `project_cc_status` — the merge is deferred
   until PR #2261 clears review).
