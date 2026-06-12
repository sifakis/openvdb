# Mesh → Signed Distance Field: overall pipeline plan

High-level design notes for the GPU mesh-to-**SDF** pipeline on NanoVDB index grids.
This is the umbrella plan; the connected-components stage (step 3 below) has its own
detailed notes in [`MeshToSDFDevelopmentPlan.md`](./MeshToSDFDevelopmentPlan.md).

> **Where we are (2026-06-12).** Steps 2 and 3 are implemented and validated. Everything
> else on this page is *planned / under design* — captured here from a design discussion so
> the reasoning isn't lost. The cross-level "flood fill" (how the invert masks actually get
> populated across the tree) is explicitly **still to be designed**; see the open items at the end.

---

## 1. The pipeline at a glance

| # | Step | Status | Where | Notes |
|---|------|--------|-------|-------|
| 1 | **Compute UDF + triangle-index sidecar** | TODO | `MeshToGrid.cuh` (extend) | Rasterize mesh → narrow-band `ValueOnIndex` grid + **two** sidecars: the UDF (float per active voxel) and the **nearest-triangle index** (uint32 per active voxel). |
| 2 | **Prune barrier voxels → derived topology** | ✅ done | `computeDerivedTopology` (example) | Drop voxels with `udf² < 0.75·voxelSize²` (the surface shell), rebuild a clean `ValueOnIndex` grid via `PruneGrid`. |
| 3 | **Connected components on the pruned topology** | ✅ done | `ConnectedComponents.cuh` | Per-leaf CC → cross-leaf edges → global union-find. Output: `deviceComponentParent[]`. Detailed in the companion doc. |
| 4 | **Sign the non-barrier voxels** | TODO | new | Identify the exterior CC (it contains the global min-x — or min-y/-z — active voxel) → `+`. All other CCs → `−`. |
| 5 | **Sign the barrier voxels** | TODO | new | Replicate OpenVDB's `ComputeIntersectingVoxelSign`: use the triangle-index sidecar + already-signed neighbors to decide each barrier voxel's side. |
| 6 | **Narrow band → "proper" (gap-free) SDF** | TODO (design) | new | Extend the level-set *representation* so inactive voxels/tiles at every tree level carry a sign. See "Level-set representation extension" below. Then a cross-level fill populates it. |

The **current task at hand** remains narrow: compute the discrete connected components of a
`ValueOnIndex` grid (step 3). Steps 4–6 are the planned trajectory beyond that.

---

## 2. What OpenVDB's `MeshToVolume` does (the reference we're mirroring)

Source: `openvdb/openvdb/tools/MeshToVolume.h` and `tools/SignedFloodFill.h`. Verified against
the code on 2026-06-12.

### 2a. Step 1 is a *purely narrow-band* UDF

`VoxelizePolygons` (the `convert()` voxelization pass) walks each triangle's voxel footprint and,
via `updateDistance()`, keeps the minimum **squared** distance per voxel:

```cpp
// VoxelizePolygons::updateDistance (MeshToVolume.h ~2195)
const ValueType dist = (voxelCenter - closestPointOnTriangleToPoint(...)).lengthSqr();
if (dist < oldDist) { data.distAcc.setValue(ijk, dist); data.indexAcc.setValue(ijk, prim.index); }
...
return !(dist > 0.75); // voxel is a surface/barrier voxel iff d² ≤ 0.75
```

The dist tree's background is `std::numeric_limits<float>::max()`. **Only** voxels within the
band of some triangle are ever activated. The deep interior of a large closed solid is **not**
present after this step — it is just inactive `+inf` background. So the output is a sparse shell
hugging both sides of the surface, all carrying **positive** (squared) distances; inside/outside
is not yet distinguished.

### 2b. The index grid (companion to the UDF)

Produced *simultaneously* with the UDF, co-topological with it: an `Int32` grid storing, per
active voxel, the **index of the nearest triangle** — whichever primitive won the distance
`atomicMin`/reduction (`data.indexAcc.setValue(ijk, prim.index)`). Background is
`util::INVALID_IDX`. This index grid is what the barrier-voxel sign step (2d) consumes: knowing
the closest triangle lets it compute the closest surface point and the surface normal direction.

### 2c. Sign convention

From `SignedFloodFill.h` (the canonical statement): **outside = `+background`, inside =
`−background`.** (`MeshToVolume.h` line ~3238 notes that `evaluateInteriorTest` uses the *reverse*
convention internally at one intermediate point, but the final emitted field is standard:
`+` exterior, `−` interior.)

### 2d. Signing — two sub-problems

1. **Non-barrier voxels** (`d² > 0.75`): signed by connectivity. `traceExteriorBoundaries`
   sweeps from outside inward; whatever the exterior sweep can't reach is interior (`−`).
   Multiple interior regions (cavities, nested shells) all get `−`.
2. **Barrier voxels** (`0 ≤ d² ≤ 0.75`, the shell straddling the surface): signed by
   `ComputeIntersectingVoxelSign` (MeshToVolume.h ~1380). For each barrier voxel `q`:
   - **Pass 1** — scan the 3×3×3 in-leaf neighborhood. For each already-`−` neighbor `n`:
     - `cp` = closest point on `n`'s nearest triangle (from the index grid);
     - `dir_n = normalize(n − cp)`, `dir_q = normalize(q − cp)`;
     - if `dir_n · dir_q > 0` (q is on the same side of the triangle as a known-interior
       neighbor) → flip `q` to `−`. Stop.
   - **Pass 2** — if pass 1 found nothing, repeat the same same-side test over the extended
     26-neighborhood that reaches *outside* the leaf.

   Heuristic in one line: **a barrier voxel is interior iff it lies on the same side of its
   nearest triangle as a known-interior neighbor.** This depends on the non-barrier voxels
   already being signed — hence the ordering (step 4 before step 5).

### 2e. `signedFloodFill` — sign propagation **and** interior tile creation

`signedFloodFillWithValues` (called at MeshToVolume.h ~3512) does the level-by-level fill that
turns the signed band into a level set:
- **Leaf pass** — z-scanline through the *dense* 8³ buffer; every **inactive** voxel slot is set
  to `±background` according to the sign of the last active voxel seen on the scanline.
- **Internal-node passes** — same scanline logic over child slots, propagating sign into inactive
  tiles.
- **Root pass** — a z-scanline over the root's sorted children inserts **inactive interior tiles**
  (`root.addTile(c, mInside, false)`) wherever two same-(x,y) children sandwich interior space.
  These "root tiles" (entries in the root node's table, as opposed to child nodes) are what cover
  the deep interior of a big solid. They carry the *constant* `−interiorWidth`, not accurate
  distances.

**Result:** every point in space resolves to a meaningful sign — active band voxels carry accurate
signed distances; everything else falls into an inactive tile (`−interiorWidth` interior or
`+exteriorWidth` exterior). The SDF is only *accurate* within the band; outside it the value is the
clamped band-width constant. That is what makes it a valid (if not Eikonal-everywhere) level set.

### 2f. The crucial simplification

**`UDF` is exactly `|SDF|`.** The entire signing problem is therefore a per-location *binary*
decision: `+` or `−`. Nothing after UDF rasterization changes magnitudes (modulo the optional
renormalization post-process) — it only assigns signs and extends coverage.

---

## 3. Our CUDA approach: connected components instead of a directional flood-fill

Steps 2–3 above replace OpenVDB's `traceExteriorBoundaries` sweep with explicit CC labeling of the
pruned (barrier-removed) topology. CC is strictly more informative than a one-directional
flood-fill: it labels *all* components at once rather than propagating a single exterior sign
inward, which is exactly the robustness win we want for thin features.

### Separation of concerns (decided)

Signing the band and extending the topology are **independent** problems and will be done
separately:

- **Sign the narrow band (steps 4–5)** — operates only on voxels that already exist. No new
  topology needed.
- **Extend the topology inward (step 6)** — purely additive coverage so off-band queries return a
  meaningful value. Does not change any band voxel's sign.

### Step 4 — signing the non-barrier voxels (planned)

- Run CC (step 3) on the pruned topology.
- The CC containing the voxel with the **globally minimum x** (or y, or z) coordinate *must* be the
  exterior component — nothing lies further out in that axis than the mesh's own extent. Sign that
  whole component `+`.
- Every other component → `−` (interior shells, cavities). This matches OpenVDB's treatment:
  anything the exterior can't reach is interior.

### Step 5 — signing the barrier voxels (planned)

Replicate `ComputeIntersectingVoxelSign` (2d above) on the GPU, consuming the triangle-index
sidecar from step 1 and the now-signed non-barrier neighbors from step 4. This is the next *major*
algorithmic component and the goal is to mirror OpenVDB's heuristic faithfully.

---

## 4. Step 1 detail — the triangle-index sidecar (planned `MeshToGrid` extension)

Today `MeshToGrid::getHandleAndUDF` returns the grid handle + a single float UDF sidecar. The UDF
kernel (`util::rasterization::cuda::ComputeUDFFunctor`, in `util/cuda/Rasterization.cuh`) keeps the
minimum distance via `atomicMin` on the bit-cast float — it does **not** record which triangle won.

Plan: a new entry point (working name `computeUDFsidecarAndIndex` / `getHandleAndUDFAndIndex`) that
returns **topology + 2 sidecars**: the float UDF and a `uint32_t` nearest-triangle index per active
voxel.

**Consistency requirement & approach.** The distance and the triangle index must update together —
no race where the kept distance came from triangle A but the index slot ends up B. Replace the
float `atomicMin` with a **64-bit CAS loop on a packed `uint64_t`**:

```text
packed = (uint64_t(__float_as_uint(distSqr)) << 32) | triangleID   // dist in high bits, comparable as uint
atomicMin-style CAS that compares the high 32 bits, swapping the whole 64-bit word on a win
```

The high-bits-are-distance layout makes an unsigned `min` on the 64-bit word do the right thing, so
distance and index stay a consistent pair. After the kernel, split the packed buffer into the two
caller-visible sidecars. No structural obstacle — straightforward addition.

---

## 5. Level-set representation extension on `ValueOnIndex` grids (the step-6 enabler)

### 5a. The problem

In a **NanoVDB FloatGrid** (or OpenVDB float tree) every leaf has a *dense* 8³ float buffer: all
512 slots are backed by memory whether active or not. `signedFloodFill` can freely write
`±background` into inactive slots. **No problem there.**

A **`ValueOnIndex` grid** allocates sidecar slots for **active voxels only**. Inactive voxels have
*no backing storage anywhere* — the leaf stores only an active mask + base index. A boundary leaf
(intersects the band but isn't fully inside it) therefore cannot represent the sign of its inactive
interior voxels: querying one returns the (unsigned, `+`) background, which is wrong. This is a
**structural limitation of the sparse index representation**, not present in dense float grids.

The same gap exists at **every** level: deep-interior space collapses to inactive tiles at the
lower-internal, upper-internal, or root level, and none of those carry a sign in a plain index grid.

### 5b. The solution — per-level `invertMask` sidecars

Extend the semantics of a level set stored on a `ValueOnIndex` grid with an **`invertMask`**: a set
of bitmask sidecars, one per tree level, parallel to the node arrays.

| Level | Node | `LOG2DIM` | Child slots | Mask type | Sidecar length |
|-------|------|-----------|-------------|-----------|----------------|
| 0 | Leaf | 3 | 8³ = 512 | `Mask<3>` | `nodeCount[0]` |
| 1 | Lower internal | 4 | 16³ = 4096 | `Mask<4>` | `nodeCount[1]` |
| 2 | Upper internal | 5 | 32³ = 32768 | `Mask<5>` | `nodeCount[2]` |
| (root) | Root | — | sparse hash map | *(implicit — see 5e)* | — |

So `Mask<LOG2DIM>` is exactly "one bit per child slot" at each fixed-fanout level — a structural
parallel to the value/child masks already inside the nodes. The value sidecar is unchanged (length
`valueCount()`, including slot 0 = background).

### 5c. Semantics

The invert bit modifies the value of a **terminal** slot (one with no real stored value and no
finer representation below it):

```text
effective_value(terminal slot) = sidecar[0] * (invertMask_bit ? -1 : +1)
```

With slot 0 holding the saturated background `+3h` (band width 3, voxel size `h`):
- bit **clear** → `+3h` (exterior) — and this is the **all-zeros default**, which is correct for
  the overwhelmingly common exterior, so a freshly zeroed mask is already right and you only set
  bits for interior regions;
- bit **set** → `−3h` (interior).

This recovers exactly the information `signedFloodFill` writes into a dense buffer, at **512 bits
per leaf** instead of 512 floats (and analogously at the internal levels).

### 5d. Which slots may carry an invert bit

The invert bit is legal **only on terminal slots** — those with no finer representation and no real
stored value. Setting it elsewhere is **undefined behavior** (our tools will never do it).

- **Leaf:** only on **inactive** voxels (active voxels read their true sidecar value).
- **Lower / upper internal:** only on **childless** slots (tiles). A slot holding a child pointer
  has its sign carried by the descendants; the bit there is meaningless.

The governing predicate at internal levels is *childless / no-child* (not merely "inactive"),
because that's what determines whether a real value could exist below. For our level-set use the
deep-interior tiles are childless-and-inactive, so the two notions coincide — but the spec should
state *childless*. Every terminal in the tree is covered by exactly one invert bit at exactly one
level: no double-counting, no gaps.

### 5e. Why `invertMask` (relative) and **not** `signMask` (absolute) — the core reason

The index grid is the **topology** half of a level set; the sidecar is the **value** half. The
invert masks are deliberately classified as **topology**.

We require this invariant: **negating the field is a pure value-side operation.** Geometrically,
`φ(x) → −φ(x)` is the complement of the solid (flip it inside-out). We want that to be achievable by
negating every sidecar value (including slot 0) with the **topology left bit-for-bit unchanged**.

The invert mask is *multiplicative* (`value = sidecar[0] * (±1)`), so this falls out automatically.
Negate the sidecar (`+3h → −3h`) and every terminal flips:
- bit clear: `+3h → −3h`
- bit set:  `−3h → +3h`

Active voxels flip too (their sidecar entries negate directly). The masks — being topology — never
move. A `signMask` storing an absolute sign and reading `±|sidecar[0]|` would break this:
`|−3h| = |+3h|`, so negating the sidecar would do nothing to inactive voxels, and complementing the
solid would force flipping every mask bit — i.e. mutating the topology. `invert` is correct
precisely because it makes negation **commute through the topology untouched**: sign lives in the
values; the mask records only *relative* orientation.

### 5f. Root-level convention (asymmetric, by design)

The root table is a **sparse hash map**, so unlike the fixed-size lower nodes it needn't represent
anything explicitly. We let **absence encode exterior**:

| Root-table state | Meaning |
|------------------|---------|
| No entry at this key | Exterior — falls through to root background `+3h`. |
| Tile **with** children | Intersects the narrow band (real values live in descendants). |
| **Childless** tile | Deep interior — **implied `invertMask = 1`** → `−3h`. |

Because exterior is represented by *not storing an entry*, the only childless root tiles that ever
exist are interior ones — so childlessness + existence carries the sign implicitly and **no explicit
root-level invertMask sidecar is needed**.

Caveat (recorded for completeness): this bakes the exterior↔interior asymmetry into the *topology
convention* at the root, so the clean "negate the sidecar to complement the field" invariant of
5e holds for the three fixed-fanout levels but **not** at the root — complementing a whole grid
would additionally require materializing the previously-absent exterior as childless root tiles and
dropping the interior ones (a root-level topology operation). That's the inherent price of
exploiting sparsity for the unbounded exterior, and it only bites at the root.

---

## 6. Open items / still to design

- **Cross-level "flood fill" — how the invert masks actually get populated.** The representation
  (section 5) is the *container*; the algorithm that fills it across leaf → lower → upper → root
  (the CC/flood-fill analogue that decides which terminal slots are interior) is **not yet
  designed**. This is the next design conversation.
- **Boundary-leaf inactive voxels.** Even within a single band-intersecting leaf, inactive interior
  voxels need their leaf `Mask<3>` invert bits set (section 5a). How this dovetails with the
  cross-level fill is part of the above.
- **Renormalization / Eikonal accuracy.** OpenVDB's output is only accurate within the band
  (clamped constants outside). If a *true* Euclidean SDF everywhere is wanted, a fast-marching /
  redistancing pass is a separate, later concern.
- **Step-5 GPU port specifics** — data layout for the 26-neighbour same-side test across leaf
  boundaries, and reuse of the cross-leaf neighbour-lookup machinery already built for CC
  (`ccNeighborLeafIndex` via `root().probeLeaf(origin.offsetBy(±8))`).
