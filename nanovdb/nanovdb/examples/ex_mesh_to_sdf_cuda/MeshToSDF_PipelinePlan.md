# Mesh → Signed Distance Field: overall pipeline plan

High-level design notes for the GPU mesh-to-**SDF** pipeline on NanoVDB index grids.
This is the umbrella plan; the connected-components stage (step 3 below) has its own
detailed notes in [`MeshToSDFDevelopmentPlan.md`](./MeshToSDFDevelopmentPlan.md).

> **Where we are (2026-07-14).** Steps 1–6 are **all implemented** and validated. The narrow band
> is signed (steps 4–5) and the sign is extended to all of space via per-level invert-mask sidecars,
> filled bottom-up leaf → lower → upper → root (step 6); a single `signedSignAt(x)` query composes
> the full level set, sidecar-only (grid topology is never rebuilt). The example driver is organized
> into three passes over an opaque `SdfPipeline`: **`buildMeshToSdf`** (run steps 1–6),
> **`validateMeshToSdf`** (independent CPU oracles + OpenVDB / analytic cross-checks), and
> **`exportMeshToSdf`** (Polyscope visualization dump, viewed with `mesh_to_sdf_viewer.py`).
> The design notes on this page match what shipped.
>
> **Known limitation.** Step 4's exterior rule is *single-seed* (the min-axis component is the only
> region called exterior), so it mislabels **separated objects** and **nested / hollow** shapes
> (reproduced by the `--two-spheres` probe). A more robust replacement is under design.

---

## 1. The pipeline at a glance

| # | Step | Status | Where | Notes |
|---|------|--------|-------|-------|
| 1 | **Compute UDF + triangle-index sidecar** | ✅ done | `MeshToGrid.cuh` (`getHandleAndUDFAndIndex`) | Rasterize mesh → narrow-band `ValueOnIndex` grid + **two** sidecars: the UDF (float per active voxel) and the **nearest-triangle index** (uint32 per active voxel), written together via a packed-64 `atomicMin`. |
| 2 | **Prune barrier voxels → derived topology** | ✅ done | `computeDerivedTopology` (`MeshToSDF.cuh`) | Drop voxels with `udf² < 0.75·voxelSize²` (the surface shell), rebuild a clean `ValueOnIndex` grid via `PruneGrid`. |
| 3 | **Connected components on the pruned topology** | ✅ done | `ConnectedComponents.cuh` | Per-leaf CC → cross-leaf edges → global union-find. Output: `deviceComponentParent[]`. Detailed in the companion doc. |
| 4 | **Sign the non-barrier voxels** | ✅ done | `MeshToSDF::signNonBarrier` | Exterior CC = the one holding the global min-x active voxel → `+`; all other CCs → `−`. **Single-seed** — see the limitation note above. |
| 5 | **Sign the barrier voxels** | ✅ done | `MeshToSDF::signBarrier` | Mirrors OpenVDB's `ComputeIntersectingVoxelSign` (double precision): each barrier voxel takes the side it shares with an already-signed neighbor via that neighbor's nearest triangle. Reads a sign snapshot → order-independent. |
| 6 | **Complete the level set (gap-free fill)** | ✅ done | `MeshToSDF::fill{Leaf,Coarse}InvertMask`, `fillRootInteriorMask` | Per-level invert-mask sidecars carry the sign into inactive voxels/tiles; filled bottom-up leaf → lower → upper → root. Sidecar-only — grid topology never rebuilt. `signedSignAt` composes the query. |

**Status:** the full steps-1→6 pipeline is implemented and validated; current work is code cleanup
and a more robust step-4 rule (the single-seed limitation above).

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

## 6. Populating the invert masks — the leaf level

This is the "fill" algorithm: given a signed narrow band (steps 4–5), decide which **inactive**
voxels are interior and set their leaf `Mask<3>` invert bits. Designed 2026-06-16.

### 6a. Seeds, walls, and the in-leaf fill

After CC, partition the full narrow band into three sets (see also step 4):

1. **Barrier voxels** (`d² < 0.75·h²`) — the surface shell, excluded from CC.
2. **Exterior component** — the non-barrier CC containing the global min-on-any-axis voxel → `+`.
3. **Deep interior** — *every* other non-barrier active voxel → `−`, however many components it
   forms (cavities, nested shells, disconnected pockets all count). This is exactly
   `narrowband − barrier − exteriorCC`.

Set (3) voxels are the interior **active** voxels. They already carry real signed values in the
value sidecar (negated UDF), so they are **not** themselves represented by the invert mask. Their
role is to be **seeds** for an in-leaf flood fill of the *inactive* voxels:

- **Seeds (injectors):** each set-(3) voxel sets `invertMask = ON` on each of its inactive
  face-neighbors.
- **Propagation:** the fill then runs purely **inactive → inactive** over 6-adjacency — any inactive
  voxel adjacent to an already-ON inactive voxel is set ON too. Iterate to convergence (a few sweeps
  in an 8³ leaf).
- **Walls:** active voxels are never propagated *through*; only set-(3) voxels act as initial
  injectors. So barrier voxels **and** exterior voxels automatically bound the fill, because the
  propagation graph is the inactive voxels alone. The (active) barrier shell is the separator that
  keeps the fill from leaking to the exterior-side inactive voxels.

We **cannot** blanket-set every inactive voxel in a band-intersecting leaf, because such a leaf
generally contains inactive voxels on *both* sides of the surface; the fill distinguishes them by
staying on the interior side of the barrier shell. Its only failure mode is a non-watertight shell
*within* the leaf (thin feature / discrete gap) — the same robustness caveat that already governs CC.

### 6b. No leaf↔leaf propagation is needed (self-sufficiency)

**Claim.** An `invertMask=ON` voxel on a leaf face never needs to reach across the boundary to set
the bit on a same-resolution neighbor voxel in an adjacent **pre-existing** leaf. Each materialized
leaf labels its own inactive interior voxels entirely from its own seeds.

**Why.** A fully-refined leaf exists **iff** it contains ≥1 active voxel, and active voxels exist
only where the UDF band reached. The relevant guarantee is *conditional* and comes from the
**distance field**, not from CC:

> If a leaf contains an interior *inactive* voxel, then it also contains an interior *active* seed.

This follows from the distance field being **monotone away from the surface**, so its shells are
nested: between a deep-interior inactive voxel (`d² > bw²`) and the surface you must cross the
non-barrier interior band (`0.75 ≤ d² ≤ bw²`, the seed shell, ~2 voxels thick at `bandWidth = 3`)
before reaching the barrier shell (`d² < 0.75`). The voxel `W` a boundary ON-voxel would want to
reach is itself interior (it is the face-neighbor of an interior voxel, and there is no surface
between two adjacent inactive voxels). So `W`'s leaf is, by assumption, in the "contains an interior
inactive voxel" case, and therefore holds its own seed and self-labels.

**Important corrections to keep the reasoning honest:**

- It is **not** true that every materialized leaf contains an interior voxel. An **exterior-only**
  leaf (outer shell + barrier, e.g. outside a convex surface) is perfectly legitimate and common —
  it is **not** a CC failure. We only ever invoke the contrapositive: an exterior-only leaf has no
  interior inactive voxels, hence nothing to propagate.
- The load-bearing fact is **nested distance shells**, not CC correctness.

### 6c. Robustness — holds for arbitrarily nasty input

The pipeline (UDF → barrier mark `d² < 0.75` → CC → label min-axis CC exterior, all other
non-barrier voxels interior) gives the self-sufficiency property **regardless of how
self-intersecting, overlapping, or disconnected** the input triangle soup is:

- **Always, unconditionally:** the labeling is well-defined and self-consistent, and the
  min-on-any-axis voxel is *provably* in the true exterior (nothing lies further out on that axis),
  so the exterior CC is correctly seeded every time.
- **Closed surfaces:** the `√0.75 = √3/2` barrier threshold (half the voxel space-diagonal) makes
  the barrier shell **watertight under 6-connectivity**, so the shell separates inside from outside
  and the labeling matches the intended solid — *including* self-intersecting, overlapping, nested,
  and disconnected closed geometry. (These are exactly the cases where CC beats a directional
  flood-fill, since interior walls don't trap a sweep.)
- **Open / non-watertight surfaces:** the exterior CC simply **leaks through the gap and wraps
  around**, so there are *fewer* interior voxels — in the limit, none. Inside/outside is
  geometrically ill-posed for an open surface (the same limitation OpenVDB has); it is not a CC
  failure.

The key consequence for 6b: degrading watertightness only ever **shrinks** the interior set; it can
never manufacture the one thing that would force cross-leaf propagation — an interior inactive voxel
sitting in a materialized leaf with no interior seed of its own. Watertightness failures *remove*
interior; they never *strand* it. So leaf↔leaf propagation is unnecessary in **all** cases, and the
only propagation that does real work is cross-**level** (into coarser tiles / unrefined space).

### 6d. Leaf → coarser-neighbor propagation (seeding the coarse levels)

Since leaf↔leaf propagation is never needed (6b), the only outward work a leaf does is **seed
coarser neighbors**. Applied per leaf, per face, after the in-leaf fill (6a) converges:

A face's neighbor is a **coarser neighbor** when the space across it is not materialized at voxel
resolution but as a **childless tile** one or more levels up — a child slot of a lower node
(`Mask<4>`), of an upper node (`Mask<5>`), or a root tile. (A lower-node child slot covers an
8³ region — exactly leaf-sized; coarser slots cover more.)

The rules:

1. **Childless ⟹ surface-free ⟹ uniform sign.** A childless tile was never refined, so *every*
   one of its voxels has `d > bandWidth`; the surface is `> bandWidth` from the whole tile. The tile
   is therefore uniformly interior or uniformly exterior, and a **single** interior contact on
   **any** face classifies the entire tile. (No need to inspect its interior or reconcile faces.)

2. **Refined neighbor ⟹ skip.** If the neighbor across the face is a materialized leaf, do nothing —
   it self-labels (6b).

3. **Barrier on the face ⟹ the neighbor is refined, so skip.** A barrier voxel (`d ≤ √0.75 ≈ 0.87`)
   forces the voxel one step across to have `d ≤ 0.87 + 1 = 1.87 < bandWidth`, i.e. **active**, i.e.
   refined. So a face bearing *any* barrier voxel never drives coarse propagation. (This is the
   contrapositive of: a childless neighbor forces all shared-face voxels to `d > bandWidth − 1 = 2`,
   comfortably non-barrier.)

4. **Trigger — interior-signed, unanimously.** Propagate `ON` to a childless neighbor **iff** the
   abutting face has ≥1 **interior-signed** voxel and **0 exterior-signed** voxels (barrier voxels
   ignored), where:
   - *interior-signed* = active set-3 (deep interior) **or** inactive `invertMask = ON`;
   - *exterior-signed* = active exterior-CC **or** inactive `invertMask = OFF` (the default).

   Read "inactive" in the trigger strictly as **inactive-and-inverted (`ON`)** — a plain inactive
   `OFF` voxel is *exterior* and must not trigger. Setting the bit means setting the one bit indexed
   by that child slot in the neighbor node's invert-mask sidecar (`Mask<4>`/`Mask<5>`; implicit at
   the root, per 5f).

**Why the unanimity clause, and why the "mixed face" fear is unfounded for valid input.** One might
worry: if a face has *both* interior and exterior voxels, do we risk marking an exterior tile
interior? For a genuine childless neighbor this **cannot happen**: rule 1 puts the surface
`> bandWidth` from the tile, so by Lipschitz the shared face voxels all have `d > bandWidth − 1 = 2`
— non-barrier and unable to straddle the surface (a straddle needs a near-zero-`d` point between two
voxels). Hence the abutting face is **uniform in sign**, and "has interior" ≡ "has no exterior". The
unanimity clause therefore only ever bites on **non-watertight / band-thin input**, where it errs
toward the safe `OFF`/exterior default — the worst it can do is fail to mark a true-interior tile,
never falsely inflate the solid.

**Free consistency check.** A face that has *both* a barrier voxel and a childless neighbor violates
the invariant (rules 1+3) — a reliable tell that the input was non-watertight / band-thin there.

**Seeding ≠ the whole fill.** These rules only light up the coarse tiles a leaf *directly abuts*. A
large interior cavity may be many coarse tiles thick, with core tiles touching no leaf at all. So
after leaf-face seeding, the fill must continue **upward, level by level** — §7.

---

## 7. The cross-level fill — bottom-up, leaf → lower → upper → root

Section 6 settles the leaf level. The same template repeats at each coarser level, terminating in a
small dense flood fill over a provisional root-tile array. Designed 2026-06-16.

### 7a. The uniform template at every level

The fill is a **bottom-up sweep**: process all leaves, then all lower nodes, then all upper nodes,
then the root array. At each level the work is identical in spirit to §6:

1. **Within-node fill.** Flood `invertMask = ON` among the node's **childless** child slots (the
   ones with an invert bit), over 6-adjacency, using **refined** child slots (childMask ON) as
   **walls** — exactly as active voxels walled the in-leaf fill. Seeds are the slots already marked
   `ON` by the level below's face propagation.
2. **Face → coarser propagation.** For each of the node's 6 faces, for slots carrying an interior
   `ON` tile, push `ON` across to the *coarser* neighbor via the probe cascade (7b).

**Same-level propagation is unnecessary at every level** — the §6b self-sufficiency proof recurs:

> A node at level L is materialized **iff** the band passes through it (it has ≥1 refined child).
> A region of pure interior with no band is *never* a materialized level-L node — it collapses to a
> childless tile one or more levels coarser. So any materialized node holding interior tiles also
> holds interior-facing refined children, which seed its interior region; the within-node fill then
> completes it.

Hence leaf↔leaf, lower↔lower, upper↔upper propagation are all skipped. (Cross-node *seeding* still
happens implicitly: a face's coarser-neighbor probe can land in an adjacent parent node, so a leaf
in one lower node can mark a tile in the next — that is part of the face→coarser pass, not a
sideways pass.) Same robustness caveat throughout: an interior region disconnected *within* one node
and joined only through a neighbor is the thin-feature edge case.

### 7b. The probe cascade (per face, after the interior-unanimous gate of §6d.4)

Starting one level below wherever we are and walking up, the neighbor across the face resolves to
exactly one terminal, and we act on it:

| Probe result at `neighborCoord` | Meaning | Action |
|---|---|---|
| same-level node exists (refined) | neighbor self-labels (§6b/7a) | **skip** |
| childless slot in a lower node | 8³ tile | set its `Mask<4>` bit `ON` |
| childless slot in an upper node | 128³ tile | set its `Mask<5>` bit `ON` |
| no upper node (`probeUpper == null`) | root-level space | mark the provisional root cell (7c) |

For a **leaf** the cascade is `probeLower → probeUpper → root`; for a **lower** node it is
`probeLower (skip if a sibling/adjacent lower exists) → probeUpper → root`; for an **upper** node it
is `probeUpper (skip) → root`. Tightening (verified in discussion): given `probeUpper != null` and
the lower probe null, the upper child slot is **guaranteed** childless (a lower child would have been
found by the lower probe), so the `Mask<5>` write is unconditional. The "wall" on a face is always a
**refined** (childMask ON) neighbor; a **barrier voxel** on a leaf face is a sufficient witness that
the neighbor is refined (`d ≤ √0.75` ⟹ the voxel one step across has `d < bandWidth` ⟹ active ⟹
refined), so such faces never drive coarse propagation.

### 7c. The provisional root-tile array

A childless **root tile** has the same 4096³ extent as an upper node, so when a face's neighbor has
no upper node (`probeUpper == null`) the neighbor lives at root granularity. Rather than insert into
the root hash map during the parallel passes (a delicate concurrent mutation), we **defer**:

- **Pre-size a dense array** once, up front, from the UDF band's index-space bbox quantized **up to
  4096-aligned root tiles**:
  - `tileMin = (⌊minX/4096⌋, ⌊minY/4096⌋, ⌊minZ/4096⌋)` (floored — mind negatives),
  - `tileMax = (⌊maxX/4096⌋, …)`, `(P,Q,R) = tileMax − tileMin + 1`,
  - cell `(i,j,k)` ↔ root tile `tileMin + (i,j,k)`.
  Example: band bbox `[-10,-20,-30]→[400,500,600]` ⟹ tiles `[-1,0]³` ⟹ a `2×2×2` array.
  Root tiles are gigantic, so this is **a handful of cells** even for huge models (a 10k³ bbox is
  `3³ = 27`); brute force is fine.
- **Marking** (during 7b's "root" action) is an **idempotent set** of cell
  `(⌊coord/4096⌋ − tileMin)` to `ON` — a plain `atomicOr`/race-to-1 on a `bool`/`uint8` cell, no
  hashing, no insertion.
- **Validity / no conflict:** `probeUpper == null` means the entire 4096³ has *no* band, so it is
  uniform; and it is mutually exclusive with the band-intersecting case (which *has* an upper node).
  The three end states — refined (has upper child) / marked-interior / absent-exterior — are
  disjoint.
- **Closure guarantee (no escape):** the neighbor we propagate into is interior, and
  `interior ⊆ object ⊆ inside-the-shell ⊆ band-bbox ⊆ provisional lattice`, so an interior neighbor
  *always* lands inside the array. Only exterior neighbors could fall outside, and we never propagate
  to those.

### 7d. The final dense flood fill on the provisional array

The leaf/lower/upper → root passes only mark cells that directly abut the band. The deep interior of
a *gigantic* solid is many root tiles thick, with core tiles touching nothing — so a last flood fill
on the dense `P×Q×R` array completes it, using the same template:

- **Medium:** provisional cells *not* backed by a pre-existing root entry.
- **Walls:** the **pre-existing root tiles** — root-table entries that carry an upper-node child
  (band-intersecting 4096³ regions). These wall the fill exactly as refined children did below.
- **Seeds:** the cells already marked `ON` in 7b/7c.
- Propagate `ON` among medium cells over 6-adjacency, blocked by walls.

Validity is the top-scale instance of the recurring argument: a cell flips interior↔exterior only
across the surface, and crossing the surface at root-tile granularity passes through a
band-intersecting (hence pre-existing, hence wall) root tile. Exterior cells need no handling — they
stay at the `OFF` default. The array is tiny, so a single GPU block iterating to convergence
suffices.

### 7e. Reconciliation and the (cheap) topology rebuild

After the fill, walk the `P×Q×R` array once:
- `ON` **and** not pre-existing → **materialize a childless root tile** (→ implicitly interior, §5f);
- pre-existing (wall) → leave as-is (keeps its upper child and subtree);
- `OFF` → leave absent (exterior).

The new childless ("dummy") tiles are the *only* topology change. **The entire delta is at the
RootNode.** Specifically:

- **Node counts `[0]/[1]/[2]` are unchanged** — no new leaf/lower/upper nodes were created (interior
  beyond the band is carried by root tiles and invert *bits*, never new interior nodes). So the
  upper/lower/leaf node arrays are **byte-identical** to the UDF tree; in the rebuilt blob they only
  **shift position** below the grown root section — a wholesale `memcpy` of that block plus a fixup
  of the root→upper relative offsets, not a re-derivation.
- **Sidecars carry over untouched:** the value sidecar (`valueCount` unchanged — dummy tiles are
  inactive and store no index; their `−background` magnitude is supplied at query time by the invert
  convention reading slot 0) and the `Mask<3/4/5>` invert sidecars (indexed by the unchanged node
  counts).

So "rebuild the SDF tree" reduces to "rebuild the RootNode, relocate everything below it verbatim."

### 7f. GPU implementation sketch

All device-side (the provisional marks are produced on-device by the propagation kernels; keeping
the rest on-device avoids a sync/transfer round-trip — the fill itself is trivially cheap):

1. **Flood fill** the dense `P×Q×R` array (7d) — a single block to convergence.
2. **Compact** the `ON` & not-pre-existing cells into a small list of new-tile root keys
   (`coordToKey` on `tileMin + (i,j,k)`, à la `PointsToGrid`).
3. **Craft the new root table** by duplicating the philosophy of the `<op>Root()` stage of the
   existing CUDA tools — `MergeGrids` (union of two root tables) and `DilateGrid` (insertion of new
   root-level topology) are the closest analogs. Feed it `{existing child keys} ∪ {new dummy keys}`;
   the sort/dedup is defensive since a marked cell that is also pre-existing was a wall (excluded in
   step 2), so keys never collide. Emit a new grid blob: augmented root + verbatim-relocated
   upper/lower/leaf block + carried-over sidecars.

**Detail to nail down:** the exact tile encoding for a childless interior root tile in a
`ValueOnIndex` grid — it must be inactive (so it doesn't consume a `valueCount` index) yet present,
and a `ValueOnIndex` accessor landing on it must resolve to the interior `−background` via the §5f
convention + the value sidecar's slot 0.

---

## 8. Open items / still to design

- **Renormalization / Eikonal accuracy.** OpenVDB's output is only accurate within the band
  (clamped constants outside). If a *true* Euclidean SDF everywhere is wanted, a fast-marching /
  redistancing pass is a separate, later concern.
- **Step-5 GPU port specifics** — data layout for the 26-neighbour same-side test across leaf
  boundaries, and reuse of the cross-leaf neighbour-lookup machinery already built for CC
  (`ccNeighborLeafIndex` via `root().probeLeaf(origin.offsetBy(±8))`).
