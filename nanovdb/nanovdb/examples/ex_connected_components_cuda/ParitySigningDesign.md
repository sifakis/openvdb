# Parity-based signing — a more robust replacement for step 4

> Design notes for replacing the mesh→SDF **sign-determination stage (step 4)** — currently
> a *min-x single-seed* rule — with **parity-based signing**: color the connected-component
> graph and flip the sign at every surface crossing. Handles **separated objects** and
> **nested / hollow** shapes correctly. **Not yet implemented**; two design points are still open (§13).
>
> Umbrella pipeline: [`MeshToSDF_PipelinePlan.md`](./MeshToSDF_PipelinePlan.md).
> CC internals: [`MeshToSDFDevelopmentPlan.md`](./MeshToSDFDevelopmentPlan.md).

---

## 0. Motivation — why replace step 4

Today's step 4 = *"the CC holding the global min-x active voxel is exterior (`+`); every
other CC is interior (`−`)."*

- ✅ Correct for a **single, simply-connected, watertight** closed surface.
- ❌ **Separated objects** — only the object containing the min-x voxel is called exterior;
  a *second* object's outer shell is mislabeled interior. (Reproduced with `--two-spheres`:
  7808 confident mismatches.)
- ❌ **Nested / hollow** shapes — a hollow cavity is filled as solid. (This particular case
  *matches* OpenVDB's flood-from-outside; it only differs from a winding-number convention.)

**Root cause:** "one min-x component is exterior, everything else is interior" is a **single
seed**. It cannot express the correct inside/outside (parity) once the domain has
disconnection or nesting.

---

## 1. Core idea — graph 2-coloring (parity)

**Crossing the surface once flips the sign** (parity / winding number). Cast as a graph:

- Partition space into pieces (nodes).
- Relate adjacent pieces with edges:
  - **flip edge** — two pieces separated *by the surface* (a barrier between them) →
    **opposite sign**.
  - **keep edge** — two pieces connected *through empty space* (same material, no surface
    between) → **same sign**. *(Also read as "same-material edge".)*
- Seed a **definitely-exterior** piece as `+`, then **BFS, flipping at every flip edge** →
  every piece gets a sign.

This 2-coloring *is* the even-odd / winding sign: a nested cavity comes out `+` (even number
of crossings), and each separated object is signed correctly on its own.

---

## 2. Voxel 3-class (the foundation)

Partition the narrow band into three classes:

| class | definition | role |
|---|---|---|
| **non-barrier active CC** | 6-connected component of active voxels with `udf² ≥ 0.75·h²` | **primary graph node** (the thing we assign a sign to) |
| **barrier** | `udf² < 0.75·h²` (within `√3/2` voxel of the surface) | **medium for flip edges** (the surface itself) |
| **inactive** | not active (empty voxels in a leaf + childless tiles + absent space) | **medium for keep edges** (empty material) + the **fill target** |

- Barrier shell is ~1–2 voxels thick (typically 1 per side of the surface = 2); the exterior
  band is ~2.1 voxels. All connectivity is **6-connectivity**.

---

## 3. Graph definition

**Nodes:**
- **CC** — a non-barrier active component; the primary thing that receives a sign.
- **inactive region** — a connected piece of empty space; doubles as the hub for keep edges
  and as the place the **fill** sign is stored (§8). Includes the exterior region (the empty
  space outside everything).

**Edges:**
- **flip edge** (CC ↔ CC) — two CCs separated by a barrier → opposite sign.
- **keep edge** (CC ↔ region, region ↔ region) — share an inactive region / joined through
  empty space → same sign.

**Seed:** the exterior region = `+` (§7).

**Result:** parity BFS assigns a sign to every CC and every region.

---

## 4. Edge detection — one principle: "label flood + collision"

Both edge types come from the *same* mechanism: **propagate CC labels through a medium; when
two different labels meet (collide), emit an edge.** Only the medium differs:

```
medium = BARRIER   → the two CCs are on opposite sides of the surface  → FLIP edge
medium = INACTIVE  → the two CCs are joined with no surface between     → KEEP edge
```

- **No union-find needed** — a label-propagation flood (the same Jacobi style used by the
  step-6 fill) suffices. CC counts are small enough that O(K²) pair testing would also work,
  but a single flood pass is cleaner.

### 4a. flip edge — two candidates (undecided)

Because the barrier is ~2 voxels thick, **a plain ±1 neighbor comparison does not work** (no
single voxel sees the CC on *both* sides). Two candidates:

**Candidate A — ray march (per barrier voxel):**
- From each barrier voxel, shoot rays along the **3 axes** (`±x, ±y, ±z`); on each ray, skip
  over barrier voxels and take the **first non-barrier CC** in each direction.
- If the `+` direction and `−` direction land on different CCs → flip edge.
- Rules: probe **all 3 axes**; if both ends hit the *same* CC, **ignore (self-loop)**; if the
  first non-barrier voxel is inactive, skip that direction; dedup edges.
- Pros: independent, parallel, simple. Cons: axis-aligned rays can self-loop on a surface
  tangent to an axis (harmless); curved surfaces are still fine in practice.

**Candidate B — barrier label-flood (race):**
- Seed barrier voxels with adjacent CC labels → flood *through* the barrier medium → a
  collision of two different labels = flip edge.
- Pros: follows the shell, so thickness / slope don't matter. Cons: multi-source flood is a
  bit more work than A.

→ **A (ray) and B (flood) give the same result** (the two CCs a barrier separates). Keep both
as candidates; pick one at implementation time.

### 4b. keep edge — flood (required)

- "Reachable?" is a **connectivity** question, so a flood is the natural tool (a ray does not
  fit — empty space has no notion of "two sides").
- Label-flood the inactive medium → CCs that reach the same region are joined by keep edges.
  Equivalently, treat the **region as a hub node** and record CC ↔ region edges (§6).

---

## 5. The "opposite side" caveat for flip (robustness)

- A flip edge joins the two CCs a barrier separates, but **pure topology cannot always tell
  which two are "opposite".**
- **Clean manifold:** exactly 2 CCs per barrier shell (one per side) → "opposite" is
  unambiguous → correct.
- **Ambiguous cases** (§11): a shell abutting >2 CCs (junction / pinch) → which two are
  opposite is undecidable from topology alone → detected as a parity conflict (§11).

---

## 6. Avoiding O(n²) — the separator as a hub node

Instead of all-pairs CC comparison (O(K²)):
- **keep:** make the inactive region a node; record only CC ↔ region adjacency (a boundary
  sweep, O(boundary)). All CCs touching one region are mutually "keep" through it →
  **clique (K²) collapses to a star (K).**
- **flip:** the ray / flood emits only the CC pairs it actually reaches (deduped) →
  O(barrier voxels).
- Total is O(voxels), independent of K². (O(K²) is fine when K is small, but this is safer.)

---

## 7. The exterior seed — finding "definitely outside"

- A **corner of the padded bounding box is unambiguously exterior** (outside the bbox that
  wraps the whole band).
- The **inactive region containing that corner = the exterior region** → parity seed `+`.
- Every object's outer band CC shares this region (keep edge) → all get `+`.
- Unlike the min-x single seed, this is **robust to multiple / surrounded objects** — every
  object's exterior connects to the same exterior region.

---

## 8. Absent space & the hierarchy (NanoVDB materialization)

Of the inactive space, the only part **not materialized** in the grid is root-absent space (a
4096³ region with no upper node):

| inactive kind | in the grid? | where its label lives |
|---|---|---|
| inactive voxel inside a leaf | ✅ | per-leaf `Mask<3>` |
| childless lower slot (8³) | ✅ | per-lower `Mask<4>` |
| childless upper slot (128³) | ✅ | per-upper `Mask<5>` |
| **absent root region (4096³)** | ❌ | **dense P×Q×R array** (bbox quantized to 4096) |

- The **empty space between scattered objects** is absent → represent it temporarily with the
  **dense array** (the step-6 "chunk C" array) to connect them. Root cells are enormous
  (4096³), so the array is only a *handful* of cells (bbox / 4096). Only extreme separations
  blow the array up — a scaling limit that would push toward a sparse root.
- **Hierarchical flood:** the fine levels (leaf / lower / upper) plus the dense root array are
  stitched by **cross-level neighbor probes** (the probe cascade), joining the fine band to
  the coarse empty space.

---

## 9. Inactive labeling = keep edges *and* the step-6 fill (two birds)

- Label-flood the inactive space → **regions**.
- Each region receives a **sign (color) from parity** → that color **is** the step-6 invert
  bit (leaf `Mask<3>` / lower `Mask<4>` / upper `Mask<5>` / root dense array).
- → **the separate step-6 flood is no longer needed.** The *same* flood produces both the keep
  edges (graph) and the fill (step 6).
- Information flow: **active CC signs (decided geometrically) → parity → region signs →
  inactive fill.** Not circular — the flood is purely topological; signs are assigned by parity
  afterward.

---

## 10. Full pipeline (new version)

```
1. UDF + nearest-tri index          (existing, MeshToGrid)   ※ the new step 4 uses no triangle
                                                                geometry → the index is for step 5 only
2. barrier prune → derived grid      (existing)
3. CC labeling (non-barrier active)  (existing, step 3)
── new step 4 from here ──
4a. flip edges:  barrier label-flood OR ray (CC ↔ CC)
4b. keep edges:  inactive label-flood (CC ↔ region, region as hub)
4c. inactive region labels (= same flood as 4b; absent space → dense array)
4d. identify the exterior region (touches a padded-bbox corner) = +
4e. parity BFS → sign every CC + region
── existing, unchanged ──
5. barrier-voxel signs (independent; mirrors ComputeIntersectingVoxelSign; unrelated to step 4)
6. fill = region color → invert-mask lookup (no separate flood, §9)
7. query: signedSignAt (existing; composes the sidecars)
```

---

## 11. Robustness & fundamental limits

### 11a. parity conflict ⟺ ambiguous topology
- 2-coloring is consistent ⟺ **every cycle has an even number of flip edges** (no odd-flip
  cycle).
- **Closed orientable manifold:** any closed loop crosses the surface an even number of times
  → always 2-colorable → **no conflict** (well-defined).
- **An odd-flip cycle (conflict)** = inside/outside was intrinsically ambiguous:
  - **non-orientable** surface (Klein-bottle-like),
  - **self-intersecting / non-manifold junction** (3+ sheets meeting at one edge),
  - (open surfaces / holes usually degrade gracefully via CC merging, not a conflict).
- **Handling:** detect the color clash during BFS → flag "ambiguous" → fall back to a local
  geometric heuristic (step-5 style).

### 11b. thin feature = a resolution issue (not signing's job)
- **thin wall / fin collapse:** if a wall is thinner than `√3/2 × 2 ≈ 1.73` voxels, its
  interior is *all* barrier → the two surfaces merge into one barrier → flip under-counts the
  crossing → parity is wrong.
- **thin exterior neck / near-contact:** if two parts are within `~1.73` voxels, the exterior
  between them is all barrier → band pinch (one material splits into 2 CCs) → a tangent ray /
  flood produces a false flip.
- Both are **discretization artifacts of a grid too coarse to represent a thin structure** →
  **higher resolution fixes them** (a non-barrier CC reappears inside). This is the **input's**
  responsibility (voxelSize / rasterization), outside the signing algorithm's contract.
- Contract: *"rasterize finely enough that the band resolves the thinnest feature."*

### 11c. merged shell (junction / near-contact)
- A barrier shell abutting >2 CCs → ambiguous flip pairing. Classify as §11a (non-manifold,
  inherent) or §11b (near-contact, resolution). Detected via a parity conflict against the
  keep edges.

---

## 12. Semantics — winding vs OpenVDB flood (to be decided)

| | separated objects | nested cavity | note |
|---|---|---|---|
| **this new algorithm (parity / winding)** | ✅ | **+ (hollow)** | even crossings = outside |
| flood-from-outside (OpenVDB) | ✅ | **− (solid)** | trapped = unreached = interior |
| current min-x | ❌ | − | single-seed limit |

- **The new algorithm = winding:** a nested cavity is `+` (a hole). This **intentionally
  differs** from OpenVDB (cavity `−`).
- If the goal is to *replicate* OpenVDB, flood-from-outside is simpler (no flip edges needed).
  If the goal is **winding (hollow-correct)**, this parity algorithm is the one. ← the choice
  depends on the final target (needs confirmation).

---

## 13. Undecided / candidates (to finalize)

1. **flip detection:** ray (A) vs barrier label-flood (B) — both candidates. Equivalent
   results; simplicity vs shell-following robustness.
2. **semantics:** winding (cavity `+`) vs OpenVDB flood (cavity `−`) — the final target (§12).
3. **conflict fallback:** the concrete local heuristic when parity conflicts.
4. **absent-root scaling:** whether the dense array suffices or a sparse root is needed for
   extreme separations.

---

## 14. Summary at a glance

- **Problem:** the current min-x single seed mislabels separated & nested shapes.
- **Fix:** **graph 2-coloring (parity)** — nodes = CCs + inactive regions; edges = flip
  (barrier between) / keep (joined through empty space); seed the exterior region `+` and BFS,
  flipping at every flip edge. This *is* the winding sign.
- **Edge detection:** "flood CC labels through a medium → a collision is an edge." Barrier
  medium ⇒ flip (ray or flood candidate); inactive medium ⇒ keep (flood). No union-find, no
  geometry, O(voxels).
- **Exterior:** the inactive region touching a padded-bbox corner = `+`. Robust to
  multiple / surrounded objects.
- **Absent space:** only the root level needs a dense array (bbox / 4096); everything else is
  already labelable via per-node masks.
- **Two birds:** inactive labeling serves *both* the keep edges and the step-6 fill (region
  color = invert bit).
- **Limits:** a parity conflict = ambiguous topology (non-orientable / self-intersecting,
  inherent); a thin feature = a resolution issue (input). A well-resolved orientable mesh is
  always 2-colorable → correct.
