# Inclusion-forest signing — the per-component-SDF approach

> Design notes for a more robust replacement of the step-4 sign rule. It handles **separated objects**
> and **nested / hollow** shapes that the current *min-x single-seed* rule mislabels. **Not yet
> implemented.**
>
> Pipeline reference: [`MeshToSDF_PipelinePlan.md`](./MeshToSDF_PipelinePlan.md).

---

## 0. Motivation — why replace the min-x single-seed rule

Today's step 4 picks **one** exterior component (the one holding the global min-x voxel) and calls
everything else interior. This is correct for a single, simply-connected, watertight surface, but it
mislabels:

- **Separated objects** — only one object is called exterior; a second disjoint object's outer shell is
  wrongly interior.
- **Nested / hollow** shapes — cavities and nested shells are signed by a single seed, which cannot
  express the correct inside/outside parity.

The core fix idea is the **even-odd / winding rule**: a point is inside iff it is enclosed by an odd
number of surfaces. This variant realises it by **decomposing
the input into one SDF per surface and composing them**, so the single-seed rule only ever runs on a
*single* surface (where it is provably correct).

---

## 1. The idea in one paragraph

Run connected components on the **un-pruned** UDF so each closed surface becomes **one component γᵢ**.
Turn **each γᵢ into its own full signed distance field φᵢ** (reusing the existing pipeline — this is a
single-object problem, so min-x is valid). Then recover how the surfaces **nest** by a cheap **sign
query**: `γⱼ ⊂ γᵢ  iff  φᵢ(V) < 0` for a voxel `V ∈ γⱼ`. The inclusions form a **forest**; **compose**
the per-surface SDFs down that forest with a **level-set boolean, flipping the sign at each nesting
level** (union at even depth, subtract at odd) — exactly the even-odd rule.

---

## 2. Step 0 — components of the *un-pruned* UDF

The current pipeline **prunes** the barrier shell (`udf² < 0.75·h²`) before CC, which splits each closed
surface into two shells (inner + outer). The inclusion approach instead runs CC on the **UDF band with
the barrier kept**, so the inner and outer shells stay glued → **one component per closed surface**:

- Surfaces that are ≥ ~2·bandwidth apart → one component each (γ₁ … γₙ).
- Surfaces closer than the band width → their bands merge into one component (the hard case, §7).

`ConnectedComponents<ValueOnIndex>` is domain-agnostic, so this is literally "run CC on the original
grid instead of the derived one." It is also a useful **first diagnostic**: *for the problem meshes
(e.g. cavities), how many components does the UDF have before barrier removal?* — which tells us the
component count `N` this whole approach must scale to.

---

## 3. Step 1 — each component γᵢ → a full SDF φᵢ

For each component γᵢ, run the **existing** mesh→SDF pipeline (prune → sub-CC → sign → fill) restricted
to γᵢ's voxels, with the mesh triangles available for barrier signing. The result φᵢ is the signed
distance field of surface Γᵢ **as if it were the only surface**, filled everywhere (queryable at any
coordinate via `signedSignAt`).

Two properties make this sound:

- **Single-seed is valid per component.** γᵢ contains exactly one surface, so the min-x rule (outer
  shell touches the domain extreme → exterior `+`, inner shell → interior `−`) is correct. The
  multi-object failure of min-x is *dissolved* by the decomposition.
- **φᵢ is a full SDF, not just a band.** The step-6 invert-mask fill makes φᵢ answer a sign query at any
  point — including points near *other* surfaces. That is what the next step needs.

**Rasterization is done once (shared), not per component.** The UDF (`|dist|` + nearest-triangle index)
is computed once on the full mesh; each γᵢ is a *subset* of that already-computed field (partitioned by
CC label). Only the cheaper downstream steps (prune / sub-CC / barrier-sign / fill) run per component,
and because the voxels are a partition, their **summed** cost is ≈ one pass over the band, not `N×`.
See §6.

---

## 4. Step 2 — inclusion test (nesting via a sign query)

For an ordered pair (i, j), ask whether Γⱼ is nested inside Γᵢ:

```
pick a voxel V in γⱼ
γⱼ ⊂ γᵢ   ⟺   φᵢ(V) < 0        (V is inside surface Γᵢ)
```

- Correct for cleanly nested surfaces: every voxel of Γⱼ is inside Γᵢ, so any V gives `φᵢ(V) < 0`.
- For disjoint surfaces: `φᵢ(V) > 0` and `φⱼ(W) > 0` → neither includes the other → siblings (→ union).
- **No flood fill.** This is the key point: the reachability / "same-material" question is answered
  by a **sign query on the already-built SDF**, reusing `signedSignAt`, not by a separate flood.

**Robustness hardening (recommended):** rather than a single voxel, query several voxels of γⱼ and take
the **majority** (or require unanimity). A single voxel near a tangency where `φᵢ(V) ≈ 0` could misfire;
a majority vote removes that fragility. (Clean nesting with a ≥1-voxel gap is unambiguous regardless.)

---

## 5. Step 3 — inclusion forest, then compose with parity

**Build the forest.** Collect the pairwise inclusions into a partial order. For cleanly nested / disjoint
surfaces this is a genuine partial order (antisymmetric, transitive: Γₖ⊂Γⱼ⊂Γᵢ ⟹ Γₖ⊂Γᵢ), whose Hasse
diagram is a **forest**. Flatten by **transitive reduction** — keep only the immediate-parent edge
(e.g. if γ₁⊃γ₂⊃γ₃ and γ₁⊃γ₃, drop the γ₁→γ₃ edge). This is a small host-side graph op (`N` is the
number of surfaces, typically tiny).

**Sanity check:** verify the computed pairwise relation is a *consistent* forest (no cycles /
contradictions). For well-behaved input it will be; a contradiction flags intersecting or ambiguous
geometry (§7).

**Compose.** Walk the forest from each root, flipping the sign at every level of descent:

```
φ_final = Γ_root                         (depth 0 = solid, interior −)
          −  (immediate children)         (depth 1 = cavity → subtract)
          +  (their children)             (depth 2 = solid → union)
          − ...
```

A point enclosed by *k* surfaces along its chain gets sign `(−1)^k` — the **even-odd / winding rule**.
The composition uses standard level-set booleans (`union = min`, `difference = max(·, −·)`); the **sign
is exact**, the magnitude is approximate near boolean seams (redistance if a true Euclidean SDF is
needed).

Disjoint surfaces at the top level are simply **unioned** (union of topologies + injected sidecars,
easy for disjoint fields).

---

## 6. Cost — rasterize once, downstream partitioned

| stage | per component? | total cost |
|---|---|---|
| rasterize (UDF + nearest-tri index) — the most expensive stage | **no, once (shared)** | **1×** |
| prune / sub-CC / barrier-sign | per component | voxels are a partition → summed ≈ **1 pass** |
| invert-mask fill (make each φᵢ full) | per component | nested interiors overlap a little, but the fill is per-node **sidecar bits** (cheap), only the root array is dense |
| inclusion tests | O(N²) pairs (or fewer) | each is one `signedSignAt` query — cheap |
| forest + compose | host graph + level-set booleans | small |

So the naive fear — `N × (whole pipeline)` including `N` rasterizations — **does not hold**. The real
shape is **`1× rasterize + ~1–2× the cheap downstream steps`**. The remaining `N`-sensitivity is: (a)
building/holding `N` per-component grids + sidecars (memory), and (b) the fill overlap for deeply nested
inputs. Both are modest as long as the fill stays sidecar-based (not densified). **The number that
decides feasibility is `N` (component count) — hence the diagnostic in §2.**

---

## 7. Weaknesses / genuinely ambiguous cases

- **Intersecting surfaces.** If Γⱼ is *partially* inside Γᵢ, the inclusion test gives different signs for
  different voxels of γⱼ — inclusion is ill-defined and the partial order breaks. This is not a failure
  of the method: inside/outside is genuinely ambiguous there. The desired behaviour is a convention
  choice (a safe one: take the union of the two solids).
- **Nested closer than the band width.** Two surfaces within ~2·bandwidth merge into a *single*
  component in §2, so they cannot be separated at all. This is a **resolution limit** (the two surfaces
  are indistinguishable at this voxel size), not an algorithm bug. Fallback: take the outermost.
- **Barrier voxel with all-same-sign neighbours.** Set it to that sign; one suggestion is to use `−epsilon`
  (not exactly 0) and possibly a redistance pass so the level set stays well-defined.
- **Open / non-watertight surfaces.** No well-defined interior → φᵢ is ill-posed; a pre-existing
  limitation of the whole SDF approach, unchanged here.

---

## 8. Implementation notes

Most of the pipeline already exists — this approach is largely orchestration on top of it:

- **§2 (CC on un-pruned UDF):** feed the original grid to `ConnectedComponents` instead of the derived
  one. Extract each γᵢ as a masked sub-grid (label == i) via `PruneGrid`.
- **§3 (per-component SDF):** re-use `buildMeshToSdf` on each γᵢ's voxels (pass the full triangle list).
- **§4 (inclusion):** re-use `signedSignAt` — one query per (i, j) on a representative voxel (or a few).
- **§5 (forest):** small host-side transitive reduction.
- **§5 (compose):** the one genuinely new piece. Two strategies:
  - **(a) Materialize** a single composed SDF grid (union topology + a CSG value pass). Harder, but the
    output is a standalone level-set grid.
  - **(b) Query-time composition** — keep the `N` per-component SDFs + the forest, and answer a sign
    query for a point P by the parity of its enclosing chain (the innermost enclosing surface's depth).
    Easier (no new grid built), at the cost of an O(depth) query and an output that is "N SDFs + a rule"
    rather than one grid — an N-fold extension of the current sidecar philosophy.

Strategy (b) keeps the whole approach at **moderate** difficulty; (a) is more work. The first concrete
step is the free diagnostic in §2 (count un-pruned UDF components) to learn `N` before committing.

---

## 9. Open questions (to resolve)

1. Desired result for **intersecting** surfaces and for **nested-closer-than-bandwidth** (union? outer
   only? redistance?).
2. Composition strategy: materialize one grid (a) vs query-time (b).
3. Performance ceiling: what is `N` on real inputs (foam / bubbles could be large), and does the
   per-component-SDF cost stay acceptable there?
