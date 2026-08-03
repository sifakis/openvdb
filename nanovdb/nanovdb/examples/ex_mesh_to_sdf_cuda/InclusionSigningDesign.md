# Inclusion-forest signing — the per-component-SDF approach

> Design notes for a more robust replacement of the step-4 sign rule. It handles **separated objects**
> and **nested / hollow** shapes that a *single* min-x seed mislabels.
>
> **Status: implemented** (§10). Surfaces are separated by labeling the un-pruned band, each is signed
> on its own by a per-surface exterior seed, and nesting is folded in by per-component sign fields, an
> inclusion test and a depth-parity flip. Self-tests cover separated, nested and doubly-nested cases.
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
grid instead of the derived one."

**Measured component counts.** The number `N` decides whether the whole approach is affordable, so it
was measured before building on it (voxel size 0.004, hairball coarser):

| mesh | un-pruned `N` | barrier-pruned components |
|---|---|---|
| sphere / bunny / armadillo / hand / cat | **1** | 2 / 4 / 2 / 1 / 2 |
| dragon | **1** | 40 |
| hairball @ 0.02 | **1** | 14,500 |
| hairball @ 0.01 (118 M active voxels) | **1** | 334,000 |
| two separated spheres | **2** | 4 |

So `N` is the count of genuinely distinct closed surfaces — not of shell fragments. The large pruned
counts (dragon's 40, bunny's ear pockets, hairball's hundreds of thousands) are all artifacts of
splitting each band into an inner and an outer shell; they vanish on the un-pruned band. Per-surface
work is therefore cheap, and for any single-object mesh (`N == 1`) this decomposition is the identity.

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

---

## 10. Implementation roadmap

The approach is largely orchestration on top of the existing pipeline (§8), so it is staged into
milestones that each build on verified ground. The current `buildMeshToSdf` runs six steps —
rasterize (UDF + nearest-tri index, shared/expensive) → prune → CC → **sign (min-x single-seed)** →
barrier-sign → invert-mask fill — and this work replaces only the signing step with the decomposition.

| id | milestone | what | reuses | depends on |
|---|---|---|---|---|
| **M0** | **un-pruned CC count** | Run CC on the un-pruned grid (`orig`, barrier kept) and report the global-label count `N` per mesh, alongside the derived-grid count. Diagnostic only — existing behaviour unchanged. | `ConnectedComponents::label()`, existing label-count code | — |
| **M1** | **partition** | Split the labeled grid into `N` per-component sub-grids (one per label). | — | M0 |
| **M2** | **per-component SDF** | For each γᵢ, run prune → sub-CC → sign → fill restricted to its voxels (triangles shared) → a full φᵢ queryable via `signedSignAt`. | `computeDerivedTopology`, `MeshToSDF::sign*/fill*` | M1 |
| **M3** | **inclusion test** | `γⱼ ⊂ γᵢ ⟺ φᵢ(V) < 0` for representative V ∈ γⱼ; majority vote over several voxels. O(N²) sign queries. | `signedSignAt` | M2 |
| **M4** | **forest + compose** | Host-side transitive reduction → inclusion forest → compose per-surface SDFs with depth-parity. Start with query-time composition (§8b), materialize (§8a) only if a standalone grid is needed. | level-set booleans | M3 |
| **M5** | **validation** | Analytic self-tests: two separated spheres, nested spheres (sphere-in-sphere), sphere-with-cavity. Extend the `--two-spheres` self-test. | existing self-test harness | M4 |

### Ordering strategy

```
M0 (diagnostic, ~free — establishes N)
  └─ M1 prototype (per-label PruneGrid)  ──┐
       └─ M2 → M3 → M4(b) → M5             │  end-to-end algorithm validation
            └─ swap M1 for a batched partition primitive (optimized)
```

**Prototype M1 with per-label `PruneGrid` first** (§8): extract each γᵢ as a masked sub-grid so M2–M5
can proceed and the whole algorithm is validated end to end. Replace that with a dedicated **batched
partition primitive** (label → `N` index grids in one pass) as a later optimization — a sibling of the
existing `prune` / `dilate` / `merge` topology ops. Decoupling M1's optimization from M2–M5 keeps the
algorithm work unblocked.

### Status

- **M0 — done.** Component counts measured for the whole mesh suite; see the table in §2.
- **Per-surface exterior seeding — done.** This is the part of the scheme that needs no inclusion
  test: separated surfaces are all at depth 0, so composing them is a plain union and the only thing
  required is *one min-x seed per surface* instead of one globally. Labeling the un-pruned band
  (§2) gives each voxel a surface id; those ids are carried onto the barrier-pruned grid with
  `util::cuda::InjectGridDataFunctor` (which is renumbering-safe), and `MeshToSDF::signNonBarrier`
  now reduces to one exterior representative per surface. Barrier signing and the invert-mask fill
  needed no change — both work off the signed band. With one surface the code path reduces exactly
  to the previous behaviour. The `--two-spheres` case went from 7808 confident sign mismatches to 0
  and is now an asserting self-test rather than a report-only probe.
- **Nesting — done.** Once every surface is signed on its own (through `signBarrier`), each one's band
  is carved out as a sub-grid with `PruneGrid`, its signs are carried across with
  `InjectGridDataFunctor`, and the step-6 fill is run on it — giving φᵢ, the sign field surface i would
  have alone, defined everywhere. Probing φᵢ at one voxel of every other surface's band yields the
  inclusion relation; a surface's nesting depth is simply how many surfaces report it as inside, and
  its signs are flipped iff that depth is odd. The composed signs then go through the ordinary step-6
  fill once, so the output has exactly the same shape as before and the whole validation harness
  applies unchanged. The three fill methods gained an optional external sign array to make this
  possible; with one surface the entire stage is skipped.
- **M5 — done for the analytic cases.** The oracle was generalized from a union (`min` over
  primitives) to the **even-odd rule** (inside iff an odd number of primitives contain the point,
  magnitude `min|dᵢ|`), which is what a closed-surface soup means and coincides with the union when
  primitives do not overlap. Self-tests, in increasing generality:
  | case | surfaces | depths | what it pins down |
  |---|---|---|---|
  | `--two-spheres` | 2 | 0, 0 | separated objects — the original failure |
  | `--multi-spheres` | 5 | all 0 | many siblings, only one owning the global min-x voxel |
  | `--nested-spheres` | 2 | 0, 1 | a cavity — the inner band must be flipped |
  | `--triple-nested` | 3 | 0, 1, 2 | parity, not mere enclosure: depth 2 is solid again |
  | `--multi-nested` | 5 | 0,1,2 + 0,1 | separation and nesting combined — the relation is a forest with two roots, so depth must be counted against a surface's own ancestors |
- **Still open:** intersecting surfaces and surfaces nested closer than the band width (§7), the
  majority-vote hardening of the inclusion probe (§4 — one representative voxel per surface is used
  today), and the transitive-reduction consistency check of §5 (depth is counted directly, so the
  forest is never materialized).

### Partition first — done

The implementation originally grew in two increments and the seams showed: multi-surface handling was
*threaded through* the single-grid pipeline rather than layered on top of it. The pipeline now has the
shape §2-§5 describes, and the whole signing sequence runs once per closed surface:

```
1  rasterize                                   (shared — the expensive step, done once)
2  connected components on the un-pruned band  → surfaces γ₁..γ_N
3  for each γᵢ:  carve a sub-grid, then run the ordinary pipeline on it  → φᵢ
4  inclusion test → nesting depth per surface
5  compose: fold each φᵢ back with its depth parity, then one final fill
```

Restructuring removed machinery rather than adding it:

- `signNonBarrier` lost its `d_surfaceLabel` / `surfaceCount` arguments — a sub-grid holds exactly one
  closed surface, so a plain global min-x seed is provably correct again. The per-surface indexing in
  `FindExteriorRepFunctor` / `SignNonBarrierFunctor` went away with it, along with
  `ExteriorRepFromKeyFunctor` and the per-surface seed buffer.
- Carrying surface ids onto the pruned grid disappeared entirely — nothing downstream needs them.
- The inclusion stage no longer rebuilds the per-surface fields at the end: they *are* the pipeline's
  output, and it only probes them, counts depths, and merges.
- The host-side barrier oracle lost its nesting-parity parameters, since each φᵢ is surface-local.

The costs are that each sub-grid needs its UDF and nearest-triangle-index sidecars remapped
(`InjectGridDataFunctor<float>` / `<uint32_t>`, mechanical), that composing means injecting each φᵢ's
signs back onto the original grid, and that prune / CC / barrier signing run once per surface instead
of once overall — though the voxels are a partition, so the summed cost is still about one pass (§6).

### The single-surface path

A mesh with one closed surface — every ordinary watertight model — skips the carve: the original grid
already *is* that surface's band. This is a memory decision, and it was measured rather than assumed.
Forcing the multi-surface path for a single surface costs, on the hairball:

| voxelSize | active voxels | carve skipped | carve forced | wall |
|---|---:|---:|---:|---|
| 0.01 | 118 M | 2,556 MiB | 3,676 MiB (+44%) | 9.11 → 9.25 s |
| 0.006 | — | 7,036 MiB | 10,268 MiB (+46%) | 19.61 → 19.94 s |

Time is unaffected (within noise; rasterization dominates), but peak memory rises by ~45% in both
cases, because carving duplicates not only the grid but the UDF (float), nearest-triangle index
(uint32) and sign (int8) sidecars — 9 bytes per voxel. Peak memory is the binding constraint at
production resolutions, so the carve stays conditional.

Everything downstream of the carve is unconditional. Composition detects that a lone uncarved surface
already holds its signs on the original grid and *aliases* them rather than allocating and gathering a
second full-length array; the final fill reads the same signal (an empty composed-sign buffer) and
adopts the fill that surface already ran. So there is exactly one branch in the flow, at the carve.

### Verification

The self-tests (`--two-spheres` … `--multi-nested`, plus the analytic, invert-mask and full-domain
checks) were the safety net for the rewrite, and a new **surface-merge check** was added: the host
gathers each surface's signs, applies the same depth parity, and compares against what the GPU
composed. After restructuring, every self-test assertion and every real-mesh figure (bunny, dragon,
armadillo, cat at 0.004 — component counts, interior/exterior counts, barrier counts, OpenVDB in-shell
ties) is unchanged.
