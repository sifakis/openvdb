# How the mesh → SDF pipeline works

A walkthrough of `nanovdb/tools/cuda/MeshToSDF.cuh` as it is implemented today: what it produces,
what each stage does, and why the stages are ordered the way they are.

The other documents here are design notes written *while* the pipeline was being built —
`MeshToSDF_PipelinePlan.md` for the original plan, `InclusionSigningDesign.md` for the reasoning
behind the multi-surface signing, `BENCHMARK.md` for measurements. This one describes the finished
thing.

---

## 1. What you get

```cpp
using namespace nanovdb::tools::cuda;

MeshToSDF<nanovdb::ValueOnIndex> sdf(d_points, pointCount, d_triangles, triangleCount, map);
sdf.setNarrowBandWidth(3.f);
sdf.build();
```

After `build()`, the result is a **narrow-band signed distance field**: exact distances near the
surface, and a correct *sign* everywhere else. It is spread across a grid and a handful of parallel
arrays ("sidecars"):

| what | accessor | length |
|---|---|---|
| the narrow band itself | `deviceGrid()` | a `ValueOnIndex` grid |
| unsigned distance, world units | `deviceUDF()` | activeVoxelCount + 1 |
| sign, +1 outside / −1 inside | `deviceSign()` | activeVoxelCount + 1 |
| sign of inactive voxels in materialized leaves | `deviceLeafInvertMask()` | one `Mask<3>` per leaf |
| sign of childless 8³ tiles | `deviceLowerInvertMask()` | one `Mask<4>` per lower node |
| sign of childless 128³ tiles | `deviceUpperInvertMask()` | one `Mask<5>` per upper node |
| sign of absent 4096³ regions | `deviceRootInterior()` | one `uint8` per root cell |

Two conventions worth internalising:

- **Sidecars are indexed by `leaf.getValue(n)`.** A `ValueOnIndex` grid stores an *index* per active
  voxel, not a value; the value lives in a flat array at that index. Slot 0 is the background.
- **The grid is never rebuilt.** Every stage writes a sidecar or a mask. Nothing mutates topology
  after the initial rasterization, so a caller who holds the grid keeps holding the same grid.

### Querying it

`sdf_detail::signedSignAt()` folds all of the above into the sign at any coordinate — one tree
descent, stopping at the first level that has no child:

```
root has no child there  ->  root-interior sidecar      ->  ±1
upper slot has no child  ->  upper invert mask bit      ->  ±1
lower slot has no child  ->  lower invert mask bit      ->  ±1
leaf voxel is active     ->  sign sidecar               ->  ±1
leaf voxel is inactive   ->  leaf invert mask bit       ->  ±1
```

A set bit means **interior**. Combined with `deviceUDF()` on the band, that is a complete level set.

> **Not yet a `NanoGrid<float>`.** The output is an index grid plus sidecars, so tools that expect an
> ordinary typed grid (samplers, file I/O, renderers) cannot read it directly. Turning this into a
> materialized level set — or wrapping it in an abstraction that carries the sidecars along — is an
> open design question, not an oversight.

---

## 2. The one idea the whole thing rests on

Signing a distance field means answering "is this point inside?". The pipeline answers it with a
connectivity argument:

> Remove a thin shell of voxels right at the surface. What is left falls apart into disconnected
> pieces. The piece containing the leftmost voxel in the whole grid must be *outside*, because
> nothing is further out than the leftmost voxel. Every other piece is *inside*.

That argument is airtight — **for one closed surface**. It breaks immediately with two:

- Two separate objects? Only one of them owns the global leftmost voxel; the other object's outer
  shell gets called interior.
- A hollow object? The cavity is "inside the outer surface" but "outside the inner one". A single
  inside/outside label cannot express that.

So the pipeline **partitions the input into closed surfaces first**, signs each one on a grid where
it is the only thing present — restoring the conditions the rule needs — and then composes the
results. Everything below follows from that decision.

---

## 3. The stages

```
1  rasterize      mesh ────────────────────────────► band grid + UDF + nearest-triangle index
2  partition      band grid ───────────────────────► which closed surface each voxel belongs to
3  per surface    ┌─ carve it out of the band
                  ├─ drop the barrier shell
                  ├─ label the pieces
                  ├─ sign them (leftmost piece = outside)
                  ├─ sign the barrier shell
                  └─ extend the sign off the band          ► one complete field per surface
4  compose        the fields tell each other who encloses whom
                  ───────────────────────────────────────► one sign array on the band grid
5  fill           extend that composed sign off the band  ► the final invert masks
```

### Stage 1 — Rasterize

`MeshToGrid` turns the triangle soup into a narrow band around the surface: a `ValueOnIndex` grid
whose active voxels are those within `bandWidth` cells of some triangle, plus two sidecars — the
**unsigned distance** and the **index of the nearest triangle** for each of them.

No sign yet. Everything after this is about deciding sign.

This stage is also where nearly all the time goes for an ordinary single-object mesh (~96%; see
`BENCHMARK.md`), because it is bound by triangle count rather than voxel count.

### Stage 2 — Partition into closed surfaces

Run connected components on the band **before** anything is removed from it.

Why before: the band wraps *around* the surface, so its inner and outer sides are joined through the
voxels sitting right on the surface. Leave those in place and each closed surface stays glued into
exactly one component. The component count is the surface count.

The result is a per-voxel label saying which surface each active voxel belongs to.

> This is a statement about **connectivity of the band**, not about closed-ness. Two surfaces closer
> together than the band width merge into one component; an open surface still yields one component
> even though it encloses nothing. See "What it does not handle" below.

### Stage 3 — Sign each surface on its own

For each surface, `signSurface()` runs the following. The result is the field that surface would
have if it were the only object in the scene.

**(a) Carve.** Build a retain mask selecting only this surface's voxels, `PruneGrid` the band with
it, and re-index the UDF and nearest-triangle sidecars onto the smaller grid. Pruning renumbers the
value slots, so the sidecar transfer goes through an injection functor that pairs leaves by origin
and remaps slots by popcount rank.

> Skipped entirely when the mesh has a single closed surface, since the band already *is* that
> surface's band. That is a memory decision: carving duplicates the grid plus three sidecars, about
> 9 bytes per voxel, and forcing it on single-surface input costs ~45% more peak memory with no time
> benefit (measured, `BENCHMARK.md`).

**(b) Drop the barrier shell.** A voxel is a *barrier* if it is within √3/2 voxels of the surface —
i.e. `udf² < 0.75·voxelSize²`. Those are the voxels that could be on either side, and they are
exactly what holds the band together. `computeDerivedTopology()` prunes them away.

**(c) Label the pieces.** Connected components on what is left. For one closed surface this splits
into an outer shell and an inner shell (plus one stranded piece per thin or concave pocket — which
is why the component count can be large and still be correct).

**(d) Sign the non-barrier voxels.** `signNonBarrier()` finds the component holding the minimum-x
active voxel and calls it exterior (+1); every other component is interior (−1). This is the rule
from §2, and it is valid here precisely because the grid holds one closed surface.

**(e) Carry those signs back and sign the barrier.** `injectSignsToOriginal()` copies the signs onto
the un-pruned surface grid, leaving barrier voxels at a sentinel 0. `signBarrier()` then fills them:
a barrier voxel is exterior if some already-signed exterior neighbour's nearest triangle places both
of them on the same side of that triangle. This mirrors OpenVDB's `ComputeIntersectingVoxelSign`,
and it reads from an immutable snapshot so the result does not depend on execution order.

**(f) Extend the sign off the band.** Three fills, coarsest information last:

- `fillLeafInvertMask()` floods interior signs from interior band voxels through the *inactive*
  voxels of each materialized leaf. Active voxels act as walls.
- `fillCoarseInvertMasks()` seeds childless lower and upper tiles from leaf faces, then floods.
- `fillRootInteriorMask()` builds a small dense cell array over the grid's root-tile range and floods
  the deep interior that lies beyond any upper node.

After (f) this surface's field can be queried anywhere, not just on its band. That is what stage 4
needs.

### Stage 4 — Compose by inclusion

Each field was built as if its surface stood alone. Side by side that is already right; enclosed it
is not — a cavity comes out signed like a solid. The fix is the even-odd rule:

> A point wrapped by *k* closed surfaces is inside iff *k* is odd.

so surface *i*'s own signs must be negated exactly when its nesting depth is odd.

Recovering the depth:

1. **Pick one representative voxel per surface** (a deterministic minimum over packed coordinates).
   Any voxel of a surface's band will do: the band hugs its own surface, so it lies wholly inside, or
   wholly outside, every *other* surface.
2. **Ask every field about every representative.** `φᵢ(Vⱼ) < 0` means surface *j* lies inside surface
   *i*. This is the only place the per-surface invert masks are used — the representative is off
   `φᵢ`'s own band, so without them `φᵢ` could not answer at all.
3. **Depth = how many other surfaces report this one as inside them.** The inclusion forest itself is
   never built; only the parity of the depth matters.

Then the merge:

4. **Scatter every surface's band signs back onto the band grid** (through the same injection
   functor, since carving renumbered the slots), and **negate the odd-depth ones in place**.

The scatter is exact because the surfaces *partition* the band grid's active voxels — connected
components labels each one exactly once — so the writes are disjoint and together total.

### Stage 5 — Fill on the band grid

Run the same three fills once more, on the band grid, seeded by the composed signs. These are the
invert masks the caller gets.

---

## 4. Two things that surprise people

### There is only one grid

The per-surface grids are not independent grids. Each is the band grid pruned by a per-surface voxel
mask, so their leaf origins coincide by construction. Nothing is ever "combined" in the CSG sense —
no min, no max, no union of two level sets. What is combined is **per-voxel signs**, scattered back
into a single array on the one grid that existed all along.

A useful consequence: the band grid is, by construction, the union of every surface's band. So a
*tile* can only exist where **no** surface has a band. The awkward case of "one field has a tile here
while another has voxels" cannot appear in the output.

### The per-surface invert masks are thrown away

They exist for exactly one purpose — letting stage 4 query a field off its own band — and are dead
the moment the nesting depths are known. The final off-band signs are recomputed from scratch in
stage 5.

They are not merged because they *cannot* be: each is indexed by its own grid's node array, so there
is no slot correspondence. And even with aligned arrays the rule would not be a bitwise operation —
the composed answer is "inside an odd number of surfaces", which would mean evaluating all *N* fields
at every point. Paying for one extra fill collapses *N* fields into one, and every later query is a
single tree descent.

> Logically discarded, but not yet *freed*: the per-surface state currently lives until the whole
> conversion finishes. That is most of the ~4 MiB-per-surface memory reported in `BENCHMARK.md`.

---

## 5. What it does not handle

Measured behaviour, not speculation:

| input | what happens |
|---|---|
| **Open / non-watertight** | The band is still one component, and nothing is enclosed, so everything comes out exterior. **No diagnostic is emitted** — every internal check passes. |
| **Self-intersecting, or objects closer than the band width** | They merge into one component and are signed as their union. Defensible, but it is a silent reinterpretation of the input. |
| **Nested / cavities** | Correct, by the even-odd rule above. |
| **Very thin features** | Correct but degenerate: if a feature is thinner than the band, it has no interior at that resolution. The hairball has zero interior voxels at every resolution tested. |

The first row is the significant one. A tool that answers confidently and wrongly on malformed input
is worse than one that refuses, so detecting and reporting non-closed input is a prerequisite for
calling this production-ready.

---

## 6. Where the code is

| | |
|---|---|
| `nanovdb/tools/cuda/MeshToSDF.cuh` | the whole pipeline |
| ‣ `MeshToSDF` | the orchestrator — stages 1, 2, 4, 5 and the per-surface loop |
| ‣ `sdf_detail::SurfaceSigner` | stage 3, the stages that sign one closed surface |
| ‣ `sdf_detail::signedSignAt` | the query |
| `nanovdb/tools/cuda/MeshToGrid.cuh` | stage 1 |
| `nanovdb/tools/cuda/ConnectedComponents.cuh` | stages 2 and 3c |
| `nanovdb/tools/cuda/PruneGrid.cuh` | the carve and the barrier prune |
| `mesh_to_sdf_cuda_kernels.cu` | the validation harness — independent CPU mirrors of each stage, an OpenVDB cross-check, and analytic ground truth |

Run `./ex_mesh_to_sdf_cuda --selftest` for the analytic suite, or pass an `.obj` to run and validate
a real mesh. `VISUALIZATION.md` covers dumping the result for the viewer.
