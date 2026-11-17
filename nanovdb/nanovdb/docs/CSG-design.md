# CSG on NanoVDB index-grid level sets: design notes

**Status:** design only; nothing here is implemented or tested. Every claim marked *argued* is a
paper argument that the first implementation must check against OpenVDB.
**Branch:** `nanovdb-csg` (fork `sifakis`), based on `nanovdb-interface-tracking` `7dbda91f3`.
**Origin:** whiteboard discussion of 2026-11-17 (the case table of section 4.1 and the conjecture
that the invert masks of a union are the OR of the input masks).

**Depends on:** the level-set state and the sign lookup of `InterfaceTracking-design.md` (sections 3
and 7.1 there), `tools::cuda::PropagateSign`, `tools::cuda::MergeGrids`, `tools::cuda::PruneGrid`, and
`isInverted()` in `tools/cuda/LevelSetTracker.cuh`.

---

## 1. Objective

Union, intersection and difference of two narrow-band level sets stored as `ValueOnIndex` grids, each
with its sidecar and invert masks, producing a level set in the same form. The counterpart of
`openvdb::tools::csgUnion`, `csgIntersection` and `csgDifference`.

The question this document answers: the two inputs have different topology, so their invert masks are
indexed by different nodes and may record the sign of the same region at different tree levels. Can the
result be computed inside the representation, without running `PropagateSign` again on the result?

**Out of scope:** inputs with different transforms or background values, more than two inputs, and
the renormalization of the result (section 7).

## 2. Inputs and assumptions

Each input is the state of `InterfaceTracking-design.md` section 3: grid `G`, sidecar `phi` with the
background `gamma > 0` in slot 0, and invert masks `M` (leaf, lower, upper, and the sorted root origin
list). Subscripts 1 and 2 name the two inputs.

Assumptions:

1. Both grids share one index space (same transform) and the same `gamma`.
2. Every active value satisfies `|phi| < gamma` (the state after `prune()`).
3. Each input satisfies the preconditions of `PropagateSign`: no empty nodes; each face-connected
   region of inactive voxels touches active voxels of one sign only; the unbounded region is positive;
   no active tiles and no root value tiles.

## 3. The full-space field

An index-grid level set with invert masks defines a value at every point of index space:

    Phi(p) =  phi[index(p)]   if p is an active voxel
              -gamma          if the inactive element holding p has its invert bit on
              +gamma          otherwise

This is the field an OpenVDB `FloatGrid` level set stores explicitly. The CSG operations are pointwise
on it, exactly as in OpenVDB:

    union          Phi = min(Phi1, Phi2)
    intersection   Phi = max(Phi1, Phi2)
    difference     Phi = max(Phi1, -Phi2)

Everything below is this definition, evaluated on whole tree slots instead of single points.

## 4. Union

### 4.1 Three states, one table for every tree level

Any slot of the tree, at any level, is in one of three states:

| state | meaning | `Phi` on the slot |
|---|---|---|
| `O` | no child, invert bit off | `+gamma` everywhere |
| `I` | no child, invert bit on | `-gamma` everywhere |
| `C` | has a child (at the leaf level: the voxel is active) | varies; described one level down |

At the root level a "slot" is an upper-node-sized region: `C` if the root table has an entry for it,
`I` if its origin is in the root origin list, `O` otherwise.

The union of two slots covering the same region:

| slot 1 | slot 2 | result |
|---|---|---|
| `I` | any | `I`; the other input's subtree, if any, is discarded |
| `O` | `X` | `X`; the other input's subtree is taken unchanged |
| `C` | `C` | `C`; apply this table to the children. At the leaf level: `min(phi1, phi2)` |

`-gamma` is below every value and `+gamma` is above every active value (assumption 2), so `I` absorbs
and `O` is the identity of `min`. The table is the whiteboard case table with "active" generalized to
"has a child": the same three rules hold for voxels, lower slots, upper slots and root regions.

### 4.2 The mask form

For a node present in the result, let `C` be its child mask (the value mask for a leaf) and `I` its
invert mask. Where an input has no node at that position, the whole node is `O` or `I` in that input
and its `C` is empty and its `I` is all-off or all-on accordingly. Then, at every level:

    I_out = I1 | I2
    C_out = (C1 | C2) & ~I_out

The first line is the conjecture from the whiteboard, and it holds per region of space. It is not a
plain OR of the stored mask arrays: the two arrays are indexed by different nodes, and a bit that is on
at one level in one input must remove the nodes the other input holds beneath it. The second line is
that removal. `MergeGrids` computes `C1 | C2`; the difference from a merge is the `& ~I_out`.

### 4.3 The result is a valid level-set state (*argued*)

Claim: in the result, two face-adjacent inactive voxels never have different signs, and the result
satisfies the preconditions of `PropagateSign`.

Argument. Let `p` and `q` be face-adjacent in the result with `q` in state `I`. Then `q` is inactive
and negative in one input, say input 1. In input 1, `p` cannot be inactive and positive, because `p`
and `q` would be one face-connected inactive region with two signs. So `p` is either `I` in input 1,
which makes it `I` in the result, or active and negative in input 1 (precondition 2 applied to the
region of `q`), which makes the result at `p` either `I` or active with a value `<= phi1(p) < 0`.
Hence an `I` voxel of the result has only `I` or active-negative face neighbours. The symmetric
statement for `O` follows the same way: a negative active neighbour of an `O` voxel would be a
negative active neighbour of a positive inactive voxel in the input that supplied the value.

Consequences:

- A leaf of the result with no active voxel left is uniformly `O` or uniformly `I` (a leaf is
  face-connected), so it collapses into one bit of its parent. The same holds one level up. The result
  therefore never needs an empty node.
- The unbounded region stays positive: a region is `I` in the result only where an input is `I`.

### 4.4 Values

A voxel that is active in the result is not `I` in either input, so in each input it is active or `O`.
Because slot 0 of a sidecar holds `+gamma` and `getValue()` returns index 0 for an inactive voxel:

    phi_out[index_out(p)] = min(phi1[index1(p)], phi2[index2(p)])

needs no case analysis.

## 5. Intersection and difference

Intersection exchanges the roles of `O` and `I`: `O` absorbs, `I` is the identity, `C`/`C` recurses
with `max`.

    O_out = O1 | O2        where O = ~C & ~I
    C_out = (C1 | C2) & ~O_out
    I_out = ~C_out & ~O_out

Difference `1 \ 2` is the intersection of input 1 with the complement of input 2. The complement is not
materialized: it would be negative at infinity, which the invert-mask convention does not represent.
Instead the table reads input 2 with `O` and `I` exchanged and its active values negated:

| slot 1 | slot 2 | result |
|---|---|---|
| `O` | any | `O` |
| any | `I` | `O` |
| `X` | `O` | `X` |
| `I` | `C` | `C`, values `-phi2` |
| `C` | `C` | `C`; recurse. At the leaf level: `max(phi1, -phi2)` |

The result is positive at infinity because input 1 is.

## 6. Implementation routes

### 6.1 Route A: existing operators (first draft)

1. `MergeGrids(G1, G2)` gives the topology `C1 | C2` at every level.
2. One kernel per merged leaf builds the retain mask: a voxel is retained if it is active in the merged
   grid and `isInverted()` is false for it in both inputs. `isInverted()` descends each input's own
   tree, so it reads the bit at whichever level that input stores it; the topology mismatch is confined
   to this lookup.
3. `PruneGrid(merged, retainMask)` removes the absorbed voxels and collapses emptied nodes.
4. One kernel per result leaf writes the values (section 4.4).
5. The invert masks of the result, one of:
   - (a) run `PropagateSign` on the result (what `track()` does today after `dilate()` and `prune()`);
   - (b) for each childless slot and each inactive leaf voxel of the result, evaluate
     `isInverted1(p) | isInverted2(p)` at one point `p` of the slot. One point suffices because the
     slot is uniform in the result (section 4.3). The root origin list needs the candidates: both input
     lists and the regions whose root-table entry was removed in step 3.

Route A adds two short kernels and no new topology operator. It builds the merged tree and then prunes
it, which is the cost route B removes.

### 6.2 Route B: native top-down construction

Build the result level by level from the root with the two mask formulas of section 4.2, creating only
the nodes that survive. This is one topology operator in the style of `MergeGrids` with the invert
masks as an additional input and output. Faster in principle; more code. To be considered after route A
is verified.

## 7. After the operation

`min` and `max` of two distance fields are not distance fields near the curve where the two surfaces
meet, and the band there can be narrower than `gamma`. The result is a valid sign field with correct
zero crossings, to be followed by `LevelSetTracker::track()` (or `normalize()` alone) when distances
are needed, as OpenVDB users do after its CSG operations.

## 8. Verification plan

Against `openvdb::tools::csgUnion` / `csgIntersection` / `csgDifference` on `FloatGrid`s converted from
the same inputs, comparing the full-space field of section 3:

1. Two overlapping spheres; two disjoint spheres; one sphere inside the other.
2. A case where one input's surface crosses the other's deep interior, so that leaves of one input lie
   under an `I` lower or upper slot of the other (the level mismatch of section 4.2).
3. A case large enough to involve root-level `I` regions in at least one input.
4. For each: active set identical, active values identical, sign of every inactive voxel and tile
   identical, and `PropagateSign` on the result reproduces the masks of step 5(b).

## 9. Open questions

1. Step 5 of route A: (a) recompute or (b) derive. (b) is the claim that the representation is closed
   under CSG; (a) is the safe first draft. Measure the difference before choosing.
2. Whether the result should be returned pruned to `|phi| < gamma` or as produced; with assumption 2 on
   the inputs the two coincide, but inputs taken straight from `MeshToSDF` may not satisfy it.
3. `MeshToSDF` and `PartitionGrid` are not on this branch. Merging `mesh-to-sdf-pr` conflicts in
   `PropagateSign.cuh`, whose two copies have diverged; examples that start from a mesh need that
   reconciled first.
4. N-ary union: `MergeGrids` is already N-ary and the table is associative, so the retain mask and the
   values generalize directly; not designed further here.
