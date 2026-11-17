# CSG on NanoVDB index-grid level sets: design notes

**Status:** design only; nothing here is implemented or tested. Every claim marked *argued* is a
paper argument that the first implementation must check against OpenVDB.
**Branch:** `nanovdb-csg` (fork `sifakis`), based on `nanovdb-interface-tracking` `7dbda91f3`.
**Origin:** whiteboard discussion of 2026-11-17 (the case table of section 5.1 and the conjecture
that the invert masks of a union are the OR of the input masks), followed by a design discussion between
Efty Sifakis and Claude the same day. Section 3 (signed background) and sections 6 to 8 are the outcome of
that discussion; JaeHyun's sections 5.2 to 5.4 and 9 are kept and revised where it changed them.
**Picture:** an interactive illustration of the node-existence rule of section 8 (1-D tree, two inputs,
the union row, click a node to see why it exists):
<https://claude.ai/artifact/B2xdqqGVHxwputZ91WUpDY>. The page is private to its owner (Efty) until shared.

**Depends on:** the level-set state and the sign lookup of `nanovdb/nanovdb/docs/InterfaceTracking-design.md` (sections 3
and 7.1 there), `tools::cuda::PropagateSign`, `tools::cuda::MergeGrids`, `tools::cuda::PruneGrid`, and
`isInverted()` in `tools/cuda/LevelSetTracker.cuh`.

---

## 1. Objective

Union, intersection and difference of two narrow-band level sets stored as `ValueOnIndex` grids, each
with its sidecar and invert masks, producing a level set in the same form. The counterpart of
`openvdb::tools::csgUnion`, `csgIntersection` and `csgDifference`. Complement is also supported, as a
trivial operation on the representation (section 3).

The question this document answers: the two inputs have different topology, so their invert masks are
indexed by different nodes and may record the sign of the same region at different tree levels. Can the
result be computed inside the representation, without running `PropagateSign` again on the result?

**Out of scope for the first version:** inputs with different transforms, more than two inputs, and
the renormalization of the result (section 11). Inputs with different `|gamma|` are handled by
preprocessing (section 2, assumption 3), not by the core algorithm.

## 2. Inputs and assumptions

Each input is the state of `InterfaceTracking-design.md` section 3: grid `G`, sidecar `phi` with the
signed background `gamma` in slot 0 (section 3), and invert masks `M` (leaf, lower, upper, and the
sorted root origin list). Subscripts 1 and 2 name the two inputs.

Assumptions:

1. Both grids have the same affine transform: the index spaces are covariant, so voxel `(i,j,k)` is the
   same point of space in both and their nodes cover identical regions. Decided; not negotiable.
2. Every active value satisfies `|phi| < |gamma|` (the state after `prune()`).
3. Both grids have the same `|gamma|`. The sign of `gamma` may differ (section 3). Different magnitudes
   reduce to this case: the input with the wider band is first pruned to `|phi| < gamma_min`, which turns
   its extra voxels into inactive ones with the sign of their `phi`, and the result takes
   `|gamma_out| = gamma_min`. The minimum is forced, not a convenience: beyond the narrower band that
   input says nothing about the sign of the other's active values, so no wider result is determined.
   The pruning keeps the `PropagateSign` preconditions as long as the band is a couple of voxels wide
   (*argued*). Not designed further here.
4. Each input satisfies the preconditions of `PropagateSign`, generalized to the sign of `gamma`: no
   empty nodes; each face-connected region of inactive voxels touches active voxels of one sign only;
   the unbounded region has the sign of `gamma` (positive if `gamma > 0`); no active tiles and no root
   value tiles.

## 3. The full-space field and the signed background

### 3.1 Convention

The grid stores a signed background `gamma`, in slot 0 of the sidecar. It is the value the implicit field
takes at infinity, so its sign says which side of the surface the unbounded region is on. `|gamma|` is
also the saturation level of the band.

The invert bit means "the sign here is the opposite of the background". The masked region `M`, where the
bits are on, is therefore always bounded, whatever the sign of `gamma`. The sign of `gamma` is the only
polarity information; no separate flag is needed.

An index-grid level set with invert masks defines a value at every point of index space:

    Phi(p) =  phi[index(p)]   if p is an active voxel
              -gamma          if the inactive element holding p has its invert bit on
              +gamma          otherwise

with `gamma` now signed. For `gamma > 0` this is the field of a bounded body whose interior is `M`.

### 3.2 Complement

The complement of a level set negates the field: negate every sidecar value and `gamma`, and leave the
masks exactly as they are. The masked region `M` is unchanged; a bounded body becomes an unbounded one
and back. Complementing twice is the identity. The stored complement of a sphere is therefore
representable with the same bounded masks (a sphere interior stays `M`; the complement has background
`-G` and `M` has value `+G`).

### 3.3 Vocabulary

The operations are pointwise on `Phi`, exactly as in OpenVDB:

    union          Phi = min(Phi1, Phi2)
    intersection   Phi = max(Phi1, Phi2)
    difference     Phi = max(Phi1, -Phi2)

To keep the geometry apart from the storage convention, this document uses three states for any element
of the tree, at any level, in addition to the stored `active`/`inverted` predicates (`G` stands for
`|gamma|`, the shared band half-width):

| state | meaning | `Phi` on the element |
|---|---|---|
| `out` | inactive, outside | `+G` everywhere |
| `in` | inactive, inside | `-G` everywhere |
| `act` | has a child (at the leaf level: the voxel is active) | varies; described one level down |

The stored bit maps to the state through the sign of that input's `gamma`, with `neg = (gamma < 0)`:

    in_i(p)  =  not active_i(p)  and  ( inverted_i(p)  XOR  neg_i )

so for `gamma > 0` a set bit is `in` and a clear bit is `out`, and for `gamma < 0` the two exchange.

## 4. What the operation does to the background and to `M`

At infinity each field is its background, so the result background is the operation applied to
`gamma1` and `gamma2`. With equal magnitudes only the signs matter:

| `g1` | `g2` | union | intersection | difference `1 \ 2` |
|---|---|---|---|---|
| `+` | `+` | `+` | `+` | `+` |
| `+` | `-` | `-` | `+` | `+` |
| `-` | `+` | `-` | `+` | `-` |
| `-` | `-` | `-` | `-` | `+` |

Rules of thumb: an intersection is bounded (positive background) if either input is. A union is
unbounded if either input is. Difference is intersection with the complement of input 2.

The result's masked region, in terms of the inputs' `M1` and `M2` (ignoring the narrow band):

| `g1` | `g2` | union | intersection | difference `1 \ 2` |
|---|---|---|---|---|
| `+` | `+` | `M1 or M2` | `M1 and M2` | `M1 minus M2` |
| `+` | `-` | `M2 minus M1` | `M1 minus M2` | `M1 and M2` |
| `-` | `+` | `M1 minus M2` | `M2 minus M1` | `M1 or M2` |
| `-` | `-` | `M1 and M2` | `M1 or M2` | `M2 minus M1` |

(`or`, `and` and `minus` are set union, intersection and difference.) Every entry is bounded, so the operations
are closed in the representation: the result is again a bounded masked region plus a signed background.
Consequences:

- The mask formula `I_out = I1 | I2` of the earlier version of this document holds only in the
  `(+,+)` union entry. In general the combining operation depends on the sign pair.
- Complement and De Morgan relate the entries: `1 | 2 = ~(~1 & ~2)`, `1 \ 2 = 1 & ~2`. A test can check
  union against intersection by complementing the inputs and the result.
- The torus (sphere minus an inner sphere) is the difference of two positive-gamma inputs. It does not
  need a stored complement.

## 5. Union

### 5.1 The per-voxel case table

In `in`/`out`/`act` terms (section 3.3), with `G = |gamma|`, per voxel, independent of the level at which an input stores the
voxel:

| input 1 | input 2 | result | value |
|---|---|---|---|
| `out` | `out` | `out` | `+G` |
| `out` | `in` | `in` | `-G` |
| `in` | `out` | `in` | `-G` |
| `in` | `in` | `in` | `-G` |
| `out` | `act` | `act` | `phi2` |
| `act` | `out` | `act` | `phi1` |
| `in` | `act` | `in` | `-G` (input 2's detail discarded) |
| `act` | `in` | `in` | `-G` (input 1's detail discarded) |
| `act` | `act` | `act` | `min(phi1, phi2)` |

`-G` is below every value and `+G` is above every active value (assumption 2), so `in`
absorbs and `out` is the identity of `min`. Equivalently, union is a maximum over the order
`out < act < in`: any `in` gives `in`, otherwise any `act` gives `act` (with value `min` of the active
`phi`), otherwise `out`.

### 5.2 Refactored rule: one table, conjugated by the signs

All four sign pairs follow from this table by reading and writing through the sign of `gamma`:

1. Read: each input slot is `act` if it has a child, otherwise `in` if `bit XOR neg_i`, else `out`.
2. Combine: as in 5.1.
3. Write: the result background has `neg_out = neg1 OR neg2`; the result bit of an inactive element is
   `isIn XOR neg_out`; `act` elements carry no bit.

In mask form, for the slots that have no child (`IN_i` is only meaningful off the child mask `C_i`):

    IN_i   = (I_i XOR (neg_i ? ~0 : 0)) & ~C_i
    IN_out = IN_1 | IN_2
    I_out  = IN_out XOR (neg_out ? ~0 : 0)

For `(+,+)` this is the plain `I_out = I1 | I2`. The full tables, per stored state, are in Appendix A.

### 5.3 Values

Wherever the result is `act` the value is the `min` of the active inputs' `phi`; an input that is not
`act` there is `out` and contributes `+G`, which never wins.

**Correction to the earlier single formula** `phi_out = min(phi1[index1(p)], phi2[index2(p)])`, which
relied on slot 0 holding `+gamma` so that an inactive voxel reads as `out`. That is true only when `out`
is the unmasked state, i.e. `gamma > 0`. For an input with `gamma < 0` an `out` voxel is a masked one and
slot 0 holds `-G`; the value kernel must take the `out` side from the state, not from index 0. In
every row where the result is `in` the value is `-G` regardless.

### 5.4 Valid level-set state (*argued*)

Claim: in the result, two face-adjacent inactive voxels never have different signs, and the result
satisfies the preconditions of `PropagateSign`.

Argument, for `(+,+)`. Let `p` and `q` be face-adjacent in the result with `q` in state `in`. Then `q` is
inactive and negative in one input, say input 1. In input 1, `p` cannot be inactive and positive, because
`p` and `q` would be one face-connected inactive region with two signs. So `p` is either `in` in input 1,
which makes it `in` in the result, or active and negative in input 1 (precondition 2 applied to the
region of `q`), which makes the result at `p` either `in` or active with a value `<= phi1(p) < 0`. Hence
an `in` voxel of the result has only `in` or active-negative face neighbours. The symmetric statement for
`out` follows the same way. The other sign pairs follow by complement (section 3.2).

Consequences:

- A leaf of the result with no active voxel left is uniformly `out` or uniformly `in` (a leaf is
  face-connected), so it collapses into one bit of its parent. The same holds one level up. The result
  therefore never needs an empty node.
- The unbounded region keeps the sign of `gamma_out`: a region is masked in the result only where the
  masked-region table of section 4 says so.

## 6. Intersection and difference

Intersection is the same structure with the roles of `in` and `out` exchanged: `out` absorbs, `in` is
the identity, the order is `in < act < out`, active values combine with `max`, and
`neg_out = neg1 AND neg2`.

Difference `1 \ 2` is the intersection of input 1 with the complement of input 2. The complement is
never materialized: read input 2 with `neg_2` flipped and its active `phi` negated (section 3.2), then
intersect. Then `neg_out = neg1 AND (NOT neg2)`, matching the last column of the table in section 4.

| slot 1 | slot 2 (as read, complemented) | result |
|---|---|---|
| `out` | any | `out` |
| any | `out` | `out` |
| `in` | `act` | `act`, values `phi2'` (the negated `phi2`) |
| `act` | `in` | `act`, values `phi1` |
| `act` | `act` | `act`, `max(phi1, phi2')` |
| `in` | `in` | `in` |

Intersection and difference are only sketched here; the per-voxel tables and the existence condition of
section 8 follow by the symmetry above and must be checked in the verification of section 12. The
difference is kept as its own entry point for the first version (it is not just intersection with a
flag), even though it shares the kernel.

## 7. The tree levels

### 7.1 Level-independent state

At any level (leaf, lower, upper, root) an element is `act` if it has a child, otherwise `in` or `out`
by section 3.3. Each stored node `n` gives two masks:

- `C(n)`, the child mask (the value mask at leaf level);
- `IN(n) = (I(n) XOR neg) & ~C(n)`, the elements that are inside.

### 7.2 Virtual nodes

Where input 1 has a node and input 2 does not, input 2 holds that region uniformly (as `in` or `out`,
inherited from the slot above). Treat it as a virtual node with `C = 0` and `IN` all ones or all zeros.
With virtual nodes both inputs always have a node at the same position, and the same formulas apply at
every level. The level mismatch is gone from the formulas and only appears in the traversal that carries
the inherited state down for whichever input ran out of tree.

At a slot with a child in one input only:

- if the other input's virtual state is `out`, the result is that input's subtree, copied unchanged;
  the exception is `neg_out != neg_i`, when every inactive bit in the copied subtree must be flipped (all
  levels), because the encoding is relative to each grid's own background; sidecar values do not change;
- if it is `in`, the slot becomes `in` and the subtree is dropped.

Where both inputs have children, recurse.

### 7.3 Leaf-only reference

If every inside region were resolved to the leaf level, union would be `MergeGrids` with a per-leaf
bitwise step: every leaf present in either input is present in the result, an absent leaf is the
background, and per leaf `IN_out = IN1 | IN2`, `C_out = (C1 | C2) & ~IN_out`, the active values being the
`min`. The multi-level algorithm is this algorithm run lazily on a compressed representation: a virtual
node is a deferred expansion of a uniform slot, the one-sided copy-or-drop is the shortcut where
expanding would change nothing, and the collapse of section 5.4 is re-coarsening. Resolving everything to
leaves costs memory proportional to the inside volume, which is what the coarser levels avoid.

It is still the right oracle: expand both inputs to leaf-level masks, combine with the bitwise formulas,
and compare the full-space field with the multi-level output.

## 8. Which nodes exist, and which slots are interior

The two questions are independent and each has its own condition.

### 8.1 Existence of a node (leaf, lower or upper)

**A node exists in the result if and only if some voxel in its extent is active in the result.**
For union with `gamma1, gamma2 > 0`, a voxel is active in the result iff

    ( active1(p) or active2(p) )  and  not inverted1(p)  and  not inverted2(p)

and for general signs, with `in_i` of section 3.3,

    ( active1(p) or active2(p) )  and  not in1(p)  and  not in2(p).

In plain terms, a voxel is active in the union if it is active in both inputs, or active in one and
exterior in the other. Sufficient: such a voxel needs a node (and so do its ancestors). Necessary: the
result has no empty node (section 5.4), so a node without such a voxel has no reason to exist.

What does not force a node: an interior voxel (it is absorbed and leaves only a bit), and an active voxel
of one input that lies in the interior of the other. Where both inputs have nodes and every active voxel
of each is absorbed by the other (*mutual absorption*), the result has no node although both inputs do.

The condition factors, which makes a one-shot build direct: a leaf exists iff its merged mask
`(C1 | C2) & ~(IN1 | IN2)` is nonzero (only leaves present in at least one input are candidates, an absent
leaf counting as background); a lower node exists iff one of its leaves exists; an upper node iff one of
its lower nodes exists. The result topology is the set of non-empty leaves and their ancestors, built
without creating nodes and pruning them afterwards. `MergeGrids` creates a node wherever any input has
one, which is the superset; the filter by non-emptiness is what it lacks.

### 8.2 Interior slots (the masks)

The invert state of a slot without a child in the result, at any level, is separate from existence. Each
level is covered uniformly by:

    C_out  = 1 iff the node under the slot exists (8.1)
    IN_out = ( IN1 | IN2 | C1 | C2 ) & ~C_out          (*argued*, for gamma1, gamma2 > 0)

That is, a childless slot of the result is interior iff it is interior in either input, or either input
has a child there (a node that vanished in the result must have had all its active voxels absorbed by an
interior region, hence the whole region is interior). A slot that is exterior in both inputs and has no
child in either is exterior. A deep-exterior and a deep-interior region fail the existence test alike;
only the bit tells them apart, in the deepest existing ancestor or in the root origin list.

### 8.3 The root level

The rule of 8.1 applies to root entries that have an upper node beneath them. A childless root tile, an
entry of the root origin list, is by our convention an upper-node-sized region that is uniformly
interior. It exists without active voxels, and is a mask state, so it belongs to 8.2, not to 8.1.

For union with `gamma1, gamma2 > 0` the result's origin list is the set union of

- the origin lists of the two inputs, and
- the origins of upper nodes of either input that has no upper node in the result (*argued*: such a
  region is uniform, and holds an interior voxel that absorbed the active ones, hence is interior).

A tile that is exterior in both inputs, and so on neither list, stays exterior: the union never creates a
cavity tile out of nothing. A sorted merge of the two lists plus the vanished entries is therefore enough;
no flood is required.

### 8.4 Using stage 4 of PropagateSign instead

Stage 4 of `PropagateSign` computes the root origin list as the interior regions without a root-table
entry. It cannot be called alone: its input is the set of root face seeds written by stages 1 to 3, and
it throws if the cavity escapes the hull of the root entries. To use it on the result one would first
compute those seeds from the result's masks (a tile across a face of an upper node is interior iff an
inactive voxel of that boundary layer is inverted, by precondition 2) with a kernel that duplicates what
`seedAcrossFace` does. The derivation of 8.3 avoids that, and is the recommended route for union. The full
`PropagateSign` stays useful as a test oracle for the masks and the root list.

## 9. Implementation routes

### 9.1 Route A: existing operators (first draft)

1. `MergeGrids(G1, G2)` gives the topology `C1 | C2` at every level.
2. One kernel per merged leaf builds the retain mask: a voxel is retained iff it is active in the merged
   grid and `in_i` is false for it in both inputs. `in_i` descends each input's own tree, so it reads the
   bit at whichever level that input stores it; the topology mismatch is confined to this lookup.
3. `PruneGrid(merged, retainMask)` removes the absorbed voxels and collapses emptied nodes (8.1).
4. One kernel per result leaf writes the values (5.3).
5. The invert masks of the result, one of:
   - (a) run `PropagateSign` on the result (what `track()` does today after `dilate()` and `prune()`);
   - (b) derive them: for each childless slot and each inactive leaf voxel of the result, evaluate
     `isInverted1(p) | isInverted2(p)` (generalized by the signs, 5.2) at one point `p` of the slot,
     one point sufficing because the slot is uniform in the result (5.4); the root origin list from 8.3;
   - (c) if the inputs' masks are first resolved to leaf level (7.3), the result's inside set is the merged
     `IN` masks directly, with no point sampling.

Route A adds two short kernels and no new topology operator. It builds the merged tree and then prunes
it, which is the cost route B removes.

### 9.2 Route B: native top-down construction

Build the result in one pass: compute leaf-level non-emptiness first (8.1), construct only the nodes that
survive, and fill the masks by 8.2 and 8.3. This is one topology operator in the style of `MergeGrids`
with the invert masks as an additional input and output. Faster in principle; more code. To be considered
after route A is verified. The sign handling of 5.2 and the copy-with-flip of 7.2 apply here.

## 10. Preconditions to check at the entry point

- Same transform (assumption 1); same `|gamma|` or pre-pruned (assumption 3).
- Each input's `|phi| < |gamma|` on active voxels, no empty nodes, no active or root value tiles.
- The generalized `PropagateSign` preconditions (assumption 4), the sign of `gamma` fixing which side is
  unbounded. `PropagateSign` and `isInverted()` need the same generalization to the sign of `gamma`.

## 11. After the operation

`min` and `max` of two distance fields are not distance fields near the curve where the two surfaces
meet, and the band there can be narrower than `|gamma|`. The result is a valid sign field with correct
zero crossings, to be followed by `LevelSetTracker::track()` (or `normalize()` alone) when distances
are needed, as OpenVDB users do after its CSG operations.

## 12. Verification plan

Against `openvdb::tools::csgUnion` / `csgIntersection` / `csgDifference` on `FloatGrid`s converted from
the same inputs, comparing the full-space field of section 3.1:

1. Two overlapping spheres; two disjoint spheres; one sphere inside the other.
2. A case where one input's surface crosses the other's deep interior, so that leaves of one input lie
   under an `in` lower or upper slot of the other (the level mismatch of section 7).
3. A case large enough to involve root-level interior tiles in at least one input.
4. Mutual absorption: a node present in both inputs, each with active voxels absorbed by the other, so the
   result has no node there (section 8.1).
5. Mixed signs: a stored complement of a sphere against a sphere, for all three operations (section 4),
   and the torus as the difference of two positive-gamma inputs.
6. Identities: `1 | 2 = ~(~1 & ~2)`, `1 \ 2 = 1 & ~2`, double complement, and union of equal inputs
   equals the input.
7. For each: active set identical, active values identical, sign of every inactive voxel and tile
   identical; the node set equals the existence condition of 8.1; `PropagateSign` on the result reproduces
   the derived masks of 8.2 and the root list of 8.3; and the leaf-resolved reference of section 7.3
   agrees with the multi-level output.

## 13. Open questions

1. Step 5 of route A: (a) recompute, (b) derive, or (c) leaf-resolved masks. (b) is the claim that the
   representation is closed under CSG; (a) is the safe first draft. Measure the difference before choosing.
2. Whether the result should be returned pruned to `|phi| < |gamma|` or as produced; with assumption 2 on
   the inputs the two coincide, but inputs taken straight from `MeshToSDF` may not satisfy it.
3. `MeshToSDF` and `PartitionGrid` are not on this branch. Merging `mesh-to-sdf-pr` conflicts in
   `PropagateSign.cuh`, whose two copies have diverged; examples that start from a mesh need that
   reconciled first.
4. N-ary union: `MergeGrids` is already N-ary and the table is associative, so the retain mask and the
   values generalize directly; not designed further here.
5. Generalizing `PropagateSign`, `isInverted()` and the root origin list to a negative `gamma`
   (assumption 4): the stored-complement case needs it, the torus (difference of positive inputs) does not.
6. The existence condition (8.1), the interior rule (8.2) and the root list derivation (8.3) are stated for
   union with positive backgrounds. Intersection, difference and mixed signs follow by the symmetries of
   sections 4 to 6 but still need the same argument, written out and tested.
7. Whether the copy-with-flip of 7.2 can be avoided by choosing `neg_out` equal to the sign of the
   larger input subtree, or whether a bit-flip pass is cheap enough.

## Appendix A. Union, per stored state

Result per input storage state (`act` active, `non` inactive with the bit off, `inv` inactive with the
bit on), against the result's own background. `val` is the value where the result is `act`.

`g1 > 0`, `g2 > 0` (`g_out > 0`):

| in 1 \ in 2 | `act` | `non` | `inv` |
|---|---|---|---|
| `act` | `act`, `min(phi1, phi2)` | `act`, `phi1` | `inv` |
| `non` | `act`, `phi2` | `non` | `inv` |
| `inv` | `inv` | `inv` | `inv` |

`g1 > 0`, `g2 < 0` (`g_out < 0`); the case `g1 < 0`, `g2 > 0` is the transpose:

| in 1 \ in 2 | `act` | `non` | `inv` |
|---|---|---|---|
| `act` | `act`, `min(phi1, phi2)` | `non` | `act`, `phi1` |
| `non` | `act`, `phi2` | `non` | `inv` |
| `inv` | `non` | `non` | `non` |

`g1 < 0`, `g2 < 0` (`g_out < 0`):

| in 1 \ in 2 | `act` | `non` | `inv` |
|---|---|---|---|
| `act` | `act`, `min(phi1, phi2)` | `non` | `act`, `phi1` |
| `non` | `non` | `non` | `non` |
| `inv` | `act`, `phi2` | `non` | `inv` |

Cross-checks against section 4: `(+,+)` gives `inv` if either input is `inv` (`M1 or M2`); `(-,-)` gives
`inv` only if both are (`M1 and M2`); `(+,-)` gives `inv` only for `non`/`inv` (`M2 minus M1`).
