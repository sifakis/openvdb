# Interface tracking on NanoVDB index grids: design notes

**Status:** requirements for the three steps of `track()` agreed; no implementation on this branch yet.
**Branch:** `nanovdb-interface-tracking` (fork `sifakis`), based on `origin/develop/nanovdb` `1ec456ef6`.

**Predecessor:** draft PR #2352 (`levelset-renormalization`), which implemented `normalize()` alone and
recorded its design in `nanovdb/nanovdb/examples/ex_renormalize_levelset_cuda/RENORMALIZATION_DESIGN.md`.
Section 8 lists which of its decisions carry over and which this design supersedes. "The reference" means
the CUDA benchmark imported by #2352 under that directory's `Benchmark/`.

---

## 1. Objective

A NanoVDB counterpart of `openvdb::tools::LevelSetTracker::track()`: the narrow-band maintenance cycle
that motion operators (advection, propagation, morphing, filtering) call after every time step. In
OpenVDB it is three steps:

    dilateActiveValues(leafs, 1, NN_FACE, IGNORE_TILES)   // grow the band by one face-connected layer
    normalize()                                           // renormalize the whole band
    prune()                                               // trim |phi| >= gamma, collapse emptied nodes

(`openvdb/openvdb/tools/LevelSetTracker.h:302-315`.)

The target is the same cycle on a `ValueOnIndex` grid plus sidecars, without the reference's
approximations: wherever a step needs a value outside the band, it reads the stored sign field rather
than inferring a sign from nearby active voxels.

**Out of scope here:** the motion operators themselves (section 9), velocity extension, a host path,
and splitting a level set whose topology changed (ConnectedComponents and PartitionGrid).

## 2. What OpenVDB gets for free

A `FloatGrid` level set stores a value at every point of space: active voxels hold distances, and every
inactive voxel and tile holds `+gamma` or `-gamma`. The sign field off the band is therefore part of the
tree, and the three steps never have to compute it:

| step | what OpenVDB does | values created |
|---|---|---|
| dilate | switches inactive voxels on; a new leaf is filled from the tile it replaces | none: the new voxels already hold `±gamma` |
| normalize | updates active values; the stencil reads `±gamma` through the accessor at inactive taps | active values only |
| prune | trim: `setValueOff(±gamma)` where `abs(phi) >= gamma`; then `pruneLevelSet` collapses emptied nodes into tiles of the right sign | none: signs are kept |

A `ValueOnIndex` grid has no values off the band. Every row above therefore needs something the tree
cannot provide, and that is the subject of this document.

## 3. The level-set state

All three steps consume and produce the same state, on the grid's own numbering:

| part | contents | indexed by |
|---|---|---|
| grid `G` | `ValueOnIndex` topology | |
| sidecar `phi` | slot 0: the background `gamma > 0`; slot `k > 0`: the value of the active voxel with index `k` | value index |
| invert masks `M` | one `Mask<3>` per leaf, `Mask<4>` per lower node, `Mask<5>` per upper node, and the origins of the upper-node-sized regions without a root-table entry whose bit is on | node index; root list sorted by `(x, y, z)` |

`M` is exactly what `tools::cuda::PropagateSign` (#2361) produces: a set bit means the inactive element
takes `-gamma` instead of `+gamma`. Bits of active voxels and of slots with a child are off.

### 3.1 The sign field and the value field

The state defines a value at every point `x`, which is the field OpenVDB stores:

    v(x) = phi[index(x)]                if x is an active voxel
         = (bit(x) ? -1 : +1) * gamma   otherwise

where `bit(x)` comes from the first level of a tree descent with no child at `x`: the leaf mask bit if
the leaf exists, else the lower slot bit, else the upper slot bit, else membership of `x`'s upper-node
region in the root list. `isInverted()` in `MeshToSDF.cuh` (#2417) implements this descent.

### 3.2 Invariants

- **I1. Structure:** no active tiles, the root holds only child tiles, and no node is empty (every leaf has
  an active voxel, every internal node a child). PropagateSign's preconditions 1 and 4; PruneGrid and
  DilateGrid output satisfies them.
- **I2. Band:** `abs(phi[k]) <= gamma` for every active voxel.
- **I3. Sign consistency:** each face-connected region of inactive voxels takes one sign, the sign of the
  active voxels it touches (zero counts as positive). This is what makes `M` current relative to `phi`;
  it is PropagateSign's preconditions 2 and 3, restated as a property of the output.
- **I4. Numbering:** `phi` and `M` are indexed by `G`'s numbering. A topology change renumbers every active
  voxel and every node, so all three are replaced together.

### 3.3 What each step must preserve

- **Dilation** preserves `v` exactly, everywhere: it only moves points from "inactive, value from `M`" to
  "active, value from `phi`".
- **Renormalization** changes active values only. `G` and `M` are unchanged.
- **Prune** preserves the sign field everywhere and clamps `v` to `[-gamma, gamma]`: a trimmed voxel moves
  from `phi` to `M` with its sign.

Each step returns a state satisfying I1-I4.

## 4. Step 1: dilation

OpenVDB's `dilateActiveValues` changes topology and values together. NanoVDB's `DilateGrid` changes
topology only, so the counterpart is:

1. `newGrid = DilateGrid(oldGrid)` with `NN_FACE`. DilateGrid's default is `NN_FACE_EDGE_VERTEX`, so the
   operation must be set explicitly.
2. Allocate `newPhi` with `newPhi[0] = gamma`.
3. Inject: every voxel active in both grids copies its value from `oldPhi`.
4. Fill: every voxel active in `newGrid` but not in `oldGrid` gets `v_old(x)`, i.e. `±gamma` with the sign
   read from the **old** grid's masks (section 3.1).
5. Produce `newM` (section 7.2).

Step 4 is the new part. The reference instead copies the sign of the first active old face-neighbor
(`Benchmark.cu:108-116`), and #2352 §3.3 records that this rule and the stencil's rule disagree.

Notes:

- Steps 3 and 4 are one pass over the new leaves, as in the reference's `DilateNarrowBandFunctor`: probe
  the old leaf once, copy where the voxel was active, fill from the old leaf's mask where it was not.
- If the old leaf does not exist, the whole new leaf lies in one childless old slot (or one absent root
  region), so all its new voxels take one sign: one descent per leaf, not per voxel.
- `track()` keeps `gamma`. OpenVDB's `dilate(n)` also grows the background by `dx` per shell; that variant
  writes the new `gamma` to slot 0 before filling (deferred, section 8.3).

## 5. Step 2: renormalization

#2352's `normalize()` (`tools/cuda/LevelSetTracker.cuh` on `levelset-renormalization`) is the base:

- `d(phi)/d(tau) = S(phi0) (1 - |grad phi|)` with `S(phi0) = phi0 / sqrt(phi0^2 + |grad phi|^2)`;
- `|grad phi|^2` by HJ-WENO5 and Godunov's scheme, in index space, rescaled by `1/dx`;
- TVD-RK2, `dt = 0.9 dx`, `normCount` steps, the ping-pong closing on `phi`;
- one thread per active voxel, decoded through the VoxelBlockManager; a 19-point gather.

**The change: taps outside the band read `v(x)`.** In #2352 a missing tap reads slot 0 and takes the sign
of the next tap inward along its axis, ring by ring. Instead it takes `±gamma` with the sign from `M`,
the value OpenVDB's accessor returns at the same tap. Consequences:

- One rule for the sign of out-of-band points, shared with step 1. #2352's B1 disappears.
- The background is passed to the kernel as a value. #2352 needs slot 0 to hold `gamma` in every buffer
  that can be a stencil source, the RK scratch included (its §3.8); that invariant goes away.
- OpenVDB becomes an exact oracle again, up to the WENO constants (#2352 C1/C2), since both stacks read
  the same field off the band. #2352 §6 had given this up.
- The step needs the masks of the dilated grid, which is why step 1 must produce them.

**Where the tap's sign comes from:** a tap in an existing leaf (the voxel's own, or the neighbor leaf the
gather already probes) reads that leaf's mask bit, which needs the neighbor's leaf index. A tap whose
leaf does not exist takes the coarser bit. Taps reach at most 3 voxels, so a voxel has at most one
neighbor leaf per axis side, and those signs can be resolved once per leaf.

`G` and `M` are unchanged. Renormalization is not supposed to change the sign of any active voxel; see
section 9.2 for when it does.

## 6. Step 3: prune

1. Retain mask: keep active voxels with `abs(phi) < threshold`. Strict, as OpenVDB's trim (`>= gamma` is
   removed) and the reference's `PruneNarrowBandFunctor`.
2. `PruneGrid` with the mask.
3. Inject `phi` into the pruned grid (`util::cuda::InjectGridDataFunctor`, which pairs leaves by origin
   and remaps slots by popcount rank).
4. Produce `newM` (section 7.2).

Notes:

- **The threshold is an argument, not slot 0.** For `track()` it is `gamma`. #2352's offset workflow
  (§3.7) prunes while slot 0 holds `gamma + D`, and the reference's `pruneNarrowBand` takes a background
  argument but reads `phi[0]` instead.
- **Collapsed nodes take one sign.** A leaf whose active voxels are all trimmed disappears, and its region
  becomes one lower slot bit; likewise for emptied lower and upper nodes, and an emptied upper node adds
  its region to the root list if negative. One bit is right because, under I2 and I3, a region with no
  remaining active voxel contains no zero crossing. OpenVDB's `pruneLevelSet` assumes the same.
- **An empty result** (everything trimmed) is a valid state: `G` has no leaves, and the sign field lives
  in the root list alone.

## 7. Cross-cutting

### 7.1 The sign lookup

Steps 1 and 2 both evaluate `bit(x)` on a grid and its masks. One device function, with a per-leaf form
for the cases where a whole leaf or a whole face neighbor shares one answer. The root-level case is a
binary search in the sorted root list.

### 7.2 Producing the new masks

Two options, for steps 1 and 3 alike:

- **Derive them from the old masks and the step's own changes.** Dilation and prune change the sign field
  nowhere, so every bit of `newM` is known: a new leaf inherits its old leaf's mask, or the old coarser bit
  for all its voxels, minus the newly active voxels; trimmed voxels set their bit from their sign;
  collapsed nodes become one coarser bit; the root list loses regions that gained an entry, and gains
  emptied negative ones. Local, exact, no flood fill.
- **Recompute them with PropagateSign** on `(newG, newPhi)`. Both outputs meet its preconditions, but it
  pays for full flood fills to recover what is already known.

Leaning: derive in the operator, use PropagateSign as the test oracle. Not decided.

### 7.3 Ownership

#2352 §3.5 makes the tracker own the grid, sidecar, RK scratch and VoxelBlockManager, because all four
depend on one numbering and silently go wrong when out of sync. The masks and the root list join that
set. The semantic-grid bundle (`nanovdb/nanovdb/docs/SemanticGrid-design.md`, branch
`semantic-grid-design`, section 11) is the natural carrier for grid, `phi`, `M` and `gamma`; this is
#2352's open question A4, now with a concrete candidate.

### 7.4 The precondition between calls

Motion between two `track()` calls changes active values but not `M`. I3 then holds only if no voxel at
the edge of the band changes sign, i.e. the interface moves less than the band half-width per step,
which the CFL condition guarantees. OpenVDB makes the same assumption implicitly: advection never
updates inactive values either.

## 8. Relation to #2352

### 8.1 Carried over

- `ValueOnIndex` grid plus sidecar, slot 0 as the background (§3.1).
- HJ-WENO5 and TVD-RK2 only, other schemes unnameable, enum values pinned to OpenVDB's
  (`nanovdb/math/FiniteDifference.h`, §3.4).
- The tracker owns its state, moved in and released (§3.5), now including the masks.
- No `LeafManager`: sequential leaves asserted (`isSequential<0>()`), isotropic voxels checked, RK
  temporaries as typed sidecars, no buffer-swap protocol, scratch count derived from the scheme (§3.8).
- Interrupter, `grainSize` and the `TrimMode` enum dropped (§3.7).
- The static `WenoStencil::normSqGrad` and `gradient` overloads over a caller buffer (commit `6f99ddff6`,
  not yet upstream).
- CUDA only.

### 8.2 Superseded

| #2352 | now |
|---|---|
| §3.2: no sign information outside the band | the invert masks are part of the state |
| §3.3 / B1: two sign-extrapolation rules to unify | one lookup into `M` (section 7.1) |
| §3.8: slot 0 holds `gamma` in every stencil-source buffer | `gamma` passed to the kernel |
| §6: OpenVDB no longer an exact oracle | exact again, up to C1/C2 |
| `pruneNarrowBand` reads the threshold from slot 0 | threshold argument |

### 8.3 Still open, unchanged

- A2: schemes as runtime enums or compile-time tags.
- B2: where `Log2BlockWidth` lives. B4: reuse the `gatherIndices` refactor (`vbm-cpu-port`). B5:
  validation cases. B7: a VoxelBlockManager over a mask. B8: debug check of I2.
- C1/C2: the WENO5 epsilon (`scale2`) and working precision.
- Deferred operations (§3.7): `dilate(n)` with band growth, `erode`/`resize`, asymmetric prune,
  masked `normalize`, `offset`, `changeLevelSetBackground`.
- The VoxelBlockManager API changed since #2352 (#2291: per-slot `decodeInverseMap`, batched builds);
  #2352's `LevelSetTracker.cuh` already uses the per-slot decode.

## 9. Beyond `track()`

### 9.1 The motion operators

Advection (`V . grad phi`, OpenVDB `LevelSetAdvection`) and normal-speed propagation (`F |grad phi|`,
the reference's `LevelSetPropagate`) call `track()` after every CFL step; OpenVDB's CFL factor is
`{0.3, 0.9, 1.0} / sqrt(3)` for TVD-RK1/2/3, against `max |V|` or `max |F|`. Not designed here. Two
points in the reference's propagation step to decide when they are, both inherited from OpenVDB's
`LevelSetMorphing`:

- Godunov's scheme is upwinded on the sign of `phi` (`GodunovsNormSqrd(v[0] > iso, ...)`), which is right
  for renormalization, whose speed is `S(phi)`. For `F |grad phi|` the upwind side follows the sign of
  `F`; the two differ inside the surface where `F > 0` and outside where `F < 0`.
- The update multiplies `F` by `|grad phi|^2` rather than `|grad phi|`; they agree only while `phi` stays a
  signed distance function.

The out-of-band taps of these stencils read `v(x)` as in step 2.

### 9.2 Renormalization and inclusion

`NewDistancingApproach.md` (branch `new-distancing-approach`), hazard 3: numerically, renormalization
can move the zero crossing it is meant to keep. For tracking this matters twice: it can break inclusion,
and if an active voxel changes sign, I3 can fail where that voxel meets inactive voxels. Position to be
decided.

### 9.3 The root level

PropagateSign returns the negative root regions as a list of origins without root-table entries. The
sanitized convention discussed for the semantic grid instead inserts interior root tiles into the root
table, which conflicts with PropagateSign's precondition 4. Steps 1 and 3 create and remove root
entries, so the choice affects them directly.

## 10. Open questions

1. Producing the new masks: derive or recompute (section 7.2).
2. The carrier of the state: the semantic-grid bundle, or a tracker-local struct first (section 7.3).
3. The root-level representation (section 9.3).
4. A position on renormalization moving the interface (section 9.2).
5. #2352's open items in section 8.3.

## 11. References

| what | where |
|---|---|
| OpenVDB `track()`, `prune()`, `dilate()` | `openvdb/openvdb/tools/LevelSetTracker.h:279-337` |
| OpenVDB trim | `openvdb/openvdb/tools/LevelSetTracker.h:473-506` |
| OpenVDB advection time loop and CFL | `openvdb/openvdb/tools/LevelSetAdvect.h:341-470` |
| #2352 `normalize()` | `nanovdb/nanovdb/tools/cuda/LevelSetTracker.cuh` (branch `levelset-renormalization`) |
| #2352 design | `nanovdb/nanovdb/examples/ex_renormalize_levelset_cuda/RENORMALIZATION_DESIGN.md` (same branch) |
| reference dilation and sign seeding | `.../Benchmark/src/Benchmark.cu:75-168` (same branch) |
| reference prune | `.../Benchmark/src/Benchmark.cu:469-555` (same branch) |
| reference propagation step | `.../Benchmark/src/Benchmark.cu:760-974` (same branch) |
| invert masks | `nanovdb/nanovdb/tools/cuda/PropagateSign.cuh` (#2361) |
| sign lookup | `isInverted()` in `nanovdb/nanovdb/tools/cuda/MeshToSDF.cuh` (#2417) |
| sidecar injection | `nanovdb/nanovdb/util/cuda/Injection.cuh`, `InjectGridDataFunctor` |
| state bundle | `nanovdb/nanovdb/docs/SemanticGrid-design.md` (branch `semantic-grid-design`) |
