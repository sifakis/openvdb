# The ball-certification sweep on the GPU: design

> Working design from the October–November 2026 design review (branch `new-distancing-approach`).
> The sweep (sections 1–12) is implemented as `tools::cuda::SignCertifier`
> (`nanovdb/tools/cuda/SignCertifier.cuh`); the inside-magnitude pass (section 14) is not yet. The theory it relies on is in `NewDistancingApproach.md`: Step 2 (the closure),
> "Stencil reach of the closure sweep" (how far a voxel must look), and "Overlap tolerance of the
> certification test". The experiments that will validate it are listed there under "Queued
> certification experiments". The current implementation, for comparison, is `BallCertifyFunctor`
> and `signBarrierWithBalls` in `nanovdb/tools/cuda/MeshToSDF.cuh` on branch `mesh-to-sdf`.

## 1. What the sweep computes

- **Input:** one connected component of the narrow-band UDF, on its own index grid (today's pipeline
  already prunes each surface into its own grid before signing). Values are exact distances to the
  mesh, in band `udf ≤ W`.
- **Seeds:** the 6-connected component of non-barrier voxels (`udf ≥ D`, with `D = 0.55`)
  containing the voxel with the smallest coordinate along some axis. Barrier voxels and
  non-barrier components matter only here; the certification itself treats every active voxel
  alike. (Reusing the seeding labels to promote whole components is an acceleration: not in
  version 1, planned for later; section 10.)
- **Two independent options.** `pedantic` changes *which* voxels end up certified: off, the sweep
  uses the `W + D` stencil only; on, it alternates the sweep with a `2W` shell pass until neither
  adds anything, which yields exactly the reachability set of the ball method (section 9).
  `promote` changes only *how fast* the fixpoint is reached (section 10). Pedantic mode turns
  promotion on by default.
- **Rule:** certify `q` from a certified `n = q + w` iff `r_n + r_q > |w| + τ`, where `r = udf`
  (voxel units) and `τ` is the overlap tolerance.
- **Outside-only.** Certification only ever adds voxels, so the result is the set reachable from the
  seeds in the ball-overlap graph, whatever the sweep order. This is what makes in-place updates,
  racy cross-block reads and early exits safe. Unlike today's kernel, nothing certifies "inside";
  every voxel left uncertified is inside.
- **Candidates:** every active, uncertified voxel. Barrier voxels are most of them, but non-barrier
  voxels must be included, because a non-barrier pocket is unlocked only when one of its own voxels
  is certified from a barrier voxel in the neck.

## 2. Stencil

- **Offsets:** primitive integer vectors with `|w| < W + D`. Multiples on a line are never needed,
  for any surface (`NewDistancingApproach.md`). At `W = 3`, `D = 0.55` that is `|w|² ≤ 11`:
  **146 taps**.
- **Halo `H`:** the stencil must fit in `[-H, H]³`. The box suffices iff
  `W + D ≤ √((H + 1)² + 1)`; beyond that, a clipped stencil loses only sub-voxel bubbles, never
  soundness.
- **Cap `H ≤ 8`**, the largest halo whose tile stays inside the 3×3×3 block of leaves around the
  centre leaf.

| `H` | tile | shared memory (float) | load overhead | max `W` (`D = 0.55`) | stored taps | resident 512-thread CTAs per SM |
|---|---|---|---|---|---|---|
| 3 | 14³ | 11 KB | 5.4× | 3.57 | 218 | 3 (warp limit) |
| 4 | 16³ | 16 KB | 8× | 4.55 | 410 | 3 (warp limit) |
| 5 | 18³ | 23 KB | 11× | 5.53 | 770 | 3 (warp limit) |
| 8 | 24³ | 55 KB | 27× | 8.5 | 2,546 | 1 (shared memory) |

"Stored taps" counts the primitive offsets in the box with `|w|² < (H + 1)² + 1`, the most any run
with that `H` can use. Residency assumes about 100 KB of shared memory and 48 warps per SM (sm_120);
every row up to `H = 5` keeps all 48 warp slots busy. Compile `H ∈ {3, 4, 5, 8}` and choose at runtime the smallest that covers
`W + D`. `H = 3` covers the default; `H = 8` is a slow catch-all.

**`W` in the code.** `W` is the band width of the input grid, given per certifier (`setBandWidth`;
default and maximum 3) and required to be at least `3D` (the primitive reduction). Only `H = 3`
(sweep) and `H = 5` (shell and recompute passes) are compiled, which is what caps `W` at 3. The tap
tables are built on the host from `W`: at `W = 2` the sweep has 74 taps (`|w|² ≤ 6`) instead of
146. Before `W` was a parameter the code used `W = 3` for every grid. On narrower bands (1.7 to
2.5, 20 inputs, both modes) that certified the same voxels, but wrote inside distances down to
`-3` voxels, beyond the background value. Hairball, band 2, resolution 5344, RTX 5090: with
`W = 2` the sweep takes 1.3 s instead of 1.5 s, and the recompute pass 0.5 s instead of 0.8 s.

**Shell stencil (pedantic mode only).** Primitive `w` with `W + D ≤ |w| < 2W`: 624 taps at `W = 3`.
It needs the box with `2W ≤ √((H + 1)² + 1)`, so `H = 5` at `W = 3`; the cap `H ≤ 8` allows
`W ≤ 4.53`. Beyond that the shell is clipped and pedantic mode is no longer exact (still sound).

The isovalue (deferred) will interact with this choice: with an offset the radius becomes
`|udf - iso|`, inside radii can exceed `W`, and the needed reach grows.

## 3. One CTA per leaf, with a halo tile

- **One CTA per leaf, 512 threads**, matching `operatorKernel` with `MaxThreadsPerBlock = LEAF_SIZE`.
  One warp per leaf was rejected: shared memory limits residency to about one tile per warp slot,
  so about 9 warps per SM at `H = 3` (19% occupancy), too few to hide the gather's latency.
- **Tile:** `(8 + 2H)³` floats in shared memory, `tile[c] = r` if cell `c` is a certified voxel and
  `-inf` otherwise (inactive, missing leaf, or uncertified). The test is then branch-free:
  `tile[c + delta] + r_q > |w| + τ`. Store `udf / h`, so the tile, `|w|` and `τ` are all in voxel
  units.
- **Only the leaf's own 8³ voxels are candidates.** The halo is read-only: halo voxels' own
  certifiers lie outside the tile, and only the owning CTA writes a voxel's state, so there are no
  write-write races between CTAs.

## 4. Global data

| array | per | type | written by |
|---|---|---|---|
| UDF | active slot | `float`, unsigned, read-only | nobody (input) |
| certified | leaf | `Mask<3>` (512 bits) | the owning CTA only |
| state | leaf | `uint32_t`: bit 0 dirty, bit 1 has candidates; double-buffered | dirty: any CTA (push); candidates: owner |
| neighbours | leaf | `uint32_t[27]`, leaf indices, `UINT32_MAX` if absent | setup kernel |
| components | active slot | label, plus `rootOutside[N]` | promotion pass (planned, not in v1; section 10) |

- **Why not keep the flag in the UDF's sign bit:** it would halve gather loads, but the UDF would no
  longer be read-only (re-runs and other consumers see a mutated buffer), and the output would be
  2 KB of floats per leaf instead of a 64-byte mask. The mask turns into a signed field in one pass
  at the end.
- **Plain stores, relaxed atomics.** Every writer of a dirty word writes the same value, and only
  owners write masks and candidate bits, so no read-modify-write atomics are needed on global
  memory. Use relaxed `cuda::atomic_ref<uint32_t>` stores and loads so cross-block reads within a
  round are well-defined; they compile to ordinary memory operations. One `uint32_t` per leaf
  avoids packing that would need `atomicOr`.

## 5. Gather

- **Neighbour leaves first.** Version 1: at CTA start, 27 threads each `probeLeaf` one neighbour
  origin into shared memory (the pattern of `util/cuda/Morphology.cuh:728`). Later: fill the global
  neighbour table once per grid, since topology never changes during the sweeps and other halo
  operations (the interior distance recompute, nesting) need the same table. Store indices, not
  pointers: half the size, and position-independent.
- **Fused index and load.** Each thread computes a cell's (neighbour leaf, offset, slot) in
  registers and loads what it needs right away. A materialised 14³ index tile would take 22 KB of
  shared memory, twice the data it feeds, to save about 10 instructions per cell
  (`ValueOnIndex::getValue` is a mask test, the packed prefix sum, and one popcount). A small
  callback-based helper keeps this reusable across operations and sidecar arrays:

  ```cpp
  gatherHalo<H>(nbrLeafIDs, [&](int cell, const LeafT& leaf, uint32_t offset, uint64_t slot) {
      const bool cert = certMask[leafID(leaf)].isOn(offset);   // flags need no slot
      sTile[cell] = cert ? dUDF[slot] * invVoxelSize : -INFINITY;
  });
  ```

- **Thread mapping: lexicographic, z fastest.** Cell `c = threadIdx.x, threadIdx.x + 512, ...`,
  decomposed as `(ti, tj, tk)` with constant divisors. With `tk` fastest, consecutive lanes follow
  each source leaf's memory order (`i·64 + j·8 + k`) within a row, so a warp sees at most three
  interleaved, nearly sequential streams (the `-z` leaf, the middle leaf, the `+z` leaf). The
  neighbour index and local offset are arithmetic (`(t >= H) + (t >= H + 8)` per axis; local
  coordinate `(t - H) & 7`), with no geometric branches.
- **Why not read whole neighbour leaves and shuffle:** coalescing is about the number of 32-byte
  sectors touched, not per-thread contiguity. `±x` and `±y` slabs touch 3/8 of a leaf's sectors;
  `±z` slabs touch the same sectors as full rows. So direct slab reads never touch more sectors than
  whole-leaf reads, and whole-leaf reads would move about twice the bytes (55 KB against 20–26 KB
  per tile at `H = 3`) plus a staging buffer. The exception is `H = 8`, where the halo *is* whole
  neighbouring leaves and sequential reads are the natural choice.
- **Details:** inactive cells get slot 0, so the load is branch-free (slot 0 must read as "not
  certified"). Full leaves (value mask all on) can skip the popcount (`slot = mOffset + offset`);
  record one bit per neighbour. Sparse leaves keep slots in offset order, so streams still advance
  monotonically.

## 6. Tap evaluation

- **One warp per candidate, one lane per tap.** Compact the leaf's candidates into a shared list
  (ballot plus popcount); each warp takes candidates in turn and covers the taps 32 at a time,
  stopping at the first success (`__any_sync`). One thread per candidate would leave most warps idle,
  since candidates are a thin layer.
- **Tap tables, per `H`, generated offline.** Entries `{ int32 delta, float len }`, where `delta` is
  the linear tile offset `dx·S² + dy·S + dz` (`S = 8 + 2H`) and `len = |w|`, sorted by `len`. A run
  with `W` and `D` uses the prefix `len < W + D`, so only its length `n(W + D)` is a runtime argument
  (binary search on the host). The sort order also puts the likeliest certifiers first. Valid while
  `W ≥ 3D` and `D ≤ 1` (the primitive reduction); check on the host.
- **Generate a header with a script** rather than computing the tables with `constexpr` functions:
  sorting about 2,500 entries at compile time can hit compiler step limits. Keep the arrays at
  namespace scope (memory-space specifiers are not allowed on class data members), with a template
  accessor:

  ```cpp
  struct Tap { int32_t delta; float len; };
  __device__ const Tap kTapsH3[218] = { /* generated, sorted by len */ };
  template <int H> __device__ const Tap* taps();
  template <> __device__ const Tap* taps<3>() { return kTapsH3; }
  ```

- **Memory space.** Not `__constant__`: the constant cache serialises when lanes read different
  entries, which lane-per-tap always does. `__device__ const` arrays read with `__ldg` are coalesced
  (32 consecutive entries per warp per round) and mostly hit L1. At `H = 3` a CTA can instead copy
  its prefix (1.2 KB) into shared memory. Check the SASS (`cuobjdump -sass`): `LDG...CONSTANT` is
  the read-only global path, `LDC` the constant bank, `LDL` local memory (bad).
- **Round `len` deliberately.** A `len` rounded up only makes the test stricter; keep `τ` well above
  the rounding either way.

## 7. Local fixpoint in shared memory

- After the first pass, a newly certified candidate writes `tile[q] = r_q` and sets its bit in a
  shared output mask, and the CTA re-tests the remaining candidates until a pass certifies nothing.
  The gather dominates the cost, so extra passes over an already loaded tile are cheap, and a chain
  crosses a whole leaf per launch.
- **Cap the passes at 4–8.** Each pass certifies at least one voxel or stops, but a winding chain in
  one leaf could take up to 512 passes; anything left over goes to the next global round. Lemma M
  suggests 1–3 passes in practice. Use the in-block latch pattern of the invert-mask flood in
  `MeshToSDF.cuh` (`sChanged`, latch then barrier).
- **Shared output mask:** several warps may set bits in one 64-bit word, so use shared-memory
  `atomicOr`, or a 512-byte flag array packed into the mask at write-back.
- **`maxLocalIters = 1` with double-buffered masks** gives a mode in which every round is a clean
  step that does not depend on scheduling. Use it for debugging and as a reference: both modes must
  produce identical certified sets.

## 8. Cost model (estimates, dense leaf, `H = 3`)

| | count | cost per item | total per leaf |
|---|---|---|---|
| gather | 2744 cells | ~30 instructions + a 4 B load, mostly L2 | ~800–1,400 SM-clocks |
| pull pass, worst case | 512 × 146 taps | ~4 instructions + a 4 B shared-memory load | ~2,300–4,500 SM-clocks |

- Candidates that succeed exit early. Candidates that fail (in the pull model, the inside voxels,
  about half the band) pay all 146 taps, and would pay them again on every pass and every round.
  A realistic first pass is about 1.5–3× the gather.
- **Later optimisation: push after the first pass.** Each voxel certified in the previous pass tests
  the uncertified voxels of this leaf's 8³ within its offsets. Work becomes proportional to the
  number of newly certified voxels, not the number of candidates. Across rounds, the same idea needs
  a per-leaf "newly certified" mask, marked in the halo during the gather.
- Profile before tuning: the numbers depend mainly on L2 bandwidth per SM and on bank conflicts in
  the scattered tap loads.

## 9. Rounds, dirty leaves and termination

**Version 1 (correctness first):**
- **Launch one CTA per dirty leaf.** Each round the host compacts the dirty flags into a leaf list
  (`cub::DeviceSelect::Flagged`, so the list is in leaf order), reads its length and launches that
  many CTAs. CTA `b` processes leaf `list[b]` and exits at once if the leaf has no candidates. An
  empty list ends the pass.
- **Dirty flags are pushed.** A CTA that certifies anything stores 1 into the next round's dirty
  word of its own leaf and its 26 neighbours. Clean leaves get no CTA.
- **Double-buffer** the dirty words: round `r` reads `dirtyCurr` and writes `dirtyNext`; swap and
  clear (`cudaMemsetAsync`) between rounds. Round 1 starts with every leaf that has candidates.
- **Has-candidates** is cleared by the owner at write-back when no uncertified active voxel remains.
- **Termination:** a global "anything changed" counter, read by the host every round.
- Keep a mode that ignores dirty flags and processes every leaf with candidates every round (the
  list is then compacted from the has-candidates flags); all modes must agree.

**Why the leaf list is in version 1** (it was the last of the later items). A launched CTA costs
about 2 ns even if it exits at once, and the number of rounds grows with resolution: a region
reached late is crossed one leaf per round. Hairball, band 2, RTX 5090: 14, 243 and 402 rounds at
resolutions 3072, 3840 and 4592; at 4592 (14.4M leaves) a round over every leaf costs about 30 ms
while a few hundred to a few tens of thousands of leaves do work. At resolution 5344 (3.23B
voxels, 19.9M leaves, 445 rounds) the list takes the sweep from 19.6 s to 1.5 s with the same
certified voxels. A list the CTAs push into themselves (guarded by a queued flag per leaf) was
0.08 s faster there and needs more code, so the compaction stays.

**Pedantic mode: the shell pass (in version 1).** After the `W + D` sweep reaches its fixpoint, run
one pass of the same kernel with the shell stencil (section 2) and `H = 5` at `W = 3`, over leaves
with uncertified non-barrier voxels:
- *candidates:* uncertified non-barrier voxels only (a hop of `|w| ≥ W + D` needs
  `r_n + r_q > W + D`, so both ends have `r > D`);
- *taps:* the shell, sorted by `|w|`, cut at `|w| < r_q + W`; first success wins as usual;
- *then:* if it certified anything, push dirty flags and resume the `W + D` rounds; repeat until a
  shell pass certifies nothing.

Usually the first shell pass finds nothing, so pedantic mode costs one pass. That pass is
expensive: its candidates are mostly interior non-barrier voxels (roughly 40% of the band), which
never succeed and so scan their whole prefix, up to 624 taps against 146. Expect a few rounds'
worth of work. It goes into version 1 because it is the same kernel with another tap table and a
candidate filter, and because it is the GPU reference for the default stencil (S2 in
`NewDistancingApproach.md`).

**Later:**
- **Fewer host syncs:** read the counter every `k` rounds (extra rounds after convergence are
  nearly free), then a CUDA graph with a conditional `while` node (CUDA 12.4+), keeping the batched
  host loop as fallback. A persistent cooperative kernel with `grid.sync()` was rejected: residency
  limits with large tiles, more complexity, no precedent in the codebase. The leaf list adds a
  second host read per round (its length), which batching must also cover.

Dropped: a fixed grid of CTAs looping over every leaf and skipping the clean ones. It targeted the
empty CTAs, which the leaf list no longer launches.

## 10. Component promotion (#4): not in version 1, a planned optimization

*Status: decided for version 1 (leave it out); to be added once the plain sweep is validated, as
the `promote` option (on by default in pedantic mode; the default otherwise to be set by S5). It is
expected to matter on meshes with slots and enclosed pockets, so it should not be dropped.*

**What promotion is.** Seeding runs `ConnectedComponents` on the non-barrier voxels (`udf ≥ D`) of
the surface's grid and keeps the piece containing the extreme voxel. That pass labels every
non-barrier piece, and each piece is provably entirely inside or entirely outside (a 6-connected
step between non-barrier voxels cannot cross `Σ`). Promotion reuses the labels: once the sweep
certifies any voxel of a piece, certify the whole piece.

**What it can change: only the number of rounds.** The certification itself has no barrier voxels
and no components: every active voxel is a candidate, and face neighbours in one piece already pass
the ball test (`udf(a) + udf(b) ≥ 2D = 1.1 > 1 + τ`). So once the sweep certifies one voxel of a
piece, it certifies the whole piece by itself, and promotion never changes the fixpoint or the reach
analysis. Equivalently: remove the barrier voxels from the final certified set `C`, and what remains
is a union of *whole* non-barrier pieces, the seed piece plus any others `C` entered. Promotion
helps only when there are others.

- *One piece (the common case):* `C` is the seed piece plus the barrier shell next to it, and the
  sweep should need only a few rounds (to be measured, S5).
- *More pieces:* each extra piece is flooded at about one leaf per round, since a CTA writes only its
  own leaf and sees its neighbours' results a round later. A pocket whose band runs 500 voxels along
  the surface takes about 60 extra rounds; a long slot through a large model, hundreds. With
  promotion, each extra piece costs the round that enters it plus one promotion step.

**Where extra pieces come from: narrow passages, only.** `C` never leaves the seed's side of `Σ`, so
an extra piece is always part of the one true exterior, cut off from the seed piece after the
barrier voxels are removed. That happens at a passage too thin to hold a 6-connected chain of voxels
all at least `D` from `Σ`: a gap of about `2D` plus up to about a voxel of slack depending on
orientation (diagonal passages are cut sooner). Typical shapes: the air pocket between two lobes of a
mesh that nearly touch at two necks, the inside of a bottle, the region behind a slot between parts.
The sweep enters such a piece through barrier voxels in the neck (lattice points there have
`udf ≈ g/2` for a gap `g`, so a gap of about a voxel suffices), or by one longer hop across a short
pinch. Behind a sub-voxel neck the entering hop can be as long as `2W`, beyond the default stencil
(figure `pocket_entry_failure_2d.svg` in `NewDistancingApproach.md`): the default mode then leaves
the pocket uncertified, labelled inside, and pedantic mode (section 9) certifies it. Promotion does
not change which pockets are entered, only how fast an entered one fills.

A sealed cavity (two lobes that actually touch, or a hollow shell modelled as two surfaces) is a
different region of the complement of `Σ`; `C` never reaches it, with or without promotion. That
belongs to the nesting question in section 13, not here.

**Cost of promotion.** It keeps the per-voxel labels from seeding alive, adds a pass that flags the
pieces holding a certified voxel and a pass that certifies their voxels, and must repeat (a promoted
pocket may have necks into further pockets). Against that, the rounds it saves are cheap ones:
with dirty-leaf early exit a round does real work only on the frontier, so the saving is mostly
launch and termination-check overhead, which the leaf list and the later item of section 9 also
reduce.

**A possible shape.** Seeding becomes the first promotion: clear `rootOutside[N]`, mark the extreme
voxel's label, run the inherit pass. Then: sweep round, during which the owning CTA marks
`rootOutside[label]` for each newly certified non-barrier voxel (`atomicExch` on a `uint32_t` to
detect new labels) → if a new label appeared, inherit pass over leaves with uncertified non-barrier
voxels (certify, push dirty flags) → repeat until neither changes anything. Labels come from
`ConnectedComponents` on the pruned grid at the same `D` as the barrier test (dense ids, so `find()`
is a plain lookup). Promotion on and off must give identical certified sets (S3).

## 11. Scope: one surface at a time

Each surface's grid holds one UDF component, so every tile sees only that component and
"certified" always means outside that component's surface. If surfaces are later batched into one
sweep on the shared grid (worthwhile when there are many small surfaces, to avoid a launch sequence
per surface), halos will pick up other components' voxels, and a cluster enclosing another would
wrongly certify its inner side from the inner cluster's exterior. The gather must then fold
other-component voxels to `-inf` by component label. Design the worklist so it does not care how
many components a grid holds.

## 12. Compared with today's `BallCertifyFunctor`

| today | this design |
|---|---|
| thread per voxel; `(2r+1)³` box, default `r = 1` | primitive stencil from `W + D`; 146 taps at `W = 3` |
| out-of-leaf taps through `ReadAccessor`, two scattered loads each | one gather per leaf into shared memory |
| `sqrtf` per tap | `\|w\|` from the tap table |
| two-sided (inside balls certify inside), no early exit | outside-only, first success wins |
| double-buffered labels, one hop per round | in place, local fixpoint per tile |
| every leaf every round | dirty leaves only |
| host sync every round, up to 32 rounds | counter every round (v1), then every `k`, then device-side loop |
| `eps = 1e-5·h` | `τ = 2δ`, default `1e-3` voxels |

At `r = 1`, a like-for-like port would gain only modestly; the tile's fixed gather cost roughly
matches today's few taps. At the reach the analysis requires (`r = 3`), today's per-candidate cost
is 342 accessor-based taps, and the tiled design should win by roughly an order of magnitude per
round (an estimate to confirm by profiling).

## 13. Unresolved questions (to revisit)

Items 1–5 of the review are settled: scope (section 11), rounds and termination (section 9),
the tolerance (`NewDistancingApproach.md`), promotion and pocket entry (sections 9 and 10), and
validation (below). Item 4b is settled (sealed cavities are filled). Item 6 is decided in part: inside magnitudes will
be recomputed (version 1 designed, section 14); the isovalue is deferred.

**Next step (decided November 2026):** integrate the sweep and the inside recompute into
`MeshToSDF` as one of two barrier-signing options, the other being the existing OpenVDB-style
intersecting-voxel heuristic. Still deferred: the isovalue, and the provisional choices (the value
of inside voxels with nothing within reach, section 14.5; the default of `promote`, after S5).

**#4 — Component promotion and pocket entry: settled** (sections 9 and 10). Promotion is left out
of version 1 and planned as the `promote` option afterwards, sized by S5. Pockets whose entry needs
a hop beyond `W + D` (sub-voxel necks) are recovered by the `pedantic` option's shell pass, which is
in version 1.

**#4b — Sealed cavities (within-cluster nesting): settled, fill them.** A region of the complement
of `Σ` that is enclosed by surface (winding number 0, not connected to the exterior at any
resolution), whose bounding surfaces share one UDF component with the outer surface. One spot
where the surfaces come within about `2W` of each other is enough to merge them. Every certified
ball is surface-free, so no ball chain reaches such a cavity, whatever the stencil, schedule or
promotion (figure `figures/sealed_cavity_2d.svg`, generated by
`scripts/distancing_experiments/sealed_cavity_2d.py`: a hollow disk with a 2.5-voxel wall, 196/196
exterior band points certified, 0/94 cavity band points, default and pedantic alike).

Decision: the cavity is filled, i.e. labelled inside (`invertMask` on). We accept that the
transition to the non-material region cannot be detected unless the cavity's inner boundary is its
own UDF component; in that case the nesting stage (`PropagateSign` and the even-odd or solid rule)
makes it a void. This is the "take the outermost" fallback of `InclusionSigningDesign.md` §7, and it
keeps the containment specification: the outside set contains the region at infinity and never
intersects `Σ` (Property 1 of `NewDistancingApproach.md`). Filling a cavity only enlarges the
inside.

**#5 — Validation: settled, deliberately frugal.** Two layers, plus the experiments T1–T3 and
S1–S5 in `NewDistancingApproach.md`.

*Unit test (`TestNanoVDB.cu`), minimal:* one sphere with an analytic UDF,
`udf = ||x - c| - R|` in float (no mesh, so distance error is float rounding, far below `τ`, and
inside/outside is exact), `R` about 20–30 voxels, centre at a non-lattice offset such as
`(0.37, 0.21, 0.64)`, `W = 3`. Assert:
- *completeness:* every outside voxel with `udf > 0.1` voxel is certified. Convexity makes the
  overlap test at least as easy as for the tangent plane (`udf_sphere(V) ≥ dist(V, H)`), but the
  same curvature can push a deep planar witness just out of the band (about `6ε/R`, roughly 0.01
  voxel at `R = 20`), and `τ` costs `τ/2`; hence the margin over the planar gap of 0.0384. On
  failure, print the smallest uncertified outside `udf`;
- *soundness:* no inside voxel is certified;
- *modes agree:* default and pedantic give identical sets (a sphere has no pockets).

*Example (`mesh_to_sdf_cuda`):* retune the existing OpenVDB `meshToLevelSet` sign cross-check
(`mesh_to_sdf_cuda_kernels.cu`, under `NANOVDB_USE_OPENVDB`) to report and characterise mismatches
rather than fail. OpenVDB is a reference with its own sign method, not ground truth.
- Classify by direction and `udf`: *we inside / OpenVDB outside* is expected below 0.1 (the
  certification gap), unexpected between 0.1 and `D` on smooth surfaces, and at `udf ≥ D` means a
  pocket not entered or a bubble; *we outside / OpenVDB inside* is the only class that could mean
  unsoundness, so each case is examined (with the existing `CC_EXPORT_VIS` viewer export).
- Report 6-connected clusters of mismatches (count, size, `udf` range) rather than voxels, and for
  `udf ≥ D` clusters whether pedantic mode recovers them.
- Behind an environment flag, rerun with `maxLocalIters = 1`, without dirty-leaf skipping, and in
  pedantic mode, and report differences in the certified set (the first two must be none).
- Counters: rounds, shell passes, non-seed components entered, near misses
  (`|r_n + r_q - |w|| ≤ τ`).

*Dropped:* an exact-match CPU reference (what it alone would catch, a bug shared by every GPU mode
such as a wrong tap table or a halo gather error, would also show up as unexplained OpenVDB
mismatch clusters or in the sphere test; add it if a cluster cannot be explained), and winding
numbers (OpenVDB stands in).

**#6 — Follow-ons that share this infrastructure.**
- *Inside magnitudes: decided, recompute; version 1 designed* (section 14). Uncertified voxels get
  `φ(p) = -minᵢ(|p - Vᵢ| - dᵢ)` over certified `Vᵢ`, the exact signed distance to `∂O` (Step 3 of
  `NewDistancingApproach.md`), rather than `-udf`. Reasons: the field is self-consistent everywhere
  (keeping `-udf` puts large negative values next to large positive ones across the neck of a missed
  pocket), inclusion holds by construction, and it can be evaluated between lattice points. For a
  truly inside `p` it is never below `udf(p)`, so it departs from the mesh distance only where `∂O`
  comes loose from `Σ`. Version 1 is an exact brute-force search; it is to be benchmarked before any
  optimization is discussed.
- *The isovalue: deferred until after the inside recompute.* Notes so far: `iso > 0` (dilation) is
  the same sweep on `udf - iso` over voxels with `udf > iso` (radius `udf - iso`, barrier
  `udf - iso < D`, reach `W + D` in those terms; the input band must widen to `W + iso` to keep an
  effective `W`, which feeds the choice of `H`; necks narrower than about `2·iso` close, correctly).
  `iso < 0` (erosion) cannot use unsigned data; it is a post-pass `φ + |iso|` on the signed result
  and needs inside magnitudes valid to `|iso|`, so it depends on the recompute above.

## 14. Inside magnitudes: the recompute pass (version 1)

*Status: designed, not implemented. Implement exactly as below, then benchmark (14.6) before
considering any of the optimizations in 14.7.*

### 14.1 What it computes

After `SignCertifier` has converged (including pedantic passes, if enabled), every uncertified
active voxel `p` gets

```
φ(p) = -min over certified V of (|p - V| - d_V),   clamped to [-W, 0]
```

in voxel units, written to the SDF sidecar as `φ(p)·h`. `d_V` is the input UDF of `V`. Certified
voxels keep `+udf`, which is already their exact distance to `∂O` (the Property section of
`NewDistancingApproach.md`). So the pass reads the UDF only at certified voxels and never uses the
UDF of `p` itself. It is one pass with no iteration: the certified set is final.

### 14.2 How far to look, and when to stop

- **Taps:** *all* integer offsets with `0 < |w| < 2W` (894 at `W = 3`), sorted by `|w|`. The
  primitive reduction does not apply: it was an argument about reachability, and this is a
  minimum over distances.
- **Why `2W`:** a certified `V` at offset `w` contributes `|w| - d_V ≥ |w| - W`, which reaches the
  clamp `W` once `|w| ≥ 2W`.
- **Stop rule:** start with `best = W` (the clamp). Scan the taps in order, keeping
  `best = min(best, |w| - d_V)` over certified `V`, and stop at the first tap with
  `|w| ≥ best + W`. Every later tap contributes at least `|w| - W ≥ best`.
- **One table for every `W`:** the constant table holds the offsets of the largest band
  (`W = 3`). A narrower band reads a prefix of it: the stop rule ends the scan no later than the
  first tap with `|w| ≥ 2W`.
- **Why `best + W` and not `best`:** a farther `V` can still win if its ball is larger. These deep
  balls are often the winners: a ball of radius `r` bulges about `s²/8r` away from `Σ` between the
  points where balls touch it (`s` about one voxel), so the deepest balls hug `Σ` best, which is the
  same mechanism as the planar infinite-lattice result.
- **Expected scan length:** for a truly inside `p` at depth `u`, `best ≥ u`, so a voxel scans at
  least the taps with `|w| < u + W`:

  | depth `u` | 0.25 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 |
  |---|---|---|---|---|---|---|---|
  | taps scanned | 146 | 178 | 250 | 388 | 484 | 738 | 894 |

  That is about 420 on average over a uniformly deep inside band, slightly more in practice, since
  `best` sits a little above `u`.

### 14.3 Tile

- One CTA per leaf that has at least one uncertified active voxel (the has-candidates bit the
  sweep already keeps).
- **Halo `H = ⌈2W⌉ - 1`**, so `H = 5` at `W = 3`. Every offset with `|w| < 2W` has all coordinates
  at most 5. The tile is `18³ = 5,832` floats, 23 KB, and fits within the 3×3×3 block of
  neighbouring leaves up to `W = 4.5`.
- The tile is the same as the pedantic shell pass's, with the same encoding: certified voxels
  hold `d` in voxel units, everything else holds `-inf`, so `|w| - tile` is `+inf` for them and the
  loop needs no branch. Reuse the gather, the neighbour table and the `H = 5` instantiation.

### 14.4 Parallel mapping: a thread per inside voxel, taps in lockstep

The sweep uses a warp per voxel and a lane per tap, because a success ends the search early. Here
there is no early success, only a running minimum, so the mapping flips:

- **Compact** the leaf's uncertified active voxels into a shared-memory list, in tile order, so
  that threads are not wasted on certified or inactive voxels.
- **A thread per listed voxel**, looping over the list if there are more voxels than threads.
- **All lanes read the same tap at the same time.** The tap table can therefore live in
  `__constant__` memory, whose reads are broadcasts when uniform across the warp (894 taps × 8 bytes
  ≈ 7 KB). This is the opposite of the sweep, where lanes read different taps. The tile reads
  `tile[cell(p) + delta]` are consecutive across lanes, as in an ordinary stencil.
- **Per-thread stop, per-warp exit:** each thread stops updating at its own cutoff; the warp
  leaves the loop when `__all_sync` says every lane is past its cutoff, checked every 16 taps.
  Voxels in one warp are spatially close and of similar depth, so lanes should finish at similar
  points.
- **Tap lengths** are stored rounded up, as in the sweep. Here that can only make `|φ|` larger by
  at most one ulp, which is conservative for inclusion.

### 14.5 Edge cases

- **Nothing within reach** (`best` stays at `W`): write `-W·h`, the band's inside limit.
  (Provisional; the alternative is the negative background.)
- **Truly outside voxels that were not certified** (sub-gap barrier voxels, bubbles, pockets
  behind sub-voxel necks, sealed cavities): they get the same formula and are consistent with
  `∂O`. For them `|φ|` can be below their UDF, which is expected.
- **Inactive voxels and tiles** are signed by `PropagateSign` as before; this pass touches only
  active voxels.

### 14.6 Validation and benchmark

- **Unit test (the sphere of #5):** for every inside voxel, `|φ| ≥ udf - 1e-5` (the lower bound
  for truly inside points) and `|φ| - udf ≤ 0.1` (looseness of `∂O`; print the largest value on
  failure).
- **Example:** report the distribution of `|φ| - udf` over inside voxels (extremes, percentiles),
  with the locations of the extremes. Large positive values mark places where `∂O` comes loose from
  `Σ` (narrow creases the balls cannot enter); negative values can only occur at truly outside
  voxels that were not certified (missed pockets, bubbles, sealed cavities).
- **Benchmark, before any optimization:**
  - the pass's GPU time, next to the other pipeline stages in `BENCHMARK.md`. Rasterization (mesh to
    UDF and index) is about 95% of the pipeline on single-surface input, so the pass is judged
    against that;
  - the number of leaves processed, inside voxels, and taps scanned per voxel (mean and
    distribution);
  - scaling with resolution (the dragon series of `BENCHMARK.md`).

### 14.7 Optimizations, deferred until the benchmark says they matter

- **Floor-based stop:** for a truly inside `p`, every candidate is at least `udf(p)`, so stopping
  once `best - udf(p) ≤ ε` bounds the error by `ε`. Exact only at `ε = 0`, where it rarely triggers;
  for missed exterior voxels the floor does not hold.
- **A tighter cap than `W`** in the stop rule: the largest `d` actually present in the tile.
  Exact, but in a full band usually close to `W`.
- **Normal-directed search:** with the foot point from the nearest-triangle sidecar, near-planar
  `Σ` puts the winning `V` within about 0.8 voxel of `p`'s normal line (`τ² ≤ 2λ(u + z) + λ²`,
  `λ = dist(foot, ∂O)`). It needs only tens of taps, but it is not exact at sharp convex edges or
  high curvature, so it needs a fallback.
- **Shell-by-shell propagation of the winning site:** the region where a given `V` wins is
  star-shaped from `V`, so candidates can come from the neighbours' winners, about 26 evaluations
  per voxel over 3–5 shells. It is approximate on the lattice (rare small errors, as with
  propagation distance transforms), and its per-voxel work is small enough that iterating over each
  shell with the `VoxelBlockManager` fits better than leaf tiles.
- **Warp per voxel** (lanes over taps) as an alternative mapping to benchmark.
- **Fusion with the pedantic shell pass:** same tile and overlapping candidates; run the recompute
  speculatively in the last shell pass and redo only leaves next to a new certification.
