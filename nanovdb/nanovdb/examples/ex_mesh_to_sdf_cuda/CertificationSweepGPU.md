# The ball-certification sweep on the GPU: design

> Working design from the October 2026 design review (branch `new-distancing-approach`). Not yet
> implemented. The theory it relies on is in `NewDistancingApproach.md`: Step 2 (the closure),
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
  alike. (Reusing the seeding labels to promote whole components is an optional acceleration, still
  unresolved: section 10.)
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
| components | active slot | label, plus `rootOutside[N]` | promotion pass (only if adopted; section 10) |

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
- Launch one CTA per leaf over all leaves each round. Thread 0 reads the leaf's state word; if it is
  not both dirty and has-candidates, the whole CTA exits together (decide through a shared flag after
  one `__syncthreads()`).
- **Dirty flags are pushed.** A CTA that certifies anything stores 1 into the next round's dirty
  word of its own leaf and its 26 neighbours. Clean leaves then exit after a single load, without
  touching the neighbour table.
- **Double-buffer** the dirty words: round `r` reads `dirtyCurr` and writes `dirtyNext`; swap and
  clear (`cudaMemsetAsync`) between rounds. Round 1 starts with every leaf that has candidates.
- **Has-candidates** is cleared by the owner at write-back when no uncertified active voxel remains.
- **Termination:** a global "anything changed" counter, read by the host every round.
- Keep a mode that ignores dirty flags and processes every leaf with candidates every round; all
  modes must agree.

**Later, in order:**
1. **A fixed grid looping over every leaf** (about SM count × occupancy CTAs, each striding over
   leaf indices and skipping clean ones). A CTA reserves its shared memory at launch even if it exits
   at once, so at `H = 8` (one CTA per SM) launching one CTA per leaf costs about a millisecond per
   round in empty CTAs for 100k leaves. The loop pays a few loads per clean leaf instead.
2. **Fewer host syncs:** read the counter every `k` rounds (extra rounds after convergence are
   nearly free), then a CUDA graph with a conditional `while` node (CUDA 12.4+), keeping the batched
   host loop as fallback. A persistent cooperative kernel with `grid.sync()` was rejected: residency
   limits with large tiles, more complexity, no precedent in the codebase.
3. **A compacted leaf list** (`cub::DeviceSelect::Flagged`), only if profiling at scale (millions of
   leaves, many rounds) shows the flag scan matters.

## 10. Unresolved (#4): component promotion, and how many non-barrier pieces to expect

*Status: discussed, not settled; to revisit.*

**What promotion is.** Seeding runs `ConnectedComponents` on the non-barrier voxels (`udf ≥ D`) of
the surface's grid and keeps the piece containing the extreme voxel. That pass labels every
non-barrier piece, and each piece is provably entirely inside or entirely outside (a 6-connected
step between non-barrier voxels cannot cross `Σ`). Promotion reuses the labels: once the sweep
certifies any voxel of a piece, certify the whole piece.

**It is optional.** The certification itself has no barrier voxels and no components: every active
voxel is a candidate, and face neighbours in one piece already pass the ball test
(`udf(a) + udf(b) > 1.1 > 1`). So promotion never changes the result or the reach analysis; it only
saves rounds. Without it, certification crosses a large piece about one leaf per round (a pocket
spreading 500 voxels along the surface takes about 60 rounds); with it, one.

**A possible shape, if adopted.** Seeding becomes the first promotion: clear `rootOutside[N]`, mark
the extreme voxel's label, run the inherit pass. Then: sweep round, during which the owning CTA
marks `rootOutside[label]` for each newly certified non-barrier voxel (`atomicExch` on a `uint32_t`
to detect new labels) → if a new label appeared, inherit pass over leaves with uncertified
non-barrier voxels (certify, push dirty flags) → repeat until neither changes anything. Labels come
from `ConnectedComponents` on the pruned grid at the same `D` as the barrier test (dense ids, so
`find()` is a plain lookup). Promotion on and off must give identical certified sets (add to S3).

**Current lean:** leave it out of version 1, so the sweep is exactly the method as stated, and use
experiment S1 to count rounds on pocket-heavy meshes before deciding.

**Open sub-question: how many non-barrier pieces, and what they are.** For one closed connected
surface there is exactly one outside region in the continuum, but after removing the barrier voxels
it can split into several pieces:
- *Narrow necks.* A passage narrower than about `2D` is all barrier, so a bottle's inner air, or the
  region behind a slot between parts of one mesh, becomes a separate outside piece. On the lattice,
  6-connectivity also cuts somewhat wider (especially diagonal) necks. These pockets are what the
  ball sweep exists to reach.
- *Several surfaces in one UDF component.* A UDF component is one connected band, not one closed
  surface: surfaces closer than about `2W` share a band. A thin-walled hollow sphere modelled as an
  outer and an inner surface has a third region, the cavity, whose band is its own piece. Nothing
  can certify it (it is not connected to the outside), so it ends up "inside". Whether that is right
  depends on what the mesh means (a solid with a void, or nested objects): the within-cluster
  nesting question, not yet addressed by the sweep or anything else in the pipeline.

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

Items 1–3 of the review are settled above: scope (section 11), rounds and termination (section 9),
and the tolerance (`NewDistancingApproach.md`). Items 4–6 are open.

**#4 — Component promotion and non-barrier pieces.** See section 10: whether promotion goes into
version 1, and what to do with pieces that are neither the outside nor solid (cavities when one UDF
component holds several surfaces).

**#5 — Validation strategy.** Queued so far: T1–T3 (tolerance) and S1–S4 (sweep) in
`NewDistancingApproach.md`. Points raised but not settled:
- the Python scripts in `scripts/distancing_experiments/` (bubbles, planes, wedges) as the CPU
  reference, which the GPU sweep must match exactly;
- equivalence between modes (one pass per launch, local fixpoint, dirty-leaf skipping, promotion on
  and off);
- soundness against winding numbers on closed meshes;
- restoring a diagnostic count (certifications per round, undecided barrier voxels at the end),
  since today's contradiction tally is gone.

**#6 — Follow-ons that share this infrastructure.**
- *Inside magnitudes:* keep the input UDF, or recompute distances to `∂O` (Step 3 of
  `NewDistancingApproach.md`)? The recompute would reuse the tiles and the neighbour table.
- *The isovalue (deferred):* with an offset the radius becomes `|udf - iso|`, inside radii can
  exceed `W`, and the needed reach grows, which interacts with the choice of `H`.
