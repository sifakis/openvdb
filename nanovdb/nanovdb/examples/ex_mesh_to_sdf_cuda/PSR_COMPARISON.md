# Four per-leaf union-find schedules, measured

Generated 2026-08-09 by `ex_mesh_to_sdf_cuda --sv-convergence <mesh> --voxel-size S`.

All four are from Liu & Tarjan, *Simple Concurrent Connected Components Algorithms* (ACM TOPC 9(2), 2022), which casts this family as a **connect** step followed by one or more **shortcut** steps, repeated until no parent changes.

| | main loop | proven step bound |
|---|---|---|
| **H** | `parent-connect; shortcut`, then `parent-root-connect; shortcut` after `SwitchToRootAfter` rounds | **O(lg n) on the tail.** P's constants while it converges fast, R's bound if it does not |
| **P** | `parent-connect; shortcut` | **none.** Section 4.2: *"we are unable to prove even an O(lg² n) bound"*, and an earlier published analysis was withdrawn as incorrect |
| **S** | `parent-connect; shortcut until flat` | O(min{d, lg n}·lg n) — theorem 4.1, shown tight |
| **R** | `parent-root-connect; shortcut` | **O(lg n)** — theorem 4.19, matching the Ω(lg n) lower bound, so asymptotically optimal |
| **R2** | `parent-root-connect; shortcut ×2` | same O(lg n); section 4.3 notes this variant admits a simpler analysis with better constants |

P and S use the **same** connect step; they differ only in how much they compress per round. R and R2 change the connect step: `parent-root-connect` hooks a vertex's parent onto a smaller neighbour **only when that parent is itself a root**. That restriction makes the algorithm monotone, which is what the O(lg n) proof rests on.

## What the words mean

| term | meaning |
|---|---|
| **union-find call** | one launch of the per-leaf counting kernel over one grid. It initialises every active voxel to its own label, runs the schedule until convergence, and counts roots. **This is the unit everything below is measured in**, timed with a CUDA event pair around the launch. |
| **case** | one (mesh, voxel size) pair. There are 188 of them that ran. |
| **placement** | where the mesh sits on the voxel lattice. A voxel's label is its offset `64x + 8y + z`, fixed by position, so moving the geometry rewrites every label without touching the surface — and that alone can change the round count. Each case is run under three: identity, mirrored in all three axes, and rotated 45° about z. |
| **round** | one pass of the main loop: a connect step plus its compress step(s). A leaf's round count is the pass on which its labels stopped changing. |
| **compress step** | one pointer-jump, `parent[v] = parent[parent[v]]`. This is the work the four schedules actually differ in. |
| **warm-up** | one connect step — the schedule's own — plus four unconditional compresses, run before the main loop. A leaf is 8 voxels across, so four pointer-jumps flatten whatever the first connect builds. P, R and R2 all have it; S does not, because compressing to flatness every round already subsumes it. |

## The short answer

| | **H** | **P** | **S** | **R** | **R2** |
|---|---:|---:|---:|---:|---:|
| **time of one union-find call, typical case** | **0.28 ms** | **0.28 ms** | **0.36 ms** | **0.29 ms** | **0.28 ms** |
| **relative to H, typical case** | **1.00x** | **1.00x** | **1.28x** | **1.02x** | **1.02x** |
| time of one union-find call, largest case (hairball @ 0.004) | 200 ms | 201 ms | 256 ms | 205 ms | 204 ms |
| time of one union-find call, smallest case | 20 µs | 20 µs | 20 µs | 20 µs | 20 µs |
| rounds needed by the worst leaf anywhere | 6 | 6 | 6 | 6 | 5 |
| compress steps, relative to H | 1.00x | 1.00x | 1.19x | 1.00x | 1.24x |
| cases where it was fastest, of 188 (ties counted for each) | 130 | 106 | 11 | 47 | 54 |

Read the first two rows as: *on a typical mesh, one leaf union-find call takes about 0.28 ms, and switching schedule changes that by the factor shown.* Times scale with the mesh — the same call is 20 µs on the smallest case here and 201 ms on the largest — so the ratio, not the absolute figure, is what transfers between meshes.

**With the same warm-up, H, P, R and R2 are indistinguishable, and S costs 28% more.**

That is a different answer from an earlier version of this measurement, which had R 27% behind. The gap was an artifact of the comparison: only P had a warm-up then, so R was being asked to start from an unflattened forest. Given the same start, the difference disappears.

The compress counts say why. P and R differ by 0.06%. After four warm-up compresses almost every vertex is its own tree's root, so `parent-root-connect`'s extra test (`v.o = v.o.o`) is satisfied nearly always and the restriction stops costing anything. It only bites on deep forests, and after a warm-up there are none.

S is slower for a plain reason: 19% more compress steps for no fewer rounds. Compressing to flatness every round is work spent on a forest that was already nearly flat.

**This is what makes the shipped arrangement cheap.** R is the only schedule here whose bound matches the Ω(lg n) lower bound, and even as a pure schedule it costs only 2% over 188 cases (1.022x averaged rather than at the median). H does better still, by not paying even that -- see below.

### Does the fallback ever engage?

No. H and P executed exactly the same number of compress steps in 188 of the 188 cases and settled in the same number of rounds in 188. The two schedules diverge only once a leaf reaches `LeafUnionFind::SwitchToRootAfter` rounds, so identical counts mean the parent-root-connect phase was never entered: on this corpus the shipped schedule *is* P, instruction for instruction, and the difference in their measured times (0%) is run-to-run noise.

That is the point of the arrangement. The bound is insurance on inputs not seen here, and insurance that is never claimed costs nothing.

All five agreed on the component count in every case, so none of this trades correctness for speed.

## What is inside the timer

Labeling a grid runs four stages, and only the first two depend on the schedule:

| stage | depends on the schedule? | timed here |
|---|---|---|
| per-leaf component **count** — init, the union-find, count roots | yes | **yes — this is the union-find call** |
| per-leaf component **mask fill** — repeats the union-find, then scatters masks and face flags | partly | no |
| cross-leaf edge gathering | no | no |
| global union-find and label scatter | no | no |

The counting kernel runs the union-find and essentially nothing else, which is why it is the one timed. Timing the full labeling call instead would dilute the difference between schedules by roughly the share the counting kernel occupies in it. Measured that way, S's 28% shrinks to 11%, because the counting kernel is only 20% of the call.

## How each case was run

- The mesh is rasterized to a narrow band, then labeled once per schedule.
- Both grids the pipeline labels are covered: the **un-pruned** band, which the surface partition runs on, and the **barrier-pruned** band, which each surface runs on. A reported time is one call over each, added.
- Each case runs under the three placements described above. Times below are the mean over them; rounds are the worst over them.
- Every timing is a CUDA event pair around the launch, after a discarded warm-up run. Nothing else is inside the timer, and nothing is averaged over repeats.
- Rounds exclude the warm-up; compress steps include its four.

## Every case

`rd` = rounds the worst leaf needed. `ms` = one union-find call, mean over the three placements. Lower is better in both.

| mesh | voxel size | leaves | H rd | H ms | P rd | P ms | S rd | S ms | R rd | R ms | R2 rd | R2 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| armadillo | 0.02 | 151 | 5 | 0.020 | 5 | 0.023 | 4 | 0.027 | 5 | 0.020 | 3 | 0.020 |
| armadillo | 0.008 | 802 | 5 | 0.050 | 5 | 0.050 | 4 | 0.060 | 5 | 0.050 | 3 | 0.050 |
| armadillo | 0.004 | 3,150 | 5 | 0.147 | 5 | 0.147 | 4 | 0.187 | 5 | 0.150 | 3 | 0.150 |
| boat | 0.02 | 3,004 | 5 | 0.127 | 5 | 0.130 | 4 | 0.163 | 5 | 0.130 | 3 | 0.133 |
| boat | 0.008 | 19,434 | 5 | 0.760 | 5 | 0.787 | 4 | 1.007 | 5 | 0.790 | 4 | 0.793 |
| boat | 0.004 | 77,816 | 5 | 3.223 | 5 | 3.257 | 5 | 4.290 | 5 | 3.273 | 4 | 3.503 |
| boot | 0.02 | 22 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| boot | 0.008 | 72 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| boot | 0.004 | 246 | 4 | 0.027 | 4 | 0.030 | 4 | 0.030 | 5 | 0.030 | 3 | 0.027 |
| brucewick | 0.02 | 1,070 | 5 | 0.060 | 5 | 0.060 | 4 | 0.073 | 5 | 0.060 | 3 | 0.060 |
| brucewick | 0.008 | 6,822 | 5 | 0.307 | 5 | 0.303 | 4 | 0.373 | 5 | 0.307 | 3 | 0.307 |
| brucewick | 0.004 | 27,653 | 5 | 1.150 | 5 | 1.143 | 4 | 1.473 | 5 | 1.203 | 3 | 1.187 |
| bunny | 0.02 | 454 | 5 | 0.040 | 5 | 0.043 | 4 | 0.040 | 5 | 0.040 | 3 | 0.040 |
| bunny | 0.008 | 2,822 | 5 | 0.140 | 5 | 0.133 | 4 | 0.170 | 5 | 0.137 | 4 | 0.133 |
| bunny | 0.004 | 11,298 | 5 | 0.497 | 5 | 0.493 | 4 | 0.647 | 5 | 0.523 | 4 | 0.520 |
| bunny_hr | 0.02 | 462 | 5 | 0.040 | 5 | 0.040 | 4 | 0.040 | 5 | 0.040 | 3 | 0.040 |
| bunny_hr | 0.008 | 2,854 | 5 | 0.143 | 5 | 0.140 | 4 | 0.177 | 5 | 0.143 | 3 | 0.143 |
| bunny_hr | 0.004 | 11,491 | 5 | 0.543 | 5 | 0.543 | 4 | 0.667 | 5 | 0.550 | 3 | 0.550 |
| cat-low-resolution | 0.02 | 1,017 | 5 | 0.063 | 5 | 0.060 | 4 | 0.070 | 5 | 0.060 | 3 | 0.060 |
| cat-low-resolution | 0.008 | 6,478 | 5 | 0.283 | 5 | 0.287 | 4 | 0.367 | 5 | 0.290 | 3 | 0.293 |
| cat-low-resolution | 0.004 | 26,138 | 5 | 1.187 | 5 | 1.180 | 4 | 1.550 | 5 | 1.233 | 3 | 1.220 |
| cat | 0.02 | 1,021 | 5 | 0.083 | 5 | 0.083 | 4 | 0.103 | 5 | 0.087 | 3 | 0.083 |
| cat | 0.008 | 6,536 | 5 | 0.433 | 5 | 0.437 | 4 | 0.573 | 5 | 0.447 | 3 | 0.443 |
| cat | 0.004 | 26,431 | 5 | 1.670 | 5 | 1.677 | 4 | 2.220 | 5 | 1.707 | 4 | 1.717 |
| cheese | 0.02 | 3,646 | 5 | 0.243 | 5 | 0.243 | 4 | 0.317 | 5 | 0.243 | 3 | 0.250 |
| cheese | 0.008 | 22,816 | 5 | 1.357 | 5 | 1.360 | 5 | 1.803 | 5 | 1.383 | 4 | 1.413 |
| cheese | 0.004 | 91,591 | 5 | 5.547 | 5 | 5.597 | 5 | 6.690 | 5 | 5.263 | 4 | 5.423 |
| cow-low-resolution | 0.02 | 8,180 | 5 | 0.360 | 5 | 0.357 | 4 | 0.460 | 5 | 0.363 | 3 | 0.363 |
| cow-low-resolution | 0.008 | 52,437 | 5 | 2.433 | 5 | 2.570 | 4 | 3.243 | 5 | 2.573 | 3 | 2.640 |
| cow-low-resolution | 0.004 | 211,338 | 5 | 10.227 | 5 | 10.093 | 4 | 12.723 | 5 | 10.753 | 4 | 11.570 |
| cow | 0.02 | 8,286 | 5 | 0.433 | 5 | 0.430 | 4 | 0.550 | 5 | 0.457 | 3 | 0.437 |
| cow | 0.008 | 53,342 | 5 | 2.953 | 5 | 2.907 | 5 | 3.700 | 5 | 2.877 | 4 | 2.773 |
| cow | 0.004 | 215,332 | 5 | 11.017 | 5 | 11.037 | 5 | 14.230 | 5 | 11.493 | 4 | 11.780 |
| cube | 0.02 | 1,960 | 4 | 0.077 | 4 | 0.077 | 3 | 0.103 | 4 | 0.083 | 3 | 0.083 |
| cube | 0.008 | 10,128 | 4 | 0.277 | 4 | 0.280 | 3 | 0.373 | 4 | 0.287 | 3 | 0.307 |
| cube | 0.004 | 48,240 | 4 | 1.677 | 4 | 1.750 | 3 | 2.350 | 4 | 1.830 | 3 | 1.970 |
| cube_no_bottom | 0.02 | 1,722 | 4 | 0.077 | 4 | 0.073 | 3 | 0.093 | 4 | 0.073 | 3 | 0.077 |
| cube_no_bottom | 0.008 | 8,268 | 4 | 0.240 | 4 | 0.240 | 4 | 0.320 | 4 | 0.243 | 3 | 0.260 |
| cube_no_bottom | 0.004 | 40,560 | 4 | 1.447 | 4 | 1.440 | 3 | 2.027 | 4 | 1.477 | 3 | 1.650 |
| demosthenes-low-res | 0.02 | 920 | 5 | 0.060 | 5 | 0.060 | 4 | 0.070 | 5 | 0.060 | 4 | 0.060 |
| demosthenes-low-res | 0.008 | 5,870 | 5 | 0.270 | 5 | 0.270 | 4 | 0.333 | 5 | 0.270 | 4 | 0.270 |
| demosthenes-low-res | 0.004 | 23,518 | 5 | 1.077 | 5 | 1.090 | 4 | 1.387 | 5 | 1.137 | 3 | 1.100 |
| demosthenes | 0.02 | 930 | 5 | 0.063 | 5 | 0.063 | 4 | 0.080 | 5 | 0.067 | 4 | 0.067 |
| demosthenes | 0.008 | 5,935 | 5 | 0.313 | 5 | 0.313 | 5 | 0.397 | 5 | 0.317 | 4 | 0.317 |
| demosthenes | 0.004 | 23,892 | 5 | 1.213 | 5 | 1.237 | 5 | 1.543 | 5 | 1.217 | 4 | 1.200 |
| dragon | 0.02 | 125 | 4 | 0.027 | 4 | 0.023 | 4 | 0.030 | 4 | 0.030 | 3 | 0.027 |
| dragon | 0.008 | 744 | 5 | 0.060 | 5 | 0.060 | 4 | 0.077 | 5 | 0.060 | 3 | 0.060 |
| dragon | 0.004 | 3,270 | 5 | 0.190 | 5 | 0.193 | 5 | 0.247 | 5 | 0.197 | 4 | 0.193 |
| falconstatue | 0.02 | 411 | 5 | 0.037 | 5 | 0.037 | 4 | 0.040 | 5 | 0.037 | 3 | 0.040 |
| falconstatue | 0.008 | 2,691 | 5 | 0.130 | 5 | 0.130 | 5 | 0.167 | 5 | 0.137 | 4 | 0.137 |
| falconstatue | 0.004 | 11,036 | 5 | 0.500 | 5 | 0.503 | 5 | 0.640 | 5 | 0.517 | 4 | 0.527 |
| falconstatue_boundary | 0.02 | 408 | 5 | 0.037 | 5 | 0.037 | 4 | 0.040 | 5 | 0.037 | 3 | 0.033 |
| falconstatue_boundary | 0.008 | 2,635 | 5 | 0.133 | 5 | 0.133 | 5 | 0.173 | 5 | 0.140 | 4 | 0.137 |
| falconstatue_boundary | 0.004 | 10,698 | 5 | 0.473 | 5 | 0.483 | 5 | 0.623 | 5 | 0.497 | 4 | 0.503 |
| fish | 0.02 | 704 | 5 | 0.050 | 5 | 0.050 | 5 | 0.060 | 5 | 0.050 | 4 | 0.060 |
| fish | 0.008 | 4,568 | 5 | 0.213 | 5 | 0.210 | 4 | 0.263 | 5 | 0.213 | 3 | 0.213 |
| fish | 0.004 | 18,368 | 5 | 0.770 | 5 | 0.767 | 4 | 0.987 | 5 | 0.803 | 3 | 0.793 |
| fish_control_mesh | 0.02 | 859 | 5 | 0.050 | 5 | 0.053 | 4 | 0.063 | 5 | 0.053 | 3 | 0.053 |
| fish_control_mesh | 0.008 | 5,637 | 5 | 0.250 | 5 | 0.247 | 4 | 0.320 | 5 | 0.253 | 4 | 0.250 |
| fish_control_mesh | 0.004 | 23,399 | 5 | 0.997 | 5 | 1.010 | 5 | 1.303 | 5 | 1.020 | 4 | 1.020 |
| fish_low_resolution | 0.02 | 649 | 4 | 0.043 | 4 | 0.047 | 4 | 0.050 | 4 | 0.047 | 3 | 0.043 |
| fish_low_resolution | 0.008 | 3,975 | 5 | 0.280 | 5 | 0.280 | 4 | 0.353 | 5 | 0.283 | 3 | 0.277 |
| fish_low_resolution | 0.004 | 16,337 | 5 | 1.033 | 5 | 1.033 | 4 | 1.330 | 5 | 1.047 | 3 | 1.040 |
| goathead | 0.02 | 30,441 | 5 | 1.937 | 5 | 1.927 | 4 | 2.523 | 5 | 1.957 | 3 | 1.960 |
| goathead | 0.008 | 190,762 | 5 | 12.343 | 5 | 12.380 | 4 | 16.063 | 5 | 12.587 | 3 | 12.610 |
| goathead | 0.004 | 763,519 | 5 | 39.913 | 5 | 38.227 | 4 | 47.153 | 5 | 40.723 | 4 | 46.537 |
| hairball | 0.02 | 89,812 | 6 | 5.603 | 6 | 5.290 | 5 | 6.947 | 6 | 5.277 | 4 | 5.217 |
| hairball | 0.008 | 879,909 | 5 | 48.280 | 5 | 48.267 | 6 | 63.430 | 6 | 51.903 | 5 | 52.827 |
| hairball | 0.004 | 3,866,231 | 6 | 200.283 | 6 | 201.337 | 5 | 255.960 | 6 | 204.530 | 4 | 203.573 |
| hammer | 0.02 | 1,704 | 4 | 0.083 | 4 | 0.083 | 4 | 0.103 | 4 | 0.087 | 3 | 0.087 |
| hammer | 0.008 | 10,842 | 4 | 0.463 | 4 | 0.457 | 4 | 0.620 | 4 | 0.487 | 3 | 0.520 |
| hammer | 0.004 | 44,135 | 5 | 1.863 | 5 | 1.903 | 4 | 2.560 | 5 | 1.963 | 3 | 2.023 |
| hand | 0.02 | 77 | 4 | 0.023 | 4 | 0.023 | 4 | 0.030 | 4 | 0.023 | 3 | 0.020 |
| hand | 0.008 | 394 | 4 | 0.033 | 4 | 0.033 | 4 | 0.040 | 4 | 0.033 | 3 | 0.033 |
| hand | 0.004 | 1,554 | 5 | 0.090 | 5 | 0.087 | 4 | 0.110 | 5 | 0.090 | 3 | 0.087 |
| hand_closed | 0.02 | 77 | 4 | 0.020 | 4 | 0.020 | 4 | 0.023 | 4 | 0.020 | 3 | 0.020 |
| hand_closed | 0.008 | 400 | 4 | 0.030 | 4 | 0.033 | 4 | 0.040 | 4 | 0.033 | 3 | 0.033 |
| hand_closed | 0.004 | 1,596 | 5 | 0.087 | 5 | 0.083 | 4 | 0.110 | 5 | 0.087 | 3 | 0.087 |
| hand_lowres | 0.02 | 79 | 4 | 0.020 | 4 | 0.020 | 4 | 0.023 | 4 | 0.020 | 3 | 0.020 |
| hand_lowres | 0.008 | 406 | 4 | 0.033 | 4 | 0.037 | 4 | 0.040 | 5 | 0.040 | 3 | 0.040 |
| hand_lowres | 0.004 | 1,598 | 5 | 0.083 | 5 | 0.083 | 4 | 0.103 | 5 | 0.083 | 3 | 0.083 |
| house | 0.02 | — | — | — | — | — | — | — | — | — | — | — |
| house | 0.008 | — | — | — | — | — | — | — | — | — | — | — |
| house | 0.004 | — | — | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.02 | — | — | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.008 | — | — | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.004 | — | — | — | — | — | — | — | — | — | — | — |
| human_man | 0.02 | 171 | 4 | 0.030 | 4 | 0.027 | 4 | 0.030 | 4 | 0.030 | 3 | 0.027 |
| human_man | 0.008 | 989 | 5 | 0.060 | 5 | 0.057 | 4 | 0.073 | 5 | 0.060 | 3 | 0.060 |
| human_man | 0.004 | 3,877 | 5 | 0.173 | 5 | 0.177 | 4 | 0.227 | 5 | 0.177 | 4 | 0.180 |
| human_neutral | 0.02 | 169 | 4 | 0.027 | 4 | 0.020 | 4 | 0.030 | 4 | 0.020 | 3 | 0.020 |
| human_neutral | 0.008 | 884 | 5 | 0.050 | 5 | 0.053 | 4 | 0.063 | 5 | 0.053 | 3 | 0.053 |
| human_neutral | 0.004 | 3,631 | 5 | 0.163 | 5 | 0.163 | 4 | 0.213 | 5 | 0.167 | 3 | 0.167 |
| human_woman | 0.02 | 148 | 4 | 0.020 | 4 | 0.020 | 4 | 0.027 | 4 | 0.023 | 3 | 0.020 |
| human_woman | 0.008 | 848 | 5 | 0.060 | 5 | 0.053 | 4 | 0.070 | 5 | 0.053 | 3 | 0.060 |
| human_woman | 0.004 | 3,401 | 5 | 0.153 | 5 | 0.153 | 4 | 0.193 | 5 | 0.153 | 4 | 0.153 |
| koala | 0.02 | 9,070 | 5 | 0.423 | 5 | 0.437 | 4 | 0.577 | 5 | 0.450 | 3 | 0.470 |
| koala | 0.008 | 56,842 | 5 | 2.627 | 5 | 2.703 | 4 | 3.403 | 5 | 2.687 | 3 | 2.760 |
| koala | 0.004 | 225,890 | 5 | 12.103 | 5 | 11.777 | 4 | 14.327 | 5 | 11.313 | 3 | 11.617 |
| koala_low_resolution | 0.02 | 8,876 | 5 | 0.383 | 5 | 0.383 | 4 | 0.490 | 5 | 0.390 | 3 | 0.390 |
| koala_low_resolution | 0.008 | 55,880 | 5 | 2.453 | 5 | 2.577 | 4 | 3.273 | 5 | 2.633 | 3 | 2.663 |
| koala_low_resolution | 0.004 | 223,453 | 5 | 10.820 | 5 | 10.673 | 4 | 13.593 | 5 | 11.243 | 4 | 11.873 |
| lionstatue | 0.02 | 666 | 5 | 0.047 | 5 | 0.050 | 4 | 0.057 | 5 | 0.050 | 3 | 0.047 |
| lionstatue | 0.008 | 3,942 | 5 | 0.193 | 5 | 0.193 | 5 | 0.250 | 5 | 0.197 | 4 | 0.200 |
| lionstatue | 0.004 | 16,036 | 5 | 0.707 | 5 | 0.710 | 4 | 0.920 | 5 | 0.720 | 4 | 0.713 |
| mountain | 0.02 | 123,940 | 5 | 5.890 | 5 | 5.970 | 4 | 7.473 | 5 | 5.997 | 3 | 6.210 |
| mountain | 0.008 | 768,838 | 5 | 38.730 | 5 | 38.690 | 4 | 47.607 | 5 | 41.283 | 3 | 45.580 |
| mountain | 0.004 | 3,068,538 | 5 | 153.643 | 5 | 156.980 | 4 | 193.440 | 5 | 156.913 | 4 | 158.680 |
| mushroom | 0.02 | 2,371 | 5 | 0.117 | 5 | 0.123 | 4 | 0.153 | 5 | 0.123 | 3 | 0.123 |
| mushroom | 0.008 | 15,124 | 5 | 0.677 | 5 | 0.673 | 4 | 0.877 | 5 | 0.700 | 3 | 0.713 |
| mushroom | 0.004 | 60,698 | 5 | 2.943 | 5 | 2.963 | 4 | 3.710 | 5 | 3.113 | 3 | 3.077 |
| nefertiti-lowres | 0.02 | — | — | — | — | — | — | — | — | — | — | — |
| nefertiti-lowres | 0.008 | — | — | — | — | — | — | — | — | — | — | — |
| nefertiti-lowres | 0.004 | — | — | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.02 | — | — | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.008 | — | — | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.004 | — | — | — | — | — | — | — | — | — | — | — |
| parsnip | 0.02 | 1,445 | 4 | 0.093 | 4 | 0.093 | 4 | 0.113 | 5 | 0.093 | 3 | 0.093 |
| parsnip | 0.008 | 9,181 | 5 | 0.427 | 5 | 0.440 | 5 | 0.580 | 5 | 0.453 | 4 | 0.457 |
| parsnip | 0.004 | 37,087 | 5 | 1.820 | 5 | 1.857 | 4 | 2.487 | 5 | 1.830 | 3 | 1.853 |
| penguin | 0.02 | 930 | 5 | 0.057 | 5 | 0.060 | 4 | 0.070 | 5 | 0.060 | 3 | 0.060 |
| penguin | 0.008 | 6,165 | 5 | 0.277 | 5 | 0.277 | 4 | 0.350 | 5 | 0.280 | 3 | 0.280 |
| penguin | 0.004 | 25,661 | 5 | 1.113 | 5 | 1.140 | 4 | 1.470 | 5 | 1.160 | 3 | 1.207 |
| penguin_control_mesh | 0.02 | 1,099 | 5 | 0.063 | 5 | 0.067 | 4 | 0.083 | 5 | 0.067 | 3 | 0.070 |
| penguin_control_mesh | 0.008 | 7,637 | 5 | 0.337 | 5 | 0.333 | 4 | 0.430 | 5 | 0.347 | 3 | 0.343 |
| penguin_control_mesh | 0.004 | 32,083 | 5 | 1.440 | 5 | 1.453 | 4 | 1.860 | 5 | 1.477 | 4 | 1.507 |
| penguin_hr | 0.02 | 927 | 5 | 0.060 | 5 | 0.060 | 4 | 0.070 | 5 | 0.060 | 3 | 0.060 |
| penguin_hr | 0.008 | 6,128 | 5 | 0.277 | 5 | 0.277 | 4 | 0.357 | 5 | 0.283 | 3 | 0.283 |
| penguin_hr | 0.004 | 25,418 | 5 | 1.157 | 5 | 1.187 | 4 | 1.480 | 5 | 1.180 | 3 | 1.183 |
| pizza | 0.02 | 132 | 4 | 0.020 | 4 | 0.023 | 4 | 0.023 | 4 | 0.023 | 3 | 0.023 |
| pizza | 0.008 | 749 | 4 | 0.050 | 4 | 0.050 | 4 | 0.060 | 5 | 0.050 | 3 | 0.050 |
| pizza | 0.004 | 2,985 | 5 | 0.143 | 5 | 0.143 | 5 | 0.193 | 5 | 0.150 | 4 | 0.150 |
| plane | 0.02 | 9,565 | 5 | 0.383 | 5 | 0.387 | 4 | 0.510 | 5 | 0.397 | 4 | 0.403 |
| plane | 0.008 | 60,792 | 5 | 2.697 | 5 | 2.720 | 4 | 3.567 | 5 | 2.743 | 4 | 2.853 |
| plane | 0.004 | 243,852 | 5 | 11.193 | 5 | 11.497 | 4 | 15.123 | 5 | 12.193 | 4 | 12.753 |
| plane_holes | 0.02 | 9,345 | 5 | 0.387 | 5 | 0.390 | 4 | 0.513 | 5 | 0.397 | 4 | 0.407 |
| plane_holes | 0.008 | 58,231 | 5 | 2.513 | 5 | 2.623 | 4 | 3.457 | 5 | 2.677 | 4 | 2.777 |
| plane_holes | 0.004 | 231,378 | 5 | 10.897 | 5 | 10.650 | 4 | 13.820 | 5 | 11.540 | 4 | 12.143 |
| scorpion | 0.02 | 6,174 | 5 | 0.300 | 5 | 0.307 | 5 | 0.390 | 5 | 0.313 | 4 | 0.310 |
| scorpion | 0.008 | 39,895 | 5 | 1.883 | 5 | 1.920 | 4 | 2.443 | 5 | 1.947 | 4 | 1.923 |
| scorpion | 0.004 | 160,853 | 5 | 8.057 | 5 | 8.143 | 5 | 9.617 | 5 | 8.243 | 4 | 8.397 |
| scorpion_low_resolution | 0.02 | 6,217 | 5 | 0.280 | 5 | 0.283 | 5 | 0.357 | 5 | 0.287 | 4 | 0.280 |
| scorpion_low_resolution | 0.008 | 40,227 | 5 | 1.813 | 5 | 1.890 | 4 | 2.417 | 5 | 1.930 | 3 | 1.940 |
| scorpion_low_resolution | 0.004 | 161,445 | 5 | 7.763 | 5 | 7.927 | 4 | 9.717 | 5 | 7.987 | 4 | 8.007 |
| skull | 0.02 | 40,486 | 5 | 1.877 | 5 | 1.933 | 4 | 2.427 | 5 | 1.977 | 4 | 2.023 |
| skull | 0.008 | 254,923 | 5 | 13.717 | 5 | 12.573 | 4 | 15.473 | 5 | 12.493 | 3 | 12.080 |
| skull | 0.004 | 1,021,048 | 5 | 53.283 | 5 | 53.303 | 4 | 69.397 | 5 | 52.697 | 3 | 54.590 |
| skull_low_resolution | 0.02 | 40,127 | 5 | 1.843 | 5 | 1.860 | 4 | 2.377 | 5 | 1.900 | 4 | 1.937 |
| skull_low_resolution | 0.008 | 252,742 | 5 | 12.387 | 5 | 12.070 | 4 | 16.097 | 5 | 13.480 | 4 | 13.470 |
| skull_low_resolution | 0.004 | 1,012,874 | 5 | 52.633 | 5 | 52.620 | 4 | 66.803 | 5 | 55.800 | 4 | 53.683 |
| sphere | 0.02 | 1,038 | 5 | 0.060 | 5 | 0.060 | 3 | 0.077 | 5 | 0.060 | 3 | 0.063 |
| sphere | 0.008 | 6,299 | 5 | 0.283 | 5 | 0.283 | 3 | 0.357 | 5 | 0.290 | 3 | 0.290 |
| sphere | 0.004 | 25,296 | 5 | 1.157 | 5 | 1.153 | 3 | 1.477 | 5 | 1.180 | 3 | 1.170 |
| spot | 0.02 | 438 | 5 | 0.040 | 5 | 0.040 | 4 | 0.040 | 5 | 0.040 | 3 | 0.040 |
| spot | 0.008 | 2,774 | 5 | 0.130 | 5 | 0.133 | 4 | 0.163 | 5 | 0.130 | 3 | 0.133 |
| spot | 0.004 | 11,155 | 5 | 0.483 | 5 | 0.483 | 4 | 0.627 | 5 | 0.490 | 3 | 0.497 |
| spot_control_mesh | 0.02 | 582 | 4 | 0.040 | 4 | 0.040 | 4 | 0.053 | 4 | 0.043 | 3 | 0.047 |
| spot_control_mesh | 0.008 | 3,829 | 5 | 0.167 | 5 | 0.167 | 5 | 0.223 | 5 | 0.173 | 4 | 0.173 |
| spot_control_mesh | 0.004 | 15,499 | 5 | 0.663 | 5 | 0.660 | 5 | 0.853 | 5 | 0.673 | 4 | 0.680 |
| spot_low_resolution | 0.02 | 436 | 4 | 0.037 | 4 | 0.037 | 4 | 0.040 | 4 | 0.040 | 3 | 0.040 |
| spot_low_resolution | 0.008 | 2,752 | 5 | 0.127 | 5 | 0.130 | 4 | 0.163 | 5 | 0.133 | 3 | 0.130 |
| spot_low_resolution | 0.004 | 11,023 | 5 | 0.480 | 5 | 0.483 | 4 | 0.607 | 5 | 0.487 | 3 | 0.487 |
| springer | 0.02 | 2,591 | 5 | 0.123 | 5 | 0.123 | 4 | 0.153 | 5 | 0.123 | 3 | 0.123 |
| springer | 0.008 | 16,543 | 5 | 0.710 | 5 | 0.720 | 4 | 0.920 | 5 | 0.737 | 3 | 0.737 |
| springer | 0.004 | 66,240 | 5 | 3.060 | 5 | 3.220 | 4 | 3.977 | 5 | 3.287 | 3 | 3.400 |
| strawberry | 0.02 | 979 | 5 | 0.073 | 5 | 0.070 | 4 | 0.090 | 5 | 0.070 | 3 | 0.070 |
| strawberry | 0.008 | 6,509 | 5 | 0.357 | 5 | 0.363 | 4 | 0.457 | 5 | 0.363 | 4 | 0.370 |
| strawberry | 0.004 | 26,762 | 5 | 1.350 | 5 | 1.363 | 4 | 1.767 | 5 | 1.373 | 3 | 1.343 |
| stuffedtoy | 0.02 | 97 | 4 | 0.023 | 4 | 0.020 | 4 | 0.027 | 4 | 0.020 | 3 | 0.020 |
| stuffedtoy | 0.008 | 555 | 5 | 0.040 | 5 | 0.040 | 4 | 0.050 | 5 | 0.047 | 3 | 0.040 |
| stuffedtoy | 0.004 | 2,309 | 5 | 0.113 | 5 | 0.113 | 4 | 0.153 | 5 | 0.120 | 3 | 0.117 |
| sword-quad-dominant | 0.02 | 34 | 3 | 0.020 | 3 | 0.020 | 4 | 0.020 | 3 | 0.020 | 3 | 0.020 |
| sword-quad-dominant | 0.008 | 80 | 4 | 0.023 | 4 | 0.020 | 4 | 0.027 | 4 | 0.020 | 3 | 0.020 |
| sword-quad-dominant | 0.004 | 197 | 4 | 0.030 | 4 | 0.030 | 4 | 0.030 | 4 | 0.030 | 3 | 0.030 |
| sword | 0.02 | 34 | 3 | 0.020 | 3 | 0.020 | 4 | 0.020 | 3 | 0.020 | 3 | 0.020 |
| sword | 0.008 | 80 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| sword | 0.004 | 197 | 4 | 0.030 | 4 | 0.030 | 4 | 0.030 | 4 | 0.030 | 3 | 0.030 |
| torus | 0.02 | 800 | 4 | 0.050 | 4 | 0.050 | 4 | 0.060 | 4 | 0.050 | 3 | 0.050 |
| torus | 0.008 | 4,971 | 5 | 0.220 | 5 | 0.227 | 3 | 0.283 | 5 | 0.230 | 3 | 0.233 |
| torus | 0.004 | 20,025 | 5 | 0.863 | 5 | 0.873 | 4 | 1.130 | 5 | 0.887 | 3 | 0.890 |
| tower | 0.02 | 5,713 | 5 | 0.303 | 5 | 0.300 | 4 | 0.400 | 5 | 0.307 | 3 | 0.310 |
| tower | 0.008 | 37,228 | 5 | 2.243 | 5 | 2.287 | 5 | 3.030 | 5 | 2.323 | 4 | 2.347 |
| tower | 0.004 | 152,469 | 5 | 7.930 | 5 | 7.567 | 5 | 9.783 | 5 | 7.350 | 4 | 7.127 |
| tower_holes | 0.02 | 4,740 | 5 | 0.247 | 5 | 0.247 | 5 | 0.310 | 5 | 0.247 | 4 | 0.247 |
| tower_holes | 0.008 | 29,371 | 5 | 1.360 | 5 | 1.353 | 5 | 1.757 | 6 | 1.360 | 4 | 1.420 |
| tower_holes | 0.004 | 118,862 | 5 | 5.757 | 5 | 5.643 | 4 | 7.147 | 5 | 5.700 | 4 | 5.713 |
| tree | 0.02 | 457 | 5 | 0.040 | 5 | 0.040 | 4 | 0.050 | 5 | 0.043 | 3 | 0.043 |
| tree | 0.008 | 2,351 | 5 | 0.147 | 5 | 0.150 | 4 | 0.187 | 5 | 0.150 | 3 | 0.150 |
| tree | 0.004 | 9,295 | 5 | 0.503 | 5 | 0.503 | 4 | 0.647 | 5 | 0.513 | 3 | 0.510 |
| tree_closed | 0.02 | 459 | 5 | 0.040 | 5 | 0.040 | 4 | 0.050 | 5 | 0.040 | 3 | 0.040 |
| tree_closed | 0.008 | 2,410 | 5 | 0.147 | 5 | 0.147 | 4 | 0.187 | 5 | 0.150 | 3 | 0.150 |
| tree_closed | 0.004 | 9,616 | 5 | 0.517 | 5 | 0.507 | 4 | 0.660 | 5 | 0.520 | 3 | 0.517 |
| violin | 0.02 | 28 | 3 | 0.020 | 3 | 0.020 | 4 | 0.020 | 3 | 0.020 | 3 | 0.020 |
| violin | 0.008 | 90 | 4 | 0.023 | 4 | 0.023 | 4 | 0.027 | 4 | 0.020 | 3 | 0.027 |
| violin | 0.004 | 295 | 4 | 0.030 | 4 | 0.030 | 4 | 0.037 | 4 | 0.030 | 3 | 0.030 |
| well | 0.02 | 1,143,998 | 5 | 57.243 | 5 | 55.323 | 5 | 76.187 | 5 | 54.877 | 4 | 57.710 |
| well | 0.008 | — | — | — | — | — | — | — | — | — | — | — |
| well | 0.004 | — | — | — | — | — | — | — | — | — | — | — |
| well_boundary | 0.02 | 1,039,667 | 5 | 51.000 | 5 | 49.173 | 5 | 70.900 | 5 | 50.417 | 4 | 52.570 |
| well_boundary | 0.008 | — | — | — | — | — | — | — | — | — | — | — |
| well_boundary | 0.004 | — | — | — | — | — | — | — | — | — | — | — |
| wingnut | 0.02 | 5,099 | 5 | 0.207 | 5 | 0.207 | 4 | 0.273 | 5 | 0.210 | 4 | 0.213 |
| wingnut | 0.008 | 31,746 | 5 | 1.233 | 5 | 1.233 | 5 | 1.603 | 5 | 1.260 | 4 | 1.287 |
| wingnut | 0.004 | 125,910 | 5 | 5.583 | 5 | 5.410 | 4 | 6.997 | 5 | 5.463 | 4 | 5.750 |

16 of 204 cases did not run, covering house, house_boundary, nefertiti, nefertiti-lowres, well, well_boundary. Those meshes sit in coordinate systems tens to hundreds of times larger than the unit-scale ones, so a fixed voxel size asks for a grid thousands of voxels across and the run exhausts GPU memory before labeling begins. Not a convergence failure.

### Why the times span four orders of magnitude

Input size, in two regimes.

Above roughly ten thousand leaves (75 cases) the kernel is throughput-bound and the cost per leaf is flat: a median of 46 ns, with the 10th and 90th percentiles at 42 and 59 ns. A large number in the table therefore means a large mesh, not a schedule behaving badly.

Below about a thousand leaves (51 cases) the call sits on a launch-latency floor -- a median of 37 µs regardless of size -- so per-leaf figures there measure the launch, not the algorithm, and the schedules are indistinguishable by construction.

| case | leaves | one union-find call (P) | per leaf |
|---|---:|---:|---:|
| boot @ 0.02 (latency-bound) | 22 | 0.020 ms | 909 ns |
| skull_low_resolution @ 0.02 (throughput-bound) | 40,127 | 1.843 ms | 46 ns |
| hairball @ 0.004 (throughput-bound) | 3,866,231 | 200.283 ms | 52 ns |

This spread is also why no total is reported anywhere: summing across cases would hand the answer to the two or three largest meshes.

## What to take from it

- **A proven bound is free here.** R matches P to within 2%, so the schedule with no proven bound has no measured advantage left to defend.

- **R is the only optimal-bound algorithm we can adopt.** The other one the paper proves O(lg n) for, RA, needs edge alteration, and our edges are implicit in the 6-neighbourhood of the voxel lattice rather than a list we can rewrite.

- **No schedule lowers the worst case much.** The worst leaf anywhere needed 6 rounds under P and 6 under R, against a cap of 64. If the worry is the cap, the fix is detecting that it was reached, which nothing currently does.

- **Switching is a one-line call:** `ConnectedComponents::setLeafSchedule()`. All four share one solver, so the counting and mask passes cannot drift apart.
