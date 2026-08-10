# Four per-leaf union-find schedules, measured

Generated 2026-08-07 by `ex_mesh_to_sdf_cuda --sv-convergence <mesh> --voxel-size S`.

All four are from Liu & Tarjan, *Simple Concurrent Connected Components Algorithms* (ACM TOPC 9(2), 2022), which casts this family as a **connect** step followed by one or more **shortcut** steps, repeated until no parent changes.

| | main loop | proven step bound |
|---|---|---|
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

| | **P** | **S** | **R** | **R2** |
|---|---:|---:|---:|---:|
| **time of one union-find call, typical case** | **0.29 ms** | **0.36 ms** | **0.30 ms** | **0.29 ms** |
| **relative to P, typical case** | **1.00x** | **1.25x** | **1.00x** | **1.00x** |
| time of one union-find call, largest case (hairball @ 0.004) | 230 ms | 286 ms | 237 ms | 236 ms |
| time of one union-find call, smallest case | 17 µs | 20 µs | 17 µs | 17 µs |
| rounds needed by the worst leaf anywhere | 6 | 6 | 6 | 5 |
| compress steps, relative to P | 1.00x | 1.19x | 1.00x | 1.24x |
| cases where it was fastest, of 188 (ties counted for each) | 137 | 12 | 72 | 100 |

Read the first two rows as: *on a typical mesh, one leaf union-find call takes about 0.29 ms, and switching schedule changes that by the factor shown.* Times scale with the mesh — the same call is 17 µs on the smallest case here and 230 ms on the largest — so the ratio, not the absolute figure, is what transfers between meshes.

**With the same warm-up, P, R and R2 are indistinguishable, and S costs 25% more.**

That is a different answer from an earlier version of this measurement, which had R 27% behind. The gap was an artifact of the comparison: only P had a warm-up then, so R was being asked to start from an unflattened forest. Given the same start, the difference disappears.

The compress counts say why. P and R differ by 0.06%. After four warm-up compresses almost every vertex is its own tree's root, so `parent-root-connect`'s extra test (`v.o = v.o.o`) is satisfied nearly always and the restriction stops costing anything. It only bites on deep forests, and after a warm-up there are none.

S is slower for a plain reason: 19% more compress steps for no fewer rounds. Compressing to flatness every round is work spent on a forest that was already nearly flat.

**The practical consequence: R is adoptable at no measured cost.** It is the only schedule here whose bound matches the Ω(lg n) lower bound, and across 188 cases its median time is 1.000x P's -- the schedule that has no proven bound at all. Averaged over cases rather than taken at the median it is 1.013x, so the gap is under 2% either way.

All four agreed on the component count in every case, so none of this trades correctness for speed.

## What is inside the timer

Labeling a grid runs four stages, and only the first two depend on the schedule:

| stage | depends on the schedule? | timed here |
|---|---|---|
| per-leaf component **count** — init, the union-find, count roots | yes | **yes — this is the union-find call** |
| per-leaf component **mask fill** — repeats the union-find, then scatters masks and face flags | partly | no |
| cross-leaf edge gathering | no | no |
| global union-find and label scatter | no | no |

The counting kernel runs the union-find and essentially nothing else, which is why it is the one timed. Timing the full labeling call instead would dilute the difference between schedules by roughly the share the counting kernel occupies in it.

## How each case was run

- The mesh is rasterized to a narrow band, then labeled once per schedule.
- Both grids the pipeline labels are covered: the **un-pruned** band, which the surface partition runs on, and the **barrier-pruned** band, which each surface runs on. A reported time is one call over each, added.
- Each case runs under the three placements described above. Times below are the mean over them; rounds are the worst over them.
- Every timing is a CUDA event pair around the launch, after a discarded warm-up run. Nothing else is inside the timer, and nothing is averaged over repeats.
- Rounds exclude the warm-up; compress steps include its four.

## Every case

`rd` = rounds the worst leaf needed. `ms` = one union-find call, mean over the three placements. Lower is better in both.

| mesh | voxel size | leaves | P rd | P ms | S rd | S ms | R rd | R ms | R2 rd | R2 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| armadillo | 0.02 | 151 | 5 | 0.020 | 4 | 0.020 | 5 | 0.020 | 3 | 0.020 |
| armadillo | 0.008 | 802 | 5 | 0.050 | 4 | 0.060 | 5 | 0.050 | 3 | 0.050 |
| armadillo | 0.004 | 3,150 | 5 | 0.157 | 4 | 0.193 | 5 | 0.160 | 3 | 0.153 |
| boat | 0.02 | 3,004 | 5 | 0.130 | 4 | 0.167 | 5 | 0.130 | 3 | 0.137 |
| boat | 0.008 | 19,434 | 5 | 0.773 | 4 | 0.993 | 5 | 0.783 | 4 | 0.810 |
| boat | 0.004 | 77,816 | 5 | 3.243 | 5 | 4.290 | 5 | 3.390 | 4 | 3.447 |
| boot | 0.02 | 22 | 4 | 0.020 | 3 | 0.020 | 4 | 0.020 | 3 | 0.017 |
| boot | 0.008 | 72 | 4 | 0.023 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| boot | 0.004 | 246 | 4 | 0.030 | 4 | 0.030 | 5 | 0.030 | 3 | 0.027 |
| brucewick | 0.02 | 1,070 | 5 | 0.060 | 4 | 0.073 | 5 | 0.060 | 3 | 0.060 |
| brucewick | 0.008 | 6,822 | 5 | 0.310 | 4 | 0.383 | 5 | 0.313 | 3 | 0.313 |
| brucewick | 0.004 | 27,653 | 5 | 1.237 | 4 | 1.560 | 5 | 1.290 | 3 | 1.297 |
| bunny | 0.02 | 454 | 5 | 0.033 | 4 | 0.040 | 5 | 0.033 | 3 | 0.033 |
| bunny | 0.008 | 2,822 | 5 | 0.140 | 4 | 0.170 | 5 | 0.140 | 4 | 0.137 |
| bunny | 0.004 | 11,298 | 5 | 0.500 | 4 | 0.623 | 5 | 0.503 | 4 | 0.503 |
| bunny_hr | 0.02 | 462 | 5 | 0.040 | 4 | 0.043 | 5 | 0.037 | 3 | 0.037 |
| bunny_hr | 0.008 | 2,854 | 5 | 0.143 | 4 | 0.180 | 5 | 0.147 | 3 | 0.147 |
| bunny_hr | 0.004 | 11,491 | 5 | 0.577 | 4 | 0.710 | 5 | 0.597 | 3 | 0.593 |
| cat-low-resolution | 0.02 | 1,017 | 5 | 0.060 | 4 | 0.070 | 5 | 0.060 | 3 | 0.060 |
| cat-low-resolution | 0.008 | 6,478 | 5 | 0.293 | 4 | 0.370 | 5 | 0.300 | 3 | 0.300 |
| cat-low-resolution | 0.004 | 26,138 | 5 | 1.207 | 4 | 1.517 | 5 | 1.227 | 3 | 1.227 |
| cat | 0.02 | 1,021 | 5 | 0.090 | 4 | 0.107 | 5 | 0.087 | 3 | 0.087 |
| cat | 0.008 | 6,536 | 5 | 0.460 | 4 | 0.590 | 5 | 0.467 | 3 | 0.463 |
| cat | 0.004 | 26,431 | 5 | 1.773 | 4 | 2.313 | 5 | 1.830 | 4 | 1.823 |
| cheese | 0.02 | 3,646 | 5 | 0.253 | 4 | 0.323 | 5 | 0.253 | 3 | 0.263 |
| cheese | 0.008 | 22,816 | 5 | 1.440 | 5 | 1.863 | 5 | 1.467 | 4 | 1.493 |
| cheese | 0.004 | 91,591 | 5 | 5.890 | 5 | 7.587 | 5 | 6.037 | 4 | 6.137 |
| cow-low-resolution | 0.02 | 8,180 | 5 | 0.410 | 4 | 0.517 | 5 | 0.420 | 3 | 0.410 |
| cow-low-resolution | 0.008 | 52,437 | 5 | 2.590 | 4 | 3.360 | 5 | 2.683 | 3 | 2.677 |
| cow-low-resolution | 0.004 | 211,338 | 5 | 10.830 | 4 | 13.147 | 5 | 10.467 | 4 | 10.500 |
| cow | 0.02 | 8,286 | 5 | 0.450 | 4 | 0.563 | 5 | 0.457 | 3 | 0.447 |
| cow | 0.008 | 53,342 | 5 | 3.227 | 5 | 4.093 | 5 | 3.170 | 4 | 3.000 |
| cow | 0.004 | 215,332 | 5 | 12.487 | 5 | 14.563 | 5 | 11.270 | 4 | 11.533 |
| cube | 0.02 | 1,960 | 4 | 0.077 | 3 | 0.103 | 4 | 0.077 | 3 | 0.083 |
| cube | 0.008 | 10,128 | 4 | 0.287 | 3 | 0.377 | 4 | 0.297 | 3 | 0.307 |
| cube | 0.004 | 48,240 | 4 | 1.767 | 3 | 2.403 | 4 | 1.840 | 3 | 1.977 |
| cube_no_bottom | 0.02 | 1,722 | 4 | 0.073 | 3 | 0.093 | 4 | 0.073 | 3 | 0.073 |
| cube_no_bottom | 0.008 | 8,268 | 4 | 0.250 | 4 | 0.323 | 4 | 0.253 | 3 | 0.260 |
| cube_no_bottom | 0.004 | 40,560 | 4 | 1.463 | 3 | 2.020 | 4 | 1.543 | 3 | 1.643 |
| demosthenes-low-res | 0.02 | 920 | 5 | 0.060 | 4 | 0.070 | 5 | 0.060 | 4 | 0.060 |
| demosthenes-low-res | 0.008 | 5,870 | 5 | 0.293 | 4 | 0.353 | 5 | 0.293 | 4 | 0.290 |
| demosthenes-low-res | 0.004 | 23,518 | 5 | 1.080 | 4 | 1.327 | 5 | 1.103 | 3 | 1.120 |
| demosthenes | 0.02 | 930 | 5 | 0.070 | 4 | 0.090 | 5 | 0.070 | 4 | 0.070 |
| demosthenes | 0.008 | 5,935 | 5 | 0.333 | 5 | 0.407 | 5 | 0.327 | 4 | 0.330 |
| demosthenes | 0.004 | 23,892 | 5 | 1.357 | 5 | 1.677 | 5 | 1.357 | 4 | 1.323 |
| dragon | 0.02 | 125 | 4 | 0.030 | 4 | 0.030 | 4 | 0.030 | 3 | 0.030 |
| dragon | 0.008 | 744 | 5 | 0.060 | 4 | 0.077 | 5 | 0.060 | 3 | 0.060 |
| dragon | 0.004 | 3,270 | 5 | 0.210 | 5 | 0.263 | 5 | 0.210 | 4 | 0.210 |
| falconstatue | 0.02 | 411 | 5 | 0.033 | 4 | 0.040 | 5 | 0.033 | 3 | 0.030 |
| falconstatue | 0.008 | 2,691 | 5 | 0.133 | 5 | 0.167 | 5 | 0.137 | 4 | 0.137 |
| falconstatue | 0.004 | 11,036 | 5 | 0.530 | 5 | 0.663 | 5 | 0.540 | 4 | 0.543 |
| falconstatue_boundary | 0.02 | 408 | 5 | 0.030 | 4 | 0.040 | 5 | 0.033 | 3 | 0.030 |
| falconstatue_boundary | 0.008 | 2,635 | 5 | 0.133 | 5 | 0.167 | 5 | 0.133 | 4 | 0.133 |
| falconstatue_boundary | 0.004 | 10,698 | 5 | 0.513 | 5 | 0.660 | 5 | 0.533 | 4 | 0.533 |
| fish | 0.02 | 704 | 5 | 0.047 | 5 | 0.050 | 5 | 0.047 | 4 | 0.047 |
| fish | 0.008 | 4,568 | 5 | 0.220 | 4 | 0.263 | 5 | 0.220 | 3 | 0.217 |
| fish | 0.004 | 18,368 | 5 | 0.890 | 4 | 1.077 | 5 | 0.890 | 3 | 0.870 |
| fish_control_mesh | 0.02 | 859 | 5 | 0.050 | 4 | 0.060 | 5 | 0.050 | 3 | 0.050 |
| fish_control_mesh | 0.008 | 5,637 | 5 | 0.257 | 4 | 0.317 | 5 | 0.260 | 4 | 0.257 |
| fish_control_mesh | 0.004 | 23,399 | 5 | 0.970 | 5 | 1.243 | 5 | 1.017 | 4 | 1.003 |
| fish_low_resolution | 0.02 | 649 | 4 | 0.040 | 4 | 0.050 | 4 | 0.040 | 3 | 0.040 |
| fish_low_resolution | 0.008 | 3,975 | 5 | 0.183 | 4 | 0.233 | 5 | 0.187 | 3 | 0.183 |
| fish_low_resolution | 0.004 | 16,337 | 5 | 0.730 | 4 | 0.940 | 5 | 0.770 | 3 | 0.763 |
| goathead | 0.02 | 30,441 | 5 | 1.440 | 4 | 1.827 | 5 | 1.480 | 3 | 1.467 |
| goathead | 0.008 | 190,762 | 5 | 10.240 | 4 | 12.210 | 5 | 9.537 | 3 | 9.490 |
| goathead | 0.004 | 763,519 | 5 | 44.420 | 4 | 52.010 | 5 | 41.377 | 4 | 43.333 |
| hairball | 0.02 | 89,812 | 6 | 5.817 | 5 | 7.540 | 6 | 5.610 | 4 | 5.330 |
| hairball | 0.008 | 879,909 | 5 | 55.047 | 6 | 68.470 | 6 | 54.280 | 5 | 58.897 |
| hairball | 0.004 | 3,866,231 | 6 | 229.763 | 5 | 285.813 | 6 | 237.427 | 4 | 235.873 |
| hammer | 0.02 | 1,704 | 4 | 0.080 | 4 | 0.103 | 4 | 0.080 | 3 | 0.083 |
| hammer | 0.008 | 10,842 | 4 | 0.480 | 4 | 0.633 | 4 | 0.497 | 3 | 0.517 |
| hammer | 0.004 | 44,135 | 5 | 2.017 | 4 | 2.707 | 5 | 2.070 | 3 | 2.077 |
| hand | 0.02 | 77 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| hand | 0.008 | 394 | 4 | 0.033 | 4 | 0.040 | 4 | 0.033 | 3 | 0.033 |
| hand | 0.004 | 1,554 | 5 | 0.093 | 4 | 0.113 | 5 | 0.093 | 3 | 0.093 |
| hand_closed | 0.02 | 77 | 4 | 0.020 | 4 | 0.020 | 4 | 0.017 | 3 | 0.020 |
| hand_closed | 0.008 | 400 | 4 | 0.030 | 4 | 0.040 | 4 | 0.030 | 3 | 0.030 |
| hand_closed | 0.004 | 1,596 | 5 | 0.087 | 4 | 0.103 | 5 | 0.087 | 3 | 0.083 |
| hand_lowres | 0.02 | 79 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.017 |
| hand_lowres | 0.008 | 406 | 4 | 0.030 | 4 | 0.040 | 5 | 0.033 | 3 | 0.030 |
| hand_lowres | 0.004 | 1,598 | 5 | 0.087 | 4 | 0.103 | 5 | 0.083 | 3 | 0.083 |
| house | 0.02 | — | — | — | — | — | — | — | — | — |
| house | 0.008 | — | — | — | — | — | — | — | — | — |
| house | 0.004 | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.02 | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.008 | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.004 | — | — | — | — | — | — | — | — | — |
| human_man | 0.02 | 171 | 4 | 0.020 | 4 | 0.023 | 4 | 0.020 | 3 | 0.020 |
| human_man | 0.008 | 989 | 5 | 0.060 | 4 | 0.070 | 5 | 0.060 | 3 | 0.060 |
| human_man | 0.004 | 3,877 | 5 | 0.177 | 4 | 0.230 | 5 | 0.177 | 4 | 0.180 |
| human_neutral | 0.02 | 169 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| human_neutral | 0.008 | 884 | 5 | 0.050 | 4 | 0.063 | 5 | 0.053 | 3 | 0.050 |
| human_neutral | 0.004 | 3,631 | 5 | 0.173 | 4 | 0.223 | 5 | 0.180 | 3 | 0.180 |
| human_woman | 0.02 | 148 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| human_woman | 0.008 | 848 | 5 | 0.050 | 4 | 0.060 | 5 | 0.050 | 3 | 0.050 |
| human_woman | 0.004 | 3,401 | 5 | 0.153 | 4 | 0.200 | 5 | 0.160 | 4 | 0.153 |
| koala | 0.02 | 9,070 | 5 | 0.440 | 4 | 0.543 | 5 | 0.457 | 3 | 0.453 |
| koala | 0.008 | 56,842 | 5 | 3.110 | 4 | 3.900 | 5 | 3.003 | 3 | 2.903 |
| koala | 0.004 | 225,890 | 5 | 13.247 | 4 | 15.937 | 5 | 12.687 | 3 | 12.550 |
| koala_low_resolution | 0.02 | 8,876 | 5 | 0.403 | 4 | 0.497 | 5 | 0.403 | 3 | 0.407 |
| koala_low_resolution | 0.008 | 55,880 | 5 | 2.607 | 4 | 3.310 | 5 | 2.703 | 3 | 2.670 |
| koala_low_resolution | 0.004 | 223,453 | 5 | 11.477 | 4 | 13.830 | 5 | 10.963 | 4 | 11.100 |
| lionstatue | 0.02 | 666 | 5 | 0.043 | 4 | 0.053 | 5 | 0.043 | 3 | 0.043 |
| lionstatue | 0.008 | 3,942 | 5 | 0.207 | 5 | 0.257 | 5 | 0.207 | 4 | 0.207 |
| lionstatue | 0.004 | 16,036 | 5 | 0.750 | 4 | 0.943 | 5 | 0.757 | 4 | 0.750 |
| mountain | 0.02 | 123,940 | 5 | 6.320 | 4 | 7.897 | 5 | 6.343 | 3 | 6.197 |
| mountain | 0.008 | 768,838 | 5 | 42.917 | 4 | 52.387 | 5 | 42.627 | 3 | 45.250 |
| mountain | 0.004 | 3,068,538 | 5 | 175.830 | 4 | 217.047 | 5 | 170.207 | 4 | 180.840 |
| mushroom | 0.02 | 2,371 | 5 | 0.120 | 4 | 0.150 | 5 | 0.120 | 3 | 0.120 |
| mushroom | 0.008 | 15,124 | 5 | 0.693 | 4 | 0.880 | 5 | 0.733 | 3 | 0.727 |
| mushroom | 0.004 | 60,698 | 5 | 2.920 | 4 | 3.783 | 5 | 3.127 | 3 | 3.067 |
| nefertiti-lowres | 0.02 | — | — | — | — | — | — | — | — | — |
| nefertiti-lowres | 0.008 | — | — | — | — | — | — | — | — | — |
| nefertiti-lowres | 0.004 | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.02 | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.008 | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.004 | — | — | — | — | — | — | — | — | — |
| parsnip | 0.02 | 1,445 | 4 | 0.090 | 4 | 0.110 | 5 | 0.090 | 3 | 0.093 |
| parsnip | 0.008 | 9,181 | 5 | 0.463 | 5 | 0.587 | 5 | 0.470 | 4 | 0.480 |
| parsnip | 0.004 | 37,087 | 5 | 2.023 | 4 | 2.620 | 5 | 2.050 | 3 | 2.047 |
| penguin | 0.02 | 930 | 5 | 0.057 | 4 | 0.070 | 5 | 0.060 | 3 | 0.057 |
| penguin | 0.008 | 6,165 | 5 | 0.283 | 4 | 0.350 | 5 | 0.293 | 3 | 0.290 |
| penguin | 0.004 | 25,661 | 5 | 1.117 | 4 | 1.407 | 5 | 1.143 | 3 | 1.150 |
| penguin_control_mesh | 0.02 | 1,099 | 5 | 0.060 | 4 | 0.073 | 5 | 0.060 | 3 | 0.060 |
| penguin_control_mesh | 0.008 | 7,637 | 5 | 0.343 | 4 | 0.433 | 5 | 0.350 | 3 | 0.343 |
| penguin_control_mesh | 0.004 | 32,083 | 5 | 1.427 | 4 | 1.793 | 5 | 1.443 | 4 | 1.483 |
| penguin_hr | 0.02 | 927 | 5 | 0.057 | 4 | 0.070 | 5 | 0.057 | 3 | 0.057 |
| penguin_hr | 0.008 | 6,128 | 5 | 0.287 | 4 | 0.350 | 5 | 0.287 | 3 | 0.283 |
| penguin_hr | 0.004 | 25,418 | 5 | 1.170 | 4 | 1.473 | 5 | 1.193 | 3 | 1.250 |
| pizza | 0.02 | 132 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| pizza | 0.008 | 749 | 4 | 0.043 | 4 | 0.053 | 5 | 0.043 | 3 | 0.043 |
| pizza | 0.004 | 2,985 | 5 | 0.137 | 5 | 0.177 | 5 | 0.140 | 4 | 0.140 |
| plane | 0.02 | 9,565 | 5 | 0.413 | 4 | 0.540 | 5 | 0.423 | 4 | 0.430 |
| plane | 0.008 | 60,792 | 5 | 2.707 | 4 | 3.690 | 5 | 2.927 | 4 | 2.953 |
| plane | 0.004 | 243,852 | 5 | 14.650 | 4 | 18.793 | 5 | 14.677 | 4 | 14.993 |
| plane_holes | 0.02 | 9,345 | 5 | 0.420 | 4 | 0.550 | 5 | 0.427 | 4 | 0.437 |
| plane_holes | 0.008 | 58,231 | 5 | 2.657 | 4 | 3.527 | 5 | 2.767 | 4 | 2.713 |
| plane_holes | 0.004 | 231,378 | 5 | 11.280 | 4 | 14.127 | 5 | 10.823 | 4 | 11.543 |
| scorpion | 0.02 | 6,174 | 5 | 0.317 | 5 | 0.393 | 5 | 0.320 | 4 | 0.317 |
| scorpion | 0.008 | 39,895 | 5 | 2.057 | 4 | 2.657 | 5 | 2.163 | 4 | 2.087 |
| scorpion | 0.004 | 160,853 | 5 | 8.893 | 5 | 10.490 | 5 | 8.233 | 4 | 8.220 |
| scorpion_low_resolution | 0.02 | 6,217 | 5 | 0.300 | 5 | 0.373 | 5 | 0.303 | 4 | 0.300 |
| scorpion_low_resolution | 0.008 | 40,227 | 5 | 1.900 | 4 | 2.467 | 5 | 2.017 | 3 | 1.983 |
| scorpion_low_resolution | 0.004 | 161,445 | 5 | 8.230 | 4 | 10.417 | 5 | 8.247 | 4 | 8.157 |
| skull | 0.02 | 40,486 | 5 | 1.977 | 4 | 2.563 | 5 | 2.130 | 4 | 2.077 |
| skull | 0.008 | 254,923 | 5 | 15.300 | 4 | 17.673 | 5 | 14.003 | 3 | 13.413 |
| skull | 0.004 | 1,021,048 | 5 | 57.523 | 4 | 75.030 | 5 | 63.873 | 3 | 59.267 |
| skull_low_resolution | 0.02 | 40,127 | 5 | 1.883 | 4 | 2.423 | 5 | 2.013 | 4 | 1.993 |
| skull_low_resolution | 0.008 | 252,742 | 5 | 12.993 | 4 | 15.413 | 5 | 12.703 | 4 | 13.547 |
| skull_low_resolution | 0.004 | 1,012,874 | 5 | 54.990 | 4 | 69.777 | 5 | 67.210 | 4 | 63.277 |
| sphere | 0.02 | 1,038 | 5 | 0.060 | 3 | 0.070 | 5 | 0.060 | 3 | 0.060 |
| sphere | 0.008 | 6,299 | 5 | 0.290 | 3 | 0.360 | 5 | 0.297 | 3 | 0.290 |
| sphere | 0.004 | 25,296 | 5 | 1.160 | 3 | 1.430 | 5 | 1.200 | 3 | 1.210 |
| spot | 0.02 | 438 | 5 | 0.037 | 4 | 0.040 | 5 | 0.037 | 3 | 0.037 |
| spot | 0.008 | 2,774 | 5 | 0.137 | 4 | 0.170 | 5 | 0.137 | 3 | 0.137 |
| spot | 0.004 | 11,155 | 5 | 0.533 | 4 | 0.657 | 5 | 0.540 | 3 | 0.527 |
| spot_control_mesh | 0.02 | 582 | 4 | 0.040 | 4 | 0.047 | 4 | 0.040 | 3 | 0.040 |
| spot_control_mesh | 0.008 | 3,829 | 5 | 0.170 | 5 | 0.223 | 5 | 0.173 | 4 | 0.173 |
| spot_control_mesh | 0.004 | 15,499 | 5 | 0.653 | 5 | 0.823 | 5 | 0.663 | 4 | 0.657 |
| spot_low_resolution | 0.02 | 436 | 4 | 0.033 | 4 | 0.040 | 4 | 0.033 | 3 | 0.033 |
| spot_low_resolution | 0.008 | 2,752 | 5 | 0.133 | 4 | 0.163 | 5 | 0.133 | 3 | 0.130 |
| spot_low_resolution | 0.004 | 11,023 | 5 | 0.483 | 4 | 0.603 | 5 | 0.490 | 3 | 0.483 |
| springer | 0.02 | 2,591 | 5 | 0.127 | 4 | 0.157 | 5 | 0.130 | 3 | 0.130 |
| springer | 0.008 | 16,543 | 5 | 0.753 | 4 | 0.943 | 5 | 0.770 | 3 | 0.767 |
| springer | 0.004 | 66,240 | 5 | 3.353 | 4 | 4.343 | 5 | 3.407 | 3 | 3.380 |
| strawberry | 0.02 | 979 | 5 | 0.073 | 4 | 0.090 | 5 | 0.073 | 3 | 0.070 |
| strawberry | 0.008 | 6,509 | 5 | 0.387 | 4 | 0.480 | 5 | 0.387 | 4 | 0.383 |
| strawberry | 0.004 | 26,762 | 5 | 1.437 | 4 | 1.793 | 5 | 1.437 | 3 | 1.420 |
| stuffedtoy | 0.02 | 97 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| stuffedtoy | 0.008 | 555 | 5 | 0.040 | 4 | 0.050 | 5 | 0.040 | 3 | 0.040 |
| stuffedtoy | 0.004 | 2,309 | 5 | 0.120 | 4 | 0.140 | 5 | 0.120 | 3 | 0.117 |
| sword-quad-dominant | 0.02 | 34 | 3 | 0.020 | 4 | 0.020 | 3 | 0.020 | 3 | 0.020 |
| sword-quad-dominant | 0.008 | 80 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| sword-quad-dominant | 0.004 | 197 | 4 | 0.023 | 4 | 0.027 | 4 | 0.020 | 3 | 0.023 |
| sword | 0.02 | 34 | 3 | 0.020 | 4 | 0.020 | 3 | 0.020 | 3 | 0.017 |
| sword | 0.008 | 80 | 4 | 0.017 | 4 | 0.020 | 4 | 0.030 | 3 | 0.020 |
| sword | 0.004 | 197 | 4 | 0.020 | 4 | 0.023 | 4 | 0.020 | 3 | 0.020 |
| torus | 0.02 | 800 | 4 | 0.050 | 4 | 0.053 | 4 | 0.053 | 3 | 0.050 |
| torus | 0.008 | 4,971 | 5 | 0.230 | 3 | 0.283 | 5 | 0.233 | 3 | 0.230 |
| torus | 0.004 | 20,025 | 5 | 0.907 | 4 | 1.130 | 5 | 0.920 | 3 | 0.933 |
| tower | 0.02 | 5,713 | 5 | 0.310 | 4 | 0.393 | 5 | 0.317 | 3 | 0.313 |
| tower | 0.008 | 37,228 | 5 | 1.933 | 5 | 2.510 | 5 | 1.940 | 4 | 1.923 |
| tower | 0.004 | 152,469 | 5 | 8.340 | 5 | 10.470 | 5 | 8.050 | 4 | 8.207 |
| tower_holes | 0.02 | 4,740 | 5 | 0.243 | 5 | 0.300 | 5 | 0.247 | 4 | 0.243 |
| tower_holes | 0.008 | 29,371 | 5 | 1.377 | 5 | 1.763 | 6 | 1.433 | 4 | 1.447 |
| tower_holes | 0.004 | 118,862 | 5 | 6.460 | 4 | 7.747 | 5 | 5.900 | 4 | 5.830 |
| tree | 0.02 | 457 | 5 | 0.040 | 4 | 0.050 | 5 | 0.040 | 3 | 0.040 |
| tree | 0.008 | 2,351 | 5 | 0.153 | 4 | 0.190 | 5 | 0.153 | 3 | 0.150 |
| tree | 0.004 | 9,295 | 5 | 0.537 | 4 | 0.670 | 5 | 0.543 | 3 | 0.533 |
| tree_closed | 0.02 | 459 | 5 | 0.040 | 4 | 0.050 | 5 | 0.040 | 3 | 0.040 |
| tree_closed | 0.008 | 2,410 | 5 | 0.153 | 4 | 0.197 | 5 | 0.153 | 3 | 0.157 |
| tree_closed | 0.004 | 9,616 | 5 | 0.537 | 4 | 0.677 | 5 | 0.543 | 3 | 0.537 |
| violin | 0.02 | 28 | 3 | 0.020 | 4 | 0.020 | 3 | 0.020 | 3 | 0.020 |
| violin | 0.008 | 90 | 4 | 0.020 | 4 | 0.020 | 4 | 0.020 | 3 | 0.020 |
| violin | 0.004 | 295 | 4 | 0.030 | 4 | 0.040 | 4 | 0.030 | 3 | 0.033 |
| well | 0.02 | 1,143,998 | 5 | 62.917 | 5 | 79.440 | 5 | 64.687 | 4 | 63.083 |
| well | 0.008 | — | — | — | — | — | — | — | — | — |
| well | 0.004 | — | — | — | — | — | — | — | — | — |
| well_boundary | 0.02 | 1,039,667 | 5 | 55.180 | 5 | 73.497 | 5 | 63.317 | 4 | 58.100 |
| well_boundary | 0.008 | — | — | — | — | — | — | — | — | — |
| well_boundary | 0.004 | — | — | — | — | — | — | — | — | — |
| wingnut | 0.02 | 5,099 | 5 | 0.213 | 4 | 0.280 | 5 | 0.223 | 4 | 0.227 |
| wingnut | 0.008 | 31,746 | 5 | 1.327 | 5 | 1.663 | 5 | 1.330 | 4 | 1.383 |
| wingnut | 0.004 | 125,910 | 5 | 5.780 | 4 | 7.370 | 5 | 5.677 | 4 | 5.733 |

16 of 204 cases did not run, covering house, house_boundary, nefertiti, nefertiti-lowres, well, well_boundary. Those meshes sit in coordinate systems tens to hundreds of times larger than the unit-scale ones, so a fixed voxel size asks for a grid thousands of voxels across and the run exhausts GPU memory before labeling begins. Not a convergence failure.

### Why the times span four orders of magnitude

Input size, in two regimes.

Above roughly ten thousand leaves (75 cases) the kernel is throughput-bound and the cost per leaf is flat: a median of 49 ns, with the 10th and 90th percentiles at 42 and 60 ns. A large number in the table therefore means a large mesh, not a schedule behaving badly.

Below about a thousand leaves (51 cases) the call sits on a launch-latency floor -- a median of 33 µs regardless of size -- so per-leaf figures there measure the launch, not the algorithm, and the schedules are indistinguishable by construction.

| case | leaves | one union-find call (P) | per leaf |
|---|---:|---:|---:|
| boot @ 0.02 (latency-bound) | 22 | 0.020 ms | 909 ns |
| plane_holes @ 0.004 (throughput-bound) | 231,378 | 11.280 ms | 49 ns |
| hairball @ 0.004 (throughput-bound) | 3,866,231 | 229.763 ms | 59 ns |

This spread is also why no total is reported anywhere: summing across cases would hand the answer to the two or three largest meshes.

## What to take from it

- **A proven bound is free here.** R matches P to within 0%, so the schedule with no proven bound has no measured advantage left to defend.

- **R is the only optimal-bound algorithm we can adopt.** The other one the paper proves O(lg n) for, RA, needs edge alteration, and our edges are implicit in the 6-neighbourhood of the voxel lattice rather than a list we can rewrite.

- **No schedule lowers the worst case much.** The worst leaf anywhere needed 6 rounds under P and 6 under R, against a cap of 64. If the worry is the cap, the fix is detecting that it was reached, which nothing currently does.

- **Switching is a one-line call:** `ConnectedComponents::setLeafSchedule()`. All four share one solver, so the counting and mask passes cannot drift apart.
