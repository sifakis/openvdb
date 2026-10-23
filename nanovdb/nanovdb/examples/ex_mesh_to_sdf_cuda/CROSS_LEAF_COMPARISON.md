# The cross-leaf union-find: four algorithms, measured

Generated 2026-08-10 by `ex_mesh_to_sdf_cuda --sv-convergence <mesh> --voxel-size S`.

Connected-component labeling runs two union-finds. The per-leaf one solves each 8³ leaf in isolation; this one merges the results across leaf boundaries, over an **explicit list of edges** between leaf-local components. The two are different algorithms today, and that is the first thing to be clear about.

| | what it does | proven bound |
|---|---|---|
| **CAS** (what ships) | Not a connect/shortcut loop at all: a classic disjoint-set forest. One thread per edge walks to both roots and compare-and-swaps the larger root under the smaller, retrying if the CAS fails, then one pass flattens every path. No rounds. | **none.** It uses union-by-minimum rather than union-by-rank and does not compress while walking, so the textbook O(α(n)) does not apply -- an adversarial edge order can build a chain of depth Θ(n). |
| **P** | Liu & Tarjan's algorithm P over the edge list: parent-connect, one shortcut per round, until nothing changes. | **none** (section 4.2) |
| **S** | algorithm S: parent-connect, then shortcut until the forest is flat. | O(min{d, lg n}·lg n) |
| **R** | algorithm R: parent-root-connect, one shortcut per round. | **O(lg n)**, matching the lower bound |

A **round** here is two kernel launches plus a convergence check that has to come back to the host -- there is no block to synchronize, unlike inside a leaf. That cost is why the answer differs from the per-leaf comparison.

## The short answer

| | **CAS** | **P** | **S** | **R** |
|---|---:|---:|---:|---:|
| **time of one cross-leaf pass, typical case** | **0.06 ms** | **0.57 ms** | **0.67 ms** | **0.57 ms** |
| **relative to CAS, typical case** | **1.00x** | **9.39x** | **10.75x** | **9.40x** |
| rounds, median | -- | 8 | 4 | 9 |
| rounds, worst anywhere | -- | 11 | 6 | 11 |
| cases where it was fastest, of 188 | 188 | 0 | 0 | 0 |

**CAS wins everywhere.** The closest is P at 9.39x. That is the opposite shape from the per-leaf result, where four schedules landed within a few percent of each other, and the reason is the barrier: inside a leaf a round costs a `__syncthreads()`, here it costs two kernel launches and a stream synchronization. CAS needs no rounds at all -- each thread loops until its own edge is resolved -- so it pays that cost zero times.

**P is strictly dominated.** It is slower than CAS and, like CAS, has no proven bound, so there is nothing it offers in exchange. R is the only alternative that buys anything: an O(lg n) bound, at 9.4x the time on this stage in the typical case and about 2x on the largest meshes.

## The gap depends on how big the mesh is

The 9.4x above is a median over cases, and most cases are small. Grouped by leaf count the picture is monotone: the more work there is per kernel, the less the per-round synchronization matters, and the four converge.

| leaves | cases | CAS, one pass | P | S | R |
|---|---:|---:|---:|---:|---:|
| **under 1 K** | 51 | 0.030 ms | 14.00x | 15.44x | 14.00x |
| **1 K - 10 K** | 62 | 0.053 ms | 10.16x | 11.70x | 10.10x |
| **10 K - 100 K** | 51 | 0.107 ms | 6.12x | 6.86x | 5.82x |
| **100 K - 1 M** | 18 | 0.348 ms | 3.13x | 3.11x | 2.78x |
| **over 1 M** | 6 | 1.260 ms | 2.02x | 1.78x | 1.63x |

On the smallest meshes a cross-leaf pass is 30 microseconds, so a round -- two kernel launches and a stream synchronization -- costs more than the work it synchronizes, and the round-based algorithms lose by 14x. On the largest they are within 2x. The trend is smooth and points the same way throughout: CAS is ahead everywhere, by less as the input grows.

This also means the headline ratio is the wrong number to quote for a production decision. Big meshes are where the time actually goes, and there the choice is worth about 2x on a stage that is under 1% of labeling -- roughly 1% of the pipeline.

## How much does this stage matter?

Very little. Across the cases measured, the cross-leaf pass is a median of **4.1%** of the whole labeling call, and on the largest meshes it falls to well under 1%:

| case | leaves | whole labeling | cross-leaf | share |
|---|---:|---:|---:|---:|
| hairball @ 0.004 | 3,866,231 | 817.7 ms | 5.20 ms | 0.64% |
| mountain @ 0.004 | 3,068,538 | 643.4 ms | 5.67 ms | 0.88% |
| well @ 0.02 | 1,143,998 | 210.9 ms | 1.27 ms | 0.60% |

So the whole choice moves the pipeline by a fraction of a percent either way. Adopting R for its bound would be cheap in absolute terms; keeping CAS for its speed wins little.

## Does the round count grow with the domain?

This was the open question -- inside a leaf the graph is capped at 512 vertices, but the component graph here grows with the mesh. The answer is that it grows, but barely:

| case | leaves | P rounds | S rounds | R rounds |
|---|---:|---:|---:|---:|
| boot @ 0.02 | 22 | 4 | 3 | 4 |
| hairball @ 0.004 | 3,866,231 | 9 | 6 | 11 |
| penguin @ 0.008 | 6,165 | 8 | 4 | 8 |

Leaf count spans 22 to 3,866,231 -- a factor of 175,738 -- while P's rounds go from 4 to 9. A rasterized narrow band is a thin shell that follows the surface, so its component graph stays shallow no matter how finely it is sampled. Domain size is not the lever here.

## How this was measured

- Both grids the pipeline labels are covered: the un-pruned band and the barrier-pruned band. A reported time is one pass over each, added.
- Each case runs under three lattice placements (identity, mirrored, rotated 45° about z). Times are the mean over them; rounds and shortcut counts are the worst.
- Timing is a CUDA event pair around the whole cross-leaf stage: init, the merge, and the final flatten. Every algorithm pays the same init and flatten.
- The convergence check is per round, which is a stream synchronization. Batching four rounds between checks was tried separately: it helps small cases (bunny 13x -> 7x) and hurts large ones, because the wasted rounds then cost real kernel time.
- The four were checked to agree on the component count in every case.

## Every case

`rd` = rounds (CAS has none). `ms` = one cross-leaf pass, mean over the three placements.

| mesh | voxel size | leaves | CAS rd | CAS ms | P rd | P ms | S rd | S ms | R rd | R ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| armadillo | 0.02 | 151 | -- | 0.027 | 5 | 0.383 | 4 | 0.453 | 6 | 0.393 |
| armadillo | 0.008 | 802 | -- | 0.030 | 7 | 0.457 | 4 | 0.557 | 7 | 0.460 |
| armadillo | 0.004 | 3,150 | -- | 0.050 | 8 | 0.543 | 4 | 0.647 | 8 | 0.553 |
| boat | 0.02 | 3,004 | -- | 0.053 | 8 | 0.517 | 4 | 0.570 | 7 | 0.520 |
| boat | 0.008 | 19,434 | -- | 0.097 | 9 | 0.600 | 5 | 0.710 | 10 | 0.617 |
| boat | 0.004 | 77,816 | -- | 0.160 | 10 | 0.747 | 5 | 0.783 | 10 | 0.670 |
| boot | 0.02 | 22 | -- | 0.020 | 4 | 0.300 | 3 | 0.327 | 4 | 0.280 |
| boot | 0.008 | 72 | -- | 0.023 | 5 | 0.350 | 3 | 0.363 | 5 | 0.307 |
| boot | 0.004 | 246 | -- | 0.030 | 6 | 0.417 | 3 | 0.403 | 6 | 0.373 |
| brucewick | 0.02 | 1,070 | -- | 0.037 | 7 | 0.483 | 4 | 0.550 | 7 | 0.447 |
| brucewick | 0.008 | 6,822 | -- | 0.063 | 8 | 0.573 | 5 | 0.677 | 8 | 0.557 |
| brucewick | 0.004 | 27,653 | -- | 0.113 | 9 | 0.657 | 4 | 0.733 | 9 | 0.630 |
| bunny | 0.02 | 454 | -- | 0.040 | 7 | 0.433 | 5 | 0.480 | 6 | 0.410 |
| bunny | 0.008 | 2,822 | -- | 0.050 | 8 | 0.517 | 4 | 0.623 | 8 | 0.520 |
| bunny | 0.004 | 11,298 | -- | 0.077 | 9 | 0.603 | 5 | 0.733 | 9 | 0.610 |
| bunny_hr | 0.02 | 462 | -- | 0.030 | 6 | 0.410 | 4 | 0.457 | 6 | 0.407 |
| bunny_hr | 0.008 | 2,854 | -- | 0.050 | 8 | 0.513 | 4 | 0.587 | 8 | 0.497 |
| bunny_hr | 0.004 | 11,491 | -- | 0.080 | 9 | 0.597 | 5 | 0.717 | 9 | 0.600 |
| cat-low-resolution | 0.02 | 1,017 | -- | 0.033 | 7 | 0.467 | 4 | 0.520 | 7 | 0.487 |
| cat-low-resolution | 0.008 | 6,478 | -- | 0.070 | 9 | 0.627 | 5 | 0.697 | 9 | 0.603 |
| cat-low-resolution | 0.004 | 26,138 | -- | 0.123 | 10 | 0.703 | 5 | 0.813 | 10 | 0.670 |
| cat | 0.02 | 1,021 | -- | 0.050 | 7 | 0.500 | 4 | 0.573 | 7 | 0.500 |
| cat | 0.008 | 6,536 | -- | 0.080 | 9 | 0.640 | 5 | 0.713 | 9 | 0.610 |
| cat | 0.004 | 26,431 | -- | 0.123 | 10 | 0.687 | 5 | 0.787 | 10 | 0.673 |
| cheese | 0.02 | 3,646 | -- | 0.057 | 8 | 0.540 | 4 | 0.653 | 8 | 0.557 |
| cheese | 0.008 | 22,816 | -- | 0.113 | 9 | 0.647 | 5 | 0.747 | 9 | 0.633 |
| cheese | 0.004 | 91,591 | -- | 0.190 | 10 | 0.773 | 5 | 0.917 | 10 | 0.693 |
| cow-low-resolution | 0.02 | 8,180 | -- | 0.070 | 9 | 0.617 | 4 | 0.720 | 9 | 0.590 |
| cow-low-resolution | 0.008 | 52,437 | -- | 0.133 | 10 | 0.743 | 5 | 0.803 | 10 | 0.693 |
| cow-low-resolution | 0.004 | 211,338 | -- | 0.333 | 10 | 1.110 | 5 | 1.157 | 10 | 1.000 |
| cow | 0.02 | 8,286 | -- | 0.070 | 9 | 0.630 | 4 | 0.720 | 9 | 0.597 |
| cow | 0.008 | 53,342 | -- | 0.127 | 10 | 0.737 | 5 | 0.803 | 10 | 0.697 |
| cow | 0.004 | 215,332 | -- | 0.340 | 10 | 1.040 | 5 | 1.037 | 10 | 0.930 |
| cube | 0.02 | 1,960 | -- | 0.040 | 7 | 0.483 | 3 | 0.397 | 7 | 0.463 |
| cube | 0.008 | 10,128 | -- | 0.073 | 9 | 0.583 | 4 | 0.497 | 8 | 0.587 |
| cube | 0.004 | 48,240 | -- | 0.147 | 10 | 0.763 | 4 | 0.563 | 9 | 0.667 |
| cube_no_bottom | 0.02 | 1,722 | -- | 0.040 | 8 | 0.507 | 3 | 0.413 | 7 | 0.470 |
| cube_no_bottom | 0.008 | 8,268 | -- | 0.063 | 9 | 0.533 | 4 | 0.467 | 9 | 0.510 |
| cube_no_bottom | 0.004 | 40,560 | -- | 0.147 | 10 | 0.773 | 4 | 0.563 | 10 | 0.667 |
| demosthenes-low-res | 0.02 | 920 | -- | 0.043 | 7 | 0.480 | 4 | 0.567 | 7 | 0.480 |
| demosthenes-low-res | 0.008 | 5,870 | -- | 0.053 | 9 | 0.540 | 5 | 0.657 | 9 | 0.537 |
| demosthenes-low-res | 0.004 | 23,518 | -- | 0.090 | 10 | 0.627 | 5 | 0.757 | 10 | 0.627 |
| demosthenes | 0.02 | 930 | -- | 0.043 | 7 | 0.477 | 4 | 0.563 | 7 | 0.487 |
| demosthenes | 0.008 | 5,935 | -- | 0.060 | 9 | 0.557 | 5 | 0.673 | 9 | 0.543 |
| demosthenes | 0.004 | 23,892 | -- | 0.103 | 10 | 0.677 | 5 | 0.783 | 10 | 0.647 |
| dragon | 0.02 | 125 | -- | 0.020 | 5 | 0.370 | 4 | 0.423 | 5 | 0.347 |
| dragon | 0.008 | 744 | -- | 0.040 | 7 | 0.477 | 5 | 0.553 | 7 | 0.463 |
| dragon | 0.004 | 3,270 | -- | 0.050 | 8 | 0.523 | 5 | 0.653 | 8 | 0.570 |
| falconstatue | 0.02 | 411 | -- | 0.030 | 6 | 0.420 | 4 | 0.447 | 6 | 0.403 |
| falconstatue | 0.008 | 2,691 | -- | 0.050 | 7 | 0.493 | 4 | 0.607 | 8 | 0.507 |
| falconstatue | 0.004 | 11,036 | -- | 0.073 | 9 | 0.593 | 4 | 0.693 | 9 | 0.623 |
| falconstatue_boundary | 0.02 | 408 | -- | 0.033 | 7 | 0.430 | 4 | 0.473 | 7 | 0.420 |
| falconstatue_boundary | 0.008 | 2,635 | -- | 0.047 | 8 | 0.527 | 4 | 0.653 | 8 | 0.537 |
| falconstatue_boundary | 0.004 | 10,698 | -- | 0.073 | 9 | 0.597 | 4 | 0.693 | 9 | 0.600 |
| fish | 0.02 | 704 | -- | 0.030 | 7 | 0.420 | 4 | 0.513 | 7 | 0.410 |
| fish | 0.008 | 4,568 | -- | 0.043 | 8 | 0.537 | 5 | 0.657 | 9 | 0.530 |
| fish | 0.004 | 18,368 | -- | 0.097 | 9 | 0.653 | 5 | 0.757 | 9 | 0.643 |
| fish_control_mesh | 0.02 | 859 | -- | 0.040 | 7 | 0.450 | 4 | 0.563 | 7 | 0.443 |
| fish_control_mesh | 0.008 | 5,637 | -- | 0.063 | 9 | 0.570 | 5 | 0.720 | 9 | 0.597 |
| fish_control_mesh | 0.004 | 23,399 | -- | 0.077 | 9 | 0.607 | 5 | 0.810 | 9 | 0.603 |
| fish_low_resolution | 0.02 | 649 | -- | 0.040 | 6 | 0.427 | 4 | 0.560 | 6 | 0.417 |
| fish_low_resolution | 0.008 | 3,975 | -- | 0.050 | 8 | 0.563 | 4 | 0.667 | 9 | 0.590 |
| fish_low_resolution | 0.004 | 16,337 | -- | 0.080 | 9 | 0.623 | 5 | 0.773 | 9 | 0.610 |
| goathead | 0.02 | 30,441 | -- | 0.107 | 10 | 0.653 | 5 | 0.767 | 10 | 0.630 |
| goathead | 0.008 | 190,762 | -- | 0.313 | 10 | 0.967 | 5 | 0.967 | 11 | 0.880 |
| goathead | 0.004 | 763,519 | -- | 1.113 | 10 | 2.037 | 5 | 1.860 | 11 | 1.600 |
| hairball | 0.02 | 89,812 | -- | 0.173 | 9 | 0.807 | 5 | 0.943 | 9 | 0.777 |
| hairball | 0.008 | 879,909 | -- | 1.077 | 9 | 2.873 | 6 | 2.400 | 10 | 2.173 |
| hairball | 0.004 | 3,866,231 | -- | 5.197 | 9 | 14.180 | 6 | 12.370 | 11 | 12.557 |
| hammer | 0.02 | 1,704 | -- | 0.053 | 8 | 0.543 | 4 | 0.553 | 8 | 0.550 |
| hammer | 0.008 | 10,842 | -- | 0.087 | 9 | 0.643 | 5 | 0.697 | 10 | 0.673 |
| hammer | 0.004 | 44,135 | -- | 0.150 | 10 | 0.700 | 5 | 0.747 | 11 | 0.703 |
| hand | 0.02 | 77 | -- | 0.020 | 5 | 0.363 | 4 | 0.467 | 5 | 0.353 |
| hand | 0.008 | 394 | -- | 0.033 | 7 | 0.440 | 4 | 0.517 | 7 | 0.470 |
| hand | 0.004 | 1,554 | -- | 0.043 | 8 | 0.513 | 4 | 0.637 | 8 | 0.537 |
| hand_closed | 0.02 | 77 | -- | 0.020 | 5 | 0.363 | 4 | 0.447 | 5 | 0.343 |
| hand_closed | 0.008 | 400 | -- | 0.030 | 6 | 0.430 | 4 | 0.520 | 7 | 0.440 |
| hand_closed | 0.004 | 1,596 | -- | 0.043 | 7 | 0.500 | 4 | 0.607 | 8 | 0.537 |
| hand_lowres | 0.02 | 79 | -- | 0.023 | 5 | 0.413 | 4 | 0.443 | 5 | 0.340 |
| hand_lowres | 0.008 | 406 | -- | 0.030 | 7 | 0.437 | 4 | 0.510 | 7 | 0.423 |
| hand_lowres | 0.004 | 1,598 | -- | 0.043 | 8 | 0.507 | 4 | 0.633 | 8 | 0.550 |
| house | 0.02 | — | — | — | — | — | — | — | — | — |
| house | 0.008 | — | — | — | — | — | — | — | — | — |
| house | 0.004 | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.02 | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.008 | — | — | — | — | — | — | — | — | — |
| house_boundary | 0.004 | — | — | — | — | — | — | — | — | — |
| human_man | 0.02 | 171 | -- | 0.030 | 6 | 0.417 | 4 | 0.440 | 7 | 0.423 |
| human_man | 0.008 | 989 | -- | 0.037 | 8 | 0.487 | 4 | 0.557 | 8 | 0.483 |
| human_man | 0.004 | 3,877 | -- | 0.057 | 9 | 0.587 | 5 | 0.750 | 9 | 0.600 |
| human_neutral | 0.02 | 169 | -- | 0.030 | 6 | 0.433 | 4 | 0.483 | 7 | 0.447 |
| human_neutral | 0.008 | 884 | -- | 0.037 | 8 | 0.500 | 4 | 0.547 | 8 | 0.483 |
| human_neutral | 0.004 | 3,631 | -- | 0.050 | 9 | 0.560 | 5 | 0.660 | 9 | 0.563 |
| human_woman | 0.02 | 148 | -- | 0.030 | 6 | 0.413 | 4 | 0.430 | 7 | 0.423 |
| human_woman | 0.008 | 848 | -- | 0.040 | 8 | 0.517 | 4 | 0.573 | 8 | 0.507 |
| human_woman | 0.004 | 3,401 | -- | 0.050 | 9 | 0.600 | 5 | 0.707 | 9 | 0.600 |
| koala | 0.02 | 9,070 | -- | 0.067 | 9 | 0.577 | 4 | 0.680 | 9 | 0.573 |
| koala | 0.008 | 56,842 | -- | 0.133 | 10 | 0.780 | 5 | 0.900 | 10 | 0.717 |
| koala | 0.004 | 225,890 | -- | 0.367 | 10 | 1.190 | 5 | 1.147 | 11 | 1.000 |
| koala_low_resolution | 0.02 | 8,876 | -- | 0.063 | 9 | 0.547 | 4 | 0.643 | 9 | 0.557 |
| koala_low_resolution | 0.008 | 55,880 | -- | 0.133 | 10 | 0.833 | 5 | 0.850 | 10 | 0.737 |
| koala_low_resolution | 0.004 | 223,453 | -- | 0.357 | 10 | 1.070 | 5 | 1.100 | 10 | 0.963 |
| lionstatue | 0.02 | 666 | -- | 0.040 | 7 | 0.450 | 4 | 0.580 | 7 | 0.490 |
| lionstatue | 0.008 | 3,942 | -- | 0.053 | 8 | 0.560 | 5 | 0.687 | 8 | 0.587 |
| lionstatue | 0.004 | 16,036 | -- | 0.080 | 9 | 0.653 | 5 | 0.777 | 10 | 0.620 |
| mountain | 0.02 | 123,940 | -- | 0.283 | 10 | 0.960 | 5 | 0.933 | 10 | 0.810 |
| mountain | 0.008 | 768,838 | -- | 1.120 | 10 | 2.217 | 5 | 1.813 | 11 | 1.657 |
| mountain | 0.004 | 3,068,538 | -- | 5.673 | 11 | 10.603 | 6 | 7.167 | 11 | 8.817 |
| mushroom | 0.02 | 2,371 | -- | 0.047 | 8 | 0.517 | 4 | 0.557 | 8 | 0.537 |
| mushroom | 0.008 | 15,124 | -- | 0.090 | 9 | 0.643 | 4 | 0.713 | 9 | 0.640 |
| mushroom | 0.004 | 60,698 | -- | 0.140 | 10 | 0.710 | 5 | 0.803 | 10 | 0.657 |
| nefertiti-lowres | 0.02 | — | — | — | — | — | — | — | — | — |
| nefertiti-lowres | 0.008 | — | — | — | — | — | — | — | — | — |
| nefertiti-lowres | 0.004 | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.02 | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.008 | — | — | — | — | — | — | — | — | — |
| nefertiti | 0.004 | — | — | — | — | — | — | — | — | — |
| parsnip | 0.02 | 1,445 | -- | 0.053 | 8 | 0.563 | 4 | 0.530 | 8 | 0.530 |
| parsnip | 0.008 | 9,181 | -- | 0.107 | 9 | 0.637 | 4 | 0.667 | 9 | 0.633 |
| parsnip | 0.004 | 37,087 | -- | 0.177 | 10 | 0.697 | 4 | 0.807 | 10 | 0.683 |
| penguin | 0.02 | 930 | -- | 0.040 | 7 | 0.463 | 4 | 0.537 | 7 | 0.447 |
| penguin | 0.008 | 6,165 | -- | 0.060 | 8 | 0.583 | 4 | 0.673 | 8 | 0.567 |
| penguin | 0.004 | 25,661 | -- | 0.097 | 9 | 0.637 | 5 | 0.777 | 10 | 0.647 |
| penguin_control_mesh | 0.02 | 1,099 | -- | 0.040 | 7 | 0.480 | 4 | 0.547 | 7 | 0.457 |
| penguin_control_mesh | 0.008 | 7,637 | -- | 0.060 | 8 | 0.560 | 4 | 0.700 | 8 | 0.557 |
| penguin_control_mesh | 0.004 | 32,083 | -- | 0.100 | 10 | 0.663 | 5 | 0.790 | 9 | 0.607 |
| penguin_hr | 0.02 | 927 | -- | 0.030 | 7 | 0.447 | 4 | 0.530 | 7 | 0.420 |
| penguin_hr | 0.008 | 6,128 | -- | 0.060 | 8 | 0.557 | 4 | 0.680 | 8 | 0.573 |
| penguin_hr | 0.004 | 25,418 | -- | 0.110 | 10 | 0.697 | 5 | 0.777 | 10 | 0.640 |
| pizza | 0.02 | 132 | -- | 0.023 | 6 | 0.360 | 3 | 0.360 | 5 | 0.343 |
| pizza | 0.008 | 749 | -- | 0.040 | 7 | 0.470 | 3 | 0.493 | 7 | 0.457 |
| pizza | 0.004 | 2,985 | -- | 0.057 | 8 | 0.573 | 4 | 0.577 | 8 | 0.553 |
| plane | 0.02 | 9,565 | -- | 0.077 | 9 | 0.600 | 5 | 0.707 | 10 | 0.593 |
| plane | 0.008 | 60,792 | -- | 0.150 | 10 | 0.723 | 5 | 0.907 | 11 | 0.750 |
| plane | 0.004 | 243,852 | -- | 0.400 | 10 | 1.120 | 6 | 1.200 | 11 | 1.027 |
| plane_holes | 0.02 | 9,345 | -- | 0.080 | 9 | 0.650 | 5 | 0.760 | 10 | 0.670 |
| plane_holes | 0.008 | 58,231 | -- | 0.153 | 10 | 0.800 | 5 | 0.917 | 11 | 0.767 |
| plane_holes | 0.004 | 231,378 | -- | 0.377 | 11 | 1.193 | 6 | 1.240 | 11 | 1.087 |
| scorpion | 0.02 | 6,174 | -- | 0.057 | 9 | 0.610 | 5 | 0.800 | 10 | 0.630 |
| scorpion | 0.008 | 39,895 | -- | 0.093 | 10 | 0.700 | 5 | 0.917 | 11 | 0.720 |
| scorpion | 0.004 | 160,853 | -- | 0.247 | 10 | 1.003 | 6 | 1.130 | 11 | 0.933 |
| scorpion_low_resolution | 0.02 | 6,217 | -- | 0.053 | 9 | 0.630 | 5 | 0.807 | 9 | 0.623 |
| scorpion_low_resolution | 0.008 | 40,227 | -- | 0.097 | 10 | 0.730 | 6 | 0.937 | 11 | 0.723 |
| scorpion_low_resolution | 0.004 | 161,445 | -- | 0.253 | 11 | 0.967 | 6 | 1.140 | 11 | 0.900 |
| skull | 0.02 | 40,486 | -- | 0.123 | 10 | 0.740 | 5 | 0.830 | 10 | 0.690 |
| skull | 0.008 | 254,923 | -- | 0.370 | 10 | 1.107 | 5 | 1.110 | 11 | 1.017 |
| skull | 0.004 | 1,021,048 | -- | 1.217 | 10 | 2.470 | 5 | 2.173 | 11 | 1.890 |
| skull_low_resolution | 0.02 | 40,127 | -- | 0.120 | 9 | 0.707 | 5 | 0.820 | 10 | 0.683 |
| skull_low_resolution | 0.008 | 252,742 | -- | 0.390 | 10 | 1.147 | 5 | 1.143 | 10 | 0.997 |
| skull_low_resolution | 0.004 | 1,012,874 | -- | 1.253 | 10 | 2.353 | 6 | 2.230 | 11 | 1.990 |
| sphere | 0.02 | 1,038 | -- | 0.040 | 7 | 0.490 | 3 | 0.483 | 6 | 0.417 |
| sphere | 0.008 | 6,299 | -- | 0.057 | 8 | 0.540 | 3 | 0.527 | 8 | 0.540 |
| sphere | 0.004 | 25,296 | -- | 0.107 | 9 | 0.613 | 4 | 0.693 | 9 | 0.610 |
| spot | 0.02 | 438 | -- | 0.033 | 6 | 0.433 | 4 | 0.470 | 6 | 0.417 |
| spot | 0.008 | 2,774 | -- | 0.050 | 7 | 0.490 | 4 | 0.600 | 8 | 0.500 |
| spot | 0.004 | 11,155 | -- | 0.067 | 8 | 0.540 | 5 | 0.697 | 9 | 0.553 |
| spot_control_mesh | 0.02 | 582 | -- | 0.030 | 6 | 0.413 | 4 | 0.487 | 6 | 0.400 |
| spot_control_mesh | 0.008 | 3,829 | -- | 0.050 | 8 | 0.517 | 4 | 0.593 | 8 | 0.537 |
| spot_control_mesh | 0.004 | 15,499 | -- | 0.073 | 9 | 0.597 | 5 | 0.793 | 9 | 0.630 |
| spot_low_resolution | 0.02 | 436 | -- | 0.027 | 6 | 0.413 | 4 | 0.443 | 6 | 0.397 |
| spot_low_resolution | 0.008 | 2,752 | -- | 0.040 | 7 | 0.493 | 4 | 0.603 | 8 | 0.527 |
| spot_low_resolution | 0.004 | 11,023 | -- | 0.073 | 9 | 0.583 | 5 | 0.713 | 9 | 0.570 |
| springer | 0.02 | 2,591 | -- | 0.040 | 7 | 0.497 | 4 | 0.583 | 7 | 0.450 |
| springer | 0.008 | 16,543 | -- | 0.087 | 9 | 0.633 | 5 | 0.767 | 9 | 0.617 |
| springer | 0.004 | 66,240 | -- | 0.147 | 10 | 0.770 | 5 | 0.850 | 10 | 0.710 |
| strawberry | 0.02 | 979 | -- | 0.040 | 7 | 0.460 | 4 | 0.540 | 6 | 0.437 |
| strawberry | 0.008 | 6,509 | -- | 0.063 | 8 | 0.557 | 4 | 0.653 | 9 | 0.550 |
| strawberry | 0.004 | 26,762 | -- | 0.103 | 9 | 0.653 | 5 | 0.750 | 9 | 0.640 |
| stuffedtoy | 0.02 | 97 | -- | 0.023 | 5 | 0.367 | 3 | 0.410 | 5 | 0.357 |
| stuffedtoy | 0.008 | 555 | -- | 0.037 | 7 | 0.440 | 4 | 0.550 | 7 | 0.443 |
| stuffedtoy | 0.004 | 2,309 | -- | 0.040 | 8 | 0.490 | 4 | 0.613 | 8 | 0.487 |
| sword-quad-dominant | 0.02 | 34 | -- | 0.020 | 5 | 0.373 | 2 | 0.290 | 5 | 0.357 |
| sword-quad-dominant | 0.008 | 80 | -- | 0.020 | 6 | 0.410 | 3 | 0.430 | 6 | 0.407 |
| sword-quad-dominant | 0.004 | 197 | -- | 0.030 | 6 | 0.410 | 4 | 0.460 | 7 | 0.427 |
| sword | 0.02 | 34 | -- | 0.020 | 5 | 0.363 | 2 | 0.280 | 5 | 0.350 |
| sword | 0.008 | 80 | -- | 0.020 | 6 | 0.440 | 3 | 0.400 | 6 | 0.400 |
| sword | 0.004 | 197 | -- | 0.030 | 6 | 0.427 | 4 | 0.473 | 7 | 0.443 |
| torus | 0.02 | 800 | -- | 0.030 | 7 | 0.450 | 3 | 0.463 | 7 | 0.417 |
| torus | 0.008 | 4,971 | -- | 0.053 | 8 | 0.530 | 4 | 0.650 | 8 | 0.553 |
| torus | 0.004 | 20,025 | -- | 0.113 | 9 | 0.650 | 4 | 0.747 | 9 | 0.620 |
| tower | 0.02 | 5,713 | -- | 0.060 | 8 | 0.623 | 5 | 0.673 | 9 | 0.563 |
| tower | 0.008 | 37,228 | -- | 0.123 | 9 | 0.677 | 6 | 0.830 | 10 | 0.663 |
| tower | 0.004 | 152,469 | -- | 0.267 | 10 | 0.890 | 6 | 0.970 | 10 | 0.820 |
| tower_holes | 0.02 | 4,740 | -- | 0.067 | 8 | 0.573 | 5 | 0.690 | 8 | 0.560 |
| tower_holes | 0.008 | 29,371 | -- | 0.107 | 9 | 0.650 | 6 | 0.833 | 10 | 0.623 |
| tower_holes | 0.004 | 118,862 | -- | 0.210 | 10 | 0.857 | 6 | 0.907 | 10 | 0.777 |
| tree | 0.02 | 457 | -- | 0.033 | 7 | 0.467 | 4 | 0.593 | 7 | 0.477 |
| tree | 0.008 | 2,351 | -- | 0.053 | 8 | 0.553 | 5 | 0.727 | 9 | 0.577 |
| tree | 0.004 | 9,295 | -- | 0.070 | 10 | 0.603 | 5 | 0.763 | 10 | 0.633 |
| tree_closed | 0.02 | 459 | -- | 0.037 | 6 | 0.440 | 4 | 0.583 | 7 | 0.457 |
| tree_closed | 0.008 | 2,410 | -- | 0.053 | 8 | 0.540 | 5 | 0.710 | 8 | 0.577 |
| tree_closed | 0.004 | 9,616 | -- | 0.070 | 9 | 0.630 | 5 | 0.790 | 9 | 0.617 |
| violin | 0.02 | 28 | -- | 0.020 | 4 | 0.300 | 3 | 0.343 | 4 | 0.287 |
| violin | 0.008 | 90 | -- | 0.020 | 6 | 0.383 | 3 | 0.407 | 6 | 0.363 |
| violin | 0.004 | 295 | -- | 0.030 | 6 | 0.433 | 4 | 0.497 | 7 | 0.440 |
| well | 0.02 | 1,143,998 | -- | 1.267 | 10 | 2.533 | 5 | 2.140 | 11 | 2.113 |
| well | 0.008 | — | — | — | — | — | — | — | — | — |
| well | 0.004 | — | — | — | — | — | — | — | — | — |
| well_boundary | 0.02 | 1,039,667 | -- | 1.103 | 10 | 2.390 | 6 | 2.163 | 11 | 1.900 |
| well_boundary | 0.008 | — | — | — | — | — | — | — | — | — |
| well_boundary | 0.004 | — | — | — | — | — | — | — | — | — |
| wingnut | 0.02 | 5,099 | -- | 0.060 | 8 | 0.567 | 4 | 0.663 | 8 | 0.547 |
| wingnut | 0.008 | 31,746 | -- | 0.120 | 9 | 0.677 | 5 | 0.823 | 10 | 0.663 |
| wingnut | 0.004 | 125,910 | -- | 0.203 | 10 | 0.843 | 6 | 0.890 | 10 | 0.750 |

16 of 204 cases did not run, covering house, house_boundary, nefertiti, nefertiti-lowres, well, well_boundary -- those meshes sit in coordinate systems far larger than the unit-scale ones, so a fixed voxel size asks for a grid thousands of voxels across and the run exhausts GPU memory before labeling begins. Not a convergence failure.

## What to take from it

- **Keep CAS.** It is faster on every case measured, it is already validated, and its result is order-independent by construction.

- **Drop P from consideration.** Slower than CAS with no bound to show for it.

- **R is the only real alternative**, at 9.40x on a stage that is a fraction of a percent of the pipeline. Whether an O(lg n) guarantee is worth that is a judgement, not a performance question.

- **CAS's own worst case is unmeasured.** Union-by-minimum without compression during the walk can degenerate to a Θ(n) chain in an adversarial edge order. Nothing like it appeared here, but nothing rules it out either -- a watchdog on walk length or retry count would turn that from an assumption into an observation.
