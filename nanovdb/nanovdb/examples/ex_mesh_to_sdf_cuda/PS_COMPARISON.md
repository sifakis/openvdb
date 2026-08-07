# Algorithm P vs algorithm S for the per-leaf union-find

Generated 2026-08-07 by `ex_mesh_to_sdf_cuda --sv-convergence <mesh> --voxel-size S`.

Both algorithms are from Liu & Tarjan, *Simple Concurrent Connected Components Algorithms* (ACM TOPC 9(2), 2022), which casts this family as a **connect** step followed by one or more **shortcut** steps, repeated until no parent changes.

| | main loop | proven step bound |
|---|---|---|
| **P** (what we ship today) | `parent-connect; shortcut` | **none.** Section 4.2: *"we are unable to prove even an O(lg² n) bound"*, and an earlier published analysis was withdrawn as incorrect |
| **S** | `parent-connect; shortcut until flat` | **O(min{d, lg n} · lg n)** (theorem 4.1), which is a constant for a fixed 8³ leaf |

So S is the one we can point at a theorem for. The question this measures is what it costs.

## The short answer

| | P | S |
|---|---:|---:|
| worst rounds, over everything | 6 | 6 |
| rounds, typical | higher | lower in 128 of 188, **higher in 4** |
| total shortcut steps | **618,049,850** | 807,329,714 &nbsp;(+31%) |
| total labeling time | **12,180 ms** | 13,200 ms |
| median time ratio S/P | | **1.08x** |

**S costs about 8% more time and does not lower the worst case.** Both top out at 6 rounds. S is
usually a round quicker, but that round is bought with several extra shortcuts, and on four
configurations it needed *more* rounds than P, not fewer.

The 11 configurations where S came out faster are all tiny -- 79 to 920 leaves -- where the pass is
dominated by launch overhead rather than by either algorithm. Excluding those changes nothing: the
median ratio is 1.08 on the 24 largest configurations and 1.04 on the 88 smallest, so the gap is
consistent rather than scale-dependent.

This is what you would expect if a hook and a shortcut cost about the same. A round of S replaces one
shortcut with as many as it takes to flatten the forest, so it trades one step for several. Hooking
uses shared-memory atomics, which are cheap; shortcutting is pointer jumping through shared memory,
which is also cheap. With neither dominating, the trade loses.

**So the choice is not about speed. It is about whether a proven bound is worth 8%.**

## How this was measured

- Each mesh is rasterized to a narrow band, then labeled twice, once under each algorithm.
- Both grids the pipeline actually labels are covered: the **un-pruned** band (what the surface partition runs on) and the **barrier-pruned** band (what each surface runs on). The figures below add the two.
- Each mesh is placed on the voxel lattice three ways — identity, mirror in all three axes, and a 45° rotation about z. A voxel's label is its offset `64x + 8y + z`, so moving the geometry relative to the lattice re-assigns every label without changing the surface, and that alone can change the round count. `rounds` below is the worst over the three; time and shortcut counts are summed over them.
- Timing is a CUDA event around the whole labeling call, after a discarded warm-up run.
- P and S were checked to agree on the component count for every run.

## Per mesh

`rounds` is the outer-loop count; `shortcuts` is how many compress steps ran, which is where the two differ; `ms` is the labeling pass. Lower is better in all columns.

| mesh | voxel size | leaves | P rounds | S rounds | P shortcuts | S shortcuts | P ms | S ms | S/P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| armadillo | 0.02 | 151 | 5 | 4 | 4,604 | 6,192 | 0.86 | 0.87 | 1.01 |
| armadillo | 0.008 | 802 | 5 | 4 | 26,771 | 35,552 | 1.03 | 1.12 | 1.09 |
| armadillo | 0.004 | 3,150 | 5 | 4 | 106,872 | 139,984 | 2.32 | 2.51 | 1.08 |
| boat | 0.02 | 3,004 | 5 | 4 | 97,212 | 124,974 | 2.07 | 2.20 | 1.06 |
| boat | 0.008 | 19,434 | 5 | 4 | 620,712 | 795,988 | 9.99 | 10.89 | 1.09 |
| boat | 0.004 | 77,816 | 5 | 5 | 2,460,361 | 3,152,807 | 42.56 | 46.11 | 1.08 |
| boot | 0.02 | 22 | 4 | 3 | 637 | 816 | 0.66 | 0.66 | 1.00 |
| boot | 0.008 | 72 | 4 | 4 | 2,217 | 2,920 | 0.82 | 0.84 | 1.02 |
| boot | 0.004 | 246 | 4 | 4 | 8,194 | 10,843 | 0.95 | 0.96 | 1.01 |
| brucewick | 0.02 | 1,070 | 5 | 4 | 36,047 | 46,811 | 1.23 | 1.28 | 1.04 |
| brucewick | 0.008 | 6,822 | 5 | 4 | 230,100 | 295,423 | 3.89 | 4.21 | 1.08 |
| brucewick | 0.004 | 27,653 | 5 | 4 | 914,424 | 1,171,643 | 14.16 | 15.60 | 1.10 |
| bunny | 0.02 | 454 | 5 | 4 | 15,132 | 19,898 | 1.03 | 1.06 | 1.03 |
| bunny | 0.008 | 2,822 | 5 | 4 | 95,419 | 124,509 | 2.06 | 2.21 | 1.07 |
| bunny | 0.004 | 11,298 | 5 | 4 | 378,905 | 490,012 | 6.08 | 6.81 | 1.12 |
| bunny_hr | 0.02 | 462 | 5 | 4 | 15,176 | 20,026 | 1.20 | 1.24 | 1.03 |
| bunny_hr | 0.008 | 2,854 | 5 | 4 | 96,671 | 126,272 | 2.35 | 2.50 | 1.06 |
| bunny_hr | 0.004 | 11,491 | 5 | 4 | 385,763 | 498,220 | 7.22 | 7.78 | 1.08 |
| cat-low-resolution | 0.02 | 1,017 | 5 | 4 | 34,147 | 45,079 | 1.27 | 1.34 | 1.06 |
| cat-low-resolution | 0.008 | 6,478 | 5 | 4 | 218,383 | 284,714 | 3.94 | 4.32 | 1.10 |
| cat-low-resolution | 0.004 | 26,138 | 5 | 4 | 878,276 | 1,139,761 | 13.84 | 15.30 | 1.11 |
| cat | 0.02 | 1,021 | 5 | 4 | 34,371 | 45,455 | 1.79 | 1.92 | 1.07 |
| cat | 0.008 | 6,536 | 5 | 4 | 220,401 | 287,962 | 5.81 | 6.44 | 1.11 |
| cat | 0.004 | 26,431 | 5 | 4 | 888,271 | 1,155,754 | 20.15 | 22.50 | 1.12 |
| cheese | 0.02 | 3,646 | 5 | 4 | 121,132 | 154,549 | 3.51 | 3.78 | 1.08 |
| cheese | 0.008 | 22,816 | 5 | 5 | 759,828 | 964,557 | 16.97 | 18.71 | 1.10 |
| cheese | 0.004 | 91,591 | 5 | 5 | 3,048,098 | 3,856,420 | 70.24 | 76.53 | 1.09 |
| cow-low-resolution | 0.02 | 8,180 | 5 | 4 | 273,189 | 358,601 | 6.82 | 7.55 | 1.11 |
| cow-low-resolution | 0.008 | 52,437 | 5 | 4 | 1,749,722 | 2,275,817 | 30.69 | 34.47 | 1.12 |
| cow-low-resolution | 0.004 | 211,338 | 5 | 4 | 7,039,743 | 9,127,934 | 127.46 | 137.03 | 1.08 |
| cow | 0.02 | 8,286 | 5 | 4 | 277,095 | 365,145 | 5.64 | 6.24 | 1.11 |
| cow | 0.008 | 53,342 | 5 | 5 | 1,785,980 | 2,340,887 | 37.00 | 40.62 | 1.10 |
| cow | 0.004 | 215,332 | 5 | 5 | 7,213,573 | 9,427,346 | 146.32 | 156.67 | 1.07 |
| cube | 0.02 | 1,960 | 4 | 3 | 58,598 | 71,332 | 1.65 | 1.68 | 1.02 |
| cube | 0.008 | 10,128 | 4 | 3 | 242,513 | 297,353 | 4.28 | 4.47 | 1.04 |
| cube | 0.004 | 48,240 | 4 | 3 | 1,500,957 | 1,817,589 | 24.31 | 25.62 | 1.05 |
| cube_no_bottom | 0.02 | 1,722 | 4 | 3 | 51,611 | 63,079 | 1.38 | 1.44 | 1.04 |
| cube_no_bottom | 0.008 | 8,268 | 4 | 4 | 202,898 | 250,736 | 3.71 | 3.96 | 1.07 |
| cube_no_bottom | 0.004 | 40,560 | 4 | 3 | 1,265,938 | 1,538,014 | 20.15 | 21.56 | 1.07 |
| demosthenes-low-res | 0.02 | 920 | 5 | 4 | 31,280 | 41,387 | 1.30 | 1.29 | 0.99 |
| demosthenes-low-res | 0.008 | 5,870 | 5 | 4 | 198,247 | 259,956 | 3.73 | 4.09 | 1.10 |
| demosthenes-low-res | 0.004 | 23,518 | 5 | 4 | 797,564 | 1,040,448 | 12.78 | 14.07 | 1.10 |
| demosthenes | 0.02 | 930 | 5 | 4 | 31,612 | 41,796 | 1.53 | 1.59 | 1.04 |
| demosthenes | 0.008 | 5,935 | 5 | 5 | 200,455 | 264,133 | 4.32 | 4.77 | 1.10 |
| demosthenes | 0.004 | 23,892 | 5 | 5 | 809,656 | 1,061,601 | 15.28 | 16.86 | 1.10 |
| dragon | 0.02 | 125 | 4 | 4 | 3,979 | 5,352 | 1.16 | 1.09 | 0.94 |
| dragon | 0.008 | 744 | 5 | 4 | 25,261 | 34,172 | 1.36 | 1.42 | 1.04 |
| dragon | 0.004 | 3,270 | 5 | 5 | 110,951 | 146,891 | 2.95 | 3.22 | 1.09 |
| falconstatue | 0.02 | 411 | 5 | 4 | 13,606 | 18,237 | 1.02 | 1.06 | 1.04 |
| falconstatue | 0.008 | 2,691 | 5 | 5 | 90,266 | 118,726 | 2.14 | 2.32 | 1.08 |
| falconstatue | 0.004 | 11,036 | 5 | 5 | 369,285 | 481,824 | 6.39 | 7.06 | 1.10 |
| falconstatue_boundary | 0.02 | 408 | 5 | 4 | 13,555 | 18,210 | 1.03 | 1.06 | 1.03 |
| falconstatue_boundary | 0.008 | 2,635 | 5 | 5 | 88,609 | 116,770 | 2.16 | 2.35 | 1.09 |
| falconstatue_boundary | 0.004 | 10,698 | 5 | 5 | 359,155 | 469,968 | 6.27 | 6.94 | 1.11 |
| fish | 0.02 | 704 | 5 | 5 | 23,967 | 31,500 | 1.06 | 1.12 | 1.06 |
| fish | 0.008 | 4,568 | 5 | 4 | 155,057 | 200,126 | 2.84 | 3.10 | 1.09 |
| fish | 0.004 | 18,368 | 5 | 4 | 626,788 | 805,039 | 9.73 | 10.76 | 1.11 |
| fish_control_mesh | 0.02 | 859 | 5 | 4 | 28,346 | 37,370 | 1.33 | 1.36 | 1.02 |
| fish_control_mesh | 0.008 | 5,637 | 5 | 4 | 188,071 | 244,730 | 3.32 | 3.63 | 1.09 |
| fish_control_mesh | 0.004 | 23,399 | 5 | 5 | 773,148 | 999,839 | 17.96 | 19.91 | 1.11 |
| fish_low_resolution | 0.02 | 649 | 4 | 4 | 21,387 | 27,704 | 1.51 | 1.61 | 1.07 |
| fish_low_resolution | 0.008 | 3,975 | 5 | 4 | 136,236 | 175,878 | 4.02 | 4.34 | 1.08 |
| fish_low_resolution | 0.004 | 16,337 | 5 | 4 | 551,157 | 708,623 | 12.81 | 14.06 | 1.10 |
| goathead | 0.02 | 30,441 | 5 | 4 | 1,027,903 | 1,328,633 | 22.81 | 25.40 | 1.11 |
| goathead | 0.008 | 190,762 | 5 | 4 | 6,449,834 | 8,317,456 | 154.21 | 169.75 | 1.10 |
| goathead | 0.004 | 763,519 | 5 | 4 | 25,817,813 | 33,263,217 | 498.82 | 543.44 | 1.09 |
| hairball | 0.02 | 89,812 | 6 | 5 | 3,234,280 | 4,684,221 | 67.75 | 75.35 | 1.11 |
| hairball | 0.008 | 879,909 | 5 | 6 | 30,377,219 | 41,986,744 | 606.88 | 684.39 | 1.13 |
| hairball | 0.004 | 3,866,231 | 6 | 5 | 131,756,005 | 177,734,288 | 2687.70 | 2935.51 | 1.09 |
| hammer | 0.02 | 1,704 | 4 | 4 | 55,186 | 71,052 | 1.70 | 1.82 | 1.07 |
| hammer | 0.008 | 10,842 | 4 | 4 | 356,002 | 455,159 | 6.42 | 6.97 | 1.09 |
| hammer | 0.004 | 44,135 | 5 | 4 | 1,399,676 | 1,785,939 | 25.80 | 27.70 | 1.07 |
| hand | 0.02 | 77 | 4 | 4 | 2,491 | 3,318 | 0.99 | 1.00 | 1.01 |
| hand | 0.008 | 394 | 5 | 4 | 13,318 | 17,549 | 1.25 | 1.02 | 0.82 |
| hand | 0.004 | 1,554 | 5 | 4 | 52,399 | 68,096 | 1.64 | 1.76 | 1.07 |
| hand_closed | 0.02 | 77 | 4 | 4 | 2,480 | 3,294 | 0.76 | 0.78 | 1.03 |
| hand_closed | 0.008 | 400 | 4 | 4 | 13,495 | 17,746 | 1.09 | 1.11 | 1.02 |
| hand_closed | 0.004 | 1,596 | 5 | 4 | 53,594 | 69,715 | 1.78 | 1.87 | 1.05 |
| hand_lowres | 0.02 | 79 | 4 | 4 | 2,550 | 3,400 | 0.99 | 0.97 | 0.98 |
| hand_lowres | 0.008 | 406 | 4 | 4 | 13,659 | 18,040 | 0.96 | 0.99 | 1.03 |
| hand_lowres | 0.004 | 1,598 | 5 | 4 | 53,962 | 70,590 | 1.61 | 1.71 | 1.06 |
| house | 0.02 | — | — | — | — | — | — | — | did not run |
| house | 0.008 | — | — | — | — | — | — | — | did not run |
| house | 0.004 | — | — | — | — | — | — | — | did not run |
| house_boundary | 0.02 | — | — | — | — | — | — | — | did not run |
| house_boundary | 0.008 | — | — | — | — | — | — | — | did not run |
| house_boundary | 0.004 | — | — | — | — | — | — | — | did not run |
| human_man | 0.02 | 171 | 4 | 4 | 5,602 | 7,487 | 0.84 | 0.82 | 0.98 |
| human_man | 0.008 | 989 | 5 | 4 | 32,475 | 43,155 | 1.21 | 1.24 | 1.02 |
| human_man | 0.004 | 3,877 | 5 | 4 | 128,294 | 167,620 | 2.84 | 3.05 | 1.07 |
| human_neutral | 0.02 | 169 | 4 | 4 | 5,318 | 7,034 | 0.91 | 0.93 | 1.02 |
| human_neutral | 0.008 | 884 | 5 | 4 | 29,730 | 39,294 | 1.16 | 1.22 | 1.05 |
| human_neutral | 0.004 | 3,631 | 5 | 4 | 120,536 | 157,235 | 2.49 | 2.73 | 1.10 |
| human_woman | 0.02 | 148 | 4 | 4 | 4,860 | 6,513 | 0.82 | 0.80 | 0.98 |
| human_woman | 0.008 | 848 | 5 | 4 | 28,549 | 37,957 | 1.26 | 1.31 | 1.04 |
| human_woman | 0.004 | 3,401 | 5 | 4 | 112,392 | 147,051 | 2.49 | 2.64 | 1.06 |
| koala | 0.02 | 9,070 | 5 | 4 | 304,463 | 392,650 | 5.88 | 6.41 | 1.09 |
| koala | 0.008 | 56,842 | 5 | 4 | 1,897,173 | 2,438,770 | 35.22 | 38.11 | 1.08 |
| koala | 0.004 | 225,890 | 5 | 4 | 7,632,406 | 9,801,871 | 138.04 | 147.33 | 1.07 |
| koala_low_resolution | 0.02 | 8,876 | 5 | 4 | 297,654 | 384,833 | 4.96 | 5.47 | 1.10 |
| koala_low_resolution | 0.008 | 55,880 | 5 | 4 | 1,869,997 | 2,408,074 | 30.85 | 34.36 | 1.11 |
| koala_low_resolution | 0.004 | 223,453 | 5 | 4 | 7,514,965 | 9,661,063 | 134.72 | 145.52 | 1.08 |
| lionstatue | 0.02 | 666 | 5 | 4 | 22,045 | 29,355 | 1.30 | 1.34 | 1.03 |
| lionstatue | 0.008 | 3,942 | 5 | 5 | 132,493 | 174,416 | 3.01 | 3.25 | 1.08 |
| lionstatue | 0.004 | 16,036 | 5 | 4 | 534,323 | 694,956 | 9.22 | 10.24 | 1.11 |
| mountain | 0.02 | 123,940 | 5 | 4 | 4,163,715 | 5,373,007 | 75.71 | 80.75 | 1.07 |
| mountain | 0.008 | 768,838 | 5 | 4 | 25,872,751 | 33,356,962 | 503.16 | 542.30 | 1.08 |
| mountain | 0.004 | 3,068,538 | 5 | 4 | 103,226,752 | 133,058,859 | 2161.45 | 2262.09 | 1.05 |
| mushroom | 0.02 | 2,371 | 5 | 4 | 80,593 | 105,337 | 2.16 | 2.28 | 1.06 |
| mushroom | 0.008 | 15,124 | 5 | 4 | 509,017 | 658,684 | 8.95 | 9.69 | 1.08 |
| mushroom | 0.004 | 60,698 | 5 | 4 | 2,044,475 | 2,636,832 | 35.79 | 39.82 | 1.11 |
| nefertiti-lowres | 0.02 | — | — | — | — | — | — | — | did not run |
| nefertiti-lowres | 0.008 | — | — | — | — | — | — | — | did not run |
| nefertiti-lowres | 0.004 | — | — | — | — | — | — | — | did not run |
| nefertiti | 0.02 | — | — | — | — | — | — | — | did not run |
| nefertiti | 0.008 | — | — | — | — | — | — | — | did not run |
| nefertiti | 0.004 | — | — | — | — | — | — | — | did not run |
| parsnip | 0.02 | 1,445 | 4 | 4 | 48,320 | 63,179 | 1.66 | 1.81 | 1.09 |
| parsnip | 0.008 | 9,181 | 5 | 5 | 308,104 | 401,339 | 6.30 | 6.96 | 1.10 |
| parsnip | 0.004 | 37,087 | 5 | 4 | 1,247,305 | 1,616,705 | 23.01 | 25.39 | 1.10 |
| penguin | 0.02 | 930 | 5 | 4 | 31,231 | 41,012 | 1.31 | 1.31 | 1.00 |
| penguin | 0.008 | 6,165 | 5 | 4 | 206,507 | 267,605 | 4.03 | 4.27 | 1.06 |
| penguin | 0.004 | 25,661 | 5 | 4 | 851,718 | 1,097,612 | 13.91 | 15.29 | 1.10 |
| penguin_control_mesh | 0.02 | 1,099 | 5 | 4 | 37,239 | 49,137 | 1.46 | 1.47 | 1.01 |
| penguin_control_mesh | 0.008 | 7,637 | 5 | 4 | 256,329 | 334,548 | 4.30 | 4.74 | 1.10 |
| penguin_control_mesh | 0.004 | 32,083 | 5 | 4 | 1,067,138 | 1,379,728 | 16.60 | 18.12 | 1.09 |
| penguin_hr | 0.02 | 927 | 5 | 4 | 31,124 | 40,723 | 1.22 | 1.23 | 1.01 |
| penguin_hr | 0.008 | 6,128 | 5 | 4 | 204,884 | 265,242 | 3.82 | 4.14 | 1.08 |
| penguin_hr | 0.004 | 25,418 | 5 | 4 | 843,754 | 1,086,233 | 13.99 | 15.38 | 1.10 |
| pizza | 0.02 | 132 | 4 | 4 | 4,096 | 5,507 | 1.03 | 1.02 | 0.99 |
| pizza | 0.008 | 749 | 5 | 4 | 24,255 | 32,591 | 1.28 | 1.52 | 1.19 |
| pizza | 0.004 | 2,985 | 5 | 5 | 97,862 | 131,264 | 2.39 | 2.60 | 1.09 |
| plane | 0.02 | 9,565 | 5 | 4 | 307,881 | 400,630 | 5.48 | 6.03 | 1.10 |
| plane | 0.008 | 60,792 | 5 | 4 | 1,946,233 | 2,518,356 | 33.75 | 37.23 | 1.10 |
| plane | 0.004 | 243,852 | 5 | 4 | 7,821,487 | 10,103,230 | 191.69 | 212.41 | 1.11 |
| plane_holes | 0.02 | 9,345 | 5 | 4 | 301,514 | 392,542 | 5.18 | 5.66 | 1.09 |
| plane_holes | 0.008 | 58,231 | 5 | 4 | 1,866,558 | 2,417,254 | 32.15 | 35.57 | 1.11 |
| plane_holes | 0.004 | 231,378 | 5 | 4 | 7,421,349 | 9,588,820 | 137.71 | 149.55 | 1.09 |
| scorpion | 0.02 | 6,174 | 5 | 5 | 210,480 | 277,806 | 4.11 | 4.56 | 1.11 |
| scorpion | 0.008 | 39,895 | 5 | 4 | 1,357,160 | 1,757,546 | 23.87 | 26.46 | 1.11 |
| scorpion | 0.004 | 160,853 | 5 | 5 | 5,474,170 | 7,048,167 | 99.14 | 107.02 | 1.08 |
| scorpion_low_resolution | 0.02 | 6,217 | 5 | 5 | 211,979 | 278,775 | 3.68 | 4.06 | 1.10 |
| scorpion_low_resolution | 0.008 | 40,227 | 5 | 4 | 1,367,032 | 1,771,549 | 21.81 | 24.19 | 1.11 |
| scorpion_low_resolution | 0.004 | 161,445 | 5 | 4 | 5,494,145 | 7,086,618 | 95.61 | 104.95 | 1.10 |
| skull | 0.02 | 40,486 | 5 | 4 | 1,352,154 | 1,743,520 | 23.65 | 26.60 | 1.12 |
| skull | 0.008 | 254,923 | 5 | 4 | 8,520,240 | 10,946,404 | 153.13 | 164.55 | 1.07 |
| skull | 0.004 | 1,021,048 | 5 | 4 | 34,100,701 | 43,787,367 | 653.08 | 683.10 | 1.05 |
| skull_low_resolution | 0.02 | 40,127 | 5 | 4 | 1,340,041 | 1,728,164 | 21.61 | 23.92 | 1.11 |
| skull_low_resolution | 0.008 | 252,742 | 5 | 4 | 8,444,618 | 10,855,608 | 153.43 | 164.73 | 1.07 |
| skull_low_resolution | 0.004 | 1,012,874 | 5 | 4 | 33,832,869 | 43,444,782 | 651.25 | 702.32 | 1.08 |
| sphere | 0.02 | 1,038 | 5 | 3 | 35,268 | 45,348 | 1.23 | 1.25 | 1.02 |
| sphere | 0.008 | 6,299 | 5 | 3 | 214,653 | 275,752 | 3.78 | 4.14 | 1.10 |
| sphere | 0.004 | 25,296 | 5 | 3 | 861,871 | 1,108,465 | 13.66 | 14.93 | 1.09 |
| spot | 0.02 | 438 | 5 | 4 | 14,882 | 19,765 | 1.00 | 1.01 | 1.01 |
| spot | 0.008 | 2,774 | 5 | 4 | 93,607 | 121,626 | 2.04 | 2.18 | 1.07 |
| spot | 0.004 | 11,155 | 5 | 4 | 372,907 | 481,823 | 6.15 | 6.73 | 1.09 |
| spot_control_mesh | 0.02 | 582 | 4 | 4 | 19,067 | 25,388 | 1.04 | 1.04 | 1.00 |
| spot_control_mesh | 0.008 | 3,829 | 5 | 5 | 128,150 | 167,397 | 2.51 | 2.73 | 1.09 |
| spot_control_mesh | 0.004 | 15,499 | 5 | 5 | 516,051 | 669,509 | 8.28 | 9.07 | 1.10 |
| spot_low_resolution | 0.02 | 436 | 4 | 4 | 14,739 | 19,545 | 1.04 | 1.06 | 1.02 |
| spot_low_resolution | 0.008 | 2,752 | 5 | 4 | 92,704 | 120,438 | 2.09 | 2.26 | 1.08 |
| spot_low_resolution | 0.004 | 11,023 | 5 | 4 | 368,753 | 477,547 | 5.99 | 6.52 | 1.09 |
| springer | 0.02 | 2,591 | 5 | 4 | 86,486 | 113,357 | 2.02 | 2.17 | 1.07 |
| springer | 0.008 | 16,543 | 5 | 4 | 548,504 | 710,218 | 9.44 | 10.27 | 1.09 |
| springer | 0.004 | 66,240 | 5 | 4 | 2,230,559 | 2,872,768 | 38.35 | 43.12 | 1.12 |
| strawberry | 0.02 | 979 | 5 | 4 | 33,462 | 44,895 | 1.47 | 1.47 | 1.00 |
| strawberry | 0.008 | 6,509 | 5 | 4 | 221,947 | 293,368 | 4.82 | 5.32 | 1.10 |
| strawberry | 0.004 | 26,762 | 5 | 4 | 910,987 | 1,188,271 | 16.79 | 18.60 | 1.11 |
| stuffedtoy | 0.02 | 97 | 4 | 4 | 3,185 | 4,256 | 0.73 | 0.76 | 1.04 |
| stuffedtoy | 0.008 | 555 | 5 | 4 | 18,596 | 24,576 | 0.87 | 0.89 | 1.02 |
| stuffedtoy | 0.004 | 2,309 | 5 | 4 | 77,873 | 102,378 | 1.94 | 2.08 | 1.07 |
| sword-quad-dominant | 0.02 | 34 | 3 | 4 | 1,013 | 1,265 | 0.79 | 0.81 | 1.03 |
| sword-quad-dominant | 0.008 | 80 | 4 | 4 | 2,481 | 3,268 | 0.87 | 0.88 | 1.01 |
| sword-quad-dominant | 0.004 | 197 | 4 | 4 | 6,195 | 8,097 | 0.87 | 0.89 | 1.02 |
| sword | 0.02 | 34 | 3 | 4 | 1,013 | 1,265 | 0.74 | 0.76 | 1.03 |
| sword | 0.008 | 80 | 4 | 4 | 2,481 | 3,268 | 0.78 | 0.76 | 0.97 |
| sword | 0.004 | 197 | 4 | 4 | 6,196 | 8,101 | 0.87 | 0.86 | 0.99 |
| torus | 0.02 | 800 | 4 | 4 | 25,060 | 32,506 | 1.08 | 1.13 | 1.05 |
| torus | 0.008 | 4,971 | 5 | 3 | 167,380 | 214,628 | 3.09 | 3.34 | 1.08 |
| torus | 0.004 | 20,025 | 5 | 4 | 662,589 | 851,528 | 10.84 | 11.91 | 1.10 |
| tower | 0.02 | 5,713 | 5 | 4 | 187,692 | 243,457 | 4.10 | 4.49 | 1.10 |
| tower | 0.008 | 37,228 | 5 | 5 | 1,216,517 | 1,569,657 | 21.65 | 23.81 | 1.10 |
| tower | 0.004 | 152,469 | 5 | 5 | 4,987,335 | 6,393,858 | 92.52 | 100.19 | 1.08 |
| tower_holes | 0.02 | 4,740 | 5 | 5 | 157,034 | 206,050 | 3.29 | 3.59 | 1.09 |
| tower_holes | 0.008 | 29,371 | 5 | 5 | 982,846 | 1,280,012 | 16.34 | 18.03 | 1.10 |
| tower_holes | 0.004 | 118,862 | 5 | 4 | 3,938,383 | 5,090,278 | 71.44 | 77.73 | 1.09 |
| tree | 0.02 | 457 | 5 | 4 | 14,682 | 19,425 | 1.04 | 1.07 | 1.03 |
| tree | 0.008 | 2,351 | 5 | 4 | 79,018 | 105,261 | 2.28 | 2.49 | 1.09 |
| tree | 0.004 | 9,295 | 5 | 4 | 315,504 | 413,716 | 6.33 | 7.03 | 1.11 |
| tree_closed | 0.02 | 459 | 5 | 4 | 14,735 | 19,458 | 1.05 | 1.09 | 1.04 |
| tree_closed | 0.008 | 2,410 | 5 | 4 | 80,267 | 106,602 | 2.29 | 2.48 | 1.08 |
| tree_closed | 0.004 | 9,616 | 5 | 4 | 326,121 | 426,441 | 6.50 | 7.20 | 1.11 |
| violin | 0.02 | 28 | 3 | 4 | 801 | 1,056 | 0.78 | 0.84 | 1.08 |
| violin | 0.008 | 90 | 4 | 4 | 2,558 | 3,315 | 0.83 | 0.82 | 0.99 |
| violin | 0.004 | 295 | 4 | 4 | 9,381 | 12,369 | 1.04 | 1.03 | 0.99 |
| well | 0.02 | 1,143,998 | 5 | 5 | 36,584,793 | 46,857,198 | 673.13 | 775.46 | 1.15 |
| well | 0.008 | — | — | — | — | — | — | — | did not run |
| well | 0.004 | — | — | — | — | — | — | — | did not run |
| well_boundary | 0.02 | 1,039,667 | 5 | 5 | 33,391,938 | 43,064,013 | 623.02 | 674.71 | 1.08 |
| well_boundary | 0.008 | — | — | — | — | — | — | — | did not run |
| well_boundary | 0.004 | — | — | — | — | — | — | — | did not run |
| wingnut | 0.02 | 5,099 | 5 | 4 | 163,934 | 210,430 | 2.84 | 3.11 | 1.10 |
| wingnut | 0.008 | 31,746 | 5 | 5 | 989,657 | 1,270,032 | 15.38 | 16.82 | 1.09 |
| wingnut | 0.004 | 125,910 | 5 | 4 | 3,977,495 | 5,098,181 | 69.40 | 75.63 | 1.09 |

16 of 204 configurations did not run, covering house, house_boundary, nefertiti, nefertiti-lowres, well, well_boundary. These meshes sit in coordinate systems tens to hundreds of times larger than the unit-scale ones, so a fixed voxel size asks for a grid thousands of voxels across and the run exhausts GPU memory before labeling starts. It is not a convergence failure.

## What to take from it

- **S buys a theorem, not speed.** Its `O(min{d, lg n} lg n)` bound (theorem 4.1) is the thing P does
  not have and, per section 4.2, nobody has managed to prove for P. The measured price is about 8%
  of the labeling pass, which is itself a small share of the mesh-to-SDF pipeline.
- **S does not improve the worst case.** Both reach 6 rounds at most, against a cap of 64. If the
  concern is the cap being too tight, neither algorithm changes the picture; what changes it is
  detecting the cap being reached, which nothing currently does.
- **The two are interchangeable at the call site:** `ConnectedComponents::setLeafSchedule()`, taking
  `LeafSchedule::Parent` (P, the default) or `LeafSchedule::Flatten` (S). Both share one solver, so
  the counting and mask passes cannot drift apart, and both were checked to produce identical
  component counts on every run here.
- **If a proven `O(lg n)` is wanted rather than `O(lg^2 n)`,** the paper's algorithm R is the target,
  not S: it is `parent-root-connect; shortcut`, which differs from what we have today only by hooking
  onto a parent when that parent is a root. That is a smaller change than S, and theorem 4.19 gives
  it `O(lg n)`. It has not been measured here.
