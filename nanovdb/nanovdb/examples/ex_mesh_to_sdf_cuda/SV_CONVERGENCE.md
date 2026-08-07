# Per-leaf union-find: how many rounds does it take to converge?

Generated 2026-08-06 by `ex_mesh_to_sdf_cuda --sv-convergence`.

`ConnectedComponents` labels each leaf's active voxels with a Shiloach-Vishkin union-find that loops until nothing changes, under a safety cap of **64** (`LeafUnionFind::MaxConvergenceIters`). The cap is there to catch a non-terminating bug, not to bound legitimate input: a leaf that ran out of rounds would be left under-labeled, and nothing would say so. No tight bound has been proven, so this measures one.

**Worst seen anywhere: 6 rounds. The cap is 64.**

## What the numbers mean

| term | meaning |
|---|---|
| **round** | one pass of the loop: a hooking step followed by one pointer-jumping step. A leaf's round count is the pass on which its labels stopped changing. Reported per leaf, then maximised. |
| **placement** | where the mesh sits on the voxel lattice. A voxel's label is its offset inside the leaf, `64x + 8y + z`, so moving the geometry relative to the lattice re-assigns every label without touching the surface. Eleven are used: the identity, three axis swaps and a 3-cycle, two mirrors, three off-axis rotations (45 deg about z, 30 about y, 17 about x), and a half-voxel shift. |
| **un-pruned grid** | the rasterized narrow band as it comes out of `MeshToGrid`. This is what the surface partition labels. |
| **pruned grid** | that band with the barrier shell removed (`computeDerivedTopology`). This is what each individual surface labels. |
| **configuration** | one (mesh, voxel size) pair. Each is measured 22 times: 11 placements x the two grids above. 68 meshes x 3 voxel sizes = 204 configurations. |
| **worst** / **best** | the largest and smallest round count among a configuration's 22 measurements. If they differ, the placement alone changed the answer. |
| **leaves** | the largest leaf count among the configuration's placements, as a sense of scale. |

## Why each mesh is measured under eleven placements

Hooking follows labels, and a label is fixed by position rather than by anything geometric. Two arrangements that are congruent as graphs -- same adjacency, same diameter, same component count -- can therefore need different numbers of rounds. Measuring one orientation of a mesh says nothing about the others.

That is not hypothetical here: **107 of the 189 completed configurations came out differently under different placements.** Swapping two axes or mirroring one is usually worth a round.

## Results by mesh

| mesh | voxel size | leaves | worst | best | notes |
|---|---:|---:|---:|---:|---|
| armadillo | 0.02 | 151 | 5 | 4 |  |
| armadillo | 0.008 | 802 | 5 | 4 |  |
| armadillo | 0.004 | 3,155 | 5 | 4 |  |
| boat | 0.02 | 3,004 | 5 | 4 |  |
| boat | 0.008 | 19,434 | 5 | 4 |  |
| boat | 0.004 | 77,816 | 5 | 4 |  |
| boot | 0.02 | 22 | 4 | 3 |  |
| boot | 0.008 | 73 | 4 | 3 |  |
| boot | 0.004 | 254 | 5 | 4 |  |
| brucewick | 0.02 | 1,070 | 5 | 4 |  |
| brucewick | 0.008 | 6,822 | 5 | 4 |  |
| brucewick | 0.004 | 27,653 | 5 | 5 |  |
| bunny | 0.02 | 454 | 5 | 4 |  |
| bunny | 0.008 | 2,844 | 5 | 4 |  |
| bunny | 0.004 | 11,298 | 5 | 4 |  |
| bunny_hr | 0.02 | 462 | 5 | 4 |  |
| bunny_hr | 0.008 | 2,867 | 5 | 4 |  |
| bunny_hr | 0.004 | 11,491 | 5 | 5 |  |
| cat-low-resolution | 0.02 | 1,044 | 5 | 4 |  |
| cat-low-resolution | 0.008 | 6,564 | 5 | 4 |  |
| cat-low-resolution | 0.004 | 26,396 | 5 | 5 |  |
| cat | 0.02 | 1,053 | 5 | 4 |  |
| cat | 0.008 | 6,630 | 5 | 4 |  |
| cat | 0.004 | 26,704 | 5 | 5 |  |
| cheese | 0.02 | 3,725 | 5 | 4 |  |
| cheese | 0.008 | 23,445 | 5 | 4 |  |
| cheese | 0.004 | 93,735 | 5 | 5 |  |
| cow-low-resolution | 0.02 | 8,223 | 5 | 5 |  |
| cow-low-resolution | 0.008 | 52,504 | 5 | 5 |  |
| cow-low-resolution | 0.004 | 211,831 | 5 | 5 |  |
| cow | 0.02 | 8,315 | 5 | 4 |  |
| cow | 0.008 | 53,430 | 5 | 5 |  |
| cow | 0.004 | 216,046 | 5 | 5 |  |
| cube | 0.02 | 1,960 | 4 | 3 |  |
| cube | 0.008 | 10,128 | 4 | 3 |  |
| cube | 0.004 | 48,240 | 4 | 3 |  |
| cube_no_bottom | 0.02 | 1,722 | 4 | 3 |  |
| cube_no_bottom | 0.008 | 8,603 | 4 | 3 |  |
| cube_no_bottom | 0.004 | 40,560 | 4 | 3 |  |
| demosthenes-low-res | 0.02 | 920 | 5 | 4 |  |
| demosthenes-low-res | 0.008 | 5,870 | 5 | 4 |  |
| demosthenes-low-res | 0.004 | 23,573 | 5 | 5 |  |
| demosthenes | 0.02 | 930 | 5 | 4 |  |
| demosthenes | 0.008 | 5,935 | 5 | 5 |  |
| demosthenes | 0.004 | 23,935 | 5 | 5 |  |
| dragon | 0.02 | 125 | 4 | 4 |  |
| dragon | 0.008 | 744 | 5 | 4 |  |
| dragon | 0.004 | 3,270 | 5 | 4 |  |
| falconstatue | 0.02 | 413 | 5 | 4 |  |
| falconstatue | 0.008 | 2,691 | 5 | 4 |  |
| falconstatue | 0.004 | 11,091 | 5 | 4 |  |
| falconstatue_boundary | 0.02 | 411 | 5 | 4 |  |
| falconstatue_boundary | 0.008 | 2,635 | 5 | 4 |  |
| falconstatue_boundary | 0.004 | 10,790 | 5 | 4 |  |
| fish | 0.02 | 704 | 5 | 4 |  |
| fish | 0.008 | 4,568 | 5 | 4 |  |
| fish | 0.004 | 18,368 | 5 | 4 |  |
| fish_control_mesh | 0.02 | 859 | 5 | 4 |  |
| fish_control_mesh | 0.008 | 5,637 | 5 | 5 |  |
| fish_control_mesh | 0.004 | 23,399 | 5 | 5 |  |
| fish_low_resolution | 0.02 | 649 | 5 | 4 |  |
| fish_low_resolution | 0.008 | 4,023 | 5 | 4 |  |
| fish_low_resolution | 0.004 | 16,337 | 5 | 5 |  |
| goathead | 0.02 | 30,737 | 5 | 5 |  |
| goathead | 0.008 | 192,936 | 5 | 5 |  |
| goathead | 0.004 | 771,405 | 5 | 5 |  |
| hairball | 0.02 | 89,838 | 6 | 5 |  |
| hairball | 0.008 | 880,394 | 6 | 5 |  |
| hairball | 0.004 | 3,866,231 | 6 | 5 |  |
| hammer | 0.02 | 1,808 | 4 | 4 |  |
| hammer | 0.008 | 11,381 | 5 | 4 |  |
| hammer | 0.004 | 44,923 | 5 | 4 |  |
| hand | 0.02 | 78 | 4 | 4 |  |
| hand | 0.008 | 406 | 5 | 4 |  |
| hand | 0.004 | 1,599 | 5 | 4 |  |
| hand_closed | 0.02 | 78 | 4 | 4 |  |
| hand_closed | 0.008 | 410 | 4 | 4 |  |
| hand_closed | 0.004 | 1,633 | 5 | 4 |  |
| hand_lowres | 0.02 | 79 | 4 | 4 |  |
| hand_lowres | 0.008 | 412 | 4 | 4 |  |
| hand_lowres | 0.004 | 1,661 | 5 | 4 |  |
| house | 0.02 | — | — | — | did not run, see below |
| house | 0.008 | — | — | — | did not run, see below |
| house | 0.004 | — | — | — | did not run, see below |
| house_boundary | 0.02 | — | — | — | did not run, see below |
| house_boundary | 0.008 | — | — | — | did not run, see below |
| house_boundary | 0.004 | — | — | — | did not run, see below |
| human_man | 0.02 | 191 | 4 | 4 |  |
| human_man | 0.008 | 989 | 5 | 4 |  |
| human_man | 0.004 | 3,877 | 5 | 4 |  |
| human_neutral | 0.02 | 172 | 4 | 4 |  |
| human_neutral | 0.008 | 899 | 5 | 4 |  |
| human_neutral | 0.004 | 3,631 | 5 | 4 |  |
| human_woman | 0.02 | 164 | 4 | 4 |  |
| human_woman | 0.008 | 848 | 5 | 4 |  |
| human_woman | 0.004 | 3,401 | 5 | 4 |  |
| koala | 0.02 | 9,100 | 5 | 4 |  |
| koala | 0.008 | 56,842 | 5 | 5 |  |
| koala | 0.004 | 227,352 | 5 | 5 |  |
| koala_low_resolution | 0.02 | 8,881 | 5 | 4 |  |
| koala_low_resolution | 0.008 | 55,880 | 5 | 5 |  |
| koala_low_resolution | 0.004 | 223,453 | 5 | 5 |  |
| lionstatue | 0.02 | 680 | 5 | 4 |  |
| lionstatue | 0.008 | 3,971 | 5 | 4 |  |
| lionstatue | 0.004 | 16,036 | 5 | 5 |  |
| mountain | 0.02 | 123,940 | 5 | 5 |  |
| mountain | 0.008 | 768,838 | 5 | 5 |  |
| mountain | 0.004 | 3,068,538 | 5 | 5 |  |
| mushroom | 0.02 | 2,385 | 5 | 4 |  |
| mushroom | 0.008 | 15,124 | 5 | 5 |  |
| mushroom | 0.004 | 60,698 | 5 | 5 |  |
| nefertiti-lowres | 0.02 | — | — | — | did not run, see below |
| nefertiti-lowres | 0.008 | — | — | — | did not run, see below |
| nefertiti-lowres | 0.004 | — | — | — | did not run, see below |
| nefertiti | 0.02 | — | — | — | did not run, see below |
| nefertiti | 0.008 | — | — | — | did not run, see below |
| nefertiti | 0.004 | — | — | — | did not run, see below |
| parsnip | 0.02 | 1,445 | 5 | 4 |  |
| parsnip | 0.008 | 9,209 | 5 | 4 |  |
| parsnip | 0.004 | 37,087 | 5 | 5 |  |
| penguin | 0.02 | 930 | 5 | 4 |  |
| penguin | 0.008 | 6,165 | 5 | 4 |  |
| penguin | 0.004 | 25,661 | 5 | 5 |  |
| penguin_control_mesh | 0.02 | 1,099 | 5 | 4 |  |
| penguin_control_mesh | 0.008 | 7,637 | 5 | 4 |  |
| penguin_control_mesh | 0.004 | 32,083 | 5 | 5 |  |
| penguin_hr | 0.02 | 927 | 5 | 4 |  |
| penguin_hr | 0.008 | 6,128 | 5 | 5 |  |
| penguin_hr | 0.004 | 25,418 | 5 | 5 |  |
| pizza | 0.02 | 132 | 4 | 4 |  |
| pizza | 0.008 | 749 | 5 | 4 |  |
| pizza | 0.004 | 2,985 | 5 | 4 |  |
| plane | 0.02 | 9,653 | 5 | 4 |  |
| plane | 0.008 | 61,778 | 5 | 5 |  |
| plane | 0.004 | 248,890 | 5 | 5 |  |
| plane_holes | 0.02 | 9,502 | 5 | 4 |  |
| plane_holes | 0.008 | 59,547 | 5 | 5 |  |
| plane_holes | 0.004 | 237,104 | 5 | 5 |  |
| scorpion | 0.02 | 6,236 | 5 | 5 |  |
| scorpion | 0.008 | 40,166 | 5 | 5 |  |
| scorpion | 0.004 | 161,596 | 5 | 5 |  |
| scorpion_low_resolution | 0.02 | 6,289 | 5 | 5 |  |
| scorpion_low_resolution | 0.008 | 40,445 | 5 | 5 |  |
| scorpion_low_resolution | 0.004 | 162,460 | 5 | 5 |  |
| skull | 0.02 | 40,486 | 5 | 5 |  |
| skull | 0.008 | 254,923 | 5 | 5 |  |
| skull | 0.004 | 1,021,048 | 5 | 5 |  |
| skull_low_resolution | 0.02 | 40,127 | 5 | 5 |  |
| skull_low_resolution | 0.008 | 252,742 | 5 | 5 |  |
| skull_low_resolution | 0.004 | 1,012,874 | 5 | 5 |  |
| sphere | 0.02 | 1,038 | 5 | 5 |  |
| sphere | 0.008 | 6,307 | 5 | 4 |  |
| sphere | 0.004 | 25,311 | 5 | 5 |  |
| spot | 0.02 | 446 | 5 | 4 |  |
| spot | 0.008 | 2,811 | 5 | 4 |  |
| spot | 0.004 | 11,155 | 5 | 4 |  |
| spot_control_mesh | 0.02 | 582 | 5 | 4 |  |
| spot_control_mesh | 0.008 | 3,916 | 5 | 4 |  |
| spot_control_mesh | 0.004 | 15,499 | 5 | 5 |  |
| spot_low_resolution | 0.02 | 438 | 5 | 4 |  |
| spot_low_resolution | 0.008 | 2,770 | 5 | 4 |  |
| spot_low_resolution | 0.004 | 11,023 | 5 | 5 |  |
| springer | 0.02 | 2,591 | 5 | 4 |  |
| springer | 0.008 | 16,543 | 5 | 4 |  |
| springer | 0.004 | 66,240 | 5 | 5 |  |
| strawberry | 0.02 | 979 | 5 | 4 |  |
| strawberry | 0.008 | 6,509 | 5 | 5 |  |
| strawberry | 0.004 | 26,828 | 5 | 5 |  |
| stuffedtoy | 0.02 | 103 | 5 | 4 |  |
| stuffedtoy | 0.008 | 564 | 5 | 4 |  |
| stuffedtoy | 0.004 | 2,320 | 5 | 4 |  |
| sword-quad-dominant | 0.02 | 35 | 3 | 3 |  |
| sword-quad-dominant | 0.008 | 88 | 4 | 3 |  |
| sword-quad-dominant | 0.004 | 204 | 4 | 3 |  |
| sword | 0.02 | 35 | 3 | 3 |  |
| sword | 0.008 | 88 | 4 | 3 |  |
| sword | 0.004 | 204 | 4 | 3 |  |
| torus | 0.02 | 800 | 4 | 4 |  |
| torus | 0.008 | 4,971 | 5 | 5 |  |
| torus | 0.004 | 20,025 | 5 | 4 |  |
| tower | 0.02 | 5,713 | 5 | 4 |  |
| tower | 0.008 | 37,228 | 5 | 5 |  |
| tower | 0.004 | 152,469 | 5 | 5 |  |
| tower_holes | 0.02 | 4,740 | 5 | 4 |  |
| tower_holes | 0.008 | 29,371 | 5 | 5 |  |
| tower_holes | 0.004 | 118,862 | 5 | 5 |  |
| tree | 0.02 | 457 | 5 | 4 |  |
| tree | 0.008 | 2,353 | 5 | 4 |  |
| tree | 0.004 | 9,301 | 5 | 5 |  |
| tree_closed | 0.02 | 459 | 5 | 4 |  |
| tree_closed | 0.008 | 2,410 | 5 | 4 |  |
| tree_closed | 0.004 | 9,633 | 5 | 5 |  |
| violin | 0.02 | 28 | 4 | 3 |  |
| violin | 0.008 | 90 | 4 | 4 |  |
| violin | 0.004 | 295 | 5 | 4 |  |
| well | 0.02 | 1,143,998 | 5 | 5 |  |
| well | 0.008 | — | — | — | did not run, see below |
| well | 0.004 | — | — | — | did not run, see below |
| well_boundary | 0.02 | 1,039,667 | 5 | 5 |  |
| well_boundary | 0.008 | 6,060,983 | 5 | 5 | 7/11 placements |
| well_boundary | 0.004 | — | — | — | did not run, see below |
| wingnut | 0.02 | 5,099 | 5 | 4 |  |
| wingnut | 0.008 | 31,746 | 5 | 4 |  |
| wingnut | 0.004 | 125,910 | 5 | 4 |  |

### How the worst-per-configuration is distributed

| worst round count | number of configurations |
|---:|---:|
| 3 | 2 |
| 4 | 26 |
| 5 | 158 |
| 6 | 3 |

The 3 configurations that reached 6 are: hairball @ 0.02, hairball @ 0.008, hairball @ 0.004. Thin strands produce long, irregular components inside a single leaf, which is the shape that costs the most rounds.

### The configurations that did not run

15 of the 204 did not produce a measurement, covering 6 meshes: house, house_boundary, nefertiti, nefertiti-lowres, well, well_boundary. **This is not a convergence failure** -- the process runs out of GPU memory before labeling begins.

The cause is that voxel size here is absolute while the meshes are not normalised, so the same setting means wildly different resolutions:

| mesh | bounding box | width at voxel size 0.02 |
|---|---|---:|
| bunny | 1.6 x 1.5 x 1.2 | 78 voxels |
| house | 92.8 x 122.1 x 135.8 | 4,638 voxels |
| house_boundary | 92.8 x 122.1 x 135.8 | 4,638 voxels |
| nefertiti | 238.7 x 494.6 x 362.6 | 11,936 voxels |
| nefertiti-lowres | 238.7 x 494.7 x 362.6 | 11,937 voxels |
| well | 46.9 x 70.0 x 45.3 | 2,347 voxels |
| well_boundary | 46.9 x 70.0 x 45.3 | 2,347 voxels |

Every mesh that failed sits in a coordinate system tens to hundreds of times larger than the unit-scale ones, so even the coarsest setting in this sweep asks for a grid thousands of voxels across. They fail at all three voxel sizes, not just the finest.

## Synthetic leaves

Rasterized narrow bands are blobby: their leaves have shallow union-find trees and few local minima, so they stress neither term of the informal `log2(tree depth) + log2(number of local minima)` estimate the cap was originally justified with. These patterns stress each term on purpose.

- **serpentine** -- the longest induced path that fits in a leaf: 16 lines two apart, joined end to end, 143 voxels, so the graph diameter is 142. Maximises tree depth; has exactly one local minimum.
- **zigzag** -- a path whose *labels* sawtooth, by alternating a +z step (label +1) with a -y step (label -8). Maximises local minima.
- **checkerboard** -- 256 voxels, no two touching: 256 components, tree depth zero.
- **comb**, **nested shells**, **solid leaf** -- intermediate shapes.

Each is run through all 48 symmetries of the cube, for the same reason the meshes are re-placed. `as-built` is the orientation the generator produces; the range beside it is over all 48.

```
Shiloach-Vishkin rounds to convergence, per leaf.   Safety cap = 64
A leaf reaching the cap would be under-labeled, i.e. a wrong result.

Hand-built single-leaf patterns, over all 48 cube symmetries
  serpentine (induced path)     143 voxels    1 comps    rounds: as-built  5, over symmetries 5- 6
  zigzag (label sawtooth)        60 voxels    4 comps    rounds: as-built  4, over symmetries 2- 4
  checkerboard (isolated)       256 voxels  256 comps    rounds: as-built  1, over symmetries 1- 1
  comb (spine + teeth)           36 voxels    1 comps    rounds: as-built  2, over symmetries 2- 3
  nested shells                 352 voxels    2 comps    rounds: as-built  3, over symmetries 3- 3
  solid leaf                    512 voxels    1 comps    rounds: as-built  3, over symmetries 3- 3

Random single-leaf fills, 20000 trials per density
  density 0.05    max = 3    mean = 1.12
  density 0.15    max = 4    mean = 2.06
  density 0.30    max = 6    mean = 3.47
  density 0.50    max = 6    mean = 4.15
  density 0.70    max = 5    mean = 3.43
  density 0.85    max = 4    mean = 3.06
  density 0.95    max = 4    mean = 3.00

Random self-avoiding walks, 20000 trials
  max = 5    mean = 3.37    longest walk = 367 voxels

Worst observed: 6 rounds, from serpentine (induced path) (worst symmetry).   Cap is 64.
```

## What this does and does not establish

It establishes coverage: 6 rounds is the worst over 189 mesh configurations, the adversarial patterns under every symmetry, and tens of thousands of random leaves. The cap is an order of magnitude above that.

It is not a proof. The one thing it does settle is that the graph diameter is the wrong quantity to bound by: the serpentine has diameter 142 and converges in 6, so a proof of "fewer rounds than the diameter" would be true but about 24x loose -- and 142 is itself above the cap, so proving it would argue against 64 rather than for it.

Reproduce with `ex_mesh_to_sdf_cuda --sv-convergence [mesh.obj] [--voxel-size S] [--trials N]`. The random sweeps use a fixed seed.
