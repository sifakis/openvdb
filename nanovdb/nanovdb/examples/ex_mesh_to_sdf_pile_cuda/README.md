# A pile of meshes in letter-shaped bins: one MeshToSDF conversion of 104 million triangles

Eight bins shaped like the letters of SIGGRAPH are filled with 1,589 triangle meshes that overlap one another, the
floors and the walls. The whole scene, 104,428,862 triangles, is one triangle soup, and a single call of
`tools::cuda::MeshToSDF` turns it into one narrow-band signed distance field with 1.4 billion active voxels in 6.4 s
on an RTX 5090. The example is a stress test of the conversion: many triangles, many meshes whose surfaces cross,
and a sign that has to be right in every cavity between them.

## What it exercises

| Tool | Use here |
|---|---|
| `tools/cuda/MeshToSDF.cuh` | One conversion of the whole scene: band of 3 voxels on either side, `BarrierSigning::Ball`, `NestingRule::Solid`. |
| `tools/cuda/SignCertifier.cuh` | The sign next to the surface comes from the ball certification (`BarrierSigning::Ball`). |
| `tools/cuda/ConnectedComponents.cuh`, `PartitionGrid.cuh` | The band has 91 connected components at 2048 voxels per unit; each is signed on its own, then `NestingRule::Solid` makes everything enclosed by a component inside. |
| `tools/cuda/PropagateSign.cuh` | The planes written for the pictures hold the sign of every voxel, also of those outside the band: of inactive voxels of leaves and of the slots of lower and upper nodes. |
| Overlap | Meshes that overlap in one soup give the union of the meshes: every voxel gets the distance to the nearest triangle, and the hidden parts of the surfaces stay in the band as negative values. |

## The scene

Units: the cap height of the letters is 1. Axes: x along the word, y up the letters, z out of the ground.
`make_scene.py` makes every letter a tray lying on the ground, a floor under the letter and a wall 0.24 high along
its outline, both 0.025 thick. The outline comes from the 2D distance field of the letter rendered with the font
DejaVu Sans Bold, on a grid of 384 cells per unit; `bin/mesh_band` (OpenVDB's `volumeToMesh`) turns the tray's level
set into triangles. The small meshes are scaled to a longest extent of 0.12 times a random factor between 0.8 and
1.15, rotated at random, and put on a jittered lattice inside the letter, in four layers. One mesh of the top layer,
Nefertiti in the letter R, stands upright and faces the front: the "hero" the movie starts from.

| | |
|---|---|
| Triangles | 104,428,862 (52,214,943 points): 94,854,194 of the 1,589 meshes, 9,574,668 of the 8 bins |
| Extents | 7.643 x 1.086 x 0.375 units; at 2048 voxels per unit 15,653 x 2,224 x 768 voxels |
| `scene/soup.bin` | 1.9 GB, SHA-256 `00b2f43ea9aad4a45d7f3575993fd410cf079f3f2663a93300d559d7f553461d` |

## Meshes

The 24 meshes are files of https://github.com/odedstein/meshes (commit `8352132`); `get_meshes.sh` downloads them
(222 MB) and checks their SHA-256 sums. They are not part of this repository. Each has its own author and license,
stated in `objects/<object>/README.md` there; the table gives them as read on 2026-11-23. All 24 are closed. A mesh
with many triangles is picked rarely (`DEFAULT_SET` in `make_scene.py` holds the weights).

| File (`objects/<object>/`) | Triangles | Uses | Author, as that repository names it | License |
|---|---|---|---|---|
| `nefertiti/nefertiti.obj` | 2,018,232 | 3 | Nora Al-Badri and Jan Nikolai Nelles | public domain |
| `cat/cat.obj` | 1,698,248 | 4 | billyd, via Thingiverse | CC BY 4.0 |
| `cheese/cheese.obj` | 474,234 | 23 | 3dvs_ch, via Sketchfab | CC BY 4.0 |
| `tree/tree_closed.obj` | 459,968 | 22 | warmarine759, via Thingiverse | CC BY 4.0 |
| `violin/violin.obj` | 255,952 | 21 | Virtual Museums of Małopolska, via Sketchfab | CC0 1.0 |
| `koala/koala.obj` | 99,994 | 61 | YahooJAPAN, via Thingiverse | CC BY 3.0 |
| `skull/skull.obj` | 100,002 | 70 | YahooJAPAN, via Thingiverse | CC BY 3.0 |
| `scorpion/scorpion.obj` | 99,990 | 67 | YahooJAPAN, via Thingiverse | CC BY 3.0 |
| `hammer/hammer.obj` | 60,160 | 92 | Andrea Spognetta (Spogna), via Sketchfab | CC BY 4.0 |
| `bunny/bunny_hr.obj` | 59,664 | 94 | The Stanford 3D Scanning Repository | Stanford repository terms: research use, with acknowledgement |
| `sword/sword.obj` | 44,582 | 53 | The Oakeshott Institute, via Sketchfab | CC BY 4.0 |
| `boot/boot.obj` | 39,906 | 87 | inciprocal, via Sketchfab | CC BY 4.0 |
| `boat/boat.obj` | 32,416 | 88 | Oded Stein, Eitan Grinspun, Keenan Crane | CC BY 4.0 |
| `armadillo/armadillo.obj` | 29,170 | 128 | The Stanford 3D Scanning Repository | Stanford repository terms: research use, with acknowledgement |
| `penguin/penguin_hr.obj` | 26,880 | 91 | Oded Stein | CC BY 4.0 |
| `spot/spot.obj` | 23,062 | 87 | Keenan Crane | public domain |
| `fish/fish.obj` | 21,568 | 112 | Keenan Crane | public domain |
| `plane/plane.obj` | 17,456 | 71 | YahooJAPAN, via Thingiverse | CC BY 3.0 |
| `hand/hand_closed.obj` | 15,982 | 64 | Max Planck Society e.V., via SMPL | CC BY 4.0 |
| `human/human_man.obj` | 13,776 | 78 | Max Planck Society e.V., via SMPL | CC BY 4.0 |
| `human/human_woman.obj` | 13,776 | 65 | Max Planck Society e.V., via SMPL | CC BY 4.0 |
| `wingnut/wingnut.obj` | 6,090 | 66 | Oded Stein, Eitan Grinspun, Keenan Crane | CC BY 4.0 |
| `goathead/goathead.obj` | 5,522 | 93 | hugoelec, via Thingiverse | CC BY 3.0 |
| `torus/torus.obj` | 4,608 | 49 | Oded Stein | CC BY 4.0 |

## Steps

```
CCCL=<a checkout of github.com/NVIDIA/cccl> OPENVDB_ROOT=<an OpenVDB installation> ./build.sh
./get_meshes.sh meshes
export SCENE_PYTHON=<python with numpy, scipy, Pillow>
./run.sh meshes                 # the paper's run: needs a 32 GB GPU -> output/pile/
./run.sh meshes 1024 R small    # one letter at 1024 voxels per unit: fits a 12 GB GPU -> output/small/
```

`run.sh <mesh directory> [voxels per unit=2048] [word=SIGGRAPH] [name=pile]` makes the scene (`make_scene.py`, 22 s
on a 16-thread laptop CPU) and converts it (`bin/mesh_to_sdf_pile`). The scene can be made on one machine and
converted on another: `run.sh` keeps an `output/<name>/scene/` that is already there, and only `make_scene.py`
needs `bin/mesh_band`, hence OpenVDB.

`CCCL` is only needed if the CUDA toolkit's own Thrust lacks the `<cuda/std/...>` headers (CUDA 12.0 does). The
example is not registered in `examples/CMakeLists.txt`: `build.sh` compiles it directly against this repository's
headers; its header lists the flags for other GPUs and compilers.

The movie of the paper is made from `output/pile/` by `figure factory/MeshToSDFPile/make_animation.py` of the
paper's repository.

## Files

| File | Content |
|---|---|
| `make_scene.py` | The scene: bins, meshes, the soup. Its defaults are the paper's scene; `--word`, `--layers`, `--size`, `--meshes` and others change it. |
| `mesh_to_sdf_pile.cu` | One `MeshToSDF` conversion of a soup; prints the stage times, the counts and the peak of the CUDA memory pool; writes the surface voxels and planes of the sign. Its header gives the arguments and the file formats. |
| `run.sh` | Both steps with the paper's settings. |
| `get_meshes.sh` | Downloads and checks the meshes. |
| `build.sh` | Builds `bin/mesh_to_sdf_pile` and `bin/mesh_band`. |

`output/<name>/` after `run.sh`:

| File | Content |
|---|---|
| `scene/soup.bin`, `owners.i32`, `scene.json` | The triangle soup, the mesh every point belongs to, counts and the hero. Also `bins.meshbin` and `meshes.meshbin`, the two parts as meshes. |
| `sdf.faces` | Every interior voxel of the band that has an exterior face neighbour, 7 bytes each: the surface as the voxels see it. 1.4 GB for the paper's run. |
| `sdf_slice_0.bin` .. `sdf_slice_71.bin` | The signed distance on 72 horizontal planes from height 0.38 down to the ground, every 4th voxel at 2048 voxels per unit: the value of an active voxel, elsewhere plus or minus the band width by the sign `PropagateSign` gives. Also, per sample, where the tree stores that sign. |
| `sdf_slice_72.bin` | The plane at height 0.2462, every voxel. |
| `run.log` | The GPU, the stage times, the `RESULT` line and the index bounding box of the active voxels. |

## Numbers

One run on an NVIDIA GeForce RTX 5090 (32 GB), CUDA toolkit 12.9, built with `-arch=sm_120`; band of 3 voxels,
`BarrierSigning::Ball`, 2048 voxels per unit, `SHIFT_VOXELS="-7905 0 0"`:

| | |
|---|---|
| Active voxels | 1,416,920,079, in an index bounding box of 15,659 x 2,230 x 774 voxels (27.0 billion) |
| Connected components of the band | 91 |
| Interior voxels of the band | 723,890,059; 202,914,956 of them have an exterior face neighbour (286,485,298 faces) |
| Rasterizing the mesh | 5028.6 ms |
| Partitioning into components | 194.4 ms |
| Signing the components | 1172.4 ms |
| Nesting rule | 10.8 ms |
| `MeshToSDF`, whole call | 6423.9 ms, including the first-call start-up of the device code |
| Peak of the CUDA memory pool | 30,524.6 MiB, of which the input mesh 1,792.6 MiB |

The same scene at other resolutions, `BarrierSigning::Heuristic`, nothing written (`bin/mesh_to_sdf_pile
scene/soup.bin <voxels per unit> - 3 heuristic`), same GPU:

| Voxels per unit | Active voxels | Components | `MeshToSDF` | Peak of the pool |
|---|---|---|---|---|
| 1024 | 347,165,527 | 45 | 4602.1 ms | 29,022.0 MiB |
| 1536 | 794,552,848 | 71 | 5476.6 ms | 30,041.4 MiB |
| 2048 | 1,416,916,188 | 91 | 6378.6 ms | 30,522.4 MiB |
| 2176, 2304 | out of memory while rasterizing | | | |

## Limits

- **The triangles set the memory.** The peak is almost the same at 1024, 1536 and 2048 voxels per unit, and it is
  reached while the mesh is rasterized, which both signing modes share. A scene with fewer triangles reaches more
  voxels on the same GPU.
- **`BarrierSigning::Ball` far from the origin.** Without `SHIFT_VOXELS` the conversion at 2048 voxels per unit gave
  wrong signs in the letters whose voxel index along x exceeds about 8192: there float32 coordinates in voxels are
  good to about 1/1000 voxel, which is the tolerance of the ball certification. `SHIFT_VOXELS` moves the soup by whole
  voxels so that it is centered along x, and the files are written in the soup's own frame. A scene wider than
  about 16,000 voxels along an axis cannot be centered that way.
- **`sdf.faces` stores voxel coordinates as int16**, so a scene must stay within 32,767 voxels of its origin.
- `MeshToSDF` takes the numbers of points and triangles as 32-bit integers.

## Checks

- `make_scene.py` of this folder, run with the meshes `get_meshes.sh` downloaded, wrote the same `soup.bin`,
  `owners.i32`, `bins.meshbin` and `meshes.meshbin`, byte for byte, as the development copy the paper's run was made
  from (Python 3.12.3, numpy 2.5.3, scipy 1.18.1, Pillow 12.3.0 with FreeType 2.14.3, DejaVu Sans Bold 2.37).
  Another version of the font or of FreeType may move the letters' outlines and change the counts; the SHA-256 sum
  above tells.
- The planes of the run at 2048 voxels per unit were looked at as pictures of the sign: in every letter the bins and
  the meshes are inside and the space between them is outside.
- Converting only the whole meshes near the hero again, on the same voxel grid, gave the same plane as the whole
  scene there: of 1,085,188 samples around the hero at height 0.2461, none differs in sign, in value or in where the
  tree stores the sign.
