# CSG viewer and its data

Two programs of this example write, as JSON, what the CSG operations produce while one mesh level set moves
through another. `make_viewer.py` turns the two files into one interactive HTML page. The same files are the
intended input of static figures: `read_frames.py` reads them with the Python standard library, and this document
describes every field.

| file | what it is |
|---|---|
| `../csg_viewer_slices.cu` | Writes one 2D slice of the inputs and of all 12 results per step. |
| `../csg_viewer_voxels.cu` | Writes every stored voxel and mask bit of the inputs and of the results per step (3D). |
| `make_viewer.py`, `viewer_template.html` | Build the interactive page. |
| `read_frames.py` | Reference reader of both files, with a self check. |

## The scene

Input 1 stays in place. Input 2 is the second mesh, translated by a whole number of voxels along +z; the
translation shrinks linearly to 0 over the frames, so in the last frame both meshes are centered at the origin.
Both are narrow bands with a half-width of 3 voxels (the constant `W` in `../CsgExampleUtil.cuh`), read from band
files written by `mesh_to_band`. The voxel size is 1, so index coordinates and world coordinates coincide.

For every frame, three operations run for each of the four sign pairs of the backgrounds, which gives 12 results:

- operations: union (`csgUnion`), intersection (`csgIntersection`), difference 1 − 2 (`csgDifference`);
- an input is used as it is, or negated with `csgComplement` first.

The generators' scene parameters are constants in their `main()`:

| | `csg_viewer_slices` | `csg_viewer_voxels` |
|---|---|---|
| frames | 30 | 15 |
| translation of input 2 in frame k | `round(210 * (29 - k) / 29)` voxels | `round(105 * (14 - k) / 14)` voxels |
| meant for meshes scaled to | 220 voxels along the longest side | 110 voxels along the longest side |
| what is written | the slice x = 0, for z in [-120, 303] and y in [-120, 119] | all stored voxels and bits |

## Commands

```
# mesh inputs (any closed OBJ meshes; the same voxel size for both)
./ex_csg_nanovdb_cuda_mesh_to_band dragon.obj    220 dragon.band
./ex_csg_nanovdb_cuda_mesh_to_band armadillo.obj 220 armadillo.band
./ex_csg_nanovdb_cuda_mesh_to_band dragon.obj    110 dragon_110.band
./ex_csg_nanovdb_cuda_mesh_to_band armadillo.obj 110 armadillo_110.band

# data: fixed input, moving input, output
./ex_csg_nanovdb_cuda_csg_viewer_slices dragon.band     armadillo.band     slices.json
./ex_csg_nanovdb_cuda_csg_viewer_voxels dragon_110.band armadillo_110.band voxels.json

# check the files, then build the page
python3 viewer/read_frames.py --slices slices.json --voxels voxels.json
python3 viewer/make_viewer.py --slices slices.json --voxels voxels.json --names dragon armadillo --out csg_viewer.html
```

Both generators print one line per frame and end with `ALL FRAMES AGREE` if every check passed. The slice file is
about 2 MB and the voxel file about 12 MB for the dragon and the armadillo. The page embeds both and loads three.js
from a CDN for the 3D view.

## What a level set stores, and what the data shows

A level set is a `ValueOnIndex` grid, one value per active voxel, a background, and invert masks:

- An **active voxel** stores a signed distance with a magnitude below the band half-width.
- An **inactive voxel** takes the background or its negation. A mask bit that is **on** means "the sign here is
  the opposite of the background". Bits are stored at three levels: one bit per inactive voxel of a leaf, one bit
  per childless slot of a lower node (8 x 8 x 8 voxels), one bit per childless slot of an upper node (128^3
  voxels).
- With a **positive background** (a bounded solid) a bit that is on means inside, and everything else that is
  inactive is outside. With a **negative background** (the complement of a bounded solid) it is the other way
  round: everything inactive whose bit is off is inside, including all of space far away, and a bit that is on
  means outside.
- **Negating** a level set flips the sign of every value and of the background. The grid and the masks do not
  change.

The value of a result at a voxel follows the rule `min(f1, f2)` for union, `max(f1, f2)` for intersection and
`max(f1, -f2)` for difference, where `f` is the stored value of an active voxel and the background or its negation
elsewhere. Both generators check every result against this rule and record the number of mismatches in each
frame; it is 0 in a correct run.

## Slice file (`csg_viewer_slices`)

Top level:

| key | meaning |
|---|---|
| `band` | band half-width in voxels (also the magnitude of every background) |
| `w`, `h` | width and height of a slice in voxels |
| `x0` | z coordinate of the first column |
| `y0` | y coordinate of the bottom row |
| `flipY` | always `true`: row 0 is the top row |
| `in1` | `{leaves, active}` of input 1 (whole level set, not the slice) |
| `i1` | the slice of input 1, encoded (see below); it is the same in every frame |
| `frames` | list of frames |

The entry at row `j`, column `i` of a slice is the voxel `(x, y, z) = (0, y0 + h - 1 - j, x0 + i)`.

Each frame:

| key | meaning |
|---|---|
| `off` | translation of input 2 along +z, in voxels |
| `in2` | `{leaves, active}` of input 2 |
| `i2` | the slice of input 2, encoded, as it is (not negated) |
| `u00` ... `d11` | the 12 results: a letter for the operation (`u` union, `i` intersection, `d` difference), then `1` if input 1 was negated, then `1` if input 2 was negated |
| `checked` | number of points at which the 12 results were compared with the rule: every second voxel in each direction of a box around both inputs, and every voxel of the slice |
| `ruleBad` | number of those points where a result differed from the rule |

Each result:

| key | meaning |
|---|---|
| `bg` | background of the result; its sign says how to read the bits |
| `leaves`, `lowers`, `uppers` | node counts of the whole result |
| `active` | active voxels of the whole result |
| `bits` | number of bits on in the leaf, lower and upper invert masks of the whole result |
| `v` | the slice of the result, encoded |

**Encoding of a slice.** A string of runs in row-major order: one letter (`A` = code 0, `B` = code 1, ...) followed
by the run length in decimal. The code of a voxel is one of the values below, plus 8 where a leaf exists at the
voxel (so the leaf layout can be drawn):

| code | meaning |
|---|---|
| 0 | inactive, bit off |
| 1 | active, negative value |
| 2 | active, positive value |
| 3 | inactive, bit on in a leaf mask |
| 4 | inactive, in a childless lower-node slot whose bit is on |
| 5 | inactive, in a childless upper-node slot whose bit is on |
| 6 | inactive, in a root-level region whose bit is on (does not occur: not supported yet) |

The slices of the inputs are those of the inputs before negation. To draw a negated input, flip the sign of the
active codes (1 and 2 trade places) and read the bits with a negative background. `read_frames.classify` does this.

## Voxel file (`csg_viewer_voxels`)

Top level: `band` as above, `i1` (input 1, a level-set record) and `frames`.

Each frame:

| key | meaning |
|---|---|
| `off` | translation of input 2 along +z, in voxels |
| `i2` | input 2, as it is |
| `u`, `i`, `d`, `e` | four results computed from the inputs as they are: `1 ∪ 2`, `1 ∩ 2`, `1 − 2`, `2 − 1` |
| `checked`, `ruleBad` | as in the slice file; here every voxel of a box around both inputs is compared |
| `mapBad` | number of the 12 results that are not byte-identical to the stored result they map to (below) |

Only four results are stored because each of the 12 is one of them or its complement, and a complement has the
same grid and masks with negated values:

| input 1, input 2 | union | intersection | difference |
|---|---|---|---|
| as is, as is | `u` | `i` | `d` |
| as is, negated | `e`, complemented | `d` | `i` |
| negated, as is | `d`, complemented | `e` | `u`, complemented |
| negated, negated | `i`, complemented | `u`, complemented | `e` |

The generator does not assume this table: it computes all 12 results and compares each, byte for byte (grid,
values, the masks of the three levels), with the stored result of the table, negated where the table says so.

A level-set record:

| key | meaning |
|---|---|
| `leaves`, `lowers`, `uppers`, `active`, `bits` | as in a result of the slice file |
| `t` | base64 of the leaves: 198 bytes per leaf |
| `l` | base64 of the lower-node slots whose bit is on: 6 bytes per slot |
| `u` | base64 of the upper-node slots whose bit is on: 6 bytes per slot |

- A leaf is its origin (3 little-endian `int16`: x, y, z) followed by three masks of 64 bytes each: the value mask
  (active voxels), the mask of the active voxels with a negative value, and the invert mask. Voxel `n` of a leaf
  is at `(n >> 6, (n >> 3) & 7, n & 7)` from the origin, and is bit `n & 7` of byte `n >> 3` of a mask.
- A slot is its origin (3 little-endian `int16`). A lower-node slot covers 8 x 8 x 8 voxels from its origin, an
  upper-node slot 128 x 128 x 128.

A stored result has a positive background. For a complemented result, swap the negative and the positive active
voxels and read the bits with a negative background.

## Reading the data in Python

`read_frames.py` needs only the standard library:

```python
import json
import read_frames as rf

slices = json.load(open("slices.json"))
frame = slices["frames"][17]
record = rf.slice_result(frame, "difference", negate1=False, negate2=True)
rows = rf.decode_slice(record["v"], slices["w"], slices["h"])        # rows[j][i] is a code
state, level = rf.classify(rows[100][200], background_negative=record["bg"] < 0)
input1 = rf.decode_slice(slices["i1"], slices["w"], slices["h"])      # draw negated: rf.classify(code, negated=True)

voxels = json.load(open("voxels.json"))
level_set, complemented = rf.voxel_result(voxels["frames"][9], "union", negate1=True, negate2=False)
lists = rf.decode_level_set(level_set)   # negative, positive, leaf_bits, lower_bits, upper_bits, leaf_origins
```

`python3 read_frames.py --slices slices.json --voxels voxels.json` decodes everything and compares the decoded
counts of every level set of the voxel file with the counts recorded in it.

## Colors of the interactive page

The page uses these colors in its light theme; a figure does not have to follow them.

| meaning | color |
|---|---|
| active, negative value | `#c9781f` |
| active, positive value | `#f2c878` |
| inside, bit of a leaf / lower / upper mask (positive background) | `#c4b6f2` / `#7b5fd0` / `#3b2787` |
| outside, bit off (positive background) | `#eef0f4` |
| inside, bit off (negative background) | `#b9b3cf` |
| outside, bit of a leaf / lower / upper mask (negative background) | `#dde6e1` / `#b4c7bd` / `#86a193` |
| leaf boundary | `#1a1f2b`, blended one third into the voxel color |

In the 3D view, a lower-node or upper-node bit is drawn as slabs one voxel thick along x, so that the cut plane
`x < c` hides part of it. The unbounded inside of a level set with a negative background is not drawn.
