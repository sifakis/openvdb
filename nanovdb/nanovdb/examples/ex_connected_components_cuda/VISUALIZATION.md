# Visualizing the mesh → SDF pipeline

Two steps: **(1)** run the CUDA example to dump the result to a file, **(2)** open that file in
Polyscope with `mesh_to_sdf_viewer.py`. The dump is written only when the `CC_EXPORT_VIS` env var is
set, so normal runs / self-tests are unaffected.

```
buildMeshToSdf ──(CC_EXPORT_VIS=path)──▶ exportMeshToSdf ──▶ path.ccvis (+ .fill, + .ovdb)
                                                                    │
                                                      mesh_to_sdf_viewer.py ──▶ Polyscope window
```

---

## Prerequisites (Python viewer)

The viewer needs `polyscope` + `numpy` and a display (it opens an OpenGL window; over SSH you need X
forwarding). A ready virtualenv already exists at `~/Desktop/work/.venv`; otherwise:

```bash
python3 -m venv .venv && .venv/bin/pip install polyscope numpy
```

Polyscope note: `#` is not allowed in structure names (the viewer already avoids it).

---

## Step 1 — export a dump

Run the example with `CC_EXPORT_VIS=<path>`:

```bash
cd ~/Desktop/Code/NVIDIA/openvdb/build/release
CC_EXPORT_VIS=/tmp/bunny.ccvis \
  nanovdb/nanovdb/examples/ex_connected_components_cuda ~/Desktop/meshes/bunny.obj 0.008
```

Arguments: `<input.obj> [voxelSize] [bandWidth]` (also works with the synthetic `--sphere` /
`--big-sphere` / `--two-spheres` self-tests). This writes up to three files:

| file | contents | written when |
|------|----------|--------------|
| `<path>.ccvis`  | one record per active voxel: `[ijk, cc, sign, udf]` | always |
| `<path>.fill`   | step-6 interior-fill boxes: `[baseVoxel, level]` (leaf 1³ / lower 8³ / upper 128³ / root 4096³) | always |
| `<path>.ovdb`   | OpenVDB `meshToLevelSet` sign+value at each voxel (for comparison) | built with `NANOVDB_USE_OPENVDB=ON` |

**Resolution ↔ voxel count (the #1 performance lever).** Band voxel count grows as ~1/voxelSize²:
bunny is ~84k voxels at `0.02`, ~8.5M at `0.002` (100×). For an interactive viewer, keep it in the
low hundred-thousands — start around `0.008`–`0.01` and go finer only if needed. The console prints
`N voxels, voxelSize=…` so you can check.

---

## Step 2 — view

```bash
~/Desktop/work/.venv/bin/python \
  ~/Desktop/Code/NVIDIA/openvdb/nanovdb/nanovdb/examples/ex_connected_components_cuda/mesh_to_sdf_viewer.py \
  /tmp/bunny.ccvis
```

(Or `source ~/Desktop/work/.venv/bin/activate` once, then `python mesh_to_sdf_viewer.py <dump>`.)

### Structures (each is an on/off toggle in the left panel)

| structure | what | default |
|-----------|------|---------|
| `band voxels`            | non-barrier active voxels (the signed narrow band) | **on** |
| `barrier voxels`         | the surface shell (`cc = -1`) | **on** |
| `interior voxels (…)`    | step-6 deep-interior fill, one box per tree level | **off** |
| `cc NN (…)`              | one connected component at a time (only with `--split-cc`) | off |

### Color modes (switch per structure via its quantity in the UI)

- **`cc`** — connected-component id (categorical; each component a different color). *Default on `band`.*
- **`sign`** — inside / outside (interior = warm, exterior = cool). Use `--sign` to enable it on
  everything at once.
- **`udf`** — unsigned distance (sequential colormap).

---

## OpenVDB comparison (needs the `.ovdb` companion)

Built with `NANOVDB_USE_OPENVDB=ON`, the export also writes `<path>.ovdb` (OpenVDB's sign/value per
voxel). The viewer then offers:

- **Side-by-side (`--side-by-side`, default on when `.ovdb` is present):** a second copy of the
  band+barrier voxels, shifted `+X`, colored by **OpenVDB's** sign — a true 1:1 *ours* vs *openvdb*
  layout (`ours: …` / `openvdb: …`). `--no-side-by-side` forces a single overlaid panel.
- **`--mismatch`:** color band voxels by disagreement — **red** where our sign differs from OpenVDB
  *beyond* the √3/2 shell (a real disagreement), soft yellow *within* the shell (expected/method-
  dependent), neutral where they agree.
- **`--no-ovdb`:** ignore the companion even if present.

---

## Options

| option | effect |
|--------|--------|
| `--sign`             | start every structure in the interior/exterior sign coloring |
| `--mismatch`         | start in the OpenVDB `ovdb mismatch` coloring (needs `.ovdb`) |
| `--side-by-side` / `--no-side-by-side` | force the 1:1 OpenVDB panel on / off |
| `--alpha A`          | start all structures at transparency `A` (0..1); tune per-structure in the UI |
| `--no-barrier`       | hide the barrier shell |
| `--no-fill`          | skip the interior-fill boxes (much lighter — recommended for large dumps) |
| `--no-ovdb`          | ignore the OpenVDB comparison companion |
| `--split-cc`         | one structure per component (isolate individual CCs) |
| `--cc RANKS`         | show only these component ranks (e.g. `--cc 2,3,4`; rank 0 = largest) |

Transparency is always available: each structure has a **Transparency** slider under its UI options
(`Appearance → Transparency` sets the global mode).

---

## Interpreting what you see

- **`cc` view — extra components.** Ideal is 2 (outer shell + inner shell). In practice you see a few
  more: tiny **trapped pockets** — non-barrier voxels the barrier shell pinches off in thin/concave
  regions. They're all correctly signed (interior) and **harmless**; a finer voxelSize removes them.
  Use `--split-cc` to isolate and locate them.
- **`sign` view.** Interior/exterior should form two clean regions; with `--sign` the fill boxes and
  the band should read as one continuous interior color.
- **`interior voxels (…)` (step 6).** Turn them on to see the deep interior filled level-by-level
  (leaf 1³ → lower 8³ → upper 128³). Coarse levels only appear for thick solids (e.g. `--big-sphere`);
  `root 4096³` essentially never appears for real meshes.

---

## Performance tips

- **Coarsen first.** Re-export at a larger voxelSize (`0.008`–`0.01`) before viewing large meshes.
- **`--no-fill`.** The fill boxes can be millions of entries; skip them for sign/CC inspection.
- Side-by-side doubles the band+barrier geometry — combine with the two tips above for big meshes.

---

## File formats (little-endian)

```
<dump>.ccvis   "CCVIS001" | u64 N | f64 voxelSize | f64 tx,ty,tz | N × { i32 i,j,k,cc,sign; f32 udf }
<dump>.fill    "CCFILL01" | u64 M | f64 voxelSize | f64 tx,ty,tz | M × { i32 baseX,baseY,baseZ, level }
<dump>.ovdb    "CCOVDB01" | u64 N | f64 voxelSize |               N × { f32 value; i32 sign; i32 mismatch }
```

`.ovdb` records are positionally aligned 1:1 with the `.ccvis` records. World position of voxel `ijk`
= `(tx,ty,tz) + ijk · voxelSize`. See `exportMeshToSdf()` in `connected_components_cuda_kernels.cu`
and the header docstring of `mesh_to_sdf_viewer.py`.
