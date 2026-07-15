#!/usr/bin/env python3
"""View a mesh->SDF pipeline dump in Polyscope as a Sparse Volume Grid.

Companion to ex_connected_components_cuda. Produce the dump by running the CUDA
example with the CC_EXPORT_VIS env var, e.g.:

    CC_EXPORT_VIS=/tmp/dragon.ccvis ./nanovdb_example_ex_connected_components_cuda dragon.obj 0.001

then view it:

    pip install polyscope numpy
    python mesh_to_sdf_viewer.py /tmp/dragon.ccvis

Each active voxel becomes one occupied cell of a Polyscope Sparse Volume Grid, with three
togglable per-cell quantities:
  * cc rank - connected-component id, remapped to a dense 0..K-1 rank by size (categorical).
              Barrier voxels (absent from the derived CC grid) get rank -1.
  * sign    - final signed level-set sign, +1 outside / -1 inside (categorical, 2 colors).
  * udf     - world-space unsigned distance (sequential scalar).

Structures (each an on/off toggle in the Polyscope panel):
  band voxels             non-barrier active voxels; color by 'cc' / 'sign' / 'udf'
  barrier voxels          the surface shell (grey; 'sign' / 'udf' available)
  interior voxels (...)   step-6 deep interior, one box per tree level (leaf 1³ / lower 8³ / ...)

If a '<dump>.ovdb' companion is present (written when the example is built with OpenVDB), each band /
barrier voxel also gets an OpenVDB comparison:
  * openvdb sign   - OpenVDB meshToLevelSet's sign at that voxel (same color scheme as 'sign').
  * ovdb mismatch  - where our sign disagrees with OpenVDB: RED beyond the √3/2 shell (a real
                     disagreement), soft YELLOW within the shell (expected/method-dependent), neutral
                     grey where they agree.
  * openvdb value  - OpenVDB's raw signed level-set value (sequential scalar).

Options:
  --sign            start with the unified interior/exterior sign coloring on every structure
  --mismatch        start with the 'ovdb mismatch' coloring on (needs the '<dump>.ovdb' companion)
  --alpha A         start every structure at transparency A (0..1); tune per structure in the UI
  --no-barrier      hide barrier voxels (cc == -1)
  --no-fill         skip the step-6 interior-fill boxes (the '<dump>.fill' companion)
  --no-ovdb         ignore the OpenVDB comparison companion even if present
  --split-cc        register EACH connected component as its own structure (toggle individually)
  --cc RANKS        show only these component ranks (comma list, e.g. --cc 2,3,4; rank 0 = largest)

Binary layout (little-endian), matching exportMeshToSdf() in the .cu:
    char   magic[8] = "CCVIS001"
    uint64 N
    double voxelSize
    double tx, ty, tz          (world position of index origin (0,0,0))
    N x { int32 i,j,k,cc,sign;  float udf }

Companion '<dump>.ovdb' (written by validateMeshToSdf when built with OpenVDB), same order as above:
    char   magic[8] = "CCOVDB01"
    uint64 N                    (must equal the .ccvis N)
    double voxelSize
    N x { float value; int32 sign; int32 mismatch }   (mismatch: 0 agree, 1 beyond-shell, 2 in-shell)
"""
import argparse
import struct
import sys

import numpy as np


REC = np.dtype([("ijk", "<i4", 3), ("cc", "<i4"), ("sign", "<i4"), ("udf", "<f4")])
REC_FILL = np.dtype([("base", "<i4", 3), ("level", "<i4")])
# Companion "<dump>.ovdb": OpenVDB meshToLevelSet sampled at each active voxel, in the SAME order as
# the .ccvis records. value = OpenVDB level-set value (world units); sign = +/-1; mismatch class below.
REC_OVDB = np.dtype([("value", "<f4"), ("sign", "<i4"), ("mismatch", "<i4")])

# Step-6 fill levels: (level id, box size in voxels, label, RGB color, transparency).
FILL_LEVELS = [
    (0, 1,    "leaf 1³",     (0.20, 0.45, 1.00), 1.0),
    (1, 8,    "lower 8³",    (0.20, 0.80, 0.35), 0.85),
    (2, 128,  "upper 128³",  (1.00, 0.60, 0.10), 0.6),
    (3, 4096, "root 4096³",  (1.00, 0.20, 0.20), 0.45),
]

# Fixed sign -> RGB so the SAME colors mean inside/outside across every structure (active voxels and
# all fill levels). Interior = warm, exterior = cool.
INTERIOR_RGB = (1.00, 0.55, 0.10)   # sign -1
EXTERIOR_RGB = (0.15, 0.45, 1.00)   # sign +1


def sign_to_rgb(sign):
    rgb = np.empty((sign.shape[0], 3), dtype=np.float32)
    rgb[sign < 0] = INTERIOR_RGB
    rgb[sign >= 0] = EXTERIOR_RGB
    return rgb


# OpenVDB comparison (from the '<dump>.ovdb' companion): per-voxel disagreement class -> RGB. The
# beyond-shell mismatch is the one to spot — our sign genuinely differs from OpenVDB where OpenVDB is
# confident. In-shell disagreement is expected (method-dependent within the √3/2-voxel shell), so it
# gets a separate, calmer color.
MISMATCH_RGB = {
    0: (0.22, 0.22, 0.25),   # agree                                   — neutral dark grey
    1: (1.00, 0.10, 0.10),   # our sign != OpenVDB, BEYOND the shell   — real disagreement (red)
    2: (0.95, 0.85, 0.20),   # disagree WITHIN the √3/2 shell          — expected tie (soft yellow)
}


def mismatch_to_rgb(mm):
    rgb = np.empty((mm.shape[0], 3), dtype=np.float32)
    for k, c in MISMATCH_RGB.items():
        rgb[mm == k] = c
    return rgb


def load(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"CCVIS001":
            sys.exit(f"bad magic {magic!r} (not a CCVIS001 dump)")
        (n,) = struct.unpack("<Q", f.read(8))
        (vs,) = struct.unpack("<d", f.read(8))
        tx, ty, tz = struct.unpack("<3d", f.read(24))
        rec = np.fromfile(f, dtype=REC, count=n)
    if rec.shape[0] != n:
        sys.exit(f"truncated: header says {n} records, read {rec.shape[0]}")
    return rec, vs, np.array([tx, ty, tz])


def load_fill(path):
    """Load the companion "<dump>.fill" step-6 interior-fill boxes, or None if absent."""
    import os
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"CCFILL01":
            sys.exit(f"bad magic {magic!r} in {path} (not a CCFILL01 dump)")
        (m,) = struct.unpack("<Q", f.read(8))
        (vs,) = struct.unpack("<d", f.read(8))
        tx, ty, tz = struct.unpack("<3d", f.read(24))
        rec = np.fromfile(f, dtype=REC_FILL, count=m)
    if rec.shape[0] != m:
        sys.exit(f"truncated fill: header says {m}, read {rec.shape[0]}")
    return rec, vs, np.array([tx, ty, tz])


def load_ovdb(path, expect_n):
    """Load the companion "<dump>.ovdb" (OpenVDB meshToLevelSet sign comparison), or None if absent.
    One record per active voxel, positionally aligned with the .ccvis records. Returns the raw record
    array, or None (with a warning) if its count disagrees with the .ccvis dump."""
    import os
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"CCOVDB01":
            sys.exit(f"bad magic {magic!r} in {path} (not a CCOVDB01 dump)")
        (m,) = struct.unpack("<Q", f.read(8))
        (_vs,) = struct.unpack("<d", f.read(8))
        rec = np.fromfile(f, dtype=REC_OVDB, count=m)
    if rec.shape[0] != m:
        sys.exit(f"truncated ovdb: header says {m}, read {rec.shape[0]}")
    if m != expect_n:
        print(f"WARNING: {path} has {m} records but the .ccvis dump has {expect_n}; skipping the "
              f"OpenVDB comparison (regenerate both from the same run).")
        return None
    return rec


def rank_components(cc, sign):
    """Return (rank_of_voxel, table) where table is a list of (rank, cc_id, count, majority_sign)
    sorted by size descending. Barrier voxels (cc<0) keep rank -1."""
    real = cc[cc >= 0]
    ids, counts = np.unique(real, return_counts=True)
    order = np.argsort(-counts)  # largest first
    ids, counts = ids[order], counts[order]
    id_to_rank = {int(c): r for r, c in enumerate(ids)}
    rank = np.full(cc.shape[0], -1, dtype=np.int32)
    for r, c in enumerate(ids):
        rank[cc == c] = r
    table = []
    for r, (c, n) in enumerate(zip(ids, counts)):
        s = sign[cc == c]
        maj = 1 if int((s > 0).sum()) >= int((s < 0).sum()) else -1
        table.append((r, int(c), int(n), maj))
    return rank, table


def main():
    ap = argparse.ArgumentParser(description="Polyscope viewer for a CCVIS pipeline dump.")
    ap.add_argument("dump", help="path to the CC_EXPORT_VIS .ccvis file")
    ap.add_argument("--no-barrier", action="store_true", help="hide barrier voxels (cc == -1)")
    ap.add_argument("--split-cc", action="store_true",
                    help="register each component as its own structure (toggle individually)")
    ap.add_argument("--cc", default=None,
                    help="show only these component ranks (comma list, rank 0 = largest)")
    ap.add_argument("--no-fill", action="store_true",
                    help="skip the step-6 interior-fill boxes (the '<dump>.fill' companion)")
    ap.add_argument("--no-ovdb", action="store_true",
                    help="ignore the OpenVDB comparison companion ('<dump>.ovdb'), even if present")
    ap.add_argument("--side-by-side", dest="side_by_side", action="store_true", default=None,
                    help="register a SECOND copy of the band+barrier voxels, translated +X (by the X "
                         "extent + ~15%% gap) and colored by OpenVDB's sign — a true 1:1 ours-vs-openvdb "
                         "comparison (default ON when the '<dump>.ovdb' companion is present)")
    ap.add_argument("--no-side-by-side", dest="side_by_side", action="store_false",
                    help="force a single overlaid panel even when the '.ovdb' companion is present")
    ap.add_argument("--mismatch", action="store_true",
                    help="start with the OpenVDB 'ovdb mismatch' coloring enabled (red = our sign "
                         "disagrees with OpenVDB beyond the √3/2 shell; needs the '<dump>.ovdb' companion)")
    ap.add_argument("--sign", action="store_true",
                    help="start with the unified sign coloring (interior/exterior) enabled on EVERY "
                         "structure — active voxels and all fill levels — to verify signs across levels")
    ap.add_argument("--alpha", type=float, default=None,
                    help="start every structure at this transparency (0=invisible..1=opaque); you can "
                         "still fine-tune per structure with the UI Transparency slider")
    args = ap.parse_args()

    rec, vs, translation = load(args.dump)
    ijk = rec["ijk"].astype(np.int32)
    cc = rec["cc"].astype(np.int32)
    sign = rec["sign"].astype(np.int32)
    udf = rec["udf"].astype(np.float32)

    # Optional OpenVDB comparison companion — positionally aligned with `rec` (before any filtering).
    ovdb = None if args.no_ovdb else load_ovdb(args.dump + ".ovdb", rec.shape[0])
    ov_sign = ov_mm = ov_val = None
    if ovdb is not None:
        ov_sign = ovdb["sign"].astype(np.int32)
        ov_mm = ovdb["mismatch"].astype(np.int32)
        ov_val = ovdb["value"].astype(np.float32)
        beyond, inshell = int((ov_mm == 1).sum()), int((ov_mm == 2).sum())
        print(f"OpenVDB comparison ('.ovdb'): {beyond} beyond-shell sign mismatches (RED), "
              f"{inshell} in-shell ties (yellow), {rec.shape[0] - beyond - inshell} agree.")

    def _filter_ovdb(keep):
        nonlocal ov_sign, ov_mm, ov_val
        if ovdb is not None:
            ov_sign, ov_mm, ov_val = ov_sign[keep], ov_mm[keep], ov_val[keep]

    if args.no_barrier:
        keep = cc != -1
        ijk, cc, sign, udf = ijk[keep], cc[keep], sign[keep], udf[keep]
        _filter_ovdb(keep)

    rank, table = rank_components(cc, sign)

    # Report: definitive component count + per-component sizes.
    print(f"{ijk.shape[0]} voxels, voxelSize={vs:g}, "
          f"{len(table)} connected components, {int((cc < 0).sum())} barrier (cc=-1)")
    print("  rank   cc_id      voxels   sign")
    for r, c, n, s in table:
        print(f"  {r:>4}  {c:>7}  {n:>10}   {'+ (out)' if s > 0 else '- (in )'}")

    # Optional rank filter.
    if args.cc is not None:
        want = {int(x) for x in args.cc.split(",") if x.strip() != ""}
        keep = np.isin(rank, list(want))
        ijk, cc, sign, udf, rank = ijk[keep], cc[keep], sign[keep], udf[keep], rank[keep]
        _filter_ovdb(keep)
        print(f"--cc {sorted(want)}: showing {ijk.shape[0]} voxels")

    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_transparency_mode("pretty")   # enables the per-structure Transparency slider in the UI

    # Polyscope cell (i,j,k) lower corner at origin + (i,j,k)*cell_width; a NanoVDB voxel center sits
    # at index (i,j,k). Offset origin by half a cell so cell centers land on true voxel centers.
    origin = tuple(translation - 0.5 * vs)
    cw = (vs, vs, vs)

    # Side-by-side: a second copy of the band+barrier voxels, translated +X so it sits NEXT TO the
    # "ours" panel (not overlaid), colored by OpenVDB's sign. Default ON when the .ovdb companion is
    # present. The copy reuses the same integer cells, only the structure origin is shifted in world X.
    side_by_side = (ovdb is not None) and (args.side_by_side is not False)
    if args.side_by_side and ovdb is None:
        print("--side-by-side requested but no '.ovdb' companion loaded; showing a single panel.")
    prefix = "ours: " if side_by_side else ""
    origin_ovdb = origin
    if side_by_side:
        xspan = (int(ijk[:, 0].max()) - int(ijk[:, 0].min()) + 1) * vs
        origin_ovdb = (origin[0] + xspan * 1.15, origin[1], origin[2])  # X extent + ~15% gap

    # ---- Structure layout (each row below is one on/off toggle in the Polyscope panel) ----
    #   band voxels              non-barrier voxels (steps 3-4): quantities cc / sign / udf
    #   barrier voxels           the surface shell (step 5), signed +/-
    #   interior voxels (...)    step-6 deep-interior boxes, one per tree level, sized to the node
    #   cc #NN ...               (only with --split-cc) one component at a time, for isolation
    # Default coloring: cc on the band, a neutral shell, level colors on the interior fill. Pass --sign
    # to start every structure in the unified interior/exterior sign coloring instead.
    band = cc >= 0
    barrier = cc < 0

    # OpenVDB comparison quantities (only when the '.ovdb' companion loaded): 'openvdb sign' mirrors the
    # 'sign' coloring but from OpenVDB's level set, and 'ovdb mismatch' flags where the two disagree
    # (red beyond the shell = real; yellow in-shell = expected). --mismatch starts it enabled.
    def add_ovdb_quantities(g, mask):
        if ovdb is None:
            return
        g.add_color_quantity("openvdb sign", sign_to_rgb(ov_sign[mask]), defined_on="cells", enabled=False)
        g.add_color_quantity("ovdb mismatch", mismatch_to_rgb(ov_mm[mask]), defined_on="cells",
                             enabled=args.mismatch)
        g.add_scalar_quantity("openvdb value", ov_val[mask], defined_on="cells", cmap="coolwarm",
                              enabled=False)

    # The right-hand "openvdb: ..." panel — same cells, shifted origin, colored by OpenVDB's sign (its
    # 'sign' quantity IS OpenVDB's). --mismatch flips both panels to the disagreement coloring.
    def register_ovdb_panel(name, mask, grey):
        go = ps.register_sparse_volume_grid(name, origin_ovdb, cw, ijk[mask], enabled=True)
        if grey:
            go.set_color((0.65, 0.65, 0.68))
        go.add_color_quantity("sign", sign_to_rgb(ov_sign[mask]), defined_on="cells",
                              enabled=not args.mismatch)
        go.add_scalar_quantity("openvdb value", ov_val[mask], defined_on="cells", cmap="coolwarm",
                               enabled=False)
        go.add_color_quantity("ovdb mismatch", mismatch_to_rgb(ov_mm[mask]), defined_on="cells",
                              enabled=args.mismatch)
        if args.alpha is not None:
            go.set_transparency(args.alpha)

    if band.any():
        g = ps.register_sparse_volume_grid(prefix + "band voxels", origin, cw, ijk[band],
                                           enabled=True)
        g.add_scalar_quantity("cc", rank[band], defined_on="cells",
                              datatype="categorical", enabled=not (args.sign or args.mismatch))
        g.add_color_quantity("sign", sign_to_rgb(sign[band]), defined_on="cells", enabled=args.sign)
        g.add_scalar_quantity("udf", udf[band], defined_on="cells", cmap="viridis", enabled=False)
        add_ovdb_quantities(g, band)
        if args.alpha is not None:
            g.set_transparency(args.alpha)
        if side_by_side:
            register_ovdb_panel("openvdb: band voxels", band, grey=False)

    if barrier.any():
        g = ps.register_sparse_volume_grid(prefix + "barrier voxels", origin, cw, ijk[barrier],
                                           enabled=True)
        g.set_color((0.65, 0.65, 0.68))              # neutral grey shell when 'sign' is off
        g.add_color_quantity("sign", sign_to_rgb(sign[barrier]), defined_on="cells",
                             enabled=args.sign)
        g.add_scalar_quantity("udf", udf[barrier], defined_on="cells", cmap="viridis", enabled=False)
        add_ovdb_quantities(g, barrier)
        if args.alpha is not None:
            g.set_transparency(args.alpha)
        if side_by_side:
            register_ovdb_panel("openvdb: barrier voxels", barrier, grey=True)

    # --split-cc: one structure per component (subset of the signed band), disabled by default, so you
    # can isolate a single component. Disable "active · signed band" and enable the one you want.
    if args.split_cc:
        if len(table) > 64:
            print(f"WARNING: {len(table)} components — that's a lot of structures; "
                  f"consider --cc to pick a few ranks instead.")
        for r, c, n, s in table:
            m = rank == r
            ps.register_sparse_volume_grid(
                prefix + f"cc {r:02d}  (n={n}, {'+' if s > 0 else '-'})", origin, cw, ijk[m], enabled=False)

    # Step-6 interior fill: one box per interior region, sized to its tree level. Deep interior lives in
    # coarse childless tiles (8³/128³/4096³), not active voxels, so this is the only way to see it.
    fill = None if args.no_fill else load_fill(args.dump + ".fill")
    if fill is not None:
        frec, fvs, ftr = fill
        forigin = tuple(ftr - 0.5 * fvs)   # same half-voxel corner alignment as the active grid
        base, lvl = frec["base"].astype(np.int64), frec["level"].astype(np.int32)
        counts = []
        for level, size, label, color, alpha in FILL_LEVELS:
            m = lvl == level
            if not m.any():
                continue
            counts.append(f"{label}:{int(m.sum())}")
            cells = base[m] // size                      # tile bases are size-aligned -> exact
            g = ps.register_sparse_volume_grid(
                prefix + f"interior voxels ({label})", forigin, (size * fvs,) * 3, cells, enabled=False)
            g.set_color(color)                           # per-level color (shown when 'sign' is off)
            a = args.alpha if args.alpha is not None else alpha
            if a < 1.0:
                g.set_transparency(a)
            # All fill boxes are interior by construction; give them the SAME sign color quantity as the
            # active grids so the unified sign view (--sign) is consistent across levels.
            interior = np.tile(np.array(INTERIOR_RGB, dtype=np.float32), (cells.shape[0], 1))
            g.add_color_quantity("sign", interior, defined_on="cells", enabled=args.sign)
        print(f"step-6 interior fill: {', '.join(counts) if counts else '(none)'} "
              f"boxes (all interior, sign -).")

    print("\nPolyscope structures (toggle each in the left panel):")
    band_quants = "'cc' / 'sign' / 'udf'" + (" / 'openvdb sign' / 'ovdb mismatch'" if ovdb is not None else "")
    if side_by_side:
        print("  == two panels, side by side (left = ours, right = OpenVDB, shifted +X) ==")
        print(f"  ours: band voxels        our result — color by {band_quants}")
        print("  ours: barrier voxels     our surface shell (grey; 'sign'/'udf')")
        print("  openvdb: band voxels     OpenVDB meshToLevelSet — color by 'sign' / 'openvdb value'")
        print("  openvdb: barrier voxels  OpenVDB surface shell")
    else:
        print(f"  band voxels              non-barrier voxels — color by {band_quants}")
        print("  barrier voxels           the surface shell (grey; 'sign'/'udf' available)")
    if fill is not None:
        print(f"  {prefix}interior voxels (...)    step-6 interior, one box per tree-level node")
    if args.split_cc:
        print(f"  {prefix}cc NN ...                one component at a time (disabled by default)")
    print("  tip: run with --sign to color everything by interior/exterior at once.")
    if ovdb is not None:
        print("  tip: run with --mismatch to highlight where our sign disagrees with OpenVDB "
              "(red = beyond the shell).")
        if not side_by_side:
            print("  tip: run with --side-by-side for a 1:1 ours-vs-OpenVDB comparison (two panels).")
    elif not args.no_ovdb:
        print("  (no '.ovdb' companion found — build with OpenVDB and rerun to get the sign comparison.)")

    ps.show()


if __name__ == "__main__":
    main()
