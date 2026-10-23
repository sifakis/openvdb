#!/usr/bin/env python3
"""View a mesh->SDF pipeline dump in Polyscope as a Sparse Volume Grid.

Companion to ex_mesh_to_sdf_cuda. Produce the dump by running the CUDA
example with the CC_EXPORT_VIS env var, e.g.:

    CC_EXPORT_VIS=/tmp/dragon.ccvis ./ex_mesh_to_sdf_cuda dragon.obj 0.001

then view it:

    pip install polyscope numpy
    python scripts/mesh_to_sdf_viewer.py /tmp/dragon.ccvis

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

If a '<dump>.ovdb' companion is present, each band / barrier voxel also gets a second opinion to
compare our sign against. The companion says which one it holds, and the quantities are named after
it -- 'openvdb ...' or 'ball ...':
  * <other> sign     - the other method's sign at that voxel (same color scheme as 'sign'; a mid grey
                       means it did not decide).
  * <other> mismatch - RED a real disagreement, soft YELLOW an expected/undecided one, neutral grey
                       where the two agree. Which is which depends on the comparison:
                         ours vs OpenVDB (written when the example is built with OpenVDB) - red is a
                           differing sign beyond the √3/2 shell, yellow a tie inside it
                         ball vs shipped (written with CC_VIS_BALL=1) - red is a sign the ball
                           certification proves and the shipped one contradicts, yellow its residue
  * <other> value    - OpenVDB's raw signed level-set value, or the UDF, as a sequential scalar.

Options:
  --sign            start with the unified interior/exterior sign coloring on every structure
  --mismatch        start with the mismatch coloring on (needs the '<dump>.ovdb' companion)
  --alpha A         start every structure at transparency A (0..1); tune per structure in the UI
  --no-barrier      hide barrier voxels (cc == -1)
  --no-fill         skip the step-6 interior-fill boxes (the '<dump>.fill' companion)
  --no-ovdb         ignore the OpenVDB comparison companion even if present
  --split-cc        register EACH connected component as its own structure (toggle individually)
  --cc RANKS        show only these component ranks (comma list, e.g. --cc 2,3,4; rank 0 = largest)
  --balls           draw the union-of-balls exterior certificate (see below)
  --ball-max R      with --balls, keep only spheres of radius <= R voxels
  --mesh OBJ        also load the input surface, to check it against the balls
  --slice           add a slice plane that cuts the spheres open

The '--balls' view exists because the containment guarantee lives on BALLS, not on cells. Ball
B(V, udf(V)) has radius equal to V's distance to the surface, so by construction it cannot contain
surface material -- every sphere is tangent to the input mesh, none swallows a piece of it. Drawing
the certified-exterior voxels at that radius therefore shows the object the guarantee is about, and
a sphere seen eating into the mesh means the UDF over-estimated the distance there. Pair it with
'--mesh' and '--slice':

    CC_EXPORT_VIS=/tmp/bunny.ccvis CC_VIS_BALL=1 CC_BARRIER=ball ./ex_mesh_to_sdf_cuda bunny.obj 0.008
    python scripts/mesh_to_sdf_viewer.py /tmp/bunny.ccvis --balls --mesh bunny.obj --slice

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
UNDECIDED_RGB = (0.45, 0.45, 0.48)  # sign 0 — undecided (a certification companion only)


def sign_to_rgb(sign):
    rgb = np.empty((sign.shape[0], 3), dtype=np.float32)
    rgb[sign < 0] = INTERIOR_RGB
    rgb[sign > 0] = EXTERIOR_RGB
    rgb[sign == 0] = UNDECIDED_RGB   # only a certification companion produces these; ours are always +/-1
    return rgb


# Comparison companion ('<dump>.ovdb'): per-voxel class -> RGB. Class 1 is always the one to spot and
# class 2 the expected/benign one, but what they mean depends on which comparison the file holds (its
# magic, see load_ovdb):
#   CCOVDB01, ours vs OpenVDB   1 = our sign differs where OpenVDB is confident (beyond the √3/2
#                                   shell); 2 = differs inside the shell, where a tie is expected
#   CCBALL01, ball vs shipped   1 = the ball certification proves a sign the shipped one contradicts;
#                                   2 = the certification proved nothing there (residue)
MISMATCH_RGB = {
    0: (0.22, 0.22, 0.25),   # agree / consistent      — neutral dark grey
    1: (1.00, 0.10, 0.10),   # real disagreement       — red
    2: (0.95, 0.85, 0.20),   # expected / undecided    — soft yellow
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
        return None, None
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"CCOVDB01", b"CCBALL01"):
            sys.exit(f"bad magic {magic!r} in {path} (not a CCOVDB01/CCBALL01 dump)")
        (m,) = struct.unpack("<Q", f.read(8))
        (_vs,) = struct.unpack("<d", f.read(8))
        rec = np.fromfile(f, dtype=REC_OVDB, count=m)
    if rec.shape[0] != m:
        sys.exit(f"truncated ovdb: header says {m}, read {rec.shape[0]}")
    if m != expect_n:
        print(f"WARNING: {path} has {m} records but the .ccvis dump has {expect_n}; skipping the "
              f"comparison (regenerate both from the same run).")
        return None, None
    return rec, magic


def load_obj(path):
    """Minimal OBJ reader: vertex positions and fan-triangulated faces. Everything else ignored."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # face entries are "v", "v/vt", "v//vn" or "v/vt/vn"; OBJ indices are 1-based
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


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
                    help="start with the companion's 'mismatch' coloring enabled (red = real "
                         "disagreement, yellow = expected/undecided; needs the '<dump>.ovdb' companion)")
    ap.add_argument("--sign", action="store_true",
                    help="start with the unified sign coloring (interior/exterior) enabled on EVERY "
                         "structure — active voxels and all fill levels — to verify signs across levels")
    ap.add_argument("--balls", action="store_true",
                    help="draw each certified-exterior voxel as a sphere of radius = its UDF, the "
                         "union-of-balls exterior certificate the containment guarantee is about")
    ap.add_argument("--ball-max", type=float, default=None, metavar="R",
                    help="with --balls, keep only spheres whose radius is <= R voxels")
    ap.add_argument("--slab", type=float, default=None, metavar="W",
                    help="with --balls, keep only spheres whose centre is within W voxels of the "
                         "slice plane (a slice view does not need the rest, and dropping them is "
                         "what keeps the window responsive)")
    ap.add_argument("--slab-axis", default="z", choices=("x", "y", "z"),
                    help="axis the --slab is measured along (default z, matching --slice)")
    ap.add_argument("--mesh", default=None, metavar="OBJ",
                    help="also load this OBJ (the input surface) to check it against --balls")
    ap.add_argument("--slice", action="store_true",
                    help="add a scene slice plane that cuts the spheres open instead of hiding them")
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
    ovdb, ov_magic = (None, None) if args.no_ovdb else load_ovdb(args.dump + ".ovdb", rec.shape[0])
    # The companion holds one of two comparisons (see the magic). Everything below is identical for
    # both -- only the wording on the structures/quantities changes, so the panel never mislabels
    # which pair of signs the red voxels come from.
    ball = ov_magic == b"CCBALL01"
    other = "ball" if ball else "openvdb"          # whose sign the right-hand panel shows
    Q_SIGN, Q_MM, Q_VAL = f"{other} sign", f"{other} mismatch", f"{other} value"
    ov_sign = ov_mm = ov_val = None
    if ovdb is not None:
        ov_sign = ovdb["sign"].astype(np.int32)
        ov_mm = ovdb["mismatch"].astype(np.int32)
        ov_val = ovdb["value"].astype(np.float32)
        c1, c2 = int((ov_mm == 1).sum()), int((ov_mm == 2).sum())
        if ball:
            print(f"Ball comparison ('.ovdb'): {c1} voxels where the ball certification disagrees "
                  f"with the shipped sign (RED), {c2} the certification left unproven (yellow), "
                  f"{rec.shape[0] - c1 - c2} agree.")
        else:
            print(f"OpenVDB comparison ('.ovdb'): {c1} beyond-shell sign mismatches (RED), "
                  f"{c2} in-shell ties (yellow), {rec.shape[0] - c1 - c2} agree.")

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

    # --balls gives the scene to the spheres, so the voxel structures start hidden (re-enable any of
    # them in the UI). Every registration below goes through this helper so none is missed.
    voxel_structs = []

    def reg_grid(*a, **kw):
        g = ps.register_sparse_volume_grid(*a, **kw)
        voxel_structs.append(g)
        return g

    ps.set_up_dir("z_up")
    # "pretty" transparency is multi-pass depth peeling: cheap for a few volume grids, but the
    # dominant per-frame cost once --balls puts hundreds of thousands of sphere impostors on
    # screen. Only turn it on when transparency is actually going to be used.
    if args.alpha is not None or not args.balls:
        ps.set_transparency_mode("pretty")   # enables the per-structure Transparency slider

    # Polyscope cell (i,j,k) lower corner at origin + (i,j,k)*cell_width; a NanoVDB voxel center sits
    # at index (i,j,k). Offset origin by half a cell so cell centers land on true voxel centers.
    origin = tuple(translation - 0.5 * vs)
    cw = (vs, vs, vs)

    # Side-by-side: a second copy of the band+barrier voxels, translated +X so it sits NEXT TO the
    # "ours" panel (not overlaid), colored by OpenVDB's sign. Default ON when the .ovdb companion is
    # present. The copy reuses the same integer cells, only the structure origin is shifted in world X.
    side_by_side = (ovdb is not None) and (args.side_by_side is not False)
    if side_by_side and args.balls and args.side_by_side is None:
        # the shifted copy would sit on top of the spheres, which are drawn unshifted
        side_by_side = False
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
        g.add_color_quantity(Q_SIGN, sign_to_rgb(ov_sign[mask]), defined_on="cells", enabled=False)
        g.add_color_quantity(Q_MM, mismatch_to_rgb(ov_mm[mask]), defined_on="cells",
                             enabled=args.mismatch)
        g.add_scalar_quantity(Q_VAL, ov_val[mask], defined_on="cells", cmap="coolwarm",
                              enabled=False)

    # The right-hand "openvdb: ..." panel — same cells, shifted origin, colored by OpenVDB's sign (its
    # 'sign' quantity IS OpenVDB's). --mismatch flips both panels to the disagreement coloring.
    def register_ovdb_panel(name, mask, grey):
        go = reg_grid(name, origin_ovdb, cw, ijk[mask], enabled=True)
        if grey:
            go.set_color((0.65, 0.65, 0.68))
        go.add_color_quantity("sign", sign_to_rgb(ov_sign[mask]), defined_on="cells",
                              enabled=not args.mismatch)
        go.add_scalar_quantity(Q_VAL, ov_val[mask], defined_on="cells", cmap="coolwarm",
                               enabled=False)
        go.add_color_quantity(Q_MM, mismatch_to_rgb(ov_mm[mask]), defined_on="cells",
                              enabled=args.mismatch)
        if args.alpha is not None:
            go.set_transparency(args.alpha)

    if band.any():
        g = reg_grid(prefix + "band voxels", origin, cw, ijk[band],
                                           enabled=True)
        g.add_scalar_quantity("cc", rank[band], defined_on="cells",
                              datatype="categorical", enabled=not (args.sign or args.mismatch))
        g.add_color_quantity("sign", sign_to_rgb(sign[band]), defined_on="cells", enabled=args.sign)
        g.add_scalar_quantity("udf", udf[band], defined_on="cells", cmap="viridis", enabled=False)
        add_ovdb_quantities(g, band)
        if args.alpha is not None:
            g.set_transparency(args.alpha)
        if side_by_side:
            register_ovdb_panel(f"{other}: band voxels", band, grey=False)

    if barrier.any():
        g = reg_grid(prefix + "barrier voxels", origin, cw, ijk[barrier],
                                           enabled=True)
        g.set_color((0.65, 0.65, 0.68))              # neutral grey shell when 'sign' is off
        g.add_color_quantity("sign", sign_to_rgb(sign[barrier]), defined_on="cells",
                             enabled=args.sign)
        g.add_scalar_quantity("udf", udf[barrier], defined_on="cells", cmap="viridis", enabled=False)
        add_ovdb_quantities(g, barrier)
        if args.alpha is not None:
            g.set_transparency(args.alpha)
        if side_by_side:
            register_ovdb_panel(f"{other}: barrier voxels", barrier, grey=True)

    # --split-cc: one structure per component (subset of the signed band), disabled by default, so you
    # can isolate a single component. Disable "active · signed band" and enable the one you want.
    if args.split_cc:
        if len(table) > 64:
            print(f"WARNING: {len(table)} components — that's a lot of structures; "
                  f"consider --cc to pick a few ranks instead.")
        for r, c, n, s in table:
            m = rank == r
            reg_grid(
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
            g = reg_grid(
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

    # ---- The union-of-balls exterior certificate (--balls) ---------------------------------------
    # A voxel's own UDF is its distance to the surface, so the ball of that radius around it cannot
    # contain surface material -- it is tangent to the mesh at worst. Drawing the certified-exterior
    # voxels at that radius shows the set the containment guarantee is stated on (the cells those
    # voxels occupy carry no such guarantee). A sphere that visibly eats into the mesh means the UDF
    # over-estimated the distance there, which is the one premise the guarantee rests on.
    balls_pc = None
    if args.balls:
        if ovdb is not None and ball:
            label, src = ov_sign, "ball certification"     # the certification's own verdict
        else:
            label, src = sign, "shipped sign"
            print("--balls: no CCBALL01 companion (rerun the example with CC_VIS_BALL=1 to get the "
                  "certification's own labels); falling back to the shipped sign.")
        keep = label > 0
        if args.ball_max is not None:
            keep = keep & (udf <= args.ball_max * vs)
        if args.slab is not None:
            # Only spheres near the cut can show up in it; the rest are pure render cost.
            ax = {"x": 0, "y": 1, "z": 2}[args.slab_axis]
            mid = 0.5 * (int(ijk[:, ax].min()) + int(ijk[:, ax].max()))
            keep = keep & (np.abs(ijk[:, ax] - mid) <= args.slab)
            print(f"--slab {args.slab:g}: keeping spheres within {args.slab:g} voxels of "
                  f"{args.slab_axis} = {mid:g}")
        if not keep.any():
            print("--balls: nothing to draw (no voxel is labelled exterior after filtering).")
        else:
            centers = translation + ijk[keep] * vs          # NanoVDB voxel centre, world units
            radii   = udf[keep]
            balls_pc = ps.register_point_cloud("exterior balls", centers)
            balls_pc.add_scalar_quantity("udf", radii, cmap="viridis", enabled=True)
            # autoscale=False keeps the radius in world units; the default would normalise it away
            balls_pc.set_point_radius_quantity("udf", autoscale=False)
            if args.alpha is not None:
                balls_pc.set_transparency(args.alpha)
            print(f"--balls: {int(keep.sum())} spheres from the {src}, radius = UDF "
                  f"({radii.min():g} .. {radii.max():g} world units)")

    if args.mesh is not None:
        mv, mf = load_obj(args.mesh)
        ps.register_surface_mesh("input mesh", mv, mf)
        print(f"--mesh: {mv.shape[0]} vertices, {mf.shape[0]} triangles from {args.mesh}")

    if args.balls and balls_pc is not None:
        for g in voxel_structs:
            g.set_enabled(False)

    if args.slice:
        ps.add_scene_slice_plane()
        if balls_pc is not None:
            # cut each sphere open at the plane rather than hiding whole ones, so the cross-section
            # circles are visible against the mesh's cross-section curve
            balls_pc.set_cull_whole_elements(False)
        print("--slice: drag the slice plane in the UI (Slice Planes section).")

    print("\nPolyscope structures (toggle each in the left panel):")
    band_quants = "'cc' / 'sign' / 'udf'" + (f" / '{Q_SIGN}' / '{Q_MM}'" if ovdb is not None else "")
    rhs = "the ball certification" if ball else "OpenVDB meshToLevelSet"
    if side_by_side:
        print(f"  == two panels, side by side (left = ours, right = {other}, shifted +X) ==")
        print(f"  ours: band voxels        our result — color by {band_quants}")
        print("  ours: barrier voxels     our surface shell (grey; 'sign'/'udf')")
        # pad to the same column as the "ours: ..." rows above, whatever the other method is called
        print(f"  {other + ': band voxels':<24} {rhs} — color by 'sign' / '{Q_VAL}'")
        print(f"  {other + ': barrier voxels':<24} {rhs}, surface shell")
    else:
        print(f"  band voxels              non-barrier voxels — color by {band_quants}")
        print("  barrier voxels           the surface shell (grey; 'sign'/'udf' available)")
    if fill is not None:
        print(f"  {prefix}interior voxels (...)    step-6 interior, one box per tree-level node")
    if args.split_cc:
        print(f"  {prefix}cc NN ...                one component at a time (disabled by default)")
    print("  tip: run with --sign to color everything by interior/exterior at once.")
    if ovdb is not None:
        if ball:
            print("  tip: run with --mismatch to highlight the certification (red = it proves a sign "
                  "the shipped one contradicts, yellow = it proved nothing there).")
        else:
            print("  tip: run with --mismatch to highlight where our sign disagrees with OpenVDB "
                  "(red = beyond the shell).")
        if not side_by_side:
            print(f"  tip: run with --side-by-side for a 1:1 ours-vs-{other} comparison (two panels).")
    elif not args.no_ovdb:
        print("  (no '.ovdb' companion found — build with OpenVDB and rerun to get the sign comparison.)")

    ps.show()


if __name__ == "__main__":
    main()
