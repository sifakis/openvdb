#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""make_scene.py <out dir> --mesh-dir DIR [options]: bins shaped like the letters of a word, filled with many small meshes, written as
one triangle soup for MeshToSDF.

Units: the cap height of the letters is 1. Axes: x along the word, y up the letters, z out of the ground. Every letter
is a tray lying on the ground: a floor under the letter and a wall of height --wall along its outline.

A bin is made as a level set on a grid of --bin-res cells per cap height (from the letter's 2D distance field,
rendered with a font) and turned into triangles by bin/mesh_band (OpenVDB volumeToMesh; build.sh). The small meshes are
closed OBJ meshes of --mesh-dir scaled to a longest extent of about --size, randomly rotated, on a jittered lattice of
spacing --spacing * --size inside the letter, in --layers layers. They overlap one another, the floor and the wall.
The defaults are the scene of the paper.

Output in <out dir>:
  soup.bin     uint64 points, uint64 triangles, float32 xyz per point, int32 abc per triangle: bins and meshes
  bins.meshbin the bins alone (the format mesh_band writes), in the scene's units; meshes.meshbin the small meshes alone
  owners.i32   int32 per point of the soup: -1 for a bin, else the number of the small mesh it belongs to
  scene.json   counts, the bounding box, and the hero: one mesh of the top layer (--hero names its OBJ file) that a
               camera can start from
Run with a Python that has numpy, scipy, Pillow.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = Path(__file__).resolve().parent
# The meshes of the paper's scene, NAME:WEIGHT (get_meshes.sh downloads them): closed meshes of
# github.com/odedstein/meshes whose README states a public-domain, CC0 or CC BY license, and the Stanford bunny and
# armadillo. The weights make the meshes with many triangles rare, so that the scene has about 104 million triangles.
DEFAULT_SET = ["nefertiti:0.003", "cat:0.003", "cheese:0.010", "tree_closed:0.010", "violin:0.015", "koala:0.035",
               "skull:0.035", "scorpion:0.035", "hammer:0.05", "bunny_hr:0.05", "sword:0.03", "boot:0.05", "boat:0.05",
               "armadillo:0.07", "penguin_hr:0.06", "spot:0.06", "fish:0.06", "plane:0.04", "hand_closed:0.04",
               "human_man:0.04", "human_woman:0.04", "wingnut:0.04", "goathead:0.05", "torus:0.03"]


def read_obj(path):
    points, faces = [], []
    with open(path) as stream:
        for line in stream:
            if line.startswith("v "):
                points.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                index = [int(t.split("/")[0]) for t in line.split()[1:]]
                index = [i - 1 if i > 0 else len(points) + i for i in index]
                for k in range(2, len(index)):
                    faces.append([index[0], index[k - 1], index[k]])
    points, faces = np.asarray(points, np.float64), np.asarray(faces, np.int32)
    lo, hi = points.min(0), points.max(0)
    return (points - 0.5 * (lo + hi)) / (hi - lo).max(), faces  # centered, longest extent 1


def read_meshbin(path):
    with open(path, "rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=3)
        vertices = np.fromfile(stream, dtype="<f4", count=3 * int(header[1])).reshape(-1, 3)
        faces = np.fromfile(stream, dtype="<i4", count=3 * int(header[2])).reshape(-1, 3)
    return vertices, faces


def write_meshbin(path, vertices, faces):
    with open(path, "wb") as stream:
        np.array([0x4D45534830303031, len(vertices), len(faces)], "<u8").tofile(stream)
        vertices.astype("<f4").tofile(stream)
        faces.astype("<i4").tofile(stream)


def letter_distance(ch, font, cap_px, pad_px):
    """2D signed distance of a letter in pixels, negative inside; row 0 is the bottom. Also the row of the baseline."""
    width, height = int(cap_px * 1.8) + 2 * pad_px, int(cap_px * 1.5) + 2 * pad_px
    image = Image.new("L", (width, height), 0)
    baseline = pad_px + int(cap_px * 1.25)
    ImageDraw.Draw(image).text((pad_px, baseline), ch, font=font, fill=255, anchor="ls")
    inside = np.asarray(image) > 127
    columns = np.where(inside.any(0))[0]
    inside = inside[:, columns[0] - pad_px:columns[-1] + pad_px + 1]
    distance = ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)
    return distance[::-1].astype(np.float32), height - 1 - baseline


def make_bin(distance, cell, wall, thickness, out_dir, name):
    """Triangles of one tray. distance: the letter's 2D distance on the bin grid, in scene units, [y, x]."""
    pad = 4
    z = (np.arange(-int(np.ceil(thickness / cell)) - pad, int(np.ceil(wall / cell)) + pad + 1) * cell).astype(np.float32)
    d = distance[None, :, :]
    zz = z[:, None, None]
    floor = np.maximum(np.maximum(d - thickness, zz), -thickness - zz)
    side = np.maximum(np.maximum(d - thickness, -d), np.maximum(zz - wall, -thickness - zz))
    phi = np.minimum(floor, side) / cell  # in cells; the zero set is the tray's surface
    k, j, i = np.nonzero(np.abs(phi) < 2.0)
    records = np.zeros(len(i), dtype=[("i", "<i2"), ("j", "<i2"), ("k", "<i2"), ("phi", "<f4")])
    records["i"], records["j"], records["k"], records["phi"] = i, j, k, phi[k, j, i]
    band, meshbin = out_dir / f"bin_{name}.band", out_dir / f"bin_{name}.meshbin"
    with open(band, "wb") as stream:
        np.array([len(records), 0, 0, 0], "<i4").tofile(stream)
        records.tofile(stream)
    subprocess.run([str(HERE / "bin" / "mesh_band"), str(band), str(meshbin)], check=True, stdout=subprocess.DEVNULL)
    vertices, faces = read_meshbin(meshbin)
    band.unlink(), meshbin.unlink()
    return vertices.astype(np.float64) * cell + np.array([0.0, 0.0, z[0]]), faces  # x, y from the grid's corner


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--mesh-dir", type=Path, required=True, help="the directory with the OBJ files")
    p.add_argument("--word", default="SIGGRAPH")
    p.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    p.add_argument("--gap", type=float, default=0.14, help="between the letters' outlines")
    p.add_argument("--wall", type=float, default=0.24, help="height of the walls")
    p.add_argument("--thickness", type=float, default=0.025, help="of the walls and the floor")
    p.add_argument("--bin-res", type=int, default=384, help="cells per cap height of the grid the bins are made on")
    p.add_argument("--size", type=float, default=0.12, help="longest extent of a small mesh")
    p.add_argument("--spacing", type=float, default=0.62, help="lattice spacing in units of --size")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--margin", type=float, default=0.42, help="centers stay this far inside the letter, in units of --size")
    p.add_argument("--meshes", nargs="*", default=DEFAULT_SET,
                   help="names of OBJ files in --mesh-dir, each optionally NAME:WEIGHT (how often it is picked, default 1)")
    p.add_argument("--hero", default="nefertiti")
    p.add_argument("--hero-turn", type=float, default=180.0, help="turns the hero about the vertical, degrees")
    p.add_argument("--hero-letter", type=int, help="which letter's bin holds the hero (default: the fifth, or the last of a shorter word)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if args.hero_letter is None:
        args.hero_letter = min(4, len(args.word) - 1)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    supersample = 4
    cap_px = args.bin_res * supersample
    font = ImageFont.truetype(args.font, 100)
    box = font.getbbox("H", anchor="ls")
    font = ImageFont.truetype(args.font, int(round(100 * cap_px / (box[3] - box[1]))))
    pad_px = int(np.ceil((args.thickness + 0.03) * cap_px / supersample)) * supersample + 2 * supersample
    cell = 1.0 / args.bin_res

    weights = np.array([float(m.split(":")[1]) if ":" in m else 1.0 for m in args.meshes])
    args.meshes = [m.split(":")[0] for m in args.meshes]
    weights /= weights.sum()
    library = {name: read_obj(args.mesh_dir / (name + ".obj")) for name in dict.fromkeys(args.meshes + [args.hero])}
    spacing = args.spacing * args.size

    turn = np.radians(args.hero_turn)
    hero_rotation = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]]) @ \
        np.array([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]])  # the hero's y up and its -z toward -y, then turned about the vertical
    points, faces, instances, letters, owners = [], [], [], [], []
    bin_points, bin_faces = [], []
    count, cursor, hero = 0, 0.0, None
    for index, ch in enumerate(args.word):
        fine, baseline = letter_distance(ch, font, cap_px, pad_px)
        ny, nx = (fine.shape[0] // supersample) * supersample, (fine.shape[1] // supersample) * supersample
        coarse = fine[:ny, :nx].reshape(ny // supersample, supersample, nx // supersample, supersample).mean((1, 3)) / cap_px
        origin = np.array([cursor, -baseline / cap_px, 0.0])  # scene position of the grid's corner
        v, f = make_bin(coarse, cell, args.wall, args.thickness, args.out, str(index))
        v += origin + np.array([0.5 * cell, 0.5 * cell, 0.0])  # cell centers
        bin_points.append(v), bin_faces.append(f + sum(len(q) for q in bin_points[:-1]))

        # centers on a jittered lattice inside the letter
        placed = 0
        for layer in range(args.layers):
            offset = 0.5 * spacing * (layer % 2)
            xs = np.arange(offset, coarse.shape[1] * cell, spacing)
            ys = np.arange(offset, coarse.shape[0] * cell, spacing)
            gx, gy = np.meshgrid(xs, ys)
            gx = gx.ravel() + rng.uniform(-0.3, 0.3, gx.size) * spacing
            gy = gy.ravel() + rng.uniform(-0.3, 0.3, gy.size) * spacing
            d = ndimage.map_coordinates(coarse, [gy / cell - 0.5, gx / cell - 0.5], order=1, mode="nearest")
            keep = d < -args.margin * args.size
            for x, y in zip(gx[keep], gy[keep]):
                top = layer == args.layers - 1
                is_hero = hero is None and top and index == args.hero_letter and y > 0.45
                name = args.hero if is_hero else args.meshes[rng.choice(len(args.meshes), p=weights)]
                scale = args.size * (1.0 if is_hero else rng.uniform(0.8, 1.15))
                rotation = hero_rotation if is_hero else random_rotation(rng)
                center = origin + np.array([x, y, 0.5 * args.size + layer * spacing * 0.85 + rng.uniform(-0.15, 0.15) * spacing])
                if is_hero:
                    center[2] += 0.25 * args.size  # stands a little above its layer
                pv, pf = library[name]
                points.append((pv * scale) @ rotation.T + center)
                faces.append(pf + count)
                owners.append(np.full(len(pv), len(instances), np.int32))
                count += len(pv)
                if is_hero:
                    hero = {"mesh": name, "center": center.tolist(), "size": scale, "letter": ch, "instance": len(instances)}
                instances.append(name)
                placed += 1
        letters.append({"letter": ch, "x0": cursor, "x1": cursor + coarse.shape[1] * cell, "meshes": placed})
        cursor += coarse.shape[1] * cell - 2 * pad_px / cap_px + args.gap

    bin_points, bin_faces = np.concatenate(bin_points), np.concatenate(bin_faces)
    write_meshbin(args.out / "bins.meshbin", bin_points, bin_faces)
    mesh_points, mesh_faces = np.concatenate(points), np.concatenate(faces)
    write_meshbin(args.out / "meshes.meshbin", mesh_points, mesh_faces)
    all_points = np.concatenate([bin_points, mesh_points]).astype("<f4")
    all_faces = np.concatenate([bin_faces, mesh_faces + len(bin_points)]).astype("<i4")
    np.concatenate([np.full(len(bin_points), -1, np.int32)] + owners).astype("<i4").tofile(args.out / "owners.i32")
    with open(args.out / "soup.bin", "wb") as stream:
        np.array([len(all_points), len(all_faces)], "<u8").tofile(stream)
        all_points.tofile(stream), all_faces.tofile(stream)
    names, uses = np.unique(instances, return_counts=True)
    scene = {"word": args.word, "points": len(all_points), "triangles": len(all_faces), "bin_triangles": len(bin_faces),
             "meshes": len(instances), "mesh_triangles": len(mesh_faces), "by_mesh": dict(zip(names.tolist(), uses.tolist())),
             "lo": all_points.min(0).tolist(), "hi": all_points.max(0).tolist(), "letters": letters, "hero": hero,
             "size": args.size, "wall": args.wall, "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    (args.out / "scene.json").write_text(json.dumps(scene, indent=1))
    extent = all_points.max(0) - all_points.min(0)
    print(f"{args.word}: {len(instances)} meshes ({len(mesh_faces)} triangles) in {len(args.word)} bins ({len(bin_faces)} triangles); "
          f"{len(all_faces)} triangles in all; extents {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}; hero {hero}")
