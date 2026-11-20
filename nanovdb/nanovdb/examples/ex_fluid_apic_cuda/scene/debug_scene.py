# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Independent particle/triangle diagnostics and allocated NanoVDB leaf blocks."""

import argparse, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from render_scene import character_pose, table


def main(args):
    meta = json.loads((args.character / "metadata.json").read_text())
    seq = np.load(args.character / "vertices.npy", mmap_mode="r")
    tri = np.load(args.character / "triangles.npy")
    settings = json.loads((args.cache / "settings.json").read_text())
    records = table(args.cache / "frames.csv")
    out = args.cache / "debug"
    out.mkdir(exist_ok=True)
    if args.representative:
        records = [records[i] for i in sorted(set([0, len(records) // 2, len(records) - 1]))]
    for record in records:
        k = int(record["frame"])
        t = float(record["time"])
        image = Image.new("RGB", (1440, 900), "#101b29")
        draw = ImageDraw.Draw(image)
        points = character_pose(seq, t + settings["animation_offset"], meta["source_fps"])
        fluid = np.loadtxt(args.cache / f"frame_{k:06d}.csv", delimiter=",", skiprows=1, ndmin=2)
        ww = table(args.cache / f"frame_{k:06d}_whitewater.csv")
        blocks = table(args.cache / f"frame_{k:06d}_blocks.csv")

        # Left: all particles in an oblique view. Right: a 12 cm slab / leaf halo.
        def projection(p, panel):
            if panel == 0:
                return np.column_stack(
                    (
                        75 + 205 * (p[:, 0] + 0.28 * (p[:, 2] - 1.2)),
                        820 - 238 * p[:, 1] + 35 * (p[:, 2] - 1.2),
                    )
                )
            return np.column_stack((765 + 195 * p[:, 0], 820 - 238 * p[:, 1]))

        for panel in (0, 1):
            xy = projection(points, panel)
            if panel == 1:
                for b in blocks:
                    if float(b["z"]) <= 1.0 < float(b["z"]) + 8 * settings["h"]:
                        x, y = float(b["x"]), float(b["y"])
                        a = projection(
                            np.array(
                                [[x, y, 0], [x + 8 * settings["h"], y + 8 * settings["h"], 0]]
                            ),
                            panel,
                        )
                        draw.rectangle(
                            (a[0, 0], a[1, 1], a[1, 0], a[0, 1]),
                            outline="#986b35" if int(b["core"]) else "#284853",
                        )
            for indices in tri[::2]:
                draw.line([tuple(v) for v in xy[indices[[0, 1, 2, 0]]]], fill="#865d59", width=1)
            p = fluid[:, 1:4]
            mask = np.ones(len(p), bool) if panel == 0 else abs(p[:, 2] - 1) < 0.06
            xyf = projection(p[mask], panel)
            draw.point([tuple(q) for q in xyf], fill="#43c7ff")
            for kind, color in [(0, "#ffffff"), (1, "#fcda8b")]:
                values = np.array(
                    [[float(q[a]) for a in ("x", "y", "z")] for q in ww if int(q["kind"]) == kind],
                    dtype=float,
                ).reshape(-1, 3)
                if panel:
                    values = values[abs(values[:, 2] - 1) < 0.06]
                if len(values):
                    draw.point([tuple(q) for q in projection(values, panel)], fill=color)
            edge = projection(
                np.array([[0, 0, 0], [3.2, 0, 0], [3.2, 3.2, 0], [0, 3.2, 0], [0, 0, 0]]), panel
            )
            draw.line([tuple(q) for q in edge], fill="#a1aab6", width=1)
        draw.text(
            (35, 18),
            f"DANCING WATER | simulation {t:.3f} s | h={settings['h']:.3f} m | {len(fluid):,} APIC particles",
            fill="white",
        )
        draw.text(
            (35, 42),
            "BLUE: liquid mass | WHITE: foam | GOLD: spray | RED: evaluated character triangles",
            fill="#b8cadc",
        )
        draw.text(
            (765, 65), "z = 1.00 +/- 0.06 m | orange: fluid blocks | cyan: halo", fill="#b8cadc"
        )
        image.save(out / f"frame_{k:06d}.png")
        print("DEBUG", k, flush=True)
    if not args.representative and len(records) > 1:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(settings["fps"]),
                "-i",
                str(out / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(args.cache / "debug.mp4"),
            ],
            check=True,
        )
    # Source poses prove extraction deformation, including the loop endpoint.
    canvas = Image.new("RGB", (1500, 760), "#152031")
    d = ImageDraw.Draw(canvas)
    for panel, k in enumerate((0, (len(seq) - 1) // 2, len(seq) - 1)):
        v = seq[k]
        xy = np.column_stack((panel * 500 + 45 + 220 * (v[:, 0] - 0.6), 700 - 350 * v[:, 1]))
        for ids in tri[::2]:
            d.line([tuple(x) for x in xy[ids[[0, 1, 2, 0]]]], fill="#c9aaa0")
        d.text(
            (panel * 500 + 25, 20),
            f'SOURCE FRAME {k+1} | {k/meta["source_fps"]:.3f} s',
            fill="white",
        )
    canvas.save(args.character / "extracted_poses.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("--character", type=Path, required=True)
    p.add_argument("--representative", action="store_true")
    main(p.parse_args())
