# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Controlled beauty/no-foam render pair: verify visible white surface foam.

Water, spray, character, lights and camera are identical. Bright positive RGB
differences measure the foam's image contribution (including indirect light),
not a particle count and not a fluid-accuracy test.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("cache", type=Path)
p.add_argument("--frame", type=int, required=True)
p.add_argument(
    "--character", type=Path, default=Path(__file__).resolve().parent.parent / "output/character"
)
p.add_argument("--reuse", action="store_true")
p.add_argument("--width", type=int, default=1600)
p.add_argument("--height", type=int, default=1200)
p.add_argument("--samples", type=int, default=192)
a = p.parse_args()
if not a.reuse:
    for mode in ("beauty", "nofoam"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("render_scene.py")),
                str(a.cache),
                "--character",
                str(a.character),
                "--only-frame",
                str(a.frame),
                "--whitewater-mode",
                mode,
                "--width",
                str(a.width),
                "--height",
                str(a.height),
                "--samples",
                str(a.samples),
            ],
            check=True,
        )
name = f"frame_{a.frame:06d}.png"
beauty = np.asarray(Image.open(a.cache / "preview" / name).convert("RGB"), dtype=float)
hidden = np.asarray(Image.open(a.cache / "preview_nofoam" / name).convert("RGB"), dtype=float)
assert beauty.shape == hidden.shape
light = np.array([0.2126, 0.7152, 0.0722])
white = (beauty.min(axis=2) > 120) & (np.ptp(beauty, axis=2) < 65)
contribution = (beauty - hidden) @ light > 30
mask = white & contribution
floor = np.zeros(mask.shape, bool)
floor[int(0.62 * mask.shape[0]) :] = True
fraction = float(mask.mean())
floor_fraction = float((mask & floor).mean())
result = {
    "pass": fraction >= 0.002 and floor_fraction >= 0.001,
    "frame": a.frame,
    "white_contribution_pixels": int(mask.sum()),
    "white_contribution_fraction": fraction,
    "minimum_white_fraction": 0.002,
    "floor_white_pixels": int((mask & floor).sum()),
    "floor_white_fraction": floor_fraction,
    "minimum_floor_fraction": 0.001,
    "criterion": "min RGB>120, channel spread<65, positive luminance difference>30; floor ROI is lower 38% of image",
    "note": "Controlled visibility change includes indirect effects; visual review remains required. Water stays visible in both renders.",
}
out = a.cache / "foam_visibility"
out.mkdir(exist_ok=True)
(out / f"frame_{a.frame:06d}.json").write_text(json.dumps(result, indent=2) + "\n")
Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"mask_{a.frame:06d}.png")
canvas = Image.new("RGB", (beauty.shape[1] * 2, beauty.shape[0] + 28), (20, 25, 30))
canvas.paste(Image.fromarray(hidden.astype(np.uint8)), (0, 28))
canvas.paste(Image.fromarray(beauty.astype(np.uint8)), (beauty.shape[1], 28))
d = ImageDraw.Draw(canvas)
d.text((12, 8), "Same water, foam hidden", fill="white")
d.text((beauty.shape[1] + 12, 8), "White foam visible", fill="white")
canvas.save(out / f"comparison_{a.frame:06d}.jpg", quality=95)
print(json.dumps(result, indent=2))
if not result["pass"]:
    raise SystemExit("Foam is not visibly white enough in the beauty render")
