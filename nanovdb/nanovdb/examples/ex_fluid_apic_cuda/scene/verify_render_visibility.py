# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Compare direct final-frame loading with an initially empty whitewater animation.

First generate --only-frame LAST --whitewater-mode diagnostic with render_scene.py.
The renderer and its library are unchanged; only a diagnostic USD copy is edited.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade
from render_scene import material, render, table

p = argparse.ArgumentParser()
p.add_argument("cache", type=Path)
p.add_argument("--reuse", action="store_true")
a = p.parse_args()
settings = json.loads((a.cache / "settings.json").read_text())
last = table(a.cache / "frames.csv")[-1]
frame = int(last["frame"])
static = a.cache / "preview_diagnostic" / f"frame_{frame:06d}.png"
animated = a.cache / "preview_animated_diagnostic" / f"frame_{frame:06d}.png"
if not a.reuse:
    s = Usd.Stage.Open(str((a.cache / "scene.usdc").resolve()))
    UsdGeom.Imageable(s.GetPrimAtPath("/World/Water")).CreateVisibilityAttr("invisible")
    m = material(s, "SprayDebug", (1.0, 0.25, 0.025), 0.4)
    UsdShade.MaterialBindingAPI.Apply(s.GetPrimAtPath("/World/Spray")).Bind(m)
    path = a.cache / "empty_mesh_diagnostic.usdc"
    s.GetRootLayer().Export(str(path.resolve()))
    s = None
    timing = json.loads((a.cache / "render/timings.json").read_text())
    args = SimpleNamespace(
        cache=a.cache,
        only_frame=frame,
        whitewater_mode="animated_diagnostic",
        samples=timing["samples"],
        width=timing["resolution"][0],
        height=timing["resolution"][1],
        initial_time=0,
    )
    render(args, path, [last], settings)
x = np.asarray(Image.open(static).convert("RGB"), dtype=float)
y = np.asarray(Image.open(animated).convert("RGB"), dtype=float)
error = float(np.abs(x - y).mean())
changed = float(np.mean(np.abs(x - y).max(axis=2) > 10))
orange = int(
    np.count_nonzero(
        (y[:, :, 0] > 60) & (y[:, :, 1] < 0.65 * y[:, :, 0]) & (y[:, :, 2] < 0.3 * y[:, :, 0])
    )
)
result = {
    "pass": error < 0.5 and changed < 0.005 and orange > 50,
    "direct_load_time": float(last["time"]),
    "animation_initial_time": 0,
    "mean_absolute_8bit_pixel_difference": error,
    "mean_difference_threshold": 0.5,
    "fraction_pixels_difference_over_10": changed,
    "fraction_threshold": 0.005,
    "orange_spray_pixels": orange,
    "minimum_orange_pixels": 50,
    "note": "Small differences include ray sampling; this verifies the empty-to-nonempty geometry update, not fluid accuracy.",
}
(a.cache / "render_visibility.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert result["pass"]
