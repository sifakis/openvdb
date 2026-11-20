#!/usr/bin/env python3
"""Embed the data written by csg_viewer_voxels and csg_viewer_slices into viewer_template.html.

    make_viewer.py --voxels voxels.json --slices slices.json --names dragon armadillo --out csg_viewer.html

Either data file may be left out; the page then shows only the other view. The result is one self-contained
HTML file, except for three.js, which the 3D view loads from a CDN.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voxels", type=Path, help="JSON written by csg_viewer_voxels (3D view)")
    parser.add_argument("--slices", type=Path, help="JSON written by csg_viewer_slices (slice view)")
    parser.add_argument("--names", nargs=2, default=["input 1", "input 2"], metavar=("FIXED", "MOVING"),
                        help="names of the two inputs, shown on the page")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.voxels and not args.slices:
        parser.error("give --voxels, --slices or both")

    def embedded(path):
        if not path:
            return "null"
        text = path.read_text()
        json.loads(text)  # fail here, not in the browser, if the file is not valid JSON
        return text.strip()

    html = (HERE / "viewer_template.html").read_text(encoding="utf-8")
    for marker, value in (("/*VOXELS*/null", embedded(args.voxels)),
                          ("/*SLICES*/null", embedded(args.slices)),
                          ('/*NAMES*/["input 1", "input 2"]', json.dumps(args.names))):
        if html.count(marker) != 1:
            raise SystemExit(f"viewer_template.html: marker {marker!r} not found exactly once")
        html = html.replace(marker, value)
    args.out.write_text(html, encoding="utf-8")
    print(f"{args.out}: {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
