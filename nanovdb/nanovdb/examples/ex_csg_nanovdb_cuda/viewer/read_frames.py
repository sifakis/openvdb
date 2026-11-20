#!/usr/bin/env python3
"""Read the JSON files written by csg_viewer_slices and csg_viewer_voxels (standard library only).

The functions here are the reference for the data format described in README.md. Other scripts, for example a
figure generator, can import this module or copy what they need.

    python3 read_frames.py --slices slices.json --voxels voxels.json     # check the files against their own counts
"""
import argparse
import base64
import json
import struct
from pathlib import Path

OPERATIONS = ("union", "intersection", "difference")

# ---- slices -----------------------------------------------------------------------------------------------------

# Slice codes, per voxel of the slice. Bit 3 (value 8) is added where a leaf exists at the voxel.
BIT_OFF, ACTIVE_NEGATIVE, ACTIVE_POSITIVE, LEAF_BIT, LOWER_BIT, UPPER_BIT, ROOT_BIT = range(7)
HAS_LEAF = 8


def decode_slice(text, width, height):
    """Run-length encoded slice -> list of `height` rows, each a bytes object of `width` codes.

    The text is a sequence of runs: one letter ('A' = code 0, 'B' = code 1, ...) followed by the run length.
    Row 0 is the top row of the picture; see README.md for the voxel each entry stands for.
    """

    flat = bytearray()
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and text[j].isdigit():
            j += 1
        flat.extend(bytes([ord(text[i]) - 65]) * int(text[i + 1:j]))
        i = j
    if len(flat) != width * height:
        raise ValueError(f"slice has {len(flat)} entries, expected {width * height}")
    return [bytes(flat[r * width:(r + 1) * width]) for r in range(height)]


def slice_voxel(data, row, column):
    """The (x, y, z) index coordinate of the entry at `row`, `column` of a slice of the slice file `data`."""
    voxel = [0, 0, 0]
    voxel[data["sliceAxis"]] = data["sliceIndex"]
    voxel[data["uAxis"]] = data["x0"] + column
    voxel[data["vAxis"]] = data["y0"] + data["h"] - 1 - row
    return tuple(voxel)


def slice_result(frame, operation, negate1, negate2):
    """The record of one of the 12 results of a frame of the slice file: keys bg, leaves, lowers, uppers, active,
    bits (invert bits on in the leaf, lower and upper masks) and v (the encoded slice)."""
    return frame["uid"[OPERATIONS.index(operation)] + str(int(negate1)) + str(int(negate2))]


def classify(code, background_negative=False, negated=False):
    """What a slice code means: (state, level).

    state is 'active-' or 'active+' (a stored value of that sign), 'inside' or 'outside' (no stored value).
    level is None for a stored value and for a bit that is off, else 'leaf', 'lower', 'upper' or 'root': the tree
    level whose invert mask marks the voxel.

    background_negative: the level set has a negative background, so a bit that is off means inside and a bit
    that is on means outside. negated: the codes are those of a level set before it was negated (the inputs are
    stored once, not negated); this flips the sign of the active values and the meaning of the bits.
    """
    code &= 7
    if code in (ACTIVE_NEGATIVE, ACTIVE_POSITIVE):
        negative = (code == ACTIVE_NEGATIVE) != negated
        return ("active-" if negative else "active+", None)
    flipped = background_negative != negated
    if code == BIT_OFF:
        return ("inside" if flipped else "outside", None)
    level = {LEAF_BIT: "leaf", LOWER_BIT: "lower", UPPER_BIT: "upper", ROOT_BIT: "root"}[code]
    return ("outside" if flipped else "inside", level)


# ---- voxels -----------------------------------------------------------------------------------------------------

# Each of the 12 results is one of the four stored results of a frame, or its complement:
# (negate1, negate2) -> per operation (union, intersection, difference): (key of the stored result, complemented).
# u = 1 | 2, i = 1 & 2, d = 1 - 2, e = 2 - 1.
VOXEL_BASE = {
    (False, False): (("u", False), ("i", False), ("d", False)),
    (False, True): (("e", True), ("d", False), ("i", False)),
    (True, False): (("d", True), ("e", False), ("u", True)),
    (True, True): (("i", True), ("u", True), ("e", False)),
}


def voxel_result(frame, operation, negate1, negate2):
    """(stored level set of the frame, complemented) for one of the 12 results of a frame of the voxel file.
    A complemented level set has the same voxels and bits; its values and its background have the opposite sign."""
    key, complemented = VOXEL_BASE[(bool(negate1), bool(negate2))][OPERATIONS.index(operation)]
    return frame[key], complemented


def decode_level_set(record):
    """A level set of the voxel file -> dict of lists of (x, y, z) index coordinates:

    negative, positive: the active voxels by the sign of their value
    leaf_bits:          inactive voxels whose bit is on in a leaf's invert mask
    lower_bits:         origins of the childless slots of lower nodes whose bit is on (each is 8 x 8 x 8 voxels)
    upper_bits:         origins of the childless slots of upper nodes whose bit is on (each is 128^3 voxels)
    leaf_origins:       origin of every leaf (each is 8 x 8 x 8 voxels)
    """
    leaves = base64.b64decode(record["t"])
    out = {"negative": [], "positive": [], "leaf_bits": [], "leaf_origins": []}
    for offset in range(0, len(leaves), 198):  # origin (3 x int16), then value, negative and invert masks (64 bytes each)
        ox, oy, oz = struct.unpack_from("<3h", leaves, offset)
        out["leaf_origins"].append((ox, oy, oz))
        value, negative, invert = (leaves[offset + 6 + 64 * k:offset + 70 + 64 * k] for k in range(3))
        for n in range(512):  # voxel n of a leaf is at (n >> 6, (n >> 3) & 7, n & 7) from its origin
            byte, bit = n >> 3, 1 << (n & 7)
            voxel = (ox + (n >> 6), oy + ((n >> 3) & 7), oz + (n & 7))
            if value[byte] & bit:
                out["negative" if negative[byte] & bit else "positive"].append(voxel)
            elif invert[byte] & bit:
                out["leaf_bits"].append(voxel)
    for name, key in (("lower_bits", "l"), ("upper_bits", "u")):
        slots = base64.b64decode(record[key])
        out[name] = [struct.unpack_from("<3h", slots, offset) for offset in range(0, len(slots), 6)]
    return out


# ---- self check -------------------------------------------------------------------------------------------------

def check_slices(path):
    data = json.loads(Path(path).read_text())
    w, h = data["w"], data["h"]
    decode_slice(data["i1"], w, h)
    results = 0
    for frame in data["frames"]:
        decode_slice(frame["i2"], w, h)
        for operation in OPERATIONS:
            for negate1 in (False, True):
                for negate2 in (False, True):
                    decode_slice(slice_result(frame, operation, negate1, negate2)["v"], w, h)
                    results += 1
    bad = sum(frame["ruleBad"] for frame in data["frames"])
    print(f"{path}: {len(data['frames'])} frames, slice {w} x {h}, {results} result slices decoded, "
          f"rule mismatches recorded by the generator: {bad}")
    return bad == 0


def check_voxels(path):
    data = json.loads(Path(path).read_text())
    records = [("i1", data["i1"])]
    for index, frame in enumerate(data["frames"]):
        records += [(f"frame {index} {key}", frame[key]) for key in ("i2", "u", "i", "d", "e")]
    wrong = 0
    for name, record in records:
        decoded = decode_level_set(record)
        counts = (len(decoded["leaf_origins"]), len(decoded["negative"]) + len(decoded["positive"]),
                  [len(decoded["leaf_bits"]), len(decoded["lower_bits"]), len(decoded["upper_bits"])])
        if counts != (record["leaves"], record["active"], record["bits"]):
            print(f"  {name}: decoded {counts}, recorded {(record['leaves'], record['active'], record['bits'])}")
            wrong += 1
    bad = sum(frame["ruleBad"] + frame["mapBad"] for frame in data["frames"])
    print(f"{path}: {len(data['frames'])} frames, {len(records)} level sets decoded, {wrong} with counts that differ "
          f"from the recorded ones; mismatches recorded by the generator: {bad}")
    return wrong == 0 and bad == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slices", help="JSON written by csg_viewer_slices")
    parser.add_argument("--voxels", help="JSON written by csg_viewer_voxels")
    args = parser.parse_args()
    ok = True
    if args.slices:
        ok &= check_slices(args.slices)
    if args.voxels:
        ok &= check_voxels(args.voxels)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
