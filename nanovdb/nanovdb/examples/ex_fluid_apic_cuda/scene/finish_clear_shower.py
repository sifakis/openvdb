# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Finish the 42 cm, 12 second refinement experiment after streaming simulation.

Run with the rendering virtualenv. This does not launch or alter simulation.
Every validation keeps the established thresholds; failures stop publication.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

EX = Path(__file__).resolve().parent.parent
SCENE = EX / "scene"
DIAG = EX / "output/stream_diagnosis"
FINE = EX / "output/scene_clear_shower_fine"
COARSE = EX / "output/scene_clear_shower_coarse"
CHARACTER = EX / "output/character_grounded"
BINS = EX.parents[3] / "build/release/nanovdb/nanovdb/examples"


def run(name, command):
    start = time.perf_counter()
    print("START", name, flush=True)
    with (DIAG / (name + ".log")).open("w") as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - start
    print("DONE", name, elapsed, flush=True)
    return elapsed


def script(name, cache, *args):
    return [sys.executable, SCENE / (name + ".py"), cache, *args]


def main():
    start = time.monotonic()
    while True:
        for name in ("fine_simulation.log", "fine_meshing.log"):
            path = DIAG / name
            if path.exists() and "FAIL" in path.read_text():
                raise RuntimeError("Failed upstream task: " + str(path))
        mesh_log = FINE / "surface/meshing.csv"
        rows = list(csv.DictReader(mesh_log.open())) if mesh_log.exists() else []
        if (FINE / "complete.json").exists() and len(rows) == 361:
            break
        if time.monotonic() - start > 21600:
            raise TimeoutError("Waiting for 12 second simulation and all 361 meshes")
        time.sleep(10)

    elapsed = {}
    elapsed["fine_affine_energy"] = run("fine_affine_energy", [
        BINS / "ex_fluid_apic_affine_verify", CHARACTER, FINE])
    for name, cache in (("coarse", COARSE), ("fine", FINE)):
        for label, validator, filename, args in (
            ("validation", "validate_scene", "validation.json", ("--workers", 3)),
            ("pool", "verify_pool", "pool_validation.json", ()),
        ):
            previous = cache / filename
            if name == "coarse" and previous.exists():
                result = json.loads(previous.read_text())
                assert result["pass"], "Previously completed coarse validation failed"
                print("REUSE", previous, flush=True)
                elapsed[name + "_" + label] = {"reused_completed_result": str(previous)}
            else:
                elapsed[name + "_" + label] = run(name + "_final_" + label, script(validator, cache, *args))
        elapsed[name + "_render"] = run(name + "_final_render", script("render_scene", cache,
            "--character", CHARACTER, "--width", 1600, "--height", 1200, "--samples", 96))
    elapsed["debug"] = run("fine_final_debug", script("debug_scene", FINE, "--character", CHARACTER, "--representative"))
    elapsed["foam_visibility"] = run("fine_final_foam", script("verify_white_foam", FINE,
        "--character", CHARACTER, "--frame", 360))

    for name, cache in (("coarse", COARSE), ("fine", FINE)):
        video = cache / "dance_water.mp4"
        run(name + "_video_decode", ["ffmpeg", "-v", "error", "-i", video, "-f", "null", "-"])
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,duration",
            "-of", "json", str(video)], text=True))
        expected = 361 if name == "fine" else 91
        assert int(probe["streams"][0]["nb_read_frames"]) == expected
        (DIAG / (name + "_video.json")).write_text(json.dumps(probe, indent=2) + "\n")

    for label, left, right, duration, titles in (
        ("resolution_motion", COARSE, FINE, 3,
         ("32 mm solver | 42 cm emitter", "20 mm solver | 42 cm emitter")),
        ("previous_vs_refined", EX / "output/scene_grounded_shower", FINE, 12,
         ("Previous: 32 mm | 52 cm emitter", "Refined: 20 mm | 42 cm emitter")),
    ):
        filt = (
            f"[0:v]scale=800:600,drawtext=text='{titles[0]}':x=15:y=15:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[a];"
            f"[1:v]scale=800:600,drawtext=text='{titles[1]}':x=15:y=15:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[b];"
            "[a][b]hstack=inputs=2[v]"
        )
        # drawtext treats ':' as an option delimiter even inside text values.
        filt = filt.replace("Previous:", "Previous").replace("Refined:", "Refined")
        run(label, ["ffmpeg", "-y", "-v", "error", "-i", left / "dance_water.mp4",
            "-i", right / "dance_water.mp4", "-filter_complex", filt, "-map", "[v]",
            "-t", duration, "-c:v", "libx264", "-crf", 18, "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", DIAG / (label + ".mp4")])

    canvas = Image.new("RGB", (1600, 1248), (18, 25, 32))
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate((0, 90, 330, 360)):
        image = Image.open(FINE / "render" / f"frame_{frame:06d}.png").convert("RGB")
        x, y = (i % 2) * 800, (i // 2) * 624
        canvas.paste(image.resize((800, 600)), (x, y + 24))
        draw.text((x + 12, y + 6), f"20 mm solver / 42 cm emitter / t={frame / 30:g} s", fill="white")
    canvas.save(DIAG / "fine_representative.jpg", quality=95)
    (DIAG / "postprocess_complete.json").write_text(json.dumps(elapsed, indent=2) + "\n")


if __name__ == "__main__":
    main()
