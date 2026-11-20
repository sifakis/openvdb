# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Reproducible scene stages. Invoke with the Python environment containing ovrtx."""

import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BINS = REPO / "build/release/nanovdb/nanovdb/examples"


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    start = time.perf_counter()
    subprocess.run(list(map(str, command)), check=True)
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--stages", default="simulate,mesh,render,debug,validate")
    p.add_argument("--output", type=Path)
    p.add_argument("--character", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--precision", choices=("float","double"))
    p.add_argument("--pressure", choices=("mgpcg","jacobi"))
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    precision=args.precision or config.get("precision","double")
    pressure=args.pressure or config.get("pressure","mgpcg")
    if precision not in ("float","double") or pressure not in ("mgpcg","jacobi"):
        raise ValueError("invalid precision/pressure option")
    config["precision"]=precision;config["pressure"]=pressure
    scene_binary="ex_fluid_apic_scene_float_cuda" if precision=="float" else "ex_fluid_apic_scene_cuda"
    character = args.character or (args.config.parent / config["character"]).resolve()
    out = args.output or HERE.parent / "output" / ("scene_" + args.config.stem)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_config.json").write_text(json.dumps(config, indent=2) + "\n")
    times = {}
    if args.build:
        times["build"] = run(
            [
                "cmake",
                "--build",
                REPO / "build/release",
                "--target",
                scene_binary,
                "ex_fluid_apic_surface",
                "ex_fluid_apic_scene_verify",
                "-j",
                "3",
            ]
        )
    for stage in args.stages.split(","):
        if stage == "verify":
            times[stage] = run(
                [BINS / "ex_fluid_apic_scene_verify", character, config["obstacle_h"]]
            )
        elif stage == "simulate":
            command = [
                BINS / scene_binary,
                "--character",
                character,
                "--output",
                out,
            ]
            for key in (
                "h",
                "obstacle_h",
                "duration",
                "fps",
                "ppc",
                "dt",
                "emitter_radius",
                "emitter_speed",
                "emitter_x",
                "emitter_z",
                "whitewater_rate",
                "animation_offset",
            ):
                command.extend(["--" + key.replace("_", "-"), config[key]])
            command.extend(["--pressure",pressure])
            if args.resume:
                command.extend(["--resume", 1])
            command.extend(["--drain-enabled", int(config.get("drain_enabled", True))])
            for key, default in (
                ("initial_depth", 0),
                ("foam_radius_scale", 1),
                ("foam_life_scale", 1),
            ):
                command.extend(["--" + key.replace("_", "-"), config.get(key, default)])
            times[stage] = run(command)
        elif stage == "mesh":
            times[stage] = run(
                [
                    BINS / "ex_fluid_apic_surface",
                    out,
                    out / "surface",
                    config["mesh_h"],
                    config["h"] ** 3 / config["ppc"],
                    config["obstacle_h"],
                    config["mesh_isovalue"],
                    config["mesh_support"],
                    config["mesh_smoothing"],
                    0,
                    100000,
                    int(not config.get("drain_enabled", True)),
                    int(config.get("mesh_floor_boundary", False)),
                    0,
                    config.get("mesh_closing_radius", 0),
                    config.get("mesh_pca_threads", 4),
                ]
            )
        elif stage == "render":
            times[stage] = run(
                [
                    sys.executable,
                    HERE / "render_scene.py",
                    out,
                    "--character",
                    character,
                    "--width",
                    config["width"],
                    "--height",
                    config["height"],
                    "--samples",
                    config["samples"],
                ]
            )
        elif stage == "debug":
            times[stage] = run(
                [sys.executable, HERE / "debug_scene.py", out, "--character", character]
            )
        elif stage == "validate":
            times[stage] = run([sys.executable, HERE / "validate_scene.py", out,
                                "--workers", config.get("validation_workers", 1)])
            if not config.get("drain_enabled", True):
                times[stage] += run([sys.executable, HERE / "verify_pool.py", out])
        else:
            raise ValueError("unknown stage " + stage)
        # Preserve completed stage costs even if a later validation fails.
        path = out / "pipeline_times.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps(previous | times, indent=2) + "\n")
    path = out / "pipeline_times.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(previous | times, indent=2) + "\n")


if __name__ == "__main__":
    main()
