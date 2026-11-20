# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Supervise an interruptible scene run. Launch under systemd-inhibit.

STOP requests a pause after the next atomic checkpoint. --resume continues it.
Numerical failures stop the run; they never cause tolerance changes or retries.
"""
import argparse
import atexit
import csv
import ctypes
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EX = HERE.parent
BINS = EX.parents[3] / "build/release/nanovdb/nanovdb/examples"


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def checkpoint(folder):
    try:
        with (folder / "checkpoint.bin").open("rb") as f:
            header = f.read(216)
            if len(header) != 216:
                return None
            fields = struct.unpack_from("<10Q", header)
            expected = 216 + fields[3] * (fields[1] + 8) + fields[4] * fields[2]
            if fields[0] != 0x464C554944303034 or os.fstat(f.fileno()).st_size != expected:
                return None
            return {"time": struct.unpack_from("<d", header, 80)[0],
                    "next_frame": struct.unpack_from("<i", header, 152)[0],
                    "particles": fields[3], "bytes": expected}
    except FileNotFoundError:
        return None


def last_metric(folder):
    path = folder / "metrics.csv"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        keys = f.readline().decode().strip().split(",")
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16384))
        lines = f.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        values = line.split(",")
        if len(values) == len(keys):
            try:
                return dict(zip(keys, map(float, values)))
            except ValueError:
                pass
    return {}


class GpuMemory:
    """NVML samples observe transient allocations, but do not prove an exact peak."""
    class Memory(ctypes.Structure):
        _fields_ = [(key, ctypes.c_ulonglong) for key in ("total", "free", "used")]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlInit_v2() or self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)):
            raise RuntimeError("NVML initialization failed")

    def sample(self):
        value = self.Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(value)):
            return None
        return value.used / 2**20


def simulation_command(config, folder, executable, resume):
    command = [str(executable), "--character", config["character"], "--output", str(folder)]
    for key in ("h", "obstacle_h", "duration", "fps", "ppc", "dt", "emitter_radius",
                "emitter_speed", "emitter_x", "emitter_z", "whitewater_rate", "animation_offset",
                "initial_depth", "foam_radius_scale", "foam_life_scale", "pressure"):
        command += ["--" + key.replace("_", "-"), str(config[key])]
    command += ["--drain-enabled", str(int(config["drain_enabled"]))]
    if resume:
        command += ["--resume", "1"]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--stream-mesh", action="store_true",
                        help="Reconstruct committed frames on the CPU while simulation runs")
    parser.add_argument("--resume-mesh", action="store_true",
                        help="Replay and verify temporal surface history before extending existing meshes")
    args = parser.parse_args()
    if args.resume_mesh and not (args.resume and args.stream_mesh):
        parser.error("--resume-mesh requires --resume --stream-mesh")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A supervisor already owns this cache")
    status_file = folder / "status.json"
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        pid = previous.get("supervisor_pid")
        try:
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes()
        except FileNotFoundError:
            command_line = b""
        if pid != os.getpid() and b"overnight_run.py" in command_line and str(folder).encode() in command_line:
            raise RuntimeError("This cache is already running")
    config = json.loads(args.config.read_text())
    config["character"] = str((args.config.parent / config["character"]).resolve())
    if config.get("precision") != "float":
        raise ValueError("This run supervisor expects the float executable")
    executable = folder / "scene_float_cuda"
    if not executable.exists():
        shutil.copy2(BINS / "ex_fluid_apic_scene_float_cuda", executable)
    if args.resume and checkpoint(folder) is None:
        raise RuntimeError("No complete checkpoint available")
    if not args.resume and (folder / "checkpoint.bin").exists():
        raise RuntimeError("Existing cache requires --resume")
    if (folder / "STOP").exists():
        raise RuntimeError("Remove STOP before resuming")
    complete = folder / "complete.json"
    if args.resume and complete.exists():
        complete.replace(folder / "previous_complete.json")
    if args.resume_mesh:
        previous_config = json.loads((folder / "pipeline_config.json").read_text())
        for key in ("h", "ppc", "mesh_h", "obstacle_h", "mesh_isovalue", "mesh_support",
                    "mesh_smoothing", "mesh_closing_radius", "drain_enabled", "mesh_floor_boundary"):
            if config.get(key) != previous_config.get(key):
                raise RuntimeError(f"Cannot resume meshes after changing {key}")
    atomic_json(folder / "pipeline_config.json", config)
    start = time.monotonic()
    base = checkpoint(folder)
    start_time = base["time"] if base else 0
    status = {"state": "simulating", "supervisor_pid": os.getpid(), "target_seconds": config["duration"],
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "resume_time": start_time}
    gpu = GpuMemory()
    peak = 0
    pause_requested = False

    def request_pause(signum, frame):
        nonlocal pause_requested
        pause_requested = True

    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    mesh_child = None
    mesh_started = None
    mesh_continuation = None
    mesh_continuation_error = None
    if args.stream_mesh:
        mesh_executable = folder / "surface_cpu"
        if not mesh_executable.exists():
            shutil.copy2(BINS / "ex_fluid_apic_surface", mesh_executable)
        mesh_first, mesh_output = 0, folder / "surface"
        if args.resume_mesh:
            from surface_resume import SurfaceContinuation
            mesh_continuation = SurfaceContinuation(folder, base["next_frame"])
            mesh_first, mesh_output = mesh_continuation.first, mesh_continuation.output
            status["surface_resume"] = {"first_replayed_frame": mesh_first,
                                         "prefix_last_frame": mesh_continuation.last,
                                         "staging": str(mesh_output)}
        mesh_command = [str(mesh_executable), str(folder), str(mesh_output),
                        str(config["mesh_h"]), str(config["h"] ** 3 / config["ppc"]),
                        str(config["obstacle_h"]), str(config["mesh_isovalue"]),
                        str(config["mesh_support"]), str(config["mesh_smoothing"]),
                        str(mesh_first), str(round(config["duration"] * config["fps"])),
                        str(int(not config["drain_enabled"])),
                        str(int(config.get("mesh_floor_boundary", False))), "1",
                        str(config.get("mesh_closing_radius", 0)),
                        str(config.get("mesh_pca_threads", 4))]
        mesh_started = time.monotonic()
        with (folder / "mesh.log").open("a") as mesh_log:
            mesh_child = subprocess.Popen(mesh_command, stdout=mesh_log, stderr=subprocess.STDOUT,
                                          start_new_session=True,
                                          env=os.environ | {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "2"})
        status["mesher_pid"] = mesh_child.pid

        def stop_mesher():
            if mesh_child.poll() is None:
                os.killpg(mesh_child.pid, signal.SIGTERM)
                try:
                    mesh_child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(mesh_child.pid, signal.SIGKILL)
                    mesh_child.wait()

        atexit.register(stop_mesher)
    with (folder / "simulation.log").open("a", buffering=1) as log:
        command = simulation_command(config, folder, executable, args.resume)
        print("START", command, flush=True)
        log.write(f"\nSTART {status['started_at']} resume_time={start_time} target={config['duration']}\n")
        log.flush()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status["solver_pid"] = child.pid
        atomic_json(folder / "status.json", status)
        pending = None
        next_report = 0
        with (folder / "monitor.csv").open("a", buffering=1) as monitor:
            if monitor.tell() == 0:
                monitor.write("wall_seconds,simulation_seconds,checkpoint_seconds,gpu_mib,sampled_peak_mib,disk_free_gib\n")
            while child.poll() is None:
                current = gpu.sample()
                if current is not None:
                    peak = max(peak, current)
                elapsed = time.monotonic() - start
                if elapsed >= next_report:
                    cp = checkpoint(folder)
                    metric = last_metric(folder)
                    sim_time = metric.get("time", start_time)
                    free = shutil.disk_usage(folder).free / 2**30
                    status.update(elapsed_hours=elapsed / 3600, simulation_seconds=sim_time,
                                  checkpoint=cp, gpu_mib=current, sampled_peak_mib=peak, disk_free_gib=free)
                    if mesh_child is not None:
                        status["mesher_returncode"] = mesh_child.poll()
                    if mesh_continuation is not None and mesh_continuation_error is None:
                        try:
                            mesh_continuation.poll()
                            status["surface_resume"]["verified"] = mesh_continuation.verified
                            status["surface_resume"]["last_published_frame"] = mesh_continuation.last_published
                        except Exception as error:
                            # A render-history discrepancy must not orphan or stop
                            # the GPU solver. Preserve it for explicit diagnosis.
                            mesh_continuation_error = str(error)
                            status["mesh_continuation_error"] = mesh_continuation_error
                            stop_mesher()
                    if sim_time - start_time > 1e-6:
                        status["linear_remaining_hours"] = elapsed / (sim_time - start_time) * (config["duration"] - sim_time) / 3600
                    reason = None
                    if pause_requested or (folder / "STOP").exists():
                        reason = "user_pause"
                    elif free < 30:
                        reason = "disk_reserve"
                    elif elapsed > args.hours * 3600:
                        reason = "wall_time_budget"
                    if reason and pending is None:
                        pending = (reason, cp["next_frame"] if cp else -1)
                        status["pause_pending"] = reason
                    # Stop immediately for low disk; otherwise wait for a new committed frame.
                    if pending and cp and (pending[0] == "disk_reserve" or cp["next_frame"] > pending[1]):
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=60)
                        status.update(state="paused", reason=pending[0], checkpoint=cp)
                        atomic_json(folder / "status.json", status)
                        print("PAUSED", status, flush=True)
                        return
                    monitor.write(f"{elapsed:.3f},{sim_time:.9f},{cp['time'] if cp else 0:.9f},{current},{peak},{free:.3f}\n")
                    atomic_json(folder / "status.json", status)
                    next_report = elapsed + 5
                time.sleep(.1)
        if child.returncode:
            status.update(state="failed", returncode=child.returncode, checkpoint=checkpoint(folder))
            atomic_json(folder / "status.json", status)
            raise RuntimeError("Simulation failed; inspect simulation.log. Last checkpoint preserved.")
    elapsed = time.monotonic() - start
    atomic_json(folder / "pipeline_times.json", {"simulate_this_session": elapsed})
    status.update(state="simulation_complete", checkpoint=checkpoint(folder), simulation_seconds=config["duration"],
                  linear_remaining_hours=0, elapsed_hours=elapsed / 3600)
    atomic_json(folder / "status.json", status)
    streamed_mesh_complete = False
    if mesh_child is not None:
        status.update(state="finishing_stream_mesh")
        atomic_json(folder / "status.json", status)
        while mesh_child.poll() is None:
            if pause_requested or (folder / "STOP").exists():
                status.update(state="postprocess_paused", next_stage="mesh")
                atomic_json(folder / "status.json", status)
                return
            time.sleep(1)
        if mesh_continuation is not None and mesh_continuation_error is None:
            try:
                mesh_continuation.poll()
                status["surface_resume"].update(verified=mesh_continuation.verified,
                                                last_published_frame=mesh_continuation.last_published)
            except Exception as error:
                mesh_continuation_error = str(error)
        if mesh_continuation_error is not None:
            status.update(state="postprocess_failed", failed_stage="mesh_resume",
                          mesh_continuation_error=mesh_continuation_error)
            atomic_json(folder / "status.json", status)
            return
        if mesh_child.returncode:
            status.update(state="postprocess_failed", failed_stage="mesh", returncode=mesh_child.returncode)
            atomic_json(folder / "status.json", status)
            return
        status["mesher_returncode"] = mesh_child.returncode
        with (folder / "surface/meshing.csv").open() as mesh_report:
            mesh_rows = list(csv.DictReader(mesh_report))
        expected_frames = list(range(round(config["duration"] * config["fps"]) + 1))
        if [int(row["frame"]) for row in mesh_rows] != expected_frames:
            raise RuntimeError("Stream mesher did not produce every frame")
        # This wall time includes waiting for the solver. Per-frame CPU costs
        # remain separately recorded in meshing.csv; do not add both to wall time.
        timing_path = folder / "pipeline_times.json"
        timing = json.loads(timing_path.read_text())
        timing["mesh_stream_wall_including_wait"] = time.monotonic() - mesh_started
        timing["mesh_cpu_stage_sum"] = sum(sum(float(row[key]) for key in
            ("read_ms", "pca_ms", "grid_ms", "clip_ms", "filter_ms", "contour_ms", "write_ms"))
            for row in mesh_rows) / 1000
        if mesh_continuation is not None:
            timing["mesh_resume_warmup_cpu_seconds"] = mesh_continuation.warmup_seconds()
        atomic_json(timing_path, timing)
        streamed_mesh_complete = True
    if args.postprocess:
        # Validate before rendering. Postprocessing is separate from the simulation time budget.
        env = os.environ | {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "2"}
        for stage in ("mesh", "validate", "render"):
            if stage == "mesh" and streamed_mesh_complete:
                continue
            if (folder / "STOP").exists():
                status.update(state="postprocess_paused", next_stage=stage)
                atomic_json(folder / "status.json", status)
                return
            status.update(state=stage)
            atomic_json(folder / "status.json", status)
            with (folder / (stage + ".log")).open("a") as log:
                result = subprocess.run([sys.executable, str(HERE / "pipeline.py"), str(folder / "pipeline_config.json"),
                                         "--output", str(folder), "--stages", stage], stdout=log, stderr=subprocess.STDOUT, env=env)
            if result.returncode:
                status.update(state="postprocess_failed", failed_stage=stage, returncode=result.returncode)
                atomic_json(folder / "status.json", status)
                return
    status.update(state="complete", total_hours=(time.monotonic() - start) / 3600)
    atomic_json(folder / "status.json", status)
    print("COMPLETE", status, flush=True)


if __name__ == "__main__":
    main()
