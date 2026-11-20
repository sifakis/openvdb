# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Measure a short, isolated checkpoint continuation through the final movie.

All outputs go into a NEW directory. The source cache is read-only. Meshing
starts with empty covariance history: report the first frame as cold startup,
and compare later frame costs with the production cache's warm history.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from overnight_run import GpuMemory, checkpoint, simulation_command

HERE = Path(__file__).resolve().parent
EX = HERE.parent
BINS = HERE.parents[4] / 'build/release/nanovdb/nanovdb/examples'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--python', type=Path, default=EX / 'output/render-venv/bin/python')
    parser.add_argument('--simulation-only', action='store_true', help='Profile simulation without meshing/rendering')
    parser.add_argument('--binary', type=Path, default=BINS / 'ex_fluid_apic_scene_float_cuda')
    args = parser.parse_args()
    if args.frames < 2:
        parser.error('--frames must be at least 2 to include movie encoding')
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    cache = out / 'cache'
    cache.mkdir()
    shutil.copyfile(args.checkpoint, cache / 'checkpoint.bin')
    cp = checkpoint(cache)
    if cp is None:
        raise RuntimeError('Invalid checkpoint')
    first, last = cp['next_frame'], cp['next_frame'] + args.frames - 1
    config = json.loads((args.cache / 'pipeline_config.json').read_text())
    original_config = dict(config)
    config['duration'] = last / config['fps']
    (cache / 'pipeline_config.json').write_text(json.dumps(config, indent=2))
    for name in ('metrics.csv', 'frames.csv'):
        with (args.cache / name).open() as stream:
            header = stream.readline()
        if name == 'metrics.csv':
            header = header.split(',meshtosdf_ms', 1)[0].rstrip('\n') + '\n'
        if name == 'metrics.csv' and args.simulation_only:
            header = header.rstrip('\n') + ',meshtosdf_ms,mesh_prepare_ms,fluid_active_voxels,obstacle_active_voxels,fluid_pressure_cells,profile_substep_ms,profile_cache_transfer_ms\n'
        (cache / name).write_text(header)
    original_paths = [args.checkpoint, args.cache / 'settings.json', args.cache / 'metrics.csv',
                      args.cache / 'frames.csv', EX.parents[1] / 'tools/cuda/MeshToSDF.cuh']
    original_hashes = {str(p.resolve()): digest(p) for p in original_paths}
    gpu = GpuMemory()
    report = {'state': 'running', 'source_cache': str(args.cache.resolve()),
              'checkpoint': cp, 'first_frame': first, 'last_frame': last,
              'configuration': config, 'source_configuration': original_config,
              'sample_interval_seconds': .1, 'stages': {}, 'source_hashes': original_hashes,
              'meshing_history': 'cold at first benchmark frame; following frames accumulate history',
              'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,power.limit',
                  '--format=csv,noheader'], text=True).strip(),
              'cpu': subprocess.check_output(['lscpu'], text=True),
              'simulation_binary': str(args.binary.resolve()),
              'binary_sha256': {'ex_fluid_apic_scene_float_cuda': digest(args.binary)}}
    if not args.simulation_only:
        report['binary_sha256']['ex_fluid_apic_surface'] = digest(BINS / 'ex_fluid_apic_surface')

    def save():
        temporary = out / 'benchmark.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(out / 'benchmark.json')

    def run_stage(name, command):
        command = list(map(str, command))
        info = {'command': command, 'sampled_gpu_peak_mib': gpu.sample(), 'started': time.time()}
        report['stages'][name] = info
        report['current_stage'] = name
        save()
        start = time.perf_counter()
        with (out / (name + '.log')).open('w') as log, (out / (name + '_memory.csv')).open('w') as samples:
            writer = csv.writer(samples)
            writer.writerow(('elapsed_seconds', 'system_gpu_used_mib'))
            proc = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(out / (name + '_resources.txt')),
                                     *command], stdout=log, stderr=subprocess.STDOUT)
            info['pid'] = proc.pid
            save()
            while proc.poll() is None:
                used = gpu.sample()
                if used is not None:
                    info['sampled_gpu_peak_mib'] = max(info['sampled_gpu_peak_mib'] or 0, used)
                writer.writerow((time.perf_counter() - start, used))
                samples.flush()
                time.sleep(.1)
            info['returncode'] = proc.returncode
        info['wall_seconds'] = time.perf_counter() - start
        save()
        if proc.returncode:
            raise RuntimeError(f'{name} failed: see {out / (name + ".log")}')
        print(name, info['wall_seconds'], flush=True)

    try:
        command = simulation_command(config, cache, args.binary, True)
        if args.simulation_only:
            command += ['--profile', '1']
        run_stage('simulate', command)
        report['barrier_signing'] = json.loads((cache / 'settings.json').read_text()).get('barrier_signing', 'unrecorded')
        if not args.simulation_only:
            run_stage('mesh', [BINS / 'ex_fluid_apic_surface', cache, cache / 'surface', config['mesh_h'],
                      config['h'] ** 3 / config['ppc'], config['obstacle_h'], config['mesh_isovalue'],
                      config['mesh_support'], config['mesh_smoothing'], first, last,
                      int(not config['drain_enabled']), int(config.get('mesh_floor_boundary', False)),
                      0, config.get('mesh_closing_radius', 0), config.get('mesh_pca_threads', 4)])
            run_stage('render', [args.python, HERE / 'render_scene.py', cache, '--character', config['character'],
                      '--width', config['width'], '--height', config['height'], '--samples', config['samples'],
                      '--tag', 'benchmark', '--foam-render-scale', .7, '--spray-render-scale', .8])
            run_stage('decode', ['ffmpeg', '-v', 'error', '-i', cache / 'dance_water_benchmark.mp4', '-f', 'null', '-'])
        report['source_hashes_unchanged'] = all(digest(Path(p)) == h for p, h in original_hashes.items())
        if not report['source_hashes_unchanged']:
            raise RuntimeError('Source data changed during benchmark')
        report['state'] = 'complete'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
