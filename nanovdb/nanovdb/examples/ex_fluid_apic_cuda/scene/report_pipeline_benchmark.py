# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Aggregate measured pipeline stages without double-counting MG or overlaps."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess

SIM = ['topology_ms', 'mesh_sdf_ms', 'boundary_ms', 'p2g_ms', 'pressure_ms',
       'g2p_ms', 'contact_ms', 'whitewater_ms', 'transfer_ms', 'io_ms']
MESH = ['read_ms', 'pca_ms', 'grid_ms', 'clip_ms', 'filter_ms', 'contour_ms', 'write_ms']


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def mean(values):
    return statistics.mean(values) if values else 0


def summarize_metrics(data, fps):
    grouped = {}
    for row in data:
        frame = math.ceil(float(row['time']) * fps - 1e-8)
        grouped.setdefault(frame, []).append(row)
    frames = []
    for frame, values in grouped.items():
        frames.append({'frame': frame, 'substeps': len(values),
                       **{key: sum(float(v[key]) for v in values) / 1000 for key in SIM}})
    return {'frames': frames, 'substeps': len(data),
            'phase_ms_per_step': {k:sum(float(v[k]) for v in data)/len(data) for k in SIM},
            'substeps_per_frame': mean([v['substeps'] for v in frames]),
            'phase_seconds_per_frame': {k: mean([v[k] for v in frames]) for k in SIM},
            'phase_p95_seconds_per_frame': {k: sorted(v[k] for v in frames)[math.ceil(.95 * len(frames)) - 1] for k in SIM},
            'phase_max_seconds_per_frame': {k: max(v[k] for v in frames) for k in SIM},
            'mean_dt': mean([float(v['dt']) for v in data]),
            'mean_pressure_iterations': mean([float(v['pressure_iterations']) for v in data]),
            'mg_setup_seconds_per_frame': sum(float(v['mg_setup_ms']) for v in data) / 1000 / len(frames),
            'mg_solve_seconds_per_frame': sum(float(v['mg_solve_ms']) for v in data) / 1000 / len(frames)}


def resources(path):
    result = {}
    for line in path.read_text().splitlines():
        for key in ('Maximum resident set size (kbytes)', 'User time (seconds)', 'System time (seconds)'):
            if key + ':' in line:
                result[key] = float(line.split(key + ':')[1].strip())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark', type=Path)
    args = p.parse_args()
    out = args.benchmark
    run = json.loads((out / 'benchmark.json').read_text())
    if run['state'] != 'complete':
        raise RuntimeError('Wait for the benchmark to finish')
    cache = out / 'cache'
    source = Path(run['source_cache'])
    fps = run['configuration']['fps']
    metrics = rows(cache / 'metrics.csv')
    old_metrics = rows(source / 'metrics.csv')
    mesh = rows(cache / 'surface/meshing.csv')
    frames = rows(cache / 'frames.csv')
    render = json.loads((cache / 'render_benchmark/timings.json').read_text())
    count = run['last_frame'] - run['first_frame'] + 1
    measured = summarize_metrics(metrics, fps)
    historical = summarize_metrics(old_metrics, fps)
    original_same_window = summarize_metrics([r for r in old_metrics
        if run['checkpoint']['time'] + 1e-10 < float(r['time']) <= run['configuration']['duration'] + 1e-10], fps)
    measured['mesh_phase_seconds_per_frame'] = {k: mean([float(v[k]) for v in mesh]) / 1000 for k in MESH}
    measured['mesh_per_frame'] = [{'frame': int(v['frame']), **{k: float(v[k]) / 1000 for k in MESH}} for v in mesh]
    measured['resources'] = {name: resources(out / (name + '_resources.txt')) for name in run['stages']}
    measured['wall_seconds_per_frame'] = {k: v['wall_seconds'] / count for k, v in run['stages'].items()}
    measured['simulation_unattributed_seconds_per_frame'] = measured['wall_seconds_per_frame']['simulate'] - sum(measured['phase_seconds_per_frame'].values())
    measured['meshing_unattributed_seconds_per_frame'] = measured['wall_seconds_per_frame']['mesh'] - sum(measured['mesh_phase_seconds_per_frame'].values())
    measured['render'] = render
    checks = {
        'frame_range': [int(f['frame']) for f in frames] == list(range(run['first_frame'], run['last_frame'] + 1)),
        'no_fluid_penetration': all(float(v['penetrations']) == 0 and float(v['penetration_depth']) <= 2e-5 for v in metrics),
        'no_secondary_penetration': all(float(v['ww_depth']) <= 2e-5 for v in metrics),
        'no_missing_stencils': all(float(v['missing']) == 0 for v in metrics),
        'mass_ledger': all(float(v['ledger_error']) == 0 for v in metrics),
        'pressure': all(float(v['residual_absolute']) <= max(1e-12, 1e-4 * float(v['rhs_norm'])) for v in metrics),
        'flux_divergence': all(float(v['flux_div_after']) <= 2e-3 for v in metrics),
        'wall_flux': all(float(v['wall_flux_error']) <= 2e-5 for v in metrics),
        'contact_normal': all(float(v['contact_normal_error']) <= 2e-5 for v in metrics),
        'cfl': all(float(v['cfl']) <= .5 for v in metrics),
        'render_converged': len(render['frame_convergence']) == count and all(v['converged'] and v['progression'] >= 96 for v in render['frame_convergence']),
        'source_unchanged': run['source_hashes_unchanged'],
        'decode': run['stages']['decode']['returncode'] == 0,
    }
    video = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames', '-of', 'json',
        str(cache / 'dance_water_benchmark.mp4')]))['streams'][0]
    checks['encoded_frame_count'] = int(video['nb_read_frames']) == count
    measured['validation'] = {'pass': all(checks.values()), 'checks': checks,
        'maxima': {k: max(float(v[k]) for v in metrics) for k in ('flux_div_after','wall_flux_error','contact_normal_error','residual','cfl')},
        'particle_range': [min(int(v['particles']) for v in metrics), max(int(v['particles']) for v in metrics)],
        'video': video}
    warm_source_mesh = [v for v in rows(source / 'surface/meshing.csv') if int(v['frame']) >= 230]
    historical['four_thread_warm_mesh_phase_seconds_per_frame'] = {k: mean([float(v[k]) for v in warm_source_mesh]) / 1000 for k in MESH}
    historical['four_thread_warm_mesh_frame_range'] = [230, 300]
    result = {'measured': measured, 'historical_full_trajectory': historical,
              'historical_same_window': original_same_window,
              'notes': ['Phase timers are synchronized wall times, not kernel hardware counters.',
                        'MG setup/solve are subsets of pressure; do not sum them twice.',
                        'Sequential benchmark; do not add historical overlapping meshing wall times.',
                        'Meshing starts cold and only accumulates history within this short benchmark.',
                        'NVML at 100 ms is sampled total GPU memory, not allocation-exact peak.',
                        'Render load is amortized over three frames here; the full movie has 301 frames.',
                        'Mesh read includes CSV parsing; write includes foam projection, normals and export.']}
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    with (out / 'frame_costs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame', 'substeps', *SIM])
        writer.writeheader(); writer.writerows(measured['frames'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    entries = [(measured['wall_seconds_per_frame'], ('simulate','mesh','render'), ('Simulation + cache','CPU meshing + export','USD + render + encode')),
               (measured['phase_ms_per_step'], SIM, ('Sparse topology','Character + MeshToSDF','Obstacle boundaries','P2G + classification','Pressure / MGPCG','Extrapolation + G2P','Particle contacts','Whitewater','Transfer / management','CSV + checkpoint')),
               (measured['mesh_phase_seconds_per_frame'], MESH, ('Read / CSV parse','Neighbors + PCA','Scalar grid','Solid clipping','SDF filter','VolumeToMesh','Foam / mesh export'))]
    for ax,(values,keys,labels),title in zip(axes,entries,('Full pipeline','Simulation phases','CPU surface reconstruction')):
        pairs=sorted([(values[k],label) for k,label in zip(keys,labels)])
        bars=ax.barh([x[1] for x in pairs],[x[0] for x in pairs],color='#397ca8')
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        ax.set_xlim(0,max(x[0] for x in pairs)*1.2)
        ax.set_title(title)
        ax.set_xlabel('Mean ms / simulation step' if title=='Simulation phases' else 'Wall seconds / output frame (reference)')
        ax.grid(axis='x',alpha=.2)
    h = run['configuration']['h']
    fig.suptitle(f"{round(3.2/h)} x {round(3.2/h)} x {round(2.4/h)} | {run['configuration']['ppc']} ppc | "
                 f"{count} frames at {run['checkpoint']['time']:g}–{run['configuration']['duration']:g} s | isolated sequential benchmark")
    fig.savefig(out/'pipeline_costs.png',dpi=160)
    fig.savefig(out/'pipeline_costs.svg')
    print(json.dumps({k:v for k,v in measured.items() if k not in ('frames','render')},indent=2))
    if not all(checks.values()):
        raise RuntimeError('Benchmark validation failed')


if __name__ == '__main__':
    main()
