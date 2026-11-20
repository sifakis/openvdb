# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Report simulation costs in ms per simulation step (substep), with exact active counts."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

PHASES = {
    'meshtosdf_ms': 'MeshToSDF getHandle pipeline',
    'mesh_prepare_ms': 'Character evaluation / BVH / upload / converter setup',
    'topology_ms': 'Fluid sparse topology',
    'boundary_ms': 'Obstacle sampling / boundary data',
    'p2g_ms': 'Classification / P2G / forces',
    'pressure_ms': 'Pressure assembly / MGPCG / projection',
    'g2p_ms': 'Extrapolation / G2P / advection',
    'contact_ms': 'Pre/post particle contact / checks',
    'whitewater_ms': 'Whitewater generation / update / checks',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('--plot', action='store_true', help='Write a PNG/SVG plot (requires matplotlib)')
    args = parser.parse_args()
    out = args.benchmark
    run = json.loads((out / 'benchmark.json').read_text())
    assert run['state'] == 'complete'
    with (out / 'cache/metrics.csv').open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    with (out / 'cache/frames.csv').open() as stream:
        exported = list(csv.DictReader(stream))
    assert rows
    grouped = {}
    fps = run['configuration']['fps']
    for row in rows:
        frame = math.ceil(row['time'] * fps - 1e-8)
        grouped.setdefault(frame, []).append(row)
    assert sorted(grouped) == [int(row['frame']) for row in exported]

    def total(data, key):
        return sum(row[key] for row in data) / 1000

    def costs(data):
        step = total(data, 'profile_substep_ms')
        io = total(data, 'io_ms') + total(data, 'profile_cache_transfer_ms')
        phases = {PHASES[k]: total(data, k) for k in PHASES}
        phases['Character handle replacement / timer overhead'] = (
            total(data, 'mesh_sdf_ms') - total(data, 'meshtosdf_ms') - total(data, 'mesh_prepare_ms'))
        phases['Emitter / drain / GPU particle management'] = (
            total(data, 'transfer_ms') - total(data, 'profile_cache_transfer_ms'))
        phases['Cache download / file and checkpoint output'] = io
        phases['Other substep work / diagnostics / allocations'] = step - sum(phases.values())
        assert all(v >= -1e-5 for v in phases.values()), phases
        return step, io, phases

    step, io, phases = costs(rows)
    compute = step - io
    pipeline = total(rows, 'meshtosdf_ms')
    wall = run['stages']['simulate']['wall_seconds']
    count = len(grouped)
    checks = {
        'pressure': all(r['residual_absolute'] <= max(1e-12, 1e-4 * r['rhs_norm']) for r in rows),
        'flux_divergence': all(r['flux_div_after'] <= 2e-3 for r in rows),
        'wall_flux': all(r['wall_flux_error'] <= 2e-5 for r in rows),
        'relative_normal_velocity': all(r['contact_normal_error'] <= 2e-5 for r in rows),
        'fluid_penetration': all(r['penetrations'] == 0 and r['penetration_depth'] == 0 for r in rows),
        'secondary_penetration': all(r['ww_depth'] == 0 for r in rows),
        'stencil_coverage': all(r['missing'] == 0 for r in rows),
        'mass_ledger': all(r['ledger_error'] == 0 for r in rows),
        'cfl': all(r['cfl'] <= .5 for r in rows),
        'source_hashes': run['source_hashes_unchanged'],
        'positive_sdf_timing': all(0 < r['meshtosdf_ms'] <= r['mesh_sdf_ms'] for r in rows),
        'active_count_bounds': all(0 < r['fluid_pressure_cells'] <= r['fluid_active_voxels'] <= r['leaves'] * 512
                                   and r['obstacle_active_voxels'] > 0 for r in rows),
    }
    frame_rows = []
    for frame, data in grouped.items():
        fs, fi, fp = costs(data)
        sdf = total(data, 'meshtosdf_ms')
        row = dict(frame=frame, time=data[-1]['time'], substeps=len(data), particles=int(data[-1]['particles']),
                   substep_seconds=fs, compute_seconds=fs-fi, cache_seconds=fi, meshtosdf_seconds=sdf,
                   meshtosdf_compute_percent=100*sdf/(fs-fi), meshtosdf_substep_percent=100*sdf/fs)
        for key in ('fluid_active_voxels', 'obstacle_active_voxels', 'fluid_pressure_cells', 'leaves'):
            row[key+'_end'] = int(data[-1][key])
            row[key+'_min'] = int(min(r[key] for r in data))
            row[key+'_max'] = int(max(r[key] for r in data))
        frame_rows.append(row)
    step_rows = []
    for row in rows:
        duration, cache, _ = costs([row])
        step_rows.append(dict(
            step=int(row['step']), time=row['time'], dt=row['dt'],
            output_frame=math.ceil(row['time']*fps-1e-8), particles=int(row['particles']),
            fluid_active_voxels=int(row['fluid_active_voxels']),
            obstacle_active_voxels=int(row['obstacle_active_voxels']),
            fluid_pressure_cells=int(row['fluid_pressure_cells']),
            compute_ms=1000*(duration-cache), substep_ms=1000*duration,
            cache_ms=1000*cache, meshtosdf_ms=row['meshtosdf_ms']))
    summary = dict(
        primary_timing_unit='ms per simulation step; step = solver substep, not output frame',
        barrier_signing=run.get('barrier_signing', 'unrecorded historical binary; see source hashes'),
        compute_ms_per_step=1000*compute/len(rows),
        substep_ms_per_step=1000*step/len(rows),
        cache_ms_per_step_amortized=1000*io/len(rows),
        phase_ms_per_step={k:1000*v/len(rows) for k,v in phases.items()},
        active_count_ranges={key:dict(min=int(min(r[key] for r in rows)),
                                      max=int(max(r[key] for r in rows)))
                             for key in ('fluid_active_voxels','obstacle_active_voxels','fluid_pressure_cells')},
        pass_all=all(checks.values()), checks=checks, frames=frame_rows, substeps=len(rows),
        numerator_scope='MeshToSDF calls within the measured continuation substeps; initial restart-time SDF build is part of process startup, not this sum',
        wall_seconds=wall, substep_seconds=step, compute_seconds=compute, cache_seconds=io,
        process_startup_teardown_and_metric_flush_seconds=wall-step,
        meshtosdf_seconds=pipeline, meshtosdf_compute_percent=100*pipeline/compute,
        meshtosdf_substep_percent=100*pipeline/step, meshtosdf_process_wall_percent=100*pipeline/wall,
        meshtosdf_ms_per_substep=1000*pipeline/len(rows),
        mean_dt=statistics.mean(r['dt'] for r in rows),
        meshtosdf_per_call_ms={
            'min':min(r['meshtosdf_ms'] for r in rows),
            'median':statistics.median(r['meshtosdf_ms'] for r in rows),
            'p95':sorted(r['meshtosdf_ms'] for r in rows)[math.ceil(.95*len(rows))-1],
            'max':max(r['meshtosdf_ms'] for r in rows)},
        phase_seconds=phases, phase_seconds_per_frame={k:v/count for k,v in phases.items()},
        phase_compute_percent={k:100*v/compute for k,v in phases.items()
                               if k != 'Cache download / file and checkpoint output'},
        maxima={k:max(r[k] for r in rows) for k in (
            'particles','fluid_active_voxels','obstacle_active_voxels','fluid_pressure_cells','leaves',
            'residual','flux_div_after','wall_flux_error','contact_normal_error','cfl','mg_bytes','gpu_used_mb')})
    (out/'simulation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (out/'simulation_steps.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=step_rows[0]);writer.writeheader();writer.writerows(step_rows)
    with (out/'simulation_frames.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=frame_rows[0]);writer.writeheader();writer.writerows(frame_rows)
    lines=['# Simulation breakdown benchmark — ms per simulation step', '',
           'Barrier signing: '+summary['barrier_signing']+'.', '',
           'One step means one solver substep, not an output frame. All primary timings below are arithmetic means over measured steps.', '',
           'MeshToSDF measures synchronized wall time of `converter.getHandle()` only. '
           'Character evaluation/BVH/upload, converter construction, old-handle replacement, '
           'converter destruction, active-count readback and obstacle sampling are excluded. '
           'Numerators cover continuation substeps only; the initial restart-time SDF build belongs to process startup.', '',
           'Compute time = measured substeps minus cache download/file/checkpoint output. '
           'Substeps include timestep selection, diagnostics and allocations. '
           'Process wall also includes checkpoint load, initialization and teardown.', '',
           f'Frames: {count}; substeps: {len(rows)}; mean dt: {summary["mean_dt"]:.9g} s.', '',
           '| Phase | mean ms/step | % compute |', '|---|---:|---:|']
    for name, seconds in phases.items():
        percentage='—' if name=='Cache download / file and checkpoint output' else f'{100*seconds/compute:.3f}'
        lines.append(f'| {name} | {1000*seconds/len(rows):.6f} | {percentage} |')
    lines += ['', f'Compute: {1000*compute/len(rows):.6f} ms/step; substeps with output: {1000*step/len(rows):.6f} ms/step. '
              f'Cache output: {1000*io/len(rows):.6f} ms/step amortized (writes occur at output frames only). '
              f'Process wall total: {wall:.6f} s, including startup/teardown.',
              f'MeshToSDF: {100*pipeline/compute:.4f}% of compute, {100*pipeline/step:.4f}% of substeps '
              f'with output, {100*pipeline/wall:.4f}% of process wall.', '',
              'The following output-frame counts are reference snapshots only. simulation_steps.csv contains every simulation step.', '',
              '| Frame | Fluid active: end / max | Obstacle active: end / max | Pressure cells: end |',
              '|---:|---:|---:|---:|']
    for r in frame_rows:
        lines.append(f'| {r["frame"]} | {r["fluid_active_voxels_end"]:,} / {r["fluid_active_voxels_max"]:,} '
                     f'| {r["obstacle_active_voxels_end"]:,} / {r["obstacle_active_voxels_max"]:,} '
                     f'| {r["fluid_pressure_cells_end"]:,} |')
    lines += ['', 'Fluid active includes halo and air; pressure cells are occupied fluid unknowns. '
              'Allocated leaf capacity is a separate quantity. Counts are measured after the substep topology build.',
              '', 'Checks: '+json.dumps(checks), '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
        labels = list(phases)
        times = [1000*phases[k]/len(rows) for k in labels]
        colors = ['#d95f02' if k == 'MeshToSDF getHandle pipeline' else '#437fa3' for k in labels]
        axes[0].barh(labels, times, color=colors)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Mean ms / simulation step (solver substep)')
        axes[0].set_title(f'MeshToSDF: {100*pipeline/compute:.2f}% of simulation compute')
        for y, value in enumerate(times):
            axes[0].text(value, y, f' {value:.2f}', va='center', fontsize=8)
        x = [r['step'] for r in step_rows]
        for key, label, color in [('fluid_active_voxels', 'Fluid (including halo/air)', '#437fa3'),
                                  ('obstacle_active_voxels', 'Obstacle', '#d95f02'),
                                  ('fluid_pressure_cells', 'Fluid classification', '#5d9644')]:
            axes[1].plot(x, [r[key]/1e6 for r in step_rows], label=label, color=color)
        axes[1].set_xlabel('Simulation step ID (solver substep)')
        axes[1].ticklabel_format(axis='x', style='plain', useOffset=False)
        axes[1].set_ylabel('Actual active / classified voxels (millions)')
        axes[1].legend(fontsize=8)
        axes[1].set_title('Actual counts for every simulation step')
        fig.suptitle(f'Float MGPCG, h={run["configuration"]["h"]:.6g} m, '
                     f'{run["checkpoint"]["time"]:.3f}-{run["configuration"]["duration"]:.3f} s')
        for suffix in ('png', 'svg'):
            fig.savefig(out/('simulation_breakdown.'+suffix), dpi=160)
        plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('frames','phase_seconds','phase_seconds_per_frame')},indent=2))
    if not summary['pass_all']:
        raise RuntimeError('Numerical or measurement checks failed')


if __name__ == '__main__':
    main()
