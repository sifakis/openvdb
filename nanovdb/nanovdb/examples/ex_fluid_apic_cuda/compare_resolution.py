# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Compare measured physical-time trajectories at two discretizations."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scene_config import load_config


def metrics(folder):
    with (folder / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low', type=Path, required=True)
    parser.add_argument('--high', type=Path, required=True)
    args = parser.parse_args()
    low = json.loads((args.low / 'summary.json').read_text())
    high = json.loads((args.high / 'summary.json').read_text())
    modes = ['stage1', 'stage2', 'static', 'moving', 'deforming']
    stage_keys = ['topology_ms', 'mesh_ms', 'boundary_ms', 'p2g_ms',
                  'pressure_ms', 'g2p_ms', 'collision_ms']
    report = {}
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for j, mode in enumerate(['static', 'moving', 'deforming']):
        for root, style in [(args.low, '--'), (args.high, '-')]:
            h = load_config(root / mode)['voxel_size']
            data = metrics(root / mode)
            axes[0, j].plot(data['time'], data['fluid_cells'] / data['fluid_cells'][0],
                            style, label=f'h = {h:g} m')
            axes[1, j].plot(data['time'], data['max_speed'], style, label=f'h = {h:g} m')
            axes[2, j].plot(data['time'], data['total_energy'], style, label=f'h = {h:g} m')
        for i, label in enumerate(['Occupied cell volume / initial', 'Max speed (m/s)',
                                   'Mean APIC energy (J/kg)']):
            axes[i, j].set_title(mode)
            axes[i, j].set_xlabel('Simulation time (s)')
            axes[i, j].set_ylabel(label)
            axes[i, j].grid(alpha=.25)
            axes[i, j].legend()
    fig.savefig(args.high / 'resolution_comparison.png', dpi=150)
    plt.close(fig)
    lines = ['# Resolution sensitivity: 0.05 m and 0.04 m', '',
             'Both resolutions use the same physical tank, dam bounds, obstacle dimensions,',
             'analytic motion, gravity, five-second duration, output schedule, and solver tolerances.',
             'The particle counts, contact clearance, and SDF cell size scale with resolution.',
             'This is a two-resolution sensitivity check; it is not a convergence proof.', '',
             '| Scene | Particles: old → new | Occupied-volume drift: old → new | Max speed: old → new (m/s) | Simulation stages: old → new (s) |',
             '|---|---:|---:|---:|---:|']
    for mode in modes:
        a, b = low[mode], high[mode]
        assert abs(a['time']['final'] - b['time']['final']) < 1e-9
        assert a['status'] == b['status'] == 'PASS'
        report[mode] = {}
        for label, summary, root in [('low', a, args.low), ('high', b, args.high)]:
            v = summary['occupied_volume']
            report[mode][label] = dict(
                voxel_size=load_config(root / mode)['voxel_size'],
                particles=int(summary['particle_count']['final']),
                occupied_volume_drift_percent=(v['final'] / v['initial'] - 1) * 100,
                occupied_volume=v,
                max_speed=summary['max_speed']['max'],
                final_second_speed=summary['final_second']['max_speed'],
                final_second_energy=summary['final_second']['total_energy'],
                simulation_stage_seconds=sum(summary[k]['mean'] for k in stage_keys
                                             if k in summary) * summary['step']['final'] / 1000,
                substeps=int(summary['step']['final']))
        x, y = report[mode]['low'], report[mode]['high']
        lines.append(f"| {mode} | {x['particles']} → {y['particles']} | "
                     f"{x['occupied_volume_drift_percent']:+.2f}% → {y['occupied_volume_drift_percent']:+.2f}% | "
                     f"{x['max_speed']:.3f} → {y['max_speed']:.3f} | "
                     f"{x['simulation_stage_seconds']:.2f} → {y['simulation_stage_seconds']:.2f} |")
    lines += ['', 'Simulation stage sums exclude file output, rendering, and independent diagnostics.',
              'The first-order occupancy mask quantizes geometric volume; conserved particle mass',
              'and small projected divergence do not establish exact material-volume conservation.',
              'Differences between these trajectories are recorded, not used as a stability pass criterion.',
              'Fixed/sparse equivalence is tested independently within each resolution.', '',
              '![Resolution sensitivity](resolution_comparison.png)', '']
    (args.high / 'resolution_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.high / 'RESOLUTION_COMPARISON.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    main()
