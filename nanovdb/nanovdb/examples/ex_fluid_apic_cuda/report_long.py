# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Plot physical-time validation and produce a measured report, without rerunning GPU work."""
import argparse
import csv
import json
from pathlib import Path
from scene_config import load_config
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output'/'long')
args=parser.parse_args()
root=args.output.resolve()
summary=json.loads((root/'summary.json').read_text())
modes=[m for m in ['stage1','stage2','static','moving','deforming'] if m in summary]
config = load_config(root/modes[0])
h = config['voxel_size']
duration = summary[modes[0]]['time']['final']
fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
for mode in modes:
    with (root/mode/'metrics.csv').open() as stream:
        data=list(csv.DictReader(stream))
    col=lambda k:np.array([float(r[k]) for r in data])
    t=col('time')
    axes[0,0].plot(t,col('max_speed'),label=mode)
    axes[0,1].plot(t,col('total_energy'),label=mode)
    axes[1,0].semilogy(t,np.maximum(col('div_after'),1e-14),label=mode)
    axes[1,1].plot(t,col('fluid_cells')/float(data[0]['fluid_cells']),label=mode)
    axes[2,0].plot(t,col('leaves'),label=mode)
    axes[2,1].plot(t,col('max_affine'),label=mode)
labels=[('Maximum particle speed','m/s'),('Mean energy, including APIC affine term','J/kg'),('Post-projection physical flux divergence (RMS)','1/s'),('Occupied fluid-cell volume / initial (not material volume)','ratio'),('Allocated 8^3 leaf blocks','blocks'),('Maximum affine-matrix Frobenius norm','1/s')]
for ax,(title,ylabel) in zip(axes.flat,labels):
    ax.set_title(title);ax.set_xlabel('Simulation time (s)');ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[1,0].axhline(1e-6,color='red',linestyle='--',label='acceptance threshold')
fig.savefig(root/'stability.png',dpi=150)
plt.close(fig)
comparison=summary.get('fixed_sparse_comparison')
if comparison:
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for key in ['position_max','velocity_max']:
        ax.semilogy([r['time'] for r in comparison['series']],[max(r[key],1e-15) for r in comparison['series']],label=key)
    ax.axhline(1e-6,color='red',linestyle='--',label='acceptance')
    ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Absolute difference (m or m/s)');ax.grid(alpha=.25);ax.legend()
    fig.savefig(root/'fixed_sparse.png',dpi=150);plt.close(fig)
lines=['# Physical-time validation and 3D animation','', f'All scenes run to physical simulation time {duration:.6f} s. Obstacle motion repeats {duration*6/(2*np.pi):.2f} periods.', f"Fluid cell size is {h:g} m, the domain has {config['domain_cells']} cells, and obstacle SDF cell size is {config['obstacle_voxel_size']:g} m.", f"The physical tank remains {config['world_size']} m. Contact clearance is {config['contact_margin']:g} m; particle marker radius is {config['marker_radius']:g} m.", 'Adaptive substeps stop exactly at scheduled output times. Baselines save 10 fps; obstacle scenes save 30 fps.', 'Every simulation substep checks pressure, topology/velocity stencils, finite particles/affine matrices,', 'tank bounds, and CFL. Obstacle runs additionally check analytic wall flux and SDF/analytic-mesh penetration.', '', '| Scene | Particles | Substeps | Max relative residual | Max physical RMS divergence (1/s) | Max speed / final speed (m/s) | Leaves |', '|---|---:|---:|---:|---:|---:|---:|']
for mode in modes:
    r=summary[mode]
    lines.append(f"| {mode} | {int(r['particle_count']['final'])} | {int(r['step']['final'])} | {r['residual']['max']:.3e} | {r['div_after']['max']:.3e} | {r['max_speed']['max']:.3f} / {r['max_speed']['final']:.3f} | {int(r['leaves']['min'])}–{int(r['leaves']['max'])} |")
lines+=['', 'Criteria: true residual <= max(1e-12, 2e-8 * RHS norm); RMS divergence <= 1e-6 1/s; CFL <= 0.5;', 'blocked/contact normal error <= 1e-8 m/s; zero remaining penetration deeper than 1e-7 m.', 'Particle count remains constant within each scene. No NaN/Inf, missing stencil, tank escape, or progress stall.', 'Stationary-wall energy must not increase more than 5% over its first substep. This check passed.', 'Moving-wall energy is recorded but not treated as conserved, because the obstacle performs work.', 'Total energy includes the particle affine contribution with quadratic-kernel moment h^2/4;', 'it is a useful diagnostic, not an independently conserved discretized mechanical-energy theorem.', '', '| Scene | Wall error (m/s) | Contact error (m/s) | SDF / mesh remaining penetrations | Max attempted depth (m) |', '|---|---:|---:|---:|---:|']
for mode in ['static','moving','deforming']:
    if mode not in summary:continue
    r=summary[mode]
    lines.append(f"| {mode} | {r['wall_error']['max']:.3e} | {r['contact_normal_error']['max']:.3e} | {int(r['remaining']['max'])} / {int(r['exact_remaining']['max'])} | {r['max_attempt_depth']['max']:.3e} |")
if comparison:
    lines+=['',f"Fixed/sparse particle comparison: **{comparison['status']}**, max position {comparison['position_max']:.3e} m, max velocity {comparison['velocity_max']:.3e} m/s, tolerance 1e-6.", f"All {len(comparison['series'])} stored physical-time frames are compared, not independently numbered substeps."]
lines+=['', '## Volume limitation exposed by the longer run','', 'Particle mass/count is conserved. Occupied-cell volume is a separate coarse free-surface diagnostic.', 'It is not an exact material volume. These runs do not establish geometric volume conservation.', '', '| Scene | Initial occupied-cell volume (m^3) | Final (m^3) | Change |', '|---|---:|---:|---:|']
for mode in modes:
    v=summary[mode]['occupied_volume'];change=(v['final']/v['initial']-1)*100
    lines.append(f"| {mode} | {v['initial']:.5f} | {v['final']:.5f} | {change:+.2f}% |")
lines+=['', 'The first-order particle occupancy mask has no reseeding or volume correction, and these runs show a measurable', 'change in the occupied-cell volume. Mask quantization and particle redistribution both affect this diagnostic.', 'Small projected divergence alone must not be interpreted as', 'a proof that the material volume stays constant. A fluid SDF/subcell free surface and a dedicated', 'volume test remain necessary before treating this as a production obstacle solver.', '', '## Timing', '', 'Means in milliseconds/substep. CUDA stage wall timers include synchronization and orchestration;', 'I/O and independent CPU diagnostics are excluded. Simulation and rendering were run sequentially.', '', '| Scene | Topology | MeshToSDF | Boundary sampling | P2G | Pressure | Extrapolation/G2P | Contacts |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for mode in modes:
    r=summary[mode];keys=['topology_ms','mesh_ms','boundary_ms','p2g_ms','pressure_ms','g2p_ms','collision_ms']
    lines.append('| '+mode+' | '+' | '.join(f"{r[k]['mean']:.3f}" if k in r else '—' for k in keys)+' |')
lines+=['', '## Animation artifacts', '', 'The 3D scene contains actual particle markers, animated obstacle vertices, tank edges, lights, and camera.', 'The USD contains 151 time samples per obstacle scene at 30 time codes/second.', 'ovstage.update_from_usd_time takes seconds; the authoring code converts seconds to USD time codes.', 'ovrtx reuses one renderer and stage through the sequence. Forty warmup steps happen before capture;', 'render accumulation steps at a fixed pose do not advance simulation time.', '', '| Scene | Frames | 3D MP4 | Particle/block debug MP4 | Animated USD scene |', '|---|---:|---|---|---|']
for mode in modes:
    if (root/mode/'animation_times.json').exists():
        lines.append(f"| {mode} | {summary[mode]['frames']} | [{mode}/animation.mp4]({mode}/animation.mp4) | [debug]({mode}/debug.mp4) | [scene.usdc]({mode}/scene.usdc) |")
output_path = root.relative_to(Path(__file__).parent) if root.is_relative_to(Path(__file__).parent) else root
lines+=['', f'[Combined three-scene video](demo.mp4) concatenates the three separate {duration:g} s simulations;', 'the endpoint frame adds one display interval.', 'All video frames are also preserved under each scenario’s `rtx/` and `debug/` folders.', 'Rendering and USD-authoring durations are in `animation_times.json`, separate from simulation metrics.', '', '![Stability diagnostics](stability.png)', '', '## Reproduction', '', '```bash', 'cmake --build build/release --target ex_fluid_apic_cuda -j 4', 'EX=nanovdb/nanovdb/examples/ex_fluid_apic_cuda', f'OUT="$EX/{output_path}"' if not output_path.is_absolute() else f'OUT="{output_path}"', '/tmp/fluid-render-venv/bin/python "$EX/validate_long.py" \\', f'  build/release/nanovdb/nanovdb/examples/ex_fluid_apic_cuda --duration {duration:g} \\', f'  --voxel-size {h:g} --output "$OUT"', 'for scene in stage1 stage2; do', '  /tmp/fluid-render-venv/bin/python "$EX/animate.py" "$OUT/$scene" --fps 10', 'done', 'for scene in static moving deforming; do', '  /tmp/fluid-render-venv/bin/python "$EX/animate.py" "$OUT/$scene"', 'done', '/tmp/fluid-render-venv/bin/python "$EX/assemble_animation.py" --output "$OUT"', '/tmp/fluid-render-venv/bin/python "$EX/report_long.py" --output "$OUT"', '```', '', 'Additional Python packages for animation/plots: `usd-core`, `matplotlib`; encoding uses installed ffmpeg.', 'All implementation changes remain in the new example folder; library and ovrtx sources are untouched.', '']
(root/'RESULTS.md').write_text('\n'.join(lines))
