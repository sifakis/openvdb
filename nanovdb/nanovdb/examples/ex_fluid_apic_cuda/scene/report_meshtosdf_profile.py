# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Summarize MeshToSDF phaseMs replay and optional MeshToGrid verbose diagnosis."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

PHASES = {
    'rasterize_ms': 'MeshToGrid topology + UDF / nearest triangle',
    'partition_ms': 'Partition surfaces / initialize composition',
    'process_surfaces_ms': 'Process surfaces: barrier / components / signing',
    'resolve_signs_ms': 'Resolve signs on original grid',
    'masks_finalize_ms': 'Interior masks + final sidecar / output packing',
}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile',type=Path)
    p.add_argument('--plot',action='store_true')
    p.add_argument('--metrics',type=Path,help='Source substep metrics (overrides environment.json)')
    args=p.parse_args();out=args.profile
    env=json.loads((out/'environment.json').read_text()) if (out/'environment.json').exists() else {}
    source_path=args.metrics or env.get('source_metrics')
    if not source_path:p.error('Provide --metrics or an environment.json with source_metrics')
    source=read(Path(source_path));rows=read(out/'phases.csv')
    passes=sorted({int(r['pass']) for r in rows})
    assert len(passes)==env.get('passes',len(passes))
    for repeat in passes:
        group=[r for r in rows if int(r['pass'])==repeat]
        assert [int(r['sample']) for r in group]==list(range(len(source)))
    for row in rows:
        origin=source[int(row['sample'])]
        assert abs(float(row['time'])-float(origin['time']))<1e-10
        assert int(row['active_voxels'])==int(origin['obstacle_active_voxels'])
        assert all(math.isfinite(float(row[k])) and float(row[k])>=0 for k in PHASES)
    mean=lambda data,key: statistics.mean(float(r[key]) for r in data)
    wall=mean(rows,'api_wall_ms')
    phase={k:mean(rows,k) for k in PHASES}
    gap=wall-sum(phase.values())
    assert gap>=-1e-3
    summary=dict(primary_timing_unit='ms per geometry replay step (one MeshToSDF call)',
                 barrier_signing=rows[0].get('barrier_signing',env.get('barrier_signing','heuristic')),
                 samples=len(rows),poses=len(source),passes=len(passes),api_wall_ms=wall,
                 phase_mean_ms=phase,phase_percent_of_api_wall={k:100*v/wall for k,v in phase.items()},
                 unassigned_wall_ms=gap,unassigned_wall_percent=100*gap/wall,
                 active_counts_match_source=True,
                 active_voxel_min=min(int(r['active_voxels']) for r in rows),
                 active_voxel_max=max(int(r['active_voxels']) for r in rows),
                 pass_means=[dict(pass_index=i,api_wall_ms=mean([r for r in rows if int(r['pass'])==i],'api_wall_ms'),
                                  **{k:mean([r for r in rows if int(r['pass'])==i],k) for k in PHASES}) for i in passes])
    diagnostic=out/'raster_diagnostic.log'
    if diagnostic.exists():
        text=diagnostic.read_text()
        details=[{'phase':name,'ms':float(ms)} for name,ms in re.findall(r'^(.+?) \.\.\. completed in ([0-9.eE+-]+) milliseconds$',text,re.M)]
        summary['raster_diagnostic']=dict(phases=details,
            raster_api_wall_ms=float(re.search(r'raster_api_wall_ms=([0-9.eE+-]+)',text)[1]),
            voxel_surface_components=int(re.search(r'voxel_surface_components=(\d+)',text)[1]),
            triangle_components=int(re.search(r'triangle_components=(\d+)',text)[1]),
            scope='Single first-pose verbose diagnostic; extra event synchronizations; not the 1260-call average')
        summary['process_surface_diagnostic']={key:float(value) for key,value in
            re.findall(r'^DIAGNOSTIC (prune_barrier_ms|nonbarrier_components_ms|sign_nonbarrier_ms|inject_sign_ms|heuristic_barrier_sign_ms|interior_barrier_sign_ms|propagate_masks_ms)=([0-9.eE+-]+)$',text,re.M)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'phase_summary.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['phase','mean_ms','percent_of_api_wall'])
        for key,value in phase.items():writer.writerow([key,value,100*value/wall])
        writer.writerow(['unassigned_wall',gap,100*gap/wall])
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        labels=list(PHASES.values())+['Event boundaries / unassigned wall']
        values=list(phase.values())+[gap]
        axes[0].barh(labels,values,color=['#d95f02','#437fa3','#d95f02','#437fa3','#437fa3','#aaa'])
        axes[0].invert_yaxis();axes[0].set_xlabel('ms / replay step (one MeshToSDF call)')
        axes[0].set_title(f'{len(source)} poses x {len(passes)} passes; API wall {wall:.2f} ms')
        for i,v in enumerate(values):axes[0].text(v,i,f' {v:.3f} ({100*v/wall:.1f}%)',va='center',fontsize=8)
        axes[0].set_xlim(0,max(values)*1.35)
        detail=summary.get('raster_diagnostic',{}).get('phases',[])
        largest=sorted(detail,key=lambda r:r['ms'],reverse=True)[:6]
        axes[1].barh([r['phase'] for r in largest],[r['ms'] for r in largest],color='#437fa3')
        axes[1].invert_yaxis();axes[1].set_xlabel('ms / call (separate verbose diagnostic)')
        axes[1].set_title('Rasterization detail: one pose\nAdditional synchronization; separate diagnostic')
        for i,r in enumerate(largest):axes[1].text(r['ms'],i,f" {r['ms']:.3f}",va='center',fontsize=8)
        axes[1].set_xlim(0,max(r['ms'] for r in largest)*1.2)
        for suffix in ('png','svg'):fig.savefig(out/('meshtosdf_breakdown.'+suffix),dpi=160)
        plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
