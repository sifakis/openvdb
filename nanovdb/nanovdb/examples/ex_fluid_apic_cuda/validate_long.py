# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Physical-time stability tests. Output times, not substep counts, define comparisons."""
import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from scene_config import load_config


def rows(path):
    with path.open() as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def check(folder, duration, fps):
    config = load_config(folder)
    h = config['voxel_size']
    data = rows(folder/'metrics.csv')
    frames = rows(folder/'frames.csv')
    assert data and abs(data[-1]['time']-duration) < 1e-9, 'did not reach requested physical time'
    assert abs(frames[-1]['time']-duration) < 1e-9
    assert all(math.isfinite(v) for r in data for v in r.values()), 'nonfinite diagnostic'
    assert all(r['residual_absolute'] <= max(1e-12, 2e-8*r['rhs_norm'])
               if 'residual_absolute' in r else r['residual'] <= 2e-8 for r in data)
    assert max(r['div_after'] for r in data) <= 1e-6
    assert max(r['missing'] + r['tank_remaining'] for r in data) == 0
    assert max(r['max_speed']*r['dt']/h for r in data) <= .5
    if 'cfl' in data[0]:
        assert max(r['cfl'] for r in data) <= .5
    assert len({r['particle_count'] for r in data}) == 1
    assert len({r['particles'] for r in frames}) == 1
    assert all(abs(r['time']-min(i/fps,duration)) < 1e-9 for i,r in enumerate(frames))
    assert all((folder/f"frame_{int(r['frame']):04d}.csv").exists() for r in frames)
    if 'remaining' in data[0]:
        assert max(r['remaining']+r['exact_remaining'] for r in data) == 0
        assert max(r['physical_divergence'] for r in data) <= 1e-6
        assert max(r['wall_error'] for r in data) <= 1e-8
        assert max(r['contact_normal_error'] for r in data) <= 1e-8
        assert sum(r['contacts'] for r in data) > 0
    if folder.name in ('stage1','stage2','static'):
        assert max(r['total_energy'] for r in data) <= data[0]['total_energy']*1.05, 'stationary-wall energy increased more than 5%'
    entry = {}
    for k in data[0]:
        values = [r[k] for r in data]
        entry[k] = dict(min=min(values),max=max(values),mean=statistics.mean(values),final=values[-1])
    entry['frames'] = len(frames)
    entry['scene_config'] = config
    entry['occupied_volume'] = dict(initial=data[0]['fluid_cells']*h**3,final=data[-1]['fluid_cells']*h**3)
    # Fixed mass/particle count is not a proof of incompressible geometric volume preservation.
    entry['final_second'] = {k:statistics.mean(r[k] for r in data if r['time'] >= duration-1) for k in ('max_speed','mean_speed','total_energy','kinetic_energy')}
    entry['status'] = 'PASS'
    return entry


def compare(root):
    a, b = root/'stage1', root/'stage2'
    if not (a/'frames.csv').exists() or not (b/'frames.csv').exists():
        return None
    fa,fb=rows(a/'frames.csv'),rows(b/'frames.csv')
    assert len(fa)==len(fb)
    import numpy as np
    series=[]
    for x,y in zip(fa,fb):
        assert abs(x['time']-y['time'])<1e-9
        name=f"frame_{int(x['frame']):04d}.csv"
        p=np.loadtxt(a/name,delimiter=',',skiprows=1);q=np.loadtxt(b/name,delimiter=',',skiprows=1)
        assert p.shape==q.shape
        diff=np.abs(p-q)
        series.append(dict(time=x['time'],position_max=float(diff[:,:3].max()),velocity_max=float(diff[:,3:].max()),position_rms=float(np.sqrt(np.mean(diff[:,:3]**2))),velocity_rms=float(np.sqrt(np.mean(diff[:,3:]**2)))))
    result=dict(series=series,position_max=max(r['position_max'] for r in series),velocity_max=max(r['velocity_max'] for r in series),threshold=1e-6)
    result['status']='PASS' if max(result['position_max'],result['velocity_max'])<=1e-6 else 'DIFFERENT'
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('executable',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output'/'long')
    parser.add_argument('--duration',type=float,default=5)
    parser.add_argument('--voxel-size',type=float,default=.05)
    parser.add_argument('--modes',nargs='+',default=['stage1','stage2','static','moving','deforming'])
    parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result={}
    for mode in args.modes:
        fps=10 if mode in ('stage1','stage2') else 30
        folder=args.output/mode
        finished=folder.exists() and (folder/'frames.csv').exists() and abs(rows(folder/'frames.csv')[-1]['time']-args.duration)<1e-9 and abs(load_config(folder)['voxel_size']-args.voxel_size)<1e-12
        if not args.summarize_only and not(args.reuse and finished):
            print('Running',mode,'for',args.duration,'seconds',flush=True)
            started=time.perf_counter()
            with (args.output/(mode+'.log')).open('w') as log:
                subprocess.run([str(args.executable.resolve()),mode,str(folder),'--duration',str(args.duration),'--fps',str(fps),'--voxel-size',str(args.voxel_size)],stdout=log,stderr=subprocess.STDOUT,check=True)
            print(mode,'wall_seconds',time.perf_counter()-started,flush=True)
        result[mode]=check(folder,args.duration,fps)
        assert abs(result[mode]['scene_config']['voxel_size']-args.voxel_size)<1e-12, 'stored resolution differs from requested resolution'
        print(mode,result[mode]['status'],'steps',result[mode]['step']['final'],'dt_min',result[mode]['dt']['min'],flush=True)
    result['fixed_sparse_comparison']=compare(args.output)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['fixed_sparse_comparison']:
        print('fixed/sparse:',{k:v for k,v in result['fixed_sparse_comparison'].items() if k!='series'},flush=True)
        assert result['fixed_sparse_comparison']['status']=='PASS', 'fixed/sparse equivalence failed'
    print('Completed physical-time validation',flush=True)
