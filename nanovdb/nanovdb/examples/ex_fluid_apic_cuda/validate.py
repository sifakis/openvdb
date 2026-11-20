# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Run stages in order, fail on numerical regressions, summarize measured costs."""
import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


def rows(path):
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def summarize(root):
    result = {}
    for stage in ('stage1', 'stage2', 'static', 'moving', 'deforming'):
        data = rows(root / stage / 'metrics.csv')
        assert data and max(r['missing'] for r in data) == 0
        assert all(r['residual_absolute'] <= max(1e-12, 2e-8*r['rhs_norm'])
                   if 'residual_absolute' in r else r['residual'] < 2e-8 for r in data)
        assert max(r['div_after'] for r in data) < 1e-6
        entry = {}
        for key in data[0]:
            values = [r[key] for r in data]
            entry[key] = {'min': min(values), 'max': max(values), 'mean': statistics.mean(values)}
        if stage in ('static', 'moving', 'deforming'):
            assert max(r['remaining'] + r['exact_remaining'] for r in data) == 0
            assert max(r['wall_error'] for r in data) < 1e-8
            assert max(r['contact_normal_error'] for r in data) < 1e-8
            assert max(r['physical_divergence'] for r in data) < 1e-6
            assert sum(r['contacts'] for r in data) > 0
            entry['total_penetration_attempts'] = sum(r['penetration_attempts'] for r in data)
            entry['total_contacts'] = sum(r['contacts'] for r in data)
        result[stage] = entry
    worst_position = worst_velocity = 0
    for path in sorted((root / 'stage1').glob('frame_*.csv')):
        if '_blocks' in path.stem:
            continue
        a, b = rows(path), rows(root / 'stage2' / path.name)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            worst_position = max(worst_position, *(abs(x[k]-y[k]) for k in ('x','y','z')))
            worst_velocity = max(worst_velocity, *(abs(x[k]-y[k]) for k in ('vx','vy','vz')))
    assert worst_position < 1e-6 and worst_velocity < 1e-6
    result['fixed_sparse_comparison'] = {'max_position':worst_position, 'max_velocity':worst_velocity, 'threshold':1e-6}
    (root / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['fixed_sparse_comparison']))
    for stage in ('stage1','stage2','static','moving','deforming'):
        r=result[stage]
        print(stage, 'max residual',r['residual']['max'],'max divergence',r['div_after']['max'])
    print('PASS all stages')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'output')
    parser.add_argument('--summarize-only', action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for stage, steps in [('stage1',80),('stage2',80),('static',160),('moving',200),('deforming',200)]:
            print('Running',stage,flush=True)
            with (args.output/(stage+'.log')).open('w') as log:
                subprocess.run([str(args.executable.resolve()),stage,str(args.output/stage),str(steps)],stdout=log,stderr=subprocess.STDOUT,check=True)
    summarize(args.output)
