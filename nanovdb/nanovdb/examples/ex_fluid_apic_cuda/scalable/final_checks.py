# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Isolated final checks; run after finish_verification.py has exited."""
import json,subprocess,sys,time
from pathlib import Path
EX=Path(__file__).resolve().parents[1];OUT=EX/'output/scalability';BIN=EX.parents[3]/'build/release/nanovdb/nanovdb/examples'
def run(name,cmd,check=True):
    print('START',name,flush=True);start=time.perf_counter()
    with (OUT/(name+'.log')).open('w') as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
    seconds=time.perf_counter()-start;print('DONE',name,r.returncode,seconds,flush=True)
    path=OUT/'final_check_times.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    previous[name]={'seconds':seconds,'returncode':r.returncode}
    path.write_text(json.dumps(previous,indent=2)+'\n')
    if check and r.returncode:raise RuntimeError(name+' failed')
    return {'seconds':seconds,'returncode':r.returncode}
def main():
    times={}
    times['base_regression']=run('base_regression',[sys.executable,EX/'validate.py',BIN/'ex_fluid_apic_cuda','--output',OUT/'base_regression'])
    for method in ('jacobi',):
        times['pressure_256_'+method]=run('pressure_256_'+method,[BIN/'ex_fluid_apic_scale_float_verify',256,0,method])
    config=json.loads((OUT/'fine_float.json').read_text());config['duration']=.2
    (OUT/'timed_float.json').write_text(json.dumps(config,indent=2))
    times['timed_float']=run('timed_float',[sys.executable,EX/'scene/pipeline.py',OUT/'timed_float.json','--stages','simulate','--output',OUT/'timed_float'])
    command=[OUT/'baseline_scene_cuda','--character',EX/'output/character_grounded','--output',OUT/'timed_baseline']
    for key in ('h','obstacle_h','duration','fps','ppc','dt','emitter_radius','emitter_speed','emitter_x','emitter_z','whitewater_rate','animation_offset','initial_depth','foam_radius_scale','foam_life_scale'):
        command.extend(['--'+key.replace('_','-'),config[key]])
    command.extend(['--drain-enabled',0])
    times['timed_baseline']=run('timed_baseline',command)
    times['comparisons']=run('scene_comparison',[sys.executable,EX/'scalable/report_scalability.py'])
    (OUT/'final_check_times.json').write_text(json.dumps(times,indent=2)+'\n')
if __name__=='__main__':main()
