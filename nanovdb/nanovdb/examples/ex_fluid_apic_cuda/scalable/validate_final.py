# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Final sequential simulation/pressure checks, using existing baked character."""
import json,subprocess,sys,time
from pathlib import Path
EX=Path(__file__).resolve().parents[1];OUT=EX/'output/scalability';BIN=EX.parents[3]/'build/release/nanovdb/nanovdb/examples'
def run(name,cmd,times):
    print('START',name,flush=True);start=time.perf_counter()
    with (OUT/(name+'.log')).open('w')as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
    times[name]={'seconds':time.perf_counter()-start,'returncode':r.returncode}
    (OUT/'final_times.json').write_text(json.dumps(times,indent=2)+'\n')
    print('DONE',name,times[name],flush=True)
    if r.returncode:raise RuntimeError(name+' failed')
def main():
    times={}
    for config,name in [('fine_float','final_float'),('fine_double','final_double'),('high_float','final_high'),('timed_float','final_timed')]:
        run(name,[sys.executable,EX/'scene/pipeline.py',OUT/(config+'.json'),'--stages','simulate','--output',OUT/name],times)
    # Preserve the earlier benchmark table; updated float arithmetic gets a new table.
    old=OUT/'pressure_benchmarks.json'
    if old.exists():old.rename(OUT/'pressure_benchmarks_initial.json')
    run('final_pressure',[sys.executable,EX/'scalable/run_verification.py','--precision','float','--max-size',256],times)
    run('final_scene_verify',[BIN/'ex_fluid_apic_scene_verify',EX/'output/character'],times)
if __name__=='__main__':main()
