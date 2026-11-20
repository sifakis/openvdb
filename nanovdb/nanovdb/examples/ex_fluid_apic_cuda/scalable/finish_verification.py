# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Run the remaining scalability checks sequentially after the float scene."""
import json,subprocess,sys,time
from pathlib import Path
EX=Path(__file__).resolve().parents[1]
OUT=EX/'output/scalability'
BIN=EX.parents[3]/'build/release/nanovdb/nanovdb/examples'
def run(name,cmd):
    print('START',name,flush=True);t=time.perf_counter()
    with (OUT/(name+'.log')).open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,time.perf_counter()-t,flush=True)
def main():
    deadline=time.monotonic()+3600
    while not (OUT/'fine_float/complete.json').exists():
        log=(OUT/'fine_float.log').read_text()
        if 'FAIL' in log:raise RuntimeError('float scene failed')
        if time.monotonic()>deadline:raise TimeoutError('float scene')
        time.sleep(5)
    run('pressure_suite',[sys.executable,EX/'scalable/run_verification.py'])
    for name in ('fine_double','high_float'):
        run(name,[sys.executable,EX/'scene/pipeline.py',OUT/(name+'.json'),'--stages','simulate','--output',OUT/name])
    run('base_regression',[sys.executable,EX/'validate.py',BIN/'ex_fluid_apic_cuda','--output',OUT/'base_regression'])
    run('water_regression',[BIN/'ex_fluid_apic_water_verify',EX/'output/character'])
if __name__=='__main__':main()
