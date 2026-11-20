# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Check GPU-resident APIC/ID/whitewater checkpoint continuity in float mode."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
EX=Path(__file__).resolve().parents[1];OUT=EX/'output/scalability'
def run(config,folder,resume=False):
    cmd=[sys.executable,str(EX/'scene/pipeline.py'),str(config),'--stages','simulate','--output',str(folder)]
    if resume:cmd.append('--resume')
    subprocess.run(cmd,check=True)
def main():
    config=json.loads((OUT/'smoke_float.json').read_text());config['duration']=.1
    short=OUT/'restart_short.json';short.write_text(json.dumps(config,indent=2))
    config['duration']=.2;long=OUT/'restart_long.json';long.write_text(json.dumps(config,indent=2))
    resumed=OUT/'restart_resumed';full=OUT/'restart_full'
    run(short,resumed);run(long,resumed,True);run(long,full)
    a=np.loadtxt(resumed/'frame_000006.csv',delimiter=',',skiprows=1)
    b=np.loadtxt(full/'frame_000006.csv',delimiter=',',skiprows=1)
    same=np.array_equal(a[:,0],b[:,0]);dx=float(np.max(np.abs(a[:,1:4]-b[:,1:4])));dv=float(np.max(np.abs(a[:,4:7]-b[:,4:7])))
    def secondary(folder):
        data=np.genfromtxt(folder/'frame_000006_whitewater.csv',delimiter=',',names=True,dtype=None,encoding=None)
        return np.sort(np.atleast_1d(data),order='id')
    wa,wb=secondary(resumed),secondary(full);same_ww=np.array_equal(wa['id'],wb['id'])
    wwerror=max(float(np.max(np.abs(wa[k]-wb[k]))) for k in ('x','y','z','vx','vy','vz','radius','age','life')) if same_ww and len(wa) else 0
    result={'same_ids':bool(same),'position_max_m':dx,'velocity_max_m_s':dv,'secondary_same_ids':bool(same_ww),
            'secondary_max_state_difference':wwerror,'float_tolerance':5e-5,
            'pass':bool(same and same_ww and dx<5e-5 and dv<5e-5 and wwerror<5e-5)}
    (OUT/'restart_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
    if not result['pass']:raise RuntimeError('GPU resident restart mismatch')
if __name__=='__main__':main()
