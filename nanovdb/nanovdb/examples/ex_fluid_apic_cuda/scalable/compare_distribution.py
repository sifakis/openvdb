# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Spatial mass/velocity comparison independent of APIC particle identities.

Keep the earlier trajectory test and its failure visible. These additional
checks characterize collision-sensitive trajectories; they do not erase it.
"""
import argparse,json
from pathlib import Path
import numpy as np

def load(path):
    d=np.loadtxt(path,delimiter=',',skiprows=1)
    return d[:,0].astype(np.int64),d[:,1:4],d[:,4:7]
def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=[]
    for frame in (0,18,36):
        ids,a,va=load(args.reference/f'frame_{frame:06d}.csv');keys,b,vb=load(args.candidate/f'frame_{frame:06d}.csv')
        assert np.array_equal(ids,keys)
        distance=np.linalg.norm(a-b,axis=1);top=np.argsort(distance)[-10:][::-1]
        metrics={'frame':frame,'fraction_over_10cm':float(np.mean(distance>.1)),
                 'fraction_over_1cell':float(np.mean(distance>.02)),
                 'top_outliers':[{'id':int(ids[i]),'distance':float(distance[i]),'reference':a[i].tolist(),
                                  'candidate':b[i].tolist(),'reference_v':va[i].tolist(),'candidate_v':vb[i].tolist()}for i in top]}
        # Fixed 8 cm diagnostic bins, independent of simulation precision.
        edges=[np.arange(0,extent+.04,.08) for extent in (3.2,3.2,2.4)]
        ha=np.histogramdd(a,bins=edges)[0];hb=np.histogramdd(b,bins=edges)[0]
        metrics['mass_histogram_l1_fraction']=float(np.sum(np.abs(ha-hb))/len(ids))
        # This counts mass changes twice when one particle crosses a bin boundary.
        metrics['mass_histogram_bound']=.05
        metrics['distribution_pass']=metrics['mass_histogram_l1_fraction']<=.05
        result.append(metrics)
    report={'reference':str(args.reference),'candidate':str(args.candidate),'bin_width_m':.08,
            'frames':result,'distribution_pass':all(r['distribution_pass']for r in result)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='frames'},indent=2))
    for r in result:print(r['frame'],r['mass_histogram_l1_fraction'],r['fraction_over_10cm'])
if __name__=='__main__':main()
