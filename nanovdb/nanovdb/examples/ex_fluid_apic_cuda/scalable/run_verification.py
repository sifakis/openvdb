# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Sequential GPU comparisons: do not contaminate timings with concurrent jobs."""
import argparse,json,subprocess,time
from pathlib import Path
EX=Path(__file__).resolve().parents[1]
ROOT=EX.parents[3]
BIN=ROOT/'build/release/nanovdb/nanovdb/examples'
def main():
    p=argparse.ArgumentParser();p.add_argument('--max-size',type=int,default=128);p.add_argument('--precision',choices=('float','double','all'),default='all');args=p.parse_args()
    output=EX/'output/scalability';output.mkdir(exist_ok=True,parents=True)
    results=[]
    for precision in (('double','float') if args.precision=='all' else (args.precision,)):
        exe=BIN/f'ex_fluid_apic_scale_{precision}_verify'
        for n,mode in [(32,1),(32,2)]+[(n,0) for n in (32,64,128,256) if n<=args.max_size]:
            for method in ('mgpcg','jacobi'):
                name=f'pressure_{precision}_{n}_{mode}_{method}'
                start=time.perf_counter()
                result=subprocess.run([str(exe),str(n),str(mode),method],capture_output=True,text=True)
                (output/f'{name}.log').write_text(result.stdout+result.stderr)
                if result.returncode:
                    raise RuntimeError(name+' failed: '+result.stderr)
                data=json.loads([s for s in result.stdout.splitlines() if s.startswith('{')][-1])
                data['process_seconds']=time.perf_counter()-start;results.append(data)
                (output/'pressure_benchmarks.json').write_text(json.dumps(results,indent=2)+'\n')
                print(name,'iterations',data['iterations'],'total_ms',data['total_ms'],flush=True)
if __name__=='__main__':main()
