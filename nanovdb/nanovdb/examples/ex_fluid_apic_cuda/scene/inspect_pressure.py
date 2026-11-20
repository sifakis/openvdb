# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Independent SciPy audit of a failed finite-volume pressure system."""
import argparse,json
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);a=p.parse_args()
x=np.loadtxt(a.input,delimiter=',',skiprows=1);ids=x[:,0].astype(np.int64);n=len(ids)
lookup=np.full(ids.max()+1,-1,np.int64);lookup[ids]=np.arange(n)
neighbor=x[:,8:14].astype(np.int64);mapped=lookup[neighbor];weights=x[:,14:20]
rows=np.repeat(np.arange(n),6);cols=mapped.ravel();values=-weights.ravel();ok=(cols>=0)&(values!=0)
A=(coo_matrix((values[ok],(rows[ok],cols[ok])),shape=(n,n))+diags(x[:,5])).tocsr()
count,labels=connected_components(A,directed=False)
air=np.sum(np.where(mapped<0,weights,0),axis=1);components=[]
for k in range(count):
 m=labels==k;b=x[m,4]
 components.append(dict(cells=int(m.sum()),air_conductance=float(air[m].sum()),rhs_sum=float(b.sum()),rhs_norm=float(np.linalg.norm(b)),true_residual_norm=float(np.linalg.norm(x[m,7])),bounds=[x[m,1:4].min(0).astype(int).tolist(),x[m,1:4].max(0).astype(int).tolist()]))
components.sort(key=lambda c:c['true_residual_norm'],reverse=True)
result={'unknowns':n,'components':count,'max_asymmetry':float(abs(A-A.T).max()),'zero_diagonal_rows':int(np.sum(x[:,5]==0)),'largest_residual_components':components[:15],'closed_components':[c for c in components if c['air_conductance']==0]}
b=x[:,4];norm=np.linalg.norm(b);history=[]
solution,info=cg(A,b,rtol=1e-10,atol=1e-12,maxiter=2500,M=diags(np.where(x[:,5]>0,1/np.maximum(x[:,5],1e-30),0)),callback=lambda p:history.append(float(np.linalg.norm(A@p-b)/norm)))
result.update(cpu_cg_info=int(info),cpu_cg_iterations=len(history),cpu_true_relative_residual=float(np.linalg.norm(A@solution-b)/norm),history_every_50=history[::50])
open(a.output,'w').write(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='history_every_50'},indent=2))
