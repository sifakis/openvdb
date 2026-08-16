import math
RD = math.sqrt(2)/2

def cands(W):
    R = int(math.ceil(W + 1.5))
    out=[]
    for a in range(-R,R+1):
        for b in range(-R,R+1):
            if a==0 and b==0: continue
            out.append((a,b,math.hypot(a,b)))
    return out

def eps_thr(al, W, C):
    n=(math.cos(al), math.sin(al)); best=None
    for a,b,L in C:
        dot=a*n[0]+b*n[1]; d0=(L+dot)/2
        if RD < d0 <= W:
            g=(L-dot)/2
            if best is None or g<best: best=g
    return best if best is not None else RD

def eps_act(al, W, C):
    """sup of eps in (0,RD] not certified by any single admissible witness (actual depths)"""
    n=(math.cos(al), math.sin(al)); iv=[]
    for a,b,L in C:
        dot=a*n[0]+b*n[1]
        g=(L-dot)/2
        lo=max(g, RD-dot); hi=min(RD, W-dot)
        if lo<hi: iv.append((lo,hi))
    if not iv: return RD
    iv.sort()
    # merge (a,b] intervals; (a1,b1] u (a2,b2] merges if a2 <= b1
    cur_lo,cur_hi=iv[0]; merged=[]
    for lo,hi in iv[1:]:
        if lo<=cur_hi: cur_hi=max(cur_hi,hi)
        else: merged.append((cur_lo,cur_hi)); cur_lo,cur_hi=lo,hi
    merged.append((cur_lo,cur_hi))
    top_lo,top_hi=merged[-1]
    return top_lo if top_hi>=RD-1e-15 else RD

def maxover(f, W, C, N=20000, lo=0.0, hi=math.pi/2):
    best=-1; ba=None
    for i in range(N+1):
        al=lo+(hi-lo)*i/N
        v=f(al,W,C)
        if v>best: best,ba=v,al
    # golden refine
    a=max(lo,ba-(hi-lo)/N); b=min(hi,ba+(hi-lo)/N)
    gr=(math.sqrt(5)-1)/2
    for _ in range(200):
        x1=b-gr*(b-a); x2=a+gr*(b-a)
        if f(x1,W,C)>f(x2,W,C): b=x2
        else: a=x1
    am=(a+b)/2
    v=f(am,W,C)
    if v<best: am,v=ba,best
    return v, math.degrees(am)

import sys
Ws = [1.0,1.2,1.35,1.4,1.41,1.4143,1.5,1.6,1.7,1.7072,1.8,2.0,2.2,2.2353,2.2354,
      2.2360679,2.2360679774997896,2.24,2.2795,2.27949,2.28,2.5,3.0,3.16,
      3.1607675,3.1607676,3.1608,3.161,3.1622776,3.1785,3.17847,3.5,3.7]
print("%-12s %-18s %-10s %-18s %-10s"%("W","eps*_threshold","alpha","eps*_actual","alpha"))
for W in Ws:
    C=cands(W)
    v1,a1=maxover(eps_thr,W,C,N=8000)
    v2,a2=maxover(eps_act,W,C,N=8000)
    print("%-12.7f %-18.12f %-10.5f %-18.12f %-10.5f"%(W,v1,a1,v2,a2))
