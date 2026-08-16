import math
from math import sqrt, hypot, cos, sin, pi, degrees, gcd, atan2

def prim_mask(p):
    S=set()
    for x in range(0,p+1):
        for y in range(0,p+1):
            if (x,y)==(0,0): continue
            if gcd(x,y)!=1: continue
            S.add((x,y))
    # full symmetry closure
    T=set()
    for (x,y) in S:
        for v in [(x,y),(y,x),(-x,y),(x,-y),(-x,-y),(-y,x),(y,-x),(-y,-x)]:
            T.add(v)
    return sorted(T)

def g(w,a):
    L=hypot(*w); return (L-(w[0]*cos(a)+w[1]*sin(a)))/2
def eps_star(mask):
    best=(-1,0)
    N=400001
    for i in range(N):
        a=(pi/2)*i/(N-1); v=min(g(w,a) for w in mask)
        if v>best[0]: best=(v,a)
    lo,hi=best[1]-pi/2/(N-1), best[1]+pi/2/(N-1)
    for _ in range(300):
        m1=lo+(hi-lo)/3; m2=hi-(hi-lo)/3
        if min(g(w,m1) for w in mask) < min(g(w,m2) for w in mask): lo=m1
        else: hi=m2
    a=(lo+hi)/2
    return min(g(w,a) for w in mask), degrees(a), sorted(mask,key=lambda w:g(w,a))[:2]

print(" p |   eps*(M_p)   | alpha* deg | tying pair        |  E_p^D (HHT)  | E_p^C (HHT) | ratio eps*/E^D")
for p in range(1,7):
    M=prim_mask(p)
    e,a,pair=eps_star(M)
    s=sqrt(p*p+1)-p
    ED=sqrt(s*s+1)-1
    EC=ED/(2+ED)
    print(" %d | %.12f | %10.5f | %-17s | %.10f | %.10f | %.6f" %
          (p,e,a,str(pair),ED,EC,e/ED))
print()
# check the closed form 2eps* = smaller root of d^2 + d(s-1) + s^2/2 = 0 (pair (1,0),(p,1))
for p in [1,2,3]:
    s=sqrt(p*p+1)-p
    disc=(1-s)**2-2*s*s
    d=((1-s)-sqrt(disc))/2 if disc>=0 else float('nan')
    print("p=%d  pair (1,0),(%d,1): predicted eps* = %.15f"%(p,p,d/2))
