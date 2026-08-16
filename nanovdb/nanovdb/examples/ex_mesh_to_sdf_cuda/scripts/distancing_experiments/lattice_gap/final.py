import math
from math import sqrt, hypot, cos, sin, pi, degrees, atan2, gcd
# 1) closed form for eps*(M_p) from the (1,0)/(p,1) tie
def cf(p):
    s=sqrt(p*p+1)-p; q=p-1; A=1+q*q; B=2*s*q-2; C=s*s
    d=(-B-sqrt(B*B-4*A*C))/(2*A)
    return d/2
num={1:0.044910139438,2:0.019202630764,3:0.010574711048,4:0.006724316817,5:0.004673711324}
for p in range(1,6):
    print("p=%d closed form %.12f  numeric %.12f  diff %.2e"%(p,cf(p),num[p],abs(cf(p)-num[p])))
print()
# 2) HHT E_p^B check: a=1 fixed, minimise max relative error, 5x5 mask
def prim(p):
    S=set()
    for x in range(0,p+1):
        for y in range(0,p+1):
            if (x,y)!=(0,0) and gcd(x,y)==1: S.add((x,y))
    T=set()
    for (x,y) in S:
        for v in [(x,y),(y,x),(-x,y),(x,-y),(-x,-y),(-y,x),(y,-x),(-y,-x)]: T.add(v)
    return sorted(T)
def relerr(mask,om,N=4001):
    worst=-9
    for i in range(N):
        th=(pi/4)*i/(N-1); u=(cos(th),sin(th)); best=1e18
        for a in range(len(mask)):
            for b in range(a+1,len(mask)):
                w1,w2=mask[a],mask[b]
                det=w1[0]*w2[1]-w1[1]*w2[0]
                if det==0: continue
                la=(u[0]*w2[1]-u[1]*w2[0])/det; mu=(w1[0]*u[1]-w1[1]*u[0])/det
                if la<-1e-12 or mu<-1e-12: continue
                v=la*om[w1]+mu*om[w2]
                if v<best: best=v
        worst=max(worst,best-1)
    return worst
M=prim(2)
def mk(a,b,c):
    om={}
    for w in M:
        k=tuple(sorted((abs(w[0]),abs(w[1]))))
        om[w]= a if k==(0,1) else (b if k==(1,1) else c)
    return om
def obj(x): return relerr(M,mk(1.0,x[0],x[1]))
x=[sqrt(2),sqrt(5)]; f=obj(x); st=0.05
while st>1e-9:
    imp=False
    for i in range(2):
        for s in (1,-1):
            y=list(x); y[i]+=s*st; fy=obj(y)
            if fy<f-1e-14: x,f=y,fy; imp=True
    if not imp: st/=2
print("E_2^B (a=1 fixed, min max rel err) computed = %.6f   (HHT state 0.0187)"%f, " weights b,c =",["%.6f"%v for v in x])
print("our eps*(M_2) = 0.019203  -- close to E_2^B=0.0187 but NOT equal")
