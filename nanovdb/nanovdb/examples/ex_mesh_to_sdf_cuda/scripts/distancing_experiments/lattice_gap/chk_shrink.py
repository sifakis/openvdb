import math
from math import sqrt, hypot, cos, sin, pi, atan2, degrees
# Claim: 2*eps* = smallest uniform shrink d s.t. chamfer norm with w_i -> |w_i|-d
# never exceeds the Euclidean norm.  d_C^LP(p) = max{p.n : w_i.n <= omega_i}
MASK5=[(1,0),(2,1),(1,1),(1,2),(0,1)]
def sym(mask):
    S=set()
    for (x,y) in mask:
        for (u,v) in [(x,y),(y,x),(-x,y),(x,-y),(-x,-y),(-y,x),(y,-x),(-y,-x)]:
            S.add((u,v))
    return sorted(S)
def maxrel(mask, d, N=200001):
    M=sym(mask); om={w:hypot(*w)-d for w in M}
    # gauge of P = conv{w/om(w)}  -> evaluate by max over unit u of gauge
    # easier: d_C^LP(u) = min over adjacent pairs; use min over ALL pairs (exact for 2D LP)
    worst=-9; arg=None
    P=[(w[0]/om[w], w[1]/om[w]) for w in M]
    for i in range(N):
        th=2*pi*i/N; u=(cos(th),sin(th))
        # gauge_P(u) = 1/rho, rho = max t with t*u in P; P convex hull of points
        # use support-function dual:  d_C(u)=max{u.n : w.n<=om(w)}  -> enumerate vertices of Q
        # cheap: 1/rho where rho = min over edges of hull... do LP by enumerating pairs:
        best=1e18
        # min over pairs (i,j) of lam*om_i+mu*om_j with lam,mu>=0 and lam w_i+mu w_j=u
        for a in range(len(M)):
            for b in range(a+1,len(M)):
                w1,w2=M[a],M[b]
                det=w1[0]*w2[1]-w1[1]*w2[0]
                if det==0: continue
                lam=(u[0]*w2[1]-u[1]*w2[0])/det
                mu=(w1[0]*u[1]-w1[1]*u[0])/det
                if lam<-1e-12 or mu<-1e-12: continue
                v=lam*om[w1]+mu*om[w2]
                if v<best: best=v
        if best-1.0>worst: worst=best-1.0; arg=degrees(th)
    return worst,arg
eps=0.0192026307637962
for d in [0.030,0.0384052615,0.0384053,0.045]:
    w,a=maxrel(MASK5,d,N=2001)
    print("d=%.10f  max(d_C - 1) = %+.10f  at %.4f deg"%(d,w,a))
print("2*eps* =", 2*eps)
