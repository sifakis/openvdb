import math, random
from math import sqrt, hypot, cos, sin, atan2, degrees, pi

# Borgefors rectilinear criterion, 5x5 mask {a:(1,0), b:(1,1), c:(2,1)}
# error along the line x = 1, y = t in [0,1]
def maxabs_rect(a,b,c,N=400001):
    m=0.0; arg=None
    for i in range(N):
        t=i/(N-1)
        if t<=0.5: d=(1-2*t)*a + t*c
        else:      d=(1-t)*c + (2*t-1)*b
        e=d-sqrt(1+t*t)
        if abs(e)>m: m=abs(e); arg=(t,e)
    return m,arg

# circular criterion: absolute error on the unit circle (== relative error)
def maxabs_circ(a,b,c,N=400001):
    m=0.0; arg=None
    for i in range(N):
        th=(pi/4)*i/(N-1)
        u=(cos(th),sin(th))
        if th<=atan2(1,2):    # cone (1,0),(2,1):  u = l*(1,0)+m*(2,1)
            mu=u[1]; la=u[0]-2*mu; d=la*a+mu*c
        else:                 # cone (2,1),(1,1)
            la=u[0]-u[1]; mu=2*u[1]-u[0]; d=la*c+mu*b
        e=d-1.0
        if abs(e)>m: m=abs(e); arg=(degrees(th),e)
    return m,arg

eps=0.0192026307637962
cand=(1-eps, sqrt(2)-eps, sqrt(5)-eps)
print("candidate weights (|w|-eps*):", ["%.7f"%x for x in cand])
print("  rectilinear maxabs = %.10f  at %s" % maxabs_rect(*cand))
print("  circular    maxabs = %.10f  at %s" % maxabs_circ(*cand))
print()

def minimize(obj, x0, step=0.05):
    x=list(x0); f=obj(*x)[0]
    while step>1e-12:
        improved=False
        for i in range(3):
            for s in (+1,-1):
                y=list(x); y[i]+=s*step
                fy=obj(*y)[0]
                if fy<f-1e-16: x,f=y,fy; improved=True
        if not improved: step/=2
    return x,f

x,f = minimize(maxabs_rect,(1.0,sqrt(2),2.2))
print("BORGEFORS rectilinear, FREE (a,b,c):")
print("   opt weights = %.9f %.9f %.9f" % tuple(x))
print("   |w|-w_i     = %.9f %.9f %.9f" % (1-x[0], sqrt(2)-x[1], sqrt(5)-x[2]))
print("   maxdiff     = %.10f     (our eps* = %.10f)" % (f, eps))
print()
x2,f2 = minimize(maxabs_circ,(1.0,sqrt(2),2.2))
print("CIRCULAR/relative, FREE (a,b,c):")
print("   opt weights = %.9f %.9f %.9f" % tuple(x2))
print("   |w|-w_i     = %.9f %.9f %.9f" % (1-x2[0], sqrt(2)-x2[1], sqrt(5)-x2[2]))
print("   max rel err = %.10f" % f2)
print()
# 3x3 sanity with same machinery
def maxabs_rect3(a,b,N=400001):
    m=0.0
    for i in range(N):
        t=i/(N-1); d=(1-t)*a+t*b; e=d-sqrt(1+t*t)
        if abs(e)>m: m=abs(e)
    return m,None
def mz(obj,x0,step=0.05):
    x=list(x0); f=obj(*x)[0]
    while step>1e-12:
        imp=False
        for i in range(len(x)):
            for s in (+1,-1):
                y=list(x); y[i]+=s*step; fy=obj(*y)[0]
                if fy<f-1e-16: x,f=y,fy; imp=True
        if not imp: step/=2
    return x,f
x3,f3=mz(maxabs_rect3,(1.0,1.4))
print("3x3 rectilinear FREE (a,b): w=%.9f %.9f  maxdiff=%.12f" % (x3[0],x3[1],f3))
print("   |w|-w_i = %.9f %.9f" % (1-x3[0], sqrt(2)-x3[1]))
