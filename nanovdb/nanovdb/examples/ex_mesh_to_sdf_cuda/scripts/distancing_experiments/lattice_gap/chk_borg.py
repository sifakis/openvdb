import math
from math import sqrt, sin, cos, atan, hypot, pi, degrees, radians

# ---- our quantity ----
MASK5 = [(1,0),(2,1),(1,1),(1,2),(0,1)]
MASK3 = [(1,0),(1,1),(0,1)]
def g(w, a):           # a = angle of unit normal n
    n = (cos(a), sin(a))
    L = hypot(*w)
    return (L - (w[0]*n[0]+w[1]*n[1]))/2.0
def eps_of_n(mask, a):
    return min(g(w,a) for w in mask)
def eps_star(mask, lo=0.0, hi=pi/2, N=2000001):
    best=(-1,None)
    for i in range(N):
        a = lo + (hi-lo)*i/(N-1)
        v = eps_of_n(mask,a)
        if v>best[0]: best=(v,a)
    # golden refine
    a=best[1]; h=(hi-lo)/(N-1)
    lo2,hi2=a-h,a+h
    for _ in range(200):
        m1=lo2+(hi2-lo2)/3; m2=hi2-(hi2-lo2)/3
        if eps_of_n(mask,m1)<eps_of_n(mask,m2): lo2=m1
        else: hi2=m2
    a=(lo2+hi2)/2
    return eps_of_n(mask,a), a

e3,a3 = eps_star(MASK3)
e5,a5 = eps_star(MASK5)
print("ours 3x3 mask  eps* = %.16f  at alpha = %.6f deg" % (e3, degrees(a3)))
print("ours 5x5 mask  eps* = %.16f  at alpha = %.6f deg" % (e5, degrees(a5)))

# Borgefors 1986 closed forms from the salvaged PDF text
borg3 = (1 - sqrt(2*sqrt(2)-2))/2
print("Borgefors (18) 3x3 free-(a,b) maxdiff = (1-sqrt(2sqrt2-2))/2 = %.16f" % borg3)
print("   a_opt=%.6f b_opt=%.6f   (paper: 0.95509, 1.36930)" % ((sqrt(2*sqrt(2)-2)+1)/2, sqrt(2)-borg3))
print("   |w|-eps*: 1-e3=%.6f  sqrt2-e3=%.6f" % (1-e3, sqrt(2)-e3))
print("   match 3x3? ", abs(borg3-e3) < 1e-15, abs(borg3-e3))
print()
# Borgefors 5x5, a fixed to 1: c_opt = 2.19691, maxdiff=(sqrt5-c)/2=0.01958
c_paper = 2.19691
print("Borgefors 5x5 (a==1 constrained): maxdiff=(sqrt5-c_opt)/2 = %.8f" % ((sqrt(5)-c_paper)/2))
print("ours 5x5 = %.8f  -> a=%.7f b=%.7f c=%.7f" % (e5, 1-e5, sqrt(2)-e5, sqrt(5)-e5))
