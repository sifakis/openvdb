from decimal import Decimal as D, getcontext
import math
getcontext().prec = 60
s5=D(5).sqrt(); s2=D(2).sqrt(); beta=(2*s5-4).sqrt()
E=(3-s5-beta)/4
c=(s5-1+beta)/2; s=(s5-1-beta)/2
rd=s2/2; W=D(3)

# exhaustive at n*, |w|^2 <= 400
R=20
best=[]; inband_le=[]
for a in range(-R,R+1):
    for b in range(-R,R+1):
        if a==0 and b==0: continue
        L=D(a*a+b*b).sqrt(); dot=a*c+b*s
        g=(L-dot)/2; d0=(L+dot)/2
        if g<=E and d0<=W:            # in-band (threshold), cost <= eps*
            inband_le.append((a,b,g,d0,rd<d0))
        best.append((g,d0,a,b))
print("in-band w with g <= eps*, |w|<=20*sqrt2:")
for a,b,g,d0,nb in inband_le: print("   (%d,%d) g=%s d0=%s nonbarrier=%s"%(a,b,g,d0,nb))
best.sort()
print("\ncheapest 8 overall (ignoring band):")
for g,d0,a,b in best[:8]: print("   (%3d,%3d) g=%s d0=%s inband=%s"%(a,b,g,d0,d0<=W))
# completeness bound
print("\nW+eps* =", W+E, " (W+eps*)^2 =", (W+E)**2, " -> |w|^2 <= 9")
print("W+2eps* =", W+2*E, " squared =", (W+2*E)**2)

# three-arc covering of the octant (upper bound)
# arc A: (1,0) has g<=E  <=> (1-cos a)/2 <= E <=> cos a >= 1-2E -> a <= aA
# arc B: (2,1): sqrt5(1-cos(a-phi))/2 <= E -> |a-phi| <= d5, cos d5 = 1-2E/sqrt5
# arc C: (1,1): |a-45| <= d2, cos d2 = 1-2E/sqrt2
def acos_d(x):
    return D(repr(math.acos(float(x))))
aA=acos_d(1-2*E); d5=acos_d(1-2*E/s5); d2=acos_d(1-2*E/s2)
phi=D(repr(math.atan(0.5)))
deg=D(180)/D(repr(math.pi))
print("\narc A = [0, %s] deg"%(aA*deg))
print("arc B = [%s, %s] deg"%((phi-d5)*deg,(phi+d5)*deg))
print("arc C = [%s, 45] deg"%((D(repr(math.pi/4))-d2)*deg))
print("A_end - B_start =", (aA-(phi-d5))*deg, " (should be ~0: arcs meet exactly at alpha*)")
print("B_end - C_start =", ((phi+d5)-(D(repr(math.pi/4))-d2))*deg, " (>0 => overlap)")
