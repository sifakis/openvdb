from decimal import Decimal as D, getcontext
from fractions import Fraction as F
import math
getcontext().prec = 70
def sq(x): return D(x).sqrt()

s5 = sq(5); s2 = sq(2); s10 = sq(10)
beta = (2*s5-4).sqrt()
epsA = (3 - s5 - beta)/4
epsB = (s5-2)/(2*(1+s5+(4+2*s5).sqrt()))
t = (1+s5).sqrt()
epsC = (2-t)/t**4
epsD = (2-t)/(6+2*s5)
# trig form: tan u = 5^(1/4) sin(phi/2)/(1+5^(1/4) cos(phi/2)), phi=arctan(1/2), eps=sin^2 u
phi = D(math.atan(0.5))  # low precision route -> use high prec via arctan series? use algebraic instead
# algebraic: cos phi = 2/sqrt5, sin phi = 1/sqrt5 ; cos(phi/2)=sqrt((1+2/sqrt5)/2), sin(phi/2)=sqrt((1-2/sqrt5)/2)
c2 = ((1+2/s5)/2).sqrt(); s2h = ((1-2/s5)/2).sqrt()
k = s5.sqrt()   # 5^(1/4)
tanu = k*s2h/(1+k*c2)
sin2u = tanu**2/(1+tanu**2)
print("epsA =", epsA)
for nm,v in [("epsB",epsB),("epsC",epsC),("epsD",epsD),("sin^2u",sin2u)]:
    print("  %-8s diff = %s" % (nm, epsA-v))

# rounding facts
quoted = D("0.019202630763796")
print("\neps* - quoted15 =", epsA - quoted)
print("round to 15 dp :", epsA.quantize(D("1e-15")))
print("round to 16 dp :", epsA.quantize(D("1e-16")))
print("16dp rounded < eps*? ", epsA.quantize(D("1e-16")) < epsA)
print("0.0192026307637964 >= eps*?", D("0.0192026307637964") >= epsA)

# min poly
x = epsA
print("\nquartic residual:", 64*x**4 - 192*x**3 + 208*x**2 - 56*x + 1)

# n*
cs = (s5-1+beta)/2; sn = (s5-1-beta)/2
print("cos a* =", cs); print("sin a* =", sn)
print("c^2+s^2-1 =", cs*cs+sn*sn-1)
print("c+s-(sqrt5-1) =", cs+sn-(s5-1))
alpha = D(180)/D(str(math.pi)) # placeholder
# high-prec atan2 via arctan series on sn/cs
def arctan(z, terms=400):
    # z small enough? z ~ 0.2854 -> series converges ok at prec 70 (0.2854^k)
    s = D(0); zp = z; 
    for k in range(0, terms):
        s += (D(-1)**k) * zp/(2*k+1)
        zp *= z*z
    return s
PI = D("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803")
a_star = arctan(sn/cs)*180/PI
print("alpha* deg =", a_star)
