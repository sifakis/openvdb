"""R1: independently verify the two closed forms, their minimal polynomials,
and the claimed critical directions -- at 200+ digits, from scratch."""
import sys
sys.path.insert(0, '/tmp/dist/d3/ref_value')
from hp import setprec, dsqrt, polyeval
from decimal import Decimal, getcontext

P = 260
setprec(P + 40)

s2 = dsqrt(2); s3 = dsqrt(3); s5 = dsqrt(5); s6 = dsqrt(6); s30 = dsqrt(30); s10 = dsqrt(10)

# --- threshold rule -----------------------------------------------------------
beta3sq = 4*s30 + 2*s5 - 26
beta3 = beta3sq.sqrt()
eps_thr = (3 - s5 - beta3)/4

nx = (s5 - 1 + beta3)/2
ny = (s5 - 1 - beta3)/2
nz = s6 - s5
norm_thr = nx*nx + ny*ny + nz*nz - 1

print("=== THRESHOLD RULE ===")
print("beta3^2 =", +beta3sq)
print("eps_thr =", (+eps_thr))
print("n* = (%s,\n      %s,\n      %s)" % (+nx, +ny, +nz))
print("|n*|^2 - 1 =", +norm_thr)
print("n_x - (1-2eps) =", +(nx - (1-2*eps_thr)))
print("n_y - (sqrt5-2+2eps) =", +(ny - (s5-2+2*eps_thr)))

# tie check: g(w,n) = (|w| - w.n)/2 for the three claimed witnesses
def g(w, n, sq):
    return (sq - (w[0]*n[0]+w[1]*n[1]+w[2]*n[2]))/2

n_thr = (nx, ny, nz)
for w, sq in [((1,0,0), Decimal(1)), ((2,1,0), s5), ((2,1,1), s6)]:
    gg = g(w, n_thr, sq)
    print("  w=%s  g-eps = %s" % (str(w), +(gg - eps_thr)))

# --- exact rule ---------------------------------------------------------------
gam3sq = 6*s2 + 2*s6 - 13
gam3 = gam3sq.sqrt()
eps_exa = (s2 + s3 - s6 - gam3)/2
ex = s6 - s3; ey = gam3; ez = s3 - s2
print()
print("=== EXACT RULE ===")
print("gamma3^2 =", +gam3sq)
print("eps_exa =", (+eps_exa))
print("n_e = (%s,\n       %s,\n       %s)" % (+ex, +ey, +ez))
print("|n_e|^2 - 1 =", +(ex*ex+ey*ey+ez*ez-1))
n_exa = (ex, ey, ez)
for w, sq in [((1,1,0), s2), ((1,1,1), s3), ((2,1,1), s6)]:
    gg = g(w, n_exa, sq)
    print("  w=%s  g-eps = %s" % (str(w), +(gg - eps_exa)))
# (2,2,1) at n_e
w=(2,2,1); sq=Decimal(3)
g221 = g(w, n_exa, sq)
print("  (2,2,1): g =", +g221, " 2g =", +(2*g221), " w.n =", +(2*ex+2*ey+ez))
print("  (2,2,1) exact-rule window top = W - w.n =", +(3 - (2*ex+2*ey+ez)))

# --- minimal polynomials ------------------------------------------------------
Pthr = [25, -800, 3400, -5040, 4624, -2944, 1344, -384, 64]   # ascending
Pexa = [9, -252, 474, -228, 79, 12, 2, 0, 1]                  # ascending
print()
print("=== MINIMAL POLYNOMIALS (residuals) ===")
print("P_thr(eps_thr) =", +polyeval(Pthr, eps_thr))
print("P_exa(eps_exa) =", +polyeval(Pexa, eps_exa))
# derivative for sensitivity
def dpoly(c):
    return [i*c[i] for i in range(1, len(c))]
print("P_thr'(eps_thr) =", +polyeval(dpoly(Pthr), eps_thr))
print("P_exa'(eps_exa) =", +polyeval(dpoly(Pexa), eps_exa))

print()
print("eps_thr (200 dig) =", str(+eps_thr)[:205])
print("eps_exa (200 dig) =", str(+eps_exa)[:205])
print("ratio exa/thr =", +(eps_exa/eps_thr))
