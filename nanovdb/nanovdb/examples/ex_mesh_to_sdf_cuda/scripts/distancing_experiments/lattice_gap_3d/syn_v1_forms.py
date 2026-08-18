"""High precision verification of the closed forms, ties and minimal polynomials."""
from decimal import Decimal as D, getcontext
getcontext().prec = 160


def dsqrt(x):
    return x.sqrt()


s2 = dsqrt(D(2)); s3 = dsqrt(D(3)); s5 = dsqrt(D(5)); s6 = dsqrt(D(6))
s10 = dsqrt(D(10)); s30 = dsqrt(D(30))
rd = s3 / 2

beta = dsqrt(4 * s30 + 2 * s5 - 26)
gam = dsqrt(6 * s2 + 2 * s6 - 13)
eps_thr = (3 - s5 - beta) / 4
eps_exa = (s2 + s3 - s6 - gam) / 2

print("eps*_thr =", str(eps_thr)[:70])
print("eps*_exa =", str(eps_exa)[:70])

n_thr = ((s5 - 1 + beta) / 2, (s5 - 1 - beta) / 2, s6 - s5)
n_exa = (s6 - s3, gam, s3 - s2)
print("|n_thr|^2 - 1 =", sum(c * c for c in n_thr) - 1)
print("|n_exa|^2 - 1 =", sum(c * c for c in n_exa) - 1)
print("n_thr =", [str(c)[:32] for c in n_thr])
print("n_exa =", [str(c)[:32] for c in n_exa])


def gg(w, n, L):
    return (L - sum(D(a) * b for a, b in zip(w, n))) / 2


print("\nthreshold ties (g - eps*):")
for w, L in [((1, 0, 0), D(1)), ((2, 1, 0), s5), ((2, 1, 1), s6)]:
    print("  ", w, str(gg(w, n_thr, L) - eps_thr)[:14],
          "   d0 =", str((L + sum(D(a) * b for a, b in zip(w, n_thr))) / 2)[:22])
print("exact ties (g - eps*):")
for w, L in [((1, 1, 0), s2), ((1, 1, 1), s3), ((2, 1, 1), s6)]:
    print("  ", w, str(gg(w, n_exa, L) - eps_exa)[:14])

# the (2,2,1) spoiler at n_exa
d221 = sum(D(a) * b for a, b in zip((2, 2, 1), n_exa))
g221 = (D(3) - d221) / 2
print("\n(2,2,1) at n_exa: g =", str(g221)[:22], " window top W - w.n =", str(3 - d221)[:22],
      " = 2g?", str(2 * g221 - (3 - d221))[:14])
print("(2,2,1) actual depth at eps_exa:", str(eps_exa + d221)[:22])

# minimal polynomials
P_thr = [64, -384, 1344, -2944, 4624, -5040, 3400, -800, 25]
P_exa = [1, 0, 2, 12, 79, -228, 474, -252, 9]


def ev(P, x):
    r = D(0)
    for c in P:
        r = r * x + D(c)
    return r


print("\nP_thr(eps_thr) =", ev(P_thr, eps_thr))
print("P_exa(eps_exa) =", ev(P_exa, eps_exa))

# derived quantities
print("\nratio exa/thr        =", str(eps_exa / eps_thr)[:22])
eps2d = (3 - s5 - dsqrt(2 * s5 - 4)) / 4
print("eps_c(2D)            =", str(eps2d)[:26])
print("thr3D/2D             =", str(eps_thr / eps2d)[:16])
print("exa3D/2D             =", str(eps_exa / eps2d)[:16])
print("2*eps_exa            =", str(2 * eps_exa)[:24])
print("2*eps_thr            =", str(2 * eps_thr)[:24])
print("r_d - eps_exa        =", str(rd - eps_exa)[:20])
print("r_d / eps_exa        =", str(rd / eps_exa)[:12])
print("r_d / eps_thr        =", str(rd / eps_thr)[:12])
print("eps_exa / W          =", str(eps_exa / 3)[:14])
print("1 - r_d              =", str(1 - rd)[:22], " (Lemma A threshold)")
print("sqrt10 - 3           =", str(s10 - 3)[:18])

# plateau endpoints
W_hi_thr = (s10 + 2 * s5 - 2 + beta) / 2
W_lo_exa = s3 - D(1) / 2 + D(3) / 2 * dsqrt(s3 - 1)
W_hi_exa = s2 + s3 - 3 * eps_exa
b_ = s3 - 2 * eps_thr
W_lo_thr = (18 + 10 * b_ + dsqrt(24 - 8 * b_ * b_)) / 12
print("\nW_lo_thr =", str(W_lo_thr)[:32])
print("W_hi_thr =", str(W_hi_thr)[:32])
print("W_lo_exa =", str(W_lo_exa)[:32], " (= sqrt3 - 1/2 + 1.5 sqrt(sqrt3-1))")
print("W_hi_exa =", str(W_hi_exa)[:32], " alt:", str((3 * s6 - s2 - s3 + 3 * gam) / 2)[:32])
print("3 - W_lo_thr =", str(3 - W_lo_thr)[:14], "   W_hi_thr - 3 =", str(W_hi_thr - 3)[:14])
print("3 - W_lo_exa =", str(3 - W_lo_exa)[:14], "   W_hi_exa - 3 =", str(W_hi_exa - 3)[:14])
print("value at the exact-rule floor: (1-sqrt(sqrt3-1))/2 =",
      str((1 - dsqrt(s3 - 1)) / 2)[:22])
print("jump factor =", str((1 - dsqrt(s3 - 1)) / 2 / eps_exa)[:14])
print("3x3x3 constant (1-sqrt(2sqrt2+2sqrt6-7))/2 =",
      str((1 - dsqrt(2 * s2 + 2 * s6 - 7)) / 2)[:24])

# float facts
import struct
def bits(x):
    return hex(struct.unpack('<Q', struct.pack('<d', x))[0])
for name, v in (("eps_thr", eps_thr), ("eps_exa", eps_exa)):
    f = float(v)
    print(f"\n{name}: nearest double {bits(f)} = {f!r}")
    print("   double - exact =", str(D(f) - v)[:14])
    import math
    nxt = math.nextafter(f, 1.0)
    print("   next double up =", repr(nxt), " - exact =", str(D(nxt) - v)[:14])
