import math
from slib import *
from decimal import Decimal as D, getcontext
getcontext().prec = 60
s3 = D(3).sqrt()
W_lo_exa = float(s3 - D(1) / 2 + D(3) / 2 * (s3 - 1).sqrt())
print("W_lo' (plateau track closed form) =", repr(W_lo_exa))

ws = witnesses(20)
nc = (math.sqrt(math.sqrt(3) - 1), (math.sqrt(3) - 1) / 2, (math.sqrt(3) - 1) / 2)
s = math.sqrt(sum(c * c for c in nc)); nc = tuple(c / s for c in nc)
print("n_c =", nc, " |n|-1 =", s - 1)

EPS_EXA = 0.038443423574719627
A = (1 - math.sqrt(math.sqrt(3) - 1)) / 2
print("A = (1-sqrt(sqrt3-1))/2 =", A, "  = 1.878 x eps_exa?", A / EPS_EXA)

print("\nF_exa at n_c and certified global max, at the three claimed floors:")
for W, tag in ((2.5151853, "certificate track floor"),
               (2.5153786883076, "e2e track floor (claimed bracket 1.4e-14)"),
               (W_lo_exa - 1e-9, "W_lo'(plateau) - 1e-9"),
               (W_lo_exa, "W_lo'(plateau)"),
               (W_lo_exa + 1e-9, "W_lo'(plateau) + 1e-9"),
               (2.52, "2.52"), (2.6, "2.6"), (3.0, "3.0")):
    f = F_exa(nc, ws, W=W)
    lo, up, ncell = bnb('exa', ws, W=W, tol=1e-11, maxcells=300000)
    print(f"  W={W:.13f} {tag:42s} F_exa(n_c)={f:.12f}   global in [{lo:.12f},{up:.12f}]")

print("\nwindow anatomy at n_c, W = 2.5153786883076 (the disputed value):")
W = 2.5153786883076
iv = []
for (a, b, c, L) in witnesses(14):
    d = a * nc[0] + b * nc[1] + c * nc[2]
    g = (L - d) / 2
    lo_ = max(g, RD - d); hi_ = min(RD, W - d)
    if hi_ > lo_ and lo_ < 0.09:
        iv.append((lo_, hi_, (a, b, c)))
for lo_, hi_, w in sorted(iv):
    print(f"   w={w} window ({lo_:.12f}, {hi_:.12f}]")
print("   -> uncovered sup =", F_exa(nc, ws, W=W))
