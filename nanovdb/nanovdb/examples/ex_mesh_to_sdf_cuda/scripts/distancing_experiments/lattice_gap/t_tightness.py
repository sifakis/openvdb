#!/usr/bin/env python3
"""t_tightness.py -- machine-checked proof of the LOWER BOUND (tightness) at W = 3.

THEOREM.  Let W = 3, r_d = sqrt(2)/2, and let n* = (cos a*, sin a*) be the unit
vector determined by      cos a* + sin a* = sqrt5 - 1,   0 < a* < 45 deg,
i.e.  cos a* = (beta^2 + 2 beta + 2)/4,  sin a* = (beta^2 - 2 beta + 2)/4  with
beta = sqrt(2 sqrt5 - 4).  Put  G := (1 - cos a*)/2 = (3 - sqrt5 - beta)/4.
Then for EVERY admissible w in Z^2\\{0}:   g(w,n*) = (|w| - w.n*)/2  >=  G,
with equality exactly for w = (1,0) and w = (2,1).

Every inequality below is decided either by exact arithmetic in Q(beta) or by
certified rational interval arithmetic (t_exact.py); no floating point enters a
decision.  Floats are printed for readability only.
"""
import math
from fractions import Fraction as F
from t_exact import *

W = F(3)
FAILS = []


def claim(name, cond, note=""):
    tag = "  [OK]   " if cond else "  [FAIL] "
    print(tag + name + (("   " + note) if note else ""))
    if not cond:
        FAILS.append(name)
    return cond


def hr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ============================================================================
hr("0.  The distinguished direction n* and the constant G  (identification)")
# ----------------------------------------------------------------------------
# a* is DEFINED here by the crossover equation g((1,0),n) = g((2,1),n):
#     (1 - cos a)/2 = (sqrt5 - 2cos a - sin a)/2   <=>   cos a + sin a = sqrt5 - 1.
# Solving with cos^2+sin^2=1:  cos,sin = ( (sqrt5-1) +- sqrt(2 sqrt5 - 4) )/2.
print("  beta   = sqrt(2 sqrt5 - 4) =", BETA_I.s18())
print("  cos a* =", COS_I.s18(), "   sin a* =", SIN_I.s18())
claim("cos^2 + sin^2 = 1 exactly in K", COS * COS + SIN * SIN == Q(1))
claim("cos a* + sin a* = sqrt5 - 1 exactly in K", COS + SIN == SQRT5 - 1)
claim("0 < sin a* < cos a* (so 0 < a* < 45 deg)",
      SIN_I.gt(0) and SIN_I.lt(COS_I))

# G in three forms
G = EPS_I                                   # (1 - cos a*)/2, certified interval
G_K = EPS_K                                 # same thing, exact in K
print("  G = eps* =", G.s18(), "  interval width %.1e" % float(G.width()))
claim("G = (3 - sqrt5 - beta)/4 exactly in K", G_K == (Q(3) - SQRT5 - BETA) / 4)
claim("G = (2 - 2 beta - beta^2)/8 exactly in K", G_K == (Q(2) - 2 * BETA - BETA * BETA) / 8)

# CONTEXT's claimed decimal and closed form
G_claimed = F("0.019202630763796")
claim("G matches the reported decimal 0.019202630763796 (to its last digit)",
      G.lo < G_claimed + F(1, 10 ** 15) and G.hi > G_claimed - F(1, 10 ** 15),
      "|G - claimed| = %.3e" % abs(float(G.mid()) - float(G_claimed)))

# closed form from CONTEXT:  (sqrt5-2) / (2 (1 + sqrt5 + sqrt(4+2 sqrt5)))
num = SQRT5_I - I(2)
den = I(2) * (I(1) + SQRT5_I + (I(4) + I(2) * SQRT5_I).sqrt())
cf = I(num.lo / den.hi, num.hi / den.lo)
print("  CONTEXT closed form   =", cf.s18())
claim("CONTEXT closed form (sqrt5-2)/(2(1+sqrt5+sqrt(4+2sqrt5))) equals G",
      cf.lo <= G.hi and G.lo <= cf.hi,
      "difference bracketed by %.3e" % float(max(abs(cf.hi - G.lo), abs(G.hi - cf.lo))))

# closed form from CONTEXT:  eps* = sin^2 u, tan u = 5^(1/4) sin(phi/2)/(1+5^(1/4) cos(phi/2))
# phi = arctan(1/2):  cos phi = 2/sqrt5, sin phi = 1/sqrt5
r5 = SQRT5_I
c_half = ((I(1) + I(2) / 1 * I(1) * (I(1) / 1)) * I(0) + (I(1) + I(2) * I(1) / 1 / 1)) * I(0)  # unused
cphi = I(2) / 1 * I(1); cphi = I(I(2).lo / r5.hi, I(2).hi / r5.lo)          # 2/sqrt5
cph2 = ((I(1) + cphi) / 2).sqrt()                                           # cos(phi/2)
sph2 = ((I(1) - cphi) / 2).sqrt()                                           # sin(phi/2)
q5 = r5.sqrt()                                                              # 5^(1/4)
tan_u_claim_num = q5 * sph2
tan_u_claim_den = I(1) + q5 * cph2
tan_u_claim = I(tan_u_claim_num.lo / tan_u_claim_den.hi, tan_u_claim_num.hi / tan_u_claim_den.lo)
tan_u_true_n = SIN_I
tan_u_true_d = I(1) + COS_I                       # tan(a*/2) = sin a*/(1+cos a*)
tan_u_true = I(tan_u_true_n.lo / tan_u_true_d.hi, tan_u_true_n.hi / tan_u_true_d.lo)
claim("CONTEXT's tan u formula gives tan(a*/2)",
      tan_u_claim.lo <= tan_u_true.hi and tan_u_true.lo <= tan_u_claim.hi,
      "tan u = %.15f" % float(tan_u_true.mid()))
# sin^2(a*/2) = (1-cos a*)/2 = G identically, so eps* = sin^2 u is the same statement.

# a* in degrees, high precision (Decimal atan series; pi digits hardcoded & cited)
from decimal import Decimal, getcontext
getcontext().prec = 50
PI = Decimal("3.14159265358979323846264338327950288419716939937510")


def atan_dec(x):
    """Taylor arctan, |x| < 0.2; alternating series -> error < first omitted term."""
    t = x
    s = Decimal(0)
    k = 0
    while True:
        term = t / (2 * k + 1)
        s += term if k % 2 == 0 else -term
        t *= x * x
        k += 1
        if abs(t / (2 * k + 1)) < Decimal(10) ** -45:
            return s


t_lo = Decimal(tan_u_true.lo.numerator) / Decimal(tan_u_true.lo.denominator)
t_hi = Decimal(tan_u_true.hi.numerator) / Decimal(tan_u_true.hi.denominator)
a_lo = 2 * atan_dec(t_lo) * 180 / PI
a_hi = 2 * atan_dec(t_hi) * 180 / PI
print("  a* = %s deg  (enclosure width %.1e)" % (str(a_lo)[:22], float(a_hi - a_lo)))
claim("a* matches the reported 15.930625116 deg",
      abs(a_lo - Decimal("15.930625116")) < Decimal("5e-10"),
      "a* - 15.930625116 = %.3e deg" % float(a_lo - Decimal("15.930625116")))

# ============================================================================
hr("1.  Non-vacuity: the target at depth eps* is a BARRIER voxel")
# ----------------------------------------------------------------------------
claim("eps* > 0", G.gt(0))
claim("eps* < sqrt(2)/2 = r_d  (target's sign is NOT settled by connected components)",
      G.lt(BAR_I), "r_d - eps* = %.15f" % float((BAR_I - G).mid()))

# ============================================================================
hr("2.  Lemma A -- for any candidate that could beat G, the non-barrier test is free")
# ----------------------------------------------------------------------------
# |w| >= 1 for every nonzero integer vector, and g >= 0 (Cauchy-Schwarz), so
#     d(w,n) = (|w| + w.n)/2 = |w| - g >= 1 - g >= 1 - G   whenever g <= G.
claim("1 - G > sqrt(2)/2, so g <= G forces d > r_d automatically",
      (I(1) - G).gt(BAR_I), "1 - G = %.15f > %.15f" % (float((I(1) - G).mid()), float(BAR_I.mid())))
# same for the "evaluated at the true target depth" variant:
#     d0 = eps* + w.n = eps* + |w| - 2g >= 1 - G  (using g <= G, eps* = G)
claim("same for d0 = eps* + w.n at the true depth: d0 >= 1 - G > r_d",
      (I(1) - G).gt(BAR_I))
print("  => the ONLY binding admissibility constraint is the in-band one, d <= W.")

# ============================================================================
hr("3.  Lemma B -- finite reduction:  g(w,n) <= G  ==>  |w|^2 <= 9")
# ----------------------------------------------------------------------------
# Admissible  ==>  d = (|w| + w.n)/2 <= W.   Since g = (|w| - w.n)/2 >= 0 we get
#     w.n = d - g <= d <= W,
# hence  2 g = |w| - w.n >= |w| - W,  i.e.   |w| <= W + 2 g.
# Therefore  g <= G  ==>  |w| <= W + 2G.        (stated form)
# Sharper (also valid):  |w| = d + g <= W + g <= W + G.
R2 = W + 2 * G                   # 3.0384052615...
R1 = W + G                       # 3.0192026307...
print("  W + 2G = %.15f      W + G = %.15f" % (float(R2.mid()), float(R1.mid())))
claim("(W + 2G)^2 < 10  (so |w|^2 <= 9 for every integer w with g <= G)",
      (R2 * R2).lt(10), "(W+2G)^2 = %.12f < 10" % float((R2 * R2).mid()))
claim("(W + 2G)^2 > 9  (bound is not vacuous: |w|^2 = 9 must be enumerated)",
      (R2 * R2).gt(9), "(W+2G)^2 = %.12f" % float((R2 * R2).mid()))

CAND = [(x, y) for x in range(-3, 4) for y in range(-3, 4)
        if (x, y) != (0, 0) and x * x + y * y <= 9]
claim("candidate disk |w|^2 <= 9 has 28 nonzero lattice points", len(CAND) == 28,
      "n = %d" % len(CAND))

# ============================================================================
hr("4.  Exhaustive certified enumeration of the 28 candidates at n = n*")
# ----------------------------------------------------------------------------
rows = []
for (x, y) in CAND:
    L = isqrt_int(x * x + y * y)
    dot = I(x) * COS_I + I(y) * SIN_I
    d = (L + dot) / 2
    g = (L - dot) / 2
    inband = d.le(W)
    inband_no = d.gt(W)
    nonbar = d.gt(BAR_I)
    nonbar_no = d.lt(BAR_I)
    if not (inband or inband_no):
        claim("in-band decision for %s is decidable" % str((x, y)), False)
    if not (nonbar or nonbar_no):
        claim("non-barrier decision for %s is decidable" % str((x, y)), False)
    adm = inband and nonbar
    gt = g.gt(G)
    lt = g.lt(G)
    if gt:
        rel = ">"
    elif lt:
        rel = "<"
    else:
        rel = "="          # provisional: settled exactly in K below
    rows.append(dict(w=(x, y), L2=x * x + y * y, L=L, dot=dot, d=d, g=g,
                     adm=adm, inband=inband, nonbar=nonbar, rel=rel))

rows.sort(key=lambda r: r["g"].mid())
print("      w     |w|^2      |w|            w.n*           d=depth        "
      "g=cost         adm?  g vs G")
print("  " + "-" * 100)
for r in rows:
    why = "yes " if r["adm"] else ("OUT-OF-BAND" if not r["inband"] else "BARRIER")
    print("  %7s  %2d   %14.12f %15.12f %14.12f %14.12f  %-11s %s G" %
          (str(r["w"]), r["L2"], float(r["L"].mid()), float(r["dot"].mid()),
           float(r["d"].mid()), float(r["g"].mid()), why, r["rel"]))

ties = [r for r in rows if r["rel"] == "="]
below = [r for r in rows if r["rel"] == "<" and r["adm"]]
claim("no admissible candidate is certifiably BELOW G", len(below) == 0,
      "violators: %s" % [r["w"] for r in below])
claim("exactly two candidates are not separated from G by intervals (the ties)",
      sorted(r["w"] for r in ties) == [(1, 0), (2, 1)], str(sorted(r["w"] for r in ties)))
for r in ties:
    claim("tie candidate %s is admissible" % str(r["w"]), r["adm"],
          "d = %.12f in (%.6f, 3]" % (float(r["d"].mid()), float(BAR_I.mid())))

# ---- the ties, settled EXACTLY in K (no numerics) ---------------------------
g10 = (Q(1) - COS) / 2                       # w=(1,0): (|w| - w.n)/2, |w| = 1
g21 = (SQRT5 - (2 * COS + SIN)) / 2          # w=(2,1): |w| = sqrt5 = (beta^2+4)/2 in K
claim("EXACT in K:  g((1,0),n*) = G", g10 == G_K, repr(g10))
claim("EXACT in K:  g((2,1),n*) = G", g21 == G_K, repr(g21))
claim("EXACT in K:  g((1,0),n*) = g((2,1),n*)  (the crossover is exact, not numeric)",
      g10 == g21)

# margins of the remaining admissible candidates
others = [r for r in rows if r["adm"] and r["rel"] == ">"]
mg = min(others, key=lambda r: (r["g"].lo - G.hi))
claim("every other admissible candidate is strictly above G", len(others) + 2 ==
      len([r for r in rows if r["adm"]]),
      "smallest excess: w=%s, g - G >= %.15f (that is %.4f x G)" %
      (str(mg["w"]), float((mg["g"] - G).lo), float((mg["g"].mid() / G.mid()))))

# ============================================================================
hr("5.  The dangerous near-miss: w = (3,1)")
# ----------------------------------------------------------------------------
L31 = isqrt_int(10)
dot31 = I(3) * COS_I + I(1) * SIN_I
d31 = (L31 + dot31) / 2
g31 = (L31 - dot31) / 2
print("  |w| = sqrt10 = %.15f     w.n* = %.15f" % (float(L31.mid()), float(dot31.mid())))
print("  g((3,1)) = %.15f  =  G / %.4f   <-- would beat G by %.1fx" %
      (float(g31.mid()), float(G.mid() / g31.mid()), float(G.mid() / g31.mid())))
print("  depth d((3,1)) = %.15f" % float(d31.mid()))
claim("g((3,1),n*) < G  (it really would win if it were allowed)", g31.lt(G))

# EXACT: 3 cos a* + sin a* = beta^2 + beta + 2 = 2 sqrt5 - 2 + beta
claim("EXACT in K:  3 cos a* + sin a* = 2 sqrt5 - 2 + beta",
      3 * COS + SIN == 2 * SQRT5 - 2 + BETA)
# so 2 d((3,1)) - 2W = sqrt10 + 2 sqrt5 + beta - 8.  Crude certified rationals:
claim("sqrt10 > 79/25 = 3.16", F(79, 25) ** 2 < 10)
claim("2 sqrt5 > 111/25 = 4.44", (F(111, 50)) ** 2 < 5)
claim("beta > 17/25 = 0.68  (beta^2 = 2 sqrt5 - 4 > 0.4624)",
      BETA2_I.gt(F(17, 25) ** 2))
lower = F(79, 25) + F(111, 25) + F(17, 25)
claim("hence sqrt10 + 2 sqrt5 + beta > %s > 8, i.e. d((3,1)) > W = 3 by hand" % str(lower),
      lower > 8, "margin >= (%s - 8)/2 = %.4f" % (str(lower), float((lower - 8) / 2)))
claim("certified interval: d((3,1)) > W = 3", d31.gt(W),
      "d - W = %.15f  (NOT a rounding-scale margin)" % float((d31 - I(W)).lo))
claim("also out of band at the TRUE target depth: eps* + w.n* > 3",
      (G + dot31).gt(W), "eps* + w.n* = %.15f" % float((G + dot31).mid()))
claim("independently, (3,1) fails Lemma B's size test: |w| = sqrt10 > W + 2G",
      L31.gt(R2), "sqrt10 - (W+2G) = %.15f" % float((L31 - R2).lo))
Wneed = d31
print("  band half-width that would be needed to admit (3,1) at n*:  W >= %.15f"
      % float(Wneed.mid()))
print("  (= (sqrt10 + 2 sqrt5 - 2 + beta)/2; CONTEXT reports the plateau ending at 3.16077)")
claim("that threshold matches CONTEXT's 3.16077 plateau endpoint",
      abs(float(Wneed.mid()) - 3.16077) < 1e-5, "%.15f" % float(Wneed.mid()))

# ============================================================================
hr("5b.  Strengthening: drop the non-barrier test entirely")
# ----------------------------------------------------------------------------
# Lemma A already showed the barrier test cannot bind on a candidate with g <= G.
# So the theorem survives the most generous witness rule imaginable: ANY in-band
# lattice point (however it got certified -- e.g. through a chain of enrichments
# via other barrier voxels) still cannot certify a target at depth eps*.
allrows = [r for r in rows]
claim("all 28 disk candidates satisfy g >= G, admissible or not",
      all(r["rel"] in (">", "=") for r in allrows))
claim("=> for EVERY in-band w (barrier or not), g(w,n*) >= G",
      len(below) == 0 and all(r["rel"] in (">", "=") for r in allrows))
# the last hop of any certification chain is governed by the same inequality
# eps + d_p > |p|  <=>  eps > g(p,n), so chaining cannot help either.

# ============================================================================
hr("6.  Verdict")
# ----------------------------------------------------------------------------
adm_rows = [r for r in rows if r["adm"]]
print("  admissible candidates in the disk: %d of 28" % len(adm_rows))
print("  minimisers of g among them: %s" % sorted(r["w"] for r in ties))
claim("MIN over admissible w of g(w,n*) = G, attained exactly twice, at (1,0) and (2,1)",
      len(below) == 0 and sorted(r["w"] for r in ties) == [(1, 0), (2, 1)] and
      g10 == G_K and g21 == G_K and all(r["adm"] for r in ties))

print()
if FAILS:
    print("*** %d CLAIM(S) FAILED ***" % len(FAILS))
    for f in FAILS:
        print("   -", f)
else:
    print("ALL CLAIMS VERIFIED (exact / certified-interval arithmetic, no float decisions)")
