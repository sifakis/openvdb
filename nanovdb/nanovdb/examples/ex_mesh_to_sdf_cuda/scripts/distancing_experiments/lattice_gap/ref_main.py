#!/usr/bin/env python3
"""Independent referee verification of the proposed tightness proof."""
from fractions import Fraction as F
from ref_lib import *

FAIL = []


def ck(name, cond, note=""):
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (("   " + note) if note else ""))
    if not cond:
        FAIL.append(name)


def hr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


hr("A.  Exact identities in K = Q(beta)")
ck("beta^4 = 4 - 8 beta^2 (reduction rule used)",
   (BETA * BETA * BETA * BETA) == (K([4, 0, 0, 0]) - 8 * B2))
ck("SQRT5^2 = 5", SQRT5 * SQRT5 == K([5, 0, 0, 0]))
ck("cos^2 + sin^2 = 1", COS_K * COS_K + SIN_K * SIN_K == ONE)
ck("cos + sin = sqrt5 - 1", COS_K + SIN_K == SQRT5 - 1)
ck("cos = (sqrt5-1+beta)/2", COS_K == (SQRT5 - 1 + BETA) / 2)
ck("sin = (sqrt5-1-beta)/2", SIN_K == (SQRT5 - 1 - BETA) / 2)
ck("G = (3 - sqrt5 - beta)/4", G_K == (K([3, 0, 0, 0]) - SQRT5 - BETA) / 4)
ck("G = (2 - 2beta - beta^2)/8", G_K == (K([2, 0, 0, 0]) - 2 * BETA - B2) / 8)

g10 = (ONE - COS_K) / 2
g21 = (SQRT5 - 2 * COS_K - SIN_K) / 2
ck("EXACT tie g((1,0)) == g((2,1))", g10 == g21, "%s vs %s" % (g10, g21))
ck("EXACT g((1,0)) == G", g10 == G_K)
d10 = (ONE + COS_K) / 2
d21 = (SQRT5 + 2 * COS_K + SIN_K) / 2
ck("d((1,0)) = 1 - G", d10 == ONE - G_K)
ck("d((2,1)) = (5 sqrt5 - 3 + beta)/4", d21 == (5 * SQRT5 - 3 + BETA) / 4)
ck("3cos+sin = 2 sqrt5 - 2 + beta", 3 * COS_K + SIN_K == 2 * SQRT5 - 2 + BETA)

hr("B.  Certified numeric values")
print("  beta   =", BETA_I.s(20), " width %.1e" % float(BETA_I.width()))
print("  cos a* =", COS_I.s(20))
print("  sin a* =", SIN_I.s(20))
print("  G      =", G_I.s(20), " width %.1e" % float(G_I.width()))
print("  r_d    =", RD.s(20))
alt = (S5 - 1 + BETA_I) / 2
ck("cos via (sqrt5-1+beta)/2 consistent", not (alt.lt(COS_I) or alt.gt(COS_I)))
ck("cos^2+sin^2 numerically = 1",
   not ((COS_I * COS_I + SIN_I * SIN_I).lt(1) or (COS_I * COS_I + SIN_I * SIN_I).gt(1)))
ck("0 < G < r_d (non-vacuity: target is a BARRIER voxel)", G_I.gt(0) and G_I.lt(RD),
   "r_d - G = %.15f" % float((RD - G_I).mid()))
ck("G matches reported 0.019202630763796344",
   abs(float(G_I.mid()) - 0.019202630763796344) < 1e-17)
# closed-form cross check
cf = (S5 - 2) / (2 * (I(1) + S5 + (I(4) + 2 * S5).sqrt()))
ck("CONTEXT closed form equals G", not (cf.lt(G_I) or cf.gt(G_I)),
   "cf = %s" % cf.s(20))

hr("C.  Lemma A: barrier test never binds for g <= G")
ck("1 - G > r_d", (I(1) - G_I).gt(RD),
   "1-G = %.15f, r_d = %.15f" % (float((I(1) - G_I).mid()), float(RD.mid())))

hr("D.  Lemma B: in-band and g <= G  ==>  |w|^2 <= 9")
W = F(3)
R2 = I(W) + 2 * G_I
ck("(W+2G)^2 < 10", (R2 * R2).lt(10), "(W+2G)^2 = %.12f" % float((R2 * R2).mid()))
ck("(W+2G)^2 > 9 (norm-9 vectors DO need enumerating)", (R2 * R2).gt(9))
CAND = [(x, y) for x in range(-3, 4) for y in range(-3, 4)
        if (x, y) != (0, 0) and x * x + y * y <= 9]
ck("disk has 28 nonzero points", len(CAND) == 28, "n=%d" % len(CAND))

hr("E.  Exhaustive certified enumeration, |w|^2 <= 9")
rows = []
for (x, y) in CAND:
    N = x * x + y * y
    L = I(N).sqrt()
    dot = I(x) * COS_I + I(y) * SIN_I
    d = (L + dot) / 2
    g = (L - dot) / 2
    inband = d.le(I(W))
    outband = d.gt(I(W))
    nonbar = d.gt(RD)
    bar = d.lt(RD)
    if not (inband or outband):
        ck("band decidable %s" % str((x, y)), False)
    if not (nonbar or bar):
        ck("barrier decidable %s" % str((x, y)), False)
    rel = ">" if g.gt(G_I) else ("<" if g.lt(G_I) else "=")
    rows.append(dict(w=(x, y), N=N, L=L, dot=dot, d=d, g=g, adm=inband and nonbar,
                     inband=inband, nonbar=nonbar, rel=rel))
rows.sort(key=lambda r: r["g"].mid())
print("      w    N     |w|              w.n              d               g          adm   rel")
for r in rows:
    tag = "yes" if r["adm"] else ("OUTBAND" if not r["inband"] else "BARRIER")
    print("  %8s %2d  %14.12f %15.12f %14.12f %14.12f  %-8s %s" %
          (str(r["w"]), r["N"], r["L"].f(), r["dot"].f(), r["d"].f(), r["g"].f(), tag, r["rel"]))

below = [r for r in rows if r["rel"] == "<"]
ties = sorted(r["w"] for r in rows if r["rel"] == "=")
ck("no disk candidate certifiably below G (even ignoring admissibility)", not below,
   str([r["w"] for r in below]))
ck("exactly (1,0) and (2,1) tie", ties == [(1, 0), (2, 1)], str(ties))
ck("both ties admissible", all(r["adm"] for r in rows if r["rel"] == "="))
ck("all 28 in band (no OUTBAND in disk)", all(r["inband"] for r in rows))
nadm = sum(1 for r in rows if r["adm"])
ck("17 of 28 admissible", nadm == 17, "n=%d" % nadm)
mind = min(r["d"].f() for r in rows if r["nonbar"])
maxd = max(r["d"].f() for r in rows)
print("  max depth in disk = %.12f (slack to W: %.12f)" % (maxd, 3 - maxd))
print("  min non-barrier depth = %.12f (slack to r_d: %.12f)" % (mind, mind - float(RD.mid())))
nxt = min((r for r in rows if r["rel"] == ">"), key=lambda r: r["g"].mid())
print("  next-cheapest: w=%s g=%.15f = %.6f G" % (nxt["w"], nxt["g"].f(), nxt["g"].f() / G_I.f()))

hr("F.  (3,1): the near miss")
L31 = I(10).sqrt()
dot31 = 3 * COS_I + SIN_I
d31 = (L31 + dot31) / 2
g31 = (L31 - dot31) / 2
print("  g((3,1)) = %.15f = G/%.4f" % (g31.f(), G_I.f() / g31.f()))
print("  d((3,1)) = %.15f" % d31.f())
ck("g((3,1)) < G", g31.lt(G_I))
ck("d((3,1)) > W = 3", d31.gt(I(W)), "d - W = %.15f" % float((d31 - I(W)).lo))
ck("eps*+w.n also > 3 (alt convention)", (G_I + dot31).gt(I(W)),
   "%.15f" % (G_I + dot31).f())
ck("|(3,1)| > W+2G (fails Lemma B size test)", L31.gt(R2))
ck("d((3,1)) matches 3.160767557306492", abs(d31.f() - 3.160767557306492) < 1e-14)
ck("sqrt10 - 2G = 3.1238... (limit of the size-test-alone argument)",
   abs(float((L31 - 2 * G_I).mid()) - 3.123872) < 1e-6, "%.15f" % float((L31 - 2 * G_I).mid()))

hr("G.  Corollary (iii): plateau in W at n*, via disk |w|^2 <= 10")
Wtop = d31
ck("(Wtop + G)^2 < 11", ((Wtop + G_I) * (Wtop + G_I)).lt(11),
   "%.12f" % float(((Wtop + G_I) * (Wtop + G_I)).mid()))
C10 = [(x, y) for x in range(-4, 5) for y in range(-4, 5)
       if (x, y) != (0, 0) and x * x + y * y <= 10]
ck("disk |w|^2<=10 has 36 points", len(C10) == 36, "n=%d" % len(C10))
bad = []
for (x, y) in C10:
    N = x * x + y * y
    L = I(N).sqrt()
    dot = I(x) * COS_I + I(y) * SIN_I
    g = (L - dot) / 2
    if g.lt(G_I):
        bad.append(((x, y), g.f(), ((L + dot) / 2).f()))
ck("only (3,1) has g < G inside |w|^2<=10", [b[0] for b in bad] == [(3, 1)], str(bad))

hr("H.  Chain argument: any p with g(p) < G is out of band")
# if g(p) < G and d(p) <= 3 then |p| = d+g <= 3+G < 3.02 -> |p|^2 <= 9 -> contradiction
ck("(W+G)^2 < 10", ((I(W) + G_I) * (I(W) + G_I)).lt(10),
   "%.12f" % float(((I(W) + G_I) * (I(W) + G_I)).mid()))

print()
print(("*** %d FAILED ***" % len(FAIL)) if FAIL else "ALL REFEREE CHECKS PASSED")
for f in FAIL:
    print("   -", f)
