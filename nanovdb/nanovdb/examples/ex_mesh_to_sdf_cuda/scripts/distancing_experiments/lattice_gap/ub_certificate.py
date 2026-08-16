#!/usr/bin/env python3
"""
UPPER BOUND CERTIFICATE for the 2D lattice certification-gap problem, W = 3.

THEOREM (upper bound).  Let W = 3, r_d = sqrt(2)/2.  For every unit vector n
there is an integer vector w != 0 which is admissible for n, i.e.

        r_d < d0(w,n) := (|w| + w.n)/2 <= W,

and whose cost satisfies

        g(w,n) := (|w| - w.n)/2 <= E1 := (3 - sqrt5 - sqrt(2*sqrt5 - 4))/4
                                       = 0.019202630763796343828423710859901657...

Only three witness directions (and their images under the 8-fold symmetry of
Z^2) are ever needed:  (1,0), (2,1), (1,1).

Everything below is exact: rational (Fraction) interval arithmetic with
certified rational enclosures of every surd.  No floating point is used in the
certificate itself (floats appear only in the printed decimal echoes and in the
independent sweep at the end of the companion script ub_sweep.py).

Parametrisation.  n is swept with the tangent half-angle t = tan(alpha/2):

        n(t) = ( (1-t^2)/(1+t^2), 2t/(1+t^2) )   -- rational in t.

Key identities used (all proved in the write-up):
    g(w,n)  = (|w| - w.n)/2 = |w| sin^2(theta_w/2),   d0(w,n) = |w| - g(w,n)
    g(w,.)  is AFFINE in n  =>  the crossover {g_w = g_w'} is the LINE
                                (w - w').n = |w| - |w'|.
    g(w, n(alpha)) = |w| (1 - cos(alpha - beta_w))/2, beta_w = angle of w,
        hence unimodal in alpha with minimum 0 at alpha = beta_w:  its max over
        any interval of length < 2pi containing no wrap is at an endpoint.
"""
from fractions import Fraction as F
from math import isqrt

DIG = 60                       # decimal digits used for surd enclosures
FAIL = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)
    return cond


# ---------------------------------------------------------------------------
# 1.  certified rational enclosures of square roots
# ---------------------------------------------------------------------------
def sqrt_encl(x, d=DIG):
    """x: nonneg Fraction.  Returns (lo,hi) Fractions with lo^2 <= x <= hi^2."""
    assert x >= 0
    p, q = x.numerator, x.denominator
    n = p * q * 10 ** (2 * d)
    r = isqrt(n)
    lo = F(r, q * 10 ** d)
    hi = F(r + 1, q * 10 ** d)
    assert lo * lo <= x <= hi * hi
    return lo, hi


def isqrt_encl(iv, d=DIG):
    """sqrt of an interval (lo,hi) of nonneg rationals -> enclosing interval."""
    return sqrt_encl(iv[0], d)[0], sqrt_encl(iv[1], d)[1]


def iadd(a, b): return (a[0] + b[0], a[1] + b[1])
def isub(a, b): return (a[0] - b[1], a[1] - b[0])
def rmul(c, a): return (c * a[0], c * a[1]) if c >= 0 else (c * a[1], c * a[0])
def const(x):   return (F(x), F(x))


SQRT2 = sqrt_encl(F(2))
SQRT5 = sqrt_encl(F(5))
SQRT10 = sqrt_encl(F(10))
RD = (SQRT2[0] / 2, SQRT2[1] / 2)          # barrier radius sqrt(2)/2
W = F(3)                                   # band half-width

# |w| enclosures for the three witnesses
NORM = {(1, 0): const(1), (1, 1): SQRT2, (2, 1): SQRT5}


def n_of_t(t):
    """exact rational unit-ish vector n(t) = ((1-t^2)/(1+t^2), 2t/(1+t^2))."""
    d = 1 + t * t
    return (1 - t * t) / d, 2 * t / d


def g_iv(w, t):
    """exact interval for g(w, n(t)) at a rational t."""
    nx, ny = n_of_t(t)
    dot = F(w[0]) * nx + F(w[1]) * ny
    lo, hi = NORM[w]
    return ((lo - dot) / 2, (hi - dot) / 2)


def d0_iv(w, t):
    """exact interval for the witness depth d0(w, n(t)) = (|w| + w.n)/2."""
    nx, ny = n_of_t(t)
    dot = F(w[0]) * nx + F(w[1]) * ny
    lo, hi = NORM[w]
    return ((lo + dot) / 2, (hi + dot) / 2)


print("=" * 78)
print("0.  sanity of the exact-arithmetic toolkit")
print("=" * 78)
check("sqrt2 enclosure is sound and tight",
      SQRT2[0] ** 2 <= 2 <= SQRT2[1] ** 2
      and F(14142135623730950488, 10 ** 19) < SQRT2[0]
      and SQRT2[1] < F(14142135623730950489, 10 ** 19),
      f"width {float(SQRT2[1]-SQRT2[0]):.1e}")
check("sqrt5, sqrt10 enclosures sound",
      SQRT5[0] ** 2 <= 5 <= SQRT5[1] ** 2 and SQRT10[0] ** 2 <= 10 <= SQRT10[1] ** 2)
check("n(t) is exactly a unit vector for rational t",
      all((lambda nx, ny: nx * nx + ny * ny == 1)(*n_of_t(F(a, b)))
          for a, b in [(0, 1), (1, 3), (7, 17), (413, 997)]))

# ---------------------------------------------------------------------------
# 2.  Lemma A -- admissibility reduces to |w| <= W  (given a small cost)
# ---------------------------------------------------------------------------
print("=" * 78)
print("2.  Lemma A: if 0 <= g(w,n) <= E and 1 <= |w| <= W then w is admissible")
print("=" * 78)
# d0 = |w| - g  in  [ |w| - E , |w| ]  subset  ( r_d , W ]  provided
#       |w| - E > sqrt2/2   (worst case |w| = 1)   and   |w| <= W.
E_UB = F(193, 10000)                      # 0.0193 > E1, a crude safe bound
check("1 - 0.0193 > sqrt(2)/2 (non-barrier holds for every |w|>=1)",
      1 - E_UB > RD[1], f"1-E={float(1-E_UB):.6f} > {float(RD[1]):.6f}")
check("sqrt5 <= W = 3 (in-band holds for all three witnesses)", SQRT5[1] <= W)
check("g >= 0 always (Cauchy-Schwarz, |n|=1)  -- symbolic, see write-up", True)
# Stronger remark, checked in section 4b below:
#   the pairwise rule is only ever NEEDED for a target that is itself a barrier
#   voxel, eps <= r_d.  Then the witness depth in the ACTUAL configuration is
#   d0_act = eps + w.n <= r_d + |w| <= sqrt2/2 + sqrt5 = 2.9432 < 3 = W, so the
#   witness is in band for the actual configuration too, not only at threshold.
check("r_d + sqrt5 < W = 3  (actual-configuration band margin)",
      RD[1] + SQRT5[1] < W, f"{float(RD[1]+SQRT5[1]):.6f} < 3")

# ---------------------------------------------------------------------------
# 3.  exact crossovers
# ---------------------------------------------------------------------------
print("=" * 78)
print("3.  exact crossovers (g affine in n  =>  crossover locus is a line)")
print("=" * 78)
# Gap 1: g_(1,0) = g_(2,1)  <=>  nx + ny = sqrt5 - 1
#        with nx^2+ny^2 = 1 and nx > ny > 0:
#        nx = ((sqrt5-1) + sqrt(2 sqrt5 - 4))/2,  ny = ((sqrt5-1) - sqrt(...))/2
#        E1 = (1 - nx)/2 = (3 - sqrt5 - sqrt(2 sqrt5 - 4))/4
S = isub(SQRT5, const(1))                       # sqrt5 - 1
Q = isqrt_encl(isub(rmul(F(2), SQRT5), const(4)))   # sqrt(2 sqrt5 - 4)
NX1 = rmul(F(1, 2), iadd(S, Q))
NY1 = rmul(F(1, 2), isub(S, Q))
E1 = rmul(F(1, 2), isub(const(1), NX1))         # (1-nx)/2
E1_alt = rmul(F(1, 4), isub(isub(const(3), SQRT5), Q))
# cross-check with the second witness: E1 = (sqrt5 - 2nx - ny)/2
E1_b = rmul(F(1, 2), isub(SQRT5, iadd(rmul(F(2), NX1), NY1)))
check("crossover-1 point is on the unit circle",
      (NX1[0] ** 2 + NY1[0] ** 2) < 1 < (NX1[1] ** 2 + NY1[1] ** 2))
check("E1 via w=(1,0) and via w=(2,1) agree (intervals overlap)",
      E1[0] <= E1_b[1] and E1_b[0] <= E1[1])
check("E1 == (3-sqrt5-sqrt(2 sqrt5-4))/4", E1[0] <= E1_alt[1] and E1_alt[0] <= E1[1])
print(f"      E1 in [{float(E1[0]):.18f}, {float(E1[1]):.18f}]  width {float(E1[1]-E1[0]):.1e}")

# Gap 2: g_(2,1) = g_(1,1)  <=>  nx = sqrt5 - sqrt2
NX2 = isub(SQRT5, SQRT2)
NY2 = isqrt_encl(isub(const(1), (NX2[1] ** 2, NX2[0] ** 2)))
E2 = rmul(F(1, 2), isub(SQRT2, iadd(NX2, NY2)))
E2_b = rmul(F(1, 2), isub(SQRT5, iadd(rmul(F(2), NX2), NY2)))
check("E2 via w=(1,1) and via w=(2,1) agree", E2[0] <= E2_b[1] and E2_b[0] <= E2[1])
print(f"      E2 in [{float(E2[0]):.18f}, {float(E2[1]):.18f}]")
check("E2 < E1  (the WIDE gap dominates)", E2[1] < E1[0],
      f"E1-E2 ~ {float(E1[0]-E2[1]):.10f}")

# CONTEXT's closed form, exactly re-derived:  (sqrt5-2)/(2(1+sqrt5+sqrt(4+2sqrt5)))
CF_num = isub(SQRT5, const(2))
CF_den = rmul(F(2), iadd(iadd(const(1), SQRT5), isqrt_encl(iadd(const(4), rmul(F(2), SQRT5)))))
CF = (CF_num[0] / CF_den[1], CF_num[1] / CF_den[0])
check("CONTEXT closed form == E1", CF[0] <= E1[1] and E1[0] <= CF[1])

STATED = F(19202630763796, 10 ** 15)      # 0.019202630763796 from CONTEXT
print(f"      stated decimal 0.019202630763796 vs exact E1:")
print(f"      E1 - stated in [{float(E1[0]-STATED):.3e}, {float(E1[1]-STATED):.3e}]")
check("HONESTY CHECK: exact E1 is STRICTLY GREATER than the stated 15-digit decimal",
      E1[0] > STATED, "the stated value is E1 truncated, not rounded up")

# ---------------------------------------------------------------------------
# 4.  the covering certificate over the fundamental domain
# ---------------------------------------------------------------------------
print("=" * 78)
print("4.  covering certificate for alpha in [0,45deg]  (t = tan(alpha/2))")
print("=" * 78)
# rational nodes.  t*_1 = tan(alpha1/2), t*_2 = tan(alpha2/2) are irrational;
# we bracket them by rational t-, t+ using the EXACT sign tests
#      D1 < 0  <=>  nx + ny < sqrt5 - 1        (g_(1,0) < g_(2,1))
#      D2 < 0  <=>  nx > sqrt5 - sqrt2         (g_(2,1) < g_(1,1))
def D1_sign(t):
    nx, ny = n_of_t(t)
    if nx + ny < S[0]:
        return -1
    if nx + ny > S[1]:
        return +1
    return 0


def D2_sign(t):
    nx, _ = n_of_t(t)
    if nx > NX2[1]:
        return -1
    if nx < NX2[0]:
        return +1
    return 0


def bracket(sign_fn, lo, hi, iters=200):
    assert sign_fn(lo) == -1 and sign_fn(hi) == +1
    for _ in range(iters):
        m = (lo + hi) / 2
        s = sign_fn(m)
        if s == -1:
            lo = m
        elif s == +1:
            hi = m
        else:
            break
        if hi - lo < F(1, 10 ** 40):
            break
    return lo, hi


t1m, t1p = bracket(D1_sign, F(0), F(1, 4))
t2m, t2p = bracket(D2_sign, F(1, 4), F(2, 5))
TMAX = F(41422, 100000)                      # > tan(22.5deg) = sqrt2 - 1
check("TMAX > tan(22.5deg) = sqrt2 - 1 (so the cover reaches past alpha=45deg)",
      F(TMAX) > SQRT2[1] - 1)
check("t1-, t1+ bracket the first crossover", D1_sign(t1m) == -1 and D1_sign(t1p) == +1,
      f"t1 in [{float(t1m):.20f}, {float(t1p):.20f}]")
check("t2-, t2+ bracket the second crossover", D2_sign(t2m) == -1 and D2_sign(t2p) == +1,
      f"t2 in [{float(t2m):.20f}, {float(t2p):.20f}]")
check("crossover order 0 < t1- < t1+ < t2- < t2+ < TMAX",
      0 < t1m < t1p < t2m < t2p < TMAX)
# uniqueness of the crossovers on the octant, and root selection:
#   D1 ~ nx+ny = sqrt2 cos(alpha-45deg) strictly INCREASING on [0,45deg)
#   D2 ~ -nx   = -cos alpha             strictly INCREASING on [0,45deg)
GRID = [TMAX * F(i, 4000) for i in range(4001)]
s_prev = None
mono1 = mono2 = True
for t in GRID:
    nx, ny = n_of_t(t)
    if s_prev is not None:
        mono1 = mono1 and (nx + ny) > s_prev[0]
        mono2 = mono2 and nx < s_prev[1]
    s_prev = (nx + ny, nx)
check("nx+ny strictly increasing on the octant (=> crossover 1 unique)", mono1)
check("nx strictly decreasing on the octant (=> crossover 2 unique)", mono2)
check("bracket 1 lies in the region nx > ny (selects the '+' root of the "
      "quadratic, i.e. alpha < 45deg)",
      all(n_of_t(t)[0] > n_of_t(t)[1] for t in (t1m, t1p)))
check("both brackets lie inside the octant t in (0, tan(22.5deg))",
      t1p < SQRT2[0] - 1 and t2p < SQRT2[0] - 1)


def beta_side(w, t):
    """sign of (alpha(t) - beta_w) via the exact rational test  ny*wx - nx*wy."""
    nx, ny = n_of_t(t)
    v = ny * F(w[0]) - nx * F(w[1])
    return (v > 0) - (v < 0)


def max_g_on(w, ta, tb):
    """certified upper bound for max of g(w,n(t)) over t in [ta,tb].
    g(w,alpha) = |w|(1-cos(alpha-beta_w))/2 is unimodal with min at beta_w, so
    the max over the interval is attained at an endpoint."""
    return max(g_iv(w, ta)[1], g_iv(w, tb)[1])


def depth_range_on(w, ta, tb):
    """certified [lo,hi] for d0 over [ta,tb]:  d0 = |w| - g, g in [0, max_g]."""
    mg = max_g_on(w, ta, tb)
    lo, hi = NORM[w]
    return (lo - mg, hi)


# --- the six pieces ---------------------------------------------------------
# On each piece we name ONE witness (the crossover pieces are split at the
# exact irrational crossover, which is legitimate because on each side the
# corresponding g is monotone toward the crossover).
#
# mode 'num'  : max g over the piece is certified numerically at the endpoints
# mode 'xover': the piece ends (or starts) at the EXACT irrational crossover
#               t1* / t2*.  There max g = g(w, t*) = E1 (resp. E2) EXACTLY, by
#               definition of the crossover plus monotonicity of g(w,.) toward
#               t*.  We therefore take the analytic value; the numeric value at
#               the rational bracket endpoint is printed as "slack" to show how
#               tight the bracket is (it only ever exceeds E by ~1e-41).
PIECES = [
    ("[0, t1-]",    (1, 0), F(0), t1m, E1, 'num',   None),
    ("[t1-, t1*]",  (1, 0), t1m,  t1p, E1, 'xover', 'up'),
    ("[t1*, t1+]",  (2, 1), t1m,  t1p, E1, 'xover', 'down'),
    ("[t1+, t2-]",  (2, 1), t1p,  t2m, E1, 'num',   None),
    ("[t2-, t2*]",  (2, 1), t2m,  t2p, E2, 'xover', 'up'),
    ("[t2*, t2+]",  (1, 1), t2m,  t2p, E2, 'xover', 'down'),
    ("[t2+, TMAX]", (1, 1), t2p,  TMAX, E2, 'num',  None),
]
print()
print(f"  {'piece':<14}{'w':>8}{'max g (certified)':>24}{'<=E1':>6}"
      f"{'depth d0 range':>26}{'adm':>5}  note")
allok = True
for name, w, ta, tb, Eiv, mode, mono in PIECES:
    if mode == 'num':
        mg = max_g_on(w, ta, tb)            # rational, certified
        ok_g = mg <= E1[1]
        note = "endpoint max"
    else:
        mg = Eiv[1]                         # analytic: max g = E exactly
        ok_g = Eiv[1] <= E1[1]
        far = t1p if (name.startswith("[t1-") ) else (
              t1m if name.startswith("[t1*") else (
              t2p if name.startswith("[t2-") else t2m))
        slack = g_iv(w, far)[1] - Eiv[0]
        note = f"analytic (bracket slack {float(slack):.1e})"
    dlo, dhi = depth_range_on(w, ta, tb)
    dlo = min(dlo, NORM[w][0] - max(mg, max_g_on(w, ta, tb)))
    ok_d = (dlo > RD[1]) and (dhi <= W)
    allok = allok and ok_g and ok_d
    print(f"  {name:<14}{str(w):>8}{float(mg):>24.18f}{str(ok_g):>6}"
          f"  [{float(dlo):.6f}, {float(dhi):.6f}]{str(ok_d):>5}  {note}")
check("every piece: certified max g <= E1 AND witness admissible throughout", allok)
# the pieces must chain up and cover [0,TMAX]
chain = [F(0), t1m, t1p, t2m, t2p, TMAX]
check("pieces tile [0,TMAX] with no hole", all(chain[i] <= chain[i + 1]
                                               for i in range(len(chain) - 1)))

# the two crossover pieces additionally need the monotonicity direction; check it
check("g((1,0),.) increasing on [0,t1+]  (alpha < beta=0 never; alpha>0)",
      beta_side((1, 0), t1p) == +1 and beta_side((1, 0), F(1, 10 ** 9)) == +1)
check("g((2,1),.) decreasing on [t1-, t_phi]  (alpha < beta=arctan(1/2) there)",
      beta_side((2, 1), t1m) == -1 and beta_side((2, 1), t1p) == -1)
check("g((2,1),.) increasing at t2-,t2+ (alpha > arctan(1/2) there)",
      beta_side((2, 1), t2m) == +1 and beta_side((2, 1), t2p) == +1)
check("g((1,1),.) decreasing on [t2-,TMAX] (alpha < 45deg... up to TMAX)",
      beta_side((1, 1), t2m) == -1 and beta_side((1, 1), t2p) == -1)

# ---------------------------------------------------------------------------
# 4b.  admissibility of ALL THREE witnesses over the WHOLE octant
# ---------------------------------------------------------------------------
print("=" * 78)
print("4b. admissibility over the whole octant (not merely at the crossovers)")
print("=" * 78)
print(f"  {'w':>8}{'max g on octant':>20}{'min d0':>14}{'max d0':>14}"
      f"{'  > r_d':>9}{'  <= W':>8}{'  eps+w.n <= W':>16}")
okall = True
for w in [(1, 0), (2, 1), (1, 1)]:
    mg = max_g_on(w, F(0), TMAX)          # endpoint max, g unimodal
    dlo = NORM[w][0] - mg
    dhi = NORM[w][1]
    act = RD[1] + NORM[w][1]              # eps + w.n <= r_d + |w| for barrier targets
    ok = (dlo > RD[1]) and (dhi <= W) and (act <= W)
    okall = okall and ok
    print(f"  {str(w):>8}{float(mg):>20.12f}{float(dlo):>14.9f}{float(dhi):>14.9f}"
          f"{str(dlo > RD[1]):>9}{str(dhi <= W):>8}{str(act <= W):>16}")
check("all three witnesses are admissible for EVERY n in the octant "
      "(threshold form AND actual-configuration form)", okall)

# ---------------------------------------------------------------------------
# 5.  redundant fine uniform cover (independent of the 6-piece argument)
# ---------------------------------------------------------------------------
print("=" * 78)
print("5.  redundant FINE uniform cover of [0,TMAX] with exact rational arithmetic")
print("=" * 78)
N = 20000
bad = []
worst_slack = None
for i in range(N):
    ta = TMAX * i // N if False else TMAX * F(i, N)
    tb = TMAX * F(i + 1, N)
    best = None
    for w in [(1, 0), (2, 1), (1, 1)]:
        mg = max_g_on(w, ta, tb)
        if best is None or mg < best[0]:
            best = (mg, w)
    slack = best[0] - E1[1]
    if worst_slack is None or slack > worst_slack[0]:
        worst_slack = (slack, i, best[1])
    if slack > 0:
        bad.append((i, float(ta), float(tb), best, float(slack)))
print(f"  N = {N} subintervals; subintervals where NO single witness stays <= E1:"
      f" {len(bad)}")
for b in bad:
    print(f"    i={b[0]:6d} t in [{b[1]:.12f},{b[2]:.12f}] best w={b[3][1]}"
          f" max g-E1 = {b[4]:.3e}")
print(f"  (these are exactly the subintervals straddling a crossover; their")
print(f"   overshoot is bounded by the subinterval width and they are covered")
print(f"   exactly by pieces 2,3,5,6 above.)")
mx = max(float(b[4]) for b in bad) if bad else 0.0
check("all failing subintervals straddle a certified crossover bracket",
      all(float(b[1]) <= float(t1p) and float(b[2]) >= float(t1m)
          or float(b[1]) <= float(t2p) and float(b[2]) >= float(t2m) for b in bad),
      f"max overshoot {mx:.3e} <= subinterval width * |dg/dt|")

# ---------------------------------------------------------------------------
# 6.  adversarial EXACT pointwise falsification search
#     (a pointwise counterexample would kill the theorem outright)
# ---------------------------------------------------------------------------
print("=" * 78)
print("6.  adversarial exact pointwise search for min_w g(w,n(t)) > E1")
print("=" * 78)
import random
random.seed(12345)
probes = []
# (a) uniform random rationals in [0,TMAX]
probes += [TMAX * F(random.randrange(10 ** 18), 10 ** 18) for _ in range(120000)]
# (b) exponentially concentrated around both crossovers
for tc in (t1m, t2m):
    for k in range(1, 40):
        d = F(1, 10 ** k)
        probes += [tc + d, tc - d, tc + 3 * d, tc - 3 * d, tc + 7 * d, tc - 7 * d]
probes = [t for t in probes if 0 <= t <= TMAX]
viol = []
mx = None
for t in probes:
    m = min(g_iv(w, t)[0] for w in [(1, 0), (2, 1), (1, 1)])   # LOWER bound of min
    if mx is None or m > mx[0]:
        mx = (m, t)
    if m > E1[1]:
        viol.append((t, m))
print(f"  probes: {len(probes)} exact rational directions")
print(f"  largest observed min-cost = {float(mx[0]):.18f}  at t = {float(mx[1]):.15f}")
print(f"  E1                        = {float(E1[1]):.18f}")
check("no exact pointwise counterexample found", not viol,
      f"{len(viol)} violations" if viol else f"max-min - E1 = {float(mx[0]-E1[1]):.3e}")

print("=" * 78)
print("RESULT:", "ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}")
print("=" * 78)
