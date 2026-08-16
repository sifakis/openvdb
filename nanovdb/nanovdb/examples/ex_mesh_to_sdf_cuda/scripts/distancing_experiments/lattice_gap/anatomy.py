#!/usr/bin/env python3
"""
ANATOMY OF THE WORST CASE at W = 3.

Everything about the critical configuration is algebraic; no trig is needed to
define alpha*.  Derivation:

    g(w,alpha) = (|w| - w.n)/2 ,  n = (cos a, sin a)
    tie (1,0) vs (2,1):  (1 - c)/2 = (sqrt5 - 2c - s)/2
                    <=>  1 + c + s = sqrt5
                    <=>  c + s = sqrt5 - 1 =: k
    with c^2+s^2 = 1:    2cs = k^2 - 1 = 5 - 2 sqrt5
                         (c-s)^2 = 1 - 2cs = 2 sqrt5 - 4 =: m2   (c>s on (0,45deg))
    =>  c = (k + sqrt(m2))/2 ,  s = (k - sqrt(m2))/2
    =>  eps* = (1-c)/2 = (3 - sqrt5 - sqrt(2 sqrt5 - 4)) / 4

High-precision (Decimal, 60 digits) plus a full double-precision cross-check and
an independent numerical global maximisation over alpha.
"""
import math, json
from decimal import Decimal as D, getcontext

getcontext().prec = 70

# ----------------------------------------------------------------- utilities
def dsqrt(x):  return D(x).sqrt()

def dsin(x):
    s, t, n = D(0), D(x), 0
    while True:
        s += t; n += 1
        t = -t * x * x / (D(2 * n) * D(2 * n + 1))
        if abs(t) < D(10) ** (-getcontext().prec + 5): break
    return s

def dcos(x):
    s, t, n = D(0), D(1), 0
    while True:
        s += t; n += 1
        t = -t * x * x / (D(2 * n - 1) * D(2 * n))
        if abs(t) < D(10) ** (-getcontext().prec + 5): break
    return s

def datan(x):
    """arctan for |x| < 1 via the alternating series (x ~ 0.285 here)."""
    x = D(x); s, t, n = D(0), D(x), 0
    x2 = x * x
    while True:
        s += t / D(2 * n + 1); n += 1
        t = -t * x2
        if abs(t) < D(10) ** (-getcontext().prec + 5): break
    return s

S5   = dsqrt(5)
S2   = dsqrt(2)
S10  = dsqrt(10)
BARR = S2 / 2                       # lattice covering radius sqrt(2)/2
W    = D(3)

# ------------------------------------------------------- exact critical data
M2    = 2 * S5 - 4                  # (c - s)^2
SM2   = dsqrt(M2)
K     = S5 - 1                      # c + s
COS_A = (K + SM2) / 2
SIN_A = (K - SM2) / 2
EPS   = (1 - COS_A) / 2             # = (3 - sqrt5 - sqrt(2 sqrt5 - 4))/4
ALPHA = datan(SIN_A / COS_A)        # radians
PI    = D("3.14159265358979323846264338327950288419716939937510582097494459")
ALPHA_DEG = ALPHA * 180 / PI

EPS_CLOSED_MINE   = (3 - S5 - SM2) / 4
EPS_CLOSED_CTX    = (S5 - 2) / (2 * (1 + S5 + dsqrt(4 + 2 * S5)))

print("=" * 96)
print("0.  EXACT CRITICAL DIRECTION  (Decimal, 60 significant digits)")
print("=" * 96)
print(f"  cos a*  = {COS_A}")
print(f"  sin a*  = {SIN_A}")
print(f"  c+s     = {COS_A + SIN_A}   (should be sqrt5-1 = {S5 - 1})")
print(f"  c^2+s^2 = {COS_A * COS_A + SIN_A * SIN_A}")
print(f"  alpha*  = {ALPHA} rad = {ALPHA_DEG} deg")
print(f"  CONTEXT claims alpha* = 15.930625116 deg   ->  delta = "
      f"{float(ALPHA_DEG - D('15.930625116')):+.3e} deg")
print(f"  eps*                 = {EPS}")
print(f"  eps* (my surd)       = {EPS_CLOSED_MINE}   delta={float(abs(EPS-EPS_CLOSED_MINE)):.2e}")
print(f"  eps* (CONTEXT surd)  = {EPS_CLOSED_CTX}    delta={float(abs(EPS-EPS_CLOSED_CTX)):.2e}")
print(f"  CONTEXT claims eps* = 0.019202630763796  ->  delta = "
      f"{float(EPS - D('0.019202630763796')):+.3e}")
# CONTEXT's second closed form: eps* = sin^2(u), tan u = 5^(1/4) sin(p/2)/(1+5^(1/4) cos(p/2))
phi   = datan(D(1) / 2)
kq    = D(5) ** D("0.25")
u     = datan(kq * dsin(phi / 2) / (1 + kq * dcos(phi / 2)))
print(f"  eps* (sin^2 u form)  = {dsin(u)**2}   delta={float(abs(EPS-dsin(u)**2)):.2e}")
print(f"  alpha* = 2u          = {2 * u * 180 / PI} deg   "
      f"delta={float(abs(ALPHA_DEG - 2*u*180/PI)):.2e} deg")

# double-precision mirrors
ca, sa = float(COS_A), float(SIN_A)
alpha_f, eps_f = float(ALPHA), float(EPS)
Wf, BARf = 3.0, math.sqrt(2) / 2

# ------------------------------------------------------------------ helpers
def cost_and_depth(a, b, ca, sa):
    wl = math.hypot(a, b)
    wn = a * ca + b * sa
    return (wl - wn) / 2, (wl + wn) / 2, wl, wn      # g, depth-at-threshold, |w|, w.n

def status(depth, Wv=Wf):
    if depth <= BARf:  return "barrier"
    if depth <= Wv:    return "in-band"
    return "out-of-band"

# ============================================================================
print()
print("=" * 96)
print("(a)  ALL LATTICE POINTS WITH |w| <= 5  AT alpha* , W = 3")
print("=" * 96)
rows = []
for a in range(-5, 6):
    for b in range(-5, 6):
        if (a, b) == (0, 0): continue
        wl = math.hypot(a, b)
        if wl > 5 + 1e-12: continue
        g, depth, wl, wn = cost_and_depth(a, b, ca, sa)
        th = math.degrees(math.acos(max(-1.0, min(1.0, wn / wl))))
        st = status(depth)
        gcd = math.gcd(abs(a), abs(b))
        rows.append(dict(w=[a, b], norm=wl, theta_deg=th, depth=depth, g=g,
                         status=st, primitive=(gcd == 1)))
rows.sort(key=lambda r: r["g"])
print(f"  {'w':>10} {'|w|':>9} {'theta(deg)':>11} {'depth@thr':>11} {'cost g':>15}  "
      f"{'status':<12} prim")
for r in rows:
    print(f"  {str(tuple(r['w'])):>10} {r['norm']:9.6f} {r['theta_deg']:11.6f} "
          f"{r['depth']:11.6f} {r['g']:15.12f}  {r['status']:<12} "
          f"{'y' if r['primitive'] else 'n'}")
print(f"  ({len(rows)} lattice points; "
      f"{sum(1 for r in rows if r['status']=='in-band')} admissible, "
      f"{sum(1 for r in rows if r['status']=='barrier')} barrier, "
      f"{sum(1 for r in rows if r['status']=='out-of-band')} out-of-band)")

# rigorous truncation bound: admissible => depth<=W => w.n <= 2W-|w| => g >= |w|-W
print(f"\n  Truncation bound: admissible w has g = (|w|-w.n)/2 >= |w| - W, so any w with")
print(f"  |w| > W + eps* = {3 + eps_f:.9f} cannot beat eps*.  |w| <= 5 covers it with room.")
big = []
for a in range(-9, 10):
    for b in range(-9, 10):
        if (a, b) == (0, 0): continue
        g, depth, wl, wn = cost_and_depth(a, b, ca, sa)
        if BARf < depth <= Wf: big.append((g, (a, b)))
big.sort()
print(f"  Exhaustive |a|,|b|<=9 admissible minimum: g={big[0][0]:.15f} at w={big[0][1]}, "
      f"runner-up g={big[1][0]:.15f} at w={big[1][1]}")

# ============================================================================
print()
print("=" * 96)
print("(b)  THE TIE  (1,0) vs (2,1)  AND THE KINK")
print("=" * 96)
g10_d = (1 - COS_A) / 2
g21_d = (S5 - 2 * COS_A - SIN_A) / 2
print(f"  exact/60-digit:  g(1,0) = {g10_d}")
print(f"                   g(2,1) = {g21_d}")
print(f"                   difference = {g10_d - g21_d}")
g10_f = (1.0 - ca) / 2
g21_f = (math.sqrt(5) - 2 * ca - sa) / 2
print(f"  double:          g(1,0) = {g10_f!r}")
print(f"                   g(2,1) = {g21_f!r}")
print(f"                   hex    = {g10_f.hex()}  /  {g21_f.hex()}")
print(f"                   bitwise identical: {g10_f == g21_f}")
print(f"                   |diff| = {abs(g10_f-g21_f):.3e}   "
      f"ulp(g) = {math.ulp(g10_f):.3e}   diff in ulps = "
      f"{abs(g10_f-g21_f)/math.ulp(g10_f):.2f}")
# also evaluate through cos/sin of the double alpha (independent route)
ca2, sa2 = math.cos(alpha_f), math.sin(alpha_f)
g10_b = (1.0 - ca2) / 2
g21_b = (math.sqrt(5) - 2 * ca2 - sa2) / 2
print(f"  via cos/sin(alpha*_double):  g10={g10_b!r}  g21={g21_b!r}  "
      f"diff={abs(g10_b-g21_b):.3e} ({abs(g10_b-g21_b)/math.ulp(g10_b):.2f} ulp)")

# one-sided derivatives:  dg/dalpha = (wx sin a - wy cos a)/2
dg10 = (1 * SIN_A - 0 * COS_A) / 2
dg21 = (2 * SIN_A - 1 * COS_A) / 2
print(f"\n  dg/dalpha (per radian):   (1,0): {dg10}")
print(f"                            (2,1): {dg21}")
print(f"  per degree:               (1,0): {dg10*PI/180}")
print(f"                            (2,1): {dg21*PI/180}")
print(f"  eps*(alpha)=min(...):  left derivative  = {dg10}  (>0, from (1,0))")
print(f"                         right derivative = {dg21}  (<0, from (2,1))")
print(f"  kink magnitude (jump in derivative) = {dg21-dg10} per rad "
      f"= {float((dg21-dg10)*PI/180):.12f} per deg")
print(f"  ratio |right|/|left| = {float(-dg21/dg10):.12f}")
# empirical check of the one-sided slopes
for h in (1e-5, 1e-6, 1e-7):
    def epsdir(a):
        c, s = math.cos(a), math.sin(a); best = 1e9
        for x in range(-9, 10):
            for y in range(-9, 10):
                if (x, y) == (0, 0): continue
                wl = math.hypot(x, y); wn = x * c + y * s
                g, d = (wl - wn) / 2, (wl + wn) / 2
                if BARf < d <= Wf and g < best: best = g
        return best
    L = (eps_f - epsdir(alpha_f - h)) / h
    R = (epsdir(alpha_f + h) - eps_f) / h
    print(f"    numeric h={h:.0e}:  left slope {L:+.9f}   right slope {R:+.9f}")

# ============================================================================
print()
print("=" * 96)
print("(c)  GEOMETRY OF THE MARGINAL CONFIGURATION")
print("=" * 96)
# surface frame: Sigma = x-axis, exterior = +y ; n = e_y ; t = 90deg CCW from n
# lattice frame: v1 = 0, v0 = w, n = (cos a*, sin a*), t = (-sin a*, cos a*)
geo = {}
for w in [(1, 0), (2, 1)]:
    wx, wy = w
    wl = dsqrt(D(wx * wx + wy * wy))
    wn = wx * COS_A + wy * SIN_A
    wt = -wx * SIN_A + wy * COS_A
    d1 = EPS
    d0 = EPS + wn
    dthr = (wl + wn) / 2
    dx = 2 * dsqrt(d0 * d1)
    dist = wl
    # tangency point of the two balls (lattice frame): T = w * d1/|w|
    Tlx, Tly = D(wx) * d1 / wl, D(wy) * d1 / wl
    # surface frame coords
    v1_s = (D(0), d1)
    v0_s = (wt, d0)
    T_s = (wt * d1 / wl, d1 + (d0 - d1) * d1 / wl)
    P1_s = (D(0), D(0))
    P0_s = (wt, D(0))
    geo[str(w)] = dict(
        w=[wx, wy], norm=float(wl), w_dot_n=float(wn), w_dot_t=float(wt),
        d1=float(d1), d0=float(d0), depth_at_threshold=float(dthr),
        d0_minus_dthr=float(d0 - dthr),
        theta_deg=float(datan(abs(wt) / wn) * 180 / PI),
        sum_radii=float(d0 + d1), dist_v0v1=float(dist),
        residual_sum_minus_dist=float(d0 + d1 - dist),
        dx_tangential=float(abs(wt)), two_sqrt_d0d1=float(dx),
        dx_residual=float(abs(wt) - dx),
        tangency_lattice_frame=[float(Tlx), float(Tly)],
        tangency_surface_frame=[float(T_s[0]), float(T_s[1])],
        v0_surface_frame=[float(v0_s[0]), float(v0_s[1])],
        v1_surface_frame=[float(v1_s[0]), float(v1_s[1])],
        foot_v0_surface_frame=[float(P0_s[0]), float(P0_s[1])],
        foot_v1_surface_frame=[float(P1_s[0]), float(P1_s[1])],
        foot_v0_lattice_frame=[float(D(wx) - wn * COS_A), float(D(wy) - wn * SIN_A)],
        foot_v1_lattice_frame=[float(-EPS * COS_A), float(-EPS * SIN_A)],
    )
    print(f"\n  witness w = {w}   |w| = {wl}")
    print(f"    w.n (= d0 - d1)        = {wn}")
    print(f"    w.t (tangential offset)= {wt}")
    print(f"    theta_w                = {float(datan(abs(wt)/wn)*180/PI):.9f} deg")
    print(f"    d1 = eps*              = {d1}")
    print(f"    d0 = eps* + w.n        = {d0}")
    print(f"      (|w|+w.n)/2          = {dthr}   residual {float(d0-dthr):+.2e}")
    print(f"    radii sum d0+d1        = {d0 + d1}")
    print(f"    dist(v0,v1) = |w|      = {dist}")
    print(f"    d0+d1-dist             = {float(d0+d1-dist):+.3e}  "
          f"(double: {float(d0)+float(d1)-float(dist):+.3e})")
    print(f"    dx = |w.t|             = {abs(wt)}")
    print(f"    2 sqrt(d0 d1)          = {dx}")
    print(f"    dx - 2sqrt(d0 d1)      = {float(abs(wt)-dx):+.3e}")
    print(f"    tangency pt (lattice)  = ({Tlx}, {Tly})")
    print(f"    tangency pt (surface)  = ({T_s[0]}, {T_s[1]})")
    print(f"    v0 (surface frame)     = ({v0_s[0]}, {v0_s[1]})")
    print(f"    foot of v0 on Sigma    = ({P0_s[0]}, 0)")

print(f"\n  target v1 = origin, depth d1 = eps* = {EPS}")
print(f"  Sigma (lattice frame): {{p : p.n = -eps*}}, i.e. "
      f"{float(COS_A):.12f} x + {float(SIN_A):.12f} y + {float(EPS):.12f} = 0")
print(f"  the two witness balls sit on OPPOSITE tangential sides of v1: "
      f"w.t = {float(-SIN_A):+.9f} and {float(-2*SIN_A+COS_A):+.9f}")
print(f"  total tangential span of the sandwich = "
      f"{float(abs(-SIN_A) + abs(-2*SIN_A+COS_A)):.12f}")

# ============================================================================
print()
print("=" * 96)
print("(d)  MARGINS: third place, (3,1), and brittleness in W")
print("=" * 96)
adm = [r for r in rows if r["status"] == "in-band"]
adm.sort(key=lambda r: r["g"])
print("  admissible ranking at alpha*, W=3:")
for i, r in enumerate(adm[:8]):
    print(f"    #{i+1:<2} w={str(tuple(r['w'])):>8}  g={r['g']:.15f}  "
          f"ratio to eps* = {r['g']/eps_f:.9f}  depth={r['depth']:.6f}")
third = adm[2]
print(f"\n  third place: w={tuple(third['w'])}, g={third['g']:.15f}")
print(f"    absolute margin  g3 - eps* = {third['g']-eps_f:.15f}")
print(f"    relative margin            = {(third['g']-eps_f)/eps_f*100:.6f} %  "
      f"(exactly a factor {third['g']/eps_f:.12f})")

g31 = (math.sqrt(10) - 3 * ca - sa) / 2
d31 = (math.sqrt(10) + 3 * ca + sa) / 2
d31_exact = (S10 + 3 * COS_A + SIN_A) / 2
print(f"\n  best inadmissible: w=(3,1)")
print(f"    g(3,1)         = {g31:.15f}  = eps* / {eps_f/g31:.9f}")
print(f"    depth@thr      = {d31_exact}  (double {d31:.12f})")
print(f"    exceeds W=3 by = {float(d31_exact-3):.12f}  ({float((d31_exact-3)/3*100):.6f} % of W)")
print(f"    would need W  >= {float(d31_exact):.12f} = D31(alpha*)")
print(f"    sqrt10         = {S10}  (max over alpha of D31, at alpha=atan(1/3)="
      f"{float(datan(D(1)/3)*180/PI):.6f} deg)")

# plateau endpoints, verified numerically
def eps_star_of_alpha(a, Wv, R=9):
    c, s = math.cos(a), math.sin(a); best = float("inf"); arg = None
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            if (x, y) == (0, 0): continue
            wl = math.hypot(x, y); wn = x * c + y * s
            g, d = (wl - wn) / 2, (wl + wn) / 2
            if BARf < d <= Wv and g < best: best, arg = g, (x, y)
    return best, arg

def eps_star_of_W(Wv, N=200000, R=9, refine=True):
    """max over alpha in [0,pi/2] of eps*(alpha,W); sup-sense (checks both sides
    of admissibility discontinuities via a fine grid + local refinement)."""
    bestv, besta = -1.0, None
    for i in range(N + 1):
        a = (math.pi / 2) * i / N
        v, _ = eps_star_of_alpha(a, Wv, R)
        if v > bestv: bestv, besta = v, a
    if refine:
        h = (math.pi / 2) / N
        for _ in range(80):
            for a in (besta - h, besta + h, besta - h / 2, besta + h / 2):
                if 0 <= a <= math.pi / 2:
                    v, _ = eps_star_of_alpha(a, Wv, R)
                    if v > bestv: bestv, besta = v, a
            h *= 0.6
    return bestv, besta

print("\n  eps*(W) around the plateau endpoints (independent scan, 2e5 angles + refine):")
Wlo_exact = float(S5)                 # conjecture: plateau starts at |(2,1)| = sqrt5
Whi_exact = float(d31_exact)          # conjecture: plateau ends when (3,1) enters band at a*
probe = [2.20, 2.23, 2.2360, Wlo_exact - 1e-6, Wlo_exact, Wlo_exact + 1e-6, 2.25, 2.5,
         3.0, 3.10, 3.16, Whi_exact - 1e-6, Whi_exact, Whi_exact + 1e-6, 3.161, 3.1622,
         float(S10) - 1e-6, float(S10), 3.17, 3.20]
plateau_scan = []
for Wv in probe:
    v, a = eps_star_of_W(Wv, N=60000)
    plateau_scan.append(dict(W=Wv, eps=v, alpha_deg=math.degrees(a)))
    print(f"    W={Wv:<14.9f} eps*={v:.12f}  alpha={math.degrees(a):9.5f} deg   "
          f"{'== plateau' if abs(v-eps_f)<1e-9 else ''}")

print(f"\n  plateau (conjectured exact):  W in [sqrt5, D31(alpha*)) = "
      f"[{Wlo_exact:.12f}, {Whi_exact:.12f})")
print(f"  W=3 sits {3-Wlo_exact:.6f} above the lower end and {Whi_exact-3:.6f} below the upper end")
print(f"  relative slack to the upper end: {(Whi_exact-3)/3*100:.4f} % of W")

# ============================================================================
print()
print("=" * 96)
print("(e)  INDEPENDENT GLOBAL CERTIFICATION OF alpha* AT W=3")
print("=" * 96)
# critical angles: pairwise cost crossings + admissibility boundaries.
# each single piece g_w(alpha) is unimodal with a MINIMUM at alpha=atan2(wy,wx),
# so a max of the lower envelope can only occur at a crossing or a discontinuity.
cands = [(x, y) for x in range(-6, 7) for y in range(-6, 7)
         if (x, y) != (0, 0) and math.hypot(x, y) <= 3.0 + 0.06]
crit = [0.0, math.pi / 2]
def solve_Acos_Bsin(A, B, C):
    """all alpha in [0,pi/2] with A cos a + B sin a = C"""
    R = math.hypot(A, B)
    if R == 0 or abs(C) > R: return []
    ph = math.atan2(B, A)
    d = math.acos(max(-1.0, min(1.0, C / R)))
    out = []
    for a in (ph + d, ph - d):
        for k in (-2, -1, 0, 1, 2):
            aa = a + k * 2 * math.pi
            if -1e-12 <= aa <= math.pi / 2 + 1e-12: out.append(min(max(aa, 0.0), math.pi / 2))
    return out
for i, (x1, y1) in enumerate(cands):
    l1 = math.hypot(x1, y1)
    crit += solve_Acos_Bsin(x1, y1, 2 * Wf - l1)      # depth == W
    crit += solve_Acos_Bsin(x1, y1, 2 * BARf - l1)    # depth == barrier
    for (x2, y2) in cands[i + 1:]:
        l2 = math.hypot(x2, y2)
        crit += solve_Acos_Bsin(x1 - x2, y1 - y2, l1 - l2)   # g1 == g2
crit = sorted(set(round(a, 14) for a in crit))
bestv, besta, bestw = -1.0, None, None
for a in crit:
    for da in (-1e-11, 0.0, 1e-11):
        aa = min(max(a + da, 0.0), math.pi / 2)
        v, wgt = eps_star_of_alpha(aa, Wf)
        if v > bestv: bestv, besta, bestw = v, aa, wgt
print(f"  critical-angle enumeration: {len(crit)} candidate angles")
print(f"  max eps*(alpha) = {bestv:.15f} at alpha = {math.degrees(besta):.9f} deg (w={bestw})")
print(f"  vs closed form  = {eps_f:.15f} at alpha = {float(ALPHA_DEG):.9f} deg")
print(f"  delta eps = {bestv-eps_f:+.3e}   delta alpha = {math.degrees(besta)-float(ALPHA_DEG):+.3e} deg")
gs, ga = eps_star_of_W(3.0, N=400000)
print(f"  brute grid (4e5 angles + refine): eps*={gs:.15f} at {math.degrees(ga):.9f} deg  "
      f"delta={gs-eps_f:+.3e}")
# mirror
print(f"  mirror direction 90 - alpha* = {90-float(ALPHA_DEG):.9f} deg: "
      f"eps*={eps_star_of_alpha(math.pi/2-alpha_f, 3.0)[0]:.15f} "
      f"(witness {eps_star_of_alpha(math.pi/2-alpha_f,3.0)[1]})")

# ============================================================================
out = dict(
    problem=dict(dim=2, W=3.0, barrier_radius=BARf,
                 rule="v1 certified by w  <=>  eps > g(w,n) = (|w| - w.n)/2",
                 admissible="barrier < (|w|+w.n)/2 <= W"),
    critical=dict(
        alpha_rad=alpha_f, alpha_deg=float(ALPHA_DEG),
        alpha_deg_str=str(ALPHA_DEG)[:40],
        cos_alpha=ca, sin_alpha=sa,
        n=[ca, sa], t=[-sa, ca],
        eps_star=eps_f, eps_star_str=str(EPS)[:40],
        closed_form="eps* = (3 - sqrt5 - sqrt(2 sqrt5 - 4))/4",
        closed_form_alpha="cos a* = (sqrt5-1+sqrt(2 sqrt5-4))/2, sin a* = (sqrt5-1-sqrt(2 sqrt5-4))/2",
        tie_identity="1 + cos a* + sin a* = sqrt5",
        tie_identity_residual=float(1 + COS_A + SIN_A - S5),
    ),
    table=rows,
    tie=dict(
        witnesses=[[1, 0], [2, 1]],
        g_double=[g10_f, g21_f], g_hex=[g10_f.hex(), g21_f.hex()],
        bitwise_identical=bool(g10_f == g21_f),
        abs_diff=abs(g10_f - g21_f), ulp=math.ulp(g10_f),
        diff_in_ulps=abs(g10_f - g21_f) / math.ulp(g10_f),
        exact_diff_60digits=str(g10_d - g21_d),
        dg_dalpha_per_rad={"(1,0)": float(dg10), "(2,1)": float(dg21)},
        dg_dalpha_per_deg={"(1,0)": float(dg10 * PI / 180), "(2,1)": float(dg21 * PI / 180)},
        left_derivative_per_rad=float(dg10), right_derivative_per_rad=float(dg21),
        kink_per_rad=float(dg21 - dg10),
    ),
    geometry=geo,
    margins=dict(
        ranking=[dict(w=r["w"], g=r["g"], depth=r["depth"]) for r in adm[:8]],
        third_place=dict(w=third["w"], g=third["g"],
                         abs_margin=third["g"] - eps_f,
                         rel_margin_pct=(third["g"] - eps_f) / eps_f * 100,
                         ratio=third["g"] / eps_f),
        best_inadmissible=dict(w=[3, 1], g=g31, depth=float(d31_exact),
                               ratio_eps_over_g=eps_f / g31,
                               depth_excess_over_W=float(d31_exact) - 3.0,
                               depth_excess_pct_of_W=(float(d31_exact) - 3.0) / 3 * 100),
        plateau=dict(W_lo=Wlo_exact, W_lo_form="sqrt5 = |(2,1)|",
                     W_hi=Whi_exact, W_hi_form="D31(alpha*) = (sqrt10 + 3cos a* + sin a*)/2",
                     W_hi_str=str(d31_exact)[:30],
                     slack_below=3.0 - Wlo_exact, slack_above=Whi_exact - 3.0,
                     slack_above_pct=(Whi_exact - 3.0) / 3 * 100),
        plateau_scan=plateau_scan,
    ),
    certification=dict(critical_angle_max=bestv, critical_angle_argmax_deg=math.degrees(besta),
                       grid_max=gs, grid_argmax_deg=math.degrees(ga),
                       n_critical_angles=len(crit)),
)
with open("/tmp/dist/anatomy.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n  wrote /tmp/dist/anatomy.json")
