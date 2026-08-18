#!/usr/bin/env python3
"""Figures for the 3D companion note. Pure Python -> SVG."""
import math, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lattice_gap"))   # SVG helpers live with the 2D figs
sys.path.insert(0, _HERE)
from figs import (Svg, polyline, INK, MUT, FAINT, BLU, BLUF, AMB, AMBF, RED, TEAL,
                  PUR, GRN, GRID)
import figs as F

OUT = F.OUT
RD = math.sqrt(3) / 2
W3 = 3.0
E_THR = 0.036662265118829642
E_EXA = 0.038443423574719627
N_THR = (0.9266754697623407, 0.3093925077374490, 0.2134217652833884)
N_EXA = (0.7174389352143008, 0.6198877800093550, 0.3178372451957822)
N_2ND = (0.821854415127, 0.528210630753, 0.213421765283)
N_3RD = (0.717438935214, 0.550510257217, 0.426872148234)


def cands(R=4.2):
    out = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                if (a, b, c) == (0, 0, 0):
                    continue
                L = math.sqrt(a * a + b * b + c * c)
                if L <= R:
                    out.append((a, b, c, L))
    return out


C = cands()


def eps_exact(n, W=W3):
    """sup of the uncertified set in (0, r_d]; returns (value, witness setting it)"""
    iv = []
    for a, b, c, L in C:
        wn = a * n[0] + b * n[1] + c * n[2]
        lo = max((L - wn) / 2, RD - wn)
        hi = min(RD, W - wn)
        if hi > lo:
            iv.append((lo, hi, (a, b, c)))
    if not iv:
        return RD, None
    iv.sort()
    e, who = RD, None
    changed = True
    while changed:
        changed = False
        for lo, hi, w in iv:
            if lo < e <= hi:
                e, who, changed = lo, w, True
    return max(e, 0.0), who


def uv(n):
    return (n[1] / n[0], n[2] / n[0])


def n_of(u, v):
    L = math.sqrt(1 + u * u + v * v)
    return (1 / L, u / L, v / L)


# ============================================================ FIG 1
def fig1():
    s = Svg(980, 668, "The exact-rule envelope on the fundamental spherical triangle")
    S, OX, OY = 470, 96, 546
    N = 210

    def P(u, v):
        return (OX + S * u, OY - S * v)

    # sample
    val = {}
    who = {}
    for i in range(N + 1):
        u = i / N
        for j in range(i + 1):
            v = j / N
            e, w = eps_exact(n_of(u, v))
            val[(i, j)] = e
            who[(i, j)] = w

    # region boundaries: argmin changes
    walls = []
    for i in range(N + 1):
        for j in range(i + 1):
            for di, dj in ((1, 0), (0, 1)):
                k = (i + di, j + dj)
                if k in who and who[k] != who[(i, j)]:
                    a = P(i / N, j / N)
                    b = P(k[0] / N, k[1] / N)
                    walls.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))

    if walls:
        s.path(" ".join(f"M {x-1.6:.1f} {y:.1f} L {x+1.6:.1f} {y:.1f}"
                        for x, y in walls), s=FAINT, w=2.4, f="none")

    # contours by marching squares on the value field
    LEVELS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.034, 0.0365, 0.038]
    for lev in LEVELS:
        segs = []
        for i in range(N):
            for j in range(min(i, N - 1)):
                pts = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                if not all(p in val for p in pts):
                    continue
                cs = []
                for k in range(4):
                    a, b = pts[k], pts[(k + 1) % 4]
                    va, vb = val[a], val[b]
                    if (va - lev) * (vb - lev) < 0:
                        t = (lev - va) / (vb - va)
                        cs.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                if len(cs) == 2:
                    segs.append((P(cs[0][0] / N, cs[0][1] / N),
                                 P(cs[1][0] / N, cs[1][1] / N)))
        col = RED if lev >= 0.034 else (AMB if lev >= 0.02 else BLU)
        wdt = 2.2 if lev >= 0.034 else 1.3
        if segs:
            d = " ".join(f"M {a[0]:.1f} {a[1]:.1f} L {b[0]:.1f} {b[1]:.1f}"
                         for a, b in segs)
            s.path(d, s=col, w=wdt, f="none")

    # triangle border
    s.path(polyline([P(0, 0), P(1, 0), P(1, 1)]) + " Z", s=INK, w=2.6, f="none")
    for (u, v), lab, dx, dy in ((( 0, 0), "(1,0,0)", -8, 22), ((1, 0), "(1,1,0)", 14, 22),
                                ((1, 1), "(1,1,1)", 14, -8)):
        p = P(u, v)
        s.circ(*p, 6, f=INK, fo=1)
        s.txt(p[0] + dx, p[1] + dy, lab, size=14, anchor="middle" if dx < 0 else "start")

    # maxima
    for n, e, col, lab, off in ((N_EXA, E_EXA, RED, "global max  ε* = 0.0384434", (16, -14)),
                                (N_2ND, 0.032074258246890, AMB, "0.0320743", (16, 16)),
                                (N_3RD, 0.018614733452036, BLU, "0.0186147", (16, 16))):
        u, v = uv(n)
        p = P(u, v)
        s.circ(*p, 13, s=col, w=2.4)
        s.circ(*p, 5.5, f=col, fo=1)
        s.txt(p[0] + off[0], p[1] + off[1], lab, size=14, fill=col, weight="600")
    u, v = uv(N_THR)
    p = P(u, v)
    s.circ(*p, 9, s=PUR, w=2.2, dash="4 3")
    s.txt(p[0] - 14, p[1] + 6, "threshold-rule max", size=13, fill=PUR, anchor="end")

    s.txt(24, 34, "1.  The lower envelope on the fundamental spherical triangle "
                  "(exact rule, W = 3)", size=18, weight="600", halo=False)
    s.txt(P(0.5, 0)[0], OY + 56, "gnomonic coordinates  u = n_y/n_x ,  v = n_z/n_x", size=14,
          fill=MUT, anchor="middle", halo=False)
    s.rect(650, 74, 312, 152, f="#fff", s=GRID, rx=8)
    s.txt(668, 102, "One triangle is the whole sphere", size=15, weight="600", halo=False)
    s.txt(668, 128, "The 48-element group B₃ maps every", size=13, fill=MUT, halo=False)
    s.txt(668, 148, "direction into 0 ≤ n_z ≤ n_y ≤ n_x.", size=13, fill=MUT, halo=False)
    s.txt(668, 174, "Grey cell walls = the winning witness", size=13, fill=MUT, halo=False)
    s.txt(668, 194, "changes. Coloured curves = level sets", size=13, fill=MUT, halo=False)
    s.txt(668, 214, "of ε*(n). Peaks sit at cell corners.", size=13, fill=MUT, halo=False)
    s.txt(24, 626, "Exactly three local maxima above 0.002, each a TRIPLE tie — three witnesses "
                   "of equal cost, the 3D analogue of 2D's two-witness", size=14, fill=MUT,
          halo=False)
    s.txt(24, 648, "crossover. The global max is (1,1,0)/(1,1,1)/(2,1,1); the threshold rule's "
                   "max sits elsewhere because (2,2,1) is admissible there.", size=14,
          fill=MUT, halo=False)
    s.save("fig3d_1_envelope.svg")


# ============================================================ FIG 2
def fig2():
    s = Svg(980, 604, "Why the two admissibility rules disagree in 3D")
    L, R, T = 168, 828, 148
    hi = 0.05

    def X(e):
        return L + (R - L) * e / hi

    n = N_EXA
    rows = []
    for a, b, c, Lw in C:
        wn = a * n[0] + b * n[1] + c * n[2]
        lo = max((Lw - wn) / 2, RD - wn)
        h = min(RD, W3 - wn)
        if h > lo and lo < hi:
            rows.append((lo, h, (a, b, c), Lw, (Lw - wn) / 2))
    rows.sort()
    s.txt((L + R) / 2, 76, "target depth  ε   (voxels)", size=15, anchor="middle",
          halo=False)
    s.line(L, T - 32, R, T - 32, c=INK, w=2)
    for e in (0, 0.01, 0.02, 0.03, 0.04, 0.05):
        s.line(X(e), T - 38, X(e), T - 26, c=INK, w=1.6)
        s.txt(X(e), T - 46, f"{e:.2f}", size=13, anchor="middle", halo=False)

    y = T
    for lo, h, w, Lw, g in rows[:5]:
        col = RED if w == (2, 2, 1) else BLU
        x0, x1 = X(lo), min(X(h), R)
        s.rect(x0, y, max(x1 - x0, 2.5), 26, f=col, fo=0.22, rx=4)
        s.line(x0, y, x0, y + 26, c=col, w=2.6)
        s.line(x1, y, x1, y + 26, c=col, w=2.6)
        s.txt(L - 14, y + 19, f"{w}", size=15, anchor="end", fill=col,
              weight="600" if w == (2, 2, 1) else "normal", halo=False)
        cap = "band" if abs(h - (W3 - sum(a * b for a, b in zip(w, n)))) < 1e-12 else "r_d"
        s.txt(R + 12, y + 19, f"|w| = {Lw:.4f}   top: {cap}", size=12, fill=MUT,
              halo=False)
        y += 38

    yb = y + 22
    s.rect(X(0), yb, X(0.003754662178453) - X(0), 30, f=INK, fo=0.16, rx=4)
    s.rect(X(0.007509324356906), yb, X(E_EXA) - X(0.007509324356906), 30, f=INK, fo=0.16,
           rx=4)
    s.txt(L - 14, yb + 21, "UNCERTIFIED", size=15, anchor="end", weight="600", halo=False)
    s.line(X(E_EXA), yb - 8, X(E_EXA), yb + 40, c=RED, w=2.6)
    s.txt(X(E_EXA) + 12, yb + 56, "ε* = 0.038443424", size=15, fill=RED, weight="600",
          halo=False)
    s.txt(X(0.0056), yb + 58, "the hole", size=13, fill=RED, anchor="middle", halo=False)
    s.path(f"M {X(0.0056):.1f} {yb+46:.1f} L {X(0.0056):.1f} {yb+32:.1f}", s=RED, w=1.8)

    s.rect(24, 384, 416, 122, f="#fff", s=GRID, rx=8)
    s.txt(42, 412, "(2,2,1) has |w| = 3 = W exactly", size=15, weight="600", halo=False)
    s.txt(42, 438, "so its window top is W − w·n = |w| − w·n = 2g:", size=13, fill=MUT,
          halo=False)
    s.txt(42, 462, "a sliver (g, 2g] = (0.003755, 0.007509].", size=13, fill=MUT,
          halo=False)
    s.txt(42, 490, "The threshold rule would call it usable for ALL ε > g.", size=13,
          fill=RED, halo=False)

    s.txt(24, 34, "2.  The (2,2,1) mechanism: a witness whose usable window is a sliver",
          size=18, weight="600", halo=False)
    s.txt(24, 552, "Each bar is J_w, the set of target depths that witness w can certify. "
                   "Under the EXACT rule the bar is bounded above, because a deeper", size=14,
          fill=MUT, halo=False)
    s.txt(24, 574, "target pushes the witness out of the top of the band. The union is not "
                   "upward closed, so the uncertified set is TWO intervals.", size=14,
          fill=MUT, halo=False)
    s.save("fig3d_2_windows.svg")


# ============================================================ FIG 3
def fig3():
    s = Svg(980, 590, "eps*(W) in 3D, both admissibility rules")
    L, R, T, B = 96, 930, 84, 400
    w0, w1, y0, y1 = 1.0, 6.5, 0.0, 0.11

    def X(w):
        return L + (R - L) * (w - w0) / (w1 - w0)

    def Y(v):
        return B - (B - T) * (v - y0) / (y1 - y0)

    THR = [(1.1547005384, 1.2544355486, None), (1.2544355486, 1.2844570504, 0.577350269190),
           (1.2844570504, 1.4142135624, 0.353553390593),
           (1.4142135624, 1.7320508076, 0.129756511997),
           (1.7320508076, 2.1780221276, 0.073559321150),
           (2.1804880080, 2.4461175157, 0.072326379719),
           (2.4494897428, 2.9962453378, 0.038443423575),
           (2.9997989956, 3.1258482886, 0.036662265119),
           (3.1259333493, 3.3098177994, 0.036619733596),
           (3.3141721917, 3.7385318295, 0.032074258247),
           (3.7416569969, 4.2351948121, 0.022218963297),
           (4.2355152670, 4.5817872842, 0.022058734637),
           (4.5824013795, 5.4747882239, 0.019127139841),
           (5.4763161023, 5.9147627519, 0.015128624096),
           (5.9151542141, 6.4006505158, 0.014298388987)]
    EXA = [(1.8660254038, 2.1935355989, 0.073559321150),
           (2.1972344195, 2.5150716684, 0.072326379719),
           (2.5154503233, 3.0309340992, 0.038443423575),
           (3.0362639127, 3.1260811821, 0.036662265119),
           (3.1262087733, 3.3396305420, 0.036619733596),
           (3.3446745744, 3.7674805304, 0.032074258247),
           (3.7702462712, 4.2499679005, 0.022218963297),
           (4.2504485827, 4.6030576080, 0.022058734637),
           (4.6030576080, 5.4914780127, 0.019127139841),
           (5.4914865000, 5.9285743448, 0.015128624096),
           (5.9285743000, 6.4124751832, 0.014298388987)]

    s.rect(L, T, R - L, B - T, f="#fcfcfd", s=GRID)
    for v in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10):
        s.line(L, Y(v), R, Y(v), c=GRID, w=1.2)
        s.txt(L - 10, Y(v) + 5, f"{v:.2f}", size=13, fill=MUT, anchor="end", halo=False)
    for w in (1, 2, 3, 4, 5, 6):
        s.line(X(w), T, X(w), B, c=GRID, w=1.2)
        s.txt(X(w), B + 22, str(w), size=13, fill=MUT, anchor="middle", halo=False)
    s.txt((L + R) / 2, B + 48, "band half-width  W  (voxels)", size=15, anchor="middle",
          halo=False)
    s.txt(L - 58, (T + B) / 2, "ε*(W)", size=15, anchor="middle", halo=False, rot=-90)

    s.rect(L, Y(RD), X(1.1547005384) - L, B - Y(RD), f=INK, fo=0.07)
    s.rect(L, T, X(1.8660254038) - L, B - T, f=RED, fo=0.05)
    s.txt(X(1.42), T + 22, "no guarantee", size=13, fill=RED, anchor="middle", halo=False)
    s.txt(X(1.42), T + 40, "(exact rule)", size=12, fill=RED, anchor="middle", halo=False)

    for tab, col, wid, lab in ((THR, PUR, 5.0, "threshold rule"), (EXA, INK, 2.6, "exact rule")):
        prev = None
        for a, b, v in tab:
            if v is None:
                continue
            s.line(X(a), Y(v), X(b), Y(v), c=col, w=wid)
            if prev is not None:
                s.line(X(prev[0]), Y(prev[1]), X(a), Y(v), c=col, w=wid * 0.55, op=0.5,
                       dash="3 3")
            prev = (b, v)

    s.line(X(3), T, X(3), B, c=RED, w=2.2)
    s.txt(X(3) + 10, T + 22, "W = 3", size=16, fill=RED, anchor="middle", weight="600",
          halo=False)
    s.circ(X(3), Y(E_EXA), 6, f=INK, fo=1)
    s.txt(X(3) + 12, Y(E_EXA) - 12, "0.0384434 (exact)", size=13, weight="600")
    s.circ(X(3), Y(E_THR), 5, f=PUR, fo=1)
    s.txt(X(3) + 12, Y(E_THR) + 22, "0.0366623 (threshold)", size=13, fill=PUR)
    s.line(L + 620, T + 16, L + 660, T + 16, c=INK, w=3.4)
    s.txt(L + 668, T + 21, "exact rule (the pipeline)", size=13, halo=False)
    s.line(L + 620, T + 40, L + 660, T + 40, c=PUR, w=2.2)
    s.txt(L + 668, T + 45, "threshold rule", size=13, fill=PUR, halo=False)

    s.txt(24, 34, "3.  ε*(W) in 3D: a staircase with ramps, and the two rules on different "
                  "plateaus at W = 3", size=18, weight="600", halo=False)
    s.txt(24, 486, "Dashed segments are transitions — 3D is far rampier than 2D, which was "
                   "mostly clean jumps. Under the EXACT rule the scheme gives no", size=14,
          fill=MUT, halo=False)
    s.txt(24, 508, "guarantee at all below W = 1 + r_d = 1.8660 (red band), yet the "
                   "threshold rule reports finite constants there — optimistic by an", size=14,
          fill=MUT, halo=False)
    s.txt(24, 530, "unbounded factor. At W = 3 the threshold plateau floor is only 2.01×10⁻⁴ "
                   "below: a knife edge. The exact rule has 16% margin below,", size=14,
          fill=MUT, halo=False)
    s.txt(24, 552, "but only 1.03% above.", size=14, fill=MUT, halo=False)
    s.save("fig3d_3_plateau.svg")


# ============================================================ FIG 4
def fig4():
    s = Svg(980, 580, "Curvature in 3D")
    L, R, T, B = 96, 916, 84, 400
    y0, y1, r0, r1 = 0.0, 0.115, 1.8, 100.0
    K = 0.063494643741112

    def X(Rr):
        return L + (R - L) * (math.log(Rr) - math.log(r0)) / (math.log(r1) - math.log(r0))

    def Y(v):
        return B - (B - T) * (v - y0) / (y1 - y0)

    def eps_R(Rr, concave):
        best = 0.0
        base = []
        for i in range(0, 61):
            for j in range(0, i + 1):
                for k in range(0, j + 1):
                    if i == 0:
                        continue
                    Lv = math.sqrt(i * i + j * j + k * k)
                    base.append((i / Lv, j / Lv, k / Lv))
        base = list(set(base)) + [N_EXA, N_THR, N_2ND, N_3RD]
        for n in base:
            m = None
            for a, b, c, Lw in C:
                wn = a * n[0] + b * n[1] + c * n[2]
                if wn <= 0:
                    continue
                g = (Lw - wn) / 2
                d0 = (Lw + wn) / 2
                if concave:
                    if d0 >= Rr:
                        continue
                    e = g / (1 - d0 / Rr)
                else:
                    e = g / (1 + d0 / Rr)
                dd = Lw - e
                if dd <= RD or dd > W3:
                    continue
                if m is None or e < m:
                    m = e
            if m is not None and m > best:
                best = m
        return best

    s.rect(L, T, R - L, B - T, f="#fcfcfd", s=GRID)
    for v in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10):
        s.line(L, Y(v), R, Y(v), c=GRID, w=1.2)
        s.txt(L - 10, Y(v) + 5, f"{v:.2f}", size=13, fill=MUT, anchor="end", halo=False)
    for Rr in (2, 3, 5, 8, 12, 20, 35, 60, 100):
        s.line(X(Rr), T, X(Rr), B, c=GRID, w=1.2)
        s.txt(X(Rr), B + 22, str(Rr), size=13, fill=MUT, anchor="middle", halo=False)
    s.txt((L + R) / 2, B + 48, "curvature radius  R  (voxels, log scale)", size=15,
          anchor="middle", halo=False)
    s.txt(L - 58, (T + B) / 2, "ε*", size=15, anchor="middle", halo=False, rot=-90)
    s.line(L, Y(E_EXA), R, Y(E_EXA), c=INK, w=1.8, dash="7 5")
    s.txt(R - 8, Y(E_EXA) - 10, "ε* = 0.0384434  (planar)", size=14, anchor="end")

    NP = 46
    for concave, col in ((True, RED), (False, TEAL)):
        ex, fo = [], []
        for i in range(NP + 1):
            Rr = math.exp(math.log(r0) + (math.log(r1) - math.log(r0)) * i / NP)
            v = eps_R(Rr, concave)
            if y0 <= v <= y1:
                ex.append((X(Rr), Y(v)))
            fv = E_EXA + (K / Rr if concave else -K / Rr)
            if y0 <= fv <= y1:
                fo.append((X(Rr), Y(fv)))
        s.path(polyline(ex), s=col, w=3.2)
        s.path(polyline(fo), s=col, w=1.8, dash="5 5", cap="butt")

    s.txt(X(2.05), Y(0.086), "concave, exact", size=15, fill=RED, weight="600")
    s.txt(X(4.6), Y(0.055), "first order  ε* + K/R", size=13, fill=RED)
    s.txt(X(2.2), Y(0.026), "convex, exact", size=15, fill=TEAL, weight="600")
    s.txt(X(2.9), Y(0.010), "first order  ε* − K/R", size=13, fill=TEAL)
    for Rr, lab in ((5, "+50.7%"), (3, "+158%")):
        v = eps_R(Rr, True)
        s.circ(X(Rr), Y(v), 5.5, f=RED, fo=1)
        s.txt(X(Rr) + 12, Y(v) + 20, lab, size=13, fill=RED, weight="600")

    s.txt(24, 34, "4.  Curvature in 3D: K = 0.0635, about 2.2× the 2D value", size=18,
          weight="600", halo=False)
    s.txt(24, 486, "Exact law ε_thr(w) = g/(1 ∓ d₀/R), identical in form to 2D. A phase-1 "
                   "table published the first-order model as if it were exact and", size=14,
          fill=MUT, halo=False)
    s.txt(24, 508, "understated the concave penalty by up to 3.7×; both referees caught it. "
                   "True excesses: +50.7% at R = 5, +158% at R = 3.", size=14, fill=MUT,
          halo=False)
    s.txt(24, 530, "The convex branch does NOT saturate — it goes to zero linearly, "
                   "ε*(R) → R·tan²(δ/2) with δ = 17.6532° the angular covering radius,", size=14,
          fill=MUT, halo=False)
    s.txt(24, 552, "because as R → 0 the length weighting vanishes and only the angle "
                   "survives.", size=14, fill=MUT, halo=False)
    s.save("fig3d_4_curvature.svg")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
