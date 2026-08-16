#!/usr/bin/env python3
"""Generate the figure set for the peer-review note. Pure Python -> SVG."""
import math, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

INK, MUT, FAINT = "#0f172a", "#475569", "#94a3b8"
BLU, BLUF = "#2563eb", "#3b82f6"
AMB, AMBF = "#b45309", "#f59e0b"
RED, TEAL, PUR, GRN = "#dc2626", "#0f766e", "#7c3aed", "#15803d"
GRID = "#e2e8f0"
FONT = ("ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, "
        "sans-serif")

PHI = math.atan2(1, 2)                       # 26.5651 deg
A_STAR = math.radians(15.930625116297946539)
A_STAR2 = math.radians(34.729139026042240)
EPS_C = 0.019202630763796344
EPS_2 = 0.011330789030059158
RD = math.sqrt(2) / 2


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.title = w, h, title
        self.b = []

    def add(self, s):
        self.b.append(s)

    def txt(self, x, y, s, size=15, fill=INK, anchor="start", weight="normal",
            halo=True, style="", rot=None):
        h = ('style="paint-order:stroke;stroke:#fff;stroke-width:4.5px;'
             'stroke-linejoin:round"') if halo else ""
        t = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}" {h} {style}{t}>'
                 f'{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, c=INK, w=1.5, dash=None, op=1.0, cap="butt"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{c}" stroke-width="{w}" stroke-opacity="{op}" '
                 f'stroke-linecap="{cap}"{d}/>')

    def circ(self, cx, cy, r, s=None, f="none", w=2.0, fo=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        st = f' stroke="{s}" stroke-width="{w}"' if s else ""
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{f}" '
                 f'fill-opacity="{fo}"{st}{d}/>')

    def path(self, d, s=INK, f="none", w=2.0, fo=1.0, dash=None, cap="round"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{f}" fill-opacity="{fo}" stroke="{s}" '
                 f'stroke-width="{w}" stroke-linejoin="round" '
                 f'stroke-linecap="{cap}"{da}/>')

    def rect(self, x, y, w, h, f="none", s=None, sw=1.5, fo=1.0, rx=0):
        st = f' stroke="{s}" stroke-width="{sw}"' if s else ""
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                 f'rx="{rx}" fill="{f}" fill-opacity="{fo}"{st}/>')

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} '
                f'{self.h}" width="{self.w}" height="{self.h}" font-family="{FONT}">'
                f'<title>{esc(self.title)}</title>'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>')
        open(os.path.join(OUT, name), "w").write(head + "".join(self.b) + "</svg>")
        print("wrote", name)


def polyline(pts, **kw):
    return "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)


# ----------------------------------------------------------------- costs
def g(w, alpha):
    """certification cost of witness w at normal angle alpha"""
    wl = math.hypot(*w)
    b = math.atan2(w[1], w[0])
    return wl * (1 - math.cos(alpha - b)) / 2


def d0_thr(w, alpha):
    wl = math.hypot(*w)
    b = math.atan2(w[1], w[0])
    return wl * (1 + math.cos(alpha - b)) / 2


# ================================================================ FIG 1
def fig1():
    s = Svg(940, 600, "The certification rule and its marginal case")
    S, OX, OY = 150, 175, 470            # px per unit, origin of v1
    nx, ny = math.cos(A_STAR), math.sin(A_STAR)
    eps = 0.42                            # exaggerated for legibility
    W = (1.9, 0.62)                       # a schematic witness offset
    d0 = eps + W[0] * nx + W[1] * ny

    def P(x, y):
        return (OX + S * x, OY - S * y)

    # Sigma : x.n = -eps  ->  draw a long segment
    tx, ty = -ny, nx
    c = (-eps * nx, -eps * ny)
    a = (c[0] - 2.0 * tx, c[1] - 2.0 * ty)
    b = (c[0] + 3.4 * tx, c[1] + 3.4 * ty)
    # solid side hatch
    s.add('<defs><pattern id="h1" width="10" height="10" '
          'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
          '<rect width="10" height="10" fill="#f1f5f9"/>'
          '<line x1="0" y1="0" x2="0" y2="10" stroke="#cbd5e1" stroke-width="1.6"/>'
          '</pattern></defs>')
    deep = 1.2
    quad = [P(*a), P(*b), P(b[0] - deep * nx, b[1] - deep * ny),
            P(a[0] - deep * nx, a[1] - deep * ny)]
    s.path(polyline(quad) + " Z", f="url(#h1)", s="none", w=0)
    s.line(*P(*a), *P(*b), c=INK, w=3.4)
    s.txt(*P(b[0] - 0.16 * tx + 0.12 * nx, b[1] - 0.16 * ty + 0.12 * ny),
          "Σ", size=22, style='font-style="italic"', anchor="middle")

    # balls
    s.circ(*P(0, 0), eps * S, s=AMB, f=AMBF, w=2.4, fo=0.20)
    s.circ(*P(*W), d0 * S, s=BLU, f=BLUF, w=2.4, fo=0.10)
    # segment
    s.line(*P(0, 0), *P(*W), c=INK, w=2, dash="7 5")
    # radii to feet
    f1 = (-eps * nx, -eps * ny)
    f0 = (W[0] - d0 * nx, W[1] - d0 * ny)
    s.line(*P(0, 0), *P(*f1), c=AMB, w=2, dash="5 4")
    s.line(*P(*W), *P(*f0), c=BLU, w=2, dash="5 4")

    # tangency point between the two balls
    t = (W[0] * eps / (eps + d0), W[1] * eps / (eps + d0))
    s.circ(*P(*t), 13, s=RED, w=2)
    s.circ(*P(*t), 6, f=RED, fo=1)

    # theta arc at the witness
    r = 78
    aP = math.atan2(*(lambda p, q: (-(q[1] - p[1]), q[0] - p[0]))(P(*W), P(*f0)))
    aQ = math.atan2(*(lambda p, q: (-(q[1] - p[1]), q[0] - p[0]))(P(*W), P(0, 0)))
    pw = P(*W)
    p1 = (pw[0] + r * math.cos(-aP + math.pi / 2), pw[1] - r * math.sin(-aP + math.pi / 2))
    p2 = (pw[0] + r * math.cos(-aQ + math.pi / 2), pw[1] - r * math.sin(-aQ + math.pi / 2))
    s.path(f"M {p1[0]:.2f} {p1[1]:.2f} A {r} {r} 0 0 1 {p2[0]:.2f} {p2[1]:.2f}",
           s=PUR, w=2.4)
    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    v = (mid[0] - pw[0], mid[1] - pw[1]); L = math.hypot(*v)
    s.txt(pw[0] + v[0] / L * (r + 20), pw[1] + v[1] / L * (r + 20), "θ",
          size=22, fill=PUR, anchor="middle", style='font-style="italic"')

    # dots + labels
    for p, col, lab, dy in ((P(0, 0), AMB, "v₁  target", -18),
                            (P(*W), BLU, "v₀ = v₁ + w  witness", -20)):
        s.circ(*p, 6.5, f=col, fo=1)
    s.txt(P(0, 0)[0] - 12, P(0, 0)[1] - 16, "v₁", size=21, fill=AMB, weight="600",
          anchor="end")
    s.txt(P(0, 0)[0] - 12, P(0, 0)[1] + 4, "target", size=13, fill=MUT, anchor="end")
    s.txt(P(*W)[0] + 14, P(*W)[1] - 12, "v₀ = v₁ + w", size=21, fill=BLU, weight="600")
    s.txt(P(*W)[0] + 14, P(*W)[1] + 8, "witness", size=13, fill=MUT)
    s.txt(*[v + o for v, o in zip(P(-0.06 * nx - 0.22 * tx, -0.06 * ny - 0.22 * ty),
                                  (0, 0))], "d₁ = ε", size=16, fill=AMB, anchor="end")
    s.txt(P(W[0] - 0.55 * nx + 0.10 * tx, W[1] - 0.55 * ny + 0.10 * ty)[0],
          P(W[0] - 0.55 * nx, W[1] - 0.55 * ny)[1], "d₀ = ε + w·n", size=16, fill=BLU)
    mp = P(W[0] / 2, W[1] / 2)
    s.txt(mp[0] - 6, mp[1] - 14, "|w|", size=16, anchor="end")

    # caption block
    s.rect(600, 44, 320, 196, f="#fff", s=GRID, rx=8)
    s.txt(618, 72, "The rule (exact)", size=15, weight="600", halo=False)
    s.txt(618, 98, "d₁ + d₀ > |w|   (strict)", size=15, fill=MUT, halo=False)
    s.txt(618, 126, "Substituting d₀ = ε + w·n:", size=14, fill=MUT, halo=False)
    s.txt(618, 152, "ε > (|w| − w·n)/2", size=17, fill=INK, weight="600", halo=False)
    s.txt(618, 178, "   = |w|·sin²(θ/2)  =:  g(w,n)", size=17, fill=INK,
          weight="600", halo=False)
    s.txt(618, 206, "θ is the angle at the witness", size=14, fill=MUT, halo=False)
    s.txt(618, 226, "between its normal drop and v₁.", size=14, fill=MUT, halo=False)

    s.txt(24, 34, "1.  The pairwise rule, and the tangency that marks its margin",
          size=18, weight="600", halo=False)
    s.txt(24, 566, "Drawn at the marginal case d₁ + d₀ = |w|: the balls touch at one "
                   "point (red) and the rule, being strict, fails.", size=14, fill=MUT,
          halo=False)
    s.save("fig1_rule_and_tangency.svg")


# ================================================================ FIG 2
def fig2():
    s = Svg(940, 560, "One witness reaches a parabola, not a ball")
    S, OX, OY = 200, 470, 430
    d0 = 1.55

    def P(x, y):
        return (OX + S * x, OY - S * y)

    s.add('<defs><pattern id="h2" width="10" height="10" '
          'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
          '<rect width="10" height="10" fill="#f1f5f9"/>'
          '<line x1="0" y1="0" x2="0" y2="10" stroke="#cbd5e1" stroke-width="1.6"/>'
          '</pattern></defs>')
    s.rect(20, OY, 900, 110, f="url(#h2)")

    # capture region: eps > tau^2/(4 d0)
    pts = []
    tau = -2.24
    while tau <= 2.2401:
        pts.append(P(tau, tau * tau / (4 * d0)))
        tau += 0.01
    top = [P(2.24, 1.30), P(-2.24, 1.30)]
    s.path(polyline(pts + top) + " Z", f=GRN, fo=0.10, s="none", w=0)
    s.path(polyline(pts), s=GRN, w=3.0)

    s.circ(*P(0, d0), d0 * S, s=BLU, f=BLUF, w=2.2, fo=0.12)
    s.line(20, OY, 920, OY, c=INK, w=3.4)
    s.txt(898, OY - 14, "Σ", size=22, anchor="end", style='font-style="italic"')

    # directrix
    s.line(20, OY + d0 * S, 920, OY + d0 * S, c=GRN, w=1.8, dash="8 6", op=0.75)
    s.txt(898, OY + d0 * S - 12, "directrix:  Σ pushed to the far side by d₀",
          size=14, fill=GRN, anchor="end")

    s.circ(*P(0, d0), 7, f=BLU, fo=1)
    s.txt(*[a + b for a, b in zip(P(0, d0), (14, -14))], "v₀  (focus)", size=18,
          fill=BLU, weight="600")
    s.line(*P(0, d0), *P(0, 0), c=BLU, w=2, dash="5 4")
    s.txt(*[a + b for a, b in zip(P(0, d0 / 2), (12, 6))], "d₀", size=16, fill=BLU)
    s.circ(*P(0, 0), 5.5, f=INK, fo=1)
    s.txt(*[a + b for a, b in zip(P(0, 0), (0, 30))], "P₀", size=16, anchor="middle")

    # a captured point far off-axis
    tt, ee = 1.75, 1.75 ** 2 / (4 * d0)
    s.line(*P(tt, 0), *P(tt, ee), c=TEAL, w=2, dash="4 4")
    s.circ(*P(tt, ee), 6, f=TEAL, fo=1)
    s.txt(*[a + b for a, b in zip(P(tt, ee), (14, -6))],
          "ε = τ²/(4d₀)", size=15, fill=TEAL)
    s.line(*P(0, -0.14), *P(tt, -0.14), c=TEAL, w=2)
    s.txt(*[a + b for a, b in zip(P(tt / 2, -0.14), (0, 22))], "τ", size=16,
          fill=TEAL, anchor="middle")

    s.rect(24, 60, 330, 176, f="#fff", s=GRID, rx=8)
    s.txt(42, 88, "A witness certifies a parabola", size=15, weight="600", halo=False)
    s.txt(42, 116, "v₁ certified by v₀  ⟺  d₀ + ε > |v₁−v₀|", size=14, fill=MUT,
          halo=False)
    s.txt(42, 140, "            ⟺  ε > τ²/(4d₀)", size=15, weight="600", halo=False)
    s.txt(42, 168, "Focus at v₀ itself; vertex ON Σ at P₀.", size=14, fill=MUT,
          halo=False)
    s.txt(42, 190, "So one witness reaches zero depth —", size=14, fill=MUT, halo=False)
    s.txt(42, 210, "but at exactly one tangential offset.", size=14, fill=MUT,
          halo=False)
    s.txt(42, 230, "The ball (blue) is strictly inside it.", size=14, fill=MUT,
          halo=False)

    s.txt(24, 34, "2.  A witness's reach is a parabola with focus at the witness",
          size=18, weight="600", halo=False)
    s.save("fig2_parabola.svg")


# ================================================================ FIG 3
def fig3():
    s = Svg(940, 560, "The 16 primitive directions and their angular gaps")
    S, OX, OY = 86, 250, 300
    R = 2.55

    def P(x, y):
        return (OX + S * x, OY - S * y)

    for i in range(-2, 3):
        s.line(*P(i, -2.4), *P(i, 2.4), c=GRID, w=1.2)
        s.line(*P(-2.4, i), *P(2.4, i), c=GRID, w=1.2)
    s.circ(OX, OY, R * S, s=FAINT, w=1.4, dash="6 5")

    prim = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
            (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]
    nonp = [(2, 0), (0, 2), (-2, 0), (0, -2), (2, 2), (-2, 2), (-2, -2), (2, -2)]

    for w in prim:
        L = math.hypot(*w)
        s.line(OX, OY, *P(w[0] / L * R, w[1] / L * R), c=BLU, w=1.4, op=0.35)
    for w in nonp:
        s.line(*P(w[0] - 0.16, w[1] - 0.16), *P(w[0] + 0.16, w[1] + 0.16),
               c=RED, w=2.2, op=0.75)
        s.line(*P(w[0] - 0.16, w[1] + 0.16), *P(w[0] + 0.16, w[1] - 0.16),
               c=RED, w=2.2, op=0.75)
        s.circ(*P(*w), 5, f=RED, fo=0.28)
    for w in prim:
        s.circ(*P(*w), 6.5, f=BLU, fo=1)
    s.circ(OX, OY, 7, f=INK, fo=1)
    s.txt(OX - 12, OY + 24, "v₁", size=15, anchor="end")

    for w, lab, dx, dy in (((1, 0), "(1,0)", 12, 22), ((2, 1), "(2,1)", 10, -12),
                           ((1, 1), "(1,1)", 10, -12), ((1, 2), "(1,2)", 10, -10),
                           ((0, 1), "(0,1)", 12, -10)):
        s.txt(*[a + b for a, b in zip(P(*w), (dx, dy))], lab, size=14, fill=BLU)
    s.txt(*[a + b for a, b in zip(P(2, 0), (10, 26))], "(2,0) = 2·(1,0)", size=13,
          fill=RED)
    s.txt(*[a + b for a, b in zip(P(2, 2), (10, -14))], "(2,2) = 2·(1,1)", size=13,
          fill=RED)

    # right panel: the octant gaps
    CX, CY, RR = 690, 330, 168
    s.path(f"M {CX} {CY} L {CX+RR} {CY} A {RR} {RR} 0 0 0 "
           f"{CX+RR*math.cos(math.pi/4):.2f} {CY-RR*math.sin(math.pi/4):.2f} Z",
           f=BLUF, fo=0.07, s="none", w=0)
    for ang, col, lab, wlab in ((0.0, BLU, "0°", "(1,0)"),
                                (PHI, AMB, "26.565°", "(2,1)"),
                                (math.pi / 4, BLU, "45°", "(1,1)")):
        x2, y2 = CX + RR * math.cos(ang), CY - RR * math.sin(ang)
        s.line(CX, CY, x2, y2, c=col, w=2.4)
        s.circ(x2, y2, 6, f=col, fo=1)
        s.txt(x2 + 14, y2 + 4, f"{wlab}   {lab}", size=14, fill=col)
    for a0, a1, lab, rr in ((0.0, PHI, "26.565°", 112), (PHI, math.pi / 4, "18.435°", 66)):
        s.path(f"M {CX+rr*math.cos(a0):.2f} {CY-rr*math.sin(a0):.2f} "
               f"A {rr} {rr} 0 0 0 {CX+rr*math.cos(a1):.2f} {CY-rr*math.sin(a1):.2f}",
               s=TEAL, w=2.2)
        am = (a0 + a1) / 2
        s.txt(CX + (rr + 20) * math.cos(am), CY - (rr + 20) * math.sin(am) + 5,
              lab, size=14, fill=TEAL, anchor="middle")
    s.txt(CX, CY + 62, "one octant", size=14, fill=MUT, anchor="middle")

    s.txt(24, 34, "3.  Keep one point per direction: the 16 primitive vectors of [−2,2]²",
          size=18, weight="600", halo=False)
    s.txt(24, 522, "Crossed out: the 8 non-primitive vectors. A multiple kw costs "
                   "exactly k·g(w,n), so it can never beat its own primitive.",
          size=14, fill=MUT, halo=False)
    s.txt(24, 544, "Gaps alternate 26.565° / 18.435° and sum to 45° per octant, so the "
                   "largest gap on the whole circle is 26.565°.", size=14, fill=MUT,
          halo=False)
    s.save("fig3_primitive_directions.svg")


# ================================================================ FIG 4
def fig4():
    s = Svg(960, 668, "Cost curves, their lower envelope, and the constant")
    L, Rt, T, B = 88, 600, 78, 440          # main panel
    a0, a1, y0, y1 = 0.0, 45.0, 0.0, 0.135

    def X(a):
        return L + (Rt - L) * (a - a0) / (a1 - a0)

    def Y(v):
        return B - (B - T) * (v - y0) / (y1 - y0)

    s.rect(L, T, Rt - L, B - T, f="#fcfcfd", s=GRID)
    for v in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        s.line(L, Y(v), Rt, Y(v), c=GRID, w=1.2)
        s.txt(L - 10, Y(v) + 5, f"{v:.2f}", size=13, fill=MUT, anchor="end", halo=False)
    for a in [0, 10, 15.93, 20, 26.57, 30, 34.73, 40, 45]:
        s.line(X(a), T, X(a), B, c=GRID, w=1.2)
    for a, lab in ((0, "0"), (10, "10"), (20, "20"), (30, "30"), (40, "40"),
                   (45, "45")):
        s.txt(X(a), B + 22, lab, size=13, fill=MUT, anchor="middle", halo=False)
    s.txt((L + Rt) / 2, B + 48, "normal direction  α  (degrees)", size=15, fill=INK,
          anchor="middle", halo=False)
    s.txt(L - 56, (T + B) / 2, "cost  g(w, n)", size=15, fill=INK, anchor="middle",
          halo=False, rot=-90)

    WS = (((1, 0), BLU, "(1,0)   |w| = 1"), ((2, 1), AMB, "(2,1)   |w| = √5"),
          ((1, 1), GRN, "(1,1)   |w| = √2"))
    for w, col, lab in WS:
        pts = []
        a = a0
        while a <= a1 + 1e-9:
            v = g(w, math.radians(a))
            if v <= y1:
                pts.append((X(a), Y(v)))
            a += 0.05
        s.path(polyline(pts), s=col, w=2.0, fo=0)

    env = []
    a = a0
    while a <= a1 + 1e-9:
        env.append((X(a), Y(min(g(w, math.radians(a)) for w, _, _ in WS))))
        a += 0.02
    s.path(polyline(env), s=INK, w=3.6)

    s.line(L, Y(EPS_C), Rt, Y(EPS_C), c=RED, w=1.8, dash="7 5")
    s.txt(Rt - 8, Y(EPS_C) - 10, "ε* = 0.0192026307637963…", size=15, fill=RED,
          anchor="end", weight="600")
    s.line(X(15.930625), Y(EPS_C), X(15.930625), B, c=RED, w=1.6, dash="5 4")
    s.circ(X(15.930625), Y(EPS_C), 13, s=RED, w=2.2)
    s.circ(X(15.930625), Y(EPS_C), 6, f=RED, fo=1)
    s.txt(X(15.930625), B + 22, "15.931", size=13, fill=RED, anchor="middle",
          weight="600", halo=False)
    s.circ(X(34.729139), Y(EPS_2), 5.5, f=INK, fo=1)
    s.txt(X(34.729139) + 10, Y(EPS_2) - 10, "0.011331", size=13, fill=MUT)
    s.txt(X(34.729139), B + 22, "34.729", size=13, fill=MUT, anchor="middle",
          halo=False)

    # nearest-angle selector, for contrast
    na = math.sqrt(5) * math.sin(math.radians(13.2825256) / 2) ** 2
    s.line(L, Y(na), Rt, Y(na), c=PUR, w=1.8, dash="3 4")
    s.txt(L + 10, Y(na) - 10, "0.029909  ←  what \"nearest angle\" would give", size=14,
          fill=PUR)

    lx, ly = 372, 100
    s.rect(lx - 12, ly - 24, 222, 96, f="#fff", s=GRID, rx=7, fo=0.94)
    for i, (w, col, lab) in enumerate(WS):
        s.line(lx, ly + i * 24 - 5, lx + 26, ly + i * 24 - 5, c=col, w=2.6)
        s.txt(lx + 34, ly + i * 24, lab, size=14, halo=False)
    s.line(lx, ly + 72 - 5, lx + 26, ly + 72 - 5, c=INK, w=3.6)
    s.txt(lx + 34, ly + 72, "lower envelope", size=14, weight="600", halo=False)

    # inset
    IL, IR, IT, IB = 660, 928, 150, 400
    iy1 = 0.0225
    s.rect(IL, IT, IR - IL, IB - IT, f="#fcfcfd", s=GRID)

    def IX(a):
        return IL + (IR - IL) * (a - 12.0) / (20.0 - 12.0)

    def IY(v):
        return IB - (IB - IT) * v / iy1

    for v in [0.005, 0.010, 0.015, 0.020]:
        s.line(IL, IY(v), IR, IY(v), c=GRID, w=1.1)
        s.txt(IL - 8, IY(v) + 4, f"{v:.3f}", size=11, fill=MUT, anchor="end",
              halo=False)
    for w, col, _ in WS[:2]:
        pts = []
        a = 12.0
        while a <= 20.0:
            v = g(w, math.radians(a))
            if v <= iy1:
                pts.append((IX(a), IY(v)))
            a += 0.02
        s.path(polyline(pts), s=col, w=2.2)
    s.line(IL, IY(EPS_C), IR, IY(EPS_C), c=RED, w=1.5, dash="6 4")
    s.circ(IX(15.930625), IY(EPS_C), 11, s=RED, w=2)
    s.circ(IX(15.930625), IY(EPS_C), 5, f=RED, fo=1)
    for a in (12, 14, 16, 18, 20):
        s.txt(IX(a), IB + 18, str(a), size=11, fill=MUT, anchor="middle", halo=False)
    s.txt((IL + IR) / 2, IT - 30, "the crossover, magnified", size=14, weight="600",
          anchor="middle", halo=False)
    s.txt((IL + IR) / 2, IT - 12, "(1,0) rising  ×  (2,1) falling", size=13, fill=MUT,
          anchor="middle", halo=False)

    s.txt(24, 34, "4.  The answer: the maximum of the lower envelope, at a two-witness "
                  "crossover", size=18, weight="600", halo=False)
    s.txt(24, 524, "Each witness costs g(w,n) = |w|·sin²(θ_w/2) — LENGTH-WEIGHTED. "
                   "The best witness at each α is the lower envelope (black); the",
          size=14, fill=MUT, halo=False)
    s.txt(24, 546, "worst α is where that envelope peaks. It peaks at a kink, where the "
                   "short misaligned (1,0) and the long well-aligned (2,1) cost the",
          size=14, fill=MUT, halo=False)
    s.txt(24, 568, "same. Ignoring the length weight and simply taking the nearest "
                   "direction gives 0.029909 — 1.56× too pessimistic (purple).",
          size=14, fill=MUT, halo=False)
    s.txt(24, 604, "By the 8-fold lattice symmetry the octant [0°,45°] is the whole "
                   "story; ε* = (3 − √5 − √(2√5 − 4))/4, a degree-4 algebraic number.",
          size=14, fill=INK, halo=False)
    s.save("fig4_envelope.svg")


# ================================================================ FIG 5
def fig5():
    s = Svg(940, 620, "The critical configuration: a triple tangency")
    S, OX, OY = 132, 285, 415
    nx, ny = math.cos(A_STAR), math.sin(A_STAR)
    tx, ty = -ny, nx

    def P(x, y):
        return (OX + S * x, OY - S * y)

    s.add('<defs><pattern id="h5" width="10" height="10" '
          'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
          '<rect width="10" height="10" fill="#f1f5f9"/>'
          '<line x1="0" y1="0" x2="0" y2="10" stroke="#cbd5e1" stroke-width="1.6"/>'
          '</pattern></defs>')
    c = (-EPS_C * nx, -EPS_C * ny)
    a = (c[0] - 2.4 * tx, c[1] - 2.4 * ty)
    b = (c[0] + 4.2 * tx, c[1] + 4.2 * ty)
    deep = 1.1
    s.path(polyline([P(*a), P(*b), P(b[0] - deep * nx, b[1] - deep * ny),
                     P(a[0] - deep * nx, a[1] - deep * ny)]) + " Z",
           f="url(#h5)", s="none", w=0)
    s.line(*P(*a), *P(*b), c=INK, w=3.2)

    for i in range(-2, 5):
        for j in range(-2, 4):
            if (i, j) in ((0, 0), (1, 0), (2, 1)):
                continue
            s.circ(*P(i, j), 3.0, f=FAINT, fo=0.85)

    dA = EPS_C + nx
    dB = EPS_C + 2 * nx + ny
    s.circ(*P(2, 1), dB * S, s=AMB, f=AMBF, w=2.4, fo=0.10)
    s.circ(*P(1, 0), dA * S, s=BLU, f=BLUF, w=2.4, fo=0.12)
    s.circ(*P(0, 0), EPS_C * S, s=RED, f=RED, w=2.2, fo=0.30)

    for pt, d, col in (((1, 0), dA, BLU), ((2, 1), dB, AMB)):
        s.line(*P(*pt), *P(pt[0] - d * nx, pt[1] - d * ny), c=col, w=1.8, dash="5 4")
        s.line(*P(0, 0), *P(*pt), c=INK, w=1.8, dash="7 5")
        s.circ(*P(*pt), 6.5, f=col, fo=1)
        # tangency with the target ball
        L = math.hypot(*pt)
        t = (pt[0] * EPS_C / L, pt[1] * EPS_C / L)
        s.circ(*P(*t), 9, s=RED, w=2)
    s.circ(*P(0, 0), 6, f=RED, fo=1)

    s.txt(*[v + o for v, o in zip(P(0, 0), (-14, -14))], "v₁", size=19, fill=RED,
          weight="600", anchor="end")
    s.txt(*[v + o for v, o in zip(P(1, 0), (6, 30))], "(1,0)", size=17, fill=BLU,
          weight="600", anchor="middle")
    s.txt(*[v + o for v, o in zip(P(2, 1), (14, -10))], "(2,1)", size=17, fill=AMB,
          weight="600")
    s.txt(*[v + o for v, o in zip(P(3.15, -0.30), (0, 0))], "Σ", size=22,
          style='font-style="italic"')

    s.rect(596, 66, 322, 250, f="#fff", s=GRID, rx=8)
    rows = (("α*", "15.930625116297946°"), ("ε* = d₁", "0.019202630763796"),
            ("d₀ for (1,0)", "0.980797369236"), ("d₀ for (2,1)", "2.216865346736"),
            ("θ for (1,0)", "15.930625°"), ("θ for (2,1)", "10.634426°"),
            ("g for both", "0.019202630763796"))
    s.txt(614, 94, "Both witnesses tie, exactly", size=15, weight="600", halo=False)
    for i, (k, v) in enumerate(rows):
        y = 122 + i * 24
        s.txt(614, y, k, size=13, fill=MUT, halo=False)
        s.txt(902, y, v, size=13, fill=INK, anchor="end", halo=False)
    s.line(610, 288, 904, 288, c=GRID, w=1.2)
    s.txt(614, 306, "slack d₀ + ε* − |w| = 0 for both", size=13, fill=RED, halo=False)

    s.txt(24, 34, "5.  At α* the target ball is tangent to Σ and to BOTH witness balls",
          size=18, weight="600", halo=False)
    s.txt(24, 556, "Every ball is tangent to Σ (its radius is its centre's true "
                   "distance). At the critical direction the target ball is "
                   "additionally", size=14, fill=MUT, halo=False)
    s.txt(24, 578, "externally tangent to both witness balls at once (red rings) — the "
                   "rule fails against each by a margin of exactly zero.",
          size=14, fill=MUT, halo=False)
    s.save("fig5_critical_configuration.svg")


# ================================================================ FIG 6
def fig6():
    s = Svg(940, 576, "eps*(W): the plateau structure")
    L, Rt, T, B = 92, 890, 80, 400
    w0, w1, y0, y1 = 1.6, 4.0, 0.0, 0.055

    def X(w):
        return L + (Rt - L) * (w - w0) / (w1 - w0)

    def Y(v):
        return B - (B - T) * (v - y0) / (y1 - y0)

    CAND = [(a, b) for a in range(-6, 7) for b in range(-6, 7)
            if (a, b) != (0, 0) and math.hypot(a, b) <= 6.5]

    def eps_star(W, coarse=1400):
        best = 0.0
        for i in range(coarse + 1):
            al = (math.pi / 2) * i / coarse
            m = None
            for w in CAND:
                gg, dd = g(w, al), d0_thr(w, al)
                if dd <= RD or dd > W:
                    continue
                if m is None or gg < m:
                    m = gg
            if m is not None and m > best:
                best = m
        return best

    s.rect(L, T, Rt - L, B - T, f="#fcfcfd", s=GRID)
    for v in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        s.line(L, Y(v), Rt, Y(v), c=GRID, w=1.2)
        s.txt(L - 10, Y(v) + 5, f"{v:.2f}", size=13, fill=MUT, anchor="end", halo=False)
    for w in [2.0, 2.5, 3.0, 3.5, 4.0]:
        s.line(X(w), T, X(w), B, c=GRID, w=1.2)
        s.txt(X(w), B + 22, f"{w:.1f}", size=13, fill=MUT, anchor="middle", halo=False)
    s.txt((L + Rt) / 2, B + 46, "band half-width  W  (voxels)", size=15,
          anchor="middle", halo=False)
    s.txt(L - 58, (T + B) / 2, "ε*(W)", size=15, anchor="middle", halo=False, rot=-90)

    pts = []
    n = 240
    for i in range(n + 1):
        W = w0 + (w1 - w0) * i / n
        pts.append((X(W), Y(eps_star(W))))
    s.path(polyline(pts), s=INK, w=3.0)

    for wv, lab, col in ((math.sqrt(5), "√5 = 2.2361", TEAL),
                         (3.160767557306, "W_hi = 3.16077", TEAL)):
        s.line(X(wv), T, X(wv), B, c=col, w=1.6, dash="6 4")
        s.txt(X(wv), T - 10, lab, size=13, fill=col, anchor="middle", halo=False)
    s.line(X(3.0), T, X(3.0), B, c=RED, w=2.0)
    s.txt(X(3.0), T - 32, "W = 3", size=16, fill=RED, anchor="middle", weight="600",
          halo=False)

    s.path(f"M {X(math.sqrt(5)):.1f} {Y(EPS_C):.1f} L {X(3.160767557306):.1f} "
           f"{Y(EPS_C):.1f}", s=RED, w=4.5)
    s.txt((X(math.sqrt(5)) + X(3.16077)) / 2, Y(EPS_C) - 14,
          "ε* = 0.0192026307637963", size=15, fill=RED, weight="600", anchor="middle")

    s.txt(X(1.72), Y(0.0449) - 12, "0.044910  =  Borgefors 1986", size=14, fill=PUR)
    s.txt(X(3.42), Y(0.01133) - 12, "0.011331", size=14, fill=MUT)

    s.txt(24, 34, "6.  ε*(W) is a staircase; W = 3 sits inside the 0.0192 plateau",
          size=18, weight="600", halo=False)
    s.txt(24, 486, "Threshold admissibility rule shown. Under the rule the pipeline "
                   "actually implements the plateau is [2.2795, 3.1785) — W = 3 is "
                   "inside", size=14, fill=MUT, halo=False)
    s.txt(24, 508, "both, with about 6% margin on the high side. The step down at "
                   "W_hi is where (3,1) — 12× cheaper, and a convergent of tan α* — "
                   "finally", size=14, fill=MUT, halo=False)
    s.txt(24, 530, "fits inside the band. That amputation of the good rational "
                   "approximations is what sets the constant at W = 3.",
          size=14, fill=MUT, halo=False)
    s.save("fig6_plateau.svg")


# ================================================================ FIG 7
def fig7():
    s = Svg(940, 576, "Curvature: a flat interface is not the worst case")
    L, Rt, T, B = 92, 890, 80, 396
    y0, y1 = 0.0, 0.055
    r0, r1 = 1.7, 100.0
    K = 0.028315226153961

    def X(R):
        return L + (Rt - L) * (math.log(R) - math.log(r0)) / (math.log(r1) - math.log(r0))

    def Y(v):
        return B - (B - T) * (v - y0) / (y1 - y0)

    CAND = [(a, b) for a in range(-7, 8) for b in range(-7, 8)
            if (a, b) != (0, 0) and math.hypot(a, b) <= 7.5]

    def eps_star_R(R, concave, coarse=900):
        best = 0.0
        for i in range(coarse + 1):
            al = (math.pi / 2) * i / coarse
            m = None
            for w in CAND:
                gg, dp = g(w, al), d0_thr(w, al)
                if concave:
                    if dp >= R:
                        continue
                    e = gg / (1 - dp / R)
                else:
                    e = gg / (1 + dp / R)
                d0 = math.hypot(*w) - e
                if d0 <= RD or d0 > 3.0:
                    continue
                if m is None or e < m:
                    m = e
            if m is not None and m > best:
                best = m
        return best

    s.rect(L, T, Rt - L, B - T, f="#fcfcfd", s=GRID)
    for v in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        s.line(L, Y(v), Rt, Y(v), c=GRID, w=1.2)
        s.txt(L - 10, Y(v) + 5, f"{v:.2f}", size=13, fill=MUT, anchor="end", halo=False)
    for R in [2, 3, 5, 8, 12, 20, 35, 60, 100]:
        s.line(X(R), T, X(R), B, c=GRID, w=1.2)
        s.txt(X(R), B + 22, str(R), size=13, fill=MUT, anchor="middle", halo=False)
    s.txt((L + Rt) / 2, B + 46, "curvature radius  R  (voxels, log scale)", size=15,
          anchor="middle", halo=False)
    s.txt(L - 58, (T + B) / 2, "ε*", size=15, anchor="middle", halo=False, rot=-90)

    s.line(L, Y(EPS_C), Rt, Y(EPS_C), c=INK, w=1.8, dash="7 5")
    s.txt(Rt - 8, Y(EPS_C) - 10, "ε* = 0.0192026  (planar)", size=14, anchor="end")

    n = 90
    for concave, col, lab in ((True, RED, "concave exterior"), (False, TEAL, "convex")):
        ex, fo = [], []
        for i in range(n + 1):
            R = math.exp(math.log(r0) + (math.log(r1) - math.log(r0)) * i / n)
            ex.append((X(R), Y(eps_star_R(R, concave))))
            v = EPS_C + (K / R if concave else -K / R)
            if y0 <= v <= y1:
                fo.append((X(R), Y(v)))
        s.path(polyline(ex), s=col, w=3.2)
        s.path(polyline(fo), s=col, w=1.8, dash="5 5", cap="butt")

    s.txt(X(2.35), Y(0.0455), "concave, exact", size=15, fill=RED, weight="600")
    s.txt(X(3.05), Y(0.0332), "first order  ε* + K/R", size=13, fill=RED)
    s.txt(X(2.15), Y(0.0128), "convex, exact — saturates", size=15, fill=TEAL,
          weight="600")
    s.txt(X(2.3), Y(0.0042), "first order  ε* − K/R", size=13, fill=TEAL)

    s.txt(24, 34, "7.  Concave curvature costs; convex helps, but only up to a point",
          size=18, weight="600", halo=False)
    s.txt(24, 486, "Exact (solid): ε_thr(w) = g/(1 ∓ d₀/R). First order (dashed): "
                   "ε* ± K/R with K = 0.0283152. The linear form is good to R ≈ 20 but "
                   "understates", size=14, fill=MUT, halo=False)
    s.txt(24, 508, "concave damage by 3.2× at R = 2 — and on the convex side it wrongly "
                   "predicts free certification at R = 1.47. The exact convex value",
          size=14, fill=MUT, halo=False)
    s.txt(24, 530, "flattens near 0.53·ε* and never reaches zero. Size any safety "
                   "factor from the tightest CONCAVE feature radius.", size=14,
          fill=MUT, halo=False)
    s.save("fig7_curvature.svg")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
