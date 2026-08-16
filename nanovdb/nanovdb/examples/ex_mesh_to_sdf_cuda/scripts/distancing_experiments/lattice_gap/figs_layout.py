#!/usr/bin/env python3
"""Layout pass applied on top of figs.py (see README)."""
import math, os
from figs import *
import figs as F
#!/usr/bin/env python3
"""Final layout pass on figures 1 and 5."""
import math







HATCH = ('<defs><pattern id="hh" width="10" height="10" patternUnits="userSpaceOnUse" '
         'patternTransform="rotate(45)"><rect width="10" height="10" fill="#f1f5f9"/>'
         '<line x1="0" y1="0" x2="0" y2="10" stroke="#cbd5e1" stroke-width="1.6"/>'
         '</pattern></defs>')


def fig1():
    s = Svg(940, 660, "The pairwise rule and its marginal case")
    S, OX, OY = 105, 200, 520
    d1, d0 = 0.5, 2.0
    dx = 2 * math.sqrt(d0 * d1)

    def P(x, y):
        return (OX + S * x, OY - S * y)

    s.add(HATCH)
    s.rect(20, OY, 900, 56, f="url(#hh)")
    s.line(20, OY, 920, OY, c=INK, w=3.4)
    s.txt(908, OY - 14, "Σ", size=22, anchor="end", style='font-style="italic"')
    s.txt(30, OY + 36, "solid", size=14, fill=MUT, halo=False)
    s.txt(30, 118, "exterior", size=14, fill=MUT)

    v1, v0 = (0.0, d1), (dx, d0)
    s.circ(*P(*v0), d0 * S, s=BLU, f=BLUF, w=2.4, fo=0.11)
    s.circ(*P(*v1), d1 * S, s=AMB, f=AMBF, w=2.4, fo=0.20)
    s.line(*P(*v1), *P(v1[0], 0), c=AMB, w=2, dash="5 4")
    s.line(*P(*v0), *P(v0[0], 0), c=BLU, w=2, dash="5 4")
    s.line(*P(*v1), *P(*v0), c=INK, w=2, dash="7 5")

    t = (v1[0] + (v0[0] - v1[0]) * d1 / (d0 + d1),
         v1[1] + (v0[1] - v1[1]) * d1 / (d0 + d1))
    pt = P(*t)
    s.circ(*pt, 13, s=RED, w=2.2)
    s.circ(*pt, 6, f=RED, fo=1)
    s.path(f"M {pt[0]:.1f} {pt[1]:.1f} L {pt[0]+82:.1f} {pt[1]-118:.1f} "
           f"L {pt[0]+128:.1f} {pt[1]-118:.1f}", s=RED, w=1.6, f="none")
    s.txt(pt[0] + 136, pt[1] - 122, "the balls merely touch —", size=14, fill=RED)
    s.txt(pt[0] + 136, pt[1] - 102, "the strict rule fails, by zero", size=14, fill=RED)

    pv0 = P(*v0)
    a1d = 90.0
    a2d = math.degrees(math.atan2(P(*v1)[1] - pv0[1], P(*v1)[0] - pv0[0]))
    r = 56
    p1 = (pv0[0] + r * math.cos(math.radians(a1d)),
          pv0[1] + r * math.sin(math.radians(a1d)))
    p2 = (pv0[0] + r * math.cos(math.radians(a2d)),
          pv0[1] + r * math.sin(math.radians(a2d)))
    s.path(f"M {p1[0]:.1f} {p1[1]:.1f} A {r} {r} 0 0 1 {p2[0]:.1f} {p2[1]:.1f}",
           s=PUR, w=2.4)
    am = math.radians((a1d + a2d) / 2)
    s.txt(pv0[0] + (r + 20) * math.cos(am), pv0[1] + (r + 20) * math.sin(am) + 7,
          "θ", size=23, fill=PUR, anchor="middle", style='font-style="italic"')

    s.circ(*P(*v1), 6.5, f=AMB, fo=1)
    s.circ(*P(*v0), 6.5, f=BLU, fo=1)
    s.circ(*P(v1[0], 0), 4.5, f=INK, fo=1)
    s.circ(*P(v0[0], 0), 4.5, f=INK, fo=1)
    s.txt(P(*v1)[0] - 16, P(*v1)[1] - 8, "v₁", size=20, fill=AMB, weight="600",
          anchor="end")
    s.txt(P(*v1)[0] - 16, P(*v1)[1] + 12, "target", size=13, fill=MUT, anchor="end")
    s.txt(P(*v0)[0] + 16, P(*v0)[1] - 6, "v₀ = v₁ + w", size=20, fill=BLU, weight="600")
    s.txt(P(*v0)[0] + 16, P(*v0)[1] + 14, "witness (already certified)", size=13,
          fill=MUT)
    s.txt(P(v1[0], d1 / 2)[0] + 10, P(v1[0], d1 / 2)[1] + 5, "d₁ = ε", size=16, fill=AMB)
    s.txt(P(v0[0], d0 / 2)[0] + 12, P(v0[0], d0 / 2)[1] + 34, "d₀ = ε + w·n", size=16,
          fill=BLU)
    mid = P((v1[0] + v0[0]) / 2, (v1[1] + v0[1]) / 2)
    s.txt(mid[0] + 26, mid[1] + 20, "|w|", size=16, anchor="middle")
    s.txt(P(v1[0], 0)[0] - 8, P(v1[0], 0)[1] + 22, "P₁", size=15, anchor="end",
          halo=False)
    s.txt(P(v0[0], 0)[0] + 10, P(v0[0], 0)[1] + 22, "P₀", size=15, halo=False)

    s.rect(596, 60, 324, 222, f="#fff", s=GRID, rx=8)
    s.txt(614, 90, "The rule, and its reduction", size=15, weight="600", halo=False)
    s.txt(614, 120, "d₁ + d₀ > |w|", size=17, weight="600", halo=False)
    s.txt(614, 146, "Substitute d₀ = ε + w·n and d₁ = ε:", size=13, fill=MUT,
          halo=False)
    s.txt(614, 176, "ε > (|w| − w·n)/2", size=17, weight="600", halo=False)
    s.txt(614, 204, "  = |w|·sin²(θ/2)  =:  g(w,n)", size=17, weight="600", halo=False)
    s.line(610, 222, 906, 222, c=GRID, w=1.2)
    s.txt(614, 246, "θ = ∠P₀v₀v₁, at the witness, between", size=13, fill=MUT,
          halo=False)
    s.txt(614, 266, "its normal drop and the target.", size=13, fill=MUT, halo=False)

    s.txt(24, 34, "1.  The certification rule, drawn at the exact margin", size=18,
          weight="600", halo=False)
    s.txt(24, 616, "Both balls are tangent to Σ — each radius IS its centre's true "
                   "distance, so no ball can cross the surface. Certification asks "
                   "whether the", size=14, fill=MUT, halo=False)
    s.txt(24, 638, "two balls OVERLAP. Here they only touch: |w| = d₀ + d₁ exactly, "
                   "with feet separated by dx = 2√(d₀d₁) = 2.", size=14, fill=MUT,
          halo=False)
    s.save("fig1_rule_and_tangency.svg")


def fig5():
    s = Svg(940, 672, "The critical configuration")
    nx, ny = math.cos(A_STAR), math.sin(A_STAR)
    tx, ty = -ny, nx

    def frame(w):
        return (w[0] * tx + w[1] * ty, w[0] * nx + w[1] * ny + EPS_C)

    S, OX, OY = 100, 300, 500

    def P(x, y):
        return (OX + S * x, OY - S * y)

    s.add(HATCH)
    s.rect(20, OY, 900, 78, f="url(#hh)")

    A, B = (1, 0), (2, 1)
    fa, fb, f1 = frame(A), frame(B), frame((0, 0))
    dA, dB = fa[1], fb[1]
    s.circ(*P(fb[0], dB), dB * S, s=AMB, f=AMBF, w=2.4, fo=0.10)
    s.circ(*P(fa[0], dA), dA * S, s=BLU, f=BLUF, w=2.4, fo=0.13)
    s.line(20, OY, 920, OY, c=INK, w=3.4)
    s.txt(908, OY - 14, "Σ", size=22, anchor="end", style='font-style="italic"')

    for c, col in ((fa, BLU), (fb, AMB)):
        s.line(*P(c[0], c[1]), *P(c[0], 0), c=col, w=1.8, dash="5 4")
        s.line(*P(f1[0], f1[1]), *P(c[0], c[1]), c=INK, w=1.7, dash="7 5")
        s.circ(*P(c[0], c[1]), 6.5, f=col, fo=1)
        s.circ(*P(c[0], 0), 4.5, f=INK, fo=1)
    s.circ(*P(f1[0], f1[1]), 5.5, f=RED, fo=1)

    s.txt(*[v + o for v, o in zip(P(fa[0], dA), (-14, -10))], "(1,0)", size=17,
          fill=BLU, weight="600", anchor="end")
    s.txt(*[v + o for v, o in zip(P(fa[0], dA), (-14, 10))], "d₀ = 0.98080", size=13,
          fill=MUT, anchor="end")
    s.txt(*[v + o for v, o in zip(P(fb[0], dB), (16, -8))], "(2,1)", size=17, fill=AMB,
          weight="600")
    s.txt(*[v + o for v, o in zip(P(fb[0], dB), (16, 12))], "d₀ = 2.21687", size=13,
          fill=MUT)
    s.txt(*[v + o for v, o in zip(P(f1[0], f1[1]), (-12, -10))], "v₁", size=17,
          fill=RED, weight="600", anchor="end")

    yb = OY + 40
    s.line(P(fa[0], 0)[0], yb, P(fb[0], 0)[0], yb, c=TEAL, w=2.2)
    for xv in (fa[0], 0.0, fb[0]):
        s.line(P(xv, 0)[0], OY + 4, P(xv, 0)[0], yb + 7, c=TEAL, w=1.3, dash="4 3")
        s.circ(P(xv, 0)[0], yb, 3.5, f=TEAL, fo=1)
    s.txt(P(fa[0], 0)[0] - 12, yb + 6, "p = −0.2745", size=13, fill=TEAL, anchor="end")
    s.txt(P(fb[0], 0)[0] + 12, yb + 6, "p = +0.4126", size=13, fill=TEAL)
    s.txt(OX, yb - 12, "feet straddle the target's foot", size=13, fill=TEAL,
          anchor="middle")

    IX, IY, IR = 728, 300, 138
    Si = 2600.0
    s.circ(IX, IY, IR + 20, f="#fff", s=GRID, w=1.6, fo=1)

    def Q(x, y):
        return (IX + Si * (x - f1[0]), IY - Si * (y - f1[1]))

    s.add(f'<clipPath id="ic"><circle cx="{IX}" cy="{IY}" r="{IR+18}"/></clipPath>')
    s.add('<g clip-path="url(#ic)">')
    for c, d, col, colf in ((fb, dB, AMB, AMBF), (fa, dA, BLU, BLUF)):
        s.add(f'<circle cx="{Q(c[0], d)[0]:.1f}" cy="{Q(c[0], d)[1]:.1f}" '
              f'r="{d*Si:.1f}" fill="{colf}" fill-opacity="0.12" stroke="{col}" '
              f'stroke-width="2.6"/>')
    yl = IY + EPS_C * Si
    s.add(f'<line x1="{IX-IR-20}" y1="{yl:.1f}" x2="{IX+IR+20}" y2="{yl:.1f}" '
          f'stroke="{INK}" stroke-width="3"/>')
    s.add(f'<circle cx="{IX}" cy="{IY}" r="{EPS_C*Si:.1f}" fill="{RED}" '
          f'fill-opacity="0.28" stroke="{RED}" stroke-width="2.4"/>')
    s.add('</g>')
    s.circ(IX, IY, 4, f=RED, fo=1)
    for w, col, lab, off in ((A, BLU, "tangent to (1,0)", (-118, -52)),
                             (B, AMB, "tangent to (2,1)", (34, -76))):
        fw = frame(w)
        d = math.hypot(fw[0] - f1[0], fw[1] - f1[1])
        tp = (f1[0] + (fw[0] - f1[0]) * EPS_C / d,
              f1[1] + (fw[1] - f1[1]) * EPS_C / d)
        q = Q(*tp)
        s.circ(*q, 8, s=RED, w=2.2)
        s.line(q[0], q[1], q[0] + off[0] * 0.55, q[1] + off[1], c=col, w=1.4)
        s.txt(q[0] + off[0], q[1] + off[1] - 6, lab, size=12, fill=col, halo=True)
    s.txt(IX, IY - IR - 36, "magnified 26× about v₁", size=14, weight="600",
          anchor="middle", halo=False)
    s.txt(IX, yl + 26, "Σ", size=15, fill=MUT, anchor="middle", halo=False)
    s.txt(IX, IY + IR + 42, "the witness balls are 51× and 115× larger,", size=12,
          fill=MUT, anchor="middle", halo=False)
    s.txt(IX, IY + IR + 60, "so at this zoom they look straight", size=12, fill=MUT,
          anchor="middle", halo=False)

    s.rect(24, 62, 252, 130, f="#fff", s=GRID, rx=8)
    s.txt(42, 90, "α* = 15.930625116297946°", size=14, weight="600", halo=False)
    s.txt(42, 114, "cos α* + sin α* = √5 − 1", size=13, fill=MUT, halo=False)
    s.txt(42, 144, "g((1,0)) = g((2,1)) = ε*", size=14, weight="600", halo=False)
    s.txt(42, 168, "slack d₀ + ε* − |w| = 0, both", size=13, fill=RED, halo=False)

    s.txt(24, 34, "5.  The critical configuration: three circles on a line, mutually "
                  "tangent", size=18, weight="600", halo=False)
    s.txt(24, 626, "Drawn in the frame where Σ is horizontal. Every ball is tangent to "
                   "Σ by construction; at α* the target ball is ALSO tangent to both",
          size=14, fill=MUT, halo=False)
    s.txt(24, 648, "witness balls at once. Their feet straddle the target's foot — the "
                   "geometric signature of the crossover.", size=14, fill=MUT,
          halo=False)
    s.save("fig5_critical_configuration.svg")



if __name__ == '__main__':
    fig1()
    fig5()
