#!/usr/bin/env python3
"""3D angular quantities: unweighted covering radius vs the weighted critical angles."""
import math

RD3 = math.sqrt(3) / 2
W = 3.0
EPS_T = 0.036662265118829642
EPS_E = 0.038443423574719627
N_T = (0.9266754697623407154, 0.3093925077374489810, 0.2134217652833884020)
N_E = (0.7174389352143008047, 0.6198877800093549910, 0.3178372451957822447)


def prim(R):
    out = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                if (a, b, c) == (0, 0, 0):
                    continue
                L = math.sqrt(a * a + b * b + c * c)
                if L <= R and math.gcd(math.gcd(abs(a), abs(b)), abs(c)) == 1:
                    out.append((a, b, c, L))
    return out


def ang(w, n):
    c = (w[0] * n[0] + w[1] * n[1] + w[2] * n[2]) / w[3]
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def cover(P, N=60):
    base = set()
    for i in range(1, N + 1):
        for j in range(0, i + 1):
            for k in range(0, j + 1):
                L = math.sqrt(i * i + j * j + k * k)
                base.add((i / L, j / L, k / L))
    best = (0.0, None)
    for n in base:
        v = min(ang(w, n) for w in P)
        if v > best[0]:
            best = (v, n)
    n0, step = best[1], 0.03
    for _ in range(200):
        imp = False
        for ax in range(3):
            for d in (-step, step):
                m = list(n0)
                m[ax] += d
                L = math.hypot(*m)
                m = tuple(x / L for x in m)
                v = min(ang(w, m) for w in P)
                if v > best[0]:
                    best = (v, m)
                    n0 = m
                    imp = True
        if not imp:
            step *= 0.6
    return best


print("(a) UNWEIGHTED: worst-case angle from an arbitrary normal to the nearest")
print("    primitive lattice direction  --  the direct analogue of 2D's 13.2825 deg")
for lab, R in (("3x3x3 shell, |w| <= sqrt3", math.sqrt(3) + 1e-9),
               ("|w| <= sqrt6 = 2.4495", math.sqrt(6) + 1e-9),
               ("|w| <= 3.04  (the W=3 budget)", 3.04)):
    P = prim(R)
    v, n = cover(P)
    print(f"    {lab:32s}: {v:8.4f} deg  ({len(P):3d} vectors)"
          f"  worst n = ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")
print(f"    {'2D: 16 primitive dirs of [-2,2]^2':32s}:  13.2825 deg\n")

print("(b) WEIGHTED: theta at the directions the weighted cost actually selects")
for lab, n, eps in (("threshold rule", N_T, EPS_T),
                    ("exact rule (what the pipeline does)", N_E, EPS_E)):
    P = prim(4.1)
    rows = []
    for w in P:
        wn = w[0] * n[0] + w[1] * n[1] + w[2] * n[2]
        if wn <= 0:
            continue
        rows.append(((w[3] - wn) / 2, w, ang(w, n), eps + wn))
    rows.sort()
    print(f"  {lab}:  eps* = {eps:.12f}")
    print(f"    {'w':>10} {'|w|':>8} {'theta':>10} {'g':>13} {'depth':>9}  admissible")
    for g, w, t, d in rows[:6]:
        print(f"    {str(w[:3]):>10} {w[3]:8.5f} {t:9.4f}d {g:13.9f} {d:9.5f}  "
              f"{'yes' if RD3 < d <= W else 'NO'}")
    adm = [r for r in rows if RD3 < r[3] <= W]
    ties = [r for r in adm if abs(r[0] - eps) < 1e-9]
    print(f"    tying witnesses and their theta: "
          f"{[(r[1][:3], round(r[2], 4)) for r in ties]}")
    print(f"    min theta over ADMISSIBLE witnesses = {min(r[2] for r in adm):8.4f} deg")
    b = min(rows, key=lambda r: r[2])
    print(f"    min theta over ALL witnesses        = {b[2]:8.4f} deg  "
          f"(w = {b[1][:3]}, depth {b[3]:.5f}, "
          f"{'admissible' if RD3 < b[3] <= W else 'BLOCKED'})\n")
