#!/usr/bin/env python3
"""Crude eps*_3D(W) map, both admissibility rules.

Seeded with the critical directions already located by the 3D workflow, because the
peaks are sharp enough that a blind grid misses them.
"""
import math

RD = math.sqrt(3) / 2                      # 3D covering radius
SEEDS = [
    (0.926675469762340715, 0.309392507737448981, 0.213421765283388402),  # n*_thr
    (0.717438935214300805, 0.619887780009354991, 0.317837245195782245),  # n*_exa
    (0.821854415127, 0.528210630753, 0.213421765283),                    # 2nd thr max
    (0.717438935214, 0.550510257217, 0.426872148234),                    # 3rd thr max
    (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (2, 2, 1),
    (3, 1, 1), (3, 2, 1), (3, 2, 2), (1, 1, 0.0001), (0.9, 0.4, 0.15),
]


def norm(v):
    L = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    a = sorted((abs(v[0]) / L, abs(v[1]) / L, abs(v[2]) / L), reverse=True)
    return (a[0], a[1], a[2])


def make_cands(Wmax):
    R = int(math.ceil(Wmax + RD)) + 1
    out = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            for c in range(-R, R + 1):
                if (a, b, c) == (0, 0, 0):
                    continue
                L = math.sqrt(a * a + b * b + c * c)
                if L <= Wmax + RD + 0.05:
                    out.append((a, b, c, L))
    return out


def eps_thr(n, W, C):
    """threshold rule: min over admissible w of g, admissible iff r_d < (|w|+w.n)/2 <= W"""
    best = None
    for a, b, c, L in C:
        wn = a * n[0] + b * n[1] + c * n[2]
        d0 = (L + wn) / 2
        if d0 <= RD or d0 > W:
            continue
        g = (L - wn) / 2
        if best is None or g < best:
            best = g
    return best if best is not None else RD


def eps_exa(n, W, C):
    """exact rule: sup of the uncertified set in (0, r_d]; may be a union of intervals"""
    iv = []
    for a, b, c, L in C:
        wn = a * n[0] + b * n[1] + c * n[2]
        lo = max((L - wn) / 2, RD - wn)
        hi = min(RD, W - wn)
        if hi > lo:
            iv.append((lo, hi))
    if not iv:
        return RD
    iv.sort(key=lambda t: -t[1])
    e = RD
    changed = True
    while changed:
        changed = False
        for lo, hi in iv:
            if lo < e <= hi:
                e = lo
                changed = True
    return max(e, 0.0)


def grid(N):
    pts = []
    for i in range(1, N + 1):
        for j in range(0, i + 1):
            for k in range(0, j + 1):
                pts.append(norm((i, j, k)))
    return sorted(set(pts))


def sweep(W, C, base, f, top=26, rounds=90):
    scored = sorted(((f(n, W, C), n) for n in base), reverse=True)[:top]
    best = scored[0]
    for v0, n0 in scored:
        n, v, step = n0, v0, 0.05
        for _ in range(rounds):
            improved = False
            for dx in (-step, step):
                for ax in range(3):
                    m = list(n)
                    m[ax] += dx
                    m = norm(m)
                    vv = f(m, W, C)
                    if vv > v:
                        n, v, improved = m, vv, True
            if not improved:
                step *= 0.55
                if step < 1e-12:
                    break
        if v > best[0]:
            best = (v, n)
    return best


BASE = grid(26) + [norm(s) for s in SEEDS]
print(f"# base directions in the fundamental domain: {len(BASE)}")
print(f"# {'W':>7} {'eps*_thr':>14} {'eps*_exact':>14}   argmax n (threshold)")
W = 1.5
rows = []
while W <= 5.001:
    C = make_cands(W)
    vt, nt = sweep(W, C, BASE, eps_thr)
    ve, _ = sweep(W, C, BASE, eps_exa)
    rows.append((W, vt, ve))
    print(f"  {W:7.3f} {vt:14.9f} {ve:14.9f}   "
          f"({nt[0]:.5f}, {nt[1]:.5f}, {nt[2]:.5f})")
    W += 0.05 if W < 3.4 else 0.2
