import math, random
from slib import *

ws14 = witnesses(14)
EPS_THR = 0.036662265118829642
EPS_EXA = 0.038443423574719627

# ---------------------------------------------------------------- 1. covering radius
prim = []
for (a, b, c, L) in witnesses(9):
    if L <= 3.0 + 1e-12 and math.gcd(math.gcd(abs(a), abs(b)), abs(c)) == 1:
        prim.append((a / L, b / L, c / L))
print("primitive directions with |w| <= 3 :", len(prim))


def cover(n):
    return max(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] for p in prim)


# exhaustive Voronoi-vertex enumeration: circumcentres of direction triples
best = (2.0, None)
N = len(prim)
for i in range(N):
    ax, ay, az = prim[i]
    for j in range(i + 1, N):
        bx, by, bz = prim[j]
        for k in range(j + 1, N):
            cx, cy, cz = prim[k]
            # solve n.(a-b)=0, n.(a-c)=0  ->  n parallel to (a-b) x (a-c)
            ux, uy, uz = ax - bx, ay - by, az - bz
            vx, vy, vz = ax - cx, ay - cy, az - cz
            nx = uy * vz - uz * vy; ny = uz * vx - ux * vz; nz = ux * vy - uy * vx
            s = math.sqrt(nx * nx + ny * ny + nz * nz)
            if s < 1e-12:
                continue
            for sg in (1.0, -1.0):
                n = (sg * nx / s, sg * ny / s, sg * nz / s)
                d = n[0] * ax + n[1] * ay + n[2] * az
                if d <= 0:
                    continue
                if abs(cover(n) - d) < 1e-12:      # genuine Voronoi vertex
                    if d < best[0]:
                        best = (d, n)
delta = math.degrees(math.acos(best[0]))
print(f"covering radius (exhaustive triple enumeration) = {delta:.9f} deg  at n = "
      f"({best[1][0]:.9f},{best[1][1]:.9f},{best[1][2]:.9f})")
print(f"  tan^2(delta/2) = {math.tan(math.radians(delta)/2)**2:.12f}"
      f"   W tan^2 = {3*math.tan(math.radians(delta)/2)**2:.12f}"
      f"   sqrt6 sin^2 = {math.sqrt(6)*math.sin(math.radians(delta)/2)**2:.12f}")

# ---------------------------------------------------------------- 2. solid-angle census
random.seed(20260817)
from collections import Counter
cnt = Counter(); tot = 0
for _ in range(200000):
    while True:
        x = random.gauss(0, 1); y = random.gauss(0, 1); z = random.gauss(0, 1)
        s = math.sqrt(x * x + y * y + z * z)
        if s > 1e-9:
            break
    n = (abs(x) / s, abs(y) / s, abs(z) / s)
    n = tuple(sorted(n, reverse=True))
    bw, bg = None, 1e9
    for (a, b, c, L) in ws14:
        d = a * n[0] + b * n[1] + c * n[2]
        d0 = (L + d) / 2
        if d0 <= RD or d0 > 3.0:
            continue
        g = (L - d) / 2
        if g < bg:
            bg, bw = g, tuple(sorted((abs(a), abs(b), abs(c)), reverse=True))
    cnt[bw] += 1; tot += 1
print("\nsolid-angle census of the winning witness (threshold rule, 200k uniform on S^2):")
for w, c in cnt.most_common():
    print(f"   {w}: {100.0*c/tot:5.2f} %")

# ---------------------------------------------------------------- 3. minimal covering set
reps = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (2, 2, 1)]
orb = set()
for r in reps:
    for p in __import__('itertools').permutations(r):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    v = (sx * p[0], sy * p[1], sz * p[2])
                    if v != (0, 0, 0):
                        orb.add(v)
print("\norbit size of the six classes:", len(orb))


def subset_ws(drop):
    out = []
    for r in reps:
        if r == drop:
            continue
        for p in __import__('itertools').permutations(r):
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        v = (sx * p[0], sy * p[1], sz * p[2])
                        out.append((v[0], v[1], v[2], math.sqrt(sum(t * t for t in v))))
    return list({(a, b, c, L) for (a, b, c, L) in out})


print("minimal-covering-set check: max-min over the fundamental domain with one class dropped")
for d in reps:
    lo, up, nc = bnb('thr', subset_ws(d), W=3.0, tol=1e-10, maxcells=200000)
    print(f"   drop {d}: [{lo:.15f}, {up:.15f}]   ratio to eps*_thr {lo/EPS_THR:.4f}")

# ---------------------------------------------------------------- 4. multi-hop lemma
print("\nmulti-hop: a BARRIER witness u has UDF(u) <= r_d, so certifying a target at depth eps")
print("   needs eps + r_d > |u| >= 1, i.e. eps > 1 - r_d = %.15f" % (1 - RD))
print("   both constants are far below that, so newly certified barrier points never help.")

# ---------------------------------------------------------------- 5. peak sharpness
def tangents(n):
    a = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
    t1 = (n[1]*a[2]-n[2]*a[1], n[2]*a[0]-n[0]*a[2], n[0]*a[1]-n[1]*a[0])
    s = math.sqrt(sum(c*c for c in t1)); t1 = tuple(c/s for c in t1)
    t2 = (n[1]*t1[2]-n[2]*t1[1], n[2]*t1[0]-n[0]*t1[2], n[0]*t1[1]-n[1]*t1[0])
    return t1, t2


nstar = (0.926675469762340715, 0.309392507737448981, 0.213421765283388402)
t1, t2 = tangents(nstar)
for frac, lbl in ((0.01, '1%'), (0.05, '5%'), (0.10, '10%')):
    worst = 1e9
    for k in range(720):
        th = 2 * math.pi * k / 720
        d = (math.cos(th)*t1[0]+math.sin(th)*t2[0], math.cos(th)*t1[1]+math.sin(th)*t2[1],
             math.cos(th)*t1[2]+math.sin(th)*t2[2])
        a, b = 0.0, 0.05
        for _ in range(60):
            m = (a+b)/2
            nn = tuple(nstar[i]*math.cos(m)+d[i]*math.sin(m) for i in range(3))
            if F_thr(nn, ws14) >= EPS_THR*(1-frac):
                a = m
            else:
                b = m
        worst = min(worst, a)
    print(f"peak sharpness thr: losing {lbl} of eps* needs {math.degrees(worst):.6f} deg")
