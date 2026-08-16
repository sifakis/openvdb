#!/usr/bin/env python3
"""STRATEGY 1: very fine deterministic sweep over [0,45deg] + local refinement
to machine precision around EVERY local maximum.  Also: symmetry check over
the full circle, and a-priori pruning justification."""
import math, sys, time
sys.path.insert(0, '/tmp/dist')
from core import BAR, cand_set, eps_dir, eps_val

W = 3.0
DEG = 180.0 / math.pi

# ---- 0. a-priori bound: w=(1,0) is admissible for all a in [0,45deg] -------
print("=" * 78)
print("0. a-priori bound and candidate pruning")
print("=" * 78)
lo, hi = 1e9, -1e9
for i in range(100001):
    a = (math.pi / 4) * i / 100000
    d0 = 0.5 * (1.0 + math.cos(a))
    lo = min(lo, d0); hi = max(hi, d0)
print(f"  w=(1,0) depth d0 over [0,45deg] in [{lo:.9f},{hi:.9f}]  (BAR={BAR:.9f}, W={W})")
print(f"  -> always admissible  => eps*(a) <= g_(1,0)(a) <= (1-cos45)/2 = {(1-math.cos(math.pi/4))/2:.12f}")
UB = (1 - math.cos(math.pi / 4)) / 2
print(f"  any w with g > {UB:.9f} is irrelevant; admissible => g = |w| - d0 >= |w| - W")
print(f"  => only |w| <= W + {UB:.9f} = {W+UB:.9f} can matter.")
Cfull = cand_set(2 * W)          # provably complete: adm & w.n>0 => |w| <= 2W
Cpru = cand_set(W + UB)
print(f"  |C_full(|w|<=2W={2*W})| = {len(Cfull)},  |C_pruned(|w|<={W+UB:.6f})| = {len(Cpru)}")
mism = 0
for i in range(200001):
    a = 2 * math.pi * i / 200000
    if abs(eps_val(a, W, Cfull) - eps_val(a, W, Cpru)) > 1e-15:
        mism += 1
print(f"  pruned vs full disagreement over 200k angles on FULL circle: {mism}")

# ---- 1. symmetry check ----------------------------------------------------
print("=" * 78)
print("1. dihedral symmetry of eps*(a)  (justifies restricting to [0,45deg])")
print("=" * 78)
worst = 0.0
for i in range(50001):
    a = 2 * math.pi * i / 50000
    v = eps_val(a, W, Cfull)
    for b in (-a, math.pi / 2 - a, math.pi / 2 + a, math.pi - a, math.pi + a):
        worst = max(worst, abs(eps_val(b, W, Cfull) - v))
print(f"  max |eps*(a) - eps*(sym a)| over 50k angles x 5 symmetries: {worst:.3e}")

# ---- 2. very fine sweep ---------------------------------------------------
print("=" * 78)
print("2. fine deterministic sweep of alpha over [0,45deg]")
print("=" * 78)
CLAIM = 0.019202630763796


def sweep(N, lo=0.0, hi=math.pi / 4, C=Cpru, W=W):
    step = (hi - lo) / N
    vals = []
    for i in range(N + 1):
        a = lo + step * i
        vals.append(eps_val(a, W, C))
    return vals, step


for N in (10000, 100000, 1000000, 4000000):
    t = time.time()
    vals, step = sweep(N)
    m = max(vals); k = vals.index(m)
    print(f"  N={N:>8}  step={step*DEG:.3e} deg   max={m:.15f}  at alpha={ (math.pi/4)*k/N*DEG:.9f} deg"
          f"   (max-claim = {m-CLAIM:+.3e})   [{time.time()-t:.1f}s]")

# ---- 3. every local maximum of the sampled curve, refined -----------------
print("=" * 78)
print("3. ALL local maxima of the fine sweep, each refined to machine precision")
print("=" * 78)
N = 1000000
vals, step = sweep(N)
locmax = []
for i in range(1, N):
    if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1] and vals[i] > 1e-9:
        locmax.append(i)
# merge adjacent plateaus
merged = []
for i in locmax:
    if merged and i - merged[-1][-1] <= 2:
        merged[-1].append(i)
    else:
        merged.append([i])
print(f"  {len(merged)} local-maximum clusters found on the sampled curve")


def refine(a0, h, C=Cpru, W=W, iters=300):
    """ternary/trisection on a bracket; eps*(.) is concave-ish near a kink max."""
    lo_, hi_ = a0 - h, a0 + h
    for _ in range(iters):
        m1 = lo_ + (hi_ - lo_) / 3.0
        m2 = hi_ - (hi_ - lo_) / 3.0
        if eps_val(m1, W, C) < eps_val(m2, W, C):
            lo_ = m1
        else:
            hi_ = m2
        if hi_ - lo_ < 1e-17:
            break
    a = 0.5 * (lo_ + hi_)
    # final micro-scan at ulp scale
    best = (eps_val(a, W, C), a)
    for k in range(-4000, 4001):
        aa = a + k * 1e-16
        v = eps_val(aa, W, C)
        if v > best[0]:
            best = (v, aa)
    return best


results = []
for cl in merged:
    a0 = (math.pi / 4) * (sum(cl) / len(cl)) / N
    v, a = refine(a0, 2 * step)
    results.append((v, a))
results.sort(reverse=True)
print(f"  {'eps':>20} {'alpha (deg)':>16}   minimizers (ties within 1e-12)")
for v, a in results:
    b, rows = eps_dir(a, W, Cpru, ret_all=True)
    ties = [(r[1], r[3], r[2]) for r in rows if r[4] and abs(r[0] - v) < 1e-12]
    print(f"  {v:20.16f} {a*DEG:16.10f}   " +
          ", ".join(f"w={w} |w|={L:.5f} d0={d:.5f}" for w, L, d in ties[:4]))

GLOBAL = results[0]
print()
print(f"  GLOBAL MAX over [0,45deg] = {GLOBAL[0]:.16f} at alpha = {GLOBAL[1]*DEG:.12f} deg")
print(f"  claim                     = {CLAIM:.16f}")
print(f"  difference (found-claim)  = {GLOBAL[0]-CLAIM:+.3e}")
print(f"  EXCEEDS CLAIM? {GLOBAL[0] > CLAIM + 1e-15}")
