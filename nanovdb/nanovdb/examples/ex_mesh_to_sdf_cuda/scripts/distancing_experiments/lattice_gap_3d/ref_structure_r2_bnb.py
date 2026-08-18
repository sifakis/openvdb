"""R2: my own rigorous-modulo-double branch & bound on S^2 (fundamental domain).

Patch = spherical triangle spanned by unit vertices v1,v2,v3.
For n in the patch, n = u/|u| with u in the PLANAR triangle conv{v1,v2,v3}.
 - w.u is linear -> extremes at vertices: [mlo, mhi]
 - |u| in [rho, 1] with rho = |v1 . N| (N unit normal of the plane) : safe lower bound
 => w.n in [min(mlo, mlo/rho), max(mhi, mhi/rho)]
Upper bound on the envelope over the patch:
  THRESHOLD: min over w that are GUARANTEED admissible over the whole patch of gmax(w).
  EXACT:     sup of (0,rd] minus union of GUARANTEED windows
             ( max(gmax, rd - dotlo), min(rd, W - dothi) ]
Both are valid upper bounds (dropping witnesses / shrinking windows can only raise them).
"""
import math, heapq, sys
sys.path.insert(0, '/tmp/dist/d3/ref_structure')
from rlib import enum_w, primitives, F_thr, F_exa, RD


def norm(v):
    L = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (v[0]/L, v[1]/L, v[2]/L)


def patch_data(tri):
    v1, v2, v3 = tri
    e1 = (v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])
    e2 = (v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2])
    N = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0])
    Ln = math.sqrt(N[0]**2+N[1]**2+N[2]**2)
    if Ln == 0.0:
        return 0.0
    N = (N[0]/Ln, N[1]/Ln, N[2]/Ln)
    rho = abs(v1[0]*N[0]+v1[1]*N[1]+v1[2]*N[2])
    return rho


def dot_range(w, tri, rho):
    a, b, c = w[0], w[1], w[2]
    ds = [a*v[0]+b*v[1]+c*v[2] for v in tri]
    mlo, mhi = min(ds), max(ds)
    lo = min(mlo, mlo/rho) if rho > 0 else -1e18
    hi = max(mhi, mhi/rho) if rho > 0 else 1e18
    return lo, hi


def ub_thr(tri, ws, W, rd=RD):
    rho = patch_data(tri)
    if rho <= 0:
        return rd
    best = rd
    for w in ws:
        lo, hi = dot_range(w, tri, rho)
        L = w[3]
        gmax = (L - lo) / 2.0
        if gmax >= best:
            continue
        d0lo, d0hi = (L + lo) / 2.0, (L + hi) / 2.0
        if d0lo > rd and d0hi <= W:       # guaranteed admissible on the whole patch
            best = gmax
    return best


def ub_exa(tri, ws, W, rd=RD):
    rho = patch_data(tri)
    if rho <= 0:
        return rd
    iv = []
    for w in ws:
        lo, hi = dot_range(w, tri, rho)
        L = w[3]
        gmax = (L - lo) / 2.0
        wlo = max(gmax, rd - lo)
        whi = min(rd, W - hi)
        if whi > wlo:
            iv.append((max(wlo, 0.0), whi))
    if not iv:
        return rd
    iv.sort()
    merged = []
    for lo, hi in iv:
        if merged and lo <= merged[-1][1]:
            if hi > merged[-1][1]:
                merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    top = rd
    for lo, hi in reversed(merged):
        if hi >= top:
            top = lo
            if top <= 0.0:
                return 0.0
        else:
            return top
    return top


def centroid(tri):
    return norm(tuple(sum(v[k] for v in tri)/3.0 for k in range(3)))


def split(tri):
    # bisect longest chordal edge
    best, bi = -1, 0
    for i in range(3):
        j = (i+1) % 3
        d = sum((tri[i][k]-tri[j][k])**2 for k in range(3))
        if d > best:
            best, bi = d, i
    i, j = bi, (bi+1) % 3
    k = 3 - i - j
    m = norm(tuple((tri[i][t]+tri[j][t])/2.0 for t in range(3)))
    return [(tri[i], m, tri[k]), (m, tri[j], tri[k])]


def bnb(ws, W, rule, tol=1e-12, maxpatch=4_000_000, seedLB=0.0, verbose=True):
    """Returns (LB, UB_certified, npatch, argmax)."""
    root = (norm((1.0, 0.0, 0.0)), norm((1.0, 1.0, 0.0)), norm((1.0, 1.0, 1.0)))
    ubf = ub_thr if rule == 'thr' else ub_exa
    evf = (lambda n: F_thr(n, ws, W)) if rule == 'thr' else (lambda n: F_exa(n, ws, W))
    LB, arg = seedLB, None
    c = centroid(root)
    v = evf(c)
    if v > LB:
        LB, arg = v, c
    heap = [(-ubf(root, ws, W), 0, root)]
    cnt = 1
    npatch = 0
    while heap:
        negu, _, tri = heapq.heappop(heap)
        u = -negu
        if u <= LB + tol:
            # everything remaining has ub <= u <= LB+tol
            return LB, LB + tol, npatch, arg, u
        npatch += 1
        if npatch > maxpatch:
            return LB, u, npatch, arg, u
        for t in split(tri):
            cc = centroid(t)
            vv = evf(cc)
            if vv > LB:
                LB, arg = vv, cc
            uu = ubf(t, ws, W)
            if uu > LB + tol:
                cnt += 1
                heapq.heappush(heap, (-uu, cnt, t))
    return LB, LB, npatch, arg, LB


if __name__ == '__main__':
    W = 3.0
    print("=" * 78)
    ws9 = enum_w(3.0)            # |w|^2 <= 9  (122 vectors)
    ws14 = enum_w(math.sqrt(14)) # |w|^2 <= 14 (250)
    ws3 = enum_w(math.sqrt(3))   # 3x3x3 neighbourhood, 26
    print("witness counts:", len(ws9), len(ws14), len(ws3))

    print("\n-- G0 = max_n min_{3x3x3} g  (unfiltered) --")
    # unfiltered: use ub with no admissibility filter
    def ub_plain(tri, ws, W, rd=RD):
        rho = patch_data(tri)
        if rho <= 0: return rd
        best = rd
        for w in ws:
            lo, hi = dot_range(w, tri, rho)
            gmax = (w[3]-lo)/2.0
            if gmax < best: best = gmax
        return best
    import rlib
    # temporary monkeypatch route: inline b&b for plain
    def bnb_plain(ws, tol=1e-13):
        root = (norm((1.,0.,0.)), norm((1.,1.,0.)), norm((1.,1.,1.)))
        evf = lambda n: min((w[3]-(w[0]*n[0]+w[1]*n[1]+w[2]*n[2]))/2.0 for w in ws)
        LB, arg = 0.0, None
        heap = [(-ub_plain(root, ws, W), 0, root)]; cnt=1; np_=0
        while heap:
            negu,_,tri = heapq.heappop(heap); u=-negu
            if u <= LB+tol: return LB, u, np_, arg
            np_ += 1
            if np_ > 3_000_000: return LB, u, np_, arg
            for t in split(tri):
                cc = centroid(t); vv = evf(cc)
                if vv > LB: LB, arg = vv, cc
                uu = ub_plain(t, ws, W)
                if uu > LB+tol:
                    cnt+=1; heapq.heappush(heap,(-uu,cnt,t))
        return LB, LB, np_, arg
    lb, ub, npp, arg = bnb_plain(ws3)
    print(f"  LB={lb:.15f}  certified UB={ub:.15f}  patches={npp}")
    print(f"  argmax {arg}")
    print(f"  claimed closed form 0.073559321149897130")

    print("\n-- threshold rule, |w|^2<=9, W=3 --")
    lb, ub, npp, arg, lastu = bnb(ws9, W, 'thr', tol=1e-12)
    print(f"  LB={lb:.16f}  certified UB={ub:.16f}  patches={npp}")
    print(f"  argmax {arg}")
    print(f"  claim   0.0366622651188296423   diff LB-claim = {lb-0.0366622651188296423:.3e}")

    print("\n-- exact rule, |w|^2<=14, W=3 --")
    lb, ub, npp, arg, lastu = bnb(ws14, W, 'exa', tol=1e-12)
    print(f"  LB={lb:.16f}  certified UB={ub:.16f}  patches={npp}")
    print(f"  argmax {arg}")
    print(f"  claim   0.0384434235747196266   diff LB-claim = {lb-0.0384434235747196266:.3e}")
