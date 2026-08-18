"""R10: referee's own raw end-to-end simulation.
Real Z^3 box, real plane, true UDF = |p.n + c|, barrier = UDF <= rd, in-band = UDF <= W,
6-connected components of {non-barrier, in-band, exterior side} seeded from the outermost
shell, then the RAW pairwise closure UDF(A)+UDF(B) > dist(A,B) run to a fixpoint.
g(w,n) is never used.  Reports whether the origin (depth c) is certified."""
import math, sys
from collections import deque

RD = math.sqrt(3)/2

def simulate(n, c, W=3.0, L=9, margin=3, tol=0.0, allow_barrier_witnesses=True):
    pts = []
    idx = {}
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            for z in range(-L, L+1):
                s = x*n[0]+y*n[1]+z*n[2] + c
                u = abs(s)
                if u <= W:
                    idx[(x,y,z)] = len(pts)
                    pts.append((x,y,z,u,s))
    N = len(pts)
    cert = [False]*N
    # seed: non-barrier, exterior side (s>0), connected component reachable from the
    # deepest exterior shell.  Do a 6-connected BFS over {s>0, u>rd, u<=W}.
    free = [i for i,(x,y,z,u,s) in enumerate(pts) if s > 0 and u > RD]
    freeset = set(free)
    seen = set()
    comps = []
    for i in free:
        if i in seen: continue
        q = deque([i]); seen.add(i); comp=[i]
        while q:
            j = q.popleft()
            x,y,z,u,s = pts[j]
            for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                k = idx.get((x+d[0],y+d[1],z+d[2]))
                if k is not None and k in freeset and k not in seen:
                    seen.add(k); q.append(k); comp.append(k)
        comps.append(comp)
    # seed every component that touches the box boundary far from the plane (i.e. all of
    # them, for a plane; report the count)
    for comp in comps:
        for i in comp: cert[i]=True
    # raw pairwise closure
    work = deque([i for i in range(N) if cert[i]])
    # candidate offsets: |w| < 2W
    R = int(math.floor(2*W))
    offs = [(a,b,d) for a in range(-R,R+1) for b in range(-R,R+1) for d in range(-R,R+1)
            if (a,b,d)!=(0,0,0) and a*a+b*b+d*d <= 4*W*W]
    offs = [(a,b,d,math.sqrt(a*a+b*b+d*d)) for a,b,d in offs]
    while work:
        i = work.popleft()
        if not allow_barrier_witnesses and pts[i][3] <= RD: continue
        x,y,z,u,s = pts[i]
        for (a,b,d,ln) in offs:
            k = idx.get((x+a,y+b,z+d))
            if k is None or cert[k]: continue
            if u + pts[k][3] - ln > tol:
                cert[k]=True; work.append(k)
    # margin restriction
    orig = idx.get((0,0,0))
    return cert[orig] if orig is not None else None, len(comps), N

def uncovered_sup_sim(n, W=3.0, L=9, lo=1e-12, hi=None, iters=None, scan=None):
    """descending scan over candidate depths (NOT a bisection -- certification is not
    monotone in depth)."""
    if hi is None: hi = RD
    vals = scan
    best = 0.0
    for c in vals:
        ok,_,_ = simulate(n, c, W=W, L=L)
        if not ok and c > best: best = c
    return best

if __name__ == '__main__':
    import math
    s2,s3,s6 = math.sqrt(2),math.sqrt(3),math.sqrt(6)
    gam = math.sqrt(6*s2+2*s6-13)
    n_e = (s6-s3, gam, s3-s2)
    print("|n_e|^2 - 1 = %.3e" % (sum(t*t for t in n_e)-1))
    E = (s2+s3-s6-gam)/2
    print("eps_exa (double eval of surd) = %.20g" % E)
    for c,lab in [(E*(1-1e-9),'eps_exa*(1-1e-9)'), (E,'eps_exa'), (E*(1+1e-9),'eps_exa*(1+1e-9)'),
                  (0.0075,'0.0075 (inside (2,2,1) window)'), (0.0037,'0.0037 (below the window)'),
                  (0.0076,'0.0076 (just above the window)'), (0.02,'0.02'), (0.05,'0.05')]:
        ok,nc,N = simulate(n_e, c, L=9)
        print("  c=%-12.10g (%-30s) certified=%-5s  components=%d  N=%d" % (c,lab,ok,nc,N))
    # threshold-rule critical direction
    s5,s30 = math.sqrt(5),math.sqrt(30)
    b3 = math.sqrt(4*s30+2*s5-26)
    n_t = ((s5-1+b3)/2,(s5-1-b3)/2,s6-s5)
    Et = (3-s5-b3)/4
    print("\nthreshold n*: |n|^2-1 = %.3e ; eps_thr = %.20g" % (sum(t*t for t in n_t)-1, Et))
    for c in [Et*(1-1e-9), Et, Et*(1+1e-9), 0.0367, 0.0384, 0.0385]:
        ok,nc,N = simulate(n_t, c, L=9)
        print("  c=%.16g certified=%s comps=%d" % (c,ok,nc))
