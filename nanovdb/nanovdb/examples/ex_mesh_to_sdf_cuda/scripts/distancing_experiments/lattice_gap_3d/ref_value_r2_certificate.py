"""R2: REFEREE'S OWN rigorous certificate for
      V(S, t) := ( exists unit n in the closed cone  n1>=n2>=n3>=0  with
                   g(w,n) >= t for all w in the B_3-closure of S )

Method (independent of anything in the work under review):
  * Rearrangement lemma: for x in the cone, max over the B_3-orbit of w of w.x is
    attained at sorted-descending |w| components.  Hence the orbit constraint set
    collapses to the single sorted representative.
  * P(t) = { x : cone constraints, w.x <= |w| - 2t for each sorted rep w }.
    P(t) is a bounded polytope (x1 <= |w_(1,0,0)| - 2t = 1-2t, 0 <= x3 <= x2 <= x1)
    containing 0 (for t <= 1/2).
  * |x|^2 is convex, so max over P(t) is attained at a VERTEX.  P(t) meets the unit
    sphere  <=>  max_{P} |x|^2 >= 1  (using 0 in P and continuity of |.| along a segment).
  * Vertices = solutions of 3 linearly independent tight constraints.  We enumerate all
    C(m,3) triples, solve by Cramer with EXACT integer determinants and rational-interval
    right-hand sides, discard triples certainly infeasible, and bound |v|^2 above.

If every triple is certainly infeasible or has |v|^2 upper bound < 1, then V(S,t) is FALSE,
hence max_n min_w g(w,n) < t.  This is a proof modulo the correctness of this program.
"""
import sys, itertools
sys.path.insert(0, '/tmp/dist/d3/ref_value')
from ivl import I, SQRT, mk
from fractions import Fraction as F

CONE = [((-1, 1, 0), 0), ((0, -1, 1), 0), ((0, 0, -1), 0)]

def dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def det3i(a, b, c):
    """exact integer determinant of rows a,b,c"""
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
          - a[1]*(b[0]*c[2]-b[2]*c[0])
          + a[2]*(b[0]*c[1]-b[1]*c[0]))

def solve3(rows, rhs, det):
    """Cramer: rows integer 3-tuples, rhs list of intervals, det nonzero int."""
    out = []
    for k in range(3):
        m = [list(r) for r in rows]
        # replace column k by rhs
        # determinant expansion with interval entries in column k
        # compute cofactor expansion along column k
        s = I(0)
        for i in range(3):
            # cofactor C_ik
            rr = [m[j] for j in range(3) if j != i]
            cc = [q for q in range(3) if q != k]
            minor = rr[0][cc[0]]*rr[1][cc[1]] - rr[0][cc[1]]*rr[1][cc[0]]
            sign = (-1)**(i+k)
            s = s + rhs[i]*(sign*minor)
        out.append(s / I(det))
    return out

def certificate(sorted_reps, t, verbose=False):
    """sorted_reps: list of (w tuple sorted desc, |w|^2 int).  t: Fraction.
    Returns (ok, info) where ok=True means NO unit vector in the cone attains
    min_w g >= t, i.e. max_n min_w g(w,n) < t."""
    cons = list(CONE)
    for w, n2 in sorted_reps:
        cons.append((w, ('sqrt', n2)))
    m = len(cons)
    rhss = []
    for w, r in cons:
        if r == 0:
            rhss.append(I(0))
        else:
            rhss.append(SQRT(r[1]) - I(2*F(t)))
    worst = None
    nfeas = 0; ninfeas = 0; ndeg = 0; nundet = 0
    survivors = []
    for tri in itertools.combinations(range(m), 3):
        rows = [cons[i][0] for i in tri]
        d = det3i(*rows)
        if d == 0:
            ndeg += 1; continue
        v = solve3(rows, [rhss[i] for i in tri], d)
        # feasibility
        infeas = False
        undet = False
        for j in range(m):
            wj = cons[j][0]
            val = v[0]*wj[0] + v[1]*wj[1] + v[2]*wj[2] - rhss[j]
            if val.certainly_pos():
                infeas = True; break
            if not val.certainly_neg() and not (val.hi <= 0):
                undet = True
        if infeas:
            ninfeas += 1; continue
        nfeas += 1
        if undet: nundet += 1
        n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
        survivors.append((tri, n2))
        if worst is None or n2.hi > worst[1].hi:
            worst = (tri, n2)
    ok = all(s[1].hi < 1 for s in survivors)
    return ok, dict(ndeg=ndeg, ninfeas=ninfeas, nfeas=nfeas, nundet=nundet,
                    worst=worst, survivors=survivors)

# ---------------------------------------------------------------------------
THR_REPS = [((1,0,0),1), ((1,1,0),2), ((1,1,1),3), ((2,1,0),5), ((2,1,1),6), ((2,2,1),9)]
EXA_REPS = [((1,0,0),1), ((1,1,0),2), ((1,1,1),3), ((2,1,0),5), ((2,1,1),6)]

def bisect(reps, lo, hi, iters=60, label=""):
    """lo: known achievable (cert fails), hi: known not achievable (cert ok)."""
    ok_hi, _ = certificate(reps, hi)
    ok_lo, _ = certificate(reps, lo)
    print("  %s: cert(t=%s) ok=%s ; cert(t=%s) ok=%s" %
          (label, float(lo), ok_lo, float(hi), ok_hi))
    assert ok_hi and not ok_lo, "bracket wrong"
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok, _ = certificate(reps, mid)
        if ok: hi = mid
        else:  lo = mid
    return lo, hi

if __name__ == '__main__':
    import time
    E_THR = F('0.03666226511882964229425993304303098231900508153686')
    E_EXA = F('0.03844342357471962656641226957329293723650910457108')
    print("=== THRESHOLD RULE, witness set |w|^2 <= 9 (unfiltered) ===")
    for d in ['1e-12', '1e-18', '1e-25', '1e-40']:
        t = E_THR + F(d)
        ok, info = certificate(THR_REPS, t)
        print("  t = eps_thr + %s : NO unit vector attains it? %s  (feas vertices %d, infeas %d, deg %d, undet %d)"
              % (d, ok, info['nfeas'], info['ninfeas'], info['ndeg'], info['nundet']))
        if not ok:
            print("     worst |v|^2 =", info['worst'][1], "at rows", info['worst'][0])
    for d in ['1e-12', '1e-18', '1e-25', '1e-40']:
        t = E_THR - F(d)
        ok, info = certificate(THR_REPS, t)
        print("  t = eps_thr - %s : NO unit vector attains it? %s   (expect False)" % (d, ok))
        if not ok:
            w = info['worst']
            print("     max |v|^2 over feasible vertices =", w[1], " rows:", [THR_REPS[i-3][0] if i>=3 else CONE[i][0] for i in w[0]])
    print()
    print("=== EXACT-RULE reduced set A(n), |w|^2 <= 6 ===")
    for d in ['1e-12', '1e-18', '1e-25', '1e-40']:
        t = E_EXA + F(d)
        ok, info = certificate(EXA_REPS, t)
        print("  t = eps_exa + %s : NO unit vector attains it? %s  (feas %d, infeas %d)"
              % (d, ok, info['nfeas'], info['ninfeas']))
    for d in ['1e-12', '1e-18', '1e-25', '1e-40']:
        t = E_EXA - F(d)
        ok, info = certificate(EXA_REPS, t)
        print("  t = eps_exa - %s : NO unit vector attains it? %s   (expect False)" % (d, ok))
