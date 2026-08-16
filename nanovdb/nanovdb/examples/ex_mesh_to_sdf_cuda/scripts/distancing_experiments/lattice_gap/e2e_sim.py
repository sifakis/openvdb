"""Independent end-to-end 2D lattice simulation of the certification closure.

NO use of the reduced formula g(w,n) = |w| sin^2(theta/2) anywhere.
Everything is built from an actual lattice + an actual straight interface.

Model
-----
Lattice Z^2 (unit spacing) restricted to [-L,L]^2.
Interface Sigma = { x : x . n + c = 0 }, n a unit vector.
Signed value s(p) = p.n + c ; exterior side is s > 0.
True UDF(p) = dist(p, Sigma) = |s(p)|   (Sigma is a full straight line, exact).

barrier(p)  <=>  UDF(p) <= r_d = sqrt(2)/2
in-band(p)  <=>  UDF(p) <= W

Certification:
  step 1 (connected components): 4-connected components of the non-barrier in-band
         points; a component containing an exterior point is certified wholesale.
  step 2 (enrichment closure): RAW pairwise rule to fixpoint:
         A certified, B in-band, UDF(A) + UDF(B) > dist(A,B)  ==>  B certified.

Numerics
--------
The strict ">" has genuine EXACT TIES (e.g. n parallel to an integer vector and
B-A parallel to n across the interface).  A tolerance `tol` makes the predicate
conservative:  certify only if (uA+uB) - dist > tol.  tol=0 reproduces naive
double arithmetic (and is then vulnerable to those ties).  Near-tie events with
|margin| <= tie_watch are counted so we can prove the answer does not depend on tol.

The arithmetic backend is pluggable: pass Decimal nx,ny,c plus a Decimal sqrt for
an arbitrary-precision run.
"""

import math

RD_F = math.sqrt(2.0) / 2.0        # lattice covering radius in 2D (float)


class Sim:
    """One trial.  nx,ny,c,W,rd may be float or Decimal (consistently)."""

    def __init__(self, nx, ny, c, L=12, W=3.0, rd=None, sqrtfn=math.sqrt,
                 tol=1e-12, tie_watch=1e-9):
        self.L, self.W, self.c = L, W, c
        self.nx, self.ny = nx, ny
        self.tol, self.tie_watch = tol, tie_watch
        self.sqrtfn = sqrtfn
        self.rd = rd if rd is not None else RD_F
        zero = nx * 0
        pts, s, u, idx = [], [], [], {}
        for x in range(-L, L + 1):
            xn = x * nx
            for y in range(-L, L + 1):
                sv = xn + y * ny + c
                uv = -sv if sv < zero else sv
                if uv <= W:
                    idx[(x, y)] = len(pts)
                    pts.append((x, y))
                    s.append(sv)
                    u.append(uv)
        self.pts, self.s, self.u, self.idx = pts, s, u, idx
        self.N = len(pts)
        self.zero = zero
        self._dcache = {}
        self.n_ties = 0
        self.ncc = None
        self.cc_full_match = None
        self.cert = None

    def dist(self, dx, dy):
        k = (dx if dx >= 0 else -dx, dy if dy >= 0 else -dy)
        d = self._dcache.get(k)
        if d is None:
            d = self.sqrtfn(self.zero + (k[0] * k[0] + k[1] * k[1]))
            self._dcache[k] = d
        return d

    # ---------------- step 1: connected components ----------------
    def connected_components(self):
        s, u, idx, pts, N = self.s, self.u, self.idx, self.pts, self.N
        z, rd = self.zero, self.rd
        free = [ui > rd for ui in u]
        comp = [-1] * N
        cert = [False] * N
        n_ext_comp = 0
        for start in range(N):
            if not free[start] or comp[start] != -1:
                continue
            stack, members = [start], []
            comp[start] = 0
            while stack:
                i = stack.pop()
                members.append(i)
                x, y = pts[i]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = idx.get((x + dx, y + dy))
                    if j is not None and free[j] and comp[j] == -1:
                        comp[j] = 0
                        stack.append(j)
            pos = any(s[m] > z for m in members)
            neg = any(s[m] < z for m in members)
            if pos and neg:
                raise AssertionError("CC step leaked across the interface")
            if pos:
                n_ext_comp += 1
                for m in members:
                    cert[m] = True
        self.ncc = n_ext_comp
        # diagnostic: single flood fill from the deepest exterior non-barrier point
        cand = [i for i in range(N) if free[i] and s[i] > z]
        if cand:
            best = max(cand, key=lambda i: s[i])
            seen, stack = {best}, [best]
            while stack:
                i = stack.pop()
                x, y = pts[i]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = idx.get((x + dx, y + dy))
                    if j is not None and free[j] and j not in seen:
                        seen.add(j)
                        stack.append(j)
            self.cc_full_match = (len(seen) == len(cand))
        return cert

    # ---------------- step 2: enrichment closure ----------------
    def enrich(self, cert, allow_barrier_witness=True, provenance=False):
        pts, u, N, rd = self.pts, self.u, self.N, self.rd
        tol, tw = self.tol, self.tie_watch
        par = {} if provenance else None
        while True:
            changed = False
            for j in range(N):
                if cert[j]:
                    continue
                xj, yj = pts[j]
                uj = u[j]
                for i in range(N):
                    if not cert[i]:
                        continue
                    ui = u[i]
                    if not allow_barrier_witness and ui <= rd:
                        continue
                    m = ui + uj - self.dist(pts[i][0] - xj, pts[i][1] - yj)
                    if -tw < m < tw:
                        self.n_ties += 1
                    if m > tol:
                        cert[j] = True
                        changed = True
                        if provenance:
                            par[j] = (i, m)
                        break
            if not changed:
                return (cert, par) if provenance else cert

    # ---------------- driver ----------------
    def run(self, allow_barrier_witness=True):
        cert = self.connected_components()
        cert = self.enrich(cert, allow_barrier_witness)
        z = self.zero
        for i in range(self.N):
            if cert[i] and self.s[i] <= z:
                raise AssertionError(
                    "SOUNDNESS VIOLATION: certified point %s has s=%.17g"
                    % (self.pts[i], float(self.s[i])))
        self.cert = cert
        return cert

    def max_uncertified_depth(self, margin):
        """Max UDF over EXTERIOR in-band lattice points that stayed uncertified,
        restricted to |x|,|y| <= L-margin so box truncation cannot contaminate."""
        lim, z = self.L - margin, self.zero
        best, arg = -1.0, None
        for i in range(self.N):
            if self.cert[i] or self.s[i] <= z:
                continue
            x, y = self.pts[i]
            if abs(x) <= lim and abs(y) <= lim and self.u[i] > best:
                best, arg = self.u[i], (x, y)
        return best, arg


def make(alpha, c, L=12, W=3.0, tol=1e-12, tie_watch=1e-9):
    """float backend, alpha in radians."""
    return Sim(math.cos(alpha), math.sin(alpha), c, L=L, W=W, rd=RD_F,
               sqrtfn=math.sqrt, tol=tol, tie_watch=tie_watch)


def origin_certified(alpha, c, L=12, W=3.0, tol=1e-12, allow_barrier_witness=True):
    sim = make(alpha, c, L=L, W=W, tol=tol)
    cert = sim.run(allow_barrier_witness)
    return cert[sim.idx[(0, 0)]]
