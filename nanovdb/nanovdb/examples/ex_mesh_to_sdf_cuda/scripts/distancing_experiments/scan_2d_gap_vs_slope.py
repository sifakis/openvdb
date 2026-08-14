#!/usr/bin/env python3
"""
2D: exact eps*(theta, W) via direct lattice-vector search (no simulation needed).

Derivation (see NewDistancingApproach.md, "Decision: restrict to 2D"):
  For a barrier point p and candidate lattice vector w=(a,b) (V=p+w), the capture
  condition |p-V| < UDF(V) reduces to
        eps > |w| - w.n
  where n=(cos theta, sin theta) is the unit normal to Sigma. So
        eps*(theta, W) = min over w in Z^2 \\ {0}, 0 < w.n <= W  of  (|w| - w.n)
  This is always >= 0 (Cauchy-Schwarz), and exactly 0 iff some integer multiple of a
  primitive lattice vector exactly parallel to n has length <= W.

This does NOT require running the CC/enrichment simulation: for a single straight line
interface, any sufficiently deep non-barrier lattice point is automatically certified,
so the direct vector search already answers the question.
"""

import math

W = 3.0
BARRIER_2D = math.sqrt(2) / 2.0  # 2D analogue of sqrt(3)/2: half-diagonal of a unit square

# search radius for candidate vectors -- must be >= W but a bit of slack is free (cheap)
SEARCH_RADIUS = int(math.ceil(W)) + 2


def eps_star(theta, W):
    n = (math.cos(theta), math.sin(theta))
    best = None
    for a in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for b in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            if a == 0 and b == 0:
                continue
            wn = a * n[0] + b * n[1]
            if wn <= BARRIER_2D or wn > W:
                continue  # must land on a non-barrier, in-band point
            wlen = math.sqrt(a * a + b * b)
            val = wlen - wn
            if best is None or val < best:
                best = val
    return best


def rational_slope_str(theta, max_den=40):
    # try to recognize theta as arctan(p/q) for small integers, for readability
    t = math.tan(theta)
    best = None
    for q in range(1, max_den + 1):
        p = round(t * q)
        if p < 0 or q == 0:
            continue
        err = abs(t - p / q)
        if err < 1e-9:
            g = math.gcd(p, q)
            return f"arctan({p//g}/{q//g})"
    return None


def main():
    N = 900
    results = []
    for i in range(N + 1):
        theta = (math.pi / 2) * i / N  # 0 .. 90 degrees
        if i == 0:
            theta = 1e-6
        e = eps_star(theta, W)
        results.append((theta, e))

    results.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
    print("Worst (largest-eps) directions found in the scan:")
    for theta, e in results[:15]:
        deg = math.degrees(theta)
        label = rational_slope_str(theta) or ""
        print(f"  theta={deg:7.3f} deg   eps*={e:.6f}   {label}")

    zeros = [r for r in results if r[1] is not None and r[1] < 1e-9]
    print(f"\n{len(zeros)}/{len(results)} sampled directions hit exactly eps*=0")
    print("A few exact-zero directions found:")
    zeros.sort(key=lambda r: r[0])
    shown = 0
    for theta, e in zeros:
        deg = math.degrees(theta)
        label = rational_slope_str(theta) or "?"
        print(f"    theta={deg:7.3f} deg   {label}")
        shown += 1
        if shown >= 12:
            break

    # specifically check theta = arctan(phi) (golden ratio) and arctan(1/phi)
    phi = (1 + math.sqrt(5)) / 2
    for label, t in [("arctan(phi)", math.atan(phi)), ("arctan(1/phi)", math.atan(1 / phi))]:
        e = eps_star(t, W)
        print(f"\n{label} = {math.degrees(t):.4f} deg -> eps*={e:.6f}")

    worst = max(r[1] for r in results if r[1] is not None)
    print(f"\nmax eps* found over the scan: {worst:.6f}")
    print("compare: 3D rigorous universal bound was 0.19325 (different dimension, not "
          "directly comparable, but same order of magnitude check)")


if __name__ == "__main__":
    main()
