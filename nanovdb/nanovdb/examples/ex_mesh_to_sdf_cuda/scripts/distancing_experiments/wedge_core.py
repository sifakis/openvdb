#!/usr/bin/env python3
"""
Core construction for a concave-wedge interface: Sigma = two rays sharing a vertex,
exterior = the open angular sector of angle alpha between them.

Reuses the exact same barrier / CC-seed / pairwise-enrichment-closure machinery already
validated on the straight-line case (2D, barrier radius sqrt(2)/2, 4-connectivity,
guarded strict predicate per LatticeCertificationGap.md section 4.6).
"""

import math
from collections import deque

BOX = 8
W = 3.0
BARRIER_R = math.sqrt(2) / 2.0
TOL = 1e-9


def dist_to_ray(p, vertex, u):
    # u must be a unit vector; ray = {vertex + t*u : t >= 0}
    wx, wy = p[0] - vertex[0], p[1] - vertex[1]
    t = wx * u[0] + wy * u[1]
    if t <= 0:
        return math.hypot(wx, wy)
    fx, fy = vertex[0] + t * u[0], vertex[1] + t * u[1]
    return math.hypot(p[0] - fx, p[1] - fy)


class Wedge:
    def __init__(self, vertex, theta1, alpha):
        self.vertex = vertex
        self.theta1 = theta1
        self.alpha = alpha
        self.theta2 = theta1 + alpha
        self.u1 = (math.cos(theta1), math.sin(theta1))
        self.u2 = (math.cos(self.theta2), math.sin(self.theta2))

    def udf(self, p):
        return min(dist_to_ray(p, self.vertex, self.u1),
                    dist_to_ray(p, self.vertex, self.u2))

    def is_ext(self, p):
        dx, dy = p[0] - self.vertex[0], p[1] - self.vertex[1]
        if dx == 0 and dy == 0:
            return False  # degenerate: exactly the vertex
        ang = math.atan2(dy, dx)
        rel = (ang - self.theta1) % (2 * math.pi)
        return 0 < rel < self.alpha


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_grid(box=BOX):
    return [(x, y) for x in range(-box, box + 1) for y in range(-box, box + 1)]


def run_construction(wedge, box=BOX, w=W):
    grid = build_grid(box)
    d = {p: wedge.udf(p) for p in grid}
    is_ext = {p: wedge.is_ext(p) for p in grid}
    in_band = {p: d[p] <= w for p in grid}
    barrier = {p: d[p] <= BARRIER_R for p in grid}
    non_barrier_in_band = {p: in_band[p] and not barrier[p] for p in grid}

    seeds = [p for p in grid if non_barrier_in_band[p] and is_ext[p]]
    if not seeds:
        return {
            "grid": grid, "d": d, "is_ext": is_ext, "in_band": in_band,
            "barrier": barrier, "certified": set(), "seed": None,
            "no_seed_at_all": True,
        }
    seed = seeds[0]

    certified = {seed}
    q = deque([seed])
    while q:
        cx, cy = q.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            np_ = (cx + dx, cy + dy)
            if np_ in non_barrier_in_band and non_barrier_in_band[np_] and np_ not in certified:
                certified.add(np_)
                q.append(np_)

    for p in certified:
        assert is_ext[p], f"CC step certified an interior point at {p}!"

    eligible = set(p for p in grid if in_band[p] and p not in certified)
    changed = True
    while changed:
        changed = False
        newly = []
        for p in eligible:
            dp = d[p]
            for c in certified:
                if d[c] + dp - dist(c, p) > TOL:
                    newly.append(p)
                    break
        if newly:
            changed = True
            for p in newly:
                certified.add(p)
                eligible.discard(p)

    for p in certified:
        assert is_ext[p], f"Enrichment closure certified an interior point at {p}!"

    return {
        "grid": grid, "d": d, "is_ext": is_ext, "in_band": in_band,
        "barrier": barrier, "certified": certified, "seed": seed,
        "no_seed_at_all": False,
    }
