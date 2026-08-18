#!/usr/bin/env python3
"""
Render a wedge (concave corner) construction to SVG, same visual language as the
straight-line figures: black boundary (here, two rays), translucent green balls =
O_outside, colored lattice points by classification.
"""

import math
from wedge_core import Wedge, run_construction, dist


def render(wedge, view_radius, scale, out_path, title, box=8, w=3.0):
    res = run_construction(wedge, box=box, w=w)
    grid, d, is_ext, in_band, barrier, certified = (
        res["grid"], res["d"], res["is_ext"], res["in_band"], res["barrier"], res["certified"])
    misses = [p for p in grid if in_band[p] and is_ext[p] and p not in certified]
    print(f"{title}: certified={len(certified)}, misses={len(misses)}, "
          f"no_seed_at_all={res['no_seed_at_all']}")
    if misses:
        worst = max(misses, key=lambda p: d[p])
        print(f"   deepest miss: {worst}, UDF={d[worst]:.4f}, "
              f"dist_to_vertex={dist(worst, wedge.vertex):.4f}")

    pad = 40
    w_px = int(2 * view_radius * scale + 2 * pad)
    h_px = w_px

    def to_svg(p):
        x, y = p
        return (pad + (x + view_radius) * scale, pad + (view_radius - y) * scale)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_px}" height="{h_px}" '
           f'viewBox="0 0 {w_px} {h_px}" font-family="Helvetica,Arial,sans-serif">']
    svg.append(f'<rect x="0" y="0" width="{w_px}" height="{h_px}" fill="white"/>')

    for k in range(-box, box + 1):
        x0, y0 = to_svg((k, -box * 2))
        x1, y1 = to_svg((k, box * 2))
        svg.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                    f'stroke="#f0f0f0" stroke-width="1"/>')
        x0, y0 = to_svg((-box * 2, k))
        x1, y1 = to_svg((box * 2, k))
        svg.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                    f'stroke="#f0f0f0" stroke-width="1"/>')

    for p in sorted(certified):
        cx, cy = to_svg(p)
        r = d[p] * scale
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
                    f'fill="#2e7d32" fill-opacity="0.10" stroke="#2e7d32" '
                    f'stroke-width="0.6" stroke-opacity="0.35"/>')

    # the two rays, drawn out to the view boundary
    L = 3 * view_radius
    for u in (wedge.u1, wedge.u2):
        a = wedge.vertex
        b = (wedge.vertex[0] + L * u[0], wedge.vertex[1] + L * u[1])
        ax, ay = to_svg(a)
        bx, by = to_svg(b)
        svg.append(f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" '
                    f'stroke="black" stroke-width="2.5"/>')

    # small arc marking the wedge (exterior) angular sector, near the vertex
    r_arc = 0.6
    n_steps = 24
    pts = []
    for k in range(n_steps + 1):
        a = wedge.theta1 + wedge.alpha * k / n_steps
        pts.append((wedge.vertex[0] + r_arc * math.cos(a),
                    wedge.vertex[1] + r_arc * math.sin(a)))
    path = " ".join(f"{to_svg(pt)[0]:.2f},{to_svg(pt)[1]:.2f}" for pt in pts)
    svg.append(f'<polyline points="{path}" fill="none" stroke="#1565c0" '
                f'stroke-width="1.5" stroke-dasharray="3,2"/>')

    for p in grid:
        if abs(p[0] - wedge.vertex[0]) > view_radius + 1 or \
           abs(p[1] - wedge.vertex[1]) > view_radius + 1:
            continue
        cx, cy = to_svg(p)
        is_miss = in_band[p] and is_ext[p] and p not in certified
        if p in certified:
            color, r = "#2e7d32", 3.2
        elif is_miss:
            color, r = "#c62828", 4.0
        elif barrier[p]:
            color, r = "#f9a825", 2.6
        elif is_ext[p]:
            color, r = "#bbbbbb", 2.0
        else:
            color, r = "#9e9e9e", 2.0
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{color}"/>')

    if res["seed"] is not None:
        sx, sy = to_svg(res["seed"])
        svg.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="6" fill="none" '
                    f'stroke="black" stroke-width="1.4"/>')

    vx, vy = to_svg(wedge.vertex)
    svg.append(f'<circle cx="{vx:.2f}" cy="{vy:.2f}" r="3" fill="#1565c0"/>')

    lx, ly = w_px - 270, 24
    items = [
        ("#2e7d32", "certified exterior (+ its ball)"),
        ("#c62828", "truly exterior, uncertified (miss)"),
        ("#f9a825", "barrier, interior side"),
        ("#9e9e9e", "interior"),
        ("#bbbbbb", "exterior, out of band"),
        ("#1565c0", "wedge vertex"),
    ]
    svg.append(f'<rect x="{lx-12}" y="{ly-16}" width="270" height="{22*len(items)+14}" '
                f'fill="white" fill-opacity="0.88" stroke="#cccccc"/>')
    for i, (color, label) in enumerate(items):
        yy = ly + i * 22
        svg.append(f'<circle cx="{lx}" cy="{yy}" r="4.5" fill="{color}"/>')
        svg.append(f'<text x="{lx+14}" y="{yy+4}" font-size="13" fill="#222">{label}</text>')

    svg.append(f'<text x="{pad}" y="{h_px-14}" font-size="13" fill="#444">{title}</text>')
    svg.append('</svg>')
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"   wrote {out_path}")
    return res


OUT = "/home/esifakis/.claude/jobs/e26d81df/tmp/wedge_experiment"

# Case 1: mild wide-angle wedge, expect 0 misses
w1 = Wedge((0.45680825991006246, 0.14333946282650079), 2.1344350717006253,
           math.radians(116.03198354346237))
render(w1, view_radius=8.6, scale=48, out_path=f"{OUT}/wedge_mild.svg",
       title=f"mild wedge, alpha=116.0 deg -- typical case")

# Case 2: narrow angle with real misses
w2 = Wedge((-0.4264167507401717, 0.24577589394916322), 1.6096923058276726,
           math.radians(29.324346691685005))
render(w2, view_radius=8.6, scale=48, out_path=f"{OUT}/wedge_narrow_misses.svg",
       title=f"narrow wedge, alpha=29.3 deg -- 3 misses, deepest UDF~0.47")

# Case 3: degenerate, alpha=6 deg, expect NO seed at all
w3 = Wedge((0.2294452894392176, -0.21206223510981348), 6.1586202002323835,
           math.radians(6.0))
render(w3, view_radius=5.5, scale=75, out_path=f"{OUT}/wedge_degenerate.svg",
       title=f"degenerate wedge, alpha=6.0 deg -- NO eligible seed at all")
