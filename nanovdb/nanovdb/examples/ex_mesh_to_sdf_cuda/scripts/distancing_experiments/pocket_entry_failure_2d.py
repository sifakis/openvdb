#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""2D pocket-entry failure of a W + D stencil (figure in NewDistancingApproach.md, "Stencil reach
of the closure sweep").  The W + D bound assumes every exterior non-barrier voxel is certified
before the sweep; a non-barrier pocket that the seeding does not reach breaks that premise.  Here
the pocket B(q, rq) hangs off B(n, rn) through a sub-voxel window, and the only ball pair that
overlaps through it is (n, q), with |w| = sqrt34 ~ 5.83, close to 2W = 6.

Exterior = tunnel u B(n, rn) u B(q, rq); Sigma is its boundary.  Computes true UDFs (dense boundary
sampling), seeds with the 4-connected component of non-barrier points (udf >= D) at the tunnel end,
runs the ball-overlap closure to its fixpoint for stencils |w|^2 <= k, prints the outcome, and
writes the SVG.

Usage: python3 pocket_entry_failure_2d.py [out.svg]    (default: pocket_entry_failure_2d.svg)
The PNG next to the SVG in figures/ was rendered with headless Chrome:
  google-chrome --headless --screenshot=out.png --window-size=1190,828 file://$PWD/out.svg
"""
import math, itertools, sys
W=3.0; D=0.55
n=(0.0,0.0); q=(5.0,3.0); rn=2.99; rq=2.85; hc=1.3          # tunnel half-width
X0,X1,Y0,Y1=-7,8,-3,6                                        # domain; tunnel opens at x=X0 (the "outside")
xt=-math.sqrt(rn*rn-hc*hc)                                   # tunnel meets the big circle here
Lw=math.hypot(*q)
d=math.dist
def inU(p):
    return d(p,n)<rn or d(p,q)<rq or (p[0]<=xt+1e-12 and abs(p[1])<hc)
# dense sample of the boundary of U (the "surface" Sigma), bucketed for the distance queries
S=[]
N=200000
for i in range(N):
    t=2*math.pi*i/N; c,s_=math.cos(t),math.sin(t)
    p=(n[0]+rn*c,n[1]+rn*s_)
    if d(p,q)>=rq and not (p[0]<=xt and abs(p[1])<hc): S.append(p)
    p=(q[0]+rq*c,q[1]+rq*s_)
    if d(p,n)>=rn: S.append(p)
for i in range(20001):
    x=X0-3+(xt-(X0-3))*i/20000
    S+=[(x,hc),(x,-hc)]
hb=0.25; B={}
for p in S: B.setdefault((math.floor(p[0]/hb),math.floor(p[1]/hb)),[]).append(p)
def udf(x):
    best=float("inf"); c=(math.floor(x[0]/hb),math.floor(x[1]/hb))
    for rad in range(80):
        for o in itertools.product(range(-rad,rad+1),repeat=2):
            if max(abs(o[0]),abs(o[1]))!=rad: continue
            for p in B.get((c[0]+o[0],c[1]+o[1]),()): best=min(best,d(x,p))
        if best<rad*hb: return best
    return best
pts={}
for x in itertools.product(range(X0,X1+1),range(Y0,Y1+1)):
    if inU(x):
        u=udf(x)
        if u<=W: pts[x]=u
# seeding: 4-connected component of non-barrier points containing the tunnel end
start=min(p for p in pts if p[0]==X0)
seeds={start}; stack=[start]
while stack:
    a=stack.pop()
    for e in ((1,0),(-1,0),(0,1),(0,-1)):
        b=(a[0]+e[0],a[1]+e[1])
        if b in pts and pts[b]>=D and b not in seeds: seeds.add(b); stack.append(b)
def closure(k):
    cert=set(seeds); sweeps=0
    while True:
        sweeps+=1; new=set()
        for x,ux in pts.items():
            if x in cert: continue
            for y in cert:
                L2=(x[0]-y[0])**2+(x[1]-y[1])**2
                if L2<=k and ux+pts[y]>math.sqrt(L2)+1e-12: new.add(x); break
        if not new: return cert,sweeps
        cert|=new
Q=(5,3)
pocket={p for p in pts if d(p,q)<rq}
kWD=math.floor((W+D)**2)                                     # |w| < W + D  (|w|^2 <= 12, i.e. <= 10 on Z^2)
cS,sS=closure(kWD)
c32,s32=closure(32)
cF,sF=closure(34)
print("udf(n)=%.4f udf(q)=%.4f  rn+rq=%.4f  |w|=%.4f  (W+D=%.2f, 2W=%.0f)"%(pts[(0,0)],pts[Q],rn+rq,Lw,W+D,2*W))
print("%d exterior band points, %d seeds, %d in the pocket"%(len(pts),len(seeds),len(pocket)))
for k,(c,s) in ((kWD,(cS,sS)),(32,(c32,s32)),(34,(cF,sF))):
    print("|w|^2 <= %2d: fixpoint after %d sweeps, %d/%d certified, pocket certified %d/%d"
          %(k,s,len(c),len(pts),len(c&pocket),len(pocket)))
pairs=sorted((d(x,y),x,y) for x in c32 for y in pocket if pts[x]+pts[y]>d(x,y)+1e-12)
print("ball pairs overlapping into the pocket:",[("%.4f"%l,x,y) for l,x,y in pairs])
m=min(d(x,n)+d(x,Q) for x in itertools.product(range(-1,7),repeat=2) if x not in ((0,0),Q))
A=(Lw*Lw+rn*rn-rq*rq)/(2*Lw); rho=math.sqrt(rn*rn-A*A)
a=(rn+rq)/2; c=Lw/2; b=math.sqrt(a*a-c*c)
print("ellipse E: a=%.4f b=%.4f, lattice-free while rn+rq <= %.4f; window half-width %.4f, lens %.4f"
      %(a,b,m,rho,rn+rq-Lw))

# ---------------- SVG ----------------
s=52; pad=40; top=70; panel=330
Wd=(X1-X0)*s+2*pad+panel; Ht=(Y1-Y0)*s+pad+top+250
def P(p): return (pad+(p[0]-X0)*s, top+(Y1-p[1])*s)
o=[]
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd:.0f}" height="{Ht:.0f}" viewBox="0 0 {Wd:.0f} {Ht:.0f}" font-family="Helvetica,Arial,sans-serif">')
o.append('<rect width="100%" height="100%" fill="white"/>')
x0,y0=P((X0,Y1)); x1,y1=P((X1,Y0))
o.append(f'<defs><clipPath id="dom"><rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}"/></clipPath>')
o.append('<marker id="ar" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#6a1b9a"/></marker></defs>')
o.append(f'<text x="{pad}" y="30" font-size="18" font-weight="bold">A whole non-barrier pocket that only a hop of |w| = √34 ≈ 5.83 can enter (2D, W = 3, D = 0.55, so W + D = 3.55, 2W = 6)</text>')
o.append(f'<text x="{pad}" y="52" font-size="13" fill="#444">Exterior = tunnel ∪ B(n, {rn}) ∪ B(q, {rq}); the surface Σ is its boundary (black). All udf values are true distances to Σ.</text>')
def shapes(stroke):
    st='stroke="black" stroke-width="3"' if stroke else 'stroke="none"'
    r=[]
    cx,cy=P(n); r.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rn*s:.2f}" fill="white" {st}/>')
    cx,cy=P(q); r.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rq*s:.2f}" fill="white" {st}/>')
    ax,ay=P((X0-1,hc)); bx,by=P((xt+0.3,-hc))
    r.append(f'<rect x="{ax:.2f}" y="{ay:.2f}" width="{bx-ax:.2f}" height="{by-ay:.2f}" fill="white" {st}/>')
    return r
o.append('<g clip-path="url(#dom)">')
o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#cfcfcf"/>')
o+=shapes(True); o+=shapes(False)
o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#888" stroke-width="1"/>')
cx,cy=P((q[0]/2,q[1]/2)); ang=-math.degrees(math.atan2(q[1],q[0]))
o.append(f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{a*s:.2f}" ry="{b*s:.2f}" transform="rotate({ang:.3f} {cx:.2f} {cy:.2f})" fill="#ff9800" fill-opacity="0.25" stroke="#e65100" stroke-width="1.2" stroke-dasharray="5,3"/>')
cx,cy=P(q); o.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{(W+D)*s:.2f}" fill="none" stroke="#1565c0" stroke-width="1.5" stroke-dasharray="2,4"/>')
o.append('</g>')
for x in range(X0,X1+1):
    for y in range(Y0,Y1+1):
        px,py=P((x,y))
        if (x,y) not in pts:
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.3" fill="#777"/>'); continue
        if (x,y)==Q:
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="#d32f2f" stroke="black" stroke-width="1.2"/>')
        elif (x,y) in c32:
            fill="#2e7d32" if pts[(x,y)]>=D else "#81c784"
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" fill="{fill}" stroke="black" stroke-width="0.8"/>')
        else:
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" fill="white" stroke="#d32f2f" stroke-width="2"/>')
nx,ny=P(n); qx,qy=P(q)
L=math.hypot(qx-nx,qy-ny); ux,uy=(qx-nx)/L,(qy-ny)/L
o.append(f'<line x1="{nx+ux*8:.2f}" y1="{ny+uy*8:.2f}" x2="{qx-ux*9:.2f}" y2="{qy-uy*9:.2f}" stroke="#6a1b9a" stroke-width="2" marker-end="url(#ar)"/>')
def T(p,txt,dx=0,dy=0,size=13,col="black",anchor="start",weight="normal"):
    x,y=P(p); o.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{txt}</text>')
T(n,"n",-16,-8,15,weight="bold"); T(n,f"udf = {rn:g}",-28,22,12)
T(q,"q",11,-9,15,weight="bold"); T(q,f"udf = {rq:g}",28,-9,12)
T((0.2,2.4),"|w| = √34 ≈ 5.83",0,0,13,"#6a1b9a",anchor="middle",weight="bold")
T((-1.5,-1.55),"B(n, 2.99)",0,0,12,"#333",anchor="middle")
T((5,5.45),"pocket B(q, 2.85)",0,0,12,"#333",anchor="middle")
T((-0.2,4.4),"W + D disk around q",0,0,11,"#1565c0",anchor="middle")
T((X0,0),"to the outside (seeds)",6,-4-0.35*s,12,"#333")
T((-4.5,-2.4),"solid (inside)",0,0,13,"#555",anchor="middle")
T((4.4,-2.4),"solid (inside)",0,0,13,"#555",anchor="middle")
wc=(A*q[0]/Lw,A*q[1]/Lw); wx,wy=P(wc)
o.append(f'<circle cx="{wx:.1f}" cy="{wy:.1f}" r="11" fill="none" stroke="#333" stroke-width="1.2"/>')
# zoom panel on the window
zs=350; zc=wc; zx0=x1+40; zy0=y0; zw=panel-50; zh=zw
o.append(f'<defs><clipPath id="zoom"><rect x="{zx0}" y="{zy0}" width="{zw}" height="{zh}"/></clipPath></defs>')
def Z(p): return (zx0+zw/2+(p[0]-zc[0])*zs, zy0+zh/2-(p[1]-zc[1])*zs)
o.append(f'<line x1="{wx+8:.1f}" y1="{wy+8:.1f}" x2="{zx0:.1f}" y2="{zy0+zh:.1f}" stroke="#333" stroke-width="1" stroke-dasharray="3,3"/>')
o.append('<g clip-path="url(#zoom)">')
o.append(f'<rect x="{zx0}" y="{zy0}" width="{zw}" height="{zh}" fill="#cfcfcf"/>')
for cen,r in ((n,rn),(q,rq)):
    cx,cy=Z(cen); o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r*zs:.1f}" fill="white" stroke="black" stroke-width="3"/>')
for cen,r in ((n,rn),(q,rq)):
    cx,cy=Z(cen); o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r*zs:.1f}" fill="white" stroke="none"/>')
cx,cy=Z((q[0]/2,q[1]/2))
o.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{a*zs:.1f}" ry="{b*zs:.1f}" transform="rotate({ang:.3f} {cx:.1f} {cy:.1f})" fill="#ff9800" fill-opacity="0.25" stroke="#e65100" stroke-width="1.2" stroke-dasharray="5,3"/>')
pa=(wc[0]-rho*q[1]/Lw,wc[1]+rho*q[0]/Lw); pb=(wc[0]+rho*q[1]/Lw,wc[1]-rho*q[0]/Lw)
ax_,ay_=Z(pa); bx_,by_=Z(pb)
o.append(f'<line x1="{ax_:.1f}" y1="{ay_:.1f}" x2="{bx_:.1f}" y2="{by_:.1f}" stroke="#d32f2f" stroke-width="2.5"/>')
ea,eb=Z((wc[0]-0.12*q[0]/Lw,wc[1]-0.12*q[1]/Lw)),Z((wc[0]+0.12*q[0]/Lw,wc[1]+0.12*q[1]/Lw))
o.append(f'<line x1="{ea[0]:.1f}" y1="{ea[1]:.1f}" x2="{eb[0]:.1f}" y2="{eb[1]:.1f}" stroke="#6a1b9a" stroke-width="2" marker-end="url(#ar)"/>')
o.append('</g>')
o.append(f'<rect x="{zx0}" y="{zy0}" width="{zw}" height="{zh}" fill="none" stroke="#333" stroke-width="1.2"/>')
ty=zy0+zh+20
for i,t in enumerate([f"Zoom ×{zs/s:.1f} on the window (square is {zw/zs:.2f} voxel wide).",
                      f"Window (red) {2*rho:.2f} voxel wide; lens {rn+rq-Lw:.3f} voxel thick.",
                      "Orange: ellipse E, here only "+f"{2*b:.2f} voxel",
                      "across and free of lattice points."]):
    o.append(f'<text x="{zx0}" y="{ty+i*17}" font-size="12" fill="#222">{t}</text>')
ly=top+(Y1-Y0)*s+30
def leg(i,svgshape,txt):
    y=ly+i*22
    o.append(svgshape(pad+10,y-4)); o.append(f'<text x="{pad+26}" y="{y}" font-size="13">{txt}</text>')
leg(0,lambda x,y:f'<circle cx="{x}" cy="{y}" r="5.5" fill="#2e7d32" stroke="black" stroke-width="0.8"/>',"certified, non-barrier (udf ≥ 0.55): the 4-connected seed component")
leg(1,lambda x,y:f'<circle cx="{x}" cy="{y}" r="5.5" fill="#81c784" stroke="black" stroke-width="0.8"/>',"certified by the sweep, barrier (udf &lt; 0.55)")
leg(2,lambda x,y:f'<circle cx="{x}" cy="{y}" r="5.5" fill="white" stroke="#d32f2f" stroke-width="2"/>',"truly outside, NOT certified at the fixpoint of any stencil shorter than |w|² = 34 (the pocket)")
leg(3,lambda x,y:f'<circle cx="{x}" cy="{y}" r="7" fill="#d32f2f" stroke="black" stroke-width="1.2"/>',"q: the only pocket point any certified ball overlaps")
leg(4,lambda x,y:f'<circle cx="{x}" cy="{y}" r="2.3" fill="#777"/>',"lattice point inside the solid")
cap=[f"Fixpoint, stencil |w| &lt; W + D (|w|² ≤ 10, the 2D analogue of the 146-tap stencil), or any stencil up to |w|² ≤ 32: {len(c32)}/{len(pts)} certified after {s32} sweeps;",
     f"none of the {len(pocket)} pocket points, a whole non-barrier component, is ever entered. With |w|² ≥ 34: n certifies q ({rn}+{rq} = {rn+rq:.2f} &gt; {Lw:.3f}), then the pocket floods ({len(cF)}/{len(pts)}).",
     f"Why nothing shorter works: (n, q) is the only ball pair that overlaps through the window. Any lattice x in E = {{|x-n|+|x-q| &lt; {rn+rq:.2f}}} would bridge",
     f"n→x→q with shorter hops, but E holds no lattice point other than n and q (that needs udf(n)+udf(q) &gt; {m:.4f}, via (2,1) or (3,2)).",
     f"Why the W + D bound does not apply: it assumes every exterior non-barrier point is certified before the sweep; the pocket's are not, and promotion cannot help",
     f"before one of them is certified. The price: the pocket hangs on a {2*rho:.2f}-voxel window, a sub-voxel neck."]
for i,c in enumerate(cap):
    o.append(f'<text x="{pad}" y="{ly+5*22+18+i*19}" font-size="12.5" fill="#222">{c}</text>')
o.append('</svg>')
open(sys.argv[1] if len(sys.argv)>1 else "pocket_entry_failure_2d.svg","w").write("\n".join(o))
