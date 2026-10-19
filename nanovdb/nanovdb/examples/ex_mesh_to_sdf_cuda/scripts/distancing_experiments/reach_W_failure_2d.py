#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""2D analogue of the reach-W failure of the ball-certification sweep (figure in
NewDistancingApproach.md, "Stencil reach of the closure sweep").

Exterior = tunnel u B(n, rn) u B(q, rq); Sigma is its boundary.  Computes true UDFs, runs the
ball-overlap closure to its fixpoint with reach |w| <= W and |w| < W + sqrt2/2, prints the outcome,
and writes the SVG.

Usage: python3 reach_W_failure_2d.py [out.svg]    (default: reach_W_failure_2d.svg)
The PNG next to the SVG in figures/ was rendered with headless Chrome:
  google-chrome --headless --screenshot=out.png --window-size=920,745 file://$PWD/out.svg
"""
import math, itertools, sys
W=3.0; rd=math.sqrt(2)/2
n=(0.0,0.0); q=(3.0,1.0); rn=2.55; rq=0.65; hc=1.3          # tunnel half-width
X0,X1,Y0,Y1=-7,5,-3,3                                        # domain; tunnel opens at x=X0 (the "outside")
xt=-math.sqrt(rn*rn-hc*hc)                                   # tunnel meets the big circle here
def d(a,b): return math.dist(a,b)
def inU(p):
    return d(p,n)<rn or d(p,q)<rq or (p[0]<=xt+1e-12 and abs(p[1])<hc)
# dense sample of the boundary of U (the "surface" Sigma)
S=[]
N=200000
for i in range(N):
    t=2*math.pi*i/N
    p=(n[0]+rn*math.cos(t),n[1]+rn*math.sin(t))
    if d(p,q)>=rq and not (p[0]<=xt and abs(p[1])<hc): S.append(p)
    p=(q[0]+rq*math.cos(t),q[1]+rq*math.sin(t))
    if d(p,n)>=rn: S.append(p)
for i in range(20001):
    x=X0-3+(xt-(X0-3))*i/20000
    S+=[(x,hc),(x,-hc)]
def udf(p): return min(d(p,s) for s in S)
pts={}
for x in range(X0,X1+1):
    for y in range(Y0,Y1+1):
        if inU((x,y)):
            u=udf((x,y))
            if u<=W: pts[(x,y)]=u
seeds={p for p in pts if p[0]==X0}
def closure(R2):
    cert=set(seeds); sweeps=0
    while True:
        sweeps+=1; new=set()
        for x,ux in pts.items():
            if x in cert: continue
            for y in cert:
                L2=(x[0]-y[0])**2+(x[1]-y[1])**2
                if L2<=R2 and ux+pts[y]>math.sqrt(L2)+1e-12: new.add(x); break
        if not new: return cert,sweeps
        cert|=new
cW,sW=closure(9)                       # |w| <= W
cF,sF=closure(math.floor((W+rd)**2))   # |w| <  W + rd   (|w|^2 <= 13)
Q=(3,1)
print("udf(n)=%.4f udf(q)=%.4f  rn+rq=%.3f  |w|=%.4f"%(pts[(0,0)],pts[Q],rn+rq,math.hypot(3,1)))
print("reach W   : fixpoint after %d sweeps, %d/%d certified, q certified=%s"%(sW,len(cW),len(pts),Q in cW))
print("reach W+rd: fixpoint after %d sweeps, %d/%d certified, q certified=%s"%(sF,len(cF),len(pts),Q in cF))
print("missing at reach W:",sorted(set(pts)-cW), " missing at full reach:",sorted(set(pts)-cF))
# best short-hop attempt on q: max over certified x with |x-q|<=W of (udf(x)+rq-|x-q|)
best=max(((pts[x]+rq-d(x,Q)),x) for x in cW if d(x,Q)<=W)
print("best short certifier of q: x=%s margin=%.4f (needs >0)"%(best[1],best[0]))
for x in [(2,1),(2,0),(1,0),(1,1)]:
    print(" ",x,"udf=%.4f  udf+rq=%.4f  |x-q|=%.4f"%(pts[x],pts[x]+rq,d(x,Q)))
a=(rn+rq)/2; c=math.hypot(3,1)/2; b=math.sqrt(a*a-c*c)
A=(10+rn*rn-rq*rq)/(2*math.sqrt(10)); rho=math.sqrt(rn*rn-A*A)
print("ellipse E: a=%.4f b=%.4f ; window half-width rho=%.4f"%(a,b,rho))

# ---------------- SVG ----------------
s=70; pad=40; top=70; Wd=(X1-X0)*s+2*pad; Ht=(Y1-Y0)*s+pad+top+215
def P(p): return (pad+(p[0]-X0)*s, top+(Y1-p[1])*s)
o=[]
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd:.0f}" height="{Ht:.0f}" viewBox="0 0 {Wd:.0f} {Ht:.0f}" font-family="Helvetica,Arial,sans-serif">')
o.append(f'<rect width="100%" height="100%" fill="white"/>')
x0,y0=P((X0,Y1)); x1,y1=P((X1,Y0))
o.append(f'<defs><clipPath id="dom"><rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}"/></clipPath></defs>')
o.append(f'<text x="{pad}" y="30" font-size="18" font-weight="bold">Reach-W ball certification misses a truly-outside point, even at the fixpoint (2D, W = 3)</text>')
o.append(f'<text x="{pad}" y="52" font-size="13" fill="#444">Exterior = tunnel ∪ B(n, {rn}) ∪ B(q, {rq}); the surface Σ is its boundary (black). All udf values are true distances to Σ.</text>')
o.append('<g clip-path="url(#dom)">')
o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#cfcfcf"/>')
def shapes(stroke):
    st=f'stroke="black" stroke-width="3"' if stroke else 'stroke="none"'
    cx,cy=P(n); r=[]
    r.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rn*s:.2f}" fill="white" {st}/>')
    cx,cy=P(q); r.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rq*s:.2f}" fill="white" {st}/>')
    ax,ay=P((X0-1,hc)); bx,by=P((xt+0.3,-hc))
    r.append(f'<rect x="{ax:.2f}" y="{ay:.2f}" width="{bx-ax:.2f}" height="{by-ay:.2f}" fill="white" {st}/>')
    return r
o+=shapes(True); o+=shapes(False)
o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#888" stroke-width="1"/>')
# ellipse E
cx,cy=P(((n[0]+q[0])/2,(n[1]+q[1])/2)); ang=-math.degrees(math.atan2(1,3))
o.append(f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{a*s:.2f}" ry="{b*s:.2f}" transform="rotate({ang:.3f} {cx:.2f} {cy:.2f})" fill="#ff9800" fill-opacity="0.18" stroke="#e65100" stroke-width="1.5" stroke-dasharray="6,4"/>')
# reach-W circle around q
cx,cy=P(q); o.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{W*s:.2f}" fill="none" stroke="#1565c0" stroke-width="1.5" stroke-dasharray="2,4"/>')
o.append('</g>')
# lattice points
for x in range(X0,X1+1):
    for y in range(Y0,Y1+1):
        px,py=P((x,y))
        if (x,y) not in pts:
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.5" fill="#777"/>'); continue
        if (x,y)==Q:
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="#d32f2f" stroke="black" stroke-width="1.2"/>')
        elif (x,y) in cW:
            fill="#2e7d32" if pts[(x,y)]>=rd else "#81c784"
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="{fill}" stroke="black" stroke-width="0.8"/>')
        else:
            o.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="white" stroke="#d32f2f" stroke-width="2"/>')
# hop arrow n -> q
nx,ny=P(n); qx,qy=P(q)
o.append('<defs><marker id="ar" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#6a1b9a"/></marker></defs>')
L=math.hypot(qx-nx,qy-ny); ux,uy=(qx-nx)/L,(qy-ny)/L
o.append(f'<line x1="{nx+ux*8:.2f}" y1="{ny+uy*8:.2f}" x2="{qx-ux*9:.2f}" y2="{qy-uy*9:.2f}" stroke="#6a1b9a" stroke-width="2" marker-end="url(#ar)"/>')
def T(p,txt,dx=0,dy=0,size=13,col="black",anchor="start",weight="normal"):
    x,y=P(p); o.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{txt}</text>')
T(n,"n",-14,-10,15,weight="bold"); T(n,f"udf = {rn:g}",-30,24,12)
T(q,"q",10,-10,15,weight="bold"); T(q,f"udf = {rq:g}",50,24,12)
T((1.2,0.4),"|w| = √10 ≈ 3.162 &gt; W",-10,-26,13,"#6a1b9a",anchor="middle",weight="bold")
T((-0.6,1.75),"B(n, 2.55)",0,0,12,"#333",anchor="middle")
A2=(10+rn*rn-rq*rq)/(2*math.sqrt(10)); wc=(3*A2/math.sqrt(10),A2/math.sqrt(10))
wx,wy=P(wc); o.append(f'<line x1="{wx:.1f}" y1="{wy:.1f}" x2="{wx+30:.1f}" y2="{wy+62:.1f}" stroke="#333" stroke-width="1"/>')
o.append(f'<text x="{wx+33:.1f}" y="{wy+76:.1f}" font-size="12" fill="#333">window ≈ {2*rho:.2f} voxel wide</text>')
x21,y21=P((2,1)); o.append(f'<text x="{x21-8:.1f}" y="{y21-14:.1f}" font-size="11" fill="#333" text-anchor="end">(2,1): 0.314 + 0.65 &lt; 1</text>')
T((0.3,-2.75),"reach-W disk around q (radius 3)",0,0,11,"#1565c0")
T((1.4,-0.3),"ellipse E (lattice-free)",0,0,12,"#e65100",anchor="middle")
T((X0,0),"to the outside (seeds)",6,-4-0.35*s,12,"#333")
T((-3.5,-2.55),"solid (inside)",0,0,13,"#555",anchor="middle")
# legend / caption
ly=top+(Y1-Y0)*s+30
def leg(i,svgshape,txt):
    y=ly+i*22
    o.append(svgshape(pad+10,y-4)); o.append(f'<text x="{pad+26}" y="{y}" font-size="13">{txt}</text>')
leg(0,lambda x,y:f'<circle cx="{x}" cy="{y}" r="6" fill="#2e7d32" stroke="black" stroke-width="0.8"/>',"certified, non-barrier (udf ≥ √2/2)")
leg(1,lambda x,y:f'<circle cx="{x}" cy="{y}" r="6" fill="#81c784" stroke="black" stroke-width="0.8"/>',"certified, barrier (udf &lt; √2/2)")
leg(2,lambda x,y:f'<circle cx="{x}" cy="{y}" r="7" fill="#d32f2f" stroke="black" stroke-width="1.2"/>',"q: truly outside, NOT certified at the reach-W fixpoint (certified with reach W+√2/2)")
leg(3,lambda x,y:f'<circle cx="{x}" cy="{y}" r="2.5" fill="#777"/>',"lattice point inside the solid")
cap=[f"Fixpoint, reach |w| ≤ W = 3: {len(cW)}/{len(pts)} exterior band points certified after {sW} sweeps (last sweep changes nothing); q is never certified.",
     f"Fixpoint, reach |w| &lt; W+√2/2: {len(cF)}/{len(pts)} certified; q is certified directly from n (udf(n)+udf(q) = {rn+rq:.2f} &gt; |w| = 3.162).",
     f"Why no short chain: n's ball is the only ball that overlaps q's. Any lattice x in the orange ellipse E = {{|x-n|+|x-q| &lt; {rn+rq:.2f}}} would bridge n→x→q",
     f"with both hops &lt; W, but E holds no lattice point except n and q (it would need udf(n)+udf(q) &gt; 1+√5 ≈ 3.236). Best short attempt: x = (2,1),",
     f"udf {pts[(2,1)]:.3f} + {rq} = {pts[(2,1)]+rq:.3f} vs |x-q| = 1, margin {pts[(2,1)]+rq-1:+.3f}. The price: q sits in a sub-voxel bubble behind a {2*rho:.2f}-voxel window."]
for i,c in enumerate(cap):
    o.append(f'<text x="{pad}" y="{ly+4*22+18+i*19}" font-size="12.5" fill="#222">{c}</text>')
o.append('</svg>')
open(sys.argv[1] if len(sys.argv)>1 else "reach_W_failure_2d.svg","w").write("\n".join(o))
