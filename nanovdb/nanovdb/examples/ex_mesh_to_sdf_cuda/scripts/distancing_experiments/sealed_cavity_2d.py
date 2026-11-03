#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""2D sealed cavity: a hollow disk (outer and inner circle) whose wall is thinner than 2W, so the
bands of the two surfaces merge into one UDF component.  The cavity has winding number 0 but is not
connected to the exterior, so no chain of surface-free balls can reach it, whatever the stencil.

Computes the analytic UDF, seeds with the 4-connected non-barrier component of the leftmost band
point, runs the ball-overlap closure to its fixpoint with the default stencil (|w| < W + D) and with
the pedantic reach (|w| < 2W), prints the outcome, and writes the SVG.

Usage: python3 sealed_cavity_2d.py [out.svg]    (default: sealed_cavity_2d.svg)
The PNG next to the SVG in figures/ was rendered with headless Chrome:
  google-chrome --headless --screenshot=out.png --window-size=860,1124 file://$PWD/out.svg
"""
import math, sys
W=3.0; D=0.55
c=(0.3,0.2); Ro=9.0; Ri=6.5                                   # wall thickness 2.5 < 2W
X0,X1,Y0,Y1=-13,13,-13,13
def r(p): return math.dist(p,c)
def udf(p): return min(abs(r(p)-Ro),abs(r(p)-Ri))
def winding(p): return 1 if Ri<r(p)<Ro else 0
def region(p):
    return "material" if winding(p) else ("cavity" if r(p)<Ri else "exterior")
pts={}
for x in range(X0,X1+1):
    for y in range(Y0,Y1+1):
        u=udf((x,y))
        if u<=W: pts[(x,y)]=u
start=min(pts)                                                # leftmost band point
seeds={start}; stack=[start]
while stack:
    a=stack.pop()
    for e in ((1,0),(-1,0),(0,1),(0,-1)):
        b=(a[0]+e[0],a[1]+e[1])
        if b in pts and pts[b]>=D and b not in seeds: seeds.add(b); stack.append(b)
def closure(R2):
    cert=set(seeds); sweeps=0
    while True:
        sweeps+=1; new=set()
        for x,ux in pts.items():
            if x in cert: continue
            for y in cert:
                L2=(x[0]-y[0])**2+(x[1]-y[1])**2
                if L2<R2 and ux+pts[y]>math.sqrt(L2)+1e-12: new.add(x); break
        if not new: return cert,sweeps
        cert|=new
cD,sD=closure((W+D)**2)
cP,sP=closure((2*W)**2)
by={k:[p for p in pts if region(p)==k] for k in ("exterior","material","cavity")}
print("band points: exterior %d, material (wall) %d, cavity %d; seeds %d"
      %(len(by["exterior"]),len(by["material"]),len(by["cavity"]),len(seeds)))
for name,(cs,s) in (("default |w| < W+D",(cD,sD)),("pedantic |w| < 2W",(cP,sP))):
    print("%-18s: fixpoint after %d sweeps; certified exterior %d/%d, material %d, cavity %d"
          %(name,s,sum(p in cs for p in by["exterior"]),len(by["exterior"]),
            sum(p in cs for p in by["material"]),sum(p in cs for p in by["cavity"])))
assert all(region(p)=="exterior" for p in cP)
cav_all=[(x,y) for x in range(X0,X1+1) for y in range(Y0,Y1+1) if region((x,y))=="cavity"]
cav_band=by["cavity"]; ext_unc=[p for p in by["exterior"] if p not in cP]
print("cavity lattice points: %d (%d in band, %d beyond it); uncertified exterior band points: %d %s"
      %(len(cav_all),len(cav_band),len(cav_all)-len(cav_band),len(ext_unc),
        ["%s udf=%.3f"%(p,pts[p]) for p in ext_unc]))

# ---------------- SVG ----------------
s=30; pad=40; top=74
Wd=(X1-X0)*s+2*pad; Ht=(Y1-Y0)*s+top+pad+230
def P(p): return (pad+(p[0]-X0)*s, top+(Y1-p[1])*s)
o=[]
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd:.0f}" height="{Ht:.0f}" viewBox="0 0 {Wd:.0f} {Ht:.0f}" font-family="Helvetica,Arial,sans-serif">')
o.append('<rect width="100%" height="100%" fill="white"/>')
x0,y0=P((X0,Y1)); x1,y1=P((X1,Y0))
o.append(f'<defs><clipPath id="dom"><rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}"/></clipPath></defs>')
o.append(f'<text x="{pad}" y="30" font-size="18" font-weight="bold">A sealed cavity: winding number 0, same UDF component, unreachable by any ball chain</text>')
o.append(f'<text x="{pad}" y="54" font-size="13" fill="#444">2D hollow disk: Σ = outer circle (R = {Ro:g}) ∪ inner circle (R = {Ri:g}); wall {Ro-Ri:g} &lt; 2W = {2*W:g}, so both bands form one UDF component. W = {W:g}, D = {D}.</text>')
o.append('<g clip-path="url(#dom)">')
cx,cy=P(c)
o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="white"/>')
o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{(Ro+W)*s:.1f}" fill="#eef6ee" stroke="none"/>')
o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{Ro*s:.1f}" fill="#bdbdbd" stroke="black" stroke-width="3"/>')
o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{Ri*s:.1f}" fill="#fff3c4" stroke="black" stroke-width="3"/>')
o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{(Ri-W)*s:.1f}" fill="#fffbe8" stroke="#b28704" stroke-width="1.3" stroke-dasharray="5,4"/>')
o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{(Ro+W)*s:.1f}" fill="none" stroke="#2e7d32" stroke-width="1.3" stroke-dasharray="5,4"/>')
# a few certified balls, touching the outer surface and never crossing it
for p in [(4,10),(10,3),(8,-6),(-5,-9),(-11,0)]:
    if p in cP:
        bx,by_=P(p); o.append(f'<circle cx="{bx:.1f}" cy="{by_:.1f}" r="{pts[p]*s:.1f}" fill="#2e7d32" fill-opacity="0.08" stroke="#2e7d32" stroke-width="1.2"/>')
o.append('</g>')
o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#888" stroke-width="1"/>')
for x in range(X0,X1+1):
    for y in range(Y0,Y1+1):
        px,py=P((x,y)); reg=region((x,y))
        if (x,y) in cP:
            fill="#2e7d32" if pts[(x,y)]>=D else "#81c784"
            o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{fill}" stroke="black" stroke-width="0.6"/>')
        elif reg=="material":
            o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="#444"/>')
        elif reg=="cavity" and (x,y) in pts:
            o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="white" stroke="#d32f2f" stroke-width="1.8"/>')
        elif reg=="cavity":
            o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="none" stroke="#d32f2f" stroke-width="1"/>')
        elif (x,y) in pts:
            o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="white" stroke="#2e7d32" stroke-width="1.8"/>')
        else:
            o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.6" fill="#bbb"/>')
def T(p,txt,dx=0,dy=0,size=13,col="black",anchor="middle",weight="normal"):
    x,y=P(p); o.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" font-weight="{weight}" style="paint-order:stroke" stroke="white" stroke-width="4" stroke-linejoin="round">{txt}</text>')
T((c[0],c[1]+0.4),"sealed cavity",0,0,15,"#8d6e00",weight="bold")
T((c[0],c[1]-0.7),"winding 0, not material",0,0,12,"#8d6e00")
T((c[0],c[1]-1.6),"udf &gt; W here: outside the band",0,0,11,"#8d6e00")
T((c[0]-7.75,c[1]+0.5),"wall",0,0,13,"#222",weight="bold")
T((c[0]-7.75,c[1]-0.4),"winding 1",0,0,11,"#222")
T((-11.6,12.2),"exterior",0,0,14,"#2e7d32",anchor="start",weight="bold")
T((5.3,-11.6),"edge of the band, udf = W",0,0,11,"#2e7d32",anchor="start")
T((c[0],c[1]-Ri+W+0.35),"udf = W",0,0,11,"#b28704")
ly=top+(Y1-Y0)*s+30
def leg(i,svgshape,txt):
    y=ly+i*21
    o.append(svgshape(pad+10,y-4)); o.append(f'<text x="{pad+24}" y="{y}" font-size="13">{txt}</text>')
leg(0,lambda x,y:f'<circle cx="{x}" cy="{y}" r="4.5" fill="#2e7d32" stroke="black" stroke-width="0.6"/>',"certified outside (dark: non-barrier; light: barrier, udf &lt; D); green circles: a few of their balls, touching Σ")
leg(1,lambda x,y:f'<circle cx="{x}" cy="{y}" r="2.6" fill="#444"/>',"material (winding 1): uncertified, inside — correct")
leg(2,lambda x,y:f'<circle cx="{x}" cy="{y}" r="4.5" fill="white" stroke="#d32f2f" stroke-width="1.8"/>',"cavity, in the band (winding 0): uncertified, labelled inside")
leg(3,lambda x,y:f'<circle cx="{x}" cy="{y}" r="2.6" fill="none" stroke="#d32f2f" stroke-width="1"/>',"cavity, beyond the band: inactive")
cap=[f"Fixpoint with the default stencil (|w| &lt; W + D) and with the pedantic reach (|w| &lt; 2W): the same {len(cP)} exterior points",
     f"certified (all of them); none of the {len(by['material'])} wall points and none of the {len(cav_band)} cavity band points.",
     "Every certified ball is surface-free, so a chain of overlapping balls never crosses Σ; the cavity is on the other side of Σ from",
     "every seed, so no stencil, schedule or promotion can reach it. The sweep therefore fills the cavity, which is the 'take the outermost'",
     "fallback of InclusionSigningDesign.md §7 for surfaces nested closer than the band. With a wall thicker than about 2W the inner circle",
     "would be its own UDF component, and the nesting stage (PropagateSign + even-odd rule) would make the cavity a void instead."]
for i,t in enumerate(cap):
    o.append(f'<text x="{pad}" y="{ly+4*21+20+i*19}" font-size="12.5" fill="#222">{t}</text>')
o.append('</svg>')
open(sys.argv[1] if len(sys.argv)>1 else "sealed_cavity_2d.svg","w").write("\n".join(o))
