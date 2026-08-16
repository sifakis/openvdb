import math
RD=math.sqrt(2)/2; E=0.019202630763796344
ASTAR=math.radians(15.930625116297946)
def cands(W):
    R=int(math.ceil(W+1.5)); return [(a,b,math.hypot(a,b)) for a in range(-R,R+1) for b in range(-R,R+1) if (a,b)!=(0,0)]
def detail(al,W,C,show=False):
    n=(math.cos(al),math.sin(al)); iv=[]
    for a,b,L in C:
        dot=a*n[0]+b*n[1]; g=(L-dot)/2
        lo=max(g,RD-dot); hi=min(RD,W-dot)
        if lo<hi: iv.append((lo,hi,(a,b)))
    iv.sort()
    if show:
        for lo,hi,w in iv[:8]: print("     J%s = (%.9f, %.9f]"%(w,lo,hi))
    merged=[]
    if iv:
        cl,ch=iv[0][0],iv[0][1]
        for lo,hi,w in iv[1:]:
            if lo<=ch: ch=max(ch,hi)
            else: merged.append((cl,ch)); cl,ch=lo,hi
        merged.append((cl,ch))
    cur=0.0; top=0.0
    for lo,hi in merged:
        if lo>cur: top=lo
        cur=max(cur,hi)
    if cur<RD: top=RD
    return top, merged
for W in [3.0,3.16,3.17,3.1784,3.178460085,3.1785,3.19]:
    C=cands(W); top,m=detail(ASTAR,W,C)
    print("W=%.9f  top at alpha* = %.12f   merged=%s"%(W,top,[("%.6f"%x,"%.6f"%y) for x,y in m][:4]))
print()
C=cands(3.17); detail(ASTAR,3.17,C,show=True)
