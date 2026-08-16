import math
RD=math.sqrt(2)/2
def cands(W):
    R=int(math.ceil(W+1.5)); return [(a,b,math.hypot(a,b)) for a in range(-R,R+1) for b in range(-R,R+1) if (a,b)!=(0,0)]
def uncovered(al,W,C):
    n=(math.cos(al),math.sin(al)); iv=[]
    for a,b,L in C:
        dot=a*n[0]+b*n[1]; g=(L-dot)/2
        lo=max(g,RD-dot); hi=min(RD,W-dot)
        if lo<hi: iv.append((lo,hi))
    iv.sort(); merged=[]
    if iv:
        cl,ch=iv[0]
        for lo,hi in iv[1:]:
            if lo<=ch: ch=max(ch,hi)
            else: merged.append((cl,ch)); cl,ch=lo,hi
        merged.append((cl,ch))
    # complement of merged inside (0,RD]
    gaps=[]; cur=0.0
    for lo,hi in merged:
        if lo>cur: gaps.append((cur,lo))
        cur=max(cur,hi)
    if cur<RD: gaps.append((cur,RD))
    return gaps

# 1) at W=3, is uncovered set a single interval starting at 0 for all alpha?
C=cands(3.0); bad=0; worst=0; wa=0; nmulti=0
for i in range(0,90001):
    al=math.radians(i*90/90000)
    g=uncovered(al,3.0,C)
    if len(g)>1: nmulti+=1
    if g:
        if abs(g[0][0])>1e-15: bad+=1
        if g[-1][1]>worst: worst=g[-1][1]; wa=math.degrees(al)
print("W=3: angles with >1 uncovered gap: %d / 90001 ; angles whose first gap doesn't start at 0: %d"%(nmulti,bad))
print("W=3: max uncovered depth = %.15f at alpha=%.9f deg"%(worst,wa))

# 2) referee counterexample W=sqrt5, alpha=24 deg, eps=0.03
W=math.sqrt(5); al=math.radians(24.0); C=cands(W)
n=(math.cos(al),math.sin(al)); eps=0.03; found=[]
for a,b,L in C:
    dot=a*n[0]+b*n[1]; g=(L-dot)/2; d=eps+dot
    if eps>g and RD<d<=W: found.append((a,b))
print("\nW=sqrt5, alpha=24deg, eps=0.03: certifying witnesses =",found)
for w in [(2,1),(1,0),(1,1)]:
    a,b=w; L=math.hypot(a,b); dot=a*n[0]+b*n[1]
    print("   w=%s g=%.9f  threshold d0=%.9f (<=W? %s)  actual depth=%.9f (<=W? %s)"%(
        w,(L-dot)/2,(L+dot)/2,(L+dot)/2<=W,eps+dot,eps+dot<=W))
print("   uncovered gaps at that alpha:",[("%.6f"%x,"%.6f"%y) for x,y in uncovered(al,W,C)])

# 3) bisect the actual-rule plateau endpoints
def eps_act_max(W,N=6000):
    C=cands(W); best=-1
    for i in range(N+1):
        al=math.pi/2*i/N; g=uncovered(al,W,C)
        v=g[-1][1] if g else 0.0
        if v>best: best=v; ba=al
    a=max(0,ba-math.pi/2/N); b=min(math.pi/2,ba+math.pi/2/N); gr=(math.sqrt(5)-1)/2
    f=lambda x:(lambda gg: gg[-1][1] if gg else 0.0)(uncovered(x,W,C))
    for _ in range(120):
        x1=b-gr*(b-a); x2=a+gr*(b-a)
        if f(x1)>f(x2): b=x2
        else: a=x1
    return max(best,f((a+b)/2))
E=0.019202630763796344
for lohi,target in [((1.6,1.8),None),((2.2,2.35),None),((3.15,3.20),None)]:
    lo,hi=lohi
    for _ in range(50):
        m=(lo+hi)/2
        if abs(eps_act_max(m,1200)-E)<1e-9: hi=m
        else: lo=m
    print("\nactual-rule transition in %s -> %.15f"%(str(lohi),(lo+hi)/2))
print("1+r_d = %.15f ; W_lo' claim 2.279483144059777 ; W_hi' claim 3.178460085208401"%(1+RD))
