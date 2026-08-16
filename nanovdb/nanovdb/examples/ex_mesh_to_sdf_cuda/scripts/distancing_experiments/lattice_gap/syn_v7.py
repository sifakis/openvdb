import math
RD=math.sqrt(2)/2
A1=math.radians(15.930625116297946)   # alpha*
A2=math.radians(24.469800521)         # (1,0)x(1,1) crossover
A3=math.radians(34.729139026)         # (2,1)x(1,1) crossover
def cands(W):
    R=int(math.ceil(W+1.5)); return [(a,b,math.hypot(a,b)) for a in range(-R,R+1) for b in range(-R,R+1) if (a,b)!=(0,0)]
def top(al,W,C):
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
    cur=0.0;t=0.0
    for lo,hi in merged:
        if lo>cur: t=lo
        cur=max(cur,hi)
    if cur<RD: t=RD
    return t
WLO=2.27948314405977707271; WHI=3.17846008520840066492
for nm,Wc in [("W_lo'",WLO),("W_hi'",WHI)]:
    for d in (-1e-7,+1e-7):
        W=Wc+d; C=cands(W); best=0; ba=0
        # coarse
        for i in range(0,90001):
            al=math.radians(i*90/90000); v=top(al,W,C)
            if v>best: best,ba=v,al
        # ultrafine around the three critical angles
        for A in (A1,A2,A3):
            for j in range(-4000,4001):
                al=A+j*1e-9
                v=top(al,W,C)
                if v>best: best,ba=v,al
        print("%-6s W=%.14f  eps*_act=%.15f at alpha=%.9f deg"%(nm,W,best,math.degrees(ba)))
# how narrow is the alpha* spike just below W_hi'
for W in [3.0,3.15,3.17,3.178,3.1784,WHI-1e-7]:
    C=cands(W); lo=0.0; hi=0.6
    # find half-width where top drops below eps_c - 1e-12 scanning downward from alpha*
    e=0.019202630763796344; d=1e-12; hw=0.0
    x=0.0
    step=1e-3
    while step>1e-13:
        while top(A1-(x+step),W,C) > e-1e-9: x+=step
        step/=10
    print("W=%.10f : angular half-width (below alpha*) of the eps_c peak = %.3e rad = %.3e deg"%(W,x,math.degrees(x)))
