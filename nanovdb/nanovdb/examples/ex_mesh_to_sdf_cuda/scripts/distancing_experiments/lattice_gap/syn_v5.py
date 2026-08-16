import math
RD=math.sqrt(2)/2; E=0.019202630763796344
def cands(W):
    R=int(math.ceil(W+1.5)); return [(a,b,math.hypot(a,b)) for a in range(-R,R+1) for b in range(-R,R+1) if (a,b)!=(0,0)]
def unc_top(al,W,C):
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
    cur=0.0; top=0.0
    for lo,hi in merged:
        if lo>cur: top=lo
        cur=max(cur,hi)
    if cur<RD: top=RD
    return top
def M(W,N=40000):
    C=cands(W); best=-1;ba=0
    for i in range(N+1):
        al=math.pi/2*i/N; v=unc_top(al,W,C)
        if v>best: best,ba=v,al
    a=max(0,ba-math.pi/2/N); b=min(math.pi/2,ba+math.pi/2/N); gr=(math.sqrt(5)-1)/2
    f=lambda x: unc_top(x,W,C)
    for _ in range(150):
        x1=b-gr*(b-a); x2=a+gr*(b-a)
        if f(x1)>f(x2): b=x2
        else: a=x1
    return max(best,f((a+b)/2)), math.degrees((a+b)/2)
WLO=2.27948314405977707271; WHI=3.17846008520840066492; WCC=1+RD
for nm,Wc in [("1+r_d",WCC),("W_lo'",WLO),("W_hi'",WHI)]:
    for d in (-1e-7,0.0,+1e-7):
        v,a=M(Wc+d,20000)
        print("%-7s W=%.14f  eps*_act=%.15f  alpha=%.6f"%(nm,Wc+d,v,a))
    print()
