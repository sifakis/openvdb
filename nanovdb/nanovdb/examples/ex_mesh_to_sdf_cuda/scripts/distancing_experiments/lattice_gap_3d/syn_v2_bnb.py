import time, math
from slib import *

ws14 = witnesses(14)
ws9 = witnesses(9)
print("witness counts:", len(ws9), len(ws14))

for rule, tgt in (('thr', 0.036662265118829642), ('exa', 0.038443423574719627)):
    for W in (3.0,):
        t0 = time.time()
        lo, up, nc = bnb(rule, ws14, W=W, tol=1e-12)
        print(f"{rule} W={W}: bracket [{lo:.15f}, {up:.15f}] cells={nc} "
              f"contains target: {lo - 1e-15 <= tgt <= up + 1e-15}  ({time.time()-t0:.1f}s)")

# strict vs non-strict band test, threshold rule
lo, up, nc = bnb('thr', ws14, W=3.0, tol=1e-11, strict_band=True)
print(f"thr STRICT band (d0 < W): [{lo:.13f}, {up:.13f}] cells={nc}")

# witness-radius stability
for n2 in (9, 14, 25, 49):
    ws = witnesses(n2)
    lo, up, nc = bnb('thr', ws, W=3.0, tol=1e-11)
    print(f"thr |w|^2<={n2:3d} ({len(ws):5d} vectors): [{lo:.13f}, {up:.13f}]")
for n2 in (14, 25, 49):
    ws = witnesses(n2)
    lo, up, nc = bnb('exa', ws, W=3.0, tol=1e-11)
    print(f"exa |w|^2<={n2:3d} ({len(ws):5d} vectors): [{lo:.13f}, {up:.13f}]")

# 3x3x3 bootstrap constant
lo, up, nc = bnb('thr', witnesses(3), W=3.0, tol=1e-11)
print(f"3x3x3 (|w|^2<=3, no filters bite): [{lo:.13f}, {up:.13f}]  vs 0.0735593211498971")
