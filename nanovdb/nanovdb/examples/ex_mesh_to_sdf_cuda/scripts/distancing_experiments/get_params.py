import math
import random

random.seed(20260817)
cases = []
for i in range(25):
    vertex = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
    theta1 = random.uniform(0, 2 * math.pi)
    alpha_deg = random.uniform(20, 160)
    cases.append((vertex, theta1, alpha_deg))

print("trial 0 (mild):", cases[0])
print("trial 6 (narrow w/ misses):", cases[6])

random.seed(7)
for alpha_deg in [2, 4, 6]:
    for trial in range(8):
        vertex = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        theta1 = random.uniform(0, 2 * math.pi)
        if alpha_deg == 6 and trial == 0:
            print(f"degenerate case alpha={alpha_deg}: vertex={vertex}, theta1={theta1}")
