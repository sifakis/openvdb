# 3D brief: does the lattice certification gap carry over, and how?

READ FIRST, in this order:
  1. `/tmp/dist/CONTEXT.md`  — original problem statement. **It contains one known error**, see
     "CORRECTION" below. Everything else in it stands.
  2. `/tmp/dist/RESULT.md`   — the completed, referee-checked 2D result (1000 lines). This is the
     template for what a finished answer looks like.

**Do NOT do any literature search.** That is explicitly out of scope for this effort.

---

## CORRECTION to CONTEXT.md (important, do not inherit the error)

CONTEXT.md asserts a "threshold admissibility rule" and justifies it with *"for larger eps the
witness depth only grows further"*. That justification is **wrong**: growing depth pushes a witness
out of the **top** of the band. There are two distinct rules and they are not equivalent.

* **Threshold rule:** `w` admissible iff `r_d < (|w| + w.n)/2 <= W`.
* **Exact rule (what the pipeline actually implements):** `w` usable against a target at depth `eps`
  iff `eps > g(w,n)` **and** `r_d < eps + w.n <= W`, evaluated at the *actual* `eps`.

Under the exact rule a witness's usable set is an interval **bounded above**:
`J_w = ( max(g, r_d - w.n),  min(r_d, W - w.n) ]`, and a union of such intervals need not be upward
closed. In 2D the two rules give the same constant at `W = 3` but different plateaus
(`[sqrt5, 3.1608)` vs `[2.2795, 3.1785)`). **Report both rules separately in 3D.**

---

## What is already PROVED to be dimension-free (re-verify, but expect these to hold)

1. **The reduction.** With target lattice point at the origin at depth `eps` and witness at integer
   offset `w`, witness depth `d0 = eps + w.n`, the pairwise rule `d1 + d0 > |w|` becomes
   ```
       certified  <=>  eps > g(w,n) := (|w| - w.n)/2 = |w| sin^2(theta_w/2)
   ```
   `theta_w = angle(n,w)`. The derivation never uses the dimension.
2. **`|w| = g + d0_threshold`** where `d0_threshold = (|w| + w.n)/2`, and the identity
   **`g * d0_threshold = p^2/4`** where `p` is the tangential component of `w` (since
   `|w|^2 = (w.n)^2 + p^2`).
3. **Finiteness.** In-band plus `g < G` forces `|w| < W + G`, under BOTH rules (under the exact rule
   `|w| = 2g + w.n <= 2g + (W - eps) < W + g`, using `eps > g`). At `W = 3` this is `|w| <= ~3.02`,
   which is **122 lattice vectors in Z^3** (28 in Z^2). Bootstrap `G` if you need to: e.g. with
   `S` = the 26 vectors of the 3x3x3 neighbourhood, `G0 = max_n min_{w in S} g(w,n)` is a valid crude
   upper bound; then re-enumerate with `|w| <= W + G0` and iterate to a fixpoint.
4. **`g(w,.)` is affine in `n`.** So a cost tie `g(w1,n) = g(w2,n)` is the linear condition
   `(w2 - w1).n = |w2| - |w1|` intersected with the sphere. In 2D that is a point pair; **in 3D it is
   a great-circle-like curve (a plane section of `S^2`)**.
5. **Ball-membership criterion = exactly 2x the pairwise constant**, in any dimension, at any `W`
   (`|w| < eps + w.n` gives `eps > 2g`; the usability window does not involve the criterion).
6. **Curvature.** The exact circular-interface result is
   `eps_thr(w) = g / (1 -+ d0/R)` (minus = concave exterior, plus = convex), first order
   `g + p^2/(4R)`. In 3D the correction is `(p . H . p)/4` with `H` the second fundamental form.

## What is 2D-ONLY and must be redone

* The constant `eps_c = (3 - sqrt5 - sqrt(2 sqrt5 - 4))/4 = 0.0192026307637963438...`
* The critical direction `alpha* = 15.9306 deg` and the tying pair `(1,0)`, `(2,1)`.
* The "16 primitive directions of `[-2,2]^2`", their `26.565 deg` max angular gap, and the
  three-witness-per-octant covering.
* The Farey / Stern-Brocot crossover characterisation.
* The plateau endpoints in `W`, and the curvature constant `K = 0.0283152`.

## What specifically needs re-checking rather than assuming

**Lemma A (the barrier test is inactive).** In 2D: a certifying witness has depth
`> |w| - g >= 1 - g`, so the barrier test only bites if `g >= 1 - r_d = 0.2929`; since
`eps_c = 0.0192 << 0.2929`, the barrier half of admissibility is irrelevant at `W = 3`. **In 3D
`r_d = sqrt3/2 = 0.8660`, so the threshold drops to `1 - r_d = 0.1340`.** Still probably above the
3D constant, but this must be *verified*, not assumed — and if the 3D constant turns out larger than
`0.1340`, the barrier test becomes active and the whole analysis changes.

## Symmetry

The relevant group is the **48-element hyperoctahedral group** `B_3` (signed coordinate
permutations), which maps `Z^3` to itself and preserves both `g` and `d0`. A fundamental domain is
the spherical triangle `0 <= n_z <= n_y <= n_x`. Everything can be reduced to it.

## The expected shape of the 3D answer

In 2D the maximum sits where **two** witnesses tie (a transversal kink of the lower envelope). On
`S^2` the lower envelope `min_w g(w,n)` is a piecewise-smooth function whose generic local maxima sit
at points where **three** cost surfaces meet — the analogue of a Voronoi vertex. So expect the 3D
critical configuration to be a **triple tie**, and expect the answer to be an algebraic number
obtained by solving three linear tie conditions together with `|n| = 1`. The critical-point
certificate is: enumerate all candidate witnesses, all pairwise tie planes, evaluate the lower
envelope at (a) every triple intersection of tie planes with the sphere, (b) every intersection of a
pair of tie planes with a band-edge or barrier-edge surface, (c) all sphere-boundary/fundamental-
domain-boundary cases, and (d) stationary points on 1D tie curves. This is `O(N^3)` in the number of
witnesses; with `N ~ 122` that is `~300k` triples, entirely feasible.

Do not assume a closed form exists. A **certified numerical value with a proof of correctness of the
enumeration** is a perfectly good result, and is the primary deliverable. A surd is a bonus.

## Ground rules

* Python 3, **no numpy**, no pip. Use `fractions`, `decimal`, `math`. Pure Python is fast enough if
  you are careful (avoid nested scans over millions of directions x hundreds of witnesses; use the
  critical-point certificate instead of brute grids where possible).
* Scratch dir `/tmp/dist/d3/`. Never write into the git repo.
* Report honestly. A negative result ("no closed form found, here is the certified interval") is
  valuable. Fabricated agreement is not. State precision for every number.
* Cross-check everything at least twice, by genuinely different methods.
