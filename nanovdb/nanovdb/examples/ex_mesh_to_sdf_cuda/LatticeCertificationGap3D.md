# The 3D lattice certification gap: what the 2D result becomes in Z^3

> Full technical record. The readable summary with figures is `CertificationGap_3D_Note.md`;
> the 2D result it builds on is `LatticeCertificationGap.md`.

Status: the 2D machinery transfers almost entirely, but the *answer* does not — and one structural
thing genuinely breaks. At `W = 3` in `Z^3` the two admissibility rules give **different** constants
(they coincided in 2D), the critical configuration is a **triple** tie rather than a pair, the
constants are algebraic of degree **8** rather than 4, and the certified set is **not** upward closed.
Both constants are pinned in closed form and certified by independent exact-arithmetic and
branch-and-bound certificates.

```
    eps*_thr(3) = (3 - sqrt5 - sqrt(4 sqrt30 + 2 sqrt5 - 26))/4  = 0.0366622651188296422942...
    eps*_exa(3) = (sqrt2 + sqrt3 - sqrt6 - sqrt(6 sqrt2 + 2 sqrt6 - 13))/2
                                                                 = 0.0384434235747196265664...
```

`eps*_exa` is the number the pipeline actually delivers. Quote it **rounded up**:
`0.03844342357471963`.

This document synthesises four phase-1 tracks, three phase-2 tracks and two adversarial referee
reports. Every number below carries an explicit confidence marker. Numbers that survived a referee
challenge are marked as such; numbers that a referee *refuted* have been replaced and the refutation
is recorded in section 6. Where I re-derived something myself for this write-up the artifact is in
`scripts/distancing_experiments/lattice_gap_3d/syn/` and is called out. **No literature search was performed** (out of scope); no
novelty is claimed or implied.

Confidence markers used throughout:

| marker | meaning |
|---|---|
| **[PROVED]** | exact arithmetic end to end (integers/rationals/algebraic), no floating point in the decision path |
| **[CERTIFIED]** | rigorous enclosure computed in IEEE double with an explicit guard band; at least two genuinely different methods agree |
| **[HIGH-NUM]** | one rigorous method, or several searches that agree, but no enclosure |
| **[NUM]** | search/grid only; stated tolerance is empirical |
| **[OPEN]** | not established |

---

## 1. TRANSFER TABLE: each ingredient of the 2D result in 3D

`r_d` = lattice covering radius (`sqrt2/2` in 2D, `sqrt3/2` in 3D); `W = 3` throughout.

| # | 2D ingredient | Verdict | One-line justification |
|---|---|---|---|
| 1 | the reduction `certified <=> eps > g(w,n)` | **CARRIES VERBATIM** | derivation uses only `\|w\|`, `w.n`, 1-Lipschitzness; 0 mismatches vs the raw rule on 300 000 random 3D configs |
| 2 | `\|w\| = g + d0_thr` | **CARRIES VERBATIM** | identity by construction; residual `1.8e-15` |
| 3 | `g * d0_thr = p^2/4` | **CARRIES VERBATIM** | `= (\|w\|^2 - (w.n)^2)/4`, dimension-free; residual `3.6e-15` |
| 4 | finiteness `\|w\| < W + G` (both rules) | **CARRIES MODIFIED** | bound is dimension-free, but the count goes 28 -> **122**, and the exact-rule *sup* needs `\|w\| < W + r_d` (250 vectors) — note (a) |
| 5 | Lemma A (barrier test inactive) | **CARRIES MODIFIED** | threshold drops `0.2929 -> 0.13397`; still inactive but with `3.5x` margin instead of `15.3x` — note (b) |
| 5b | its 2D corollary "the short witness is admissible everywhere on the domain" | **FAILS** | `(1,0,0)` is barrier-excluded on `19.63 %` of `S^2`; repaired only *conditionally*, by Lemma A — note (b) |
| 6 | symmetry group `D4` (order 8), octant | **CARRIES MODIFIED** | `B_3`, order **48**; fundamental spherical triangle `n_x >= n_y >= n_z >= 0`; invariance to `4.4e-16` — note (c) |
| 7 | tie loci (affine `g` => linear tie conditions) | **CARRIES MODIFIED** | still a plane section, but of `S^2`: a **small** circle in general, great **iff** `\|w1\| = \|w2\|` — note (d) |
| 8 | primitivity `g(kw) = k g(w)`; reduce to primitives | **CARRIES VERBATIM** / **MODIFIED** | identity dimension-free; the exact-rule reduction is now *unconditional* for `r_d < 1`, the threshold-rule caveat is `4x` tighter — note (e) |
| 9 | the small upper-bound witness set (3 per octant, 16 total) | **CARRIES MODIFIED** | **6** classes / 98 vectors (threshold), **5** / 74 (exact); minimality proved by deleting each class — section 2.8 |
| 10 | ball membership `=` exactly `2x` pairwise | **CARRIES VERBATIM** | `\|w\| < eps + w.n <=> eps > 2g`; the window never mentions the criterion; confirmed end-to-end to `2.6e-11` |
| 11 | the constant `eps_c = 0.0192026307637963` | **FAILS** (2D-only, as expected) | replaced by **two** constants; the 2D value survives in 3D only as a *saddle* on the mirror face `n_z = 0` |
| 12 | critical configuration: kink, **two** tied witnesses | **CARRIES MODIFIED — as CONTEXT_3D predicted** | a **triple** tie / conic point under both rules; no fourth witness within `1e-60` over `\|w\|^2 <= 400` |
| 13 | the `W`-plateau (staircase, algebraic endpoints) | **CARRIES MODIFIED** | plateaus and algebraic endpoints survive under both rules, but 3D is far rampier and the two rules sit on **different** plateaus at `W = 3` |
| 14 | curvature law `g/(1 -+ d0/R)`, first order `g + p^2/(4R)` | **CARRIES VERBATIM** (sphere) | the sphere law is dimension-free (centre, target, witness span a 2-plane); checked against raw geometry to `2.16e-12` |
| 14b | the curvature constant `K = 0.0283152` | **FAILS** (2D-only) | `K_thr = 0.0607370863`, `K_exa = 0.0634946437` — `2.15x` / `2.24x`; curvature hurts **more** in 3D |
| 14c | "a plane is not the worst case — is a sphere?" | **STILL OPEN** (restricted result) | proved for `R > W = 3`; certificate + adversarial search below; general surfaces untouched — section 4.5 |
| 15 | Farey / Stern-Brocot crossover characterisation | **STILL OPEN** | no 3D analogue attempted; both tying triples have `det = +-1`, suggesting a unimodular-triple theory |
| 16 | *(2D open item)* upward-closedness of the certified set | **FAILS in 3D** | the uncovered depth set is not an interval at `8.45 %` of directions (2D: 0 of 90 001) — note (f) |
| 17 | *(2D open item)* multi-hop enrichment | **CLOSED — now PROVED inert** | a barrier witness needs `eps > 1 - r_d = 0.13397 >> eps*`; Lemma M, section 2.9, dimension-free |
| 18 | *(2D hazard)* `tol = 0` unsound at lattice-parallel normals | **CARRIES MODIFIED — much worse** | same mechanism, but in 3D one 1-ulp error cascades to the whole interior band — section 8 |

**Notes on the modified rows.**

**(a) Finiteness.** In band plus `g < G` still forces `\|w\| < W + G` under both rules, and the
bootstrap still closes in one step: the exact `3x3x3` constant is
`G0 = (1 - sqrt(2 sqrt2 + 2 sqrt6 - 7))/2 = 0.0735593211498971297`, and
`W + G0 = 3.0736 < sqrt10`, so `\|w\|^2 <= 9` — **122** lattice vectors, 98 primitive, in six `B_3`
orbits. Re-running with `G = eps*` returns the same set, and the answer is bit-identical for witness
sets out to `\|w\|^2 <= 256` (17 076 vectors). *New 3D bookkeeping caveat:* to compute the **sup of
the uncovered set** under the exact rule (as opposed to the cheapest witness) one must enumerate
`\|w\| < W + r_d = 3.866`, i.e. `\|w\|^2 <= 14`, 250 vectors — deeper targets need longer witnesses.
This distinction is invisible in 2D at `W = 3` and it caught one track out.

**(b) Lemma A and its corollary.** The lemma itself transfers unchanged in form: a certifying witness
has depth `> \|w\| - g >= 1 - g`, so the barrier test bites only when `g >= 1 - r_d`. In 3D that is
`0.133974596215561`, and `eps*_thr / eps*_exa` clear it by factors `3.65 / 3.49` (2D: `15.3`). So the
barrier half is inactive at `W = 3` — **verified, not assumed**: dropping the barrier test entirely,
or substituting the 2D `r_d`, changes `eps*` by 0 ulp (200 000 random directions and a `K = 400`
grid); the cheapest barrier-excluded witness at `n*_thr` is `(0,1,0)` at `g = 0.345303746 = 9.4x
eps*`. But the *corollary* the 2D proof leaned on fails: `(1,0,0)` has
`min d0_thr = (1 + 1/sqrt3)/2 = 0.788675134594813 < r_d` over the fundamental domain, so its whole
orbit is barrier-excluded whenever `max\|n_i\| <= sqrt3 - 1 = 0.732050807568877` — a set of
`19.63 % +- 0.08 %` of all directions. The repair (where `(1,0,0)` is excluded its own cost is
`>= (2 - sqrt3)/2 = 0.133974596 >` both constants) is *conditional*, not unconditional, and any
write-up must say so.

**(c) Symmetry.** `\|B_3\| = 48`; each element is orthogonal and maps `Z^3` to `Z^3`, so `g` and `d0`
are invariant and admissibility is equivariant under *both* rules. Fundamental domain: the spherical
triangle with vertices `(1,0,0)`, `(1,1,0)/sqrt2`, `(1,1,1)/sqrt3`, interior angles `45/60/90 deg`,
spherical excess `15 deg = 4 pi/48`. Both critical directions have three distinct nonzero coordinates,
hence full 48-element orbits. **Bonus with no 2D content:** on the fundamental domain the
rearrangement inequality gives `min over orbit(w) of g(w',n) = g(sort_desc\|w\|, n)`, which cuts the
98 primitives to **6 sorted representatives** — the step that makes the exact-rational vertex
certificate cheap enough to run.

**(d) Tie loci.** `g(w1,n) = g(w2,n)` is `(w2-w1).n = \|w2\|-\|w1\|` intersected with `S^2`: a circle
at signed plane-distance `h = (\|w2\|-\|w1\|)/\|w2-w1\|` from the origin, of radius `sqrt(1-h^2)`, empty
if `\|h\| > 1`. It is a **great** circle **iff `\|w1\| = \|w2\|`**, which is the exception, not the rule
(`(1,0,0)~(2,1,0)`: `h = 0.874032`, radius `0.4859`; `(2,1,0)~(2,1,1)`: `h = 0.213422`, radius
`0.97696`; `(2,1,0)~(1,2,0)`: `h = 0`, great). CONTEXT_3D's "great-circle-like curve" should be read
as "small circle in general". Census over the 98 primitives: `C(98,2) = 4753` pairs, 0 degenerate,
all meeting `S^2`, collapsing to **2086** distinct planes.

**(e) Primitivity.** `g(kw,n) = k g(w,n)` is an identity in any dimension. Under the **exact** rule
the reduction to primitives is now *unconditional*: if `kw` certifies at depth `eps` then
`eps > k g(w)`, hence `eps + w.n > 2g(w) + w.n = \|w\| >= 1 > r_d` and `eps + w.n <= eps + k w.n <= W`,
so `w` itself is usable and strictly cheaper. This works in any dimension with `r_d < 1`, i.e.
`d <= 3`; it would break at `d >= 4`. Under the **threshold** rule the 2D caveat survives but `4x`
tighter: a multiple rescued by the barrier test costs `>= 2(1 - r_d) = 0.267949192431 = 7.3x eps*`
(2D: `30.5x`). Still not binding.

**(f) Upward-closedness.** `eps*_exa` is *defined* as the sup of the uncovered set, so
"`eps > eps*_exa` implies certified" is exactly what the certified global bound delivers; what fails
is the converse reading. My own count: the uncovered set is not an interval at `8.45 %` of 30 000
random directions (tracks: `8.6 %`, `8.53 %`; a grid-based `7.1 %` is biased). Practical consequence:
**any measurement of `eps*` in 3D that bisects on depth is unsound** — and this actually produced a
wrong published number (section 3.2).

---

## 2. The 3D constants at `W = 3`

### 2.1 Statement

> **THEOREM (both rules).** Lattice `Z^3`, unit spacing, `r_d = sqrt3/2`, `Sigma` a plane with unit
> normal `n` into the exterior, `W = 3`.
>
> ```
> THRESHOLD rule  (w admissible iff r_d < (|w|+w.n)/2 <= W):
>     eps*_thr(3) = ( 3 - sqrt5 - beta )/4 ,      beta  = sqrt( 4 sqrt30 + 2 sqrt5 - 26 )
>                 = 0.036662265118829642294259933043030982319005081536864039123027...
>     n*_thr = ( (sqrt5-1+beta)/2 , (sqrt5-1-beta)/2 , sqrt6-sqrt5 )
>            = ( 0.926675469762340715 , 0.309392507737448981 , 0.213421765283388402 )
>     equivalently   n_x + n_y = sqrt5 - 1 ,  n_z = sqrt6 - sqrt5 ,  |n| = 1
>     TIED WITNESSES: exactly THREE -- (1,0,0), (2,1,0), (2,1,1)
>
> EXACT rule  (w usable at depth eps iff eps > g and r_d < eps + w.n <= W):
>     eps*_exa(3) = ( sqrt2 + sqrt3 - sqrt6 - gamma )/2 ,  gamma = sqrt( 6 sqrt2 + 2 sqrt6 - 13 )
>                 = 0.038443423574719626566412269573292937236509104571086545610762...
>     n*_exa = ( sqrt6-sqrt3 , gamma , sqrt3-sqrt2 )
>            = ( 0.717438935214300805 , 0.619887780009354991 , 0.317837245195782245 )
>     TIED WITNESSES: exactly THREE -- (1,1,0), (1,1,1), (2,1,1)
>
>     ratio  eps*_exa / eps*_thr = 1.048582880793559752
> ```
>
> Both maxima are **attained**, each on the full 48-element `B_3` orbit of its direction (all three
> coordinates distinct and nonzero, so the orbit is free). Both constants are algebraic of degree
> exactly 8 over `Q`.

**The pipeline implements the exact rule.** `eps*_exa` is `4.86 %` larger, and it is the number to
quote. The threshold rule at exactly `W = 3` is *optimistic*, for a reason that is a clean arithmetic
accident (section 2.7).

### 2.2 How the constants were determined, and to what precision

Five genuinely different routes were run, listed in order of logical strength.

| route | what it delivers | status |
|---|---|---|
| A. polytope-vertex certificate in **exact rational** interval arithmetic | rigorous upper bound for explicit rational `t` | **[PROVED]** |
| B. critical-point certificate (a superset of every local-max family, evaluated at the true `F`) | global max + full local-max census | **[CERTIFIED]** |
| C. branch and bound on `S^2` (encloses, never samples) | certified bracket, any `n` | **[CERTIFIED]**, 4 implementations |
| D. exact tie algebra, plus 130- and 250-digit Newton done blind | the closed forms | **[PROVED]** |
| E. end-to-end pipeline simulation on a real `Z^3` lattice, `g` never used | behavioural confirmation | **[CERTIFIED]**, 4 implementations |

**A — the exact-rational certificate.** For a `B_3`-closed witness set `A` put
`P(t) = {x : w.x <= \|w\| - 2t for all w in A}`. `P(t)` is bounded (the `+-e_i` constraints give
`\|x_i\| <= 1-2t`), contains `0`, and `\|x\|^2` is strictly convex, so its maximum over `P(t)` is attained
only at a vertex — a solution of 3 linearly independent constraints. Hence *if no feasible vertex has
`\|v\|^2 >= 1` then `P(t)` misses `S^2`*, i.e. `eps* < t`. The rearrangement lemma (note (c)) cuts the
constraint list to 6 (resp. 5) sorted representatives plus the 3 cone faces, leaving only
`C(9,3) = 84` candidate vertices. Two independent implementations: `audit/a5b_cert.py` +
`a16_cert2.py` (rational enclosures at scale `1e-40`) and `ref_value/r2_certificate.py` + `r22b.py`
(bracket width `3.04e-64`). At `t = eps* + 1e-30`: **0 surviving vertices**. At `t = eps*`: **exactly
48** survivors, one per orbit element — so `{n in S^2 : min_A g >= eps*} = orbit(n*)` exactly.

**B — the critical-point certificate.** At a local max of `F\|_{S^2}` the active set must satisfy
`0 in conv{w_T}` (Danskin), which forces one of: `w_T = 0` (`n = +-w/\|w\|`); two active with
antiparallel `w_T` (a stationary point of `g_w` on a tie circle); or `>= 3` active (a triple-tie
system). Enumerating a superset of each family and evaluating the *true* `F` there is a complete
certificate, because evaluating `F` at a real point can only under-estimate the max.

| family | points evaluated | above the cut |
|---|---|---|
| face-stationary `+-w/\|w\|` | 196 | 0 |
| edge-stationary on 4753 tie circles | 9 506 | 0 |
| triple ties (132 596 solvable triples) | 265 192 | 96, all the same point |
| fundamental-domain boundary | 29 109 | 0 |
| barrier edges (all combinations) | 841 456 | 0 |
| band edges | none exist at `W = 3` | - |

**Exactly one** maximising point. Runner-up critical value `0.036630163` (`3.2e-5` below, at a nearby
saddle `6e-4` away); next *separated* critical value `0.035901725`. Conditioning is benign: the
triple-solve denominator `\|a1 x a2\|^2` is a non-negative **integer**, so it is either 0 (skipped) or
`>= 1` — there is no near-degeneracy.

**C — branch and bound.** Mine (`syn/slib.py`, `syn/v2_bnb.py`; 250 witnesses, best-first, enclosures
of `w.n` from vertex extremes plus a plane-distance lower bound on `\|u\|`) gives, at `tol = 1e-13`:

```
    eps*_thr in [0.0366622651187939, 0.0366622651188939]   (857 cells)
    eps*_exa in [0.0384434235746857, 0.0384434235747857]   (617 cells)
```

Both surds lie inside. Three other implementations (two tracks, one referee) agree. Because this
*bounds* rather than samples, **no angular feature of any width can hide** — which matters, since the
peak's 1 %-loss radius is `0.07 deg`.

**D — exact algebra.** The three tie equations plus `\|n\| = 1` were solved generically by Gaussian
elimination over a multiquadratic field with `Fraction` coefficients, taking only the witness triples
as input. Blind Newton at 130 and 250 digits reproduces the closed forms to `-5.5e-130` / `+2.2e-129`
and `7.4e-260` / `1.2e-260`. 220 digits were certified independently of `Decimal` by exact-rational
Sturm bisection (enclosure width `6.9e-226`).

**E — end-to-end.** Real `Z^3` lattice in `[-L,L]^3`, real plane, true Euclidean UDFs, barrier test
`UDF <= r_d`, band test `UDF <= W`, 6-connected components of the non-barrier in-band points seeded
from the exterior, then the **raw** closure `UDF(A) + UDF(B) - dist > tol` to fixpoint by a worklist.
**The formula `g` appears nowhere in the simulator.** A blind 45-digit `Decimal` bisection at `n*_exa`
(bracket `[1e-30, 0.5]`) agrees with the closed form to **45 significant digits** (`4e-46`). With
`tol = 1e-13` the measured critical depth is `eps* + 5.00e-14 = eps* + tol/2` exactly — the guard band
is conservative. Bit-identical for box half-widths `L = 5..16` and margins 4 and 6. *(Caveat: a bare
bisection on depth is the unsound method of section 3.2; here it agreed with four other routes, but
the descending-scan variants are the ones to trust.)*

**Attainment (lower bound) is exact, not numerical.** At `n*_thr`, exhaustive enumeration over
102 322 lattice vectors (`\|w\| <= 30`) in 60-80 digit arithmetic finds exactly three admissible
witnesses at cost `eps*_thr` and **nothing** cheaper; the next-cheapest admissible cost is *exactly*
`2 eps*_thr`, at `(2,0,0) = 2(1,0,0)`. At `n*_exa` (exhaustive to `\|w\|^2 <= 400`) **zero** witnesses
have `eps*_exa` inside their exact-rule window; at `eps*_exa (1 + 1e-30)` exactly three appear, and
the runner-up usable cost is again exactly `2 eps*_exa`, at `(2,2,0) = 2(1,1,0)`. The same factor-2
isolation as 2D.

**Search found nothing above either constant.** A 1 200 000-direction Fibonacci sweep of the full
sphere: 0 exceedances (max `0.036638349` / `0.038425770`, deficits `2.4e-5` / `1.8e-5` — exactly what
a grid does against a kink whose 1 %-loss radius is `0.07 deg` while the sweep spacing is `0.18 deg`).
75 640 blind end-to-end trials: 0 soundness violations. A 70 000-direction adversarial hunt
concentrated log-uniformly at radius `1e-11..0.2` around both critical directions: closest approaches
`-6.8e-13` and `-6.4e-13`, nothing above. 150 annealing chains all reached the same maximum.

**Search-radius stability.** `eps*` is bit-identical for witness sets `\|w\|^2 <= 9, 14, 25, 49, ...,
256` (122 to 17 076 vectors), and pointwise `f_r(n) = f_{3.1}(n)` to `0.000e+00` at 40 000 Fibonacci
directions. My own B&B reproduces this (`syn/v2_bnb.py`). The finite reduction is exact, not a
heuristic.

### 2.3 The critical configurations

**Threshold rule, `n*_thr`** (60-digit exhaustive anatomy):

| `w` | `\|w\|` | `theta_w` | `w.n` | `d0` | `g - eps*` (residual) | `\|p\|` |
|---|---|---|---|---|---|---|
| `(1,0,0)` | 1 | `22.077626918 deg` | `0.926675469762` | `0.963337734881` | `0E-80` | `0.375862440` |
| `(2,1,0)` | `sqrt5` | `14.713423848 deg` | `2.162743447262` | `2.199405712381` | `0E-80` | `0.567926739` |
| `(2,1,1)` | `sqrt6` | `14.054449106 deg` | `2.376165212545` | `2.412827477664` | `0E-80` | `0.594843578` |

All three are non-barrier with margin `>= 0.0973` and in band with margin `>= 0.5872`. Barycentric
weights of `0` in `conv{w_T}`: `(0.498783334757, 0.155472851768, 0.345743813475)`, all strictly
positive. Tangent-ball identity `\|p\| = 2 sqrt(d0 eps)` holds to `0.00e+00`.

**Exact rule, `n*_exa`:**

| `w` | `\|w\|` | `theta_w` | `w.n` | `eps* + w.n` | `g - eps*` (residual) | `\|p\|` |
|---|---|---|---|---|---|---|
| `(1,1,0)` | `sqrt2` | `18.979889671 deg` | `1.337326715224` | `1.375770138798` | `5E-121` | `0.459953538` |
| `(1,1,1)` | `sqrt3` | `17.135750518 deg` | `1.655163960419` | `1.693607383994` | `5E-121` | `0.510325645` |
| `(2,1,1)` | `sqrt6` | `14.393567252 deg` | `2.372602895634` | `2.411046319208` | `5E-121` | `0.608896953` |

Barycentric weights `(0.487266477182, 0.355364466110, 0.157369056708)`.

**The peak is a conic point, not a smooth maximum.** Envelope slopes at `n*_exa` range over
`0.0514 .. 0.3044` per radian depending on tangential direction. Sharpness (threshold rule, worst
tangential direction; my own recomputation, `syn/v5_misc.py`):

| loss of `eps*` | misorientation needed (3D, worst direction) | 2D reference |
|---|---|---|
| 1 % | `0.070802 deg` | `0.0802 deg` |
| 5 % | `0.357594 deg` | `0.4009 deg` |
| 10 % | `0.724589 deg` | `0.8017 deg` |

### 2.4 Closed forms, minimal polynomials, and the degree

Both constants are the **smaller** root of a quadratic over a real biquadratic field:

```
    eps*_thr = smaller root of  4x^2 - 2(3 - sqrt5) x + (10 - 2 sqrt5 - sqrt30) = 0   over Q(sqrt5,sqrt6)
    eps*_exa = smaller root of   x^2 -  (sqrt2+sqrt3-sqrt6) x + (6 - 3 sqrt2 - sqrt3) = 0
                                                                                over Q(sqrt2,sqrt3)
```

**[PROVED]** minimal polynomials, both of degree exactly 8, both **irreducible over `Q`**:

```
    eps*_thr :  64x^8 - 384x^7 + 1344x^6 - 2944x^5 + 4624x^4 - 5040x^3 + 3400x^2 - 800x + 25
    eps*_exa :     x^8          +   2x^6 +  12x^5 +   79x^4 -  228x^3 +  474x^2 -  252x +  9   (monic)
```

Irreducibility is **proved**, not a numerical certificate (this closes a phase-1 open item): `D` is
not a square in `K` (a Galois conjugate of `D` is negative in a real field), so `L = K(sqrt D)` has
degree 8; and the exact `8 x 8` determinant of the coordinates of `1, eps, ..., eps^7` in the `Q`-basis
of `L` is nonzero — `1009375/262144` and `62647344` respectively. Hence `deg(eps*) = 8` and
`Q(eps*) = L`. Both splitting fields have `\|Gal\| = 64` (proved by an exact square-class computation;
Chebotarev-consistent over 3 240 primes `< 30000`). `eps*_exa` is an algebraic integer; `eps*_thr` is
not, but `2 eps*_thr` is.

*Worth recording:* the standard mod-`p` (Dedekind) irreducibility test **provably cannot** settle
these octics — over all primes `< 2000` every factorisation-degree pattern admits a sub-multiset
summing to 4, so the achievable rational-factor degrees never shrink below `{0,4,8}`. Anyone
retracing the phase-1 open item will otherwise waste time there.

**Why 8 and not 4 — a rule that also predicts where 16 appears.** The rule itself is a
generic statement; each row of the table below is **[PROVED]** individually by the same exact
machinery (null-space minimal polynomials over the relevant multiquadratic field).

> For a `d`-fold tie in dimension `d` with witnesses `w_1..w_d`, put `K = Q(\|w_1\|,...,\|w_d\|)`, a
> multiquadratic field of degree `2^r`. Then `eps` lies in `K(sqrt D)` and `deg(eps) \| 2^(r+1)`,
> with equality generically.

| configuration | tie | `K` | deg |
|---|---|---|---|
| 2D `eps_c` | `(1,0),(2,1)` | `Q(sqrt5)` | 4 |
| 2D Borgefors `3x3` | `(1,0),(1,1)` | `Q(sqrt2)` | 4 |
| **3D threshold global max** | `(1,0,0),(2,1,0),(2,1,1)` | `Q(sqrt5,sqrt6)` | **8** |
| **3D exact-rule global max** | `(1,1,0),(1,1,1),(2,1,1)` | `Q(sqrt2,sqrt3)` | **8** |
| 3D 2nd threshold local max | `(1,1,0),(2,1,0),(2,1,1)` | `Q(sqrt2,sqrt3,sqrt5)` | **16** |
| 3D 3x3x3 mask constant | `(1,0,0),(1,1,0),(1,1,1)` | `Q(sqrt2,sqrt3)` | 8 |

So the worry "the degree may be much higher in 3D" is justified in general — the runner-up is degree
16 — but the two constants that matter are the algebraically *simplest* triples available, because
their norms use only two independent surds.

### 2.5 Precision, and how to hard-code the number

```
    eps*_exa correctly rounded to double = 0x1.3aedb3dffd82ep-5 = 0.038443423574719623
       -> the nearest double lies 3.107e-18 BELOW the true value.  As an upper bound it is UNSAFE.
       conservative: next double up, 0x1.3aedb3dffd82fp-5 = 0.03844342357471963
    eps*_thr nearest double 0x1.2c5657b5d84f0p-5 lies 2.285e-18 BELOW; use 0.036662265118829647

    decimal: round(eps*_exa, 17) = 0.03844342357471963  is ABOVE the true value  -> safe
             round(eps*_exa, 16) = 0.0384434235747196   is BELOW                -> unsafe
             round(eps*_thr, 19) = 0.0366622651188296423                        -> safe
```

Conditioning of the algebraically equivalent surd forms in IEEE double (1 ulp `= 6.94e-18` here):

| route | error |
|---|---|
| `(3 - sqrt5 - beta)/4` | `-16.3 ulp` |
| `(10 - 2sqrt5 - sqrt30)/(3 - sqrt5 + beta)` | `-29.3 ulp` |
| `(sqrt2 + sqrt3 - sqrt6 - gamma)/2` | `-6.5 ulp` |
| `2(6 - 3sqrt2 - sqrt3)/(sqrt2 + sqrt3 - sqrt6 + gamma)` | `-94.5 ulp` |

The plain nested-radical form is the best conditioned, but none is accurate to the last bit. **Do not
compute these at runtime; hard-code the rounded-up decimal.** This is exactly the 2D trap, one order
of magnitude worse.

### 2.6 The whole envelope, not just its maximum

**Local-max census (threshold rule, fundamental domain).** **[CERTIFIED]** — there are exactly
**three** local maxima above `0.002`, established twice independently (a branch-and-bound
superlevel-set enumeration refined from every surviving cell at `T = 0.010 / 0.005 / 0.002`, and a
triple-tie enumeration over `C(98,3)` with a strict "0 in the interior of `conv{p_k}`" test):

| # | value | direction | ties | closed form |
|---|---|---|---|---|
| 1 | `0.036662265118829642` | `(0.926675470, 0.309392508, 0.213421765)` | `(1,0,0)(2,1,0)(2,1,1)` | `(3-sqrt5-beta)/4` |
| 2 | `0.032074258246890076` | `(0.821854415, 0.528210631, 0.213421765)` | `(1,1,0)(2,1,0)(2,1,1)` | `(2sqrt2-sqrt5-sqrt(2sqrt10+2sqrt30-17))/2` |
| 3 | `0.018614733452036116` | `(0.717438935, 0.550510257, 0.426872148)` | `(1,1,1)(2,1,1)(2,2,1)` | `(2sqrt3-3-sqrt(6sqrt2+6sqrt6-23))/2` |

Maxima 1 and 2 both lie on `n_z = sqrt6 - sqrt5`, which is exactly the `(2,1,0)/(2,1,1)` tie plane:
they are the two ends of one ridge. The **exact-rule** census (4 maxima: `0.038443, 0.036662,
0.032074, 0.018615`) rests on search only — the exact-rule envelope is discontinuous, so a very narrow
peak cannot be excluded by search; only the global bound is certified. **[HIGH-NUM]**

Mirror-edge structure (all saddles in 3D):

| edge | value | closed form | ties |
|---|---|---|---|
| `n_y = n_z` | `0.034938388835508` | `(4 - sqrt6 - 2sqrt(sqrt6-2))/6` | `(1,0,0)(2,1,1)` |
| `n_z = 0` | `0.019202630763796` | **exactly the 2D constant `eps_c`**, at exactly the 2D critical direction | `(1,0,0)(2,1,0)` |
| `n_x = n_y` | `0.014316063476010` | `(5sqrt2 - 6 - 2sqrt(3sqrt2-4))/6` | `(1,1,0)(2,2,1)` |

So the entire 2D solution survives verbatim on the mirror face `n_z = 0` — and is a *saddle* there,
a factor 1.909 / 2.002 below the 3D answers. Hill climbing from it, and from every lift of it into
other lattice planes, lands on the 3D maxima.

**Which witness actually wins, by solid angle** (threshold rule; my Monte-Carlo over 200 000 uniform
directions, `syn/v5_misc.py`, agreeing with a referee's deterministic quadrature to `0.2 pp`; this
**replaces** a phase-1 census that was wrong in every entry, see finding V7):

```
    (2,1,0) 24.6 %   (2,1,1) 23.6 %   (1,0,0) 15.8 %
    (1,1,0) 15.6 %   (2,2,1) 14.0 %   (1,1,1)  6.5 %
```

The *set* of six winners is the 3D analogue of 2D's "three witnesses per octant". Under the exact
rule the `(2,2,1)` cell disappears and five remain.

### 2.7 Why the two rules disagree in 3D but not in 2D — the `(2,2,1)` mechanism

This is the substantive new phenomenon, and it is a clean arithmetic fact.

> A witness with `\|w\| = W` **exactly** has exact-rule window `(g, W - w.n] = (g, 2g]`, because
> `W - w.n = \|w\| - w.n = 2g`. It can never certify a target deeper than `2g`.
>
> At `W = 3` in `Z^3` the vector `(2,2,1)` is **primitive** with `\|w\| = 3 = W` exactly. In `Z^2`
> there is no primitive vector of norm 3 (`9 = 9 + 0` only), so this case simply does not occur in 2D.

Under the threshold rule `(2,2,1)` is admissible for every `n` (`d0 = (3 + w.n)/2 <= 3`), and it is
*essential* to the threshold bound — dropping it raises the max-min to exactly `eps*_exa`. Under the
exact rule its window is a bounded sliver that shrinks to nothing as `g -> 0`, so near the
`(2,2,1)/3` direction the exact rule gets no help from it at all.

Concretely, at `n*_exa` the **cheapest** witness is `(2,2,1)`, with `g = 0.003754662178453` — ten
times cheaper than the tie — but its usable window is only `(0.003754662178, 0.007509324357]`.
The uncertified set there is therefore

```
    ( 0, 0.003754662178453 ]  u  ( 0.007509324356906, 0.038443423574720 ]      -- NOT an interval
```

and the exact-rule constant is the sup of the second piece. Formally **[PROVED]**:

```
    A(n) := min{ g(w,n) : w primitive, |w|^2 <= 6 }          (74 vectors)
    eps*_exa(n) <= A(n)  for every n ,   and   max_n eps*_exa = max_n A = eps*_exa(3).
```

*Proof sketch (referee-strengthened; the phase-1 version only checked equality at the maximiser).*
Since `max_n A = 0.0384434`, the `A`-minimiser `w0` has `w0.n = \|w0\| - 2A >= 1 - 0.0769 = 0.9231 >
r_d`, so its window bottom is `g` and its top is `min(r_d, 3 - w0.n) >= 3 - sqrt6 = 0.5505`. The axis
witness `e_i` with `\|n_i\| >= 1/sqrt3` has window bottom `<= 0.2887 < 0.5505` and top `r_d`. The two
overlap, so `(A(n), r_d]` is always covered and `sup(uncovered) <= A(n)` everywhere. QED

I re-checked the inequality `F_exa(n) <= A(n)` on 40 000 random directions: **0 violations**
(`syn/v8`).

**Practical reading.** The pipeline's true worst case at `W = 3` is `0.0384434235747196266`. The
threshold rule is optimistic here because it wrongly credits `(2,2,1)` with unlimited usability.

### 2.8 The minimal covering set is six (threshold) / five (exact)

Max-min over the fundamental domain with one class deleted (my own certified B&B, `syn/v5_misc.py`;
this **corrects** the phase-1 `drop (1,1,0)` row, finding V8):

| dropped | max-min (certified, `+-1e-10`) | ratio to `eps*_thr` |
|---|---|---|
| `(1,0,0)` | `0.118033988749895` | 3.2195 |
| `(1,1,0)` | `0.059160352656443` | 1.6137 |
| `(1,1,1)` | `0.056624327025935` | 1.5445 |
| `(2,1,0)` | `0.049958329422190` | 1.3627 |
| `(2,1,1)` | `0.072326379713903` | 1.9728 |
| `(2,2,1)` | `0.038443423541913` | 1.0486 (`= eps*_exa`) |
| none (all six) | `0.036662265118830` | 1.0000 |

Every 5-subset is strictly worse, so six is minimal under the threshold rule; longer vectors cannot
help (an in-band `\|w\| > 3` costs `>= sqrt10 - 3 = 0.162277660`), and non-primitives are dominated.
Under the exact rule the last row shows `(2,2,1)` contributes nothing, so five classes (74 vectors)
suffice — confirmed independently by instrumenting a covering certificate: only
`(1,0,0),(1,1,0),(1,1,1),(2,1,0),(2,1,1)` are ever used, at most 3 per patch. **A `W = 3`
implementation never needs to look beyond `\|w\| <= sqrt6 = 2.449`.**

### 2.9 Lemma M: multi-hop enrichment provably cannot help (new; closes a phase-1 open item)

Three of the four phase-1 tracks left "multi-hop chaining" as empirical-only. It is provable, in one
line, and the argument is dimension-free.

> **LEMMA M.** Let the closure start from the set of non-barrier in-band exterior lattice points
> (all certified by plain connected components) and add points by the pairwise rule. Any point added
> by enrichment that was not already certified is a **barrier** point, hence has `UDF(u) <= r_d`.
> Certifying a target at depth `eps` from such a `u` requires `eps + UDF(u) > dist = \|u\| >= 1`
> (distinct lattice points, unit spacing), i.e. `eps > 1 - r_d`.
>
> Therefore **no target at depth `<= 1 - r_d` is ever certified by anything but a first-hop,
> non-barrier witness.** In 3D `1 - r_d = 0.13397459621556135`, and both constants are far below it,
> so the single-hop analysis is complete at `W = 3`. **[PROVED]**

Consistent with everything measured: the end-to-end closures converge in 2 passes and reproduce both
constants; allowing newly certified barrier points as witnesses changes `eps*` at 0 of 451+250
sampled directions. Lemma M also tells you exactly *when* multi-hop starts to matter: once `eps*`
climbs above `0.13397`, which happens only for concave radii `R <~ 2.6` (section 4.4) — precisely
where the curvature track flagged its two small-`R` rows as upper bounds.

Caveat: Lemma M assumes the exterior non-barrier in-band set is fully seeded. At `W = 3` that set was
a **single 6-connected component** in every one of ~78 000 trials, and a component provably cannot
straddle the interface (6-neighbours differ in `s` by `\|n_i\| <= 1`, while a straddling non-barrier
pair differs by `> 2 r_d = 1.732`). It fragments for narrower bands: `W = 1.0` up to 94 components,
`W = 1.4` up to 313, `W = 2.0` one trial in 360 with 2, `W >= 2.2` always 1. **A pipeline that seeds
from a single known-outside point rather than from every positive component silently loses fragments
below `W ~ 2.2`.** **[HIGH-NUM]**

---

## 3. The `W`-plateau in 3D, both rules

`eps*(.,W)` is non-increasing in `W` under both rules **[PROVED]** (threshold: admissible sets are
nested; exact: every window grows). All plateau endpoints below were located by bisecting a monotone
predicate on a certified bracket, then pinned by closed forms.

### 3.1 The two plateaus containing `W = 3`

```
THRESHOLD rule
    eps*_thr = 0.0366622651188296423   for   W in [ W_lo , W_hi ]
    W_lo = (18 + 10 b + sqrt(24 - 8 b^2))/12 ,   b = sqrt3 - 2 eps*_thr = 1.658726277331218
         = 2.999798995571213626345506931224
    W_hi = (sqrt10 + 2 sqrt5 - 2 + beta)/2  =  d0_thr( (3,1,0), n*_thr )
         = 3.125848288596425229615513740496      <-- LITERALLY the 2D formula with beta2 -> beta3

EXACT rule
    eps*_exa = 0.0384434235747196266   for   W in [ W_lo' , W_hi' )
    W_lo' = sqrt3 - 1/2 + (3/2) sqrt(sqrt3 - 1)
          = 2.515450323319905582981299990822636930707572325346574438021
          attained at n_c = ( sqrt(sqrt3-1), (sqrt3-1)/2, (sqrt3-1)/2 )
                          = (0.855599677167352192, 0.366025403784438646, 0.366025403784438646)
    W_hi' = sqrt2 + sqrt3 - 3 eps*_exa = (3 sqrt6 - sqrt2 - sqrt3 + 3 gamma)/2
          = 3.030934099217813462629898256996  =  eps*_exa + (2,2,1).n*_exa
```

All four endpoints **[CERTIFIED]**, each by at least three routes (monotone-predicate bisection of a
certified bracket, an explicit critical configuration solved in closed form, and either an end-to-end
lattice simulation or an independently written scratch evaluator). I re-confirmed all four myself
(`syn/v3_plateau.py`, `syn/v6`).

**New here:** `W_lo'` is algebraic of degree exactly **4**, with minimal polynomial

```
    16 x^4 + 32 x^3 - 448 x - 23          (residual 0E-76 at 80 digits)
```

**[PROVED]** by exact rational linear algebra in `Q(sqrt(sqrt3-1))` — the null-space system has a
unique solution, so `1, x, x^2, x^3` are independent and the quartic is the minimal polynomial. This
closes an open item carried by both the `plateau` and `closedform` tracks (`syn/`, this write-up).

**What sets each endpoint:**

* `W_hi` (threshold): `(3,1,0)` **enters the band** at `n*_thr`. It costs `g = 0.036429371572` there
  — only `0.635 %` cheaper than `eps*_thr` — so crossing `W_hi` buys almost nothing: a ramp of width
  `8.506e-5` down to `0.036619733596489`, a `0.116 %` improvement.
* `W_lo` (threshold): `(2,2,1)` **leaves the band**. Below `W_lo` the transition is **continuous**,
  with two branches: `eps*(W) = (3 + sqrt2 + sqrt3 - 2W)/4` (slope exactly `-1/2`) on
  `[2.996245337822, 2.999798546802)`, then a steep branch of slope `-10.148` up to `W_lo`. Below
  `2.996245337822 = d0_thr((2,2,1), n*_exa)` the threshold constant equals `eps*_exa`.
  I verified the linear ramp to 9 digits at four values of `W` (`syn/v6`): `W = 2.9965 ->
  0.038316092446..546` vs formula `0.038316092485`; `W = 2.999 -> 0.037066092464..564` vs
  `0.037066092485`.
* `W_hi'` (exact): `(2,2,1)`'s bounded window grows until its **top reaches `eps*_exa`**, closing the
  gap. Above it, `eps*(W) = (sqrt2 + sqrt3 - W)/3` exactly — a linear ramp of slope `-1/3`, width
  `5.32981e-3` — down to `sqrt2/2 - sqrt(sqrt6-2) = 0.036666819084662` at
  `W = sqrt3 - sqrt2/2 + 3 sqrt(sqrt6-2) = 3.036263912688`, where the branch ceases to exist
  geometrically and `eps*` **jumps down** by `4.554e-6` onto the `eps*_thr` plateau. Verified
  (`syn/v3b`): `W = 3.031 -> 0.038421456` vs formula `0.038421457`; `W = 3.036 -> 0.036754790` vs
  `0.036754790`.
* `W_lo'` (exact): at `n_c` the witnesses `(1,0,0)` and `(1,1,1)` tie at
  `A = (1 - sqrt(sqrt3-1))/2 = 0.072200161416324`, and `(2,1,1)` covers `(g, W - (2,1,1).n]`. The gap
  is open iff `W < A(n) + (2,1,1).n`, maximised on `n_y = n_z`. Crossing `W_lo'` downward `eps*`
  **jumps by a factor 1.8781**, to `0.0722001614163239`.

**A clean identity linking the two rules** (verified to `1e-15`): for a plateau of value `V` with
maximiser `n_V` and cheapest spoiler `w`,

```
    W_hi(threshold) = d0_thr(w, n_V) = (|w| + w.n_V)/2
    W_hi(exact)     = V + w.n_V
    W_hi(exact) - W_hi(threshold) = V - g(w, n_V)  >  0
```

so the exact rule always holds each plateau **longer at the top** and starts it **later at the
bottom**. At `W = 3` the two rules therefore sit on *different* plateaus — that, and not any subtlety
of the constant, is the whole reason they disagree in 3D.

### 3.2 A phase-1 error that a referee caught, and I reproduced

Two phase-1 tracks reported the exact-rule plateau floor as `2.5153786883076` ("e2e", with a claimed
bracket of width `1.4e-14`) and `~2.5151853` ("certificate"). **Both are false**, by `7.2e-5` in `W`
— i.e. the e2e bracket is wrong by ~`5e9` of its own stated widths. My independent check
(`syn/v3_plateau.py`), at the direction `n_c` above:

| `W` | `F_exa(n_c)` | certified global max |
|---|---|---|
| `2.5151853` (certificate track's floor) | `0.072200161416` | `[0.072288502522, 0.072288502532]` |
| `2.5153786883076` (e2e track's floor) | `0.072200161416` | `[0.072224039752, 0.072224039762]` |
| `W_lo' - 1e-9` | `0.072200161416` | `[0.072200161743, 0.072200162009]` |
| `2.52` | `0.003119790440` | `[0.038443423515, 0.038443424515]` |
| `2.6, 2.9, 3.0` | `0.003119790440` | `[0.038443423515, 0.038443424515]` |

At both claimed floors the true value is `1.878 x` the claimed plateau constant. Both referees found
this independently, one of them confirming it end-to-end on a real lattice (uncertified at depth
`0.0722133595`, certified at `0.0722133739`). **The root cause is a method error worth remembering:
bisecting on the target depth, when the uncovered depth set is not an interval** — exactly the
failure mode the same track documented elsewhere in its own report.

### 3.3 The certified map over `W in [1, 7]`

Values are exact roots of the stated triple-tie systems (40-80 digits) and lie inside certified
`1e-12` brackets. I spot-checked 16 rows myself with an independent branch-and-bound
(`syn/v6`, `syn/v7`): `W = 1.15, 1.1547, 1.5, 1.85, 1.866, 1.87, 2.0, 2.52, 2.6, 2.9, 3.0, 3.13,
3.5, 4.0, 4.5, 5.0, 6.0` — every one inside the claimed plateau. **[CERTIFIED]**

Consecutive plateaus are separated by transitions, and only the two bounding the `W = 3` plateaus are
listed explicitly below; the omitted gaps are narrow (`8.5e-5` to `5.3e-3` wide) and are classified
as follows. `eps*(W)` is **neither a pure step function nor continuous**: over `[0.9, 7.2]` the
threshold rule has 3 pure jumps (at `(sqrt2+2/sqrt3)/2`, `sqrt2`, `sqrt3` — each a witness entering
the band at its own direction, where `d0 = \|w\| = W`), 10 pure ramps and 2 ramp-then-jump; the exact
rule has 3 jumps, 3 ramps and 4 ramp-then-jump. **3D is much rampier than 2D**, which was mostly
jumps. On a ramp the supremum is not attained (the maximiser sits on the open side of a band-edge
surface); the same is true at the upper endpoint of every plateau, so each plateau is a *maximum* on
its half-open interval and a *supremum* on the closed one. **[CERTIFIED]** near `W = 3`,
**[HIGH-NUM]** elsewhere.

**Threshold rule:**

| `eps*` | plateau in `W` | tied witnesses |
|---|---|---|
| `r_d = 0.866025403784439` — **no guarantee** | `W < 2/sqrt3 = 1.154700538379252` | none admissible |
| ramp `0.7545144 -> 0.6306096` | `(2/sqrt3, 1.2544355486)` | |
| `1/sqrt3 = 0.577350269189626` | `[1.2544355486, 1.2844570504)` | `(-1,1,1)(1,-1,1)(1,1,-1)` |
| `sqrt2/4 = 0.353553390593274` | `[1.2844570504, sqrt2)` | `(0,1,+-1)(1,0,+-1)` |
| `sqrt2/2 - 1/sqrt3 = 0.129756511996922` | `[sqrt2, sqrt3)` | `(0,1,1)(1,0,1)(1,1,0)` |
| `0.073559321149897` (the `3x3x3` constant) | `[sqrt3, 2.1780221276)` | `(1,0,0)(1,1,0)(1,1,1)` |
| `0.072326379719341` | `[2.1804880080, 2.4461175157)` | `(1,0,0)(1,1,1)(2,1,0)` |
| `0.038443423574720` | `[sqrt6 = 2.449489742783, 2.996245337822)` | `(1,1,0)(1,1,1)(2,1,1)` |
| ramp `(3+sqrt2+sqrt3-2W)/4`, then a steep branch | `[2.996245337822, 2.999798995571)` | band-edge configurations |
| **`0.036662265118830`** | **`[2.999798995571, 3.125848288596]`** | `(1,0,0)(2,1,0)(2,1,1)` |
| `0.036619733596489` | `[3.1259333493, 3.3098177994]` | `(1,0,0)(2,1,1)(3,1,0)` |
| `0.032074258246890` | `[3.3141721917, 3.7385318295]` | `(1,1,0)(2,1,0)(2,1,1)` |
| `0.022218963297108` | `[3.7416569969, 4.2351948121]` | `(1,0,0)(3,1,0)(3,1,1)` |
| `0.022058734636762` | `[4.2355152670, 4.5817872842]` | `(2,1,0)(3,1,1)(3,2,1)` |
| `0.019127139840725` | `[4.5824013795, 5.4747882239]` | `(3,1,0)(4,1,1)(4,2,1)` |
| `0.015128624095725` | `[5.4763161023, 5.9147627519]` | `(2,1,0)(4,2,1)(4,3,1)` |
| `0.014298388986994` | `[5.9151542141, 6.4006505158]` | `(3,1,0)(4,1,1)(5,2,1)` |
| `0.012339715139592` | `[6.4015640014, 7.3453660812]` | `(4,1,0)(5,1,1)(6,2,1)` |

**Exact rule:**

| `eps*` | plateau in `W` | tied witnesses |
|---|---|---|
| `r_d` — **no guarantee at all** | `W < 1 + r_d = 1.866025403784439` | — |
| `0.073559321149897` | `[1 + r_d, 2.1935355989)` | `(1,0,0)(1,1,0)(1,1,1)` |
| `0.072326379719341` | `[2.1972344195, 2.5150716684)` | `(1,0,0)(1,1,1)(2,1,0)` |
| **`0.038443423574720`** | **`[2.515450323320, 3.030934099218)`** | `(1,1,0)(1,1,1)(2,1,1)` |
| linear ramp `(sqrt2+sqrt3-W)/3`, then a `4.554e-6` jump | `(3.030934099218, 3.036263912688)` | |
| `0.036662265118830` | `[3.0362639127, 3.1260811821)` | `(1,0,0)(2,1,0)(2,1,1)` |
| `0.036619733596489` | `[3.1262087733, 3.3396305420]` | `(1,0,0)(2,1,1)(3,1,0)` |
| `0.032074258246890` | `[3.3446745744, 3.7674805304]` | `(1,1,0)(2,1,0)(2,1,1)` |
| `0.022218963297108` | `[3.7702462712, 4.2499679005]` | `(1,0,0)(3,1,0)(3,1,1)` |
| `0.022058734636762` | `[4.2504485827, 4.6030576080]` | `(2,1,0)(3,1,1)(3,2,1)` |
| `0.019127139840725` | `[4.6030576080, 5.4914780127)` | `(3,1,0)(4,1,1)(4,2,1)` |
| `0.015128624095725` | `[5.4914865 +- 3.2e-7, 5.9285743448]` | `(2,1,0)(4,2,1)(4,3,1)` |
| `0.014298388986994` | `[5.9285743 +- 1.3e-8, 6.4124751832]` | `(3,1,0)(4,1,1)(5,2,1)` |
| `0.012339715139592` | `[6.4124828 +- 1.2e-7, 7.3546026492]` | `(4,1,0)(5,1,1)(6,2,1)` |

**Every plateau maximiser in the whole range is a triple tie.** CONTEXT_3D's structural prediction is
not a `W = 3` accident.

Three phase-1 large-`W` figures were grid artifacts and are **replaced** by the certified values
above: `W = 2.99980 -> 0.0366519` (impossible, since `eps*` is non-increasing and equals `0.0366623`
at `W = 3`), `W = 3.5 -> 0.03204` (true `0.0320742582`), `W = 6 -> 0.01419` (true `0.0142983890`). My
own B&B reproduces the certified values at `W = 3.5, 4, 4.5, 5, 6` (`syn/v6`, `syn/v7`).

### 3.4 Where the scheme gives no guarantee at all

```
    threshold rule:  eps*(W) = r_d  for  W < 2/sqrt3 = 1.154700538379251529
    exact rule:      eps*(W) = r_d  for  W < 1 + r_d = 1.866025403784438647   (identical to 2D)
```

* **Threshold, `2/sqrt3`.** The obstruction is the body diagonal `n = (1,1,1)/sqrt3`: there `(1,0,0)`
  has `d0_thr = (1 + 1/sqrt3)/2 = 0.7887 <= r_d` (barrier-excluded), and the first witnesses to become
  admissible are `(1,1,-1)` and its two mirrors, at `d0_thr = (sqrt3 + 1/sqrt3)/2 = 2/sqrt3`.
  Exhaustive: 0 admissible witnesses below, exactly 3 at that `W`.
* **Exact, `1 + r_d`.** At `n = (1,0,0)` a target at depth `eps` needs a witness with integer
  `w.n = m >= 1` and `eps + m <= W`; worst case `eps -> r_d`. Confirmed end-to-end.

I confirmed both boundaries independently (`syn/v7`): at `W = 1.1547005384` the threshold value drops
from `r_d` to `0.7545144`; at `W = 1.8660254038` the exact value drops from `r_d` to `0.0735593211`.

**Consequence worth stating loudly:** on `[2/sqrt3, 1 + r_d)` the threshold rule reports finite
constants (`0.577`, `0.354`, `0.130`, `0.0736`) while the true answer is still *no guarantee at all*.
**The threshold rule is optimistic by an unbounded factor for narrow bands.**

### 3.5 Brittleness of `W = 3`

```
THRESHOLD rule   plateau [2.999798995571, 3.125848288596]
    margin below W = 3 :  2.0100442878e-4  =  0.0067 % of W        <-- KNIFE EDGE
    margin above W = 3 :  0.1258482886     =  4.195  % of W
    falling off the bottom: eps* rises continuously to 0.0384434235747 (+4.86 %) by W <= 2.996245
    going over the top    : eps* falls to 0.0366197335965 (-0.116 %) by W = 3.12593

EXACT rule       plateau [2.515450323320, 3.030934099218)
    margin below W = 3 :  0.4845496767     = 16.152  % of W
    margin above W = 3 :  0.0309340992     =  1.031  % of W
    falling off the bottom: a JUMP by factor 1.8781 to 0.0722001614163
    going over the top    : eps* falls along (sqrt2+sqrt3-W)/3 to 0.0366622651 (-4.63 %) by W = 3.0363
```

Both edges are driven by the same object: `(2,2,1)` having `\|w\| = 3 = W` exactly.

**The threshold-rule figure of `0.0366623` at exactly `W = 3` is a knife-edge accident** — any
effective shrinkage of the band by more than `2.01e-4` voxels moves it up. The number the pipeline
actually delivers (`eps*_exa`) is **not** brittle downward (16 % of `W` of slack) but has only
`1.031 %` of headroom upward.

*A phase-1 track reported the downward margin as `0.003755` (`0.125 %` of `W`), on the theory that
the threshold rule jumps at `2.996245`. It does not — the transition is a continuous ramp. The true
margin is `18.7x` smaller. My `W`-scan (`syn/v6`) confirms the ramp and the floor.*

**Design reading.** If you want the better constant `0.0366622651` under *both* readings with real
margin, use `W ~ 3.08`: the threshold plateau then has `2.60 % / 1.49 %` margins and the exact
plateau `1.42 % / 1.50 %`, for a `4.63 %` gain in guaranteed depth. If you want maximum robustness at
the current constant, `W ~ 2.77` (the centre of the exact-rule plateau) gives `+-9.3 %` of `W` of
slack. Do **not** shrink `W` below `~2.2` (seed fragmentation, section 2.9) and do not expect much
from `W = 3.13`.

### 3.6 One convention that matters and one that does not

* **The band test `<= W` vs `< W` does NOT change either constant at `W = 3`.** A phase-1 track
  recommended forcing the non-strict form on the grounds that `(2,2,1)` attains `d0_thr = W` exactly.
  It does — but only at the single tangency point `n = (2,2,1)/3`, where `g((2,2,1),n) = 0`, so
  deleting the witness there lifts `F` from 0 to `0.0326920705`, still far below the maximum. My
  certified B&B with the strict test gives `eps*_thr in [0.0366622651172, 0.0366622651242 + 1e-11]`
  — the same constant (`syn/v2_bnb.py`), reproducing a referee's refutation and a phase-1
  cross-check that had reached the same conclusion. **[CERTIFIED]**
* **The guarded predicate `UDF(A)+UDF(B)-dist > tol` is mandatory.** See section 8.

---

## 4. Curvature in 3D

### 4.1 The exact sphere law is dimension-free

For `Sigma` a sphere of radius `R`, with the **planar** quantities `q = w.n`, `g = (\|w\|-q)/2`,
`d0 = (\|w\|+q)/2`:

```
    concave exterior (exterior = inside the ball) :  c(w,n) = R g/(R - d0) = g / (1 - d0/R)
    convex  exterior (exterior = outside the ball):  c(w,n) = R g/(R + d0) = g / (1 + d0/R)
```

**[PROVED]** and re-derived from scratch. The structural reason it is identical to 2D: only
`\|v0 - C\|` enters, and the sphere centre, target and witness span at most a 2-plane, so the whole
threshold computation happens inside that plane. **The dimension enters only through which integer
vectors exist** — i.e. only through `eps*`, never through `c(w,n)`.

Verified two ways: against raw Euclidean geometry with no formula (4 000 random `(n, R in [1.2,300],
w)` x both signs, worst relative error `2.16e-12` / `1.74e-12`, bisection-limited), and by 60-digit
`Decimal` bisection of the same root (`5.6e-73`).

A second exact identity does most of the later work, and holds for **any** surface:

```
    d0_curved(w) = |w| - c_w          (at the threshold; verified geometrically to 5.6e-14)
```

i.e. admissibility is a statement about `c_w` alone.

### 4.2 The general law `(p . H . p)/4`

With `phi` the signed distance (positive on the exterior), `n` the unit normal into the exterior, and
`H` defined by `phi(P0 + u + t n) = t - (1/2) u.H.u + O(3)` (so `H = (1/R) I` for a concave sphere;
`H > 0` = concave):

```
    v0 = P0 + p + (eps+q) n  =>  phi(v0) = (eps+q) - (1/2) p.H.p + O(3)
    eps + phi(v0) > |w|      =>  eps > g + (p . H . p)/4
```

**[PROVED]** to first order, and **verified beyond an assertion**:

* **Cylinder** — an exact closed form was derived: `c = (R g - p_a^2/4)/(R - d0)` (concave), which
  expands to `g + p_c^2/(4R)`, i.e. `(p.H.p)/4` with `H = diag(1/R, 0)`. Exact form vs raw geometry:
  max relative error `2.25e-12` over 1500 random orientations. Fitted `H` vs `diag(1/R,0)`:
  `7.4e-8`. And `\|second-order model - exact\| * R^2` stays **bounded** (max 5.3) over
  `R in [60,200]` — so the model really is `O(1/R^2)`-accurate, not merely plausible.
* **Saddle** — a torus with the exact distance function; analytic principal curvatures re-derived and
  matched to a fitted `H` to `2.4e-7` over 900 points. Under a global scaling by `s` (so `H ~ 1/s`),
  `\|c - g - (p.H.p)/4\| * s^2` stays bounded (3.2-4.0) for `s = 30, 100, 300`, at the inner equator,
  at `psi = 130 deg`, and for both exterior choices.

### 4.3 The 3D curvature constants

Writing `M = sum_k lambda_k p_k p_k^T` (a `2x2` PSD form on the tangent plane, `lambda_k` the
barycentric weights of the triple tie), Danskin's theorem gives

```
    K_sphere(k,k)  = tr(M)/4        K_cylinder(k,0) = mu_max/4
    K_saddle(k,-k) = (mu_max - mu_min)/4     K_convex(-k,-k) = -tr(M)/4
```

| rule | `mu_max` | `mu_min` | `K_sphere` | `K_cylinder` | `K_saddle` |
|---|---|---|---|---|---|
| threshold | `0.188852002827` | `0.054096342354` | **`0.060737086295149`** | `0.047213000706674` | `0.033688915118199` |
| exact | `0.202191219637` | `0.051787355327` | **`0.063494643741112`** | `0.050547804909338` | `0.037600966077564` |

**[CERTIFIED]** — `K_sphere` was derived twice algebraically (Danskin/barycentric and an
implicit-function derivative), confirmed to **9 significant digits** by Richardson extrapolation of
the *exact* `eps*(R)` (`0.060737086020`, `0.063494643499`), and reproduced independently by a referee
to 32 digits. `K_cyl` and `K_saddle` were confirmed to 7-8 digits by Richardson extrapolation of the
second-order model with the principal frame held fixed. Second-order term: `eps*(R) = eps* +- K/R +
C/R^2`, `C ~ 0.1099 / 0.1087`. **[NUM]**

```
    K_thr / K_2D = 2.145 ,   K_exa / K_2D = 2.242         (K_2D = 0.028315226153961)
```

**Curvature hurts about 2.2x more in 3D than in 2D.**

### 4.4 The finite-`R` tables (exact sphere geometry)

> **A phase-1 table was wrong here, in the unsafe direction, and both referees caught it.** One track
> published a finite-`R` exact-rule table that is actually the **second-order (paraboloid) model**
> `c = g + p^2/(4R)`, labelled as "a sphere of radius `R`" with excess percentages. It understates the
> concave penalty by up to a factor 3.7. The correct tables, from the exact law `c = g/(1 - d0/R)`,
> are below.
>
> I settled this myself with a **raw-geometry** evaluator that never uses `g` or `c` — it places a
> real sphere, computes true Euclidean UDFs, and scans the depth (`syn/v4_curv.py`):
>
> | `R` | raw geometry (mine) | correct table | the refuted table |
> |---|---|---|---|
> | 5 | `0.057951775943` | `0.057951775943` | `0.051099734` |
> | 3 | `0.099168924645` | `0.099168924645` | `0.059499359` |
>
> A coarse grid search over the fundamental domain at `R = 5` found nothing above (`0.056659793`,
> grid-limited, argmax at the reported direction).

**TABLE 1 — THRESHOLD RULE, exact sphere geometry, `W = 3`.** `eps*_thr(inf) = 0.036662265118829642`

| `R` | CONCAVE | `eps* + K/R` | excess | exact/1st | CONVEX | benefit |
|---|---|---|---|---|---|---|
| 100 | `0.037280843814662` | `0.037269636` | `+1.69 %` | 1.0003 | `0.036065670088845` | `-1.63 %` |
| 50 | `0.037922759490198` | `0.037877007` | `+3.44 %` | 1.0012 | `0.035489815121403` | `-3.20 %` |
| 21 | `0.039829590486036` | `0.039554507` | `+8.64 %` | 1.0070 | `0.033997974067224` | `-7.27 %` |
| 12 | `0.042638737603488` | `0.041723689` | `+16.30 %` | 1.0219 | `0.032257455468803` | `-12.01 %` |
| 8 | `0.046548557748125` | `0.044254401` | `+26.97 %` | 1.0518 | `0.030452376166466` | `-16.94 %` |
| 5 | `0.056295244198017` | `0.048809682` | `+53.55 %` | 1.1534 | `0.027688733018959` | `-24.48 %` |
| 4 | `0.066467539884322` | `0.051846537` | `+81.30 %` | 1.2820 | `0.026120601686713` | `-28.75 %` |
| 3 | `0.099168924645437` | `0.056907961` | `+170.49 %` | 1.7426 | `0.023880776079030` | `-34.86 %` |
| 2.5 | `0.155733879014328` (b) | `0.060957100` | `+324.78 %` | 2.5548 | `0.022355879591306` | `-39.02 %` |
| 2 | `0.282527922054699` (b) | `0.067030808` | `+670.62 %` | 4.2149 | `0.020410423458827` | `-44.33 %` |

**TABLE 2 — EXACT RULE (what the pipeline delivers), exact sphere geometry, `W = 3`.**
`eps*_exa(inf) = 0.038443423574719627`

| `R` | CONCAVE | `eps* + K/R` | excess | exact/1st | CONVEX | benefit |
|---|---|---|---|---|---|---|
| 100 | `0.039089435425096` | `0.039078370` | `+1.68 %` | 1.0003 | `0.037819156907046` | `-1.62 %` |
| 50 | `0.039758393503250` | `0.039713316` | `+3.42 %` | 1.0011 | `0.037215519389909` | `-3.19 %` |
| 21 | `0.041736254066915` | `0.041466978` | `+8.57 %` | 1.0065 | `0.035647187385586` | `-7.27 %` |
| 20 | `0.041916417955` | `0.041618` | `+9.03 %` | | | |
| 12 | `0.044621250541646` | `0.043734644` | `+16.07 %` | 1.0203 | `0.033810186977294` | `-12.05 %` |
| 10 | `0.046116224664` | `0.044793` | `+19.96 %` | | | |
| 8 | `0.048568390373873` | `0.046380254` | `+26.34 %` | 1.0472 | `0.031898289739415` | `-17.03 %` |
| 5 | `0.057951775942861` | `0.051142352` | `+50.75 %` | 1.1331 | `0.028961359852068` | `-24.66 %` |
| 4 | `0.066467539884322` | `0.054317085` | `+72.90 %` | 1.2237 | `0.027291500136729` | `-29.01 %` |
| 3 | `0.099168924645437` | `0.059608305` | `+157.96 %` | 1.6637 | `0.024904517469410` | `-35.22 %` |
| 2.5 | `0.152621157907922` (m) | `0.063841281` | `+297.00 %` | 2.3906 | `0.023279478404394` | `-39.44 %` |
| 2 | `0.252790790191106` (m) | `0.070190745` | `+557.57 %` | 3.6015 | `0.021207904434300` | `-44.83 %` |

`(b)` barrier filter active; value from brute force over the full 250-vector witness set,
**[HIGH-NUM]** not certified. `(m)` single-hop value only. By Lemma M the single-hop analysis is
complete exactly while `eps* <= 1 - r_d = 0.13397`; the `R = 2.5` and `R = 2` rows of **both**
tables exceed that, so all four are **upper bounds** on the true multi-hop answer. Every row with
`R >= 3` is below the threshold and is therefore exact in that respect.
All other rows **[CERTIFIED]** — computed by a polytope-vertex method (the sphere condition
`c_w(n) >= t` is again *linear* in `n`), cross-checked by an independent grid+refinement with the
full 250-vector witness list (`<= 1.3e-16` agreement), reproduced independently by both referees, and
confirmed **end-to-end** at `R = 100, 21, 12, 8, 5, 4, 3` in both signs (origin uncertified at
`eps*(1-1e-11)`, certified at `eps*(1+1e-8)`, single seed component, 20/20 configurations).

**Practical calibration:**

```
    concave excess reaches  +1% / +5% / +10% / +25% / +50% / +100%   at   R = 167.5 / 34.9 / 18.4 /
                                                                              8.5 /  5.2 /   3.7
    first-order error (concave): +0.03% at R=100, +0.7% at 21, +2.2% at 12, +5.2% at 8,
                                 +15% at 5, +74% at 3
        -> use eps* + K/R only for R >~ 15.
    crude rigorous closed-form bound (concave, R > 3):  eps*(R) <= eps*(inf) * R/(R-3)
        (1.4% loose at R = 100, 121% loose at R = 4)
```

Compared with 2D at the same radii: `R = 100` `+1.50 %` (2D) vs `+1.69 %` (3D); `R = 21` `+7.58 %` vs
`+8.64 %`; `R = 8` `+22.86 %` vs `+26.97 %`; `R = 5` `+42.92 %` vs `+53.55 %`; `R = 3` `+105.99 %` vs
`+170.49 %`.

### 4.5 Is the concave sphere really the worst case?

**Partly. The naive argument is incomplete, the repaired argument works only for `R > W = 3`, and
searches support the conclusion further down.** This is the one place where a referee finding remains
**[OPEN]** rather than fixed.

The naive argument — `p.H.p <= kappa \|p\|^2` pointwise, and `min`/`max` are monotone — ignores that
`H` also moves witness *depths*, hence the admissible **set**. Using the exact identity
`d0_curved = \|w\| - c_w`, admissibility becomes a statement about `c_w` alone, yielding two
shape-independent lattice constants at `W = 3`:

```
    any witness with |w| > W that IS in band costs   c_w >= sqrt10 - 3 = 0.162277660168380
    any barrier-excluded witness costs               c_w >= 1 - r_d    = 0.133974596215561
```

> **THEOREM (threshold rule, `Z^3`, `W = 3`).** Let `V_S(n)` be the concave-sphere value at `n`.
> If `V_S(n) < 1 - r_d` **and the curvature class is bounded so that `c_H >= 0`**, then
> `eps*(n,H) <= V_S(n)` for every `H <= kappa I`.
> *Proof.* (1) The sphere's argmin `w*` has `\|w*\| <= 3`, else `c >= \|w*\| - W >= 0.1623 > V_S(n)`.
> (2) `c_H(w*) <= c_S(w*) = V_S(n)`. (3) `w*` is admissible for `H`: band, `\|w*\| - c_H <= \|w*\| <= 3`;
> barrier, `\|w*\| - c_H >= 1 - V_S(n) > r_d`. Hence `eps*(n,H) <= c_H(w*) <= V_S(n)`. QED

**The referee-found gap, not repaired:** step (3)'s band half tacitly needs `c_H >= 0`. Under the
literal hypothesis `H <= kappa I` with *no lower bound* on the principal curvatures, `c_H = g +
(p.H.p)/4` can go negative; for a witness with `\|w*\| = 3 = W` exactly — and `(2,2,1)` is exactly
that, and it **is** the argmin on a 14 % solid-angle cell — one gets `c_H = g(1 - kappa d0) < 0` as
soon as `kappa > 1/d0 ~ 1/3`. So:

* **[PROVED]** for `R > W = 3` (equivalently `\|k_i\| <= 1/W`).
* **[HIGH-NUM]** for `2.5638 < R <= 3`: the sharp guaranteed-window certificate closes the gap to
  `<= 7e-17` for the exact rule at `R = 100, 50, 21, 12, 8, 5, 4, 3, 2.5, 2` and for the threshold
  rule at `R >= 3`; at `R = 2.5, 2` the threshold-rule certificate has `3.5e-3` of slack and is
  **inconclusive** — a certificate limitation, not an observed violation.
* **[NUM]** further support: direct adversarial multistart over `(n, theta, k1, k2)` with
  `\|k_i\| <= kappa` in the second-order model at `R = 100, 12, 5, 3, 2.5, 2`, both rules — the sphere
  wins every time (`best_adversary - V_S <= -5e-15`); exact-geometry cylinder-vs-sphere over 1200
  random `(n, axis)` per `R` — worst difference `-9.2e-14 .. -8.1e-8`, always `<= 0`; a cylinder
  optimised over `(n, axis)` reaches `cyl/sph = 0.9951` at `R = 100` falling to `0.8546` at `R = 3`,
  never above 1.
* **[OPEN]** in general, and for surfaces not in `{sphere, cylinder, torus}` under exact geometry.

**The admissibility shifts are real, they are just inert.** Census at the concave sphere's own
optimum: the admissible set already differs from the planar one at `R = 100` (2 changes) and then 18
(`R = 12`), 32 (8), 66 (5), 102 (4), 147 (3), 160 (2.5). The sharpest case is explicit: `(3,1,0)` at
`n*_thr` is out of band in the plane (`d0_thr = 3.1258 > W`), and concave curvature lowers its depth
until it **re-enters** the band at exactly `R = d0/(1 - g/(\|w\|-W)) = 4.030689269` — at which instant
its cost is exactly `sqrt10 - 3 = 0.162277660`, i.e. `2.46x` the sphere's own `eps*(R)`. Completely
inert. The best adversarial construction — a cylinder whose axis is parallel to `p_{(3,1,0)}`, so that
`(3,1,0)` gets zero penalty and stays out of band while the tied witnesses are penalised — does keep
`(3,1,0)` out and still loses badly (`0.0263` vs the sphere's `0.0665` at `R = 4`).

Shape ordering at equal maximum principal curvature, as the `K` formulas predict and exact geometry
confirms: `sphere(k,k) > cylinder(k,0) > saddle(k,-k) > convex(-k,-k)`.

*One further correction:* the refuted table's companion claim that "the 3-tie remains the global min
over all 74 witnesses down to `R = 2`" is also false under the exact law — the argmax leaves the tie
branch by `R ~ 4-5`, moving to `(0.899,0.379,0.221)` at `R = 4`, `(0.863,0.443,0.243)` at `R = 3`,
`(0.410,0.792,0.453)` at `R = 2`.

### 4.6 Convex saturation: it reproduces, and it is not saturation

The 2D observation ("flattens near `0.53 eps_c` instead of going to 0") reproduces almost exactly in
3D, and the mechanism is now identified. As `R -> 0` for a convex sphere,

```
    c_w = R g/(R + d0)  ->  (R/d0) g = R p^2/(4 d0^2) = R tan^2(theta_w/2),
```

i.e. **the length weighting disappears and only the angle survives**, so

```
    eps*(R) ~ R tan^2(delta/2),   delta = angular covering radius of the admissible directions.
```

| | `delta` | `tan^2(delta/2)` | measured `eps*(R)/R` at `R = 1e-5` |
|---|---|---|---|
| 3D (98 primitive directions, `\|w\| <= 3`) | **`17.653171084 deg`** | `0.024112915270` | `0.024112744` |
| 2D (16 primitive directions, `\|w\| <= 3`) | `13.282525589 deg` | `0.013556834781` | `0.013556735` |

So the convex value goes to **zero, linearly** — but with a slope `2.52x` smaller than `K` in 3D
(`2.09x` in 2D), which is exactly why the exact curve looks like a plateau while `eps* - K/R` dives.
At `R_0 = K/eps*` (where the first-order formula predicts zero benefit): 2D `R_0 = 1.4745`,
`exact/eps_c = 0.5051`; 3D `R_0 = 1.6567`, `exact/eps* = 0.5109`. **Same ratio, phenomenon
reproduced.**

*Correction carried:* the covering radius is `17.653171084 deg`, **not** the `17.633655203 deg`
reported by two phase-1 tracks. I re-derived it by exhaustive Voronoi-vertex enumeration over all
`C(98,3)` direction triples (`syn/v5_misc.py`): deep hole at `n = (0.952909655, 0.224951455,
0.203371661)`, `delta = 17.653171084 deg`. Two referees agree, and the `R -> 0` asymptote above is an
independent corroboration to 5 digits. Downstream, the naive bounds become

```
    W tan^2(delta/2)     = 0.072338745809   (was quoted as 0.072176339)
    sqrt6 sin^2(delta/2) = 0.057673658579   (was quoted as 0.057547218)   <- nearest-ANGLE worst case
```

versus the true cheapest-witness answer `0.036662265`. The gap between `0.0577` and `0.0367` is
exactly the length-weighting, as in 2D.

### 4.7 A 3D-only bonus: curvature merges the two admissibility rules

At `W = 3` the rules differ in the plane only because the `\|w\| = 3` witnesses have their exact-rule
window truncated to `(c, 2c]`. Concave curvature lowers witness depths and lifts that truncation:
`R = 4.8, 4.6, ..., 4.1` still distinct; at `R = 4.0` the two tables are **identical to the last
bit**, and below `R = 4` they coincide (until barrier/multi-hop effects take over near `R ~ 2.5`).
**[HIGH-NUM]**

---

## 5. What is proved, what is certified, what is open

| claim | status |
|---|---|
| The reduction, `\|w\| = g + d0`, `g d0 = p^2/4`, ball-membership `= 2x` | **[PROVED]** (dimension-free algebra), each independently re-checked numerically |
| Candidate set `= 122` vectors (`\|w\|^2 <= 9`), fixpoint, completeness; `250` (`\|w\|^2 <= 14`) for the exact-rule sup | **[PROVED]** (exact integer census; `sqrt10 - 3 = 0.1623 > G0 = 0.0735593211498971297`, the latter now an exact surd, not a grid bound) |
| Band test vacuous at `W = 3`; Lemma A (barrier inactive) in 3D | **[PROVED]** (`g >= \|w\| - r_d >= 0.1340 >> eps*`), plus 0-ulp empirical confirmation |
| `B_3` reduction; rearrangement lemma; reduction to primitives; critical-point families exhaust the local-max condition | **[PROVED]** |
| The exact-rule reduction `eps*_exa(n) <= A(n)` for **every** `n`, with equality at the maximiser | **[PROVED]** (section 2.7), plus 40 000-direction check with 0 violations |
| Upper bound `eps* <= ` the surd, all `n` | **[PROVED]** — machine-checked exact-rational polytope-vertex certificates, two independent implementations (brackets `1e-30` and `3.04e-64`). I re-confirmed only at double precision (`+-1e-13`), so read "proved" as "proved modulo the correctness of two independently written certificate programs". |
| Lower bound (attainment) at both `n*` | **[PROVED]** (exact algebraic ties + exhaustive `Decimal` enumeration to `\|w\| <= 30`) |
| Minimal polynomials, degree exactly 8, irreducibility | **[PROVED]** (exact `8x8` determinant + non-square test; this upgrades phase 1's numerical certificate) |
| `\|Gal\| = 64` for both splitting fields | **[PROVED]** (exact square classes), Chebotarev-consistent |
| Local-max census (threshold rule): exactly three above `0.002` | **[CERTIFIED]** (two independent enumerations) |
| Local-max census (exact rule) | **[HIGH-NUM]** — search only; the envelope is discontinuous |
| `W_lo`, `W_hi`, `W_lo'`, `W_hi'` closed forms | **[CERTIFIED]** (three routes each); `W_lo'` minimal polynomial `16x^4+32x^3-448x-23` **[PROVED]** |
| The `W`-map over `[1,7]`, plateau values | **[CERTIFIED]** (each an exact tie root inside a `1e-12` bracket; 16 rows re-checked here) |
| Three exact-rule plateau starts above `W = 5.4` | **[NUM]**, bracketed to `1.3e-8 / 1.2e-7 / 3.2e-7` |
| Ramp formulas `(3+sqrt2+sqrt3-2W)/4` and `(sqrt2+sqrt3-W)/3` | **[PROVED]** as suprema of an explicit branch; matched to certified brackets at 12+ values of `W` |
| Exact sphere law; `(p.H.p)/4`; `K_thr`, `K_exa` | **[PROVED]** / **[CERTIFIED]** (see 4.1-4.3) |
| Finite-`R` sphere tables (all rows but 4) | **[CERTIFIED]** (polytope method + independent grid + end-to-end, two referees reproducing) |
| Finite-`R` rows marked `(b)`, `(m)` | **[HIGH-NUM]** / upper bounds only |
| Concave sphere is the worst case | **[PROVED]** for `R > W = 3`; **[HIGH-NUM]** to `R = 2.5638`; **[OPEN]** below and for general surfaces |
| Multi-hop enrichment cannot help at `W = 3` (Lemma M) | **[PROVED]** (given full seeding) |
| Single 6-connected exterior seed component at `W = 3` | **[HIGH-NUM]** (~78 000 trials, 0 violations; the no-straddling half is **[PROVED]**) |
| Upward-closedness of the certified set | **FALSE** in 3D — the uncovered set is not an interval at `8.45 %` of directions. `eps*_exa` is still exactly the sup of the uncovered set, which is the quantity the guarantee needs. |
| `tol = 0` hazard and the `tol/2` cost | **[CERTIFIED]** experimentally (three independent simulators) |
| Worst case over all surfaces | **[OPEN]**, as in 2D |
| 3D Farey / unimodular-triple structure | **[OPEN]**, not attempted |

---

## 6. Referee findings and their disposition

Two adversarial referees reviewed the seven tracks. One returned `sound = true`, one `sound = false`;
**the `false` verdict is about the surrounding claims (plateau floor, curvature table, band-test
recommendation, `W`-sensitivity), not about the two constants**, which both referees certified
independently with their own branch-and-bound and (in one case) an exact-rational certificate.

| # | Finding | Disposition |
|---|---|---|
| V1 | The upper bound can be upgraded from "certified-numerical" to **proved** by an exact-rational polytope-vertex certificate | **FIXED** — status upgraded in section 5; both implementations recorded |
| V3 | The exact-rule reduction `eps*_exa <= A(n)` holds for *every* `n`, so the phase-1 open item "equality only checked at the maximiser" is a non-issue | **FIXED** — proof in 2.7; open item closed |
| V4 / S5 | **The finite-`R` exact-rule curvature table in one phase-1 track is the second-order model, not the sphere.** Understates the concave penalty by up to 3.7x (`+55 %` vs the true `+158 %` at `R = 3`) | **FIXED** — TABLE 2 in 4.4 replaces it; I confirmed the replacement by raw geometry with no formulas at all |
| S5b | The same track's claim "the 3-tie remains the global min down to `R = 2`" | **FIXED** — false under the exact law; the argmax leaves the tie branch by `R ~ 4-5` (4.5) |
| V5 / S4 | **The exact-rule plateau floor is wrong in two tracks** (`2.5153786883076` with a claimed `1.4e-14` bracket, and `~2.5151853`) | **FIXED** — correct value `sqrt3 - 1/2 + (3/2)sqrt(sqrt3-1) = 2.515450323319905583`; I reproduced the refutation (3.2). Root cause recorded: bisection on depth when the uncovered set is not an interval |
| V6 / S9 | Angular covering radius is `17.653171084 deg`, not `17.633655203 deg`; the derived naive bounds change | **FIXED** — 4.6; I re-derived it by exhaustive `C(98,3)` Voronoi-vertex enumeration |
| V7 | The winning-witness solid-angle census is wrong in every entry (up to `6.7 pp`; `(1,0,0)` understated `1.75x`) | **FIXED** — 2.6, corrected census confirmed by my own Monte Carlo |
| V8 | The `drop (1,1,0)` row of the minimal-covering-set table is `0.0591603527`, not `0.0586054173` (a local search that stopped early) | **FIXED** — 2.8, my certified B&B gives `0.059160352656`. The conclusion (six is minimal) is unaffected |
| V9a / S10 | The large-`W` table is 30-100x worse than its stated `~1e-6`; `W = 2.99980 -> 0.0366519` is *impossible* (below the plateau value, contradicting monotonicity); `W = 3.5 -> 0.03204` vs true `0.0320742582` | **FIXED** — the certified map in 3.3 replaces those rows; I re-checked five of them |
| V9b | One coarse staircase says `W = 2.0-2.5 -> 0.0723`; at `W = 2.0` the exact rule gives `0.0735593211` | **FIXED** — 3.3; confirmed by my B&B |
| V10 | Convention clash: threshold plateau written `[W_lo, W_hi)` in two tracks and closed in another | **FIXED** — stated as a **sup**: `eps*(W_hi) = eps*_thr` as a supremum that is not attained; the maximum is attained on `[W_lo, W_hi)` |
| S3 | The `min d0_thr((1,0,0))` over the fundamental domain is exactly `(1+1/sqrt3)/2 = 0.788675134594813`; a quoted `0.789135` was a grid artifact | **FIXED** — transfer table row 5b |
| S7 | **The recommendation "the band test must be non-strict" is refuted:** strict `d0 < W` gives the same constant at `W = 3` | **FIXED** — 3.6; my certified B&B with the strict test returns the same value; the phase-1 track that concluded "both constants unchanged" was the correct one |
| S8 | The `W`-sensitivity table asserting a *jump* at `2.996245` and a `0.125 %` downward margin — it is a continuous ramp, and the true margin is `18.7x` smaller | **FIXED** — 3.5; I confirmed the ramp formula to 9 digits at four values of `W` |
| S11 | The exact-rule bootstrap certificate's step "`J_w = (g, r_d]`" is false as written (the window top is `min(r_d, W - w.n)`) | **FIXED** — restated: one only needs the window to reach `E`, and `W - w.n >= W - \|w\| >= E` delivers exactly that. The certificate's numerical output was correct |
| V11 / S6 | **The "concave sphere is worst" theorem has a hole**: step (3) tacitly assumes `c_H >= 0`, which fails for the `\|w\| = 3 = W` witnesses once curvature exceeds `~1/3` | **OPEN — NOT FIXED.** The theorem is restated in 4.5 for `R > W = 3` only; `2.5638 < R <= 3` rests on a certificate plus adversarial search, and the threshold-rule certificate is explicitly inconclusive at `R = 2.5, 2` |
| V2, V12, V13, S1, S2, S12 | Everything else the referees re-implemented checked out (list below) | **Noted**, and reflected in the confidence markers |

Independently re-implemented by a referee and found correct, at the stated precision or better:
the closed forms at 300 digits (residuals `~3e-297`); both octics vanishing **identically** in exact
ring arithmetic; the degree-8 irreducibility determinants `1009375/262144` and `62647344` reproduced
exactly; tie multiplicity exactly three, exhaustive to `\|w\| <= 25` in 80-digit `Decimal`, with
runner-up cost exactly `2 eps*`; the full critical-point census on the fundamental cone (three
interior local maxima with matching barycentric weights, all mirror-edge criticals, all closed forms);
`K_thr` and `K_exa` to 32 digits; peak sharpness; `W_lo`(threshold) to 31 digits by an independent
derivation (so the `2.010e-4` brittleness headline is real); both ULP facts; the `19.6 %`
barrier-exclusion measure; `r_d/eps*_exa = 22.527`; the non-interval uncovered set (`8.53 %`); a
70 000-direction adversarial hunt that found nothing above either constant (closest approaches
`-6.8e-13`, `-6.4e-13`); and the `tol = 0` hazard, reproduced independently.

**Nothing in this document presents a challenged step as settled.** The one finding that is not fixed
(V11/S6) is flagged in the transfer table (row 14c), in section 4.5, and in the open-questions list.

---

## 7. 3D versus 2D, side by side

| quantity | 2D (`Z^2`, `W = 3`) | 3D (`Z^3`, `W = 3`) | change |
|---|---|---|---|
| `r_d` | `0.707106781186548` | `0.866025403784439` | `+22.5 %` |
| `1 - r_d` (Lemma A threshold) | `0.292893218813452` | `0.133974596215561` | `-54 %` — margin over `eps*` drops from `15.3x` to `3.5x` |
| constant, threshold rule | `0.019202630763796` | `0.036662265118830` | `1.909 x` |
| constant, exact rule | `0.019202630763796` (same) | `0.038443423574720` | `2.002 x`; **the two rules now differ** (`+4.86 %`) |
| ball-membership constant `2 eps*` | `0.038405261527593` | `0.076886847149439` | *(note `2 eps_c(2D) = 0.0384053` vs `eps*_exa(3D) = 0.0384434` — a numerical near-coincidence, not a theorem)* |
| algebraic degree | 4 | **8** (runner-up configuration: 16) | `deg = 2 [Q(\|w_i\|):Q]` |
| critical configuration | pair tie, transversal kink | **triple tie**, conic point | as CONTEXT_3D predicted |
| symmetry group / fundamental domain | `D4`, order 8, `45 deg` arc | `B_3`, order **48**, spherical triangle | orbit of `n*` is 48 |
| minimal covering witness set | 3 per octant, 16 total | 6 per fundamental domain, 98 total (5 / 74 under the exact rule) | |
| angular covering radius of the primitive `\|w\| <= 3` directions | `13.2825 deg` | `17.6532 deg` | `S^2` is worse covered |
| nearest-**angle** worst case vs cheapest-witness answer | `0.029908` vs `0.019203` | `0.057674` vs `0.036662` | length-weighting gap grows |
| `3x3(x3)` mask constant | `0.044910139437773` (shown in the 2D write-up to equal Borgefors 1986) | `0.073559321149897` | `1.638 x` |
| depth gain over plain connected components `r_d/eps*` | `36.82 x` | `22.53 x` (exact rule) | **3D is harder** |
| `eps*/W` | `0.0064` | `0.012814` | |
| threshold-rule plateau | `[2.2361, 3.1608)`, width `0.925` | `[2.99980, 3.12585]`, width `0.126` | `W = 3` margin below: `0.7639 -> 0.000201` (**knife edge**) |
| exact-rule plateau | `[2.2795, 3.1785)`, width `0.899` | `[2.51545, 3.03093)`, width `0.516` | `W = 3` margin above: `5.95 % -> 1.03 %` |
| "no guarantee at all" below | `W < 1 + r_d = 1.7071` (exact rule) | `W < 1 + r_d = 1.8660` (exact); `W < 2/sqrt3 = 1.1547` (threshold) | same mechanism, larger `r_d` |
| curvature constant `K` (concave sphere) | `0.028315226153961` | `0.060737086295149` (thr) / `0.063494643741112` (exa) | `2.15x` / `2.24x` |
| concave excess at `R = 8` | `+22.9 %` | `+26.3 %` (exact rule) | |
| concave excess at `R = 3` | `+106 %` | `+158 %` | curvature matters more |
| uncovered depth set at `W = 3` | an interval at all 90 001 sampled directions | **not** an interval at `8.45 %` of directions | upward-closedness fails |
| `tol = 0` hazard | 1 wrongly certified point observed | 1300-1700 of ~3000 — the whole interior band | cascades in 3D |
| peak sharpness (1 % loss) | `0.0802 deg` | `0.0708 deg` | slightly sharper |

**What changed qualitatively, in one list.**

1. **The two admissibility rules split.** At `W = 3` they agreed in 2D and disagree in 3D, entirely
   because `(2,2,1)` is a primitive vector of norm exactly `3 = W` and `Z^2` has none.
2. **Pair tie becomes triple tie.** The maximum is a conic point of the lower envelope (a spherical
   Voronoi vertex), with all three barycentric weights strictly positive.
3. **Degree 4 becomes degree 8**, and the degree is now predicted by the arithmetic of the tying
   norms; nearby configurations reach degree 16.
4. **Upward-closedness fails.** The uncovered depth set is generically two intervals. Any measurement
   that bisects on depth is unsound — and this actually produced a wrong published number (3.2).
5. **`W = 3` becomes knife-edge for the threshold rule** (`2.01e-4` of margin, versus `0.76` in 2D),
   though not for the exact rule.
6. **Curvature matters about 2.2x more**, and the first-order model degrades much faster.
7. **The scheme is less powerful in absolute terms**: `22.5x` deeper than connected components,
   versus `36.8x` in 2D, because `S^2` is worse covered by cheap lattice directions than `S^1`.
8. **The implementation hazard is qualitatively worse**: in 3D a single 1-ulp error at a `45 deg` wall
   or a corner bevel destroys the sign of the entire interior band.
9. **Multi-hop enrichment is now settled** (Lemma M) — and the same argument works in 2D, so this is a
   retroactive improvement to the 2D document as well.
10. **The threshold rule is not merely a different convention** — for narrow bands it is optimistic by
    an unbounded factor (it reports finite constants where the true answer is "no guarantee at all").

---

## 8. What this means for the pipeline

1. **Guaranteed capture depth, planar interface, `W = 3`, 3D: `0.03844342357471963` voxels**
   (rounded up). Every exterior lattice point deeper than that is certified; some configuration — a
   plane with normal `(sqrt6-sqrt3, gamma, sqrt3-sqrt2)`, target at depth exactly `eps*_exa` — leaves
   one uncertified at exactly that depth. Do not quote `0.0366622651`: that is the threshold rule,
   which is optimistic at exactly `W = 3`.
2. **Use a guarded predicate `UDF(A) + UDF(B) - dist > tol`, with `tol >= 4e-15`; recommend
   `1e-12`.** With `tol = 0` the rule is unsound at every normal parallel to an integer vector, and in
   3D one 1-ulp error cascades: `(1,1,0)`, `(1,1,1)`, `(2,1,0)`, `(2,1,1)`, `(2,2,1)` and many others
   fail at **100 %** of tested offsets, wrongly certifying 1300-1700 of ~3000 in-band points out to
   `UDF ~ 3.0`. Measured thresholds: `tol = 0` and `1e-16` -> 397 959 wrong points in a 13-normal
   battery; `1e-15` -> 261 822; `4e-15` -> 0. Only `(1,0,0)`-type normals are structurally safe (all
   304 200 tied pairs evaluate to exactly `0.0`); generic normals are fine at `tol = 0` (1500 trials,
   0 violations), but generic is not what artists model. The cost of the guard is a conservative
   `tol/2` shift in the guaranteed depth, measured exactly.
3. **`W = 3` has only `1.03 %` of headroom above** (`W_hi' = 3.030934`) and `16 %` below. Raising `W`
   to `3.1` gives a continuous ramp, not an improvement; `W ~ 3.08` would buy the better constant
   `0.0366622651` (`+4.63 %` depth) under both readings with real margin. Do **not** shrink `W` below
   `~2.2`: the exterior seed set stops being a single 6-connected component and a
   single-point-seeded flood fill loses fragments.
4. **Size any concave-feature safety factor from TABLE 2, not from `eps* + K/R`.** The first-order
   formula is good to `2 %` only for `R >~ 12`; at `R = 3` it understates by `74 %`. Concrete: `R = 8`
   `+26 %`, `R = 5` `+51 %`, `R = 3` `+158 %`. Convex features are symmetrically better.
5. **`r_d - eps*_exa = 0.8276 > 0`**, so the worst-case target really is a barrier voxel that
   connected components cannot settle; the pairwise rule reaches `22.5x` deeper. `2 eps*_exa =
   0.0768868471494393` if the weaker ball-membership criterion is used instead.
6. **Do not compute the constant at runtime.** Every algebraically equivalent surd form is `6-95` ulp
   off in double, and the correctly rounded double lies *below* the true value. Hard-code
   `0.03844342357471963`.

---

## 9. Open questions, and what it would take to close them

**Ranked by how much they could change the number.**

1. **The worst case over all surfaces (V11/S6).** Most likely place where the reported constant is
   not conservative. Needed: (i) repair the "sphere is worst" theorem below `R = 3` by adding a lower
   bound on the principal curvatures (`\|k_i\| <= kappa`) and re-running the band step, or find a
   counterexample; (ii) push the sharp guaranteed-window certificate to the threshold rule at
   `R = 2.5, 2` where it currently has `3.5e-3` of slack; (iii) go beyond `{sphere, cylinder, torus}`
   in *exact* geometry. Estimated effort: days, not weeks — the machinery exists.
2. **Small-`R` concave, multi-hop regime.** By Lemma M the single-hop analysis is complete only while
   `eps* < 1 - r_d = 0.13397`, i.e. `R >~ 2.6` concave. The `R = 2.5` and `R = 2` rows of both
   tables are therefore upper bounds; the true values require running the closure. Needed: a curved-geometry end-to-end
   sweep at `R = 2, 2.5`. Also, `(b)` rows of TABLE 1 are brute-force, not certified.
3. **Upgrade the whole `W`-map and the curvature tables to interval arithmetic.** Everything except
   the double-precision *evaluation* is already exact; redoing the branch-and-bound patch enclosures
   with `Fraction`/`Decimal` directed rounding would upgrade every **[CERTIFIED]** row to
   **[PROVED]**. The two `W = 3` constants have already had this done; the rest has not.
4. **The exact-rule local-max census.** Currently search-only, because the exact-rule envelope is
   discontinuous; a narrow peak cannot be excluded by search. The B&B *global* bound is certified, so
   this affects only the secondary structure. Needed: a discontinuity-aware certificate.
5. **Closed forms for the remaining plateau endpoints.** `W_lo'` is now degree 4 with minimal
   polynomial `16x^4+32x^3-448x-23` (done here). Still missing: `1.2544355486`, `2.1804880080 /
   2.1972344195`, `3.3141721917 / 3.3446745744`, `3.7416569969 / 3.7702462712`, `4.2355152670 /
   4.2504485827`, `4.5824013795`, `4.6030576080`, and the three above `W = 5.4` (which are only
   bracketed to `1e-8..3e-7`). Each is the terminus of a ramp branch and is solvable by the same
   generic tie solver with one extra linear constraint.
6. **Structure: is there a 3D Farey/Stern-Brocot characterisation?** Both tying triples
   (`(1,0,0),(2,1,0),(2,1,1)` and `(1,1,0),(1,1,1),(2,1,1)`) have determinant `+-1`. A
   unimodular-triple (Delone/Selling) theory would explain *why* these triples and would predict the
   answer for other `W` and other lattices without enumeration. Nobody attempted it; it is the only
   item here that is genuinely research rather than computation.
7. **The single-6-connected-component assumption at `W = 3`.** Empirical over ~78 000 trials. A proof
   needs a lattice-slab connectivity argument for `{r_d < x.n <= W}`, uniform in `n`. Also untested:
   whether seeding from one known-outside point is equivalent to seeding from every positive
   component at `W = 3` (it is provably *not* below `W ~ 2.2`).
8. **Degenerate exterior pockets** (in-radius `<= r_d`, so no certified seed exists at all) were not
   tested in 3D. In 3D such pockets are geometrically richer than in 2D (thin sheets, tubes) and may
   be more common. This is a distinct failure mode from the depth constant.
9. **The cheapest out-of-band witnesses** — `(9,3,2)` at `n*_thr` (`g = 1.297e-4`) and `(7,6,3)` at
   `n*_exa` (`g = 2.244e-4`) — were found by brute enumeration over `\|w\|^2 <= 400`, not by a
   simultaneous-Diophantine-approximation argument. Irrelevant at `W = 3` (their depths are `~9.7` and
   `~7`), but the missing argument is the same one item 6 would supply.
10. **Irreducibility of the degree-16 family members** (the second threshold local max, `W_hi`) rests
    on the same null-space argument, which is correct, but the non-square test was not separately
    unit-tested at `r = 3`.

---

## Appendix A: constants

| symbol | closed form | value (30 digits) |
|---|---|---|
| `r_d` | `sqrt3/2` | `0.866025403784438646763723170753` |
| `beta` | `sqrt(4 sqrt30 + 2 sqrt5 - 26)` | `0.617282962024891734413786599097` |
| `gamma` | `sqrt(6 sqrt2 + 2 sqrt6 - 13)` | `0.619887780009354990999026451863` |
| **`eps*_thr(3)`** | `(3 - sqrt5 - beta)/4` | `0.036662265118829642294259933043` |
| **`eps*_exa(3)`** | `(sqrt2 + sqrt3 - sqrt6 - gamma)/2` | `0.038443423574719626566412269573` |
| `n*_thr` | `((sqrt5-1+beta)/2, (sqrt5-1-beta)/2, sqrt6-sqrt5)` | `(0.926675469762340715411480133913, 0.309392507737448980997693534817, 0.213421765283388401788110405974)` |
| `n*_exa` | `(sqrt6-sqrt3, gamma, sqrt3-sqrt2)` | `(0.717438935214300804669837733200, 0.619887780009354990999026451863, 0.317837245195782244725757617296)` |
| `2 eps*_thr` (ball membership) | | `0.073324530237659284588519866086` |
| `2 eps*_exa` (ball membership) | | `0.076886847149439253132824539147` |
| `1 - r_d` (Lemma A / Lemma M threshold) | | `0.133974596215561353236276829247` |
| `sqrt10 - 3` (cheapest in-band `\|w\| > W`) | | `0.162277660168379331998893544433` |
| `3x3x3` bootstrap constant `G0` | `(1 - sqrt(2 sqrt2 + 2 sqrt6 - 7))/2` | `0.073559321149897129681409626999` |
| `W_lo` (threshold) | `(18 + 10b + sqrt(24-8b^2))/12`, `b = sqrt3 - 2 eps*_thr` | `2.999798995571213626345506931224` |
| `W_hi` (threshold) | `(sqrt10 + 2 sqrt5 - 2 + beta)/2` | `3.125848288596425229615513740496` |
| `W_lo'` (exact) | `sqrt3 - 1/2 + (3/2) sqrt(sqrt3 - 1)` | `2.515450323319905582981299990823` |
| `W_hi'` (exact) | `sqrt2 + sqrt3 - 3 eps*_exa` | `3.030934099217813462629898256996` |
| value just below `W_lo'` | `(1 - sqrt(sqrt3-1))/2` | `0.072200161416323903515382116894` |
| value just below `W_hi'`'s ramp end | `sqrt2/2 - sqrt(sqrt6-2)` | `0.036666819084661702` |
| threshold no-guarantee bound | `2/sqrt3` | `1.154700538379251529018297561004` |
| exact no-guarantee bound | `1 + r_d` | `1.866025403784438646763723170753` |
| `K_thr` (concave sphere) | | `0.060737086295148778284871358617` |
| `K_exa` (concave sphere) | | `0.063494643741112051049616127551` |
| `delta` (covering radius, 98 primitive `\|w\| <= 3`) | | `17.653171083837 deg` |
| `eps_c` (2D reference) | `(3 - sqrt5 - sqrt(2 sqrt5 - 4))/4` | `0.019202630763796343828423710860` |

**Minimal polynomials** (all **[PROVED]** by exact linear algebra):

```
    eps*_thr : 64x^8 - 384x^7 + 1344x^6 - 2944x^5 + 4624x^4 - 5040x^3 + 3400x^2 - 800x + 25
    eps*_exa :    x^8         +    2x^6 +   12x^5 +   79x^4 -  228x^3 +  474x^2 -  252x +  9
    2 eps*_thr : x^8 - 12x^7 + 84x^6 - 368x^5 + 1156x^4 - 2520x^3 + 3400x^2 - 1600x + 100
    2 eps*_exa : x^8 + 8x^6 + 96x^5 + 1264x^4 - 7296x^3 + 30336x^2 - 32256x + 2304
    n*_thr x,y : x^8 + 4x^7 + 28x^6 + 60x^5 + 226x^4 + 260x^3 + 132x^2 - 852x + 241
    n*_thr z   : x^4 - 22x^2 + 1
    n*_exa x   : x^4 - 18x^2 + 9      n*_exa y : x^8 + 52x^6 + 822x^4 + 3796x^2 - 1583
    n*_exa z   : x^4 - 10x^2 + 1
    W_hi'      : x^8 + 58x^6 - 252x^5 + 4899x^4 + 4356x^3 + 75670x^2 - 590364x + 571849
    W_lo'      : 16x^4 + 32x^3 - 448x - 23                                    (degree 4; new here)
    W_hi       : degree 16 (256x^16 + 4096x^15 + ... + 701250625)
```

**220 certified digits** (exact-rational Sturm bisection, enclosure width `6.9e-226`):

```
eps*_thr = 0.0366622651188296422942599330430309823190050815368640391230273809195713756102184085771904
           554297811587631221966134782829052668430930912460067129003472427474511300769565219308949150
           003244600687101100656993982687698020602308...
eps*_exa = 0.0384434235747196265664122695732929372365091045710865456107623270725430704657704284828303
           430972191028997906244792835558797693002458925171015866388253027297193386907918532675234320
           790309670719638273797358512654680811602663...
```

## Appendix B: verification artifacts

Written for this synthesis (independent re-derivations, no reuse of track code):

| file | what it checks |
|---|---|
| `scripts/distancing_experiments/lattice_gap_3d/syn/slib.py` | from-scratch library: witness enumeration, both rule evaluators, window algebra, rigorous spherical-triangle branch-and-bound for both rules |
| `scripts/distancing_experiments/lattice_gap_3d/syn/v1_forms.py` | closed forms, unit norms, triple ties, both octics, all derived ratios and rounding facts at 160 digits |
| `scripts/distancing_experiments/lattice_gap_3d/syn/v2_bnb.py` | certified brackets for both constants (`+-1e-13`); strict-vs-non-strict band test; witness-radius stability `\|w\|^2 <= 9..49`; the `3x3x3` constant |
| `scripts/distancing_experiments/lattice_gap_3d/syn/v3_plateau.py`, `v3b` | refutation of the two wrong exact-rule plateau floors; the correct floor; the exact-rule map at 12 values of `W`; both ramp formulas |
| `scripts/distancing_experiments/lattice_gap_3d/syn/v4_curv.py` | **raw-geometry** concave-sphere check (real sphere, true Euclidean UDFs, no `g`, no `c`): confirms TABLE 2 at `R = 5, 3`, refutes the second-order table |
| `scripts/distancing_experiments/lattice_gap_3d/syn/v5_misc.py` | exhaustive `C(98,3)` Voronoi-vertex covering radius; solid-angle census; minimal-covering-set subset table; peak sharpness; Lemma M |
| `scripts/distancing_experiments/lattice_gap_3d/syn/v6`, `v7`, `v8` | threshold-rule `W`-scan (ramp vs jump, plateau floor, large `W`); no-guarantee boundaries; `F_exa <= A(n)` on 40 000 directions; non-interval fraction `8.45 %` |

From the seven tracks and two referees:

```
scripts/distancing_experiments/lattice_gap_3d/cp/       critical-point certificate, bootstrap, anatomy, W-plateau, curvature (s1..s31)
scripts/distancing_experiments/lattice_gap_3d/*.py      independent search track: scans, annealing, B&B, exact-rule bootstrap, e2e
scripts/distancing_experiments/lattice_gap_3d/e2e/      end-to-end Z^3 simulators (three), covering certificate, tol-hazard battery
scripts/distancing_experiments/lattice_gap_3d/audit/    lemma-transfer audit, polytope-vertex certificate in exact rationals (a5b, a16)
scripts/distancing_experiments/lattice_gap_3d/alg/      closed forms, LLL/PSLQ from scratch, the irreducibility PROOF (m5_degree.py)
scripts/distancing_experiments/lattice_gap_3d/w/        the certified W-map, plateau endpoints, gap classification
scripts/distancing_experiments/lattice_gap_3d/curv/     exact sphere/cylinder/torus geometry, K constants, TABLES 1 and 2 (k22)
scripts/distancing_experiments/lattice_gap_3d/ref_value/, scripts/distancing_experiments/lattice_gap_3d/ref_structure/    the two referee re-implementations
```
