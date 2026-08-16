# The 2D lattice certification gap: exact constant at W = 3

Working note for the `new-distancing-approach` branch. Companion to `NewDistancingApproach.md`,
which poses the problem; this document answers it in 2D. Prior art is in
`LatticeCertificationGap_PriorArt.md`. The 3D question is open and under separate investigation.

**Status.** The main theorem (upper bound, tightness, and the `W`-range under both admissibility
readings) is proved, with the exceptions explicitly flagged in section 3.8. Three adversarial
referees re-implemented the checks independently; their findings and dispositions are in 3.9. The
curvature law of section 4.5 is exact, not an expansion. Every claim carries an explicit confidence
marker where it is not fully proved.

**Reader's shortcut.** The result is section 1; the practical consequences for the pipeline are
section 6; the two things most likely to bite an implementer are 4.5 (a straight interface is not
the worst case) and 4.6 (a genuine floating-point soundness violation at lattice-aligned normals).

---

## 1. The result

### 1.1 Statement

Setting: lattice `Z^2` with unit spacing, covering radius `r_d = sqrt(2)/2`, narrow-band half-width
`W`, `Sigma` a straight line with unit normal `n` pointing into the exterior. For a witness offset
`w in Z^2 \ {0}` define

```
    g(w,n)  = (|w| - w.n)/2 = |w| sin^2(theta_w / 2)      certification cost
    d0(w,n) = (|w| + w.n)/2                               witness depth at its own threshold
```

so that `|w| = g + d0` identically. Call `w` *admissible* (CONTEXT's threshold rule) iff
`r_d < d0(w,n) <= W`, and put

```
    eps*(n,W) = min { g(w,n) : w admissible },     eps*(W) = max over unit n of eps*(n,W).
```

> **THEOREM.** For `W = 3`,
> ```
>     eps*(3) = eps_c := (3 - sqrt5 - sqrt(2*sqrt5 - 4)) / 4
>             = 0.019202630763796343828423710859901656903488071689333632072250...
> ```
> The maximum is **attained**, exactly on the 8-element dihedral orbit of the direction
> ```
>     alpha* = 15.930625116297946538778691390428 deg,
>     n*     = (cos alpha*, sin alpha*),   defined exactly by   cos alpha* + sin alpha* = sqrt5 - 1,
> ```
> where two witnesses tie: `w = (1,0)` (short, misaligned) and `w = (2,1)` (long, well aligned).
> Only three witnesses per octant are ever needed: the dihedral orbit of `{(1,0), (2,1), (1,1)}`,
> each of which is admissible for *every* `n` in its octant. `eps_c` is an algebraic number of
> degree 4 with minimal polynomial `64x^4 - 192x^3 + 208x^2 - 56x + 1` over `Q`.
>
> **Range of W.** Under the threshold admissibility rule the same value holds exactly on
> ```
>     W in [ sqrt5, W_hi ) = [ 2.236067977499789696, 3.160767557306491827 ),
>     W_hi = (sqrt10 + 2*sqrt5 - 2 + sqrt(2*sqrt5 - 4))/2 .
> ```
> Under the **exact (actual-depth) rule** — the one the pipeline really implements, where a witness
> must be in band at the *target's* depth, not at its own threshold — the plateau is instead
> ```
>     W in [ W_lo', W_hi' ) = [ 2.279483144059777073, 3.178460085208400665 ),
>     W_lo' = (3/2)*sqrt(2*sqrt2 - 2) + sqrt2 - 1/2,   W_hi' = (7*sqrt5 - 5 + 3*sqrt(2*sqrt5-4))/4 .
> ```
> `W = 3` lies comfortably inside both. The two rules give the same constant at `W = 3`; they do
> **not** agree everywhere (section 3.7).

### 1.2 Closed forms

All five expressions below are the same real number; verified to agree to 70 significant digits
(`scripts/distancing_experiments/lattice_gap/syn_v1.py`, pairwise differences `< 1.1e-70`):

```
    eps_c = (3 - sqrt5 - sqrt(2*sqrt5 - 4)) / 4                        [simplest]
          = (sqrt5 - 2) / (2 * (1 + sqrt5 + sqrt(4 + 2*sqrt5)))        [CONTEXT's conjecture]
          = (2 - t) / t^4          with t = sqrt(1 + sqrt5) = sqrt(2*phi), phi = golden ratio
          = (2 - sqrt(2*phi)) / (4*phi^2)
          = sin^2(u),   tan u = 5^(1/4) sin(psi/2) / (1 + 5^(1/4) cos(psi/2)),  psi = arctan(1/2),
                        alpha* = 2u
```

CONTEXT's conjectured surd is therefore **confirmed exactly**, not merely to `1e-17`.

### 1.3 Precision warning (do not quote the 15-digit decimal as a bound)

```
    eps_c                        = 0.0192026307637963438284237108599016569034880716893...
    CONTEXT's quoted decimal     = 0.019202630763796
    eps_c - quoted               = +3.4382842371085990e-16
```

The quoted decimal is simultaneously the truncation *and* the correctly rounded 15-decimal value,
and it lies **below** `eps_c`. So the literal inequality `g <= 0.019202630763796` is **false** on an
angular window about `alpha*` of total width

```
    3.4382842371086e-16 / 0.137236619513691  +  3.4382842371086e-16 / 0.206324130208821
        = 2.5054e-15 + 1.6664e-15 = 4.1718e-15 rad.
```

Rounding to 16 decimals does not repair it either: `round(eps_c, 16) = 0.0192026307637963 < eps_c`.
One must round **up** — use `0.0192026307637964`, or state the theorem with the exact surd. (This
corrects a mislabelled repair in the upper-bound track's own honesty caveat; see 3.9.)

Practical corollary: `eps_c` in IEEE double is `0.019202630763796344 = 0x1.3a9dabc82bb9cp-6`. A
naive double-precision evaluation of the pipeline measures `0.01920263076379647`, about 36 ulps
high, purely from rounding in `cos/sin/hypot`. Do not hard-code a float threshold at `eps_c`
without a guard band.

### 1.4 What is proved and what is not

| Claim | Status |
|---|---|
| `eps*(3) <= eps_c` (upper bound, all `n`) | **Proved.** Three-arc covering of the octant + dihedral reduction. |
| `eps*(n*,3) >= eps_c` (tightness at `n*`) | **Proved.** Finiteness bound `\|w\|^2 <= 9` + exhaustive 28-point enumeration + exact tie in `Q(beta)`. |
| Closed forms of `eps_c`; minimal polynomial | **Proved.** Elementary surd algebra; irreducibility argument checked. |
| Threshold-rule plateau `[sqrt5, W_hi)` | **Proved.** Monotonicity in `W` + the two endpoint evaluations. |
| Exact-rule plateau `[W_lo', W_hi')` | **High-confidence numerical.** Three independent event-based evaluators agree; endpoints are exact "touching interval" identities, but no closed exhaustive argument was written for `W_lo'`. |
| `eps > eps_c` implies certified, at `W = 3`, for every `n` | **High-confidence numerical.** Verified on 90 001 directions: the uncertified set is a single interval `(0, eps*(n,3)]` at every one. Not proved in general — certification is *not* monotone in `eps` at other `W` (section 3.7). |
| Planar `Sigma` is the worst case | **FALSE.** A concave exterior of curvature radius `R` gives `eps_c + K/R`, `K = 0.0283152` (section 4.5). |

---

## 2. Setup and the reduction

### 2.1 The reduction

CONTEXT's derivation is reproduced here for completeness; it was re-verified numerically against
the **raw** pairwise rule `UDF(v1) + UDF(v0) > dist(v1,v0)` by three independent implementations
(400 000, 300 000 and 200 000 random configurations respectively) with **0 mismatches** in every
case, and by a from-scratch end-to-end simulation of the whole pipeline (lattice, true UDF, barrier
marking, connected-components seeding, enrichment to fixpoint) that never uses the formula `g` at
all.

Put the target at the origin at depth `eps > 0`. A witness at integer offset `w` has depth
`d0 = eps + w.n`, so the pairwise rule reads

```
    eps + (eps + w.n) > |w|   <=>   2 eps > |w| - w.n   <=>   eps > g(w,n).
```

Two consequences used constantly:

```
    (I1)  |w| = g(w,n) + d0(w,n),   g >= 0 (Cauchy-Schwarz),  with g = 0 iff w is a positive
          multiple of n.
    (I2)  g(w, .) is the restriction to the unit circle of an AFFINE function of n. Hence the
          locus g(w1,n) = g(w2,n) is the straight line (w2 - w1).n = |w2| - |w1| intersected
          with the circle: a quadratic, so all crossovers are surds of degree <= 4.
    (I3)  g(w, n(alpha)) = |w| (1 - cos(alpha - beta_w))/2, beta_w = angle of w. So g(w, .) is
          unimodal in alpha, with minimum 0 at beta_w; its max on any interval of length < 2*pi
          is at an endpoint.
```

`(I2)` is what makes the whole problem solvable in closed form; `(I3)` is what makes the case
analysis in section 3.5 a two-line endpoint argument.

### 2.2 Two admissibility rules — keep them apart

This distinction is the single biggest source of error in the material this document synthesises,
and it is the one real defect all three referees converged on.

* **Threshold rule** (CONTEXT): `w` is admissible iff `r_d < (|w| + w.n)/2 <= W`, i.e. the witness
  is non-barrier and in band *at the target depth `eps = g(w,n)` where it first certifies*.
* **Exact rule** (what the pipeline does): `w` may be used against a target at depth `eps` iff
  `eps > g(w,n)` **and** `r_d < eps + w.n <= W`, evaluated at the actual `eps`.

CONTEXT justifies the threshold rule with "for larger `eps` the witness depth only grows further".
That is true but **not harmless**: growing depth pushes the witness out of the *top* of the band.
The exact certifying set of a witness is therefore an **interval bounded above**,

```
    J_w(alpha,W) = ( max( g(w,n), r_d - w.n ) ,  min( r_d, W - w.n ) ],
```

and the union of the `J_w` need not be upward closed. Consequently `eps*` is not a priori a `min`
of `g` at all, and the "guarantee" reading (`eps > eps*` implies certified) needs separate
justification. At `W = 3` the two rules coincide and the union *is* upward closed; away from `W = 3`
they can differ drastically. Sections 3.7 and 4.4 give the exact ranges and an explicit
counterexample.

---

## 3. Proof

Throughout, `beta := sqrt(2*sqrt5 - 4) = 0.687121499445024928277131487829`, and

```
    cos alpha* = (sqrt5 - 1 + beta)/2 = 0.961594738472407312343152578280
    sin alpha* = (sqrt5 - 1 - beta)/2 = 0.274473239027382384066021090451
    eps_c      = (1 - cos alpha*)/2   = 0.019202630763796343828423710860
```

### 3.1 Lemma B (dihedral reduction)

Let `D4` be the order-8 group of signed coordinate permutations. Each `R in D4` is orthogonal and
maps `Z^2` onto `Z^2`, so `|Rw| = |w|` and `(Rw).(Rn) = w.n`, whence `g(Rw,Rn) = g(w,n)` and
`d0(Rw,Rn) = d0(w,n)`. Admissibility (either rule) is therefore `D4`-equivariant, and every unit
`n` can be mapped into the closed octant `C = {alpha in [0 deg, 45 deg]}`. **It suffices to prove
everything on `C`.**

(The 16 primitive directions in `[-2,2]^2` listed in CONTEXT are exactly the `D4`-orbits of
`(1,0)`, `(2,1)`, `(1,1)`, of sizes 4 + 8 + 4 = 16; `(0,1)` and `(1,2)` are already inside those
orbits and must not be counted again.)

### 3.2 Lemma A (the barrier test is inactive)

*If `g(w,n) <= 0.2929` then `w` is automatically non-barrier whenever it certifies.*

Proof. A certifying witness has `eps > g`, so its actual depth is `eps + w.n > g + w.n = |w| - g
>= 1 - g` (every nonzero integer vector has `|w| >= 1`). For the barrier test to bite we would need
`1 - g <= r_d`, i.e. `g >= 1 - sqrt(2)/2 = 0.292893218813452476`. QED

Since `eps_c = 0.0192 << 0.2929`, **the entire barrier half of admissibility is irrelevant at
`W = 3`**. This was independently confirmed empirically: dropping the barrier test, making it
non-strict, or substituting the 3D value `sqrt3/2` leaves `eps*(3)` bit-identical; the cheapest
barrier-excluded witness at `n*` costs `0.292893219`. The constant at `W = 3` is set purely by the
band cutoff, as CONTEXT says.

Corollary (finiteness). If `w` is in band and `g(w,n) <= eps_c` then by `(I1)`
`|w| = g + d0 <= eps_c + W = 3.019202630763796`, so `|w|^2 <= 9`: a set of **28** lattice vectors.
The bound is not vacuous at the top — `(3.0192...)^2 = 9.1156 > 9` — so norm-9 vectors genuinely
must be enumerated.

*(An interior-side witness, `eps + w.n < 0`, can never certify: the rule would demand
`-w.n > |w|`. In the minimisation it is harmless because then `2g = |w| - w.n >= |w|`, i.e.
`g >= |w|/2 >= 1/2`. Note `g >= |w| >= 1`, as written in one of the source proofs, is **false**:
at `n*`, `w = (-1,0)` has `|w| = 1` but `g = 0.980797369236204 < 1`.)*

### 3.3 Lemma C (crossover lemma)

Let two witness directions be separated by `psi in (0,pi)`, the witnesses having lengths `A`, `B`.
Parametrise `n` inside the sector by `theta in [0,psi]` measured from the `A`-ray. Then
`g_A(theta) = A sin^2(theta/2)` is strictly increasing from 0 and `g_B(theta) = B sin^2((psi-theta)/2)`
strictly decreasing to 0, so `min(g_A,g_B)` is maximised at the unique tie, and with `u = theta_x/2`,
`k = sqrt(B/A)`:

```
    tan u = k sin(psi/2) / (1 + k cos(psi/2)),
    X(A,B,psi) = A sin^2 u = B (1 - cos psi) / ( 2 (1 + 2 k cos(psi/2) + k^2) ).
```

Proof. `min(increasing, decreasing)` increases before the tie and decreases after it. At the tie,
`sqrt(A) sin u = sqrt(B) sin(psi/2 - u)`; expanding and dividing by `cos u` gives the `tan u`
formula. With `N = k sin(psi/2)`, `D = 1 + k cos(psi/2)`, `N^2 + D^2 = 1 + 2k cos(psi/2) + k^2` and
`A N^2 = B (1 - cos psi)/2`, so `A sin^2 u = A N^2/(N^2+D^2)` is the stated `X`. QED

Equivalently, and more usefully for exact work, by `(I2)` the tie of `w1, w2` is the **linear**
condition `(w2 - w1).n = |w2| - |w1|`. The three ties in the octant are:

| pair | tie equation | crossover cost | alpha |
|---|---|---|---|
| `(1,0), (2,1)` | `n_x + n_y = sqrt5 - 1` | `eps_c = (3 - sqrt5 - beta)/4 = 0.019202630763796` | `15.930625116298 deg` |
| `(2,1), (1,1)` | `n_x = sqrt5 - sqrt2` | `eps_2 = (2sqrt2 - sqrt5 - sqrt(2sqrt10-6))/2 = 0.011330789030059` | `34.729139026042 deg` |
| `(1,0), (1,1)` | `n_y = sqrt2 - 1` | `eps_1 = (1 - sqrt(2sqrt2-2))/2 = 0.044910139437773` | `24.469800520702 deg` |

Specialising Lemma C to `A = 1`, `B = sqrt5`, `psi = arctan(1/2)` gives `k = 5^(1/4)`,
`2k cos(psi/2) = sqrt(4 + 2 sqrt5)`, `B(1 - cos psi) = sqrt5 - 2`, i.e. **exactly** CONTEXT's
conjectured surd. Alternatively, and with no trigonometry at all: `n_x + n_y = sqrt5 - 1` together
with `n_x^2 + n_y^2 = 1` gives the quadratic `2 n_x^2 - 2(sqrt5-1) n_x + (5 - 2 sqrt5) = 0` (note the
constant term is `(5 - 2 sqrt5)/2` after halving, **not** `3 - sqrt5`), whose discriminant is
`2 sqrt5 - 4 = beta^2`, so `n_x = ((sqrt5-1) + beta)/2` and `eps_c = (1 - n_x)/2`.

Ordering, certified by exact sign tests: `0 < alpha* = 15.9306 < arctan(1/2) = 26.5651 <
alpha** = 34.7291 < 45`, and `eps_2 < eps_c` with margin `eps_c - eps_2 = 0.007871841733737`. This
last inequality is the "the wide gap dominates" step: the wide angular gap `[0, arctan(1/2)]`
(26.565 deg) produces a *larger* crossover cost than the narrow gap `[arctan(1/2), 45 deg]`
(18.435 deg), despite length-weighting favouring `(2,1)` in the narrow gap.

### 3.4 Simplification of the surd

With `phi = (1+sqrt5)/2` and `t = sqrt(1+sqrt5) = sqrt(2 phi)`:

```
    phi^3 = 2 + sqrt5,  sqrt5 - 2 = phi^-3,  3 - sqrt5 = 2 phi^-2,  2 sqrt5 - 4 = 2 phi^-3,
    t^2 = 2 phi,  t^4 = 6 + 2 sqrt5 = 2 t^2 + 4,
    sqrt(4 + 2 sqrt5) = t^3/2,   beta = sqrt(2 sqrt5 - 4) = t (3 - sqrt5)/2.
```

Then `(sqrt5-2)/(2(1+sqrt5+sqrt(4+2sqrt5))) = (t^2-3)/(t^2(2+t)) = (2-t)/t^4` (cross-multiplying
reduces to `t^4 = 2t^2 + 4`), and `(3 - sqrt5 - beta)/4 = (3-sqrt5)(2-t)/8 = (2-t)/(6+2sqrt5) =
(2-t)/t^4`. All five forms coincide.

**Minimal polynomial.** From `4x = 3 - sqrt5 - beta`: `beta = 3 - 4x - sqrt5`; squaring gives
`8 sqrt5 (1-x) = 16x^2 - 24x + 18`; squaring again and dividing by 4 gives

```
    64 x^4 - 192 x^3 + 208 x^2 - 56 x + 1 = 0.
```

Its roots are `(3 - sqrt5 +- beta)/4` (real: `0.0192026307637963` and `0.3627633804863088`) and
`(3 + sqrt5 +- i sqrt(4+2sqrt5))/4` (non-real). A rational factorisation into quadratics would have
to pair the complex conjugates, hence also the two real roots — but their sum `(3-sqrt5)/2` is
irrational. Rational roots are excluded by the rational-root test. Hence the quartic is irreducible
and `eps_c` has degree 4.

### 3.5 Upper bound: `eps*(n,3) <= eps_c` for every `n`

By Lemma B work in `C = [0 deg, 45 deg]`. Split it at the two crossovers and use `(I3)`:

* `alpha in [0, alpha*]`: `beta_(1,0) = 0 <= alpha`, so `g((1,0),.)` is nondecreasing here and
  `g((1,0), n(alpha)) <= g((1,0), n(alpha*)) = eps_c`, strictly for `alpha < alpha*`.
* `alpha in [alpha*, alpha**]`: use `(2,1)`, whose direction `arctan(1/2) = 26.5651 deg` lies
  *inside* this interval. By `(I3)` the max is at an endpoint:
  `g((2,1),.) <= max(eps_c, eps_2) = eps_c`.
* `alpha in [alpha**, 45 deg]`: `beta_(1,1) = 45 deg >= alpha`, so `g((1,1),.)` is nonincreasing and
  `g((1,1), n(alpha)) <= g((1,1), n(alpha**)) = eps_2 < eps_c`.

The three intervals cover `C`, and equality with `eps_c` holds only at the single point `alpha*`.

Equivalently, as an arc-covering statement (independently checked): the sets where each witness has
`g <= eps_c` are `[0, alpha_A]`, `[psi - d5, psi + d5]`, `[45 - d2, 45]` with `cos alpha_A = 1-2eps_c`,
`cos d5 = 1 - 2 eps_c/sqrt5`, `cos d2 = 1 - 2 eps_c/sqrt2`; arcs A and B meet **exactly** at
`alpha*` (that is literally the defining identity `2 cos alpha* + sin alpha* = sqrt5 - 2 eps_c`),
and B, C overlap by 5.58 deg since `cos(d5+d2) = 0.913419569177 < 3/sqrt10 = 0.948683298051`.
Measured: arc A `= [0, 15.93062511629795]`, arc B `= [15.93062511629795, 37.19947723785803]`,
arc C `= [31.61669868685788, 45]` deg.

**Admissibility of the three witnesses is unconditional on the octant**, not merely at the
crossovers — this closes the "admissibility only checked at the optimum" hole:

| `w` | `max g` on the octant | `d0` range over the whole octant | `r_d < d0 <= 3` |
|---|---|---|---|
| `(1,0)` | `(1 - cos45)/2 = 0.146450494871` | `[0.853553390593, 1.000000000000]` | yes |
| `(2,1)` | `(sqrt5 - 2)/2 = 0.118033988750` | `[2.118033988750, 2.236067977500]` | yes |
| `(1,1)` | `(sqrt2 - 1)/2 = 0.207106781187` | `[1.207106781187, 1.414213562373]` | yes |

Combining with Lemma B: for every unit `n` there is an admissible `w` (in the `D4`-orbit of those
three) with `g(w,n) <= eps_c`. Hence `eps*(3) <= eps_c`.

### 3.6 Lower bound (tightness): `eps*(n*,3) = eps_c`

By the Corollary to Lemma A, any in-band `w` with `g(w,n*) <= eps_c` has `|w|^2 <= 9`: 28 vectors.
Exhaustive evaluation at `n*` in 60-digit arithmetic (and, redundantly, over the far larger set
`|a|,|b| <= 20`, `scripts/distancing_experiments/lattice_gap/syn_v2.py`) yields **exactly two** in-band vectors with `g <= eps_c`:

```
    (1,0)  |w| = 1        g = 0.0192026307637963438...   d0 = 0.9807973692362036562...
    (2,1)  |w| = sqrt5    g = 0.0192026307637963438...   d0 = 2.2168653467359933526...
```

Both are non-barrier and in band. Every other in-band vector has `g >= 2 eps_c`; the next-cheapest
admissible cost is **exactly** `2 eps_c = 0.038405261527593` at `w = (2,0) = 2*(1,0)`, a factor-2
isolation. (The cheapest *primitive* runner-up is `(1,1)` at `(sqrt2 + 1 - sqrt5)/2 =
0.089072792437 = 4.6386 eps_c`.)

**The tie is exact, not numerical.** Work in the field `K = Q(beta)`, `beta = sqrt(2 sqrt5 - 4)`, so
`sqrt5 = (beta^2+4)/2` and `beta^4 = 4 - 8 beta^2`. Then

```
    cos alpha* = (beta^2 + 2 beta + 2)/4,     sin alpha* = (beta^2 - 2 beta + 2)/4,
    g((1,0),n*) = (1 - cos alpha*)/2                = 1/4 - beta/4 - beta^2/8,
    g((2,1),n*) = (sqrt5 - 2 cos alpha* - sin alpha*)/2 = 1/4 - beta/4 - beta^2/8.
```

Identical reduced tuples, hence identical real numbers. Equivalently, `cos + sin = sqrt5 - 1` and
`cos^2 + sin^2 = 1` are both exact identities in `K` (verified: residual `0E-69` at 70 digits).

**The near-miss `(3,1)` is excluded with a large margin, not by rounding.** At `n*`:

```
    g((3,1), n*)  = 0.001510102861888  = eps_c / 12.716     (12.7x cheaper -- it would win)
    d0((3,1), n*) = (sqrt10 + 2 sqrt5 - 2 + beta)/2 = 3.160767557306491827  >  W = 3
```

a band excess of `0.160767557306` — about `1e15` ulps, fifteen orders of magnitude from any rounding
scale. Two independent exclusions: the direct band violation above, and the size test
`|(3,1)| = sqrt10 = 3.162277660168 > W + eps_c = 3.019202630764`. Under the exact rule the exclusion
is wider still: `eps_c + (3,1).n* = 3.178460085208 > 3`. The hand-checkable rational version:
`sqrt10 > 79/25`, `2 sqrt5 > 559/125` (from `559^2 = 312481 < 312500 = 5*250^2`, so
`sqrt5 > 559/250`), `beta > 17/25` (from `beta^2 = 2 sqrt5 - 4 > 0.472 > 0.4624`), summing to
`> 8.28 > 8`, i.e. `d0((3,1)) > 3`. Note that `(4,1)` is even cheaper (`g = 0.001126716350`) but
even further out (`d0 = 4.121978909267`), and `(7,2)` cheaper still (`g = 1.2096e-7`,
`d0 = 7.280109768321`).

Combining 3.5 and 3.6: `eps*(3) = eps_c`, attained exactly on the `D4`-orbit of `alpha*`. QED

**Strengthened form (chaining does not help).** The last hop of *any* certification chain obeys the
same inequality `eps > g(p,n)` with `p` the lattice point holding the stored UDF. By the
Corollary to Lemma A, `g(p,n*) < eps_c` plus in-band forces `|p|^2 <= 9`, contradicting the
enumeration. So every candidate final hop has depth `>= 3.160767557306 > W` and holds no stored
UDF. Independently confirmed end-to-end: allowing newly certified *barrier* points to act as
witnesses in the closure changes `eps*(alpha)` at **0 of 451** sampled angles.

### 3.7 The range of W — corrected

This is where the source material was wrong, and the correction matters.

**Threshold rule.** `eps*(.,W)` is non-increasing in `W` (admissible sets are nested), so the
plateau follows from the two endpoint evaluations:

* *Lower endpoint `sqrt5`, sharp.* At `W = sqrt5` the covering argument of 3.5 still goes through
  (`d0((2,1)) <= sqrt5` with equality exactly at `alpha = arctan(1/2)`), so `eps*(sqrt5) <= eps_c`;
  and 3.6 gives equality. For any `W < sqrt5`, take `n = (2,1)/sqrt5`: then `(2,1)` itself has
  `d0 = sqrt5 > W` and is excluded, and by `(I1)` any admissible `w` with `g < 0.0363` needs
  `|w|^2 <= 5`; enumerating those 20 vectors gives a minimum of
  `sqrt2/2 - 3 sqrt5/10 = 0.036286387936611` at `w = (1,1)`. So `eps*(W) >= 0.0362864 > eps_c` for
  every `W < sqrt5`, i.e. the plateau jumps by a factor 1.89 exactly at `sqrt5`.
  *Robustness caveat:* this endpoint is **knife-edge on the non-strict band test**. `(2,1)` attains
  `d0 = sqrt5` exactly; with `d0 < W` instead of `d0 <= W` the value at `W = sqrt5` would be
  `0.0362864`, not `eps_c`.
* *Upper endpoint `W_hi`, sharp.* For `W < W_hi` every `w` with `g(w,n*) < eps_c` has
  `d0(w,n*) >= W_hi > W` (the `|w|^2 <= 10` disk contains exactly one such vector, `(3,1)`), so
  `eps*(n*,W) = eps_c`. For `W >= W_hi`, `(3,1)` is admissible at `n*` and `eps*(n*,W) <= 0.0015101`;
  by 3.5, `alpha*` is the only octant direction where `eps_c` can be attained, so the max drops
  below `eps_c`.
  `W_hi = (sqrt10 + 2sqrt5 - 2 + beta)/2 = 3.160767557306491827`, an algebraic number of degree 8
  with minimal polynomial `x^8 + 8x^7 + 2x^6 - 120x^5 - 209x^4 + 240x^3 + 1188x^2 + 664x + 1`
  (computed twice, independently, by exact linear algebra on the 8-monomial basis of
  `Q(sqrt2, sqrt5, beta)`).
  *Small open item:* the statement "the plateau is `[sqrt5, W_hi]` in the sup sense", i.e.
  `sup_n eps*(n,W) < eps_c` for `W > W_hi`, needs a compactness remark that was never written
  (`C` minus a neighbourhood of `n*` is compact and `min(g1,g2)` is continuous there). The remark is
  routine but should be stated. Note also that for `W > W_hi` the sup is likewise not attained, and
  it tends continuously to `eps_c` as `W` decreases to `W_hi` — there is no uniform gap.

**Exact rule.** The claim that appeared in one source — "the proof uses `W` only through `sqrt5 <= W`,
so the bound holds for every `W >= sqrt5`" — is **false** under the exact rule. Explicit
counterexample, reproduced independently here (`scripts/distancing_experiments/lattice_gap/syn_v4.py`):

```
    W = sqrt5 = 2.2360679774997897,  alpha = 24 deg,  target depth eps = 0.03 = 1.56 * eps_c

    (2,1): g = 0.001120210 < eps   (cheap enough)
           threshold d0 = 2.234947768 <= W    -> "admissible" under the threshold rule
           ACTUAL depth  = 0.03 + 2.233827558 = 2.263827558 > W  -> OUT OF BAND, unusable
    (1,0): g = 0.043227271 > eps   -> does not certify
    (1,1): g = 0.046965731 > eps   -> does not certify
    all other w (exhaustive):      -> none

    Uncertified intervals at that direction: (0, 0.001120] and (0.002240, 0.043227].
```

The uncovered set is *not* an interval — this is exactly the non-monotonicity warned about in 2.2.
Under the exact rule the failure region is `W in [sqrt5, W_lo')`, and the plateau is

```
    W_lo' = (3/2) sqrt(2 sqrt2 - 2) + sqrt2 - 1/2 = 2.279483144059777073
    W_hi' = (7 sqrt5 - 5 + 3 beta)/4              = 3.178460085208400665
```

Both endpoints are "touching interval" events: at `W_lo'` the top of the `(2,1)` interval meets the
bottom of the `(1,0)` interval exactly (`W_lo' - (2 cos alpha_c + sin alpha_c) = eps_1`), and at
`W_hi'` the top of the `(3,1)` interval meets `eps_c` exactly
(`W_hi' - (3 cos alpha* + sin alpha*) = eps_c`). Since `(a,c] u (c,b] = (a,b]`, the interval is
closed on the left and open on the right. Verified directly here at `W_lo' +- 1e-7` and
`W_hi' +- 1e-7` with ultrafine local scans around the three critical angles:

| `W` | `eps*_exact` | at `alpha` |
|---|---|---|
| `W_lo' - 1e-7` | `0.044910139436` | `24.469801 deg` |
| `W_lo' + 1e-7` | `0.019202630764` | `15.930625 deg` |
| `W_hi' - 1e-7` | `0.019202630764` | `15.930625 deg` |
| `W_hi' + 1e-7` | `0.011330789030` | `34.729139 deg` |

Confidence: **high, numerical.** Three independent event-based evaluators (two from the source
tracks, one written for this document) agree on both endpoints to `<1e-13`, and both endpoints have
exact symbolic identities. What is missing is a closed exhaustive argument that no *other* critical
direction transitions later than `W_lo'`.

**Summary of `eps*(W)` on both readings** (all rows re-measured for this document,
`scripts/distancing_experiments/lattice_gap/syn_v3.py`, `syn_v5.py`, `syn_v7.py`):

| `W` range | `eps*` threshold rule | `eps*` exact rule |
|---|---|---|
| `[1, sqrt2)` | `(2-sqrt2)/4 = 0.146446609407` @ 45 deg | `r_d = 0.707106781187` — **no guarantee at all** |
| `[sqrt2, 1 + r_d = 1.707106781187)` | `0.044910139438` @ 65.530 deg | `r_d` — no guarantee at all |
| `[1 + r_d, W_c1 = 2.235320491061)` | `0.044910139438` | `0.044910139438` @ 65.530 deg |
| `[W_c1, sqrt5)` | falling ramp `0.0449101 -> 0.0362864` | `0.044910139438` |
| `[sqrt5, W_lo' = 2.279483144060)` | **`eps_c = 0.019202630764`** | `0.044910139438` |
| `[W_lo', W_hi = 3.160767557306)` | **`eps_c`** | **`eps_c`** |
| `[W_hi, ~sqrt10)` | falling ramp `eps_c -> 0.0113308`, width `0.0015101` | **`eps_c`** |
| `[~sqrt10, W_hi' = 3.178460085208)` | `0.011330789030` @ 34.729 deg | **`eps_c`** |
| `[W_hi', 3.605254829609)` | `0.011330789030` | `0.011330789030` |
| `[3.605254829609, ...)` | `0.010574711048` @ 11.805 deg | `0.010574711048` |

Corrections to CONTEXT's coarse `0.01`-step table that this establishes:

* `W >= 1.30 -> 0.146446609` is wrong under both readings: under the threshold rule the plateau
  starts at `sqrt2`, and under the exact rule the scheme gives **no guarantee whatsoever** below
  `W = 1 + r_d = 1.707106781187` (at `alpha = 0` the only useful witnesses sit at depth `eps + 1`,
  so a barrier target with `eps in (W-1, r_d]` has no witness at all — verified end-to-end at
  `W = 1.35, 1.5, 1.70`, flipping at `W = 1.71`).
* `W >= 2.24` is a `0.01`-grid artifact; the exact threshold-rule breakpoint is `sqrt5`.
* `W >= 3.17` is likewise an artifact; the exact breakpoints are `W_hi` (threshold) and `W_hi'`
  (exact).
* `eps*(W)` is **not** a pure step function. Above `W_hi` there is a strictly decreasing continuous
  ramp of width `sqrt10 - W_hi = g((3,1),n*) = 0.0015101` (sampled: `W_hi + 1e-6 -> 0.019199644`,
  `+1e-4 -> 0.018900068`, `+1e-3 -> 0.015609881`). There is a second ramp just below `sqrt5`. The
  two tracks that bisected the top of the first ramp disagree at `2.8e-7` (`3.162277382` vs exactly
  `sqrt10 = 3.162277660`); the discrepancy is immaterial here and is left unresolved.

### 3.8 Explicit list of what remains unproved

1. **Exact-rule plateau endpoints** (`W_lo'`, `W_hi'`): high-confidence numerical, symbolic
   endpoint identities, no exhaustive critical-direction argument. Does not affect `W = 3`.
2. **Upward-closedness of the certified set at `W = 3`** (i.e. that `eps > eps_c` really does imply
   certified, for every `n`, not just that `eps_c` is the min cost): verified on 90 001 directions —
   the uncertified set was a single interval `(0, eps*(n,3)]` at every one, with 0 exceptions — but
   not proved. It is genuinely a theorem that needs proving, because certification is **not**
   monotone in `eps` in general: at `n*`, witness `(2,1)` certifies a target at `eps = 0.8`
   (depth `2.997663 <= 3`) but not at `eps = 1.0` (depth `3.197663 > 3`); `(1,0)` certifies at
   `eps = 2.0` but not at `eps = 2.05`. At `W = 3` the coverage is rescued by a witness hand-off
   chain `(1,0) -> (0,1) -> (0,-1) -> (-1,1) -> (-1,-1) -> (-2,1) -> (-2,0) -> (-2,-1)` as `eps`
   grows; nobody wrote the argument that this chain always exists.
3. **`sup` versus `max` for `W > W_hi`**: needs the (routine) compactness remark of 3.7.
4. **The `Q(beta)` tie argument** assumes only that reduced tuples denote equal reals, which is
   immediate; no irreducibility of `beta`'s minimal polynomial is needed for that direction. This
   is fine as written.
5. **The pipeline facts imported from CONTEXT** (1-Lipschitz UDF, barrier definition, "non-barrier
   in-band exterior points are certified by plain connected components") were not re-derived. They
   were, however, exercised end-to-end by a from-scratch simulation that reproduces `eps_c` to 40
   digits without ever using the formula `g`.
6. **Everything is 2D and planar.** See sections 4.5 and 7.

### 3.9 Referee findings, and their disposition

| Finding | Disposition |
|---|---|
| "The bound holds verbatim for every `W >= sqrt5`" — non sequitur, and false under the exact rule on `[sqrt5, 2.27)` | **Fixed.** Section 3.7 states both rules separately with their own ranges; the counterexample is reproduced. |
| Rounding-direction error in the honesty caveat: `round(eps_c,16) = 0.0192026307637963 < eps_c`, so "correctly rounded" is not a repair | **Fixed.** Section 1.3: one must round *up*. |
| Truncation window misquoted as `3.3e-15` rad | **Fixed:** `4.1718e-15` rad. |
| "`g >= |w| - W = 0.1607`" for `(3,1)` | **Fixed:** `sqrt10 - 3 = 0.162277660168`. |
| "`g >= |w| >= 1`" for interior-side witnesses | **Fixed:** correct bound is `g >= |w|/2 >= 1/2`. Section 3.2. |
| Quadratic constant term stated as `3 - sqrt5` | **Fixed:** `(5 - 2 sqrt5)/2`. Section 3.3. |
| "The 16 primitive directions are the orbit of `{(1,0),(2,1),(1,1)}` **together with** `(0,1),(1,2)`" — double counts | **Fixed.** Section 3.1. |
| Machine certificate contains two tautological rows (`E1 <= E1` at the crossover pieces) | **Disclosed.** The two crossover pieces are asserted analytically, not independently checked; the analytic justification (monotonicity toward the crossover + the crossover value) is correct and is reproduced as section 3.5. The "32/32 checks pass" headline should be read as 30 independent checks plus 2 analytic assertions. |
| Sharpness declared "open" in one source | **Closed.** The finiteness bound `\|w\| <= W + eps_c` gives `\|w\|^2 <= 9`; enumeration then gives equality. Section 3.6. |
| Global upper bound declared "out of scope" in one source | **Closed.** Section 3.5. |
| Table row "`W < 2.2353 -> 0.0449101`" false on `[1, sqrt2)`; row "`then -> 0.0113308`" false for `W >= 3.6053` | **Fixed.** Section 3.7 table. |
| Octic conjugate list included 4 non-roots (wrong middle sign) | **Noted.** The octic itself is correct (recomputed independently); the verification narrative was wrong. Correct conjugates: `+-sqrt10/2 + (sqrt5-1) +- beta/2` (real) and `+-sqrt10/2 - (sqrt5+1) +- i sqrt(4+2sqrt5)/2` (complex). |
| `sup` vs `max` for `W > W_hi` needs compactness | **Stated as an open item**, 3.7 and 3.8.3. |
| "`eps_c` is the sup of uncertified depths at `n*`" inferred from a single point check | **Stated as an open item**, 3.8.2, with the non-monotonicity counterexample. |
| 18-digit values of `beta`, `cos alpha*`, `sin alpha*` wrong in their last two digits (float round-trip in the printer) | **Fixed.** Recomputed at 70 digits; section 3 header carries the correct 30-digit values. |
| `W_lo = sqrt5` is knife-edge on the non-strict band test | **Stated**, 3.7. |

---

## 4. Tightness, the critical configuration, and brittleness

### 4.1 Independent confirmation that `eps_c` is attained and not exceeded

| strategy | largest `eps` found | `alpha` |
|---|---|---|
| End-to-end pipeline simulation, 40-digit Decimal, no use of `g` | `0.019202630763796343828423710859901656903` | `15.93062511629794653877869139042797 deg` |
| Complete closed-form critical-angle certificate (all cost ties, band and barrier switches, stationary points), full circle, witness radii 3.146 / 4 / 6 | `0.0192026307637964` | `15.930625116298 deg` |
| Hill climbing, 4000 restarts + simulated annealing, 60 chains | `0.0192026307637964` | `15.930625116 deg` |
| Fine sweep, 4e6 directions, every local max refined | `0.0192026307637964` | `15.930625116298 deg` |
| Exact rational adversarial probe, 120 461 directions concentrated to `1e-40` from the crossovers | `eps_c - 2.8e-40` | — |
| Brute force on the original problem (random lines, real lattice, real distances) | `0.0192026307637963` | `15.930625116298 deg` |
| Blind `(alpha, offset)` grid sweeps, ~300 000 trials | max `0.018759` | grid-limited |

**Nothing exceeded `eps_c`.** The transition is sharp and the supremum is attained: at depth exactly
`eps_c` the point is uncertified (the pairwise rule is strict and the slack is exactly 0); at
`eps_c*(1+1e-9)` it is certified by both minimisers simultaneously.

Search-radius stability: witness sets `|w| <= 3.146`, 4, 6, 8, 16 and boxes `[-4,4]^2`, `[-8,8]^2`,
`[-16,16]^2` give **bit-identical** `eps*(alpha)` at 60 000 directions on the full circle. The
finite reduction is exact, not a heuristic: the cheapest cost any witness with `|w| > 3.146447` can
attain while in band is `sqrt10 - 3 = 0.162277660`, above the octant-wide bound `0.146446609`.

Box-size independence of the end-to-end simulation: identical to the last double bit for lattice
boxes `L = 5, 6, 7, 8, 10, 12, 16, 20`. The exterior non-barrier in-band set was a single
4-connected component in all 36 100 trials of the first scan, with no leaks across the interface.

### 4.2 Anatomy of the critical configuration

At `alpha*` the marginal configuration is a **double tangency**: the target ball `B(v1, eps_c)` is
tangent to `Sigma` and simultaneously externally tangent to *both* witness balls, which sit on
opposite tangential sides of the target.

| quantity | `w = (1,0)` | `w = (2,1)` |
|---|---|---|
| `\|w\| = dist(v0,v1)` | `1` | `2.236067977500` |
| `theta_w` (angle to `n`) | `15.930625116 deg` | `10.634426061 deg` |
| `w.n` | `0.961594738472` | `2.197662715972` |
| `w.t` (tangential offset) | `-0.274473239027` | `+0.412648260418` |
| witness depth `d0` | `0.980797369236 = (1 + cos alpha*)/2` | `2.216865346736 = (sqrt5 + 2cos + sin)/2` |
| `d0 + eps_c - \|w\|` (rule slack) | `0` exactly | `0` exactly |
| `\|w.t\|` vs `2 sqrt(d0 * eps_c)` | equal to `2e-70` | equal to `4e-70` |

The two tangential offsets have opposite signs and span
`0.274473239027 + 0.412648260418 = 0.687121499445 = cos alpha* - sin alpha* = beta`. Both slack
values are exactly zero in `Q(beta)`; in IEEE double they evaluate to `0.0` and `-4.44e-16` (one ulp
of `sqrt5`).

**The tie is exact but not bitwise identical in floating point.** Symbolically the difference is
`0E-70`. Naive double evaluation gives

```
    g((1,0), n*) = (1 - cos alpha*)/2              -> 0x1.3a9dabc82bba0p-6   ( 4 ulp high)
    g((2,1), n*) = (sqrt5 - 2cos - sin)/2          -> 0x1.3a9dabc82bbb0p-6   (20 ulp high)
    correctly rounded eps_c                        -> 0x1.3a9dabc82bb9cp-6
```

16 ulps apart, `5.551e-17` absolute, `2.891e-15` relative. This is entirely cancellation noise: the
amplification factors are `1/(2 eps_c) = 26.0` and `sqrt5/(2 eps_c) = 58.2`, predicting `5.7e-15` and
`1.3e-14` relative error. "Bitwise identical in double" would be a false claim.

**The peak is a kink, not a smooth maximum.** One-sided slopes of the lower envelope at `alpha*`:

```
    d g((1,0))/d alpha = + sin(alpha*)/2              = +0.137236619513691 / rad
    d g((2,1))/d alpha = (2 sin alpha* - cos alpha*)/2 = -0.206324130208821 / rad
    derivative jump    = -beta/2 = -(cos alpha* - sin alpha*)/2 = -0.343560749722512 / rad
```

Peak sharpness: losing 1% / 5% / 10% of `eps_c` requires only `0.0802` / `0.4009` / `0.8017` deg of
misorientation on the left branch, `0.0533` / `0.2666` / `0.5333` deg on the right.

**Why `(3,1)` and `(7,2)` are the cheap directions.** The certified continued fraction of
`tan alpha* = 0.285435462618495` is

```
    [0; 3, 1, 1, 72, 1, 1, 1, 1, 1, 5, 1, 2, 13, ...]
```

Its convergent directions `(3,1)`, `(4,1)`, `(7,2)` are exactly the ultra-cheap witnesses
(`g = 1.51e-3`, `1.13e-3`, `1.21e-7` — the huge partial quotient 72 explains the last one), and all
of them are far out of band (`d0 = 3.16`, `4.12`, `7.28`). **The `W = 3` constant is set purely by
the band cutoff amputating the good rational approximations**, leaving the two short vectors to
fight it out. Note also that `(1,0)`/`(2,1)` and `(2,1)`/`(1,1)` are unimodular (Farey-neighbour)
pairs — `det = 1` in both cases — so the octant partition `[0, alpha*] u [alpha*, alpha**] u
[alpha**, 45]` is exactly a Stern-Brocot / Farey bracketing of the direction. See section 5.

Mirror structure: the tie identity `1 + cos + sin = sqrt5` is symmetric under `cos <-> sin`, so the
mirror pair `(0,1)`/`(1,2)` also ties at the *same* `alpha*`, at cost `0.362763380486` (18.9x
`eps_c`, non-binding). The mirrored worst case is the direction `90 - alpha* = 74.069374883702 deg`
listed in CONTEXT's table — the same configuration, not a second one.

### 4.3 Non-vacuity and scale

```
    r_d - eps_c = 0.707106781186548 - 0.019202630763796 = 0.687904150422751 > 0
```

so the worst-case target is genuinely a barrier voxel whose sign connected components cannot settle.
`eps_c / r_d = 0.02716`: the pairwise rule reaches `r_d / eps_c = 36.82x` deeper than plain
connected components. `eps_c / W = 0.0064`.

Robustness of the verdict to convention: no in-band candidate at `n*` lies within `0.0576` of the
band edge (`(3,0)` at `2.942392`) and none within `0.0184` of the barrier edge (`(0,-2)` at
`0.725527`), so swapping `<=` for `<` in either admissibility test changes nothing at `W = 3`.

### 4.4 Brittleness with respect to `W`

At `W = 3` the constant is robust:

* `0.763932` of slack below the threshold-rule lower edge `sqrt5`, `0.160768` (5.36% of `W`) below
  the upper edge `W_hi`;
* `0.720517` above the exact-rule lower edge `W_lo'`, `0.178460` (5.95% of `W`) below `W_hi'`.

Outside those windows the behaviour degrades non-gracefully:

* Crossing `sqrt5` downward (threshold rule) the constant **jumps** by a factor 1.89, to
  `sqrt2/2 - 3 sqrt5/10 = 0.036286387936611`, because `(2,1)` leaves the band in a sliver around its
  own direction where its depth reaches `|w| = sqrt5`.
* Crossing `W_lo'` downward (exact rule) the constant jumps by a factor 2.34, to `0.044910139438`.
* Crossing `W_hi` upward the constant *falls* continuously along a `0.0015101`-wide ramp — the
  benefit of admitting `(3,1)` arrives gradually, not as a step.
* The width of the angular window in which the worst case lives collapses as `W` approaches `W_hi'`
  from below. Near `alpha*` the gap between the top of the `(3,1)` coverage interval and the
  `(1,0)`/`(2,1)` envelope has width `W_hi' - W - h'*|alpha - alpha*|` with `h' = 0.275412` on the
  left and `0.068149` on the right. At `W = 3` that window is tens of degrees wide; at
  `W = W_hi' - 1e-7` it is about `1e-4` deg wide (`2.1e-5` deg on the left, `8.4e-5` deg on the
  right). A grid search coarser than that silently misses the maximum (this
  happened during the verification for this document, and is why the endpoint checks in 3.7 use
  ultrafine local scans around the known critical angles).

### 4.5 Non-planar `Sigma`: the exact curvature law

A straight `Sigma` is **not** the worst case, and the correction is available in closed form -- no
expansion needed. The key is the identity noted in `(I1)`: since `|w|^2 = (w.n)^2 + p^2` with `p` the
tangential component of `w`,

```
    g * d0  =  (|w| - w.n)/2 * (|w| + w.n)/2  =  (|w|^2 - (w.n)^2)/4  =  p^2 / 4      (exact)
```

For a circular `Sigma` of radius `R`, solving `eps + d0 = |w|` against the true circle rather than
against the tangent line gives, exactly:

```
    eps_thr(w)  =  R(|w| - w.n) / (2R - |w| - w.n)  =  g / (1 - d0/R)     concave exterior
                =  R(|w| - w.n) / (2R + |w| + w.n)  =  g / (1 + d0/R)     convex exterior
```

so **curvature acts as a pure multiplicative rescaling of each witness's cost**, by a factor that
depends only on that witness's own threshold depth relative to `R`. Both reduce to `g` as
`R -> infinity`. To first order, using `g*d0 = p^2/4`,

```
    eps_thr  ~  g + g*d0/R  =  g + p^2/(4R)
```

i.e. the penalty is exactly **half the sagitta** of the arc spanning the witness's tangential offset.
Concave hurts because the witness sits on a chord and the surface bends toward it, shrinking its
ball; convex helps symmetrically.

Perturbing the tie at `alpha*` (one-sided slopes `s1`, `s2`; tangential offsets `p1 = -0.274473`,
`p2 = +0.412648` from the table in 4.2) gives the first-order constant

```
    eps*(R) ~ eps_c + K/R,   K = (s1 p2^2 - s2 p1^2) / (4 (s1 - s2)) = 0.028315226153961
```

which is also the slope-weighted mean of `p^2`, divided by 4. Because `s1 > 0 > s2`, both terms are
positive: `K > 0` always. Note the two tied witnesses have *different* offsets, `p^2 = 0.0753` vs
`0.1703`, so curvature penalises the long well-aligned witness `(2,1)` about **2.3x harder** and
breaks the tie toward `(1,0)`, sliding the critical direction.

**Exact values** (from the closed form above, `scripts/distancing_experiments/lattice_gap/curv2.py`; the planar control reproduces
`eps_c` to all printed digits):

| `R` | concave, exact | `eps_c + K/R` | exact / first order | convex, exact | `eps_c - K/R` | `alpha*` (deg) |
|---:|---:|---:|---:|---:|---:|---:|
| `1e6` | `0.019202659` | `0.019202659` | `1.0000` | `0.019202602` | `0.019202602` | `15.93` |
| `100` | `0.019490203` | `0.019485783` | `1.0002` | `0.018923759` | `0.018919479` | `15.97` |
| `21`  | `0.020657739` | `0.020550975` | `1.0052` | `0.017945945` | `0.017854287` | `16.13` |
| `12`  | `0.021910948` | `0.021562233` | `1.0162` | `0.017109682` | `0.016843029` | `16.31` |
| `8`   | `0.023593185` | `0.022742034` | `1.0374` | `0.016230198` | `0.015663227` | `16.55` |
| `5`   | `0.027444227` | `0.024865676` | `1.1037` | `0.014860752` | `0.013539586` | `17.09` |
| `4`   | `0.030887021` | `0.026281437` | `1.1752` | `0.014071978` | `0.012123824` | `17.58` |
| `3`   | `0.039555994` | `0.028641039` | `1.3811` | `0.012931420` | `0.009764222` | `18.82` |
| `2.5` | `0.053001449` | `0.030528721` | `1.7361` | `0.012145990` | `0.007876540` | `20.77` |
| `2`   | `0.106511765` | `0.033360244` | `3.1928` | `0.011133999` | `0.005045018` | `--`    |
| `1.6` | `0.167777143` | `0.036899647` | `4.5468` | `0.010086136` | `0.001505614` | `--`    |

Three consequences, all of which matter for the pipeline:

1. **`eps_c + K/R` is optimistic exactly where it is needed.** It is excellent down to `R ~ 20`
   (0.5% error) but understates by 38% at `R = 3` and by a factor **3.2** at `R = 2`. Do not quote
   the linear form as a bound for tight concave features; use the exact rescaling.
2. **The convex benefit saturates.** The linear form predicts `eps* -> 0` at `R = K/eps_c = 1.47`,
   i.e. that tight convex features certify for free. The exact answer flattens near `0.010`, about
   `0.53 eps_c`, and never approaches zero. Convexity is worth at most a factor of two, ever. This is
   a qualitative error of the first-order picture, in the unsafe direction.
3. **Curvature moves the critical direction**, from `15.93` deg to `16.55` deg at `R = 8` and
   `18.82` deg at `R = 3`. A search that assumes the planar critical angle will miss the worst case
   on curved geometry.

Degradation budget, first order (read as optimistic):

| tolerated excess over `eps_c` | required concave radius `R` |
|---|---|
| `+5%`  | `> 29.5` voxels |
| `+10%` | `> 14.7` voxels |
| `+25%` | `>  5.9` voxels |
| `+50%` | `>  2.9` voxels |

**3D outlook.** The penalty generalises to `(p . H . p)/4` with `H` the second fundamental form. At
a fixed bound `kappa` on the principal curvatures, a concave sphere gives `kappa |p|^2/4` for every
offset direction, a cylinder gives `kappa p1^2/4 <= kappa |p|^2/4`, and a saddle can be negative --
so the sphere penalty dominates pointwise in `p`. Since `eps* = max_n min_w [g + penalty]` is
monotone under a pointwise-larger penalty, **the concave sphere should be the worst case among
surfaces of bounded principal curvature**. (Confidence: the pointwise domination is elementary; the
monotonicity step ignores that a changed depth also changes *admissibility*, so this is flagged for
the 3D effort rather than asserted. It corrects an earlier off-hand guess that a concave edge would
be worse -- an edge only wins by leaving the bounded-curvature class entirely.)

### 4.5b Degenerate pockets have no certified seed at all

For an exterior pocket with in-radius at or below `r_d`, no lattice point inside it can be
non-barrier, so connected components certifies nothing and the closure has nothing to start from.
This is a failure mode distinct from the `eps_c` constant and is not covered by the model here. The
same mechanism explains the exact-rule row `W < 1 + r_d -> no guarantee` in 3.7.

### 4.6 A real implementation hazard: exact ties at lattice-aligned normals

With plain double arithmetic and the literal strict test `UDF(A) + UDF(B) > dist(A,B)`, the
end-to-end simulation produced a genuine **soundness violation** on its first scan:

```
    alpha = 45 deg, offset 0.005, A = (-10,12) with signed distance +1.4192135623730939
                                  B = (-12,10) with signed distance -1.4092135623730968
    B - A = (-2,-2) is parallel to n = (1,1)/sqrt2, so in exact arithmetic
        UDF(A) + UDF(B) = |s_A - s_B| = dist(A,B) = 2 sqrt2   EXACTLY,
    and the strict '>' correctly fails. Double rounding produced a margin of +4.44e-16 and
    certified a point on the WRONG side of the interface.
```

The tie set is nonempty for **every** `n` parallel to an integer vector — i.e. for the most common
axis-aligned and diagonal surface orientations. Any implementation must use a guarded predicate
(`UDF(A) + UDF(B) - dist > tol`); with `tol = 1e-12` the whole scan is sound and the measured
critical depth shifts by exactly `tol/2 = 5e-13`, i.e. conservatively. With `tol = 0` the scan
cannot be completed without violations. This is the single most actionable experimental finding for
the 3D pipeline.

---

## 5. Prior art

A full, source-tagged prior-art review is in **`LatticeCertificationGap_PriorArt.md`**. Summary of
its verdict:

**Partly a rediscovery, mostly new, with entirely classical machinery.** Three things must be kept
apart.

* **The machinery is classical and should be cited, not re-derived.** Decomposing directions into
  angular sectors indexed by the Farey / Stern-Brocot ordering of primitive ("visible") lattice
  vectors, reducing each sector to its two generating vectors, and locating the optimum at the
  equal-cost crossover of two adjacent primitive directions is textbook chamfer-mask and
  digital-medial-axis material: Montanari (1968), Borgefors (1986), Thiel-Montanvert (1992),
  Remy-Thiel (2000), Hulin-Thiel (2009). Our 16 primitive directions are the Farey sequence `F_2`
  under the dihedral group, and the octant-split-by-the-knight-direction picture is their "influence
  cone" decomposition.
* **One of our constants is not new.** The plateau value `0.0449101394377726...` that holds for
  `W` in `[1.42, 2.24)` -- our `(1,0)`/`(1,1)` crossover `eps_1 = (1 - sqrt(2 sqrt2 - 2))/2` -- is
  **exactly** Borgefors 1986, Eq. (18), at her critical angles `24.5 deg` / `65.5 deg`, with her
  optimal `3x3` chamfer weights `a_opt = 0.95509`, `b_opt = 1.36930` equal to our witness depths
  `1 - eps_1` and `sqrt2 - eps_1`. Verified independently to `5.6e-17`. **This must be attributed.**
  The identity is an accident of the `3x3` case: it holds because the trajectory direction `(0,1)`
  coincides with `w2 - w1 = (1,1) - (1,0)`, and it breaks for the `5x5` pair `(1,0)`, `(2,1)` where
  `w2 - w1 = (1,1)`. Borgefors never optimised `5x5` with free `a` ("The local distances have not
  been optimized for `a != 1`"); her `a=1`-constrained `5x5` value `0.019579` is close to but **not**
  our `0.0192026`.
* **The functional and the `W = 3` constant appear nowhere.** Every optimality criterion found in
  this literature is multiplicative or relative -- maximum relative error, percentage error, error on
  a circular trajectory. Ours is a worst-case **additive**, length-weighted deficiency
  `g(w,n) = (|w| - w.n)/2`, minimised subject to a budget on a **projected** quantity
  `sqrt2/2 < (|w| + w.n)/2 <= W` rather than on neighbourhood size `max(|x|,|y|) <= p`. That
  difference is not cosmetic: the `W` budget interleaves plateaus the `M_p` mask family cannot
  produce, e.g. `0.011330789` at `W >= 3.17` (the `(2,1)`/`(1,1)` crossover).

**Two results worth carrying forward from the review.**

1. **A closed form for the whole plateau family.** With `s_p = sqrt(p^2+1) - p` and `q = p-1`, the
   tying witness pair for the `(2p+1)x(2p+1)` mask is always `{(1,0), (p,1)}`, and

   ```
       2 eps*(M_p) = ( -B - sqrt(B^2 - 4AC) ) / (2A),
       A = 1 + q^2,   B = 2(s_p q - 1),   C = s_p^2
   ```

   reproduces all five observed plateau values `0.0449101394, 0.0192026308, 0.0105747110,
   0.0067243168, 0.0046737113` to `3e-13`. This also matches Hajdu-Hajdu-Tijdeman's Lemmas 10-11,
   which pin their extremal path to steps `(p,0)` and `(p,1)` -- the same extremal pair, a different
   functional.
2. **A new chamfer-theoretic reading of our constant.** Via LP duality, the chamfer distance with
   weights `omega_i` is the support function of `Q = {n : w_i . n <= omega_i}`. Taking the *additive*
   family `omega_i = |w_i| - delta` makes `Q_delta` a shrunken circumscribed polygon of the unit
   circle, and `2 eps*` is exactly the threshold `delta` at which the chamfer norm stops
   over-estimating the Euclidean norm in every direction. Equivalently: **`2 eps*(W)` is the smallest
   uniform subtraction from exact Euclidean weights that turns the chamfer norm into a global
   under-estimate.**

**Calibration.** High confidence on the Borgefors identity and on the relative-vs-additive contrast
(both checked against primary full texts and reproduced numerically). **Medium confidence on the
absence claim**: the search was cut short by a tool denial in each of its three areas and never
reached MathSciNet/zbMATH, venue-scoped DGCI/IWCIA indexes, non-English sources, or the 3D case.
Treat "not found" as "not exhaustively searched for".

## 6. What this means for the pipeline

**The practical reading at `W = 3`, in 2D.** Every exterior lattice point deeper than
`0.0192026307637964` voxels from the surface is certified by the connected-components +
pairwise-enrichment scheme; there exists a configuration (a straight interface at `15.9306 deg`,
target at depth exactly `eps_c`) that leaves one uncertified at exactly that depth. Compare with the
previous rigorous figure of `~0.193` voxels.

**These two numbers are not directly comparable, and the write-up should say so.** The old figure is
3D and uses the weaker ball-membership criterion; the new one is 2D and uses pairwise ball overlap.
Two of the three differences can be quantified exactly:

* **Criterion (exact factor of 2).** Ball membership asks `v1 in B(v0, d0)`, i.e.
  `|w| < eps + w.n`, i.e. `eps > |w| - w.n = 2 g(w,n)`. Pairwise overlap asks `eps > g(w,n)`.
  Under the exact rule the usability window `r_d < eps + w.n <= W` does not depend on the
  criterion, so the guaranteed depth under ball membership is *exactly* twice the pairwise one, in
  any dimension and at any `W`. Verified directly in 2D at `W = 3`: re-running the exact-rule
  evaluator with the criterion `eps > 2g` gives `0.0384047` (grid-limited) at the same `alpha*`,
  i.e. exactly `2 eps_c = 0.038405261527593`. (Under the *threshold* rule the correspondence is not
  clean, because the self-consistent witness depth would become `|w|` rather than `(|w|+w.n)/2`.)
* **Truncation (not a factor).** The `0.019202630763796` decimal is a lower approximation to
  `eps_c`; use `0.0192026307637964` or the exact surd if the number is used as a bound.
* **Dimension (not quantified).** The remaining discrepancy is the 2D -> 3D change. If the `~0.193`
  figure uses the same admissibility and band, then the 3D pairwise constant it implies is
  `~0.0965`, i.e. about `5x` the 2D value — but this is arithmetic on a figure supplied to me, not
  a verified claim. **I did not verify `0.193`**; it does not appear in CONTEXT.

**Caveats that must travel with the number:**

1. It is a **planar** bound. A concave exterior of curvature radius `R` gives `eps_c + 0.0283/R`;
   at `R = 8` voxels that is `+23%`, at `R = 3` voxels `+106%`. In a real 3D mesh the worst case is
   the tightest concave feature, not a flat wall. Any safety factor should be sized from the
   smallest concave curvature radius the pipeline is expected to resolve.
2. It is a **2D** bound. Nothing here transfers to 3D automatically (section 7).
3. It requires `W` inside the plateau. Under the rule the pipeline actually implements, the plateau
   is `[2.279483144060, 3.178460085208)`. `W = 3` is inside with ~6% margin on the high side. If
   the band ever narrows below `2.2795` the constant jumps by 2.34x; below `1 + r_d = 1.7071` the
   scheme provides **no guarantee at all**.
4. It requires a **guarded strict predicate**. Section 4.6: with plain doubles the rule is
   unsound at exactly the lattice-aligned normals that dominate real content. Use
   `UDF(A) + UDF(B) - dist > tol` with `tol` a few ulps of the largest term; the cost is a
   conservative `tol/2` shift in the guaranteed depth.
5. Do not hard-code `eps_c` as a float threshold without a guard band: naive double evaluation of
   the pipeline lands `~36 ulps` above the true value.
6. The scheme also fails outright for exterior pockets whose in-radius is at or below `r_d`, where
   there is no non-barrier seed at all. That is a separate failure mode from the depth constant.

---

## 7. Open questions

**The 3D generalisation.** This is the real target and nothing here transfers automatically.

* The affine-in-`n` structure `(I2)` and Lemma A (barrier test inactive) **do** carry over verbatim:
  `g(w,n) = (|w| - w.n)/2` is still affine in `n`, still `|w| = g + d0`, and still `g >= 1 - r_d` is
  needed for the barrier test to bite (with `r_d = sqrt3/2 = 0.8660`, so the threshold becomes
  `g >= 0.1340` — noticeably smaller than the 2D `0.2929`, so the barrier test may *not* be
  automatically inactive in 3D; this must be checked, not assumed).
* The direction space is `S^2` (2-parameter) rather than `S^1`, the symmetry group is the 48-element
  hyperoctahedral group, and the "gap combinatorics" become the Delone/Voronoi structure of the
  lattice directions rather than a Farey sequence. The right tool is **simultaneous Diophantine
  approximation**: the 2D answer is set by the crossover of two Farey neighbours, and the 3D answer
  should be set by the crossover of three witnesses spanning a spherical triangle of the direction
  "Farey" (Stern-Brocot / continued-fraction-of-two-numbers) subdivision. There is no canonical
  2D continued-fraction algorithm, which is exactly why the 3D problem is hard.
* The critical-point certificate approach used here (enumerate all pairwise cost ties, all band
  edges, all barrier edges, evaluate at each plus one-sided offsets) generalises in principle: in
  3D the ties are great circles on `S^2` and the critical points are their triple intersections.
  The enumeration is `O(N^3)` in the number of candidate witnesses; with `|w| <= W + r_d` and
  `W = 3` that is a few thousand witnesses, so a few times `1e10` triples — feasible with pruning.
* The band cutoff will again be what sets the constant (it amputates the good rational directions),
  so the answer should again be a plateau in `W` with algebraic endpoints.

**Left open by the referees, still open here:**

1. Prove that the certified set is upward closed in `eps` at `W = 3` (section 3.8.2). Numerically
   certain on 90 001 directions; not proved. This is what turns "`eps_c` is the min cost" into
   "`eps > eps_c` implies certified".
2. Write the exhaustive critical-direction argument for the exact-rule endpoints `W_lo'`, `W_hi'`
   (section 3.8.1).
3. Add the compactness remark for `sup` vs `max` when `W > W_hi` (section 3.8.3).
4. Prove the two "sliver" continuation formulas for `eps*(W)` just outside the plateau; they are
   currently local derivations verified to `~2.5e-12`.
5. Resolve the `2.8e-7` disagreement between tracks on the top of the ramp above `W_hi`
   (`3.162277382` vs `sqrt10 = 3.162277660`).
6. Multi-hop enrichment is only handled at `n*` (where it provably does not help) and empirically
   elsewhere (0 changes at 451 angles). A general statement is missing.

**New questions raised by the experiments:**

7. **Curvature.** Make the `eps_c + K/R` bound rigorous (the derivation in 4.5 is a clean
   first-order expansion, but the finite-`R` behaviour, including small `R` where the whole
   configuration degenerates, is only sampled). Then decide whether the pipeline's guarantee should
   be stated as a curvature-dependent bound rather than a constant.
8. **Non-planar `Sigma` generally.** Is the worst case over all surfaces attained by a sphere, or
   can a saddle/edge do worse? In 3D there are two principal curvatures and the correction term
   becomes `(p . H . p)/4` for the second fundamental form `H`, so an edge-like feature may be worse
   than any sphere.
9. **Optimal `W`.** Given the plateau structure, `W = 3` buys nothing over `W = 2.28` except margin,
   and `W = 3.18` would buy a 1.7x improvement (`eps_c -> 0.0113`). Is the extra band worth it in
   the 3D pipeline's memory budget? The 3D plateau structure needs to be computed before this can
   be answered.

---

## Appendix A: constants

| symbol | closed form | value (30 digits) |
|---|---|---|
| `eps_c` | `(3 - sqrt5 - beta)/4` | `0.019202630763796343828423710860` |
| `beta` | `sqrt(2 sqrt5 - 4)` | `0.687121499445024928277131487829` |
| `cos alpha*` | `(sqrt5 - 1 + beta)/2` | `0.961594738472407312343152578280` |
| `sin alpha*` | `(sqrt5 - 1 - beta)/2` | `0.274473239027382384066021090451` |
| `alpha*` | `arcsin((sqrt5-1)/sqrt2) - 45 deg` | `15.930625116297946538778691390428 deg` |
| `r_d` | `sqrt2/2` | `0.707106781186547524400844362105` |
| `2 eps_c` (ball-membership constant) | | `0.038405261527592687656847421720` |
| `eps_1` = `(1,0)x(1,1)` crossover | `(1 - sqrt(2 sqrt2 - 2))/2` | `0.044910139437772658695642242178` |
| `eps_2` = `(2,1)x(1,1)` crossover | `(2 sqrt2 - sqrt5 - sqrt(2 sqrt10 - 6))/2` | `0.011330789030059157650556477544` |
| `sqrt2/2 - 3 sqrt5/10` (value just below `sqrt5`) | | `0.036286387936610615478092261485` |
| `W_c1` | `(sqrt5 + 2 sqrt(2 sqrt2 - 2) + sqrt2 - 1)/2` | `2.235320491060897055214146712115` |
| `W_lo` (threshold) | `sqrt5` | `2.236067977499789696409173668731` |
| `W_lo'` (exact rule) | `(3/2) sqrt(2 sqrt2 - 2) + sqrt2 - 1/2` | `2.279483144059777072714761997677` |
| `W_hi` (threshold) | `(sqrt10 + 2 sqrt5 - 2 + beta)/2` | `3.160767557306491826547186184862` |
| `W_hi'` (exact rule) | `(7 sqrt5 - 5 + 3 beta)/4` | `3.178460085208400664923902536152` |
| `1 + r_d` (below this: no guarantee) | | `1.707106781186547524400844362105` |
| `K` (curvature constant) | `(s1 p2^2 - s2 p1^2)/(4(s1-s2))` | `0.028315226153961` |

Minimal polynomials: `eps_c`: `64x^4 - 192x^3 + 208x^2 - 56x + 1` (degree 4, irreducible).
`W_hi`: `x^8 + 8x^7 + 2x^6 - 120x^5 - 209x^4 + 240x^3 + 1188x^2 + 664x + 1` (degree 8).

## Appendix B: verification artifacts

Written for this document (independent re-derivations, no reuse of track code):

| file | what it checks |
|---|---|
| `scripts/distancing_experiments/lattice_gap/syn_v1.py` | five closed forms agree to 70 digits; rounding/truncation facts; quartic; `alpha*` |
| `scripts/distancing_experiments/lattice_gap/syn_v2.py` | exhaustive tightness at `n*` over `\|a\|,\|b\| <= 20`; finiteness bound; three-arc covering |
| `scripts/distancing_experiments/lattice_gap/syn_v3.py` | `eps*(W)` under both rules at 33 values of `W` |
| `scripts/distancing_experiments/lattice_gap/syn_v4.py` | upward-closedness at `W = 3` (90 001 directions); the `W = sqrt5, alpha = 24 deg` counterexample |
| `scripts/distancing_experiments/lattice_gap/syn_v5.py`, `syn_v7.py` | exact-rule plateau endpoints with ultrafine local scans |
| `scripts/distancing_experiments/lattice_gap/syn_v6.py` | the `(3,1)` coverage hole at `alpha*` as a function of `W` |

From the three tracks (proofs, experiments, prior art):

`ub_certificate.py`, `ub_sweep.py` (upper bound; exact `Fraction` interval certificate);
`t_exact.py`, `t_tightness.py`, `t_tightness2.py`, `t_extra.py`, `t_e2e.py`, `t_selftest.py`
(tightness; `Q(beta)` tower and certified intervals);
`lib_dist.py`, `part_a..part_d3`, `minpoly.py` (closed forms, plateau, minimal polynomials);
`ref_*.py`, `ref2/*` (adversarial referee re-implementations);
`e2e_sim.py`, `e2e_scan*.py`, `e2e_refine.py`, `e2e_hp*.py`, `e2e_mono.py` (end-to-end pipeline
simulation, no use of `g`);
`core.py`, `s1..s9*.py` (adversarial search: critical-angle certificates, hill climbing, annealing,
rational probes, radius stress, curved `Sigma`, brute force);
`anatomy.py`, `extras.py`, `finish.py`, `surds.py`, `anatomy.json` (critical-configuration anatomy);
`priorart/borg3x3.py`, `priorart/verify1..3.py`, `priorart/hulin.txt`, `sm4n2x.txt`, `5dqxoe.txt`
(prior art).
