# Prior art: the lattice certification gap

Companion to `LatticeCertificationGap.md`. Citations are tagged `[PRIMARY]` (read
from a full text), `[SEARCH-SUMMARY]` (search-engine prose only -- bibliographic
details may be wrong), or `[OURS]` (our own derivation, not a literature claim).

## 0. Verdict

Our question is **partly a rediscovery and mostly new, with entirely classical machinery**. Three
things must be separated. (i) The *machinery* — decomposing directions into angular sectors indexed
by the Farey / Stern–Brocot ordering of primitive ("visible") lattice vectors, reducing each sector
to its two generating vectors, and locating an optimum at the equal-cost crossover between two
adjacent primitive directions — is textbook material in the chamfer-mask and digital-medial-axis
literature, going back to Montanari (1968), Borgefors (1986) and Thiel–Montanvert (1992). We should
cite it, not re-derive it. (ii) One of our numbers is *not new*: the plateau value
`eps* = 0.0449101394377726…` that holds for `W ∈ [1.42, 2.24)` is exactly Borgefors' 1986 Eq. (18),
`(1 − sqrt(2 sqrt2 − 2))/2`, attained at her critical angles 24.5°/65.5°, with her optimal weights
`a_opt = 0.95509`, `b_opt = 1.36930` equal to `1 − eps*` and `sqrt2 − eps*`. This is an exact
identity, verified independently to 5.6e-17, and it must be attributed. (iii) The *functional* and
the constant `eps*(W = 3) = 0.0192026307637963438…` appear nowhere. Every optimality criterion we
found in this area is multiplicative/relative (maximum relative error, percentage error, error on a
circular trajectory, error normalised by local feature size); ours is a **worst-case additive,
length-weighted misalignment deficiency** `g(w, n) = (|w| − w·n)/2`, minimised over primitive
lattice vectors subject to a **budget on a projected quantity** `sqrt2/2 < (|w| + w·n)/2 ≤ W`
rather than on the neighbourhood size `max(|x|,|y|) ≤ p`. That combination — additive cost,
projection budget, worst case over directions, exact algebraic constant, interpreted as a
sign-certification depth for an unsigned distance field on a lattice — was not found in any of the
95 web calls in the salvage or in three supplementary calls. Calibration: **high confidence** on
the Borgefors identity and on the relative-vs-additive contrast (both checked against primary full
texts and reproduced numerically); **medium confidence** on the absence claim, because the search
was cut short by a tool denial in each of its three areas and never touched MathSciNet/zbMATH,
venue-scoped DGCI/IWCIA indexes, non-English sources, or the 3D case.

Throughout, citations are tagged `[PRIMARY]` (read from a full text recovered in the salvage),
`[SEARCH-SUMMARY]` (known only from search-engine prose — bibliographic details may be wrong), or
`[OURS]` (our own derivation or numerical check, not a literature claim). Bibliographic details not
literally present in the salvage are omitted rather than reconstructed.

---

## 1. Closest prior work

### 1.1 Same constant, different problem: Borgefors 1986 (the one genuine collision)

> G. Borgefors, "Distance transformations in digital images", *Computer Vision, Graphics, and Image
> Processing* **34** (1986) 344–371. `[PRIMARY]` — full text at `scripts/distancing_experiments/lattice_gap/sm4n2x.txt`; the
> ACM record URL in the salvage encodes DOI `10.1016/S0734-189X(86)80047-0`.

**What it proves.** Section 3.1 optimises the real-valued 3×3 chamfer weights `(a, b)` by minimising
the maximum absolute difference `|DT − EDT|` along the rectilinear trajectory `x = M`. Her Eq. (17)
gives `a_opt = (sqrt(2 sqrt2 − 2) + 1)/2 ≈ 0.95509`, `b_opt = sqrt2 + (sqrt(2 sqrt2 − 2) − 1)/2 ≈
1.36930`; Eq. (18) gives `maxdiff = (sqrt(2 sqrt2 − 2) − 1)·M/2 ≈ 0.04491·M`. She states that the
maximum "occurs along the lines angled 0° + n·45°, 24.5° + n·90° and 65.5° + n·90° from the
horizontal", and that the optimum is characterised by the equal-ripple condition
`−Diff1 = Diff2 = −Diff3`.

**How it matches us.** `(1 − sqrt(2 sqrt2 − 2))/2 = 0.0449101394377726`; our
`eps*(W ∈ [1.42, 2.24)) = 0.0449101394377727` at `alpha = 65.5302°` (CONTEXT.md line 98). Same
constant to 5.6e-17, same critical angles (24.47°/65.53° are exactly our `alpha*` and its 45° mirror),
and her optimal weights are exactly `|w| − eps*` for `w ∈ {(1,0), (1,1)}` — i.e. the "depth of each
witness at its own certification threshold" that CONTEXT.md lines 90–91 tabulate. `[OURS: verified
by scripts/distancing_experiments/lattice_gap/priorart/chk_borg.py]`

**Why the identity is an accident that does not generalise.** `[OURS]` Borgefors' optimum sits at
`omega_i = |w_i| − eps`; her per-cone error is `A(t) − eps` where `A = 2[lambda·g(w1,n) + mu·g(w2,n)]`
is the geometric path deficiency with exact Euclidean weights. In the 3×3 octant the trajectory
direction `(0,1)` coincides with `w2 − w1 = (1,1) − (1,0)`, so the stationarity condition of the
path deficiency and our two-witness crossover become the *same* equation, `sin phi = sqrt2 − 1`. For
the 5×5 pair `(1,0), (2,1)` we have `w2 − w1 = (1,1) ≠ (0,1)` and the two problems separate. Checks:
Borgefors never optimised 5×5 with free `a` — she writes "The local distances have not been optimized
for a ≠ 1" `[PRIMARY, sm4n2x.txt line 899]` — and her `a = 1`-constrained 5×5 optimum
`c_opt = 2.19691` gives `(sqrt5 − c_opt)/2 = 0.019579`, close to but **not** our `0.0192026`. Freely
optimising `(a,b,c)` under her rectilinear criterion gives `0.0142190` with *non-uniform* deficits
`(0.01388, 0.00000, 0.02844)`, i.e. not of the form `|w| − eps`; under a circular/relative criterion
it gives `0.0136125`, which reproduces Hajdu–Hajdu–Tijdeman's `E_2^C ≈ 0.0136` and thereby
cross-validates the code. Neither equals `0.0192026`. `[OURS:
scripts/distancing_experiments/lattice_gap/priorart/chk_borg5.py]`

**Action:** any writeup must attribute the `W ∈ [1.42, 2.24)` plateau value to Borgefors 1986
Eq. (18), and must state explicitly that the 5×5-regime constant is *not* hers.

### 1.2 Same mask family and same extremal pair, different functional: Hajdu–Hajdu–Tijdeman 2012

> A. Hajdu, L. Hajdu, R. Tijdeman, "Approximation of the Euclidean distance by chamfer distances",
> *Acta Cybernetica* **20**(3) (2012) 399–417; DOI `10.14232/actacyb.20.3.2012.3`; preprint
> arXiv:1201.0876. `[PRIMARY]` — full text at `scripts/distancing_experiments/lattice_gap/5dqxoe.txt`.

**What it proves.** They work on exactly our mask family `M_p = {(x,y) ∈ Z^2 : max(|x|,|y|) ≤ p}`
(so `M_2` is the 5×5 mask, our 16 primitive directions), and determine sharp constants for the
maximum relative error `E := limsup_{|v|→∞} ( W(v)/|v| − 1 )` in three regimes: (B) with `W(x,0) =
|x|` forced, `E_1^B ≈ 0.0551`, `E_2^B ≈ 0.0187`, `E_3^B ≈ 0.0089`; (D) the conservative regime,
`E_p^D` in closed form, `≈ 0.0824 / 0.0275 / 0.0131`; (C) unconstrained, `E_p^C = E_p^D/(2 + E_p^D)`,
`≈ 0.0396 / 0.0136 / 0.0065`. Their weight family is an explicit **multiplicative** shrink: beside
`N_c(p,0) = p`, the other admissible neighbourhood vectors get "their Euclidean lengths, multiplied
by a factor `c ≤ 1`".

**How it overlaps.** Their Lemmas 10 and 11 pin the extremal path to steps of the form `(p,0)` and
`(p,1)`. That is exactly our binding tie: computing `eps*(M_p)` by brute force gives tying witness
pairs `{(1,0),(1,1)}, {(1,0),(2,1)}, {(1,0),(3,1)}, {(1,0),(4,1)}, {(1,0),(5,1)}, {(1,0),(6,1)}` for
`p = 1..6`. `[OURS: scripts/distancing_experiments/lattice_gap/priorart/chk_hht.py]` Both constants live in the algebraic family
generated by `s_p = sqrt(p^2+1) − p`. Theirs is `E_p^D = sqrt(s_p^2 + 1) − 1`; ours is, with
`q = p−1`, `A = 1 + q^2`, `B = 2(s_p q − 1)`, `C = s_p^2`,

```
2 eps*(M_p) = ( −B − sqrt(B^2 − 4AC) ) / (2A)
```

which reproduces CONTEXT.md's plateau values `0.0449101394, 0.0192026308, 0.0105747110,
0.0067243168, 0.0046737113` to 3e-13. `[OURS: scripts/distancing_experiments/lattice_gap/priorart/final.py]`

**How it differs.** Three ways. (a) Their functional is a maximum **relative** error; ours is a
worst-case **additive** deficiency. Their own conclusion says they "determined the smallest possible
maximum relative error". (b) Their budget is the neighbourhood size `max(|x|,|y|) ≤ p`; ours is a
budget on the projected depth `(|w| + w·n)/2 ≤ W`, plus a *lower* cutoff at the lattice covering
radius `sqrt2/2`. This is not cosmetic: the `W`-budget step function interleaves extra plateaus that
the `M_p` family does not produce — e.g. `eps* = 0.011330789` at `W ≥ 3.17` (crossover of `(2,1)` and
`(1,1)`) is absent from `{eps*(M_p)}`. `[OURS]` (c) `E_2^B ≈ 0.0187` is numerically close to our
`0.0192026` but is a **genuinely different constant**; do not let a reader conflate them.

### 1.3 Closest machinery for a lattice ball predicate: Hulin–Thiel 2009

> J. Hulin and É. Thiel, "Farey Sequences and the Planar Euclidean Medial Axis Test Mask", 13th
> IWCIA, Cancún, Mexico, Nov 2009. `[PRIMARY]` — full text at `scripts/distancing_experiments/lattice_gap/priorart/hulin.txt`;
> venue from Thiel's publication list as fetched into `scripts/distancing_experiments/lattice_gap/lit/chamfer.txt` line 167.

**What it proves.** The Euclidean test mask `T(r)` is "the minimum neighbourhood sufficient to
detect the Euclidean Medial Axis of any discrete shape whose inner radius does not exceed `r`". Their
Theorem 1: for any visible (primitive) vector `v` other than `(1,0)` and `(1,1)`, the set
`{pred(v), succ(v)}` of its two **Farey neighbours** is a *set of keys* — it dominates every other
candidate in the parallelogram of keys (Lemmas 1–2, by a perpendicular-bisector argument). They also
define the **appearance radius** `r_app(v)`, the smallest `r` at which `v` must enter `T(r)`, give an
`O(r^3)` algorithm for it, and cite their own earlier proof that `T(r)` is unique in any dimension
and tends to the set of visible vectors as `r → ∞`.

**Why it is the best structural precedent we found.** It is a published, proved statement, in `Z^2`,
of exactly the principle our argument needs: *for a ball-based local predicate, only the two
Farey/Stern–Brocot neighbours of a direction can bind.* Their `r_app` plays the role of our band
half-width `W` deciding which `(p,1)` is admissible — and they observe empirically (Fig. 11, Sect. 6)
that vectors `(x,1)` have the smallest appearance radii of all visible vectors, conjecturing
`r_app(v) > v_1^2` with the bound tight on that family. That is the same "(p,1) dominates"
phenomenon we see.

**How it differs.** Their predicate is ball **inclusion** (`d(p,q) ≤ R_q − R_p`, a difference of
radii); ours is ball **overlap** (`d1 + d2 > |w|`, a sum). For a straight interface the inclusion
predicate degenerates, so none of their quantitative results transfers. They never extremise an
additive per-direction cost, and there is no analogue of `eps*`.

Companion references, verified from Hulin–Thiel's own reference list and Thiel's publication page
`[SEARCH-SUMMARY for venues]`: J. Hulin and É. Thiel, "Visible vectors and discrete Euclidean medial
axis", *Discrete and Computational Geometry* **42**(4) (2009) 759–773 (source of "T(r) tends to the
visible vectors"); É. Remy and É. Thiel, "Medial axis for chamfer distances: computing look-up tables
and neighbourhoods in 2D or 3D", *Pattern Recognition Letters* **23**(6) (2002) 649–661; É. Remy and
É. Thiel, "Exact Medial Axis with Euclidean Distance", *Image and Vision Computing* **23**(2) (2005)
167–175; N. Normand and P. Évenou, "Medial axis LUT computation for chamfer norms using
H-polytopes", 14th DGCI, Lyon, LNCS **4992** (2008) 189–200.

### 1.4 The same union of balls, but with the sign given as input

> M. Kohlbrenner and M. Alexa, "Isosurface Extraction for Signed Distance Functions using Power
> Diagrams", *Computer Graphics Forum* **44**(2), article e70037, 2025. `[SEARCH-SUMMARY]` — the
> Wiley fetch was **denied** and the TU-Berlin PDF exceeded the fetch size limit; the paper text was
> never read.

Defines the **feasible region** as the region between the union of positive spheres and the union of
negative spheres (radii `|SDF|` at grid sites); every surface consistent with the samples lies there,
and its boundary consists of spherical caps ("feasible arcs"). Our exterior set `O` is exactly their
positive-sphere union. **Decisive difference:** the sign at each sample is *input*. They ask which
surfaces are consistent with given signed samples; we ask which signs are *forced* by unsigned ones.
Nobody found runs the implication in our direction, so there is no analogue of a certification depth.

> S. Sellán, Y. Ren, C. Batty and O. Stein, "Reach For the Arcs: Reconstructing Surfaces from SDFs
> via Tangent Points", *SIGGRAPH 2024 Conference Papers*, article no. 25, 2024,
> DOI `10.1145/3641519.3657419`; and S. Sellán, C. Batty and O. Stein, "Reach For the Spheres:
> Tangency-Aware Surface Reconstruction of SDFs", arXiv:2308.09813, 2023.
> `[SEARCH-SUMMARY]` — both PDFs exceeded the fetch size limit.

"Reach For the Spheres" states our per-sample constraint one step further: each SDF sample
corresponds to a spherical region that must lie fully inside or outside the surface *depending on its
sign*, and must be tangent to the surface somewhere. Constraint (1) is precisely "`B(V_i, d_i)` is
surface-free". But the side is read off the given sign; there is no argument that overlapping
spheres force a common side, and no bound on how deep a sample can be before its side is
undecidable.

### 1.5 The nearest named relative of our pairwise rule

> J. Bialkowski, M. Otte, S. Karaman and E. Frazzoli, "Efficient collision checking in
> sampling-based motion planning via safety certificates", *International Journal of Robotics
> Research*, 2016, DOI `10.1177/0278364915625345`. `[SEARCH-SUMMARY]`

A **safety certificate** is the stored lower bound on a checked point's distance to the nearest
obstacle, defining a ball guaranteed collision-free; a new point inside an old point's certificate
ball skips its collision check. This is the **one-ball containment** test `|p − a| < d(a)`, strictly
weaker than our two-ball overlap test `d(a) + d(b) > |a − b|`. It is used to skip work, never to
propagate an inside/outside label, and there is no worst-case analysis of which points fail to be
certified.

> J. C. Hart, "Sphere tracing: a geometric method for the antialiased ray tracing of implicit
> surfaces". `[SEARCH-SUMMARY]` — venue and year not recorded in the salvage, deliberately omitted.

Contains the overlap argument in 1D: each point carries an "unbounding sphere" of radius equal to the
distance underestimate, guaranteed disjoint from the surface, and consecutive unbounding spheres
must intersect so that no intersection is skipped. That is our pairwise rule restricted to a ray,
stated as a step-size/completeness condition rather than a same-side lemma, and never on a lattice.

**Assessment of novelty for the rule itself.** Three separate targeted searches for the exact form
`d(a) + d(b) > dist(a,b)` returned nothing. Given how elementary the 1-Lipschitz proof is, the honest
framing in a writeup is **folklore-strength** — "the two-ball form of the standard clearance
certificate" — claiming novelty only for (i) using it as a *sign* certificate on a lattice and
(ii) the extremal analysis `eps*(W)`. That framing is robust against a reviewer producing an obscure
precedent.

### 1.6 The classical visibility analogue: Pólya's orchard

> T. T. Allen, "Pólya's orchard problem", *American Mathematical Monthly* **93** (1986) 98–104;
> problem posed by G. Pólya, 1918. `[SEARCH-SUMMARY]`
>
> "The orchard visibility problem and some variants", *Journal of Computer and System Sciences*
> **74** (2008) 587–597. `[SEARCH-SUMMARY]` — **authorship unconfirmed**: the salvaged summary
> attributes it to Kruskal, but that name was supplied by the query and never corroborated by the
> results; the paper itself was never retrieved (ScienceDirect 403).

Allen: trees of radius `r` at every nonzero lattice point of a disc of radius `R`; for integer `R`
you can see out iff `r < 1/sqrt(R^2 + 1)`. The JCSS paper reproves this via a "Stern–Brocot wreath",
recasts Stern–Brocot/Farey in terms of primitive lattice points, generalises to parallelogram
lattices, and studies the radius needed to block the view *between* trees (the ratio of block-all to
block-some radii tends to 2).

**How it differs, precisely.** The quantifier structure is the same — for every direction `n`, is
there a lattice point `w` within budget whose obstacle captures the ray? — but the blocking law is
fundamentally different. Ours, `eps > (|w| − w·n)/2 = |w| sin^2(theta/2)`, gives angular half-width
`≈ 2 sqrt(eps/|w|)` and perpendicular reach `≈ 2 sqrt(eps·|w|)`: the effective tree radius **grows
like sqrt(|w|)**, whereas Pólya's is constant. That is exactly why our extremum is a length-weighted
tie between a short and a long primitive vector, while Pólya's is settled by the single
worst-aligned short vector. `[OURS]` No orchard variant with a distance-dependent obstacle radius was
found, in the salvage or in supplementary searches.

### 1.7 The remaining chamfer optimality literature — all relative criteria

- É. Thiel and A. Montanvert, "Chamfer masks: discrete distance functions, geometrical properties
  and optimization", 11th ICPR, The Hague, Sept 1992; and É. Thiel and A. Montanvert, "Étude et
  amélioration des distances du chanfrein pour l'analyse d'images", *Technique et Science
  Informatiques*, 1992. `[SEARCH-SUMMARY]` — venues from Thiel's own publication page; the saved
  ICPR PDF is a CCITT-fax scan with **no text layer** (`pdftotext` yields 4 bytes), so the primary
  definitions of "influence cone" and "rational ball" are **unverified**. This is the canonical
  source for the influence-cone / visible-point-triangulation / Farey-set decomposition, which is
  the same sector decomposition our per-direction witness selection uses. Their optimisation
  criterion could not be checked.
- É. Remy and É. Thiel, "Optimizing 3D chamfer masks with norm constraints", IWCIA, Caen, France,
  July 2000. `[SEARCH-SUMMARY]` Confirms the construction: "the splitting into influence cones
  corresponds by construction to the Farey triangulation, and every triangle of the triangulation is
  examined". **Terminological trap:** their "norm constraints" means the weights must actually
  induce a norm (convexity / triangle inequality), *not* a budget on `|w|` or `w·n`. Do not cite it
  as prior art for our band constraint.
- B. J. H. Verwer, "Local distances for distance transformations in two and three dimensions",
  *Pattern Recognition Letters* **12**(11) (1991) 671–682. `[SEARCH-SUMMARY]`; the page range is
  also confirmed inside HHT's reference list `[PRIMARY]`. He minimises maximum error and unbiased
  mean-square error, and minimises "over circles and spheres to preserve the symmetries of the
  neighborhoods" — i.e. the scale-invariant/relative criterion. HHT note that Verwer's and
  Butt–Maragos' arguments "contain some hidden assumptions"; for a rigorous citation, cite HHT.
- M. A. Butt and P. Maragos, "Optimum design of chamfer distance transforms", *IEEE Transactions on
  Image Processing* **7**(10) (1998) 1477–1484. `[SEARCH-SUMMARY]` — **no full text obtained**;
  every fetch of the CVSP PDF returned the publications index page. (The salvage contains an IEEE
  Xplore file path `iel4/83/15529/00718487.pdf`, from which the DOI `10.1109/83.718487` can be read
  off; flagged as inferred.) Two things are worth chasing: their "new geometric approach ... which
  is easier to visualize than previous approaches", and their concept of **critical local
  distances**, which "reduces the computational complexity of the chamfer distance transform without
  increasing the maximum approximation error" — the closest published notion to "which mask vectors
  actually bind". Their error function is, per HHT `[PRIMARY]`, the reciprocal-relative
  `limsup(|v|/W(v) − 1)`, plus a disk-area measure.
- G. Malandain and C. Fouard, "On optimal chamfer masks and coefficients", INRIA Research Report
  RR-5566. `[SEARCH-SUMMARY]` — **the full-document fetch is the exact call that was denied and cut
  off the previous agent**; a fresh attempt in this session was blocked by Anubis. This was flagged
  as the highest-risk unread source, because its advertised content ("closed forms for local errors
  related to the chamfer mask geometry", keywords "chamfer distance, anisotropic lattice, Farey
  triangulation") is structurally the nearest published object to our per-direction `g(w,n)`. A
  supplementary search in this session partially closes that gap: the returned summary states the
  work allows "deriving analytically the **relative error** with respect to the Euclidean distance in
  any 3-D anisotropic lattice", with "decomposition into regular cones allow[ing] analytical
  expressions of the error extrema to be derived". So their per-cone closed forms are *relative*
  error extrema, consistent with the rest of the area. This is second-hand and does not fully
  discharge the caveat.
- B. J. Maiseli, "Optimization of chamfer masks using Farey sequences and kernel dimensionality",
  *Scientific Reports* **12**, article 7639 (2022), DOI `10.1038/s41598-022-11807-3`.
  `[SEARCH-SUMMARY, PMC fetch]` The most explicit statement in the salvage of the fact our premise
  rests on: an `Ω × Ω` chamfer kernel corresponds to a Farey sequence, with 5×5 → `F_2` and
  7×7 → `F_3`. Its criterion is an "RLog" relative-accuracy cost, minimising maximum *relative*
  error and achieving **equioscillation** by setting `Z(0°) = max|Z(theta)|`. **Two warnings.** The
  recorded fetch states the rule as `n = floor(Ω^2/2)` while simultaneously computing
  `floor(5^2/2) = 2`, which is arithmetically wrong (the intended rule is plainly
  `n = floor(Ω/2)`), so the paper's formulas as recorded should not be quoted. And the fetch records
  that the paper "cites no previous work connecting Farey sequences to chamfer masks" with no
  citation to Thiel, Normand, Nagy or Hajdu — given Thiel–Montanvert 1992 and Rémy–Thiel 2000, it
  should not be treated as the authority on the Farey connection.
- B. J. Maiseli, "Optimum design of chamfer masks using symmetric mean absolute percentage error",
  *EURASIP Journal on Image and Video Processing* (2019), DOI `10.1186/s13640-019-0475-y`.
  `[SEARCH-SUMMARY]` Surveys the cost functions actually used ("for years, mean absolute error and
  mean squared error have been used"), criticises them, and proposes symmetric mean absolute
  *percentage* error. Corroborating evidence that every functional in this lineage is relative,
  mean-relative or area-based.

### 1.8 Grid-resolution correctness guarantees — the right question, the wrong hypothesis

> Latecki, Conrad and Gross (topology-preserving digitization of 2D r-regular objects yields
> well-composed images); P. Stelldinger and U. Köthe, "Towards a general sampling theory for shape
> preservation" and "Connectivity preserving digitization of blurred binary images in 2D and 3D".
> `[SEARCH-SUMMARY]` — venues not recorded; these were found by a supplementary search, not in the
> salvaged file.

These are the only results found that answer "what grid resolution makes the discrete
inside/outside classification provably correct". Digitizing an `r`-regular set preserves the number
and inclusion structure of connected components of the set and its complement, under conditions
relating `r` to the grid size. But (i) the classification is by **occupancy** (Gauss digitization),
not from distance samples, and (ii) the guarantee is bought with **r-regularity**, a reach/smoothness
hypothesis — precisely the assumption our problem refuses to make.

The feature-size-conditioned reconstruction literature has the same shape. `[SEARCH-SUMMARY for all
of the following]` N. Amenta, S. Choi and R. K. Kolluri, "The power crust, unions of balls, and the
medial axis transform", *Computational Geometry: Theory and Applications*, 2001 — approximates both
the object and its **complement** by unions of polar balls, with bounded geometric error and proved
topological correctness, all conditioned on an `epsilon`-sampling hypothesis; N. Amenta and R. K.
Kolluri, "Accurate and Efficient Unions of Balls" (venue/year not in the salvage) — the union
guarantee is quantified in units of **local feature size**; F. Chazal, D. Cohen-Steiner and A.
Lieutier, "A Sampling Theory for Compact Sets in Euclidean Space", *Discrete & Computational
Geometry*, 2009 — the state of the art *without* smoothness, but still with a positive
(parameterised) feature size, and the certified object is an offset, i.e. a union of **equal-radius**
balls; N. Amenta, M. Bern and M. Kamvysselis, "A new Voronoi-based surface reconstruction algorithm"
and N. Amenta and M. Bern, "The crust algorithm for 3D surface reconstruction", SoCG 1999,
DOI `10.1145/304893.305002` — the origin of `epsilon`-sampling, the archetype of the guarantee we
cannot use, since a lattice is a fixed uniform sample and `Sigma` is arbitrary.

The competing ways to *derive* the sign are all unbounded in our sense: J. A. Bærentzen and H.
Aanæs, "Signed distance computation using the angle weighted pseudonormal", *IEEE TVCG* **11**(3)
(2005) 243–253 `[SEARCH-SUMMARY]` — proven-correct sign rule, but it needs the mesh normal at the
closest feature, i.e. oriented geometry, and fails on non-manifold/inconsistently-oriented input;
A. Jacobson, L. Kavan and O. Sorkine-Hornung, "Robust inside-outside segmentation using generalized
winding numbers", *ACM TOG* **32**(4) (2013) 33:1–33:12 `[SEARCH-SUMMARY]` — consumes the oriented
geometry, guarantee is exactness for watertight input, no lattice-resolution statement; P. Mullen,
F. de Goes, M. Desbrun, D. Cohen-Steiner and P. Alliez, "Signing the Unsigned: Robust Surface
Reconstruction from Raw Pointsets", *Computer Graphics Forum* **29**(5) (2010) 1733–1741
`[SEARCH-SUMMARY]` — literally the same *task*, but the signing is a global stochastic procedure plus
a sparse solve, with no per-sample certificate and no worst-case correctness statement.

### 1.9 Adjacent lattice-visibility and Diophantine results, all with the wrong scaling

`[SEARCH-SUMMARY for all of this subsection]`

- **Dense forests / Danzer.** "Dense forests and Danzer sets"; "Around the Danzer Problem and the
  Construction of Dense Forests", arXiv:2010.06756; "Danzer's Problem, Effective Constructions of
  Dense Forests and Digital Sequences", arXiv:2111.02773, *Mathematika* 2022; "Uniformly Discrete
  Forests with Poor Visibility", *Combinatorics, Probability and Computing*. The visibility function
  `V(eps)` — the length a ray must travel before coming `eps`-close to a point of the set — is the
  asymptotic, uniform-over-basepoints version of our condition. Mismatch: constant `eps` along an
  *unbounded* ray, versus our bounded budget and `O(1)` range.
- **View obstruction / covering radius.** "On the covering radius of lattice zonotopes and its
  relation to view-obstructions and the lonely runner conjecture", arXiv:1609.01939, *Aequationes
  Mathematicae*, DOI `10.1007/s00010-016-0458-3`. Literally our quantifier structure — every ray
  must hit an obstacle centred at (half-)integer points — reduced to a covering-radius computation
  for lattice zonotopes. A possible template for reformulating `eps*(W)`, but the obstacles are
  fixed-side cubes and the extremal quantity `1/(m+1)` is unrelated.
- **Dirichlet spectrum.** H. Davenport and W. M. Schmidt, "Dirichlet's theorem on diophantine
  approximation", *Symposia Mathematica*, Vol. IV, 1970; descendants "Exact uniform approximation
  and Dirichlet spectrum in dimension at least two", *Selecta Mathematica*,
  DOI `10.1007/s00029-023-00889-0`; "Zero-One Law for Uniform Diophantine Approximation in Euclidean
  Norm", *IMRN*, DOI `10.1093/imrn/rnaa256`. This is the branch that studies our exact quantifier
  order — sup over the target of the best achievable approximation with a *bounded* budget — and the
  Euclidean-norm-sensitive variants explicitly replace the sup norm by an arbitrary norm. If
  `eps*(W)` has a classical home, it is as a "spectrum" of this family, for the weight
  `(|w| − w·n)/2` rather than `q‖q·alpha‖`. Nothing in the salvage treats that weight, and none of
  these give finite-budget exact constants.
- **Lattice-free convex sets / flatness.** "Inequalities for the lattice width of lattice-free
  convex sets in the plane", arXiv:1003.4365; "Lattice-free simplices with lattice width
  2d − o(d)", arXiv:2111.08483; "The Exact Lattice Width of Planar Sets and Minimal Arithmetical
  Thickness", DOI `10.1007/11774938_3`. Correct *framing* — our capture region
  `{x : |x| − x·n < 2 eps}` is convex, so `eps*(W)` is the largest lattice-point-free member of a
  one-parameter family — but the literature bounds widths of bounded bodies with asymptotic
  constants in the dimension and has no mechanism for the admissibility budget.
- **Lattice points near parabolas.** "On two lattice points problems about the parabola",
  *International Journal of Number Theory*, DOI `10.1142/S1793042120500360` (arXiv:1902.06047);
  "Diophantine approximation on the parabola with non-monotonic approximation functions",
  arXiv:1802.00525; "Lattice points in thickened parabolas and rational points near hypersurfaces",
  arXiv:2512.00202. **Negative result, checked deliberately** because our capture region is a
  parabola: all of this is lattice-point *counting* under or near a dilated parabola. None is an
  extremal "largest empty parabola" problem. Our parabola formulation does not connect here.
- **Explicit negative.** B. Zadeh and G. A. Constantinides, "Direction-Preserving Number
  Representations", arXiv:2605.07662, 2026. The most promising-looking title for "worst-case angular
  error of a bounded set of integer direction vectors"; the fetched summary states plainly that it
  does *not* address that — it optimises scalar alphabets for low-precision ML formats, and even its
  worst-case angular error is estimated by sampling one million random unit vectors. Do not cite it.

---

## 2. Shared machinery — what we are re-using rather than inventing

**(a) Equioscillation / crossover optimality.** The characterisation "the optimum is where two
adjacent primitive directions have equal cost" is the standard optimality condition in this area,
under several names. Borgefors 1986 solves `−Diff1 = Diff2 = −Diff3` and reports the maximum
attained simultaneously at both endpoints of a sector and at an interior point `[PRIMARY]`. Maiseli
2022 states it as equioscillation, `Z(0°) = max|Z(theta)|` `[SEARCH-SUMMARY]`. HHT's Lemmas 10/11
pin the extremum to the adjacent pair `(p,0), (p,1)` `[PRIMARY]`. Our crossover between `(1,0)` and
`(2,1)` is the same mechanism on a different cost, and should be presented as such.

**(b) Farey / Stern–Brocot structure of primitive directions.** Standard and published. The
correspondence "`(2p+1)×(2p+1)` mask ↔ Farey sequence `F_p`" is stated explicitly in Maiseli 2022
`[SEARCH-SUMMARY]`, so our 16 candidate directions being `F_2` under the 8-fold dihedral group is
not a new observation. The angular-sector ("influence cone") decomposition indexed by the Farey /
visible-point triangulation is Thiel–Montanvert 1992, reused by Rémy–Thiel 2000
`[both SEARCH-SUMMARY]`. Hulin–Thiel's Theorem 1 gives a *proved* two-Farey-neighbour domination
result for a lattice ball predicate `[PRIMARY]`. The Klein polyhedron / sail picture supplies the
convergent-as-vertex scaffold (Wikipedia "Klein polyhedron"; O. N. German, "Klein polyhedra and
lattices with positive norm minima", *Journal de théorie des nombres de Bordeaux*,
DOI `10.5802/jtnb.580`) `[SEARCH-SUMMARY]`. Note our own structural reading `[OURS]`: the tied pair
`(1,0), (2,1)` has determinant 1, i.e. they are Farey neighbours and consecutive sail vertices; the
witness that would beat them, `(3,1)`, is exactly their **mediant** `(0+1)/(1+2) = 1/3`, excluded
only by the band. The whole `W = 3` answer is "the mediant is out of band, so the two Farey parents
tie", which is also the cleanest way to predict the breakpoints of `eps*(W)`.

**(c) Reduction to two vectors per sector.** U. Montanari, "A method for obtaining skeletons using a
quasi-Euclidean distance", *J. Assoc. Comput. Mach.* **15** (1968) 600–624 — cited as ref. [5] in
Borgefors' salvaged full text `[PRIMARY]`, where "Montanari's theorem, [5, Theorem 1]" is invoked
repeatedly to guarantee that "the minimal path consists of two straight lines". This is the classical
antecedent for any proof that only adjacent Farey pairs can tie.

**(d) Restriction to primitive / visible vectors.** CONTEXT.md line 111 justifies dropping
non-primitive witnesses ad hoc via `g(kw, n) = k·g(w, n)`. The corresponding standard fact — the
minimal test mask tends to exactly the set of visible vectors as the budget grows — is proved in
Hulin–Thiel, *DCG* **42**(4) (2009) 759–773, and quoted in the salvaged IWCIA text `[PRIMARY for the
quotation]`. Cite it.

**(e) Angular-gap structure.** The three-distance (three-gap) theorem is the standard tool for "the
gaps between the admissible primitive directions take few values". P. Alessandri and V. Berthé,
"Three distance theorems and combinatorics on words", *Enseignement Mathématique* **44**:1–2 (1998)
103–132; V. Berthé and C. Reutenauer, "On the Three Distance Theorem", *Mathematical Intelligencer*,
DOI `10.1007/s00283-023-10316-z`; original proofs by Sós (1957), Surányi, Świerczkowski (1956), on a
conjecture of Steinhaus. `[SEARCH-SUMMARY]` Our `W = 3` instance has gaps alternating
`arctan(1/2) = 26.565°` and `18.435°` — a two-value structure of exactly this type. **But** the
three-gap theorem bounds gaps only; a pure nearest-angle argument gives `0.029908`, not `0.019203`,
so it certifies the loose bound and not the constant. The average-case counterpart is F. P. Boca,
C. Cobeli and A. Zaharescu, "Distribution of Lattice Points Visible from the Origin",
*Communications in Mathematical Physics* (2000) `[SEARCH-SUMMARY]`, which determines the limiting
distribution of normalised angular gaps between primitive lattice points; and J. Marklof and A.
Strömbergsson, "The distribution of free path lengths in the periodic Lorentz gas and related
lattice point problems", arXiv:0706.4395, *Annals of Mathematics* **172** (2010) 1949–2033
`[SEARCH-SUMMARY]`, for general dimension and general lattices. Neither can produce `eps*(W)`, an
extremum at a specific algebraic direction under a hard budget.

**(f) The Lipschitz two-point free-segment lemma and its lower envelope.** The algebraic identity is
standard: the Piyavskii–Shubert lower envelope is `min_i { f(x_i) + L‖x − x_i‖ }`, whose sublevel
sets are exactly our union of balls; the salvaged summary notes the algorithm "works by building
unions of balls around sampled points, with radii determined by the Lipschitz constant and function
values". Verified in the salvage only via analyses — "Regret analysis of the Piyavskii-Shubert
algorithm for global Lipschitz optimization", arXiv:2002.02390, and "Cumulative Regret Analysis of
the Piyavskii–Shubert Algorithm and Its Variants for Global Optimization", AAAI 2024,
DOI `10.1609/aaai.v38i18.30057` (arXiv:2108.10859); **the original Piyavskii and Shubert papers do
not appear with citation details in the salvage.** `[SEARCH-SUMMARY]` That literature analyses
regret for *adaptively chosen* samples; nobody analyses the envelope on a **fixed lattice** of
sample sites, which is precisely what makes `eps*` a number-theoretic quantity rather than a
convergence rate. Related one-sided-safety constructions: Hart's unbounding spheres (§1.5);
G. Coiffier et al., "1-Lipschitz Neural Distance Fields", *CGF* 2024, DOI `10.1111/cgf.15128`
(arXiv:2407.09505), and Ludwig et al., "Strictly Conservative Neural Distance Fields", *CGF*,
DOI `10.1111/cgf.70528` — conservative by construction so the field cannot *over*estimate distance,
but the guarantee is on the value, never the sign `[SEARCH-SUMMARY]`.

**(g) Unions of balls, lower envelopes and their computation.** H. Edelsbrunner, "The union of balls
and its dual shape", *Discrete & Computational Geometry*, 1995 `[SEARCH-SUMMARY]` — dual complex from
the power diagram / regular triangulation, with short inclusion–exclusion formulas for topological,
combinatorial and metric properties of a union of finitely many balls; the right toolkit if `O` ever
needs to be computed or measured exactly. On the lattice side, D. Coeurjolly et al., "Optimal
Separable Algorithms to Compute the Reverse Euclidean Distance Transformation and Discrete Medial
Axis in Arbitrary Dimension", arXiv:0705.3343, plus Coeurjolly–Montanvert's 1D parabola covering
test `[SEARCH-SUMMARY]` — when does a union of balls centred at **lattice** points, with radii from
an exact EDT, cover a region, decided by upper-envelope-of-paraboloids tests. The paraboloid
formulation is a cousin of our constraint geometry, but there is no external surface, no sign to
certify and no extremal constant. Also standard in robotics practice: partitioning free space into
overlapping spheres obtained from an ESDF and planning inside the union ("Shape-aware Safe Corridors
Generation using Voxel Grids", arXiv:2208.06111, and others) `[SEARCH-SUMMARY]` — the *construction*
of `O` is folklore engineering there; what is absent is any same-side/sign claim from overlap, and
any worst-case analysis of what the union fails to cover.

**(h) Polyhedral / support-function view.** N. Normand and P. Évenou (§1.3) give the H-polytope
description of chamfer balls `[SEARCH-SUMMARY]`, which is the right vocabulary if we want to state
our result in chamfer language. Our own reading `[OURS]`: by LP duality the chamfer distance with
weights `omega_i` is the support function of `Q = {n : w_i·n ≤ omega_i}`; taking the **additive**
family `omega_i = |w_i| − delta` makes `Q_delta` a shrunken circumscribed polygon of the unit circle,
tangent at the mask directions, and `2 eps*` is exactly the `delta` at which its largest vertex
reaches the unit circle — equivalently, the smallest uniform subtraction from exact Euclidean weights
that turns the chamfer norm into a global under-estimate. Numerically the sign of
`max(d_C − 1)` flips at `delta = 0.0384052615 = 2 eps*` to within 1e-8. `[OURS:
scripts/distancing_experiments/lattice_gap/priorart/chk_shrink.py]` This gives our constant a clean chamfer meaning that, as far as
the salvage shows, has not been stated before.

---

## 3. What appears genuinely absent from the literature

Stated as a conjunction, because each ingredient separately exists somewhere:

1. **An additive, length-weighted deficiency as the optimised cost.** Every criterion found is
   multiplicative or relative: HHT's `limsup(W(v)/|v| − 1)` with a multiplicative weight family
   `[PRIMARY]`; Butt–Maragos' reciprocal-relative `limsup(|v|/W(v) − 1)` plus a disk-area measure
   `[PRIMARY, via HHT]`; Verwer's minimisation over circles and spheres; Maiseli's RLog and symmetric
   mean absolute percentage error; Malandain–Fouard's per-cone *relative* error extrema. Even
   Borgefors' absolute `|DT − EDT|` comes out proportional to `M` — she notes "as `M` is arbitrary
   the difference have been minimized everywhere" `[PRIMARY]` — so it is scale-free, i.e. relative in
   disguise; and the salvage records the area's own standard equivalence, that "optimizing the
   absolute error on a circular trajectory [is] equivalent to optimizing the relative error on a
   linear trajectory" `[SEARCH-SUMMARY]`.
2. **A budget on a projected quantity rather than on neighbourhood size.** In every chamfer paper
   the mask is constrained by `max(|x|,|y|) ≤ p` (stated verbatim in HHT `[PRIMARY]`), never by a
   bound on `w·n` or on `(|w| + w·n)/2`. The nearest thing to a budget anywhere is Hulin–Thiel's
   appearance radius, which is a budget on the *shape's inner radius*, not on the witness. Our
   two-sided band — upper cutoff `W`, lower cutoff at the lattice covering radius `sqrt2/2` — has no
   counterpart. Consequence, verified `[OURS]`: the `W`-indexed step function is strictly finer than
   the `M_p` mask family, producing plateaus (e.g. `0.011330789` at `W ≥ 3.17`) that no `M_p` yields.
3. **A name for `min_w (|w| − w·n)` over a bounded lattice region.** None found. The named lattice
   minima are all norm- or quadratic-form-based (successive minima, `lambda_1` of a form, norm
   minimum, covering radius); our objective is not a norm, is one-sided, vanishes identically along
   `+n`, and is minimised over a region cut out by an inhomogeneous constraint. Two dedicated
   searches for it returned explicitly empty on the intended meaning.
4. **The pairwise two-ball overlap rule as a named sign-propagation lemma.** Three separate targeted
   searches found only the strictly weaker one-ball "safety certificate" and the along-a-ray version
   in sphere tracing. Recommend claiming it as folklore, not as new (§1.5).
5. **Any error bound for an inside/outside classification derived only from exact point-to-surface
   distances at lattice sites, with no smoothness assumption.** Every guarantee found falls into one
   of four buckets, none of which is ours: feature-size-conditioned reconstruction; regularity
   (`r`-regularity) conditioned digitization by occupancy; sign presupposed as input; or sign derived
   from extra oriented geometry with no resolution-dependent bound. The flood-fill signing baseline
   is described in the salvage **with no attribution at all**, i.e. treated as folklore.
6. **A visibility problem with obstacle radius growing like the square root of distance.** Our
   blocking law gives effective radius `≈ 2 sqrt(eps·|w|)`; Pólya/Allen, dense forests, view
   obstruction and the Lorentz-gas literature all use *constant*-radius obstacles. No variant with a
   distance-dependent radius was found. If one exists it is the true classical analogue of our
   problem.
7. **The constant itself.** Grepping the three salvaged search-result files for `0192026`, `019202`,
   `15.9306` and `0449101` returns **zero hits**. (`0.04491` does appear, but only inside Borgefors'
   own recovered full text, per §1.1.)

---

## 4. Unverified leads — from model memory or from our own derivations, NOT confirmed by search

**Quarantined. Do not cite any item in this section as prior art without independent verification.**

*From model memory (no supporting document in the salvage):*

- **Reveillès / Debled-Rennesson arithmetic digital lines and DSS recognition** — the standard
  digital-geometry formalism for "which lattice points lie within a slab of given thickness around a
  line of slope `beta`", organised by the Stern–Brocot tree. Plausibly the readiest home for a proof
  that only Stern–Brocot-adjacent primitive vectors can be extremal for a slab/parabola capture
  condition. Not present in the corpus.
- **Voronoi "relevant vectors" and obtuse superbases** — the finite set of lattice vectors
  determining the Voronoi cell (in 2D exactly the three superbase differences); the standard rigorous
  answer to "which bounded set of lattice offsets suffices" for nearest-point/covering questions, and
  the machinery behind Mirebeau's stencils. Worth checking whether our admissible witness set is
  likewise superbase-generated.
- **Federer's sets of positive reach / `mu`-reach** — the canonical hypothesis replacing smoothness
  in this kind of argument. If a curved-interface version of `eps*` is ever needed, "reach ≥ R" is
  the assumption to look for. Never searched.
- **Davenport–Schmidt's non-improvability of Dirichlet's theorem** — the closest classical statement
  to "the extremal direction is where two consecutive convergents equioscillate". The salvage
  confirms the 1970 reference exists but contains no statement of the theorem.
- **Hurwitz / Markov–Lagrange spectrum.** Tempting because our closed form contains `sqrt5`, but
  **caution**: the `sqrt5` plausibly comes only from `|(2,1)| = sqrt5`, i.e. from the specific
  witness at `W = 3`, not from any golden-ratio extremality. Do not assert a Markov connection
  without checking whether `sqrt5` persists in the other plateaus.
- **Nooruddin–Turk parity/ray-stabbing voxelization repair; Tao Ju's octree-based robust repair of
  polygonal models** — both do sign propagation on a grid and might contain an unclaimed resolution
  condition. Neither appeared in any search result.
- **Nielson–Hamann asymptotic decider and the marching-cubes topological-ambiguity line** — "what can
  grid samples decide" in graphics, but as far as I know only for trilinear interpolants, not for
  distance-valued samples with a Lipschitz constraint. Not searched.
- **Interval / branch-and-bound root isolation (Moore-style)** — uses the 1D analogue of the overlap
  rule to prove an interval root-free; a plausible home if the rule is named anywhere.
- **C. O. Kiselman on regularity/metric properties of digital distance transformations**; **best
  inscribed/circumscribed lattice-vertex polygonal approximations to a circle** (the natural home of
  the *unweighted* version of our functional, `1 − cos theta_w`); **Chalk / Cassels, "An Introduction
  to the Geometry of Numbers"** and the inhomogeneous-minimum literature. All vague recollections; no
  specific reference offered.

*Our own derivations, offered as reformulations rather than literature claims `[OURS]`:*

- **Parabola reformulation.** Rotating so `n = (0,1)` with `Sigma = {y = −eps}`, the certification
  condition `eps > (|w| − w·n)/2` is equivalent to `w` lying strictly above the parabola
  `y = x^2/(4 eps) − eps` — focus at the target lattice point, axis `+n`, vertex at `−eps·n`,
  directrix `x·n = −2 eps`. So `eps*(W)` is the largest such parabola empty of *admissible* lattice
  points: a lattice-point-free convex region question with an extra admissibility constraint. The
  lattice-points-near-parabolas literature does **not** cover this (§1.9).
- **Asymptotic rate.** In slope coordinates the cost linearises to
  `g ≈ (q tan alpha − p)^2 / (2|w|) ≈ ‖q beta‖^2/(2q)` up to bounded factors — i.e. the classical
  `‖q beta‖^2/q` with a hard budget `q ≲ W`. Since best approximations give `‖q beta‖ ≈ c/q`, the
  cost falls like `q^{-3}`, predicting `eps*(W) ≈ c W^{-3}`: the crude `1/(2W^3) = 0.0185` against
  the true `0.0192` at `W = 3`, and `0.0038` against `0.0047` at `W = 5.1`. This is the precise sense
  in which the problem is length-weighted, and why the nearest-angle argument (`0.0299`) overshoots
  the truth (`0.0192`). A search for `max_alpha min_{q ≤ Q} q‖q alpha‖` and for `‖q alpha‖^2/q`
  returned nothing on point.
- **Support-function / additive-shrink reading** of `2 eps*` — see §2(h).

*Blocked or unfetched sources worth one more attempt, in priority order:*

1. **Malandain and Fouard, INRIA RR-5566** (`https://inria.hal.science/inria-00070440/document`) —
   the call that was denied and terminated the previous agent; blocked again by Anubis in this
   session. Its per-cone closed forms for local error are the one place an explicitly per-direction
   additive error formula might already exist. A supplementary search indicates they are *relative*
   error extrema (§1.7), which partially but not fully discharges the caveat.
2. **Butt and Maragos 1998 full text** — specifically their "critical local distances" (the closest
   published notion to "which mask vectors actually bind") and their per-sector geometric
   construction.
3. **Thiel and Montanvert 1992** (ICPR and TSI versions), and Thiel's 1994 Grenoble PhD "Les
   Distances de Chamfrein en Analyse d'Images" / 2001 HDR "Géométrie des distances de chanfrein" —
   the saved ICPR PDF is a fax scan with no text layer, so "influence cone" and "rational ball"
   remain unverified at source. The HDR is likely a readable, comprehensive substitute.
4. **S. Scholtus, "Chamfer Distances with Integer Neighborhoods", MSc thesis, Leiden University
   2006** — cited as [4] in HHT; credited there for the `E^C` / Butt–Maragos equivalence, and
   reportedly tabulates `(2p+1)×(2p+1)` neighbourhoods for `1 ≤ p ≤ 10`, i.e. our mask family at
   every budget level. Never fetched.
5. **"The orchard visibility problem and some variants", JCSS 74 (2008) 587–597** — never read
   (ScienceDirect 403), authorship unconfirmed; its Stern–Brocot-wreath technique is the most likely
   to transfer.
6. **Kohlbrenner–Alexa, Reach For the Arcs, Reach For the Spheres** — all three PDFs unread (size
   limit / denial); every statement about their theorems here is from search snippets.
7. **Whoever first wrote down the narrow-band flood-fill signing algorithm** — presented as folklore
   in the salvage with no attribution. It is our connected-components baseline; finding its origin
   would be worth 20 minutes.
8. **"Meshing Unsigned Distance Fields with Regular Triangulations"** (Kohlbrenner, *CGF*,
   DOI `10.1111/cgf.70524`), plus arXiv:2506.09579, arXiv:2605.01919, arXiv:2604.19568 — titles
   appear in the salvaged link lists but no summary or fetch exists. The first meshes *unsigned*
   fields and could plausibly contain a sign-recovery argument.

---

## 5. Search coverage and its limits

**How much was actually run.** The salvage contains **74 WebSearch queries and 21 WebFetch calls**
across three areas (chamfer/digital geometry: 15 + 12; computational geometry/graphics/robotics:
29 + 6; lattice geometry and Diophantine approximation: 30 + 3). Three primary full texts were
recovered locally with `pdftotext` after the fetch summariser failed on them: Borgefors 1986
(`scripts/distancing_experiments/lattice_gap/sm4n2x.txt`), Hajdu–Hajdu–Tijdeman 2012 (`scripts/distancing_experiments/lattice_gap/5dqxoe.txt`) and Hulin–Thiel 2009
(`scripts/distancing_experiments/lattice_gap/priorart/hulin.txt`). This merge added 2 WebSearch queries and 1 WebFetch. All numerical
claims attributed to us were re-run in this session (`scripts/distancing_experiments/lattice_gap/priorart/chk_borg.py`,
`chk_borg5.py`, `chk_hht.py`, `chk_shrink.py`, `final.py`).

**The search was cut short by a tool denial and is therefore incomplete.** Each of the three area
searches terminated on a *denied* WebFetch, so none of them reached a self-declared stopping point:

| Area | Denied call |
| :--- | :--- |
| chamfer | `https://inria.hal.science/inria-00070440/document` (Malandain–Fouard RR-5566) |
| compgeom | `https://onlinelibrary.wiley.com/doi/10.1111/cgf.70037` (Kohlbrenner–Alexa) |
| diophantine | `https://mathworld.wolfram.com/OrchardVisibilityProblem.html` |

The previous agent's own interim reasoning was lost — the "AGENT REASONING" sections of the salvaged
files are empty — so anything it had already ruled in or out is unrecoverable, and only raw results
survive.

**Other failed retrievals.** ScienceDirect returned 403 on the JCSS orchard paper; the INRIA HAL
landing page was behind Anubis (again in this session); the Thiel–Montanvert ICPR PDF is a CCITT-fax
scan with no text layer; three graphics PDFs (Reach For the Arcs, Reach For the Spheres,
Kohlbrenner–Alexa) exceeded the 10 MB fetch limit; Butt–Maragos failed on four separate attempts, all
returning the CVSP publications index page.

**What was not covered at all** — weigh the novelty verdict accordingly:

- **No scholarly index was searched.** No MathSciNet, zbMATH, Google Scholar, arXiv full-text search
  or citation-graph walk (e.g. who cites Hulin–Thiel, HHT or Bialkowski et al.). Everything is
  general English-language web search plus a handful of fetches.
- **No venue-scoped searching.** DGCI, IWCIA, *Discrete & Computational Geometry*, *Pattern
  Recognition Letters* and *Image and Vision Computing* are exactly where a lattice-certification
  result would live, and plain web search under-indexes them.
- **Nothing non-English**, and nothing systematic pre-1990. The Russian school on Klein polyhedra and
  one-sided approximation, and French digital-geometry theses (e.g. Rémy's "Normes de chanfrein et
  axe médian dans le volume discret", which appears only as a title), were not pursued.
- **The additive-deficiency vocabulary was barely probed.** Terms that would find our functional
  under different phrasing — "support function slack", "circumscribed polygon tangent at lattice
  directions", "weight offset", "uniform additive shrink of chamfer weights" — were mostly untried.
  Two supplementary searches on additive/offset phrasing returned only the same relative-error
  papers, which is *weak* evidence of absence.
- **Nobody searched for the extremal framing itself.** No query asked whether the *constant* exists
  ("largest depth at which a sample is unclassified", "certification radius", "capture depth", or
  the spectrum framing `{ sup_n min_w f(w,n) }` for a non-norm `f`). Every query asked whether the
  *method* exists. This is the single largest gap in the compgeom coverage.
- **The decisive rule was searched only as a literal quoted string** (`"d(a) + d(b)"`), which search
  engines handle badly, and never in prose form against a restricted domain, nor inside the full text
  of Bialkowski et al., which may well contain the two-ball variant as a remark.
- **No 3D coverage.** The real pipeline is 3D (covering radius `sqrt3/2`, mask = primitive vectors of
  a `5×5×5` or larger neighbourhood). The 3D chamfer references (Rémy–Thiel 2000,
  Malandain–Fouard) are both unread; the 3D analogue of "the two Farey neighbours suffice" is
  precisely where multidimensional continued fractions are known to be hard, so expect the 3D
  constant to require direct computation rather than transfer of a 2D theorem.
- **No search of the implementation literature this work sits inside** — OpenVDB mesh-to-volume sign
  propagation, Houdini/Bridson level-set signing, narrow-band conventions. If any of them documents a
  known failure depth, nobody looked.

**Net.** The verdict "the functional and the `W ≥ 2.24` constant are new; the machinery and the
`W ∈ [1.42, 2.24)` constant are not" rests on solid primary evidence for the *positive* claims and on
broad-but-shallow evidence for the *negative* claim. Before publishing a novelty assertion, at
minimum items 1–4 of §4's blocked-sources list should be read, and one venue-scoped pass over
DGCI/IWCIA should be run.
