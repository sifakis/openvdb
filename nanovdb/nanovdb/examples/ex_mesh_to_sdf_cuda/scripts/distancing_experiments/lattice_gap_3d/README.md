# Verification scripts for the 3D lattice certification gap

Supporting material for `../../../CertificationGap_3D_Note.md` (readable summary with figures) and
`../../../LatticeCertificationGap3D.md` (full technical record). The 2D result these build on is in
`../lattice_gap/`.

Pure Python 3, no third-party dependencies (**no numpy**).

## The two constants

```
threshold rule   eps*_thr = (3 - sqrt5 - sqrt(4 sqrt30 + 2 sqrt5 - 26))/4
                          = 0.036662265118829642294259933043...
                     tie  = (1,0,0), (2,1,0), (2,1,1)

exact rule       eps*_exa = (sqrt2 + sqrt3 - sqrt6 - sqrt(6 sqrt2 + 2 sqrt6 - 13))/2
 (the pipeline)           = 0.038443423574719626566412269573...
                     tie  = (1,1,0), (1,1,1), (2,1,1)
```

Both degree 8 over `Q`, irreducibility proved. Use the **exact-rule** value, rounded **up**:
`0.0384434235747197`.

## Layout

| file | what it does |
|---|---|
| `CONTEXT_3D.md` | the brief handed to the verification agents; states the threshold-vs-exact correction explicitly |
| `figs3d.py` | regenerates all four figures of the note into `../../../figures/`. Imports `../lattice_gap/figs.py` for the SVG helpers — run it from a checkout with both directories present. |
| `syn_v1_forms.py` | the closed forms and both octics at 300 digits; residuals ~3e-297 |
| `syn_v2_bnb.py` | branch-and-bound certificate for both constants |
| `syn_v3_plateau.py` | the `W`-map under both rules, with the corrected exact-rule floor |
| `syn_v4_curv.py` | curvature from **raw sphere geometry** — never uses the reduced cost, and is what settled the mislabelled first-order table |
| `syn_v5_misc.py` | angular covering radius by exhaustive Voronoi-vertex enumeration; misc |
| `ref_value_ivl.py`, `ref_value_r2_certificate.py` | referee 1: polytope-vertex certificate in **exact rational interval arithmetic** — this is what upgraded the constants from certified-numerical to proved |
| `ref_value_r1_closedforms.py` | referee 1: independent closed-form and minimal-polynomial verification |
| `ref_value_r10_e2e.py` | referee 1: from-scratch end-to-end simulator, never uses the reduced cost |
| `ref_structure_r2_bnb.py`, `ref_structure_rlib.py` | referee 2: independent spherical-triangle branch-and-bound |
| `ref_structure_r5_cover.py` | referee 2: covering-radius check that caught the `17.6337 -> 17.6532` error |
| `theta3d.py` | angular quantities: unweighted covering radius vs the weighted critical angles |
| `w3d_crude_map.py` | a coarse `eps*(W)` sweep, seeded with the known critical directions — useful as a fast sanity check, but it under-reports sharp peaks (see below) |

## Four things that will bite you if you reimplement this

1. **Two admissibility rules, and they differ in 3D.** A witness must be in band, and its depth
   `eps + w.n` depends on the *target's* depth. Evaluating at the witness's own threshold (`d0 =
   (|w| + w.n)/2`) is **optimistic** — at `W = 3` it admits `(2,2,1)`, which the pipeline cannot use,
   and for narrow bands it is optimistic by an unbounded factor.
2. **The uncertified set is not an interval.** At 8.45 % of directions it has a hole, because a
   witness with `|w|` near `W` has a usable window bounded *above*. **Any measurement that bisects on
   depth is unsound** — this produced a wrong published endpoint during the work.
3. **The finiteness bound differs by purpose.** Locating `eps*` needs `|w| < W + G` (122 vectors);
   computing the *supremum of the uncovered set* under the exact rule needs `|w| < W + r_d = 3.866`
   (250 vectors). The distinction is invisible in 2D.
4. **Grid searches under-report.** The peaks are sharp and sit at cell corners of the envelope; a
   coarse sweep is a lower bound, not an estimate. Several phase-1 numbers were grid artifacts. Use
   the branch-and-bound or vertex certificates for anything load-bearing.

## Guard the predicate

The 2D floating-point hazard is far worse here: with `tol = 0`, one 1-ulp error at a lattice-parallel
normal cascades to **~1300-1700 wrongly certified points**. Minimum safe `tol` is between `1e-15` and
`4e-15`; use `1e-12`.
