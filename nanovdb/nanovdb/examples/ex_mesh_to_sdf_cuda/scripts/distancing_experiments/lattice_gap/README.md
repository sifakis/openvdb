# Verification scripts for the lattice certification gap

Supporting material for `../../../LatticeCertificationGap.md` (result) and
`../../../LatticeCertificationGap_PriorArt.md` (prior art).

Pure Python 3, no third-party dependencies (**no numpy** — it is not installed in the development
environment these were written in, and `pip`/`ensurepip` are unavailable there). Everything runs
standalone from this directory in seconds to a couple of minutes:

```bash
python3 curv2.py          # exact curvature law, and the table in section 4.5
python3 syn_v1.py         # five closed forms of eps_c agree to 70 digits; quartic; alpha*
python3 syn_v2.py         # exhaustive tightness at n*; finiteness bound; three-arc covering
```

## The problem in one line

On `Z^2` with a straight interface of unit normal `n`, a target lattice point at depth `eps` is
certified exterior by a lattice point at integer offset `w` iff `eps > g(w,n) := (|w| - w.n)/2`,
subject to the witness being non-barrier and in band. `eps*(W) = max_n min_w g(w,n)` is the deepest
an exterior lattice point can be and still escape certification.

## Layout

| file | what it does |
|---|---|
| `CONTEXT.md` | the problem statement handed to the verification agents. **Contains one known error** — its "threshold admissibility rule" justification — corrected in section 2.2 of the result document. |
| `CONTEXT_3D.md` | the 3D brief, with that error called out explicitly |
| `curv2.py` | exact closed form for a circular `Sigma`, `eps_thr = g/(1 -+ d0/R)`; the concave/convex tables |
| `syn_v1.py` … `syn_v7.py` | independent re-derivations written for the final document: closed forms, tightness, `eps*(W)` under both admissibility rules, upward-closedness, plateau endpoints |
| `ub_certificate.py` | upper bound as an exact `Fraction` interval certificate |
| `t_tightness.py` | tightness via the `Q(beta)` field tower and certified intervals |
| `e2e_sim.py` | end-to-end pipeline simulation — real lattice, real UDF, barrier marking, connected-components seed, enrichment closure. **Never uses the formula `g`.** |
| `anatomy.py`, `anatomy.json` | the critical configuration: tie, tangency, slopes, margins |
| `minpoly.py` | minimal polynomials of `eps_c` and the plateau endpoints |
| `core.py`, `s1_sweep.py`, `lib_dist.py`, `ref_main.py` | adversarial search and referee re-implementations |
| `chk_borg.py`, `chk_borg5.py`, `chk_hht.py`, `chk_shrink.py`, `final.py` | prior-art checks: the Borgefors 1986 identity, the `5x5` non-identity, the Hajdu–Hajdu–Tijdeman extremal pair, the chamfer-shrink reading, and the `M_p` plateau closed form |

## Two gotchas that cost real time

1. **Do not use the 15-digit decimal `0.019202630763796` as a bound.** It is *below* `eps_c`, so the
   inequality `g <= 0.019202630763796` is false on a window of width `4.2e-15` rad around `alpha*`.
   Rounding to 16 digits does not fix it either — one must round **up**, to
   `0.0192026307637964`, or use the exact surd `(3 - sqrt5 - sqrt(2 sqrt5 - 4))/4`.
2. **The strict predicate needs a guard.** With plain doubles, `UDF(A) + UDF(B) > dist(A,B)`
   certifies points on the wrong side whenever `n` is parallel to an integer vector, because the
   comparison is an exact tie there and rounding breaks it the wrong way. Use
   `UDF(A) + UDF(B) - dist > tol`. See section 4.6 of the result document.
