# Qualified Bi2Se3 fixed/free structure-factor comparison

## Current stop decision (2026-09-16)

The user separates the simulator optimization freeze from the six-sample fitting
campaign. Stop further optimization and fitting in this phase. The frozen software
replays the complete preserved Bi2Se3 N13 baseline under identical numerical rules;
this is a software-regression MATCH, not an independent quadrature-convergence
certificate or acceptance of any fitted parameters. The broader matched-fit,
N-comparison, identifiability and accepted-image campaign is deferred and incomplete.

`bi2se3_final_baseline_agreement.ra_diag.npz` in the external current-task
visualization directory records all 1250 observations and 85 rods. Current-proposal
center plus two gamma probes agree with preserved vectors to at most 3.68e-15
whitened RMS; contrast RMS is at most 4.99e-15 and objective-contrast error is
2.92e-11. Historical guard decisions/feasibility are unchanged. Replaying the old
proposal against the original saved raw vector agrees to 6.12e-15 sigma. The
entire gate takes 149.36s under a 180s hard limit, with 1,272,709,120-byte peak
working set. These values comfortably pass the existing nominal tolerances;
byte equality was neither required nor used as the acceptance criterion.

All 371 permanent tests remain current: every one of the software evidence's
149 source/test/script hashes matches the frozen checkout. No production code
changed after that proof. Only measured arithmetic speedups are claimed: the
cone optimization is 2.75x on its recorded equivalent-work benchmark and spatial
order8 reduces sampled projection cost by about 30%; neither is a whole-fit
speedup. Existing PNGs remain provisional and unchanged. Regional convergence
evidence below is retained, including failed and incomplete cases; no full
convergence claim is inferred from the software replay.

Retained permanent coverage protects physical panel measure/support and native
mass conservation; elastic-cutoff seeding; adaptive stencil error/work limits;
zero-safe reference correction and exact proposal rejection; anomalous signed
atomic amplitudes; source partition/rebinding and render recovery; and the
Gaussian-underflow and spatial rectangle oracle boundaries. No exploratory
production tests, new dependency, detector-image mirroring or physical model
was introduced. Legacy physics classifications are unchanged; new numerical
boundaries remain NO_ORACLE against original-RASIM and use independent identities
or same-core parity. No branch integration into main is implied by this freeze.

## Authorized noise-bounded acceleration

The user requests error-driven axial integration, reference-based acceleration of
nearby fits, and safe reuse of in-plane/00L symmetry. Continue the isolated T35
worktree from main `2b80a2c`; `5d9912e` is the unchanged software baseline.
The main agent remains the only writer. Read-only agents challenge integration,
reference correction and symmetry independently.

1. Add immutable, explicitly rod-keyed physical axial panels to the existing
   quadrature. Compare parent and child panel contributions after the existing
   detector projection, in the observation covariance. Freeze a common accepted
   mesh across the declared candidate stencil. Seed complete peak-aware support;
   fail explicitly at work limits and retain independent refinement checks.
2. Accelerate proposals with `high(reference) + low(candidate) - low(reference)`.
   This zero-safe defect correction implements reference-based reuse without
   assuming smooth ratios at structure-factor zeros. Freeze each reference during
   a bounded optimizer call. Validate proposed improvements with the high rule;
   never write approximate predictions into exact recovery or use them for final
   qualification. The final ordinary all-active stage remains authoritative.
3. Reuse exact inversion-related atomic geometric sums while keeping complex
   species scattering factors unchanged. Preserve every signed rod, termination,
   wavelength and separate signed intensity. Detector-image mirroring, Friedel
   intensity equality and new physical symmetry assumptions are prohibited.

Use the existing physics and SciPy dependency. No reduced-order model, generic
backend, new optimizer or physical approximation. Permanent tests cover the
distinct mesh measure/support boundary, adaptive error response, exact/approximate
state separation and anomalous signed-amplitude identity. External benchmarks
include mesh/reference construction, repeated fitting work, rejection/restarts
and peak memory. Keep changes only when equivalent-work performance improves
without weakening the existing prediction/contrast/objective gates. Numerical
estimates remain empirical, not certified global bounds or fitted uncertainty.

Base: `05de593df436f9f9d0ea8d6482f96acb041bf1d7`. One writer; agents read-only.
User authorizes completing the full39 fit, fixed13 control, N alternatives,
ambiguity profiles and detector comparison. Relevant ledger: PHY-FIT-023/024/025.

## Plan before implementation

1. Replay the saved full endpoint through consolidated production code. Recenter
   the numerical mosaic proposal on the saved physical mosaic, freeze it across
   actual Lorentzian-width optimizer steps, and refine angular order independently.
   Preserve observations, covariance, source, N, full rods, normalized proposal,
   training split and existing numerical gates. Stop broad fitting while they fail.
2. Close two demonstrated public-runner gaps: an explicit fixed-parameter control
   roster and a local-m0 axial panel cap. Preserve ordinary full-release validation;
   conditional controls retain fixed values throughout fitting and qualification.
   Identification profiles remain the responsibility of the released full model.
3. Resolve remaining numerical components at the saved endpoint, then qualify the
   actual optimizer stencil. Refit fixed/free models under identical accepted rules,
   compare neighboring N, profile weak coordinates, and render accepted candidates.

No physical equations, thresholds, observations, or dependencies change. Numerical
proposal adjustment is not qualification by itself. No new optimization project.
Permanent regressions cover distinct public boundaries: channel-specific integration
equals explicit additive partitions; fixed-control declarations preserve fixed values
and cannot bypass full-release validation. External numerical artifacts retain input,
code and candidate identities, once-profiled scales, timing, memory and gate results.
Run compact suite, formatting/lint, registered proofs and inventory before handoff.

## Implemented boundary and measured outcome

The public runner now supports an explicit fixed-parameter control roster and a
local-m0-only axial panel cap. The latter reproduces the previously split raw
response through one evaluator: relative differences below 2.7e-17 at the saved
full candidate and two gamma probes. No numerical threshold or physics changed.
The compact suite passes 366 tests; formatting/lint and the six scientific proofs
that permit an uncommitted checkout pass. The two clean-checkout wrappers and
inventory are final handoff gates, not fitting qualification.

The consolidated old-rule replay agrees with the saved raw prediction to 4.0e-17
relative norm. Its angular proposal Lorentzian width was approximately 585 times
the fitted width. Recentring the numerical proposal is necessary but insufficient:
successive angular g5/g6, g6/g7 and g7/g8 comparisons have endpoint disagreements
6.80, 1.69 and 0.548 sigma. At g7/g8 the gamma objective-contrast error is 3.95,
above the existing 0.5 gate. The isolated local-m0 contribution passes that last
comparison; other rods do not. Axial and source qualification remain unresolved.
The previous gamma step is about 8.8% of its fitted width and also needs a smaller
step and a step-halving check before derivative-based termination is credible.

An independent fixed-Q/source slice identifies narrow detector-acceptance windows
undersampled by the global angular proposal. Physical midpoint integration at
32768/65536 angles agrees to 2.1e-10 sigma over all 1250 observation rows. Retaining
all region endpoints, tighter geometric boxes or uniformly finer physical panels
does not establish a lower-cost replacement. A temporary adaptive GL8 experiment
uses 920 retained nodes versus 2048 for CDF g10, with 1.56e-5 versus 2.60e-5 sigma
slice error. However, adaptive construction costs 4968 evaluations and 0.567 s;
the 0.100 versus 0.269 s frozen projection timings are not optimizer speedups.
The production fitter already caches those projections. No adaptive engine,
alternative physical model, fitted result or qualified uncertainty is introduced.

User priority: lightweight, fast full-SF optimization. Preserve the persistent
evaluator for response-only blocks and use existing staged fitting, finishing
with all admitted coordinates released. Parallel prediction groups create new
evaluators and must not be presumed faster than persistent serial evaluation.
Any future integration change must beat equivalent response construction and
actual cached fitting work, including memory, before adoption. Avoid further
blind global order sweeps or a new adaptive framework based on one slice.

The measured full/free and fixed-SF fits, N alternatives, ambiguity profiles and
accepted detector comparison are not completed by this software boundary. Failed
numerical qualification remains explicit; neither old candidate is promoted.

Evidence resides outside the repository under
`C:/Users/Kenpo/.codex/visualizations/2026/09/11/01a09194-718e-7e83-ab46-0fc959c9d928`:

- `bi2se3_current_proposal_qualification_20260914.ra_diag.npz`: complete predictions,
  actual probes, partition attribution, timing and 5.03 GB peak working set.
- `bi2se3_qualified_runner_boundary.ra_diag.npz`: unified/split public-boundary parity;
  three evaluations take 90.34 s with concurrent work, not a speedup measurement.
- `bi2se3_first_family_current_proposal_20260914.ra_diag.npz` and
  `bi2se3_first_family_angular_slice_20260914.ra_diag.npz`: source/axial attribution
  and independently converged angular slice.
- `bi2se3_angular_slice_cost_screen_20260914.ra_diag.npz`,
  `bi2se3_angular_all_regions_cost_screen_20260914.ra_diag.npz` and
  `bi2se3_angular_parent_panels_cost_screen_20260914.ra_diag.npz`: rejected production
  candidates and the distinction between construction and cached evaluation cost.
- `qualified_bi_software_proofs_20260914.ra_diag.npz`: software checks and final
  handoff evidence. Each artifact embeds its own numerical arrays and JSON manifest.

Permanent coverage added: one fixed-control public-CLI boundary test. The existing
native-response test additionally protects local-m0 partition additivity and
numerical-revision invalidation. Existing legacy classifications remain unchanged;
these new boundaries are NO_ORACLE against legacy and use exact same-core parity.

## Resumed execution plan (2026-09-15)

Base `2b80a2c3d3f6877988e233b99e60021d11fc58d9`; branch `codex/bi-fit-execution`.
Use existing quadrature and fitter. Compare local m0 g7/g8 and nonzero rods g9/g10
at both retained candidates and response-reusing optimizer perturbations. Check
axial/source refinement next, then execute fixed13/free39 at N=13 with the same
accepted rules. Add only a local-m0 angular-order override so the stock evaluator
can express these existing additive rules without over-resolving m0. Extend the
existing partition/revision regression. Numerical correctness comes first;
performance is judged by total fitting cost, not each individual evaluation.

The local angular override passes the existing additive-partition and revision
regression, all 366 permanent tests, and independent review. No new test function
or physical equation is added. At the retained full candidate, the combined
m0 g7/g8 and regular g9/g10 comparison passes: maximum prediction disagreement
0.029744 sigma, perturbation disagreement 0.000034702 sigma, and objective-contrast
error 0.015793. The same-rule gamma step-halving disagreement is 2.807%.
Uncapped regular axial p7/p8 fails at 6.7035 sigma; source divergence GH2/GH3
fails at 1.2909 sigma, concentrated around the central (003) peak. Existing
physical axial panel caps and finer source quadrature are being checked externally.
These are numerical checks at old parameters, not new fits. The matched plans
retain initial and fitted qualification gates; the full fit will use the completed
fixed-fit optimum as its sole initial seed and finish with all 39 coordinates free.

The measured source-order requirement is concentrated in stitched m0: regular
GH2/GH3 differs by 0.0077 sigma, while m0 GH4/GH5 still has a 0.649 objective-
contrast discrepancy. Avoid multiplying regular-rod work by applying the higher
source order globally. Add one optional local-m0 divergence order, one explicit
physics partition method, and use the existing integration and response caches
for both partitions. Fitting, qualification, controls, stitch records and rendering
must consume those same partitions and sum raw intensities before one scale.
Prove equal-order parity, unequal-order additivity and physical-source rebinding;
benchmark matched partition work. No new physical model or generic backend.

## Lean continuation

User requests nominal scientific agreement, not byte equality. Stop resolution
refinement once prediction changes are below measurement noise and optimizer
contrasts are stable under the declared gates. Existing source GH5/GH6 predictions
at the saved full candidate and gamma probes pass: 0.046122 sigma prediction RMS,
0.00014319 sigma contrast RMS and 0.372461 objective-contrast error. No further
source-order sweep is needed there. This does not qualify another candidate.
Finish the local-source boundary, assess the already-running axial comparisons,
then use identical accepted rules for fixed/free fitting. Reuse persistent responses
in response-only stages and release every admitted coordinate in the final stage.

Live CPU sampling identifies cone averaging as the remaining axial-evaluation
bottleneck. The narrow implementation skips Gaussian cone integration only when
the minimum tilt exceeds 40 sigma, where every wrapped-image exponential already
underflows in float64. Lorentzian tails and broad Gaussian cases retain their
equations. The existing cone/conservation regression covers the near-antipodal
limit and mixed-component result. Benchmark equivalent cone work before adoption.

Validation: all 367 permanent tests pass; the updated cone oracle/conservation
test additionally passes after the underflow optimization. Formatting/lint and
the six ordinary registered scientific proofs pass. No type checker is configured.
Independent review finds no remaining source-partition, cache or render issue.
The near-antipodal extension protects a distinct zero-Gaussian/nonzero-Lorentzian
limit; no separate test file or exploratory test is retained.

`bi2se3_lean_boundary_20260915.ra_diag.npz` records an alternating five-repeat
32768-cone benchmark: median 0.10012 s before, 0.03639 s after (2.75x), with agreement
within 2e-12 relative. This is an arithmetic benchmark, not a whole-fit speedup.
A coarse full-observation-support partition check agrees with independently
assembled parts for the center and two gamma probes. The unified evaluator builds
exactly two responses (3.67 s total compilation), then reuses them for both probes
(2.45/2.39 s); whole-process peak working set is 541024256 bytes. Explicit split
and unified cached costs are comparable; first-build timings include warmup and
concurrent work and establish no additional speedup.

The completed .04 axial-cap comparison disagrees with uncapped p8 by 2.5552 sigma;
neither comparison qualifies axial integration. Its first prediction took 5560.55 s,
then cached gamma probes took 64.87/65.24 s (9.32 GB peak). Its redundant .02 work
was stopped after all three .04 vectors were safely checkpointed. The independently
started .02 run remains the next numerical result to consume. Fixed/free fits,
N alternatives, uncertainty profiles and accepted detector images remain pending.
These numerical software boundaries remain NO_ORACLE against legacy; same-core
partition parity and independent cone identities supply proof. Existing reference
classifications and physical equations are unchanged.

## Bounded axial repair and image preview (2026-09-15)

Mesh preparation was discarding the input rule's physical panel-width caps when
switching to explicit meshes. It now applies global/local caps to every seed gap,
preserves existing and elastic-cutoff edges, and rejects an excessive seed before
prediction work. The existing public CLI regression first failed on an uncapped
7.5 A^-1 gap, then passed with independent regular/local caps and budget rejection.
Five focused tests, Ruff and format checks pass. No physics or tolerance changes.

Actual Bi2Se3 N13 center/gamma checks use the worst remaining m0 row 1035 and
in-plane row 58. Complete cropped observables, all 85 rods and the bound source
partitions are retained. Physical caps .003 and .0025 A^-1 agree at 3.12e-12 sigma
prediction RMS; contrast and profiled-objective checks also pass. Individual
successful checks take 54--153 seconds; every diagnostic has a hard wall limit.
This is two-row empirical evidence, not full-observation or fit qualification.

The separate full-panel render uses the same starting candidate and scalar,
all rods, no fit-ROI response, and an explicitly coarser image quadrature. Its
30-by-30 native-pixel macrobins preserve the physical panel and Gaussian box
measure; a small independent block-sum check agrees within 1.8e-15 absolute.
Images remain provisional: no refit, accepted N ranking or uncertainty claim.
The tensor image rule exposed visible quadrature spokes. The existing joint
Sobol rule (axial power 12, angular power 4) removes that structured display
aliasing without smoothing and renders all components in 119 seconds. This is
a separate image estimate, not qualification of the native fitting rule.
Evidence and requested PNGs are external under the current task visualization
directory `01a0a308-8d41-7f73-bb85-92277ba73dd6`. Full-suite/branch integration
gates were not run under the user's short-diagnostic budget; draft remains uncommitted.

## Bounded continuation (2026-09-16)

Saved real Gaussian-kernel batches reject the presumed zero-call bottleneck:
almost all candidate rectangles contribute; CSR entries combine many rectangles.
A scalar-index rewrite was reverted because its timing gain was not robust.
The retained narrow change makes the explicit spatial order govern both Gaussian
rectangle integrals. Order 16 is unchanged; order 8 reduces the sampled projector
cost by about 30%, with maximum sampled rectangle disagreement 7.46e-9. This is
not an end-to-end speedup or an intensity-error certificate. The existing oblique
beam oracle now also checks order 8; the nine source-spatial tests pass.

At the unchanged N13 center and actual gamma probe, complete in-plane parent26
(rows 53--62, all 85 rods) gives these independent local comparisons: axial caps
.01/.003 at g6/m5 agree to 9.04e-5 sigma; angular g7/m6 versus g8/m7 at cap .01
agrees to .001790 sigma; local source GH2/GH5 agrees to 6.57e-8 sigma; spatial
orders 8/16 agree to 6.98e-7 sigma. All corresponding contrast gates pass.
The lower g6/m5 angular rule fails at .1753 sigma and is not selected.

Full-branch and high-order peak pilots hit their 180-second deadlines. Splitting
branch 0 at a parent boundary gives a complete first-half center/probe in 172.5s.
Its cap .02/spatial 8 version agrees with cap .01/spatial 16 to .000601 sigma and
finishes in 48.4s (same g7/m6, source GH2); these timings include panel/ordinary
prediction parity checks. Individual-parent subdivision was rejected by a
geometry work-count audit: 677 nonempty jobs would multiply total work. Whole
branches, split only when needed, are the remaining bounded numerical pilot.
No full 1250-row qualification, fit, N ranking or uncertainty claim follows from
these local results. All retained evidence is external under the current task's
visualization directory; the isolated draft remains uncommitted.

The continued bounded software gate completes all 371 unique permanent tests in
disjoint chunks. Maintained-code Ruff checks and formatting pass (166 files), as
does `git diff --check`. The six ordinary scientific commands pass their numerical
checks; geometry-optics and mosaic-ewald still report BLOCKED solely for their
clean-checkout wrapper. Four other commands return PASS. No fitted result follows
from these software checks.

Whole branch0 at cap .04/g7 completes in 80.5s, but disagrees with the previously
cropped half by .672 sigma. Raising only its angular rule to g8 reduces that
disagreement and takes 117.6s; g9 exceeds the 180s limit and is not repeated.
The first half at cap .02 requires g8: g7/g8 differs by .2505 sigma, whereas
g8/g9 passes at .03607 sigma, .00005271 contrast sigma and .000676 objective
contrast error (73.6s/131.6s). Whole-group and cropped quadrature are not assumed
interchangeable merely because they cover overlapping observations.

The first central half has a separate resolved cause for axial under-sampling:
its fitted film thickness is 2548.736 A, producing approximately .002465 A^-1
far-from-critical Parratt fringes. Refraction makes the shortest local external-Q
period smaller. Caps .04/.03 span many fringes per GL8 panel and disagree by
3.011 sigma; a true bisection of the .04 mesh also fails, at 3.592 sigma. All
this difference is local m0. The existing stitch state confines the Parratt
contribution to external Q below .30393330 A^-1 for this center/gamma stencil.
The next external seed preserves all original/elastic edges, inserts exact
critical/blend edges, and bounds both external-Q and the authoritative refracted
phase-Q increments only below that upper blend edge. It changes no equation.
Central m5/m7 angular rules agree at .01454 sigma with passing contrast gates
on the identical .03 mesh; lower angular work makes this targeted fringe check
practical. This angular comparison must also survive the resolved axial mesh.

Rejected additions: dictionary corner caching is slower; compact array corner
caching is essentially neutral for local-m0 order8, so neither enters production.
A 39-probe bind-only audit finds 17 exact response-reusing probes and 22 genuine
geometry/source/optics changes, not false material-revision invalidation. Keep
cache keys intact. The two-entry per-rod response LRU can evict the center during
finite differences; exact evaluation scheduling is a possible later fitting
improvement, not a reason to change physics or increase caches speculatively.

The first central half now passes two resolved checks. With phase-Q and external-Q
increments capped at .0025 A^-1 below the Parratt blend end, ordinary axial caps
.01/.005 agree to .03620 sigma (contrast .00001247 sigma, objective contrast
.00805). On the coarser passing .01 mesh, m5/m6 angular integration agrees to
.01636 sigma (contrast .00002483 sigma, objective contrast .05993). Stop refining
those dimensions for this center/gamma group; these remain local empirical checks.

External source-row checkpoints preserve original source indices and normalized
weights, retain full-source support bounds and all rods, and reuse responses only
within an identical-physics source chunk. The same-rule first in-plane half agrees
with the unchunked prediction to 1.10e-14 sigma. Each invocation is capped at 180s;
completed group/source/candidate arrays resume only under matching code, inputs,
driver, mesh, candidate and chunk identities. No partial artifact is a prediction.
The second half of branch0 and both halves of branch1 are being completed with
this existing-physics diagnostic. Full-observation covariance, the actual fitting
stencil, matched fits, other samples, N comparisons and accepted images remain
unresolved; software-test success does not qualify any fitted result.
