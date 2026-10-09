# Fitting workflow and manuscript figures

This is the working guide distilled from the September–October 2026 studies of
Bi2Se3, Bi2Te3 and PbI2. It records useful methods and failed interpretations;
it does not declare any current specimen fit physically qualified. Historical
campaign instructions are archived as described in [REPOSITORY_HISTORY.md](REPOSITORY_HISTORY.md).

## Use the supported numerical owners

Use `scripts/prepare_native.py`, `scripts/refine_native.py` and
`scripts/render_native.py` as documented in [NATIVE_REFINEMENT.md](NATIVE_REFINEMENT.md).
The owners are `NativeFitObservations`, the native physical evaluator,
`score_native_prediction` and `fit_native_parameters`. Keep one implementation
of each physical equation, coordinate transformation and measurement operator.

For a practical launch, recovery and result-inspection sequence, follow
[running a fit with automatic device selection](NATIVE_REFINEMENT.md#run-a-fit-with-automatic-device-selection).
Native batches select CPU/GPU automatically; the observation and background
recipe still determines what is fitted.

`render_native.py` writes numerical render diagnostics. The recent manuscript
figures use external Matplotlib callers saved inside their diagnostic packs.
Reuse their presentation conventions below; do not copy experiment-specific
physics, fitting orchestration or frozen parameters into production modules.
The supported exponential-background profiler and the external 41-knot monotone
background solve are distinct nuisance models, not interchangeable implementations.

## Procedure for a new or revised fit

1. Bind the acquisition, specimen, raw counts, CIF/structure, geometry, source,
   wavelength/exposure, numerical rule and candidate state by their saved identities.
   The newer Bi2Se3 5/10/15-degree acquisitions share a specimen; the April 6-degree
   film is a different specimen. Both are continuous films on SiO2, which does not
   imply a single crystal or exposed escape surfaces on every tilted crystallite.
2. Freeze native pixel memberships, support areas, overlap/covariance ownership,
   likelihood flags and collection validity before fitting. A changed physical
   candidate must not move the observations or silently change their masks.
3. Treat holder-obscured support as unavailable, not zero signal. Use the same
   acquisition-specific visibility rule for peak and background-control supports.
   Record boundary and guard assumptions. Do not apply an off-axis shadow rule
   automatically to the clear positive 00L sector, or exclude bins because of their
   residual sign. Keep the image visible even where fitting support is excluded.
4. Predict signal on controls as well as peaks. Fit `raw = physical signal + background`,
   or an explicitly frozen signed background-subtracted target with its covariance.
   Use the declared shared physical scale per acquisition; no independent peak or
   family scales, shifts, clipped measurements or inflated noise to manufacture agreement.
5. Establish a matched baseline using the same operator, priority weights,
   background policy and numerical rule as the candidate. A previous displayed
   figure with a different background, normalization or window is a separate comparison.
6. Use a finite stage to answer a specific scientific question. Declare fixed/free
   coordinates, bounds, parameter coupling, actual prediction count, best evaluated
   state, returned endpoint and unfinished starts. Optimizer evaluation limits may
   exclude derivative probes. An evaluation limit is not convergence.
7. Assess positions, areas, core/flank shape, widths and tails jointly across 00L
   and indexed off-axis families. A lower total score can hide worse protected
   regions. Six off-axis L cuts do not independently resolve radial in-plane and
   azimuthal widths, and overlapping profiles are not independent evidence.
8. Apply nominal numerical screens at fixed physical state, scale and background.
   Report numerical completion, sampling consistency, physical adequacy and
   parameter identification separately. History is useful nominal evidence even
   across engine changes, but cannot substitute for a matched causal comparison.
9. Render the saved state through the same physical implementation. Integrate
   display profiles using their actual native operators, not reconstructed pixels
   or subdivisions of saved coarse receiver totals. Do not refit while plotting.

For new numerical work, freeze inputs and finite work/memory limits, include
termination/publication in one all-in deadline, retain completed components and
record actual work. Failure ends the admitted stage; no automatic heavy retry.
The stopped local-m0 integration campaign is historical evidence, not an implicit
next step. Another numerical-method investigation needs a named current defect
or a discrepancy under the same integrand, anchor, measure and observable.

## Background and validation controls

A smooth, nonnegative, nonincreasing radial component is a useful empirical
restriction. Estimate it from declared off-peak evidence while protecting Bragg
rings across all azimuths. Add a directional strip or other component only when
its spatial pattern is supported and separable from sample signal. Retain the
original radial resolution: coarse interpolation has reversed a saved net
integral's sign. Do not assume every halo, tail or low-angle feature is background.

Predict physical signal on all control supports; only declared training controls
enter fitting. Diffuse stacking signal can otherwise be absorbed into background
or compensated by structure parameters.
Monotonicity alone does not validate the background under peaks or imply that
the true whole-detector background is radial. Collection loss, unidentified
scattering and fitted nuisance remain possible confounders.

Preserve the complete background model when handing a saved fit to another caller,
including angular components, coefficients, native operators and qualification.
Replacing a radial-plus-angular model with a radial-only curve is a new nuisance
model, even if geometry and the acquisition are unchanged. Bind that change explicitly
and repeat its control checks before reuse. Background adequacy and physical-model
preservation are separate decisions; neither can compensate for the other.

Freeze spatial selection and audit supports at the native-pixel level before
choosing a correction. Row-held-out bins can still share training pixels.
Preserve cross-covariance or use disjoint support; label repeatedly inspected
same-image controls as development evidence. Stronger evidence is prediction of
reserved spatial regions or a separately bound acquisition.

### Required background checks

Before solving, freeze support, visibility, control selection and acceptance limits.
After solving, complete prediction and residual checks before adopting a background/
refit or exporting its figure. These are operator/caller requirements; existing
numerical validation alone does not establish that they ran. Reuse a recorded check
only when its acquisition, controls, background, physical prediction and observation
identities remain unchanged. Reuse the procedure across samples; transferring fitted
coefficients requires acquisition-specific evidence or an explicitly shared model.

1. **Establish local support.** Record actual native-pixel footprints, radial coverage
   and gaps, azimuthal coverage, and the background coefficients contributing to each
   fitted or displayed region. Knot positions, control centres and a global minimum/
   maximum radius are insufficient. Distinguish data-supported prediction, partial
   support and extrapolation, including constant endpoint extension. A monotone curve
   can have an unsupported level. Declare transfer assumptions under protected peaks;
   nearby controls do not prove radial symmetry or identify the background there.
   Near a holder edge, match control and target clearance as well as radius and
   azimuth. Count distinct spatial blocks, not just adjacent control cells. For a
   spatial basis, report training support for the actual coefficients contributing
   to each target footprint; smoothness regularization is not measured support.
2. **Separate collection from propagation.** Bind the observed holder boundary and
   guard to the acquisition and apply consistent visibility rules to peaks and controls.
   Record an assumed film horizon separately: it may constrain physical propagation
   but must not silently become measured collection loss. Clear pixels do not certify
   every modeled escape path. Do not transfer a guard width blindly between images.
3. **Check prediction on separate controls.** Freeze native control membership and
   acceptance limits before choosing a correction; include predicted sample scattering
   on controls. Where coverage permits, fit one angular block and predict another,
   then reverse, reporting regional signed bias and error in declared count units.
   Reserved checks must share no positive-membership native pixels with training.
   Preserve covariance for overlapping development checks; this does not make them
   independent validation. If disjoint blocks are unavailable, record limited evidence
   rather than inventing them. Reused development controls are not untouched evidence.
4. **Investigate signed residual patterns.** Report broad negative troughs, negative-bin
   fractions and signed interval masses with their spatial context. Use available full
   covariance; otherwise label summaries descriptive, not significance tests. Correlated
   bins are not independent trials. These alarms trigger diagnosis, not clipping,
   residual-selected masks, or a requirement that every net bin be positive. Background
   corrections require control evidence, not improved peak appearance.
5. **Keep unsupported regions explicit.** Preserve raw counts and signed diagnostic
   displays, marking extrapolation or inadequate angular support. Do not silently drop
   those observations, change likelihood flags or claim validated subtraction. Any
   fit relying on them remains conditional on the stated background assumptions; a
   support change needs a separately declared comparison. Restrict empirical spread
   bands to their supported regions. Elsewhere state uncertainty is unresolved; do not
   copy an outer-region band inward or inflate errors until the physical model agrees.
6. **Bind the figure to the checked state.** Save the check outcomes in the existing
   diagnostic manifest with control/support and background identities. Within each
   bin, data, background and physical signal use identical native weights and area.
   Preserve the fit memberships and record each display operator, including diagnostic
   extensions. Keep background resolution and saved physical scale; plotting never
   refits. Report background predictive adequacy separately from physical
   preservation gates. Better subtraction can expose a worse physical residual, and a
   rejected physical candidate must not change the accepted state or fitted-bin labels.

## Measurements and the current error bars

Let `c` be detector-native raw intensity and `W` the fractional pixel-membership
matrix. For each fitted or displayed bin:

\[
Y=Wc,\qquad A=W\mathbf{1},\qquad
d_i=(Y_i-B_i)/A_i,\qquad m_i=S_i/A_i.
\]

`B` is background mass and `S` is scaled physical signal mass under the same
operator. Fit in the declared count measure; divide by area only for the displayed
counts per native pixel. Signed negative measurements remain visible. Invalid or
zero-area bins remain gaps.

The current imaging-plate count proxy is

\[
C_{\mathrm{count}}=W\,\operatorname{diag}(\max(c,1))\,W^\mathsf{T},
\qquad e_i=\sqrt{(C_{\mathrm{count}})_{ii}}/A_i.
\]

The one-count floor is explicit. Preserve full covariance for overlapping
observations; adjacent bins and profiles can share pixels. Where matched dark
subtraction is declared, propagate its variance and shared-exposure covariance
once, following [RESULT_MEASURE.md](RESULT_MEASURE.md).

These whiskers are **conditional count proxies in imaging-plate units**. They are
not calibrated total error bars: gain/readout response, estimated background,
source/geometry and physical-parameter uncertainty are not all included. Do not
interpret the fraction of overlapping points inside them as independent trials,
or enlarge them until the model agrees. Agreement within a defensible total
uncertainty remains the goal; the current bars alone do not establish that result.

### Blue empirical background-spread band

The October 8 figure uses current-fit audit-control residual densities
`r = (Y-S-B)/A`, with their 16th and 84th percentiles `q16` and `q84`. The band is

\[
[d_i-\max(q84,0),\ d_i-\min(q16,0)].
\]

This is descriptive residual spread, not a fitted-background covariance,
confidence interval or model posterior. Extending one global range into the
low-angle sector does not calibrate local uncertainty there.

The audited figure used 71 controls: `q16=-0.344470139`, `q84=0.327240118`,
MAE `0.488833213` counts/pixel. Three audit rows overlap training pixels; one
selection-control row shares all its native pixels with training, despite its
held-out row flag. Excluding the three audit rows without refitting
leaves 68 controls, `q16=-0.377657967`, `q84=0.309749234`, MAE `0.450763068`.
This small band sensitivity does not repair physical residuals. Preserve the
historical figure; a future corrected version must identify its changed control
support and use disjoint native supports for genuinely reserved checks.

## Manuscript figure recipe

Preserve the October 8 newer-5-degree figure style:

| Element | Convention |
|---|---|
| Layout | White 14 by 16 inch canvas, 11-point base text; two image panels, three paired family rows, one full-width 00L panel. Export PNG and PDF. |
| Images | Left: signed data minus fitted background. Right: physical signal alone. No fitting-mask cutouts in the displayed images. |
| Image measure | Mean counts per native pixel: 12 by 12 cell mass divided by 144. |
| Extent | Shared upper-detector crop, columns 0–2999 and rows 0–1715 (143 by 250 cells), not the whole 3000-square detector. |
| Orientation | Top-origin native rows and horizontal native columns. Never add another rotation or flip after OSC ingestion. |
| Colors | Shared `viridis`, `AsinhNorm(linear_width=3)`; lower limit `min(-5, net_display.min())`, upper limit `max(5, percentile(raw_display,99.7))`. |
| Profile labels | Simple `r = 1`, `r = 3`, `r = 4` left/right branches against L; 00L against 2theta in degrees. Avoid redundant letter labels and repeated shared axis labels. Family labels do not replace individual rod identities. |
| Profile y scale | Signed `symlog`, `linthresh=1`, `linscale=0.8`, with a zero line. Each left/right family pair shares the union of its y limits. |
| Measurements | Small open circles, gray capped count-proxy whiskers; no data smoothing. |
| Model | Rust-colored line through the evaluated physical prediction; no independent model error points or plotting-only convolution. |
| Background spread | Transparent blue empirical range defined above, identified separately from the count whiskers. |
| Fit support | Match family-colored profile ticks to outlines on both detector panels. Derive outlines from positive native pixel memberships of fitted profile rows; retain holes and disconnected regions. Partial overlap does not mean the entire pixel or bin was fitted. Comparison-only regions are not fitted support. Invalid support remains gaps. |

The current upper-sector 00L display integrates 1–22 degrees in 2theta and
minus 10 to plus 10 degrees in azimuth. Its 210 centers run from 1.05 to 21.95
degrees; the plotted axis spans 0–22. Do not fold it with the lower detector half.
Data, background and model have identical fractional native weights.

The 1–3 degree extension has **zero overlap with the current fitting pixels**.
Label it diagnostic. The saved kinematic/Parratt composite is included once;
displaying a low-angle feature does not establish that it is reflectivity.
Connecting bin values is a graphical line, not additional numerical resolution.
A denser model curve requires a declared corresponding measurement operator.

Retain each diagnostic as one external `.ra_diag.npz` with numeric arrays and
one JSON manifest. Save counts, predictions, background, weights/areas,
covariance ownership, validity, fit overlap, style and source/input hashes needed
to reproduce the figure. PNG/PDF files are presentation deliverables, not new
fit inputs. Never put diagnostic output or experiment callers in the repository.

## Lessons to carry forward

| Evidence | Working conclusion |
|---|---|
| Bi2Te3 inner radial correction: left predicts right MAE 6.747 to 2.087; right predicts left 5.851 to 2.183 counts/pixel. | Earlier local background success. Keep the predictive method; it does not identify the physical background or validate every peak's subtraction. |
| October 8 Bi2Te3 inner plateau: 75.32350 to 57.24841 counts/pixel from disjoint shoulder controls; cross-side MAE fell 95–96%, negative 2.5–7.5-degree bins fell 49/50 to 1/50 without clipping. | Monotonicity did not establish the inner level. Keep local support checks and explicit inward extrapolation. The proposed scale refit failed the 00L preservation gate; the accepted physical state stayed unchanged. |
| Holder screening and restoring omitted r1 fitting support changed which counts were actually fitted. | Keep explicit measurement ownership and visibility. Correct support is necessary even when the resulting residual worsens. |
| Protected monotone background improved several off-peak diagnostics but could increase negative r3 bins or fail unchanged adoption gates. | Keep protection and independent controls; reject automatic global adoption or post-hoc gate changes. |
| October 7 staged Se fit improved a matched objective by 2.284%, while protected 003/006 residuals worsened. Latest family refinement improved only 0.0167656%, with all seven profile RMS values slightly worse. | Staging and a lower scalar score are not a demonstrated physical repair. Report regional tradeoffs and the previous displayed state separately. |
| Latest newer-5-degree displayed model/net areas are approximately 0.98294 for 003 and 0.90812 for 006. The restored 1–3 degree sector is about 8.43854 times the signed measured mass. | Area agreement does not imply shape agreement or low-angle adequacy. Different operators cannot be interchanged to claim improvement. |
| Restricted lateral-slip transition arithmetic passed saved enumeration/conservation checks; fitted fault probability changed with background and support. | Keep the finite transition model and ordered limits; fault fraction, termination and structure remain conditional specimen inferences. |
| Pure lateral registry slips leave ideal 00L phases unchanged; some r3 rods also have registry-invariant phase. Observed strips can mix rods. | Use the invariant families as linked constraints; do not claim disorder is absent from a mixed measured strip solely from its label. |
| PbI2 64/72/80-layer fixed-mean mixture returned zero side population and unchanged objective; earlier fringes changed strongly with sampling. Stronger 00L weights worsened r1. | Neither size spread nor reweighting was a demonstrated general fringe repair. Check numerical sensitivity and coupled preservation before adding parameters. |
| Gemmi atomic and BornAgain slab checks validate selected components. GSAS/MAUD restricted orientation fits had poor held-out predictions. | Component checks and a roughly one-degree fitted core do not identify a full ODF or measured structure factors. Narrow radial windows are not integrated pole figures. |

Do not rank microscopic causes from time spent on an implementation, from an
atomic point ratio versus a detector-window ratio, or from a timeout. Source,
geometry, attenuation, structure, nuisance and detector response remain coupled.
Physical absorption is already included; adding a second factor is not a remedy.
Old complete predictions are useful nominal history but different anchors,
operators and nuisance estimates prevent isolated causal attribution.

## Reproduction evidence

The measured lessons and figure definitions were audited in
`Bi2Se3_recent_decisions_independent_audit_20261008.md` and
`bi2se3_recent_decisions_independent_audit.ra_diag.npz`
(pack SHA256 `49e3169a5c6740ad89ddda11f734a221563398ec4aa16a83c73d2af096d885be`).
These and the following packs are external, under
`C:/Users/Kenpo/.codex/visualizations/2026/10/01/01a0f7e3-277a-7070-9b34-7ded40ce414d/`:

- `bi2se3_newer5_lowtheta_errors.ra_diag.npz`: `presentation_source_utf8` holds
  the actual plotting caller and band calculation.
- `bi2se3_newer5_family_fit.ra_diag.npz`: `family_callers_zip_bytes` holds
  `b5_family_fit.py`, the actual fitted objective and image cell operator.
- `bi2se3_newer5_lowtheta_display.ra_diag.npz`: native weights, area, raw counts,
  physical signal, angle and fitted-pixel overlap for the low-angle display.

The October 8 inner-background repair is retained separately in
`bi2te3_5deg_inner_corrected.ra_diag.npz`
(SHA256 `0bb5d743cc762d23bffb8b37579a8cf9f4d428fb39501f57f0ab8d8d5502d14e`)
and `Bi2Te3_5deg_inner_corrected_fit_report_20261008.md` in the same external directory.
Its controls provide local development evidence, not calibrated whole-image uncertainty.

The current figure is `Bi2Se3_newer5_lowtheta_with_errors_20261008.png`/`.pdf`.
Preserve its frozen evidence rather than replacing it with a visually similar
new calculation. The process above guides subsequent work; it does not authorize
unbounded fitting, new observations or revival of retired campaigns.


## PbI2 background-transfer investigation (2026-10-08)

The GD1 library-polytype caller dropped the preceding fit's two angular broad
background components. Subsequent retained-background figures inherited its
radial-only model. At 00L 30.1 degrees, GD1 raw/background intensities were
139.222/129.196 counts per native pixel; SiD1 values were 51.294/63.100.
The discrepancies have opposite signs and do not support a common offset.
Exact native extraction and the original 50-pixel radial resolution were intact.

A controls-only restoration of the two broad components improved GD1 global
control prediction but did not repair r3 or SiD1. A second, explicitly different
comparison used whole 12-by-12 native cells contained in the original holder-clear
control union, excluding every original profile pixel. It reused matching saved
12-by-12 raw/model means, not subdivisions of coarse control totals. Predicted
nominal sample intensity at most 2 counts/pixel was a declared control screen,
not a bound on true contamination. Training, selection and audit used disjoint
48-pixel spatial blocks; these are development checks on a previously inspected image.

The nonnegative radial-by-azimuth background retained 50-pixel radial knots with
30-degree angular knots and a fixed second-difference penalty. Its audit MAE fell
from 2.8273 to 0.8151 counts/pixel for GD1 and from 1.0223 to 0.4973 for SiD1
on these changed, matched control operators. SiD1's 26-33-degree negative bins
fell from 32/35 to 0/35 without clipping; local audit MAE fell 9.0282 to 1.2671.
Physical parameters, scale, profile operators and count covariance stayed fixed.

Neither candidate was globally adopted. GD1 r3 lacks local controls matched in
radius and holder clearance; much of its angular prediction is regularization
or extrapolation. Its corrected 00L also retains a negative residual. SiD1
r3-right audit MAE worsened 0.9005 to 1.0188, failing the frozen 5% regional gate
despite global improvement. No physical fit or calibrated uncertainty follows.
An older May background exposure is bound in historical inputs, but its transfer
to these June acquisitions is unverified; it was not subtracted by assumption.

Evidence and conditional figures are in the external
`2026/10/09/01a11e7a-43d2-7b23-a889-d369a898d3a1` visualization directory,
consolidated as `pbi2_background_investigation.ra_diag.npz`. It retains both
finite attempts, native support, coefficient leverage, local block counts,
unchanged physical state, source hashes and exact temporary caller bytes.
The prior figures remain immutable. Do not present these candidates as a
resolved whole-detector subtraction or resume their fitting stages implicitly.

## Simultaneous native Bragg regions and profiles (2026-10-09)

For a new combined fit, use this sequence:

1. Start from the matching specimen/acquisition's saved best physical vector and
   its separately saved integer repeat count `N`, then
   retain its qualification. Freeze the original Bragg-region identities and
   training/selection/audit controls before assigning any extra profile pixels.
2. Partition native support as described below, keeping Bragg regions and controls
   protected. Add only unused profile support to the likelihood. Recompute raw
   counts, areas and full covariance from those exact memberships; do not concatenate
   overlapping peak and profile measurements as independent observations.
3. Bind one physical predictor to the combined projection and one declared shared
   scale/background model. Evaluate Bragg and additional profile rows at the same
   physical candidate during each objective evaluation. With linear background
   profiling, use `NativeLinearBackgroundProblem` through the Python fitting API;
   retain raw counts and the required background checks. There is no CLI switch
   that constructs this combined observation or the study's 41-knot design.
4. For fast repeated fits on frozen geometry, explicitly select the
   [fixed-importance response](NATIVE_REFINEMENT.md#explicit-fast-combined-region-fitting).
   It evaluates the same combined memberships, with fresh candidate strengths and
   mosaic factors. Its region compilation uses CPU; native raster deposition retains
   the automatic selector. Hardware selection does
   not change memberships, covariance, weights, quadrature or nuisance definitions.
   Complete the starting prediction and numerical/preservation checks before a
   bounded optimization, keeping 003 and 006 checks separate for Bi2Se3.
5. Compare the saved starting and returned states on both Bragg-region shapes and
   profile core/flank/tail support, with independent selection/audit accounting.
   Detector outlines above a plot must show the native support actually used below;
   whole display profiles remain distinct from partial-bin likelihood additions.

The earlier attempts below retain their historical failure states. The latest
restored-response continuation is recorded after them; do not confuse a previous
incomplete prediction with the current implementation or its qualification.

Use `partition_native_region_support` to assign each positive-membership native
pixel to the first declared observation group. Preserve fractional weights and
full within-group covariance; rebuild raw counts, areas and validity after the
partition. Additional profiles may contribute only their unassigned pixels.
Reserve training, selection and audit controls before adding profile support.
Whole original display profiles may remain diagnostic rows, excluded from the
likelihood. Partial-bin additions are not whole new measurements.

`NativeLinearBackgroundProblem` jointly profiles one nonnegative physical scale
and a caller-supplied nonnegative linear background design against raw-count GLS.
For a nonincreasing radial field, multiply the integrated knot basis by the
upper-triangular cumulative-sum matrix; nonnegative coefficients then represent
successive density drops. The solver checks normalized NNLS optimality and
reports conditional design rank. This does not establish joint physical
identifiability or background adequacy. Independent backgrounds per peak and
unpropagated double counting are not introduced.

For the newer Bi2Se3 5-degree acquisition, the prepared operator retains 997
previous signal rows (979 image cells and 18 profile bins), adds 464 partial
profile bins, and includes 131 disjoint training controls: 1592 training rows.
Selection and audit retain 39 and 71 nonempty controls. Original indexed Bragg
regions must remain separate even when adjacent masks touch; connected components
merged 18 original regions into 13 and are not suitable preservation groups.
Freeze separate 003/006 checks before any continuation. Audit controls must not
select candidates; their previously inspected-image status remains explicit.

That initial attempt produced no new fit. Starting from the saved family-fit
parameters, the resolved specular rule exceeded its 4194304-node limit. The supported nominal
importance rule failed inverse-CDF qualification. Its width-only termination was
inconsistent with the caller's unchanged 2e-15 CDF threshold; importance inversion
now requires that threshold explicitly. A retry still could not resolve a
quantile: residual -2.1094237467877974e-15 within an angular bracket of
8.881784197001252e-16 radians. The error stays explicit, with no clipping or
relaxed tolerance. Analytic scalar checks establish the stopping-rule repair,
not detector integration accuracy.

All three physical predictions failed before a complete baseline; no optimizer
was entered, no candidate objective exists, and the previous nominal fit remains
unchanged and physically unadopted. The finite attempt is closed. A continuation
requires a concrete numerical repair, a supervised caller binding inputs and the
complete numerical rule, failed-attempt accounting, and fixed preservation and
accuracy gates. Do not automatically increase numerical budgets or reuse the
unexecuted archived optimization branch as a qualified fit workflow.

The external `bi2se3_newer5_hybrid_attempt.ra_diag.npz` in the October 9
`01a11e7a-43d2-7b23-a889-d369a898d3a1` visualization directory retains the prepared
operator, covariance, original region identities, unchanged state, source bytes,
checks and failure manifest. `Bi2Se3_newer5_hybrid_support.png`/`.pdf` show the
previous nominal curves and exact detector support; they are not a new fit.


### October 9 numerical repair continuation

The authorized continuation kept the expanded observation operator and the saved
family-fit state. The first divergent stage was angular inverse-CDF evaluation
(`CORRECTED`, analytic 90-digit wrapped-Cauchy oracle). Near 2-pi, adjacent float64
angles can both miss the unchanged 2e-15 CDF tolerance: one actual target had
nearest-bracket residual magnitudes 2.8194e-15 and 2.9888e-15. A compensated signed
periodic chart resolves that target without changing its original arc mass or
stratum. The two captured failures have repaired oracle residuals below 1.09e-17.
This is a scalar numerical correction, not detector or physical-fit validation.

Compiled resolved-panel preparation and 16-node streaming preserve global masses
and the cumulative work limit. On the captured first axial coordinate repeated
17 times (25,296 angular nodes), all panel and emitted-node arrays matched
bitwise across Python/compiled, sliced/unsliced and two output batch sizes. A
one-node-short global limit rejected the calculation. The full captured resolved
rule still exceeds its declared limit; no limit was relaxed.

The full nominal (4096 by 32) and finer (8192 by 64, strength order 6) starting
predictions were each partitioned over their four original source rows. All eight
CPU workers remained unfinished at the declared 900-second check limit and were
terminated. An earlier superseded serial comparison was also stopped incomplete.
No complete baseline, optimizer evaluation, candidate objective or new fit exists.
The previous nominal fit remains unchanged and physically unadopted.

A separately capped 88-second source-zero profile sampled Gaussian native-pixel
deposition repeatedly from 30 through 80 seconds, following initial compilation.
This identifies a concrete acceleration target; it is not a complete-work runtime
benchmark. The explicit user-requested GPU branch is a separate implementation
and validation task, not an accepted replacement for this failed CPU run.

A final bounded CPU continuation omitted only prediction rows unused by the
likelihood and every frozen gate. All 2,274 required rows, their 4,257,757 native
memberships, and 2,613,949 pixels were preserved. The omitted support included 64
invalid diagnostic 00L bins at 2.35--8.65 degrees. All eight nominal/finer source
jobs were still incomplete at the 15:18 UTC cutoff; no source result or objective
was obtained. Their completion mask is retained. A live stack sample still found
CPU local-m0 deposition, while the separate hybrid GPU workers had advanced to
CPU regular-rod deposition. These samples locate work, not timing fractions.

The external `bi2se3_newer5_joint_repair.ra_diag.npz` retains the prepared data,
unchanged initial state, numerical proofs, bounded attempts, input/source hashes,
and caller bytes. Any continuation must bind a new finite compute budget and
complete the frozen baseline, preservation and numerical checks before fitting
or adopting a candidate. Partial source contributions are never complete fits.


### October 9 restored fast-response continuation

The restored explicit `fixed_importance.v1` rule uses strength-independent nodes
and sparse native-region probabilities. It shares the repaired periodic inverse
CDF, conditional geometry, event-mass and spatial-probability owners. The adaptive
pixel-error estimator remains the default; its qualifications are not transferred
to this nominal option. Geometry/source/optics/proposal changes invalidate the
response, while candidate strengths, mosaic, attenuation and stitching are
contracted again. Region compilation uses CPU; automatic CPU/CUDA raster execution
remains separate. Forced CUDA region compilation rejects explicitly.

For the frozen newer 5-degree Bi2Se3 combined projection above, a complete nominal
4096-by-32 response compiled in 385.99 seconds. An unchanged contraction took
7.24 seconds and a changed atomic coordinate took 7.18 seconds without recompilation.
The retained response is about 459 MB, including grid/node/CSR arrays; this is not
a process-memory bound. The preceding adaptive attempt had no complete prediction
after 3806.47 seconds, so this comparison establishes restored usable latency,
not a speed ratio between equivalent estimators.

Eight actual same-node events agreed between sparse region contraction and the
canonical native raster to maximum absolute mass error 2.04e-15. Eighteen
reused-versus-fresh first-batch checks across the nine active coordinates, for
regular and local-m0 support, had maximum relative L1 error 1.36e-16. These are
response implementation checks, not whole-image convergence proofs. Explicit
seed/proposal/spatial changes invalidate reuse, forced CUDA and insufficient
retained-memory budgets reject, and inactive numerical-refinement controls reject.
Ruff, the wheel build and native refinement CLI import checks passed; no retained
assessment harness was added.

The bounded simultaneous fit keeps N=13, all 85 signed rods, all four original
source rows, 1592 likelihood rows and 2274 required prediction rows. It profiles
one global scale and 41 shared nonincreasing radial background knots against the
full raw-count covariance, without historical 10-times 00L weighting. The starting
state passed all 28 declared regional nominal-versus-finer screens using
8192-by-64 rules: 18 original Bragg regions, seven full diagnostic profiles,
003, 006 and additional profile support. These screens use the fixed nominal
scale and the declared 5% signal plus one-count-per-native-pixel allowance; they
are nominal checks, not source convergence or physical qualification.

The supervised run completed 161 fresh candidate predictions and 163 scoring
callbacks before its 1800-second optimization-stage deadline interrupted a
finite-difference Jacobian. Total run time was 3748.09 seconds, including the
1487.55-second finer-response and 396.03-second nominal-response compilations.
Maximum sampled RSS was 3.11 GiB; it is not a continuously measured process peak.
The last full trial lowered the combined objective from 243064.3682 to
241525.5335 (0.6331%). All trial preservation checks passed, including separate
003/006 checks. The solver did not return a converged endpoint. No final candidate
numerical or audit check ran, and no candidate was selected or physically adopted.
The last full trial, not its slightly lower finite-difference probe, is shown in
the comparison figure. This remains a bounded incomplete optimization, even though
the complete fast prediction and repeated candidate evaluation are restored.

The 003 background still relies on inner radial hats without training-control
support; 006 still lacks matching-radius held-out controls. The source rule is
nominal, the same-image audit was previously inspected, and the physical parameters
are not identified. Signed counts and negative-run diagnostics remain unmodified;
figure bars are conditional imaging-plate count proxies, not calibrated uncertainty.
The detector overlays show exactly the 482 fitted-profile rows below them. The
979 native Bragg cells also remain in the joint objective, but are not displayed
as those profile rows. Read the regional checks separately.

All evidence is external in the October 9 visualization directory named above:
`bi2se3_restored_fixed_hybrid.ra_diag.npz`, `restored_response_checks.ra_diag.npz`,
`bi2se3_restored_combined_fit.ra_diag.npz`, and
`bi2se3_restored_signed_report.ra_diag.npz`. The last pack binds the closed run,
contains the hash-verified execution source and exact report caller, and accompanies
`Bi2Se3_restored_combined_profiles.png`. Runtime sources differ from the final
commit only by later type/docstring changes and the separately checked admission
of fixed-rule numerical controls. Historical failed attempts are preserved.
A future optimization must budget complete objective/Jacobian work rather than
only the measured raw contraction; resume from a declared saved trial with fresh
endpoint qualification, not from a silently promoted finite-difference probe.


### October 9 safeguarded continuation

The next authorized stage resumes the last complete trial above, not its lower
finite-difference probe. Its seven active coordinates are the Bi and outer-Se
fractional positions, Gaussian width, Lorentzian width and probability, surface
fraction, and extra film thickness. Top/bottom roughness are fixed for this stage
because all 32 preceding local roughness probes produced identical raw predictions;
this does not establish global insensitivity or identify those parameters.

Declare derivative steps separately from search bounds. This stage uses half of
1e-4 times each declared physical sensitivity scale, converted to normalized bound
units. At the resumed seed, all seven full-step/half-step profiled-residual slope
checks passed the predeclared 5% relative L2 limit; the largest change was 0.0420%.
This is local derivative stability, not refined-rule gradient qualification.
TRF retains its existing sensitivity scaling and 1e-6 stopping tolerances. Its
iteration callback can return an accepted but unconverged endpoint before a caller
deadline; see [bounded continuation controls](NATIVE_REFINEMENT.md#parameter-steps-and-iteration-boundary-stopping).
Do not promote an evaluation or derivative probe over the returned endpoint.

The frozen observation, N=13, source/rod roster, covariance and shared background
are unchanged. Every preservation threshold remains anchored to the original
243064.3682 combined-objective baseline, never reset to the resumed trial.
The finite contract is 10800 seconds overall, 7200 seconds for optimization,
80 full solver evaluations and 680 fresh predictions, with a complete derivative
batch reserved before stopping. A returned endpoint must pass the original
regional, separate 003/006, selection and final audit checks, all 28 numerical
screens, and nominal/finer improvement. The signed completion report additionally
requires finer-rule non-regression against the resumed seed.

Compile the finer response once and retain it for both seed and endpoint checks.
Checkpoint only changing arrays plus exact caller/runtime provenance; bind the
immutable prepared observation through its verified parent-pack hash. Recompressing
all prepared memberships/covariance during every checkpoint is unnecessary work.
The first continuation checkpoint took 0.017 seconds and held about 1.3 MB of
uncompressed changing data, compared with the preceding 242 MB whole-pack payload.


This continuation returned normally: four full evaluations/four Jacobians and
three accepted iterations took 251.89 seconds. TRF stopped on its unchanged ftol
condition, with optimizer_status=2 and minimum_resolved=true. The objective fell
from the resumed 241525.5335 to 241365.5419, or 0.06624% further improvement;
relative to the original 243064.3682 baseline the improvement is 0.69892%.
All 28 seed and 28 endpoint nominal/finer screens passed. The finer objective
fell from 243006.9896 originally and 241624.7347 at the resumed seed to
241493.3347 at the endpoint, passing both improvement comparisons.

The candidate was nevertheless not selected: the protected 00L detector region
(original Bragg_0, 498 cells) has a 1.05267375 RMS ratio against baseline,
exceeding the frozen 1.05 limit (8.0285381 versus an 8.0081460 RMS ceiling).
Its 00L identity is retained in the September 30 diagonal-background input.
This region is distinct from the separately checked 003/006 profile windows.
All other protected Bragg/profile RMS ratios, separate 003/006 checks and
selection/audit checks passed. Audit MAE changed from 0.4753871 to 0.4793336,
within the original 5% allowance. This is an explicit tradeoff in the combined
objective, not a failed optimization or numerical crash. A total-score optimum
need not satisfy separately checked regional inequalities; this TRF stage checks
those inequalities for admission rather than enforcing them during each step.
Do not relax the guard, substitute a derivative probe or label this candidate
selected. Any further constrained stage needs its own declared bounded caller
and final checks against the same original baseline.

Nominal/finer compilation took 389.97/1466.46 seconds; total elapsed time was
2300.77 seconds (38.35 minutes), including setup, 14 derivative-screen probes
and final checks. There were 38 fresh predictions including the prechecks.
This is a warm-start continuation with changed steps/active set, not an
isolated speed benchmark against the earlier run. The extra film thickness
reached its 500-angstrom search boundary. N remains 13; physical adoption and
parameter identification remain unavailable. All preceding background/source
qualifications, signed-data requirements and same-image audit limitations apply.

Evidence is retained externally as
`bi2se3_continuation_search_checks.ra_diag.npz`,
`bi2se3_continued_combined_fit.ra_diag.npz`, and
`bi2se3_continued_signed_report.ra_diag.npz` in the October 9 directory above.
The report preserves exact code/caller provenance and the unselected endpoint.
`Bi2Se3_continued_combined_profiles.png` compares original, resumed and returned
curves on exactly the highlighted detector support. Software checks cover scalar
step compatibility, vector steps, public iteration stopping, early rejection,
formatting/lint, wheel construction and CLI import; they do not qualify the
physical model. No assessment harness is retained in the repository.
