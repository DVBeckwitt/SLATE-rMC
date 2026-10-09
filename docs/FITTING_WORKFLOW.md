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

No new fit completed. Starting from the saved family-fit parameters, the current
resolved specular rule exceeded its 4194304-node limit. The supported nominal
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
