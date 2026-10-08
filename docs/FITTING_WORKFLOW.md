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

Freeze spatial selection and audit supports at the native-pixel level before
choosing a correction. Row-held-out bins can still share training pixels.
Preserve cross-covariance or use disjoint support; label repeatedly inspected
same-image controls as development evidence. Stronger evidence is prediction of
reserved spatial regions or a separately bound acquisition.

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
| Fit support | Green ticks mean any overlap with fitting pixels; partial overlap does not mean the entire bin was fitted. Invalid support remains gaps. |

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
| Bi2Te3 inner radial correction: left predicts right MAE 6.747 to 2.087; right predicts left 5.851 to 2.183 counts/pixel. | Strongest measured local background success. Keep the predictive method; it does not identify the physical background or validate every peak's subtraction. |
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

The current figure is `Bi2Se3_newer5_lowtheta_with_errors_20261008.png`/`.pdf`.
Preserve its frozen evidence rather than replacing it with a visually similar
new calculation. The process above guides subsequent work; it does not authorize
unbounded fitting, new observations or revival of retired campaigns.
