# Physical detector-native refinement

The fitting path binds physical candidates into the same finite-structure,
optical transport, spherical mosaic and native-pixel integration used by the
renderer. A fit predicts the frozen native observations; moving geometry or a
lattice never moves measured pixels into another fitting region.

## Admitted parameters

| Model | Continuous specimen coordinates | With acquisition coordinates |
|---|---:|---:|
| Bi2Te3 / Bi2Se3 | 21 | 39 |
| GD1 / SiD1 Pb recipes | 21 | 39 |
| Clean1 Pb recipe | 27 | 45 |
| B4 Pb recipe | 25 | 43 |

`BiNativeStructureModel` exposes a,c, two symmetry-preserving internal z values,
three orbit occupancies, and six independent radial/normal site ADPs. The joint
model adds sigma/gamma/eta mosaic, two termination-simplex coordinates, extra
film thickness and two interface roughnesses. Bi repeats are conventional CIF
cells. The three finite termination windows retain their original integer lifts.
The empirical Parratt/kinematic composite remains a named model assumption.

`PbNativeStructureModel` exposes a,c, the signed iodine height, Pb/I occupancies,
and four orbit ADPs. `PbJointModel` adds mosaic, surface fractions, extra film
thickness, initial plus orientation, every outer phase fraction, an independent
epsilon per phase, and all parent mixture shares. B4's two handed 6H phases can
have distinct epsilon values even when their archived seed shared one value.
Transition-law mixtures are averaged before the recurrence; independent parents
and phases are averaged as intensities. Pb repeats are individual c-axis layers.
Pb has no reflectivity/roughness fit in this path.

Both models enforce `film_thickness_A = N*c_A + extra_film_thickness_A`, with
nonnegative extra thickness and positive integer N. The selected finite-total
or per-layer normalization is preserved. Fixed-N normalization changes can be
absorbed by exposure scale; size-population fractions cannot be reinterpreted
that way.

`NativeInstrumentModel` supplies 18 acquisition coordinates:

- Native detector centre column/row corrections, distance along the nominal
  detector normal, and active intrinsic local-x/current-local-y tilts.
- Active intrinsic sample tilts and translation along the nominal sample normal.
- Two spatial sigmas, two divergence sigmas, and two position/divergence
  correlations at the declared fixed source plane.
- Two spectral line centres, their relative probability, and common Gaussian
  intraline wavelength width.

The source plane, direction and transverse axes stay fixed. A separate incidence
offset would duplicate a sample-tilt coordinate. Detector roll, goniometer-axis
and pivot corrections require different acquisition/calibration information.
Every changed source is resampled through the authoritative conditional source
sampler, with explicit numerical rule and exact line masses. Material optics are
rebuilt at the actual candidate wavelengths from the occupied candidate cell.

Units are supplied by `NativeJointEvaluator.parameter_units`; parameter ordering
comes from `parameter_names`. Specimen coordinates belong to
`specimen:<sample_id>`. Acquisition coordinates belong to the SHA256 identity of
the frozen raw acquisition. Matching starting values do not justify sharing
parameters across acquisitions.

## Search, profiles and evidence

`fit_native_parameters` uses full count-plus-background covariance once and
profiles one nonnegative intensity scale exactly. It accepts multiple starts,
explicitly fixed initialization coordinates and separately declared calibration
blocks. `refit_native_choices` refits every continuous coordinate for each N or
other explicit discrete choice. It never rounds a continuous N.

Bounds are declared search ranges, not confidence intervals. Each bound endpoint
has its own `physical` or `search` classification. `sensitivity_scale` is a
separate physical scale for comparing Jacobian columns. A final joint stage must
release every admitted coordinate. Temporary fixing is initialization only.

Results distinguish best evaluated, historical-guard feasible, and optimizer
converged candidates. A converged start does not resolve a minimum if another
unfinished start found a lower objective. Optimizer success does not establish
numerical accuracy, predictive adequacy, parameter identification or global
optimality.

`profile_native_parameter` fixes one coordinate and refits all remaining
coordinates and intensity scale at every discrete alternative. Both sweep
directions use neighbor starts plus the declared multistarts. Failed or lower
unfinished minima remain unresolved. The output contains raw objective curves;
there is no automatic chi-square threshold or confidence-interval claim at
mixture boundaries, under discrete alternatives, or with historical guards.

`GaussianCalibration` requires an artifact SHA256, target acquisition ownership,
coordinate names/units/owners, mean and full positive-definite covariance. Use
one joint block per artifact/acquisition; overlapping or reordered copies cannot
be counted again. Independent measurements and assumptions contribute separately
reported objective terms. Ring residuals, search widths, and robust background
regression weights are not calibration parameter covariances.

The present acquisitions have no independently bound parameter covariance in
their full-fit declarations. Available direct-beam offset images constrain an
effective beam/resolution description at a declared source plane. They do not
separate intrinsic beam moments from detector blur or establish an absolute
source-plane position. Existing hBN ring evidence does not identify detector
roll about the beam. Spectral shape requires same-optics evidence. No priors are
manufactured to fill these gaps.

## Validation and background controls

`training_observations` creates a new objective revision with the training
marginal covariance. It cannot reactivate excluded rows. Historical guards stay
available for diagnostics and are forbidden as training constraints: they use
the already explored full acquisition. The acquisition plans declare complete
held-out branch groups before the new fit. This is prospective validation within
an explored dataset, not an untouched experimental test.

`conditional_validation` retains cross-row covariance:

`mean_V_given_T = model_V + C_VT solve(C_TT, data_T - model_T)`

`C_V_given_T = C_VV - C_VT solve(C_TT, C_TV)`

The scale and physical prediction must come from training alone. This conditional
noise score excludes fitted-parameter uncertainty, which needs profiles or a
separately defined bootstrap. Validation counts must not select a starting point,
N or a model and then be reported as a held-out score of that same selection.

`load_native_background_controls` retains the frozen control supports. Its
signal diagnostic applies the signal-fit scale to predicted diffraction on those
controls. Statistical denominators use measurement variance, not the archived
robust-regression calibration weights. The 48- and 80-pixel layouts overlap each
other and are reported separately. Without a predeclared contamination budget,
the result is diagnostic only. A failed budget leaves fixed-background validity
unresolved; it cannot silently change net counts or their covariance. In
particular, Te's exported incumbent control-background array describes an earlier
background field and must not be treated as the selected field.

## Numerical qualification and execution

Run `scripts/refine_native.py --physics PHYSICS.json --observations OBSERVATIONS.json
--plan PLAN.json --output EXTERNAL.ra_diag.npz` from a configured environment.
The plan schema is `rasim-native-refinement-plan-v1`. The external diagnostic
contains numeric arrays and one embedded JSON manifest, atomically checkpointed.
Task 30 records the six concrete acquisition-bound plan and result locations.

Required plan fields are `parameters` (serialized `FitParameter` records),
`starts`, `repeat_choices`, `proposal_mosaic`, `acquisition_id`, `fit_instrument`,
`workers`, `finite_difference_step` and `schema`. Stages declare their name,
active parameter names, iteration budget and historical-guard mode. Optional
blocks declare numerical checks/probes/tolerances, calibration, prospective
training indices/groups, sensitivity, profiles, and real-forward synthetic data.
An empty stage list executes baseline/numerical diagnostics only.

Each validation group is `{"name": "branch:m3_plus", "indices": [2, 7, 9]}`
with the actual frozen row indices. Groups must partition all valid rows outside
training, without overlap. The result retains the joint conditional score plus
correlated marginal group scores. Profiles record their own guard policy and
observation revision. Synthetic noise is drawn only on the supported covariance.

Check axial, outgoing-angle, inner-cone, spatial and source integration separately,
then independent seed and a combined stricter rule. Compare both absolute
predictions at one fixed physical scale and parameter-induced prediction changes
in the full covariance measure. Profiled objective contrasts and individual guard
decisions/scale feasibility are separate checks. These are empirical comparisons,
not certified integration error bounds. Larger N and narrow candidate widths need
their own qualification. Numerical agreement at initial probes does not certify
a fitted candidate or a profile interval.

The runner repeats these checks around every fitted N candidate against its actual
training target, then compares the numerical center-score offsets across N. The
overall numerical status cannot inherit a passing initial status when fitted or
discrete-ranking checks fail. A source refinement must change effective sampled
rays, masses or conditional position, rather than only a provenance label.

`stitch_grid_size` refines the empirical Bi handoff calculation. Baseline, numerical
probes, reported fits, local-sensitivity probes and profile candidates record the
actual surface/wavelength interval selection, overlap scale and zero-strength
normalization. Pb's non-reflectivity model returns no handoff records. A switch in
the empirical handoff remains a potential source of nonsmooth parameter response.

The explicit response cache retains at most two geometry responses and 64 small
prediction vectors. Source, material, reciprocal basis or rigid instrument
changes invalidate geometry reuse. Atomic z/ADPs, stacking, population weights,
thickness and roughness still recompute their physical intensity. Gaussian and
Lorentzian cone components reuse only their own unchanged width/order within one
response owner. Exact pole identities and float64-underflow shortcuts preserve
the unoptimized mosaic equation.

## Unresolved physical freedoms

Common occupancy and exposure scale form an exact gauge when optics are fixed;
composition-derived optics can break it weakly, but do not guarantee measurable
vacancy fractions. Ideal m0 has no radial-ADP sensitivity. Eta=0 makes the unused
Lorentzian width inactive, zero phases hide their private parameters, and an
empty simplex remainder hides later shares. Initial orientation can be equivalent
to handed-population changes in family-summed Pb data. These require explicit
weak-direction/profile reporting, not artificial small error bars.

`NativeJointEvaluator.inactive_parameters` reports exact zero-mixture/remainder
cases at each reported candidate. Coordinates remain free because later joint
steps can make them active. A parameter absent from that report may still be weak
or unidentifiable; the local SVD and profiles are separate diagnostics.

The model does not yet define a calibrated detector PSF, finite footprint in the
conditional source backend, lateral coherence shape, domain-size distribution,
microstrain ensemble, or full off-specular DWBA. ADPs do not produce the missing
diffuse intensity of those effects. Select an extension only after resolving
numerical residuals and establishing its operator, measure, calibration evidence
and independent proof. No generic extra blur or background term is added merely
to reduce the objective.
