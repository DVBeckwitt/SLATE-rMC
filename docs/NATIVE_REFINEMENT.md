# Physical detector-native refinement

The fitting path binds physical candidates into the same finite-structure,
optical transport, spherical mosaic and native-pixel integration used by the
renderer. A fit predicts the frozen native observations; moving geometry or a
lattice never moves measured pixels into another fitting region.

## Shared material boundary

`NativeRefinementModel` supplies ordered parameter names/units, `bind(values, N)`
and exact mixture-boundary inactivity. Binding returns the typed physical inputs,
finite-structure arguments, physical mosaic and optional specular stack. Built-in
`BiJointModel` and `PbJointModel` own their symmetry and population rules.
`make_native_evaluator(..., model=...)` and `native_prediction_group(..., model=...)`
accept another explicit model; the command-line JSON assembly currently covers
the built-in Bi/Pb recipes. A new material still needs a valid scientific binding
and numerical qualification. It does not need another optimizer or detector kernel.

## Portable inputs and saved images

`configs/native_experiments.json` lists the six frozen acquisition descriptors
and physical-input hashes relative to the declared external input root.
Adopt an existing calibrated experiment with:

```powershell
uv run --frozen python scripts/prepare_native.py `
  --observations C:\external\original\sample_observations.json `
  --output-directory C:\external\prepared
```

Optional `--raw` and `--dark` resolve relocated files; their original SHA256 must
match. OSC conversion happens once at the existing I/O boundary. Decoded NPZ inputs
already contain detector-native counts. Preparation verifies frozen raw projection,
count/dark covariance, net/background arrays and control support, then copies
unchanged payloads and publishes the descriptor last. The descriptor references
hash-prefixed local filenames. This is frozen-calibration adoption, not a new
background estimator or a substitute for validating a new acquisition's calibration.

Recover the catalogued September 11 Bi2Te3 baseline with one command:

```powershell
uv run --frozen python scripts/prepare_native.py `
  --sample bi2te3 --input-root C:\external\september11 `
  --with-baseline --output-directory C:\external\prepared-baseline
```

The input root contains `native_fit_inputs_20260911/` and the archived fit/figure files
named in the catalog. Catalog paths must stay inside that root and their hashes must
match. Existing `--raw`/`--dark` overrides still support relocated acquisition files.
`--sample` works for all six catalog entries; `--with-baseline` currently binds only
Bi2Te3. Without that option, preparation adopts observations and physics only.

The prepared observation descriptor's `archived_baseline` field links the unchanged
parameter JSON, already-scaled `selected` count vector and original profile/detector
figures. Open its `figures` PNG/PDF paths directly. Preparation checks the historical
objective and guards through the existing observation owner, without refitting the
scale or evaluating physics. Its separately reported GLS score is a diagnostic; the
historical result did not optimize that objective. The original JSON retains the
source rule, structural parameters, scale and nominal-acceptance limitations.

This is saved-result recovery, not a new optimizer result or numerical qualification.
Historical paths embedded inside unchanged archives remain provenance, not relocated
runtime dependencies. Only the references in the prepared descriptor are relocated.
To prepare another baseline copy, repeat the catalog command against its original input
root; baseline-bearing prepared descriptors are rejected as preparation sources.
The baseline is not a v2 refinement result for `render_native.py`; that command
recomputes a modern saved fit. A subsequent refit still requires an explicit plan and
objective. No plan or optimizer-success record is manufactured during preparation.

Render a saved fit with the same physical and observation inputs:

```powershell
uv run --frozen python scripts/render_native.py `
  --physics C:\external\prepared\PHYSICS.json `
  --observations C:\external\prepared\sample_observations.json `
  --result C:\external\fit.ra_diag.npz --output C:\external\image.ra_diag.npz `
  --full-image
```

Use the actual physics filename referenced by the prepared descriptor. Add
`--candidate` only to render the separately recorded optimizer candidate. The
renderer reproduces its saved region prediction before carrying its fit status.
Physical mosaic, roughness, scale and numerical proposal remain distinct. Full
images integrate the continuous distribution over each native pixel using one
whole-panel proposal and additive batches. `--resume` restarts from the last
consistent atomic checkpoint; `--checkpoint-seconds` defaults to 60. Source files,
dependencies, inputs and candidate identity must match. The image is explicitly
unqualified until its own observable convergence is established. Synthetic results
retain their saved synthetic target and target-kind label.

Historical orchestration was retired at T34. Recover archived source without
executing it using `scripts/restore_native_archive.py --bundle ALL_REFS.bundle
--manifest SATELLITE_MANIFEST.json --worktree OLD_NAME --destination NEW_DIRECTORY`.
The resolver verifies archived bytes, restores the recorded Git tree and dirty
overlay, and records the old-to-new source-root mapping in `RECOVERY.json`.
Original virtual-environment bytes remain archived; recreate environments from
the restored dependency metadata because launchers contain absolute paths.
Immutable examples and scientific result artifacts are not rewritten.

## September 11 G/L continuation

`configs/bi2te3_historical_native.json` is the executable conditional recipe for
that experiment. Run `scripts/refine_native.py --physics PHYSICS.json
--observations bi2te3_observations.json --plan configs/bi2te3_historical_native.json
--output EXTERNAL/fit.ra_diag.npz`, then render with `--candidate` until a qualified
selection exists. The recipe binds both original input hashes, including through
preparation. A corrected-geometry continuation needs a new, explicit binding.

The five free coordinates are Gaussian sigma, Lorentzian HWHM, mixture probability
and two termination shares. N=15, total film thickness=500 A, geometry/source,
atomic sites/ADPs/occupancies and roughness remain fixed. This is a conditional
mosaic/termination result. It cannot identify general atomic structure or disorder.
The starting vector is the delivered result. Current SLSQP and exact scale profiling
are used; this does not reproduce the sequence of historical optimizer iterations.
Original final-structure guards apply without the earlier mosaic stage's 0.01 reserve.
The full historical N sweep is not replayed: that sweep changed extra thickness for
each N to hold total film thickness fixed.

The declared nominal proposal uses 32 conditional source states, p12/g5, cone16,
axial half-width 1/32 L and the September 9 proposal mosaic. The exported physical
file's 1/30 L was based on selected N15, whereas the original proposal used N16.
`film_phase_q_first_source` maps local-m0 proposal centers with the shared phase-Q
function and first source wavelength. For this zero-bandwidth archive that is the
first declared Cu line. For a broadened spectrum it is only the first sampled row.
This alters importance sampling, never the physical density or support. The default
proposal coordinate remains `external_q`. Neither proposal establishes convergence.

An unfinished fit retains its best feasible evaluated candidate (or best evaluated
candidate if no feasible point exists), including its convergence and guard status.
It can be rendered for inspection; numerical selection remains separately gated.
The baseline diagnostic scale is unconstrained. Compare archive replay at the
archived fixed scale, rather than mistaking that diagnostic for the accepted scale.

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

`fit_native_parameters` profiles one nonnegative intensity scale exactly. Plans
select `"objective": "gls"` (the unchanged default, using full count-plus-background
covariance once) or `"objective": "historical"` (the frozen operator and target).
The historical operator requires the complete original observation roster; training
splits and synthetic targets are rejected. Both use the same search and optional
exact guard-feasible scale interval, with no blended loss or duplicated optimizer.
Results report `data_objective`, `objective_kind` and diagnostic `data_chi_square`
separately. Plan bytes, including objective selection, bind raw checkpoint reuse.
Historical loss and guards do not define a noise likelihood or confidence interval. It accepts multiple starts,
explicitly fixed initialization coordinates and separately declared calibration
blocks. `refit_native_choices` refits every continuous coordinate for each N or
other explicit discrete choice. It never rounds a continuous N.

The common runner defaults to public SciPy `least_squares(method="trf")` for
unguarded stages and SLSQP for explicit historical inequalities. `method` may be
declared per stage/profile; TRF rejects historical inequality constraints. Its
`maximum_function_evaluations` bounds public solver evaluations; finite-difference
probes are additional forward work and are counted separately by the raw ledger.
Guarded SLSQP varies one dimensionless nonnegative scale coordinate alongside
the active physical coordinates; raw predictions remain keyed only by the latter.
TRF uses physical sensitivity scales, linear residual loss and bound-aware signed
or shortened differences. It never fixes a weak coordinate merely to obtain rank.

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

`prediction_workers` defaults to 1. Larger values use one persistent isolated
process pool for the run; `prediction_group_size` defaults to 16. Each group owns
an evaluator and reuses exact dependent state across its candidates. Processes
persist, while evaluator caches are bounded to the group. Numerical projection
`workers` is a separate setting; avoid nested oversubscription. Only the parent
profiles scale, computes objectives, records status and writes checkpoints.
Compilation/cache counters in the runner are explicitly parent-only and cannot
be interpreted as total worker cost.

`--resume` restores exact completed raw vectors indexed by full float64 parameter
bytes and integer N. Forward identity includes plan, input hashes, source files,
runner and numerical dependencies. Pending work is regenerated by the public
optimizer. Raw predictions are rescored with the current stage objective and
fixed-coordinate scope; residuals, Jacobians and private optimizer state are not
restored. Lower unfinished candidates remain evidence against a resolved minimum.

Required plan fields are `parameters` (serialized `FitParameter` records),
`starts`, `repeat_choices`, `proposal_mosaic`, `acquisition_id`, `fit_instrument`,
`workers`, `finite_difference_step` and `schema`. Stages declare their name,
active parameter names, iteration budget and historical-guard mode. Optional
blocks declare numerical checks/probes/tolerances, calibration, prospective
training indices/groups, sensitivity, profiles, and real-forward synthetic data.
An empty stage list executes baseline/numerical diagnostics only.

`require_initial_qualification` defaults to true. When fitting or profiles are requested,
a failed initial numerical screen writes an incomplete checkpoint and exits with status 2
before optimization. Explicit false permits a bounded execution exercise; it does not
qualify the resulting predictions. Result schema `rasim-native-refinement-result-v2`
retains an `optimizer_candidate` separately from `selected`.

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

`qualify_profile_candidates` defaults to true. Every nuisance-refitted profile center
and N receives numerical checks, including the objective offsets across the complete
profile curve. Conditional validation also checks numerical predictions with each scale
fitted on training alone. A better admissible profile point for any N invalidates that
N's resolved minimum and requires an all-active joint refit. Failed numerical checks or
unresolved alternatives leave `selected` null, even if an optimizer reports convergence.

Optional `regular_q_bounds_Ainv` and `local_m0_q_bounds_Ainv` freeze conservative proposal
domains in `integration_override`. They must enclose actual source-region bounds for every
candidate and refinement; insufficient coverage raises. This stabilizes axial nodes across
source sample counts without dropping support. `quadrature_kind` defaults to `sobol`;
`composite_gauss` uses axial CDF panels and separately integrates reachable angular arcs.
The latter requires seed zero and independent order refinements. Neither rule is qualified
merely by being deterministic or by passing its analytic quadrature invariant.
For composite quadrature, `maximum_axial_panel_width_Ainv` caps the physical width of
inverse-CDF panels through recursive bisection. Refine the cap independently of the
angular order. This prevents large physical gaps between concentrated proposal peaks;
it does not replace the observable accuracy gate.

Use `angular_support="fixed_union"` to keep the conservative angular domain independent
of individual observation Q boundaries. The default `"q_conditioned_union"` preserves
compatibility. Fixed support removes a demonstrated remeshing discontinuity, but its
angular resolution must still be independently qualified. Observation grouping was an
unsuccessful numerical prototype and is not a supported control.

By default (`angular_integration="native_panels"`), native fitting and rendering
resolve the angular proposal against
their complete requested detector support. Geometry-only tiles (at most 64 pixels
per side, at most 65536 tiles per source/channel) provide corner/center rates of
Gaussian-kernel and pixel motion. Each tile's physical-angle width applies only
inside its original conservative Q/angular envelope. These envelopes refine the
mesh; they do not prune the integration union or omit observations. Whole-panel
rendering uses the same selector with its own complete panel support.
Witness rates are normalized by their physical Ewald-circle radius, then scaled
to each axial node's radius. This avoids applying the motion of a large circle to
a nearly collapsed endpoint circle. At zero radius only the geometry cap is
removed; angular measure and proposal panels remain. Tiles without propagating
witnesses add no geometry cap and still retain their original proposal support.
An entirely degenerate set of propagating witnesses fails explicitly. These
sampled seeds do not replace observable convergence checks.

`angular_resolution_fraction` defaults to 0.5 and controls panel width relative to
the local spatial response scale; halve it for independent angular refinement.
The ordinary angular rule uses GL8 in each proposal-CDF panel, with inverse-density
weights restoring physical `dphi`. `angular_power` supplies
`2**max(0, angular_power-3)` initial panels per arc. Powers below 3 therefore share
the same initial angular rule. The existing Sobol axial grid is retained when
`quadrature_kind="sobol"`; its angular integration is now deterministic conditional
quadrature. Low-level coordinate calls without resolution inputs retain the paired
proposal. The explicit `angular_integration="nominal"` option exposes that same
existing coordinate rule through the shared fitter and renderer, without native
resolution panels. It changes the numerical proposal only, not the source
distribution, supported domain, signed rods, structure, optics or pixel integral.
Here nominal does not mean a nominal source ray. It is an exploratory quadrature,
not a numerical qualification or an exact historical replay unless all inputs match.
Use angular power/order refinement; resolution-fraction changes are ineffective
in this mode and rejected as numerical checks. Explicit angular edges cannot be
combined with nominal mode. Mode identity is bound into cached responses.
Select the nominal route explicitly in a fit plan, for example:

```json
{"integration_override": {"angular_integration": "nominal"}}
```

The saved effective rule is also used by `render_native.py`. Source sample count,
spectral lines, specimen parameters, observations and objective stay independently
declared; this switch does not recreate the September 11 fit by itself. Keep the
existing initial/refined numerical checks and candidate-versus-selected distinction.

In native-panel mode, numerical-check overrides that change power within that
ineffective range are rejected. A frozen support envelope alone does not freeze the adaptive panels
when detector or source geometry changes; qualify the actual parameter contrasts.

Corner/center rates are resolution seeds, not a supremum or numerical certificate.
Refine and qualify the actual observable and parameter contrasts. Fit support is
not full-image evidence, and image qualification remains separate. No universal
image tolerance follows from the local angular regression.

`angular_panel_edges_rad` optionally supplies an immutable increasing physical-angle
partition from exactly `0` to `2*pi`, requiring `composite_gauss`. The sampler merges
the original support first, then intersects each arc with the partition. Each panel
uses `2**angular_power` physical Gauss nodes (use power 3 for GL8), or the declared
local-m0 power where applicable. Weights are `dphi` times the original axial measure;
the angular proposal CDF/PDF is bypassed, not multiplied in again. No supported arc,
signed rod, source mass, observation or physical factor is removed. Omitting edges
uses automatic native resolution. Explicit edges override that selector for
reproducibility; refine their panel widths and per-panel order explicitly.

`seed_angular_panel_edges(bounds, maximum_panel_width_rad=..., maximum_panels=...)`
constructs candidate edges from complete native source/region bounds, retaining
internal boundaries even where support intervals overlap. Supply all required
observation/source/probe bounds, not only a selected parent or training rows. It caps
physical widths inside their angular union and retains the outside gaps as panels.
This is geometry seeding, not a kernel-adaptive selector or accuracy certificate.
The supplied width must pass full-observable numerical checks; a fine angular rule
does not qualify axial/source/cone integration. No universal width is prescribed.

The rule's `maximum_angular_panel_nodes` defaults to 4194304 per source/rod-group
sampler call and rejects excessive automatic or explicit-panel work before yielding
any coordinates. It is a work guard, not a numerical-refinement override. The seeder
also fails when its declared panel budget is exceeded; neither uses a coarse fallback.
Edges are included in native revisions, cache keys and serialized rules. For geometry
or source probes, fixed edges alone do not freeze changing support: prepare a common
enclosing `frozen_ewald_bounds_Ainv_rad` and common axial rule as well. Reuse one mesh
across the planned candidates, and independently qualify its observable contrasts.
The bound applies to the entire sampler call, not each axial row. A fine mesh over
a large axial domain may exceed it; the demonstrated one-row cost is not a bound
for a full family. Do not silently raise the budget or relabel such a rejection
as numerical convergence.

Coordinates and transfers stream in batches, including splits within an axial row
or panel. Every batch retains the complete axial grid and global axial indices,
so cached structure factors cannot alias a slice-local grid. Automatic preparation
still stores five arrays per accepted panel (40 bytes per panel, plus construction
storage); retained sparse responses have a separate memory cost. Batch size only
bounds temporary joint-coordinate/transfer arrays. Numerical algorithm v2 and the
resolution fraction are bound to native identities and serialized rules, preventing
reuse of a pre-migration response or checkpoint.

`stitch_grid_size` refines the empirical Bi handoff calculation. Baseline, numerical
probes, reported fits, local-sensitivity probes and profile candidates record the
actual surface/wavelength interval selection, overlap scale and zero-strength
normalization. Pb's non-reflectivity model returns no handoff records. A switch in
the empirical handoff remains a potential source of nonsmooth parameter response.

`stitch_overlap_measure="continuous_q_median"` opts into a uniform-Q overlap median,
using separately interpolated Parratt numerator and internal-phase denominator on the
complete `[5Qc,10Qc]` interval. This named extension avoids counting grid points as the
measure. The default `sampled_log_median` preserves manuscript compatibility. The first
divergence is overlap normalization; automatic/fallback blend-window selection is unchanged.

The explicit response cache retains at most two geometry responses per rod partition
and 64 small prediction vectors. Response v2 excludes source masses: line probability updates
reuse per-line geometry and multiply current aligned normalized weights exactly
once, including zero-probability endpoints. Atomic z/ADPs, stacking, mosaic,
population weights, thickness and roughness recompute their physical intensity.

`FiberScatteringCache` separately retains pre-projection scattering, bounded by
256 MiB and 4096 entries. Detector corrections, sample normal translation and
spatial beam moments can reuse this state when their actual quadrature nodes and
upstream optical inputs match. New origins, Gaussian kernels, visibility and
region probabilities are still computed. Cell, occupancy, sample tilt, divergence
and wavelength update their dependent scattering/optics. No stale surviving-event
mask is reused. Cache owners are explicit; there is no hidden global cache.

`source_override.local_m0_divergence_order` optionally sets a positive Gauss-Hermite
divergence order for stitched `(0,0)` alone. The regular rods keep `divergence_order`.
`None`, equal orders, or an absent stitched rod use the ordinary unsplit path.
`NativeFitPhysics.integration_parts()` applies the distinct rules after binding the
physical candidate; each partition samples the same physical source distribution
with normalized weights. Predictions sum raw partition intensities before one scale.
Fitting, numerical checks, controls and rendering consume those same partitions.
Numerical-check `source_partitions_candidate_index` identifies the recorded probe
in its scope's `qualification_candidates` array. Rendering checkpoints retain each
partition's revision, completed batch count and completion flag.

For stationary proposals across transport perturbations, declare
`frozen_ewald_bounds_Ainv_rad=[Q_low,Q_high,arc_start,arc_width]` in the integration
rule. It must enclose each candidate's conservative source/region support; otherwise
evaluation raises. Default adaptive proposals remain available but may miss this
cache when geometry changes. Domain coverage does not establish quadrature accuracy.
There is no permanent finite response valid for arbitrary materials or parameters.
Gaussian corner reuse is restricted to adjacent native pixels; fitting rectangles
retain the direct probability loop. Stable cancellation handling and independent
conditional-CDF fallback remain shared.

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

## Matched fixed-parameter controls

An optional plan `fixed_parameters` object maps explicit parameter names to fixed
physical values. Every start must contain those exact values, within declared bounds.
No stage may release them; the final stage must release every remaining coordinate
in the declared order. Omitting this object preserves the full-release requirement.
Results label these experiments `fixed_parameter_control` and retain the values.
Numerical qualification, scale profiling, N refits and conditional validation retain
their ordinary gates. Sensitivity and identification profiles belong to the separately
released full model; fixed controls reject those options rather than silently freeing SF.

`local_m0_maximum_axial_panel_width_Ainv` optionally overrides the global axial panel
cap for the local-lamella m0 channel. `null` inherits the global cap. For example,
global `null` plus local `0.02` reproduces the prior explicit m0/nonzero partition
using one shared evaluator. Both caps require composite Gauss quadrature. Each
channel retains its full support and contributes raw mass before one fitted scale.

`local_m0_angular_power` similarly overrides `angular_power` only for the declared
local-lamella m0 channel; `null` inherits the global order. This lets the same
evaluator retain an adequate m0 rule while refining nonzero rods. The numerical
revision includes the override; neither domain nor physical weighting changes.

## Qualification of optimizer decisions

A numerical pass at a large parameter perturbation does not establish finite-difference
gradient accuracy. SLSQP uses normalized search coordinates `(value-lower)/(upper-lower)`
and the declared absolute `finite_difference_step`. At `eps=1e-4`, its physical step is
normally `1e-4*(upper-lower)`, reversed near the upper bound. Sensitivity uses its own
`sensitivity_scale` and relative step; those are different diagnostic stencils.

Use one existing diagnostic plan per center and integer N: retain the full parameter
roster and frozen training split, set `starts=[center]`, `repeat_choices=[N]`,
`stages=[]`, `profiles=[]`, `sensitivity=false` and `controls=false`. Supply the center
followed by the actual one-sided optimizer stencil points in `qualification_candidates`.
Omit `qualification_scale` so the runner profiles a reference training scale and reuses it.
Preserve independently justified numerical refinements and the existing acceptance gates.
Use actual recorded evaluation vectors when replaying an existing optimizer decision.

For each numerical rule, subtract the center's profiled objective from each stencil
objective and divide by the valid training count times the signed normalized step.
Report gradient differences and sign changes separately. Se's 884 training rows and
`eps=1e-4` mean an objective-contrast error of `0.5` still permits a normalized gradient
error of `5.656`; the existing inference gate alone is not a gradient criterion. Rebuild
stencils at every audited center because a translated/clipped probe need not match SLSQP's
bound-aware step. Sensitivity singular values likewise require their own refinement check.

Unqualified initialization remains diagnostic-only. Independent refinements must preserve
the proposed improvement direction and meet a predeclared relative decision-error budget.
It cannot produce selected estimates, profiles or identification claims. Restart the
normal qualified workflow from any retained warm start; never relax final inference gates.

## Batched finite-difference predictions

`fit_native_parameters(..., predict_many=callable)` optionally batches SciPy's bounded
finite-difference predictions. The callable receives a two-dimensional array of complete
physical parameter vectors, including fixed coordinates, and returns finite real raw
predictions with shape `(candidate_count, observation_count)` in the same order. SciPy
1.16 or newer is required for this option; ordinary serial fitting retains the existing
dependency range. The optimizer, finite-difference steps and statistical objective are
unchanged.

Only prediction calculation may be concurrent. Scale scoring, calibration, guards,
best-point updates and callbacks execute serially in the original candidate order.
Callbacks must not alter predictor state within a batch. The caller owns worker isolation:
native response caches and XrayDB sessions must not be shared between concurrent workers.
Process workers should receive explicit model state and return prediction arrays; the
parent owns diagnostic writes. Reuse an existing response for inexpensive coordinates
before distributing geometry-changing predictions. Historical-guard constraint derivatives
remain serial in SciPy. Batching changes execution cost, not numerical qualification.
The multi-choice refit/profile wrappers reject a shared batch callable when more than
one discrete choice is supplied. Bind scalar and batched predictions to the same N and
call the single-choice search separately; a callable bound to one N cannot supply the
finite differences of another.

## Opt-in axial preparation and reference proposals

The declared spatial quadrature order controls both Plackett-angle and
conditional-CDF rectangle integration, identically for native regions and pixels.
Order 16 is unchanged. Lower orders no longer inherit a hidden angle-order floor;
previous lower-order qualifications must therefore be repeated. Spatial-order
checks remain independent of source, angular and axial refinement, and the
Gaussian tail bound does not include rectangle-quadrature error.

`AxialPanelMesh` declares a complete radial rod group, a physical coordinate and
strictly increasing edges in inverse angstroms. Use `positive_phase_axial` for
regular rods or `external_local_m0_q` for local m0. Every signed rod remains present.
Edges must enclose source/detector support throughout the intended neighborhood;
insufficient meshes raise rather than crop the model.

`FiberIntegrationRule.axial_meshes` selects physical GL8 panels. It requires
`quadrature_kind="composite_gauss"`, `axial_power=3`, zero seed and no width caps.
Refine edges, not the now-inactive axial importance-proposal metadata. Angular,
source and cone integration remain independently controlled. The public
`NativeJointEvaluator.predict_axial_panels(values, N)` returns nonnegative raw
`(panel, observation)` contributions through the ordinary response and factors.
Their sum recovers `predict`; source partitions retain global panel ordering.

For preparation, use a measured-data plan with one `repeat_choices` entry,
center-first `qualification_candidates` containing the actual optimizer stencil,
existing `numerical_tolerances`, and an `axial_adaptation` object containing:

- `initial_meshes`: complete rod groups, each with `rods_hk`, `coordinate` and
  `edges_Ainv`; seed known narrow peaks, optical transitions and support boundaries.
- `fixed_scale`: the positive reference scale in the objective's count units.
- `maximum_panels` (default 2048) and `maximum_passes` (default 12): explicit work limits.

Run `scripts/refine_native.py` with the usual inputs/output and `--prepare-axial-mesh`.
Preparation applies the input rule's global/local-m0 physical panel-width limits
to every initial gap before adaptation, retaining all supplied and elastic-cutoff
edges. The local limit overrides the global limit only for local m0. Excessive
seed size fails before evaluating predictions; the work budget includes children.
The returned explicit mesh has no separate width cap because its edges carry it.
Each GL8 panel is compared with two GL8 children. Covariance-whitened prediction and
stencil-contrast errors prioritize refinement; the whole-vector objective gate also
must pass. Training splits use training rows only. Each N is prepared separately so
its own center defines contrasts. Work-budget exhaustion raises explicitly.

The external `rasim-native-axial-mesh-v1` diagnostic retains plan/input/source hashes,
stencil vectors, parent/child evidence, costs and the prepared `integration_override`.
Copy that override into the normal fit/render plan and freeze it during nearby steps.
Qualify independently: **both parent and child rules can miss a narrow peak.**
Preparation is neither a certified bound nor fit selection. Cross-N ranking,
source/angular refinement and held-out validation keep their ordinary gates.

An optional stage `reference_correction` object contains `numerical_override`
(the cheaper source/integration rule), positive `trust_radii` in physical parameter
units and full parameter order, optional `maximum_updates` (default 2) and
`maximum_function_evaluations` (default 30). `reference_corrected_start` freezes
`high(reference) + low(candidate) - low(reference)` inside each bounded optimizer
call. This additive correction handles zeros without dividing by intensity.
Out-of-box/negative corrections use an explicitly recorded high-rule fallback.
Only an improving high-rule objective passing prediction/contrast gates moves the
reference; rejected moves shrink the trust box.

Here “exact” means the declared high numerical rule, not the continuous integral.
Corrected vectors never enter exact recovery, normal fit history or qualification.
The helper produces only a warm start; the ordinary all-active high-rule stage
follows. Fixed controls, calibration ownership and guards remain unchanged.
Resume restores exact rows and restarts proposal work. Include setup, cheap calls,
rejections and final exact fitting in benchmarks: short/cheap fits can be slower.
Both accelerators are opt-in, never enabled automatically by a preparation pass.

Long inversion-paired axial rows share atomic geometric sums automatically:
per-species `G(-Q)=conj(G(Q))`, but anomalous `f` stays unchanged. Generally
`F(-Q) != conj(F(Q))`. This covers mirrored in-plane rods and 00±L without assuming
centrosymmetry, mirroring detector pixels or merging signed intensities. Finite
stacking already uses its authoritative closed-form repeat factor; no duplicate
stacking implementation or intensity-equality assumption is introduced.


## Full display support

`render_native.py --candidate --full-image --bin-size-px 12` reproduces the saved
unqualified candidate before integrating complete 12x12 native-pixel rectangles.
Bin size must divide the detector dimensions. Output is `simulated_detector_cell_count`
with native-coordinate cell centers; size1 retains `simulated_detector_native_count`.
Mass is integrated, not sampled/interpolated. No display-only scale or background is fitted.

`--profile-projection PATH` optionally supplies a numeric NPZ with integer
`detector_shape_rc[2]`, scalar integer `observation_count`, and the usual projection
arrays `flat_pixel_index`, `observation_row`, `pixel_column_index`,
`detector_area_weight_px2`. Its SHA binds the projection and checkpoint. The output
`display_profile_count` uses the same bound physical candidate and fitted scale;
profiles are integrated separately, never reconstructed from coarse image cells.
Resume preserves completed profiles and image batches and rejects changed identities.
The display projection does not change the fitting observations or objective.

See [supported staged fitting](STAGED_FITTING.md) for Bi ordered continuation,
Pb disorder/control recipes and the shared generic-CIF/acquisition boundary.
