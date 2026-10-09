# Physical detector-native refinement

The fitting path binds physical candidates into the same finite-structure,
optical transport, spherical mosaic and native-pixel integration used by the
renderer. A fit predicts the frozen native observations; moving geometry or a
lattice never moves measured pixels into another fitting region.

## Run a fit with automatic device selection

1. Prepare the acquisition using [the input-adoption commands](#portable-inputs-and-saved-images).
   Use its actual hash-prefixed physics filename, frozen observations and a complete
   acquisition-bound fitting plan. To continue a saved best fit, place its physical
   parameter vector in `starts` as one row, in the current declared parameter order:
   `starts = [saved_point["parameters"]]`. The integer repeat count is separate:
   put `saved_point["N"]` first in `repeat_choices` for a matching saved-state
   baseline; other declared choices may follow. Retain the specimen identity,
   units, bounds and qualification. A saved nominal candidate
   remains nominal; changing the observation/background operator needs a new matched
   baseline at that starting state.
2. Keep automatic selection, which is already the default when `spatial_execution`
   is omitted. The following is a fragment to merge into a complete plan, not a
   standalone fitting plan:

   ```json
   {
     "spatial_execution": "auto",
     "workers": 1,
     "prediction_workers": 1
   }
   ```

   Inner `workers` must be 1. Start with one candidate worker unless separate
   measurements justify more: candidate processes can contend for the same GPU
   and each evaluator/group owns its calibration. Keep the existing physical
   support, precision, quadrature and acceptance limits. `auto` requires no
   manually selected event threshold or device. Without an available compatible
   CUDA device/toolkit it chooses CPU and records the admission reason.
3. Run the declared bounded fit from the repository root, writing outside it:

   ```powershell
   uv run --frozen python scripts/refine_native.py `
     --physics C:\external\prepared\PHYSICS.json `
     --observations C:\external\prepared\sample_observations.json `
     --plan C:\external\fit_plan.json `
     --output C:\external\fit.ra_diag.npz
   ```

   The first substantial batch performs bounded local timing calibration; later
   batches owned by that evaluator reuse it. Small batches can stay on CPU while
   larger batches use GPU within the same prediction. This changes deposition
   execution only. Numerical qualification and the complete baseline must still
   finish before a fit can be accepted. Leave `require_initial_qualification` at
   its true default for qualification-gated fitting.
4. Inspect `manifest_json` in the result/checkpoint. `spatial_execution` records
   the requested mode; `parent_spatial_execution` and `worker_spatial_execution`
   report decisions and calibration. Refined and reference-correction evaluators
   retain summaries in their respective records. Counts describe attempted device
   selections, not completed predictions. Assess `execution_status`,
   `numerical_status`, `identification_status` and `selected` separately. A device
   choice, low objective or optimizer termination does not establish an accepted fit.
5. Resume an interrupted run by repeating the same command with `--resume`, the
   same output, and unchanged plan/input/code/dependency identity. It restores
   completed raw predictions and restarts the public optimizer; it does not restore
   a partly evaluated source/batch. Changing `auto` to an explicit mode, changing
   starts/support, or updating the engine requires a new output rather than reusing
   that checkpoint. Preserve previous execution lineage when reporting reused data.
6. Render the returned selection through the same saved plan and input bindings:

   ```powershell
   uv run --frozen python scripts/render_native.py `
     --physics C:\external\prepared\PHYSICS.json `
     --observations C:\external\prepared\sample_observations.json `
     --result C:\external\fit.ra_diag.npz `
     --output C:\external\render.ra_diag.npz
   ```

   Use `--candidate` only for the separately recorded optimizer candidate and retain
   that label. Add `--full-image` only when a full detector image is needed; it adds
   work. Rendering inherits the saved execution mode and records its own calibration.

For an explicit comparison or diagnosis, set the plan field to `"cpu"` or `"cuda"`
and use a new output. Explicit CUDA requires a working supported device and raises
on failure; neither it nor an automatically selected CUDA execution silently retries
on CPU. If a node/time budget is exhausted, inspect the recorded failure and actual
work before another bounded attempt. Faster deposition does not resolve excessive
adaptive refinement or justify relaxed tolerances.

For Python callers, `NativeJointEvaluator(...)`, `ConditionalStructureDetector(...)`
and `kernels.integrate_native_pixels(...)` select automatically with no extra setting.
Read `evaluator.spatial_executor.summary()` or `kernels.spatial_executor.summary()`
for decisions. When creating many separate kernel objects, reuse one
`NativeSpatialExecutor` via `executor=` to retain calibration across their calls;
the native evaluator already handles that sharing. The desktop native Simulator
continues to request CPU under its existing numerical resource reservation.

Automatic execution does not create or merge observation regions. For a combined
Bragg-region/profile fit, first follow the
[simultaneous-observation recipe](FITTING_WORKFLOW.md#simultaneous-native-bragg-regions-and-profiles-2026-10-09).
The standard CLI retains its declared observation/background objective. Joint
`NativeLinearBackgroundProblem` profiling is a Python fitting API; the external
41-knot Bi2Se3 study is not enabled by a new CLI plan switch. Its caller can use
the same automatic native evaluator without changing its raw-count objective.

## Automatic CPU/CUDA spatial execution

Native fitting and rendering default to `spatial_execution="auto"`, selecting
CPU or CUDA for each Gaussian deposition batch, for both regular rods and the
local-lamella composite. Plans may explicitly set `spatial_execution` to `cpu`
or `cuda`. The same modes are available on `ConditionalStructureDetector`; the
low-level `DetectorSpatialKernels.integrate_native_pixels` keyword is `execution`.

`NativeSpatialExecutor` is an explicit execution resource, separate from physical
state. Evaluators share it across candidate detectors; standalone kernel calls
reuse their own `spatial_executor` automatically. An optional `executor=` shares
calibration across different low-level kernel objects. There is no global cache,
disk cache, import-time device initialization or background calibration. A resource
is confined to its first calling process/thread. Different processes own their
own calibration; a changed device or quadrature rule conservatively uses CPU.

Auto first excludes small batches (fewer than 64 active events or 262144 clipped
pixel visits) and near-degenerate conditional widths. These are conservative
admission floors, not universal crossover claims. It counts actual column-conditioned
pixel visits in four correlation branches and checks device capability, toolkit
availability and free memory. Memory admission includes calibration's largest
1024-by-1024 float64 transfer window, actual parameter/coefficient buffers and 20%
headroom; it checks free memory again after calibration. CUDA errors after selection
propagate without a CPU retry.

The first admitted batch calibrates production CPU/CUDA functions on four bounded
256-event, 256-by-256 cases, plus zero-mass calls measuring launch/transfer/event
overhead. Timings include transfers, synchronization and readback; setup/compilation
time is recorded separately. Calibration checks finite CPU/CUDA outputs and maximum
pixel disagreement at most `1e-11 * CPU_peak`. An unresolved or nonpositive compute
rate selects CPU; a numerical parity failure raises. The frozen local timing model
requires predicted CPU time greater than `1.35 * predicted_CUDA_time` to choose CUDA.
This is a conservative heuristic, not a guarantee under every workload or changing
system load. No physical fit or numerical convergence follows from calibration.

`spatial_executor.summary()` reports calibration, most recent decision and counts
by device/reason. Counts mean attempted selections, not completed predictions.
Low-level decisions also use the ordinary debug logger; diagnostics stay off by
default. Fit checkpoints retain parent, completed worker-group, refinement and
reference-correction summaries; rendering retains summaries and resume lineage.
Failed worker groups do not return a completed execution summary. Cached raw
predictions retain their previous execution provenance. The desktop native Simulator
continues to request CPU explicitly under its existing CPU-only numerical reservation;
its resource-admission policy is separate from the fitting/batch APIs.

Canonical source, geometry, optics, signed strengths, spherical mosaic and angular
preparation remain on the CPU. CUDA receives those already contracted event masses
and the unchanged Gaussian means/factors, spatial order, tail radius and native
window. It returns the full pixel patch used by the existing adaptive pixel-L1
criterion before fractional observation memberships apply. There is no center
sampling, rectangle substitution, surviving-mass normalization or altered tolerance.

The CPU and CUDA compilers share scalar Gaussian interval, conditional-CDF,
correlation-corner and cancellation arithmetic. CUDA uses float64 event atomics,
so independent positive contributions may sum in a different order. CUDA is loaded
lazily. Explicit `cuda` raises for unavailable hardware rather than substituting;
auto records CPU admission when hardware is unavailable. The Numba CUDA runtime needs a compatible
CUDA toolkit and compute capability at least 6.0. No new package dependency is added.
The requested execution mode participates in the detector revision. External raw
prediction stores must bind it and retain execution summaries with completed evidence.

The October 9 RTX 3060 comparison retained the actual Bi2Se3 first local batch:
16,352 events and a 1579-by-3000 native window, spatial order 16 and radius 8.
CUDA differed by at most 3.39e-21 A2, or 3.49e-16 of the fixed CPU peak.
The refactored CPU output was bitwise identical to the prior CPU implementation.
Warm terminal times, including transfers/readback, were 2.78–2.90 s CUDA versus
13.03 s CPU before concurrent loading increased CPU time. Cold compilation is
separate. A 128-patch regular comparison retained identical 1024 accepted event
nodes and patch bounds; its maximum difference was 2.03e-15 of the CPU peak.
Those small patches took 1.84 s CUDA versus 1.65 s CPU, so GPU acceleration is
workload-dependent. These timings do not establish a full prediction or fit speedup.

The subsequent automatic-selector check reused that full saved 16,352-event window
without changing masses or spatial rules. Default calls chose CUDA and reused the
same calibration: 9.82 s first call including 5.86 s calibration, then 2.65 s warm,
versus 13.15 s explicitly warmed CPU. The CPU result remained bitwise identical to
the saved comparison. Tiny eight-event batches selected CPU without importing CUDA
and matched explicit CPU bitwise. Admission, override, failure-propagation, clipped
visit-count and worker-summary checks are implementation evidence. The external
`auto_spatial_selection.ra_diag.npz` retains the checks and measured outputs; no fit,
whole prediction or general optimal-routing claim is made.

Targeted comparisons also cover both correlation signs, near-unit correlation,
off-window centers, native offsets, zero event mass, caller-owned accumulation,
an independent separable Gaussian integral and unchanged conditional-CDF panels.
The external `gpu_native_validation.ra_diag.npz` retains numeric results, exact
callers, execution/source hashes and completed-source flags. Implementation
equivalence does not establish angular convergence, physical adequacy or a qualified
fit. In particular, the nominal local-m0 rule retains its existing qualification
limitations, and a changed floating-point sum can affect a near-threshold adaptive
decision; assess the declared complete observable when qualifying a run.

The subsequent all-spatial CUDA attempt retained all 85 rods and four original
source probabilities on the required training, held-out and valid profile supports.
Four nominal source processes ran from 15:20 to 15:40 UTC and completed no source
vector. The fine rule remained queued and the optimizer never started. A live sample
at 15:29 found 28,184 evaluated angular nodes in one early regular axial interval,
with the unchanged full-pixel relative error tolerance of 5e-5 and the GPU busy.
This bounds implementation availability, not end-to-end capacity: the large adaptive
work count remains unresolved. No completed nominal/fine observable or new fit is
claimed, and incomplete source arrays must be read with their completion flags.

## Optional positive background profiling

`NativeBackgroundProblem` is an explicit opt-in to `score_native_prediction` and
`fit_native_parameters`. Supply raw measured counts in `NativeFitObservations.net_count`,
one acquisition, GLS, no historical guards, and the declared native count covariance.
Any discrepancy covariance needs a separately justified measurement model; it is not
required by the profiler.
The problem binds actual pixel design X, sparse ownership W, immutable beta0 and absolute
quadratic penalty R. Background mass is W exp(X beta); the 44-column broad empirical design
in `native_background_design` preserves the existing 3000-pixel acquisition conventions.
It is not an instrument-independent background calibration.

Each physical prediction profiles one nonnegative exposure and runs bounded linear-loss
TRF from beta0. The physical optimizer consumes the same data-plus-R-beta residual,
divided throughout by the square root of the fixed valid data-row count, so its squared
norm is divided by that count. Results retain unscaled physical
prediction, exposure-scaled signal, background, total, coefficients, residual blocks,
objective components and inner termination/work. An unsuccessful inner solve raises
`BackgroundProfileError` with its result; it cannot become an outer optimization point.
The default `None` path preserves the frozen-background behavior.

The caller owns geometric support, count covariance including overlaps, discrepancy-mode
projection and held-out diagnostics. Fitting beta explicitly excludes its old jackknife
modes; a shared residual-discrepancy field must enter the joint covariance only once.
Controls must use current physical predictions whenever physical parameters change.
Conditional background improvement does not qualify the physical integration or identify
signal/background contributions under protected peaks. The expanded-support Bi2Se3
procedure and comparison limits are recorded in
[STAGED_FITTING.md](STAGED_FITTING.md#expanded-support-bi2se3-background-fit).

## Declared spatial count discrepancy

`SpatialCountDiscrepancy` is an opt-in zero-mean Gaussian residual field. The caller
freezes native `(column,row)` centers, `basis_width_px` and `sigma_count_per_pixel`.
For native pixel p and center c_j, define g_j(p) = exp(-||p-c_j||^2/(2w^2)) and
Phi_j(p) = sigma g_j(p)/sqrt(sum_k g_k(p)^2). Thus each pixel has variance sigma^2;
the induced covariance is Phi(p) dot Phi(q). The finite normalized grid gives a
nonstationary cosine kernel. The basis width is not its correlation length.

`project_modes` returns U = W Phi through the literal fractional or signed native
pixel memberships. Shared pixels share one field; no footprint-area normalization
is applied. Form C = C_working_count + U U^T once and use the existing full SPD
`NativeFitObservations` Cholesky and quadratic GLS. No diagonal jitter, extra
whitening path, residual variance rescaling or robust loss is introduced.

For joint exponential-background profiling, supply raw counts and the working count
covariance. Do not use the historical loader's background-subtracted counts or add
its beta jackknife modes: beta is already profiled in the mean. Bind raw counts,
validity/order, working covariance, mode arrays, construction and scenario to
`input_revision`. Keep C fixed during physical optimization. Its Gaussian log determinant
is constant within that scenario; record it, and never compare unlike scenario
quadratic scores as a physical improvement. Estimating C jointly would require the
Gaussian determinant and conditioning terms as well as independent identification.

Use the same field to propagate covariance into protected signed contrasts and
continuous profiles through their actual overlaps. They remain diagnostics rather
than duplicate likelihood rows. Control-derived amplitudes can contain diffraction
and counting fluctuations; explicit assumptions support conditional sensitivity,
not calibrated coverage or identification of detector noise. A second scenario with
larger amplitude and identical modes/basis width has a positive-semidefinite covariance
increment. Qualification must still bound absolute count/feature numerical error and
the error under the scenario covariance; a larger C cannot qualify unchanged quadrature.

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

For a new mosaic fit conditional on a saved OSC position result, prepare one
matching native acquisition with an explicit image ID:

```powershell
uv run --frozen python scripts/prepare_native.py `
  --sample bi2se3 --input-root C:\external\native-inputs `
  --geometry-position C:\external\geometry.json `
  --geometry-manifest configs/bi2se3_osc_geometry_fit_model_limited.yaml `
  --geometry-image-id Bi2Se3_5m_5d `
  --output-directory C:\external\prepared-current-pose
```

The adapter verifies that the selected native counts equal the manifest OSC in
detector-native orientation, applies the fitted position once from the configured
base, and replays that image's saved integer-`L` markers before publishing new
hash-bound physics. It leaves the measured projection, background, covariance,
source sampling, structure, and numerical integration unchanged. Its descriptor
records the position, selection, configuration, OSC, and marker-replay provenance.
Use `refine_native.py` with a new plan bound to the prepared physics hash and
original observation lineage; the old plan or archived baseline belongs to the
old pose. The catalogued Bi2Se3 observation contains only the five-degree
acquisition, so this route supports a conditional single-image mosaic estimate.
The saved three-image position result remains model-limited, and this preparation
does not make either position or mosaic scientifically qualified.

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

## Conditional Gaussian/Lorentzian amplitudes

The Python APIs opt into complete component predictions with
`NativeJointEvaluator.predict(values, N, resolve_mosaic_components=True)`. The
result has shape `(2, observation)`, ordered Gaussian then Lorentzian, with each
orientation law normalized separately. Both widths must be positive, even at
eta endpoints. Each pure law prepares its own dependent axial/angular response,
retaining every source, rod, signed local-m0 term and optical factor. Default calls still return
the mixed raw masses. `evaluation_count` counts completed uncached point requests;
`contraction_count` counts the completed component contractions in every disjoint
source/rod partition. Compilation remains a separate counter.

`profile_mosaic_amplitudes` solves conditional single-exposure GLS for
`signal = u*F_G + v*F_L`, with `u,v >= 0`, exposure `a=u+v`, and
`eta=v/a` when `a>0`. Narrower eta bounds are enforced by nonnegative coefficients
on their two extreme rays. Zero exposure reports eta as `None`. The returned
rank/null-direction diagnostics concern the admitted conditional amplitude cone;
they do not establish joint physical identification. Proportional unequal
columns can leave exposure unidentified. Equal columns can identify exposure
while leaving eta unresolved.

Pass `mixture_parameter="lorentzian_probability"` to the shared scorer/search,
and provide the ordered raw columns from the predictor. The search removes only
that coordinate from its nonlinear vector and reconstructs it in the complete
physical result. At zero exposure its input coordinate remains provenance, with
no fitted fraction claimed. Raw columns, signal, background and total predictions
remain separate. Guards, literal exposure, multiple exposure groups and a
calibration coupled to eta require a different inner problem and are rejected.
CLI plans do not yet enable this option.

With `NativeBackgroundProblem`, both amplitudes are re-solved at every beta. Its
existing exponential background remains nonlinear, restarted deterministically
from beta0, with the unchanged penalty, box and termination/failure behavior.
The analytic beta derivative projects off the active amplitude rays. It is valid
on a stable active face; zero-amplitude/zero-multiplier transitions require
bound-aware directional checks. Every physical finite difference must reprofile
amplitudes and background. Software checks do not qualify those measured
derivatives, an integration rule, a fit or a background subtraction.

`NativePredictionStore(..., resolve_mosaic_components=True)` retains the same
ordered two columns and completed partial batches. Bind its revision to full
inputs, source/rod support, numerical rule, implementation, parameter ordering
and component order. Scalar and component checkpoints cannot cross-replay.
Recovery retains raw work only; it does not prepend an incumbent or recover an
optimizer state, fitted amplitudes or beta.

A constrained linear-background candidate additionally needs a nonnegative
basis integrated over the exact footprints, a penalty in its own coefficient
units, and evidence-derived feature inequalities. Those data-dependent choices
are not supplied by the component API or exponential profiler. An unavailable
control-transfer envelope leaves background-dependent fitting inadmissible;
archived-model sensitivity ranges are not calibrated coverage bounds.

## Numerical qualification and execution

Run `scripts/refine_native.py --physics PHYSICS.json --observations OBSERVATIONS.json
--plan PLAN.json --output EXTERNAL.ra_diag.npz` from a configured environment.
The plan schema is `rasim-native-refinement-plan-v1`. The external diagnostic
contains numeric arrays and one embedded JSON manifest, atomically checkpointed.
Task 30 records the six concrete acquisition-bound plan and result locations.

`prediction_workers` defaults to 1. Larger values use one persistent isolated
process pool for the run; `prediction_group_size` defaults to 16. Each group owns
an evaluator and reuses exact dependent state across its candidates. Processes
persist, while evaluator caches are bounded to the group. Inner preparation
`workers` must be 1; candidate parallelism uses `prediction_workers`. Only the parent
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

The default regular estimator prepares positive strength-weighted Gaussian axial rules.
Its scalar measure is W(u) du, W=S+(u)+S-(u), from the canonical physical rod,
population and incoherent-mixture table. Cheap physical GL16 seed masses resolve
repeat/endpoint fringes using actual finite phase-depth extent, including unwrapped
CIF sites and Pb partial endpoints. Response panels include elastic, visibility and
mosaic-fold landmarks. No strength threshold removes a panel. Provably zero
populations/occupancies contribute zero; unresolved sampled-zero measures fail.

`strength_gauss_order=4` constructs interior positive nodes by reorthogonalized
Stieltjes recurrence and tridiagonal eigensolve. Coordinate scaling conditions the
recurrence while physical mass is restored without another Jacobian. Legendre
moments through degree 2n-1 check the discrete scalar measure. They do not establish
continuous scalar resolution or detector-observable axial accuracy. Refine
`strength_scalar_order`, `strength_scalar_phase_step_rad` and `strength_gauss_order`
against independent observable evidence. CIF, Pb finite surfaces and Bi2X3 finite
stacks use their authoritative strengths; unsupported custom providers raise.

S+/W and S-/W accompany W du through the shared transport compiler. Original source,
optical/polarization, cone density, attenuation, phase and deposition factor owners
remain. Each angular GL8 parent and its two children compile jointly. Tiled native
L1 parent/child indicators use `pixel_error_rtol=5e-5`, `pixel_error_atol=1e-15`;
absolute tolerance divides over all source/group/response-panel slots. Accepted
children retain patches; rejected children become cached coarse values. The empirical
indicator can miss common unresolved structure. It is not an axial error bound,
continuous-density proof or numerical fit qualification.

`angular_initial_power=5` and `pixel_error_initial_width_rad=0.1` seed conditional
angular panels. `pixel_error_maximum_depth=32`, `maximum_angular_panel_nodes=4194304`
and `pixel_error_maximum_bytes=268435456` guard work and live patch storage.
The byte guard excludes caller images, transfer arrays and small metadata; it is
not an RSS ceiling. Exhaustion raises; accumulated prefixes remain incomplete.
Spatial order stays 16 with complementary-residual corners and rolling edge/tail reuse.

Optional `regular_q_bounds_Ainv`, `local_m0_q_bounds_Ainv` and
`frozen_ewald_bounds_Ainv_rad` must enclose actual candidate support. They do not
freeze W-dependent nodes or acceptance. Every distinct candidate rebuilds dependent
preparation, including sites, repeats, lattice, populations, optics, source, geometry,
thickness and actual mosaic changes.

The required local-m0 endpoint chart keeps `local_m0_axial_power=12`,
`local_m0_angular_power=5`, `local_m0_seed=0`, `local_m0_peak_spacing_L=1`,
`local_m0_peak_half_width_L=0.02`, `local_m0_angular_resolution_fraction=0.5` and
explicit `local_m0_axial_peak_coordinate`. Its external-Q endpoint transformation
and empirical composite remain; it is not a competing regular engine. Ordinary
historical selectors and manual meshes have retired. The explicit fixed-importance
option below restores bounded sparse responses under a new identity. Migrate inputs
explicitly as documented in [ENGINE_MIGRATION.md](ENGINE_MIGRATION.md).

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

The exact candidate cache retains at most 64 completed small prediction vectors,
keyed by float64 parameters, integer N and mixed/component observable within one
immutable evaluator/projection owner. Complete G/L output independently prepares
both pure laws, even at eta endpoints. Source probabilities apply once without
survivor renormalization. Adaptive rules require fresh preparation when strength or
mosaic changes; the fixed rule below contracts those candidate factors afresh.

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
evaluation raises. Candidate-specific preparation remains required when geometry changes. Domain coverage does not establish quadrature accuracy.
There is no permanent finite response valid for arbitrary materials or parameters.
Native pixels reuse integrated corners on two rolling column edges. Native region
rectangles share physical corner identities within one Gaussian. Near unit
correlation those caches use complementary residuals; large standardized endpoints
use direct conditional integration without changing the cached representation.
Requested-order refinement, cancellation handling and conditional-CDF fallback
remain shared, with no probability clipping or normalization.

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
candidate/scattering caches and XrayDB sessions must not be shared between workers.
Processes receive explicit model state and return prediction arrays; the parent owns
diagnostic writes. Each distinct candidate reprepares dependent strength/angular state. Historical-guard constraint derivatives
remain serial in SciPy. Batching changes execution cost, not numerical qualification.
The multi-choice refit/profile wrappers reject a shared batch callable when more than
one discrete choice is supplied. Bind scalar and batched predictions to the same N and
call the single-choice search separately; a callable bound to one N cannot supply the
finite differences of another.

## Reference proposals

Adaptive regular panels are prepared internally by the strength-Gauss estimator.
Manual axial-mesh preparation and panel-prediction APIs have retired. Historical
diagnostics are evidence only. Current qualification still needs independently
declared source/angular/axial comparisons.

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
Reference correction is opt-in; low and high rules use the current engine.

Long inversion-paired axial rows share atomic geometric sums automatically:
per-species `G(-Q)=conj(G(Q))`, but anomalous `f` stays unchanged. Generally
`F(-Q) != conj(F(Q))`. This covers mirrored in-plane rods and 00±L without assuming
centrosymmetry, mirroring detector pixels or merging signed intensities. Finite
stacking already uses its authoritative closed-form repeat factor; no duplicate
stacking implementation or intensity-equality assumption is introduced.


## Full display support

`render_native.py --candidate --full-image --bin-size-px 12` replays a saved
current-engine candidate under its bound input/source identity before summing integrated native pixel masses into 12x12 display cells.
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


## Independent desktop native drafts

The desktop Simulator binds the six admitted native recipes without prepared observations,
fit records or nuisance scale. It reuses `load_native_fit_physics` on hash-bound immutable bytes,
the actual Bi/Pb joint model coordinates, `integration_parts`, each part's detector and
`iter_native_pixel_batches`. Bound Bi specular stack overrides precede source partitioning.
Native drafts and configured YAML drafts are separate schema-10 project states. Source/integration
rules, repeats, proposal mosaic and integrated rectangle width are explicit; current support is
CPU deterministic integration with one worker and nested BLAS thread. See the
[desktop workflow](../interactive/README.md#independent-native-simulator-and-reviewed-experiment-transfer).

Exact closed surface/phase declarations are checked against canonical shares before detector
binding. An unchanged Pb parent simplex preserves its exact declared weights; changed shares
still use the canonical simplex equation. This avoids last-bit reconstruction of a declared
reference simplex and retains tiny probabilities without normalization or pruning. The retained
fixed-rule GD1/Bi2Te3 comparisons establish adapter fidelity for those inputs only. Partial native
snapshots sum complete-panel additive event batches; integration completion is not convergence
or fit qualification. Native Bi disorder and Pb reflectivity stay unsupported.

Desktop delivery uses the user's representative nominal-proof policy in
[VALIDATION.md](VALIDATION.md#desktop-implementation-verification). Existing numerical validation,
rank/covariance checks and truthful fitting qualification states remain authoritative; this UI
binding adds no fit qualification or new observation-preparation recipe.


## Conditional finite background hull

`NativeBackgroundHull(columns_count, weights0)` profiles one exposure for a fixed
scalar raw diffraction shape and solves an unpenalized GLS simplex problem. Each
column contains count masses integrated over the same declared observation
footprints. One global nonnegative weight vector sums to one; callers must use
that same vector on any separately integrated display or control columns.
The finite family is a conditional model assumption, not a simultaneous
uncertainty bound. Control and overlapping feature diagnostics are not silently
added as likelihood rows.

`profile(observations, raw, callback=...)` admits one raw GLS acquisition without
guards. It returns exposure, weights, raw-model decomposition, data objective,
zero penalty, simplex feasibility, a convex objective-gap bound and fixed-shape
affine rank diagnostics. Rank deficiency leaves coefficient uniqueness unresolved;
full rank establishes uniqueness only within this fixed-shape conditional model.
Mosaic-component and outer fitting integration are not enabled by this interface.

SLSQP uses an analytic envelope gradient and deterministic `weights0`, with default
limits of 300 iterations and 600 distinct evaluations. Objective scaling is fixed
from the initial residual. The default tolerance `1e-8` applies to simplex feasibility
and the objective-gap bound divided by that scale; it is a solver criterion, not a
physical acceptance threshold. Coefficients are never clipped or normalized after
solving. A feasible accepted SLSQP iterate can terminate on the unchanged gap
certificate with `termination_kind="convex_certificate"`; its optimizer success/status
are `None`, because SLSQP did not return. Otherwise success requires optimizer
convergence and the same feasibility/gap gates. Failed checks return `success=False`;
evaluation-budget exhaustion raises `BackgroundProfileError` with the last point.
Callbacks receive every distinct evaluation for caller-owned accounting. Objective,
gradient and accepted-iteration counts remain separate from distinct evaluations.


## Candidate preparation and cache limits

The adaptive estimator retains no strength-free detector response. The explicit
fixed-importance option described below restores bounded nominal response reuse.
For adaptive rules, each pure or mixed law prepares its own angular acceptance
with candidate and observable identity. Exact completed prediction and upstream scattering reuse do
not imply convergence. Work accounting includes preparation, rejected panels,
source channels, failed calls and caller-owned image storage.


## Explicit fast combined-region fitting

For a frozen Bragg-cell plus unused-profile observation, explicitly set
`regular_rule="fixed_importance.v1"`, `regular_axial_power=12`,
`regular_angular_power=5`, and `regular_seed=0`. A stitched Bi fit also declares
`local_m0_angular_rule="cdf_stratified_importance.v1"` with its separately recorded
proposal, seed and existing minimum 4096 by 32 rule. This is a nominal estimator,
not the adaptive pixel-error estimator. Keep their numerical qualifications distinct.

Use `NativeJointEvaluator` on the complete combined projection and full covariance.
The first prediction compiles sparse native region probabilities on CPU. Later
candidates reuse them only when geometry, optics, source, proposal, nodes and
observation memberships match; structure, mosaic, attenuation and the specular
stitch update at every candidate. `auto` selects this available CPU region compiler;
forced `cuda` is unsupported here and raises. The pixel-raster path still supports
automatic CPU/CUDA execution. No per-peak scaling or additional background is implied.

Complete one bounded starting prediction, compare cached and direct same-node
contractions, then measure a changed-parameter update before budgeting optimization.
Allow for all finite-difference probes and reserve final numerical/preservation
checks. Refine axial and angular rules on the complete frozen support; never freeze
strength-dependent Gaussian nodes and call them this fixed rule. All signed counts,
background support limitations and nominal/selected distinctions remain in force.
The retained response cap excludes transient allocations, so callers separately
supervise RSS and elapsed time.
