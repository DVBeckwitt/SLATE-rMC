# CPU/GPU-compatible simulation and staged geometry fitting plan

Status: PROPOSED

Entry gate: begin after the stitch/integration work is accepted and the T07 detector-native
observable is proven.

This plan records the agreed path for:

1. making the detector simulation efficient on the CPU and compatible with measured GPU
   acceleration;
2. recovering known geometry blindly from detector-native expected peak centroids; and
3. recovering the same geometry from expected angle-space `(2theta, phi)` peak centroids.

The work must preserve one authoritative physics path. CPU rendering, GPU execution,
detector-space fitting, and angle-space fitting are consumers of the same candidate masses and
candidate-specific detector hits; none may implement alternate scattering physics.

## Governing constraints

- Complete and prove the detector-native simulation before fitting.
- Prove detector-native fitting before adding angle-space fitting.
- Parallelize over rays and bounded candidate tiles, not over the number of selected peaks, rods,
  or HKL groups.
- Keep fixed-seed source and mosaic sampling during every fit.
- Do not use randomly generated geometric truth values, random optimizer starts, or random
  geometric proposals for the current fitting proof.
- Use deliberately chosen geometric combinations whose truth is known but hidden from the fitter.
- Keep rod, reflection-group, branch, and `COLLAPSED_00L` identities frozen inside one optimizer
  run.
- Treat invalid or ambiguous topology as an invalid evaluation; never reassign identities inside
  the objective.
- Use radians internally and canonical detector-native `(column_px, row_px)` coordinates.
- Keep detector-to-angle transfer as a measurement transform. It cannot redefine upstream detector
  orientation, branch identity, event mass, or normalization.
- Use one exact full-pixel-splitting detector-to-angle transform and one normalized angle-space
  observable.
- Keep raw azimuth `chi_raw` and fitting/display azimuth `phi` as separately named quantities.
- Select CPU/GPU technology only after the accepted integrated workload is profiled, as required
  by [the performance strategy](../docs/PERFORMANCE.md).

## Dependency graph

```text
accepted stitch/T07 detector reference
    -> profiled and optimized candidate work count
    -> CPU ray-block execution
    -> measured GPU proof path
    -> T08 frozen selection identities
    -> T09 fit contracts and invalidation
    -> T10 accepted detector calibration
    -> detector expected-moment predictor
    -> detector blind geometry recovery
    -> frozen angle-space coordinate contract
    -> angular expected-moment predictor
    -> angle-space blind geometry recovery
    -> exact normalized-angle-field proof
```

Phase 1 follows [T07](07_integration.md) and a subsequent performance-focused change after T07 is
accepted. Phase 2 dovetails with [T09](09_fit_foundation.md), [T10](10_instrument_calibration.md),
and [T11](11_sample_geometry_fit.md). Phase 3 extends the later detector-to-angle transform in T15;
its owned fitting paths must be approved before implementation.

## Shared simulation boundary

Every execution mode consumes the same authoritative sequence:

```text
fixed source rays
    -> individual rods
    -> all valid m=0 and m!=0 mosaic/Q candidates
    -> complete physical candidate mass
    -> candidate-specific outgoing wave
    -> candidate-specific continuous detector hit
```

The complete candidate mass includes every applicable source, phase/parent, reciprocal/mosaic,
scattering-strength, optical, polarization, attenuation, and population factor exactly once. It
does not include detector solid angle for the raw image.

From this boundary, explicit consumers may:

- select stochastic outgoing events and render an image;
- accumulate deterministic detector expected moments; or
- construct the deterministic normalized angle-space field and its matched expected moments.

No fitting module may reimplement source sampling, mosaic support, Ewald geometry, optical
transport, detector intersection, or candidate mass.

## Shared centroid-and-line objective

Phases 2 and 3 use the same post-centroid objective structure in different coordinate spaces.

### Raw centroid component

For peak `i`, let the known target centroid be `mu_i*` and the trial expected centroid be
`mu_i(p)`. The point residual is

\[
\mathbf r_{i,\mathrm{point}}
=
\boldsymbol\mu_i(\mathbf p)-\boldsymbol\mu_i^*.
\]

In detector space this is

\[
(\Delta\mathrm{column}_{px},\Delta\mathrm{row}_{px}).
\]

Its least-squares contribution is the squared raw centroid distance

\[
d_i^2=\left\|\mathbf r_{i,\mathrm{point}}\right\|^2.
\]

Retaining two coordinate components gives the optimizer directional information while producing
the same unweighted squared-distance penalty as a scalar Euclidean distance. Measurement
covariance may later whiten the components, but raw coordinate distance remains a required
diagnostic.

### Nonzero-m two-branch line

For every locked nonzero-`m` reflection group, define target and trial lines using the final two
expected branches and the current two predicted branches:

\[
\mathbf u_g^*
=
\boldsymbol\mu_{g,1}^*-\boldsymbol\mu_{g,0}^*,
\qquad
\mathbf u_g(\mathbf p)
=
\boldsymbol\mu_{g,1}(\mathbf p)-\boldsymbol\mu_{g,0}(\mathbf p).
\]

Branch identities are exactly `0` and `1`. A swap is invalid rather than silently made
equivalent. Compare the line directions after centering each segment on its own midpoint:

\[
\Delta\alpha_g
=
\operatorname{atan2}
\left(
\det(\widehat{\mathbf u}_g^*,\widehat{\mathbf u}_g),
\widehat{\mathbf u}_g^*\cdot\widehat{\mathbf u}_g
\right).
\]

Use the midpoint-rotation residual

\[
r_{g,\mathrm{line}}
=
L_g^*\sin\left(\frac{\Delta\alpha_g}{2}\right),
\qquad
L_g^*=\|\mathbf u_g^*\|.
\]

This is the corresponding-endpoint displacement when two equal-length segments rotate around a
common midpoint. The fixed target length prevents a trial from reducing the angle penalty by
collapsing its predicted branch separation. The point residuals already constrain the segment
midpoint and length, so no separate line-offset residual is included.

### m=0 line

The `m=0` observations are branchless `COLLAPSED_00L` peaks. They must not be assigned artificial
branches. For every dataset and source/phase containing at least two distinct `00L` observations:

- two points define their connecting line directly;
- three or more points define a two-dimensional total-least-squares line;
- increasing `L` fixes the otherwise ambiguous line direction;
- the target line span supplies `L*`; and
- the same half-angle line residual is used.

For centered points `z_j`, form

\[
\mathbf C
=
\sum_j
(\mathbf z_j-\bar{\mathbf z})
(\mathbf z_j-\bar{\mathbf z})^{\mathsf T}.
\]

The two-dimensional principal-line angle can be evaluated without a general SVD:

\[
\beta
=
\frac12\operatorname{atan2}
\left(2C_{xy},C_{xx}-C_{yy}\right).
\]

Orient the unit direction so that

\[
\sum_j
(L_j-\bar L)
(\mathbf z_j-\bar{\mathbf z})\cdot\widehat{\mathbf u}
>0.
\]

Reject the line when it has fewer than two finite points, repeated `L` identities, negligible
span, an insufficient eigengap, or ambiguous angular wrapping.

Use separate typed eligibility policies for the nonzero-`m` two-branch line and the branchless
`m=0` axis line.

### Composite residual

The optimizer receives

\[
\mathbf r(\mathbf p)
=
\left[
\{\mathbf r_{i,\mathrm{point}}\},
\{r_{g,\mathrm{line}}\}_{m\ne0},
\{r_{a,\mathrm{line}}\}_{m=0}
\right].
\]

The line components are dependent guidance derived from the same centroids, not statistically
independent observations. The half-angle construction gives them the same coordinate units as the
centroids: pixels in Phase 2 and radians in Phase 3.

Use unit line weight initially because the residual is already in coordinate units. Introduce a
different declared weight only if the chosen training recovery cases show a conditioning need.
Freeze it before held-out recovery. A point-only final audit must confirm that line guidance did
not pull the solution away from the raw-centroid optimum.

## Deterministic blind-recovery protocol

The fitting proof uses only named, prescribed geometry cases.

| Case class | Purpose |
|---|---|
| Nominal control | Confirm zero displacement and no optimizer drift |
| Signed single-parameter cases | Isolate each parameter's observable effect |
| Small and medium excursions | Test local and moderately nonlinear recovery |
| Chosen pairwise sign combinations | Expose parameter coupling |
| Deliberately difficult safe combinations | Test conditioning near safe bounds |
| Held-out combinations | Test recovery without tuning against those cases |

Candidate signed levels are fixed fractions of each scientifically safe interval, such as 20%,
50%, and 80%, but a case is admitted only when its topology remains valid and the active block is
identifiable. The exact values and rationale are recorded in the truth manifest.

Use the named control, single-parameter, deliberately chosen coupled, difficult, and held-out
cases. Do not add random Latin-hypercube truth cases to the current proof.

Truth generation and fitting remain isolated:

1. Generate observations from the chosen truth combination.
2. Store truth only in a separate audit artifact.
3. Produce a sanitized fitting input containing no truth parameters or truth-like keys.
4. Start from a prescribed nominal or prescribed displaced initial point.
5. Run the fitter deterministically.
6. Reveal truth only after termination.
7. Compare parameters, fitted observations, and held-out observations.

No random optimizer starts or random geometric proposals are used. If multistart is needed, use a
small predeclared set of nominal, signed-axis, and selected-corner starting points. Audit sanitized
fit inputs recursively for truth-like keys or values.

Two proof tiers are required:

- Tier A uses the same deterministic source/mosaic sample revision for truth and fitting, isolating
  optimizer correctness.
- Tier B generates truth with a denser or different fixed sample revision, testing robustness to
  finite-sampling differences and avoiding an overly easy inverse test.

# Phase 1: working CPU/GPU-compatible simulation

## Goal

Complete and prove the detector-native simulation, then make its measured bottlenecks parallel
without changing its observable.

## Task 1.1: accept the integrated scalar reference

Description: complete the stitch/T07 integration and freeze the scalar/tiny reference before
performance work.

Acceptance criteria:

- [ ] T07 complete-pool selection, once-only mass, candidate-specific hit, deposition, clipping,
      and repeatability obligations pass.
- [ ] One ray reaches eligible `m=0` and `m!=0` rods and every valid candidate-specific mosaic/Q
      solution.
- [ ] The reference exposes stage timings, work counts, selected counts, and peak memory.

Verification:

- [ ] Run the eight T07 proof obligations in [T07](07_integration.md).
- [ ] Record the reference trace and benchmark artifacts outside the repository root.

Dependencies: accepted stitch work.

Likely owned paths: T07 `measurement/`, `render/`, `pipeline/`, and `tests/test_integration.py`.

## Task 1.2: reduce mathematical work

Description: profile the accepted reference and remove avoidable work before adding executors or a
GPU path.

Required order:

1. compile instrument transforms once;
2. localize Ewald/mosaic support;
3. avoid impossible rod-orientation combinations;
4. stream or use two passes instead of materializing the full Cartesian product;
5. separate candidate geometry from scattering strength; and
6. profile candidate generation, transport, detector intersection, selection, and deposition.

Acceptance criteria:

- [ ] A dense scalar enumerator remains the independent proof path.
- [ ] Localized support reproduces the accepted observable within frozen tolerance.
- [ ] Peak memory is bounded for the large-forward benchmark.

Verification:

- [ ] Compare equivalent candidate work, wall time, peak memory, and final output.

Dependencies: Task 1.1.

## Task 1.3: define the narrow numeric batch boundary

Description: expose compact contiguous inputs and outputs for the measured hot kernel without
creating a general backend framework.

```text
input:
    ray block
    phase/parent
    rod block
    localized mosaic/Q support
    compiled instrument and material state

output:
    validity/status
    candidate identity
    candidate mass
    outgoing direction
    continuous detector hit
```

Direct, tangent, or numerically uncertain cases retain a scalar robust fallback.

Acceptance criteria:

- [ ] The full diagnostic wrapper and streaming integration wrapper consume the same internal
      equations.
- [ ] Canonical state -> rod -> orientation candidate ordering is preserved.
- [ ] No public generic backend or arbitrary callback layer is added.

Verification:

- [ ] Compare batch and scalar validity, roots, masses, hits, identities, and first divergence.

Dependencies: Task 1.2.

## Task 1.4: implement CPU ray-block execution

Description: benchmark serial tiled execution before adding the smallest justified fused/parallel
CPU path.

Benchmark order:

1. scalar reference;
2. serial tiled/vectorized execution;
3. fused compiled CPU execution if profiling justifies it; and
4. increasing ray-block worker counts.

The implementation uses coarse ray batches, a bounded producer, explicit maximum in-flight
workspace, per-worker output ownership, no atomic image writes, and deterministic canonical
merging.

Acceptance criteria:

- [ ] CPU utilization scales over rays even when one or two peaks are active.
- [ ] Worker count does not change candidate identities, selected events, assigned mass, or image.
- [ ] Peak memory remains under the declared limit.

Verification:

- [ ] Benchmark serial and selected worker counts on small, medium, and large forward workloads.

Dependencies: Task 1.3.

## Task 1.5: preserve stochastic semantics

Description: key randomness to physical identity instead of execution order.

RNG keys depend on source sample, phase/parent, selection pool, and draw index. They do not depend
on worker count, chunk size, scheduling, or backend.

Acceptance criteria:

- [ ] Fixed-seed rendering is invariant to CPU chunking and worker count.
- [ ] Complete-pool candidate order and cumulative selection are stable.
- [ ] A changed sample revision changes provenance explicitly.

Verification:

- [ ] Hash selected candidate IDs and final pixels for multiple execution layouts.

Dependencies: Tasks 1.3-1.4.

## Task 1.6: expose stochastic rendering and deterministic moments

Description: feed two explicit consumers from the common candidate kernel.

The stochastic renderer retains complete-pool selection, configured outgoing-event count,
selected mass `T/N`, and conservative detector deposition.

The deterministic reducer accumulates

\[
M=\sum_i m_i,
\qquad
\mathbf S=\sum_i m_i\mathbf x_i,
\qquad
\mathbf H=\sum_i m_i\mathbf x_i\mathbf x_i^{\mathsf T}.
\]

This reducer establishes simulation capability; it is not yet a fitter.

Acceptance criteria:

- [ ] Moments equal direct complete-candidate enumeration.
- [ ] High-count stochastic image centroids converge to expected moments.
- [ ] `m=0`, `m!=0`, and combined pools are each proven.

Verification:

- [ ] Compare mass, first moment, second moment, centroid, and convergence rate.

Dependencies: Tasks 1.1-1.5.

## Task 1.7: add and evaluate GPU execution

Description: prototype only the sufficiently large, regular numeric kernel identified by CPU
profiling.

Requirements:

- keep immutable compiled states resident during repeated work;
- use float64 for proof;
- avoid unordered atomic reductions;
- preserve stable compaction and reduction order;
- record compile, transfer, execution, and reduction time separately; and
- define a measured CPU/GPU crossover.

The production path may remain CPU-based when strict-float64 branch-heavy geometry is faster on
the CPU. GPU compatibility means the same numeric contract executes and reproduces the accepted
observable; it does not require forcing every workload onto the GPU.

Acceptance criteria:

- [ ] GPU and CPU reproduce the proof observable within frozen tolerance.
- [ ] Near-boundary cases use the declared robust fallback or satisfy the same classification.
- [ ] A measured crossover determines automatic or explicit backend choice.

Verification:

- [ ] Run small proof, medium forward, large forward, and repeated geometry-fit benchmarks.

Dependencies: Tasks 1.3-1.6.

## Phase 1 checkpoint

- [ ] T07 remains fully passing.
- [ ] CPU execution uses ray/candidate parallelism rather than peak parallelism.
- [ ] RNG, candidate order, selected events, and reductions are reproducible.
- [ ] Expected moments match direct enumeration and stochastic convergence.
- [ ] CPU/GPU agreement, crossover, wall time, and peak memory are recorded.

# Phase 2: detector-space geometry fitter

## Goal

Blindly recover prescribed geometric truth combinations from detector-native expected peak
centroids, using raw centroid displacement plus the nonzero-`m` and `m=0` line-angle constraints.

T08 selection, T09 fitting contracts, and accepted T10 detector calibration are prerequisites.

## Task 2.1: create known detector truth observations

Description: generate chosen geometry cases with the accepted forward model and hide their truth
from the fitter.

For every case:

1. run the accepted forward model at hidden truth parameters;
2. use a declared high-accuracy fixed source/mosaic revision;
3. freeze rod, reflection-group, branch, and `00L` identities;
4. store detector-native expected peak moments;
5. reserve peaks or datasets for held-out prediction; and
6. build a sanitized fit input with no truth values.

Acceptance criteria:

- [ ] Truth cases are named and prescribed; no random LHS or random geometry is used.
- [ ] Truth and sanitized fit inputs have separate hashes and provenance.
- [ ] Tier A and Tier B cases are available.

Verification:

- [ ] Run a recursive truth-token and truth-key leak audit.

Dependencies: Phase 1 and T08-T10.

## Task 2.2: compute expected detector moments

Description: replace stochastic second-stage event selection with complete-pool conditional
expectation for every locked peak.

When deposition, masks, or clipping matter, accumulate

\[
M_i=\sum_{j,p}m_jD_{jp},
\]

\[
\mathbf S_i=\sum_{j,p}m_jD_{jp}\mathbf x_p,
\]

\[
\mathbf H_i=\sum_{j,p}m_jD_{jp}\mathbf x_p\mathbf x_p^{\mathsf T},
\]

and

\[
\boldsymbol\mu_i=\frac{\mathbf S_i}{M_i}.
\]

Use continuous-hit moments only after proving equivalence for unclipped interior bilinear
deposition. Each ray block returns partial `M`, `S`, and `H` for every active observation, so the
number of peaks does not control CPU utilization.

Acceptance criteria:

- [ ] Detector moments match direct candidate enumeration.
- [ ] Edge clipping and masks use their actual deposited support.
- [ ] Zero or negligible mass produces an explicit invalid observation.

Verification:

- [ ] Compare scalar, chunked CPU, and eligible GPU moment summaries.

Dependencies: Phase 1, T08, and T09.

## Task 2.3: construct detector-native lines

Description: build line constraints from expected peak centroids, never from individual rays.

- Every nonzero-`m` group requires locked branches `0` and `1`.
- Every `m=0` group requires at least two distinct locked `00L` centroids.
- Lines are evaluated directly in detector-native coordinates.
- Increasing `L` orients the `m=0` line.
- Group membership remains frozen for the fit.

Phase 2 lines remain genuinely detector-native and never transform their points into angular space.

Acceptance criteria:

- [ ] Both line types are available simultaneously through explicit typed policies.
- [ ] Missing, swapped, nonfinite, duplicate, or degenerate lines are rejected explicitly.
- [ ] No `m=0` branch identity is fabricated.

Verification:

- [ ] Prove two-point, multi-point, reversed-input, branch-swap, low-eigengap, and zero-span cases.

Dependencies: Tasks 2.1-2.2.

## Task 2.4: assemble the detector composite objective

Description: concatenate raw detector point residuals and coordinate-unit line residuals.

```text
all delta-column and delta-row components
    + all m!=0 branch-line half-angle residuals in pixels
    + all m=0 axis-line half-angle residuals in pixels
```

Invalid topology returns a typed invalid evaluation or fixed declared penalty. It never triggers
dynamic reassignment.

Acceptance criteria:

- [ ] Raw centroid RMS/distance and both line-angle diagnostics are reported separately.
- [ ] The residual length and ordering remain fixed during one run.
- [ ] The line terms use fixed target spans and no extra line-offset component.

Verification:

- [ ] Evaluate analytic translations, rotations around the midpoint, length changes, and combined
      perturbations.

Dependencies: Tasks 2.2-2.3 and T09.

## Task 2.5: implement deterministic optimizer rungs

Description: recover increasingly coupled parameter blocks without random starts.

Rung order:

1. one active parameter at a time;
2. selected two-parameter combinations;
3. small identified parameter blocks; and
4. the complete chosen geometry block only after Jacobian rank passes.

Every rung has explicit units, bounds, transformations, parameter scales, and a predeclared start
set. Geometry-invalidating stages are recomputed; immutable states are reused. Finite-difference or
other independent trial columns may be batched only after the serial objective is proven.

Acceptance criteria:

- [ ] No random parameter proposals or starts are generated.
- [ ] Active blocks pass rank and conditioning checks before expansion.
- [ ] Bounds and invalid evaluations are reported.

Verification:

- [ ] Run nominal, signed single-parameter, chosen coupled, and held-out cases.

Dependencies: Tasks 2.1-2.4.

## Task 2.6: evaluate blind detector recovery

Description: reveal truth only after termination and record scientific and numerical evidence.

Report:

- parameter error in physical units;
- normalized parameter error relative to the allowed span;
- raw centroid RMS and maximum distance;
- nonzero-`m` branch-line angle error;
- `m=0` axis-line angle error;
- held-out peak error;
- Jacobian rank and condition;
- active bounds and correlations;
- selection audit result; and
- CPU/GPU prediction agreement.

A final point-only audit must not materially change the composite-objective solution. If it does,
the line weight, model, or active block is not accepted.

Acceptance criteria:

- [ ] Nominal and every signed single-parameter case recover within frozen tolerance.
- [ ] Chosen coupled and held-out cases pass their declared tolerances.
- [ ] The outer selection audit remains stable or creates an explicit new manifest revision.

Verification:

- [ ] Run the compact permanent recovery suite and external broader proof matrix.

Dependencies: Tasks 2.1-2.5.

## Phase 2 checkpoint

- [ ] No random geometric truth or optimizer start was used.
- [ ] Raw detector centroid error decreases.
- [ ] Nonzero-`m` and `m=0` line angles decrease.
- [ ] Held-out detector observations improve.
- [ ] Identity and topology remain stable.
- [ ] CPU and eligible GPU predictors agree.

# Phase 3: angle-space geometry fitter

## Goal

Recover the same known geometry cases from expected angle-space peak moments while preserving the
detector-proven physics and identities.

## Angle-space coordinate contract

Use one exact full-pixel-splitting detector-to-angle transform for angle-space images and fitting.
Construct the sparse splitting operator from the accepted `CompiledInstrument` physical pixel
corners, including rectangular row/column pitch and the declared detector pose. Simulation-native
hits remain in the native detector frame and receive no display rotation.

For each detector pixel, transform all four physical corners into `(2theta, chi_raw)`, unwrap its
angular footprint across the fixed azimuth seam, and distribute its signal and normalization over
the intersected angle-space bins. Sum signal and normalization separately before division. The same
bin edges, seam, mask, corrections, and validity policy serve image construction and fitting.

The current `CompiledInstrument` has no calibrated distortion or non-planar-pixel map. Distortion is
out of scope for this phase. If required later, add one typed
detector-coordinate-to-physical-corner calibration boundary before `M`; never absorb distortion
into beam center, pitch, or Euler angles.

## Task 3.1: freeze the angular coordinate contract

Description: define the exact polar transform and fitting view before using either in an objective.

Let `(t1, t2, t3)` be the vector from the nominal sample/beam origin to a detector point in the
declared beam basis: `t1` follows detector row-down on the flat reference detector, `t2` follows
column-right, and `t3` follows the direct beam. Define

\[
2\theta=\operatorname{atan2}\!\left(\sqrt{t_1^2+t_2^2},t_3\right),
\qquad
\chi_{\rm raw}=\operatorname{atan2}(t_1,t_2).
\]

The fitting/display azimuth is

\[
\phi
:=
\operatorname{wrap}_{[-\pi,\pi)}\!\left(-\frac{\pi}{2}-\chi_{\rm raw}\right).
\]

For the flat reference detector this makes `phi=0` upward, `-pi/2` rightward, `-pi` downward, and
`+pi/2` leftward. Keep `chi_raw` and `phi` distinct in names, types, axes, diagnostics, and
serialization. The angle-space array is indexed `[raw_chi_bin, two_theta_bin]`; a display reorder to
`phi` is a view, not a change to the physical transfer operator. When extracting a fitter moment,
transform the raw-`chi` axis, sort it into increasing `phi` order, and apply the identical
row permutation to `S`, `N`, `I`, and the valid mask. Transforming only the axis is invalid.

SLATE-rMC uses center-based continuous detector coordinates: integer `(c,r)` is the center of
pixel `[r,c]`, whose corners are `(c +/- 0.5, r +/- 0.5)`. Construct every angular footprint from
those physical corners. For a flat detector, pixel-center offsets are
`x=(c-C_center)*column_pitch` and `y=(r-R_center)*row_pitch`.

Freeze and record tuple order `(2theta, phi)`, radians, wrap interval, detector shape and pitch,
accepted instrument revision, nominal angle-space origin, direct-beam axis, ordered transverse basis
and their revision, angle-space bin edges and centers, seam, mask and correction policy,
projector/LUT revision, dtype, sparse engine, summation policy, local unwrap origin for every peak
and line group, and the exact observable kind. The nominal angle-space origin/basis is independent
of per-event sample intersections and cannot move with a trial. Detector calibration remains fixed
for the first angle-space fit. Azimuth at the direct beam is undefined; a low circular resultant or
direct-beam point is an explicit invalid angular observation. A detector corner exactly at the
angular origin follows one frozen polygon-seam tie policy but does not create a physical azimuth
observation.

Acceptance criteria:

- [ ] Coordinate transforms have explicit frame, unit, revision, axes, and provenance.
- [ ] The nominal angle-space origin and ordered beam basis are immutable, orthonormal, have
      explicitly tested handedness, are versioned, and remain independent of event transport
      origins.
- [ ] Flat, tilted, rectangular-pitch, and non-square-detector center/corner maps match independent
      scalar geometry and direct corner enumeration within frozen tolerance.
- [ ] Physical-corner construction produces no half-pixel offset.
- [ ] `phi` never becomes branch identity and is never mislabeled as `chi_raw`.
- [ ] Raw-`chi` to `phi` display ordering applies one identical permutation to axes, fields, and
      validity masks.

Verification:

- [ ] Prove corners, interior points, pixel centers, all four cardinal directions, wrap boundaries,
      direct-beam invalidation, and accepted reference points.
- [ ] Compare full physical-corner geometry rather than only center-angle arrays.

Dependencies: accepted Phase 2 and T15 coordinate ownership.

## Task 3.2: transform each candidate before reduction

Description: construct one exact sparse detector-to-angle operator and apply it to
candidate-specific hits before any angular reduction.

Let `D_ki` be the same bilinear deposition weight used by the detector renderer from continuous
candidate hit `i` to detector pixel `k`. Let `M_bk` be the seam-safe full-pixel-splitting weight
from the four physical corners of detector pixel `k` to angle-space bin `b`. Do not materialize
their Cartesian product. In bounded tiles, the angular footprint of candidate `i` is

\[
a_{bi}=\sum_k M_{bk}D_{ki}.
\]

SLATE-rMC's valid detector support is `[-0.5,W-0.5] x [-0.5,H-0.5]`. Apply the accepted deposition
and clipping contract over that entire support, including all four half-pixel edge strips.

Use increasing uniform bin edges and center-valued axes. Radial bins cover zero through the maximum
detector-corner `2theta`; raw-`chi` bins cover `[-pi,pi)`. Before polygon overlap, locally unwrap a
pixel's four raw-`chi` corners across the seam exactly once so the footprint remains contiguous,
then wrap bin ownership back to the frozen interval. Record boundary/tie policy because exact
seam and beam-center ties can otherwise become order- or dtype-dependent.

The authoritative order is candidate deposition, detector-signal summation, application of `M`,
normalization, and then the matched bin moment. Detector centroids remain detector-space
observables. Cache `M` by the complete instrument,
nominal angle-space origin, direct-beam/transverse basis, detector shape, physical pixel-corner
calibration, bin edges, seam, dtype, and summation-engine revisions. If a later fit varies any
parameter that changes those quantities, rebuild or select the correctly keyed operator for that
trial; a stale LUT is an invalid evaluation.

Acceptance criteria:

- [ ] Every angular moment comes from candidate contributions passed through `D` and `M`.
- [ ] `D` is bitwise/order-equivalent to accepted detector image deposition.
- [ ] `M` is a sparse full-pixel-splitting operator built from physical corners with deterministic
      seam handling and canonical bin ordering.
- [ ] For every valid unmasked pixel with full angular support, `sum_b M_bk = 1` within frozen
      tolerance; clipped axes expose `lost_support_k = 1 - sum_b M_bk` and never silently
      renormalize it.
- [ ] A nonlinear counterexample proves the full `D -> M -> S/N -> moment` order.
- [ ] The angular summary carries its projector, grid, and instrument revisions.
- [ ] Operator cache invalidation covers the angle-space origin/basis, center, distance, pitch, pose,
      physical corners, shape, bin edges, seam, dtype, and summation engine.

Verification:

- [ ] Compare direct polygon/bin enumeration, sparse-operator application, and the optimized tiled
      reducer.
- [ ] Prove integer-hit, fractional bilinear-hit, and circular-seam fixtures by direct enumeration.
- [ ] Test detector interior points and all four half-pixel edge strips.
- [ ] Prove unity-field, total-signal, and per-pixel column-sum conservation; masks remove identical
      support from `S` and `N`, and angular clipping reports the exact lost support.
- [ ] Compare `M` and angle-space fields with an independent direct full-pixel-splitting oracle.

Dependencies: Task 3.1 and the Phase 2 candidate/moment boundary.

## Task 3.3: freeze the normalized angle-space observable

Description: define the one angle-space field and moment that observation and prediction may
compare.

Construct expected detector signal

\[
s_k=\sum_i m_iD_{ki},
\]

then accumulate angle-space signal and normalization separately:

\[
S_b=\sum_kM_{bk}s_k,
\qquad
N_b=\sum_kM_{bk}n_k,
\qquad
I_b=\frac{S_b}{N_b}
\quad\text{for valid }N_b.
\]

`NormalizedAngleFieldMoment` is the authoritative synthetic target and prediction. Signal and
normalization must be summed before division. Freeze the detector mask, invalid-bin rule,
corrections, clipping, bin grid, and bin measure. For the displayed-array centroid the default
measure is one per valid bin; an angular-area or spherical measure is a different, explicitly named
observable. A constant uniform-bin factor may cancel algebraically but must remain declared.

The current centroid contract requires nonnegative `I_b` within each fitted support. This holds for
the synthetic proof before background subtraction. Experimental dark/background-subtracted
angle-space fields may contain negative bins, for which probability-like centroid weights can
cancel or leave the support. Before fitting such data, either retain/model a nonnegative background
in the observation model or approve a separately named signed-data estimator with its own oracle
and uncertainty contract. Reject incompatible signed input; never silently clip negative bins.

Acceptance criteria:

- [ ] Observation and prediction use the same named observable, grid, mask, corrections,
      normalization, and bin measure.
- [ ] Signal and normalization are accumulated separately and divided only after reduction.
- [ ] Invalid/empty normalization bins cannot contribute to a moment.
- [ ] Negative angle-space intensity is rejected by this observable unless an explicit, separately
      proven signed-data/background policy is selected.
- [ ] No normalization, detector solid angle, or angle-space measure is applied twice or silently.
- [ ] The production residual accepts only the `NormalizedAngleFieldMoment` type.

Verification:

- [ ] Compare exact `S`, `N`, `I`, valid-bin masks, and centroids with independent direct
      enumeration for square, rectangular, masked, seam-crossing, and partial-pixel cases.
- [ ] Include a nonlinear counterexample that detects an incorrect transform or division order.
- [ ] Test negative bins, signed cancellation, zero total weight, and a modeled nonnegative
      background; silent clipping must be detected.

Dependencies: Tasks 3.1-3.2 and the accepted T15 angle-space measure.

## Task 3.4: accumulate expected angular moments

Description: reproduce the local angle-space expected-moment observable with a frozen chart and an
explicit measure.

For frozen center `(t0, phi0)`, define

\[
\Delta t_q=2\theta_q-t_0,
\qquad
\Delta\phi_q=\operatorname{wrap}(\phi_q-\phi_0).
\]

Here `q` is a valid normalized angle-space bin. Use the frozen Gaussian weighting

\[
K_q
=
\exp\left[
-\frac{\Delta t_q^2+\Delta\phi_q^2}{2\sigma^2}
\right],
\qquad
w_q=I_q\,\omega_qK_q.
\]

Here `omega_q` is the frozen bin measure. Never feed raw signal `S_b` into a
`NormalizedAngleFieldMoment`.

Accumulate

\[
M=\sum_qw_q,
\qquad
\mathbf S
=
\sum_qw_q
\begin{bmatrix}
\Delta t_q\\
\Delta\phi_q
\end{bmatrix},
\]

then

\[
\widehat{2\theta}=t_0+\frac{S_0}{M},
\qquad
\widehat\phi
=
\operatorname{wrap}\left(\phi_0+\frac{S_1}{M}\right).
\]

Use `sigma=1 degree` with no hard angular membership cutoff in the proof path. If a finite Gaussian
tail cutoff is later needed for speed, freeze it before fitting and prove a bound on discarded mass
and final centroid error. Use radians in the numerical core.

Acceptance criteria:

- [ ] The normalized angle-space moment is reproduced by direct enumeration of valid bins.
- [ ] Zero/negligible effective mass and wrap ambiguity are explicit invalid states.
- [ ] Angular covariance uses wrapped local `phi` residuals.
- [ ] ROI center, Gaussian scale, unwrap origin, projector, grid, mask, normalization, and
      instrument revision cannot change during one fit.

Verification:

- [ ] Compare mass, centroid, covariance, Gaussian-tail fraction, valid-bin fraction, and wrap
      diagnostics.

Dependencies: Tasks 3.1-3.3.

## Task 3.5: make observed and predicted quantities identical

Description: use one matched moment definition and prevent target data from overwriting a fresh
trial prediction.

Generate synthetic targets with the same authoritative `NormalizedAngleFieldMoment` definition as
the prediction. Extract experimental angle-space peaks with the same ROI, normalization, bin
measure, and moment definition.

Use immutable, separate `AngleSpaceObservation` and `AngleSpacePrediction` values. The prediction
constructor accepts only fresh trial outputs and frozen measurement metadata; it cannot accept
target centroid fields. Residual assembly receives both and subtracts them explicitly.

Acceptance criteria:

- [ ] Every observation records its measure, ROI, kernel, normalization, and extraction method.
- [ ] Measurement covariance and predicted peak spread remain distinct quantities.
- [ ] A target value cannot enter, mutate, default, or overwrite a trial prediction.

Verification:

- [ ] Use intentionally asymmetric peaks to prove observation and prediction apply the identical
      extraction operator.
- [ ] Inject distinct target and trial centroids and prove the residual contains their difference,
      not zero and not two copies of either value.

Dependencies: Task 3.4.

## Task 3.6: construct angle-space lines

Description: reuse the post-centroid line reducer in the frozen local angular chart.

```text
coordinate 0: 2theta
coordinate 1: locally unwrapped phi
```

For nonzero `m`, require branches `0` and `1`, direct branch `0 -> 1`, and reject swaps. For
`m=0`, retain `branch_id=None`, require at least two distinct `00L` centroids, and orient the line
by increasing `L`.

Near the direct beam, `phi` may be poorly conditioned. A low circular resultant, inadequate line
eigengap, or excessive propagated `phi` uncertainty invalidates the line constraint.

Before optimization, freeze a per-observation point-component mask and a per-group line-component
mask from the accepted target/nominal conditioning. A branchless low-angle `00L` observation may be
radial-only from the start; its `phi` remains diagnostic and cannot appear later because a trial
moves away from the beam. A line group participates only if its target span/eigengap and member
uncertainties pass the frozen gate. If a trial loses conditioning for a required component, return
the fixed-length invalid-evaluation result defined by T09; never add or drop residual components
dynamically.

Acceptance criteria:

- [ ] The frozen local unwrap cut cannot move during optimization.
- [ ] Point and line component masks, residual slots, and conditioning thresholds are frozen before
      the first trial.
- [ ] Both line types can participate simultaneously.
- [ ] No angle-display side label becomes physical branch identity.

Verification:

- [ ] Test wrap-crossing, reversed inputs, branch swap, low-angle `00L`, and degenerate spans.

Dependencies: Tasks 3.1-3.5.

## Task 3.7: assemble the angle-space composite objective

Description: concatenate angular point components and angular-unit line residuals.

```text
all delta-2theta and wrapped delta-phi point components
    + all m!=0 branch-line half-angle residuals
    + all m=0 axis-line half-angle residuals
```

All quantities use radians internally, and the fixed target angular span gives each line term
angular coordinate units. For branchless low-angle `00L`, individual `phi` is diagnostic-only
only when the frozen preparation mask says so; the collective line is used only when its frozen
group gate is accepted.

Acceptance criteria:

- [ ] Residual ordering and length are fixed during a run.
- [ ] Trial-dependent singularity can invalidate an evaluation but cannot alter the frozen residual
      schema.
- [ ] Raw angular centroid and line-angle metrics are reported separately.
- [ ] Invalid topology cannot trigger reassignment.

Verification:

- [ ] Repeat the analytic translation, midpoint rotation, length, and wrap tests from Phase 2.

Dependencies: Tasks 3.4-3.6 and T09 objective contracts.

## Task 3.8: replay deterministic blind recovery

Description: run the exact Phase 2 truth matrix from two prescribed angle-space starts.

1. Start from the same blinded nominal initial state used in detector fitting.
2. Start from the accepted detector-space result to prove staged dovetailing.

Compare recovered parameters, angular centroid residuals, nonzero-`m` line angles, `m=0` line
angles, held-out detector predictions, and detector-versus-angle-space parameter differences.

Acceptance criteria:

- [ ] Detector-space and angle-space fits recover the same chosen truths within frozen tolerance.
- [ ] Angle-space fitting cannot alter accepted detector calibration or selection identity.
- [ ] Held-out detector-native predictions remain correct.

Verification:

- [ ] Run nominal, signed, coupled, difficult, and held-out Tier A/Tier B cases.

Dependencies: Tasks 3.1-3.7 and accepted Phase 2 recovery matrix.

## Task 3.9: evaluate GPU angular fitting

Description: reuse the Phase 1 device-resident candidate kernel, the fixed sparse splitting
operator, and deterministic signal/normalization/moment reductions.

For the first angle-space fit, keep `M`, bin centers, masks, normalization state, and immutable
simulation state device-resident across optimizer evaluations. Stream only bounded
candidate/deposition tiles and the small residual vector. Batch independent trial points or
finite-difference/Jacobian columns when memory permits. Do not build a giant
candidate-by-angle-bin matrix.

Because detector calibration, masks, corrections, grid, and peak supports are frozen, precompute
`N` once and derive a CSR row view containing only the union of active angle-space fit bins. Retain
the full-field proof path, and permit the active-row path only after it reproduces the same `S`,
`I`, moments, and residuals. This avoids recomputing irrelevant angle-space regions on every trial
while preserving the frozen angle-space measure.

The initial fit freezes detector calibration and the nominal angle-space origin/basis, so `M` is
constant. If a later fit varies that origin/basis, distance, beam center, detector pitch/pose,
physical corners, shape, or another splitting-transform parameter, that trial must use a newly
built or correctly keyed `M`. Benchmark that rebuild cost separately; never silently reuse the
frozen operator.

Acceptance criteria:

- [ ] CPU and GPU angular moments agree within frozen tolerance.
- [ ] CPU and GPU `S`, `N`, valid masks, `I`, centroids, and residuals agree stage by stage.
- [ ] Transfer, compile, and repeated-evaluation times are reported separately.
- [ ] GPU use is selected only above the measured crossover.
- [ ] Sparse operator memory and rebuild cost remain within the declared fit budget.

Verification:

- [ ] Benchmark repeated parameter batches, finite-difference/Jacobian columns, cached-operator
      evaluations, and forced operator rebuilds.

Dependencies: Phase 1 GPU proof and Tasks 3.1-3.8.

## Phase 3 checkpoint

- [ ] Every candidate is transformed before angular reduction.
- [ ] `chi_raw` and fitting/display `phi` are never conflated.
- [ ] The center/edge adapter produces no half-pixel shift or downstream display rotation.
- [ ] The nominal angle-space origin/basis and component masks remain frozen, keyed, and independent
      of trial event origins.
- [ ] Full-support `M` conserves signal and any clipped support is explicit.
- [ ] Observed and predicted peaks share the `NormalizedAngleFieldMoment` contract.
- [ ] Detector-space and angle-space fits recover the same chosen truths.
- [ ] Nonzero-`m` and sufficiently conditioned `m=0` line angles converge.
- [ ] `phi` wrapping, bin axes, local unwrap origins, and LUT revision remain fixed during each
      run.
- [ ] Exact normalized angle-space fields match independent direct enumeration within frozen
      tolerance.
- [ ] CPU and eligible GPU paths agree.

## Cross-phase proof matrix

Every relevant implementation path must report:

| Workload | Required comparison |
|---|---|
| Tiny scalar | analytic identities and direct enumeration |
| CPU tiled | scalar proof path |
| CPU parallel | serial tiled path under equivalent work |
| GPU | accepted CPU path, including boundary classifications |
| Detector moments | direct candidate sums and stochastic convergence |
| Detector fit | hidden known truth and held-out detector observations |
| Angular moments | direct sparse-transfer and normalized-bin enumeration |
| Angle-space fit | same hidden truth used by detector-space fit |
| Binned angle field | independent direct `S`, `N`, valid mask, `I`, axes, and centroid |

Record wall time, peak memory, setup/compile time, transfer time where applicable, candidate and
selected counts, precision, hardware, code/data/configuration revisions, and error versus the
accepted reference.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CPU parallelism is limited by Python overhead | High | Localize/vectorize first; benchmark fused execution before selecting workers |
| GPU is slow for branch-heavy float64 work | High | Record crossover and retain CPU production path |
| Candidate order changes with chunking | Critical | Canonical keyed ordering and deterministic merge |
| Parallel reduction changes complete-pool selection | Critical | Stable compaction, scan, and reduction order; proof hashes |
| Monte Carlo noise destabilizes geometry | High | Fixed samples and conditional expected moments |
| Line residual double-counts centroid data | Medium | Label it dependent guidance, use coordinate-unit scaling, freeze weight, and run point-only audit |
| Trial collapses a line to evade angle penalty | High | Use fixed target span in the half-angle residual |
| m=0 observations acquire fake branches | Critical | Separate typed `COLLAPSED_00L` line contract |
| `chi_raw` is mislabeled as fitting/display `phi` | Critical | Separate names/types and cardinal-direction fixtures |
| Pixel-coordinate handling adds a half-pixel shift | Critical | Direct physical-corner construction and scalar geometry tests |
| RA interior support is imposed on valid SLATE edge strips | High | Domain-specific classification and four edge-strip fixtures |
| A GUI/OSC rotation is applied during detector-to-angle transfer | Critical | Native-frame-only operator; no downstream rotations |
| Angle-space `phi` crosses a moving wrap cut | High | Frozen local unwrap origin per observation and group |
| A stale angle-space LUT is reused after transform geometry changes | Critical | Complete immutable cache key or explicit rebuild per trial |
| Nominal angle-space origin/basis follows per-event trial geometry | Critical | Freeze and version the nominal origin/basis independently |
| Direct-beam azimuth is treated as physical data | High | Undefined/low-resultant invalidation; radial-only `00L` handling |
| Trial conditioning changes residual length | Critical | Freeze point/line component masks; fixed-schema invalid evaluation |
| Target centroid overwrites a trial prediction | Critical | Immutable observation/prediction types and injected-value regression |
| Synthetic recovery is an inverse crime | High | Add different-sample Tier B and held-out peaks |
| Truth leaks into optimizer state | Critical | Separate truth process/artifact and sanitized-state audit |
| A normalized angle-space field is mistaken for raw event mass | Critical | Explicit signal/normalization contract and equivalence proof |
| Signed background-subtracted bins are used as centroid mass | High | Reject or use a separately proven background/signed-data model; never clip |
| Unsupported detector distortion is hidden in rigid geometry | High | Declare out of scope or add a typed physical-corner calibration boundary |

## Permanent proof and cleanup policy

- Retain only compact tests protecting unique contracts: once-only mass, deterministic selection,
  candidate order, detector moments, line grouping, angular wrapping, and one representative blind
  recovery per distinct long-term failure mode.
- Keep broad truth matrices, large images, GPU sweeps, convergence studies, and profiling output as
  external proof artifacts.
- Remove temporary benchmarks, exploratory scripts, generated dumps, redundant tests, and unused
  backend experiments before each phase handoff.
- Do not retain a GPU dependency unless it wins a declared workload or is necessary for an accepted
  later fitting workload.

## Final completion gate

- [ ] One authoritative candidate-physics implementation serves all paths.
- [ ] Individual images use full CPU capacity through ray/candidate work.
- [ ] GPU acceleration is available only where measured and scientifically equivalent.
- [ ] Detector expected moments and detector blind recovery pass first.
- [ ] Exact normalized angle-space expected moments and angle-space blind recovery pass second.
- [ ] Chosen known geometric combinations recover blindly without random truth or starts.
- [ ] Raw centroid distance, nonzero-`m` branch-line angle, and `m=0` line angle all participate as
      declared.
- [ ] Detector-space and angle-space fits agree on held-out detector-native predictions.
- [ ] The sparse detector-to-angle operator and normalized fields reproduce independent direct
      enumeration at their named stages.
- [ ] Branch, rod, reflection-group, frame, unit, measure, normalization, and revision provenance
      remain explicit.
- [ ] Every optimized path reproduces the proof path within frozen tolerance.
