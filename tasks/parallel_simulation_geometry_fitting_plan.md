# CPU/GPU-compatible simulation, representative detector labels, and staged geometry fitting plan

Status: PROPOSED

Entry gate: BKI-15 first branches into the corrected Task 1.1 scalar reference and continuous-
coating validation. Task 1.2 begins only after both branches are accepted and the T07 detector-
native observable is proven.

This plan records the agreed path for:

1. making the detector simulation efficient on the CPU and compatible with measured GPU
   acceleration;
2. placing one stable zero-mass `(branch, m, L)` representative label for every coating component
   with detector support;
3. recovering known geometry blindly from detector-native expected peak centroids; and
4. recovering the same geometry from expected angle-space `(2theta, phi)` peak centroids.

The work must preserve one authoritative physics path. CPU rendering, GPU execution,
detector-space fitting, and angle-space fitting are consumers of the same continuous coating
measure, sampled event masses, and event-specific detector hits; none may implement alternate
scattering physics.

## Governing constraints

- Complete and prove the detector-native simulation before fitting.
- Prove detector-native fitting before adding angle-space fitting.
- Parallelize bounded bulk batches of incident states, continuous-component integration records,
  and sampled detector events, not the number of selected peaks, rods, or HKL groups.
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
- Keep representative detector labels separate from Monte Carlo photon mass. They are computed
  from one nominal incident ray, never deposited, and never require retained provenance for every
  sampled event.
- Generate and hash the complete canonical source realization once before Stage-A dispatch.
  Workers receive canonical row-index views plus inherited parent revisions and never regenerate
  or rehash source slices.
- `geometry/transport.py` is owned by the BKI remediation through BKI-15 and is frozen again after
  that gate; later edits require a separately reviewed scientific need.
- Keep raw azimuth `chi_raw` and fitting/display azimuth `phi` as separately named quantities.
- Select CPU/GPU technology only after the accepted integrated workload is profiled, as required
  by [the performance strategy](../docs/PERFORMANCE.md).

## Continuous-coating compatibility amendment

### Problem and decision

How might the complete forward and fitting workload use the continuous mosaic--Ewald coating
without discarding the accepted transport, detector, RNG, and angle-space work or creating a
second simulation architecture?

Use one narrow pivot: replace the finite orientation/candidate-pool work unit in Tasks 1.2--1.7
with the accepted continuous component measure from
[the coating replacement plan](continuous_ewald_coating_replacement_plan.md), evaluated as a
bounded staged dataflow. Batch active incident-state/component/node records together, then flatten
all sampled draws into large event batches for exact transport, detector projection, and
deposition. This plan remains `PROPOSED`; the amendment does not promote either implementation.

“Parallelize the whole thing” means that every expensive numerical stage is eligible for bounded
bulk execution: incident transport, coating root and density evaluation, exact rod strength,
outgoing optics, continuous sampling, sampled-event transport, detector projection, deposition or
moment partials, representative-label component evaluation, and downstream sparse angle
projection. Small control
operations may remain serial where required for correctness: deterministic adaptive refinement
decisions, prefix construction, canonical segmented reduction, and optimizer coordination.

### Smallest useful scope

1. Accept the continuous coating cutover and its scalar oracle before defining the optimized batch
   boundary. The accepted finite-pool implementation remains temporary comparison evidence, not a
   future runtime interface.
2. Stage A batches work from the self-contained incident batch accepted after BKI-08. One logical controller
   owns each `(incident_state, coating_component)` mass/error/CDF, while pure density rows may be
   packed across states/components when profiling justifies it. This batches the beam sample; it
   does not replace it with a different global incident-state sampler.
3. Stage B generates canonical state/draw samples with physical-keyed randomness, flattens them into
   bounded exact-event-to-detector batches, and reduces contributions in stable state/draw or
   pixel/event order. Coating mass is detector-unconditioned: outgoing propagation and optical
   factors apply once before selection, while detector projection occurs after exact sampled `kf`.
   Detector misses are retained as rejected mass and are never resampled or used to renormalize
   survivors.
4. Exit transport, deposition, and T18 stay unchanged consumers. The initial nonzero-`m` coating is
   only an intermediate milestone; full completion still requires an accepted physical `m=0`
   support contract, with no legacy fallback or arbitrary epsilon.
5. A compact representative-label pass uses one nominal incident ray and returns at most one
   zero-mass detector label per exact `(family_id, intersection_branch_id)`. It reuses Stage A/B
   equations and execution seams without retaining per-photon branch, `m`, or `L` after deposition.

### Objectives

- Preserve one continuous mosaic measure and one implementation of every physics equation.
- Scale one forward evaluation across all available incident states, component-node records, and
  sampled events even when few peaks are selected.
- Bound memory by the active state tile, integration-record batch, compact component/CDF state, and
  sampled-event batch, never by a retained ray-by-rod-by-orientation product.
- Make stochastic images and deterministic detector/angle moments invariant to execution layout.
- Make representative label coordinates and `(branch, m, L)` values invariant to Monte Carlo seed,
  draw count, batch layout, and worker completion order.
- Keep GPU work limited to measured regular batches while retaining the same scalar oracle and
  robust boundary classification.

### Non-goals

- Do not retain or generalize `MosaicOrientationBatch`, `AttemptBatch`, `CandidatePool`, or a finite
  state-to-rod-to-orientation execution contract.
- Do not add a backend framework, scheduler API, plugin registry, compatibility facade, permanent
  old/new switch, or second coating implementation.
- Do not launch one task per draw, rod, adaptive panel, or fitted peak, and do not create nested CPU
  pools. Parallelism comes from contiguous stage batches.
- Do not move, duplicate, approximate, or re-normalize source, mosaic, coarea, structure, optical,
  assigned-mass, or deposition factors. Detector validity is a post-sampling classification, not a
  coating factor.
- Do not change the accepted detector-native observable, T18 projector/profile contract, fitting
  residuals, or physical branch convention.
- Do not require a GPU when the measured crossover favors the CPU.
- Do not add a per-pixel tag raster, retain provenance for every sampled ray, group by continuous
  `L`, or let a diagnostic label ray contribute detector mass.

### Affected files

This planning change edits this file plus one conflicting execution sentence in
`continuous_ewald_coating_replacement_plan.md`. A later implementation slice is expected to touch
the smallest subset of:

- `src/rasim_next/pipeline/simulate.py` for staged batch orchestration and canonical reduction;
- `src/rasim_next/reciprocal/coating.py` for bounded component-point evaluation, without executor
  ownership;
- `src/rasim_next/sampling/mosaic.py` only for the accepted continuous density contract;
- `src/rasim_next/pipeline/proof.py` for compact equivalence and performance evidence;
- `tests/test_mosaic_ewald.py` and `tests/test_integration.py` for unique permanent invariants.
- `scripts/generate_bi2se3_detector_image.py` for the one retained annotated default figure.

After BKI-15, `geometry/transport.py` is a frozen consumer. Task 1.1 still owns
`render/deposition.py` and `measurement/angle_space.py`; those two files freeze only after Task 1.1
acceptance. Any later edit to a frozen consumer requires separate profiling evidence and scope
approval.

No new production module is planned for parallel execution. A new file requires evidence that the
existing pipeline orchestration cannot remain cohesive.

### Interfaces

The implementation freezes only four narrow internal seams:

1. **Continuous density batch:** contiguous `(state, component, node, alpha, beta)` work records
   plus canonical ragged-rod offsets enter the accepted coating equations and return node-level
   exact roots, rod-summed strengths, once-only source/reciprocal/outgoing factors, physical
   validity, and detector-unconditioned density in the same row order. It owns no scheduler state.
2. **Component state:** canonical density results enter one component's private adaptive state;
   component mass/error/CDF support exits under a complete revision key. It is deterministic
   orchestration, not a backend or executor interface.
3. **Sampled-event batch:** canonical `(state, draw)` records enter exact event reconstruction,
   transport, and detector projection; compact event geometry, detector contributions, and
   separately aligned assigned mass exit in the same row order.
4. **Representative-label batch:** canonical `(family, intersection branch)` rows use one nominal
   incident ray to find the detector-valid continuous orientation nearest the mosaic mode. Exact
   branch `0/1/2`, displayed `m`, `L`, mosaic coordinates, and detector coordinates exit in the
   same row order with no assigned photon mass.
Stochastic deposition, deterministic moments, and T18 consume the merged contributions without
calling coating equations or reinterpreting identities. The accepted ordered/stacking intensity
seam remains unchanged and receives exact event-aligned `q`/`L`.

The existing canonical incident-state batch remains the input to Stage A. “All `ki` together”
means vectorizing/batching that accepted row set and preserving its identities, source weights,
refraction, attenuation, validity, and order; it does not introduce another public beam contract.

### Tests and proof cadence

- A one-state tiny proof compares scalar and packed density/CDF/event rows, once-only factors,
  exact identities, per-state `T_i/N_draw,i`, and detector contributions.
- A 33-state checkpoint permutes packet/worker layouts and compares component ledgers, RNG draws,
  images, deterministic `M/S/H`, and T18 outputs under predeclared exact/tolerance policies.
- One clean 129-ray/full-catalog milestone records equivalent work, wall time, utilization, and peak
  memory. Broad sweeps and convergence grids remain external proof artifacts.
- One nominal-ray proof checks exact center and nearest-supported labels for branches `0`, `1`, and
  `2`, unchanged raw detector pixels, and serial/parallel label equality.

### Amendment acceptance criteria

- [ ] The continuous coating is the sole production path; serial and packed layouts preserve exact
      identities/order and reproduce mass, draws, detector/T18 outputs, and moments under frozen
      policies.
- [ ] Both expensive stages scale under bounded memory, while deterministic control/reduction stays
      execution-order invariant and a GPU is retained only above its measured crossover.
- [ ] Independent continuous integrals and stochastic convergence prove detector and angle moments.
- [ ] Full completion includes physical `m=0` and leaves no duplicate physics, executor framework,
      compatibility facade, or unnecessary production module.
- [ ] Every detector-supported coating component has one stable zero-mass `(branch, m, L)` label;
      unsupported components have no fabricated detector coordinate, and the raw image is
      unchanged.

### Directions considered

| Direction | Decision | Reason |
|---|---|---|
| One end-to-end worker per ray block | Retain only as baseline/fallback | Simple ownership, but scaling is capped by ray count and adaptive tails fragment SIMD/GPU work |
| Per-node/draw tasks or interchangeable backends | Reject | Scheduling overhead and duplicate lifecycle/invalidation surfaces add bloat and nondeterminism |
| Pivot Task 1.3 to staged continuous work records after coating acceptance | Choose | It batches all `ki`/component math and all sampled detector events while preserving the useful physics seams |

## Dependency graph

```text
BKI-15 accepted incident boundary
    |-> corrected Task 1.1 stitch/T07 scalar reference -----------\
    `-> accepted continuous-coating validation -------------------> Task 1.2 profiled work reduction
    -> profiled and optimized continuous component work count
    -> narrow staged continuous batch boundary
    -> CPU bulk state/component/event execution
    -> stochastic rendering and deterministic continuous moments
    -> measured eligible GPU proof path
    -> nonzero-m parallel milestone --------------------\
accepted physical m=0 contract and implementation ------> full Phase 1 checkpoint
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

Task 1.1 and coating validation branch independently from BKI-15 and join at Task 1.2. The rest of
Phase 1 follows [T07](07_integration.md), the accepted continuous-coating cutover, and a
performance-focused change after the replacement workload is re-profiled. Phase 2 dovetails with
[T09](09_fit_foundation.md), [T10](10_instrument_calibration.md), and
[T11](11_sample_geometry_fit.md). Phase 3 consumes the accepted finite-bin detector-to-angle
producer; its owned fitting paths must be approved before implementation.

## Shared simulation boundary

Every execution mode consumes the same authoritative sequence:

```text
fixed source rays
    -> continuous family/root coating components
    -> exact per-rod detector-unconditioned density with once-only outgoing factors
    -> component mass, error, and continuous CDF support
    -> exact sampled rod/root/mosaic event with assigned T_i/N_draw,i mass
    -> event-aligned outgoing wave and continuous detector hit
```

The complete continuous measure includes every applicable source, phase/parent,
reciprocal/mosaic/coarea, scattering-strength, optical, polarization, attenuation, population, and
outgoing-propagation factor exactly once. Detector projection is post-sampling and detector misses
are retained as `detector_rejected_mass_A2`; neither detector validity nor detector solid angle is
part of the raw coating mass. Recomputing exact sampled geometry or a detector hit does not reapply
a factor already represented in the draw probability or assigned mass.

From this boundary, explicit consumers may:

- sample stochastic outgoing events and render an image;
- integrate deterministic detector expected moments over the same measure; or
- pass detector contributions to T18 and consume its deterministic normalized angle-space profile
  and matched expected moments.

No fitting module may reimplement source sampling, continuous coating support, Ewald geometry,
optical transport, detector intersection, or event mass.

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

Dependencies: accepted BKI-15 incident boundary and stitch work. Continuous-coating validation
runs as the sibling BKI-15 branch and does not gate this scalar-reference task.

Likely owned paths: T07 `measurement/`, `render/`, `pipeline/`, and `tests/test_integration.py`.

## Task 1.2: reduce mathematical work

Description: retain the accepted finite-pool result only as a comparison oracle, then profile the
accepted continuous coating and remove avoidable work before adding executors or a GPU path.

Required order:

1. compile instrument transforms once;
2. compile family/rod metadata and invariant reciprocal geometry once;
3. localize continuous component support and split declared physical boundaries;
4. batch exact component-density evaluations in bounded point tiles;
5. reuse immutable root/transport node data only under a complete revision key;
6. retain one state/component CDF at a time instead of a Cartesian product; and
7. profile root, strength, optics, post-sampling detector projection/rejection, integration,
   sampling, deposition, and moment work separately.

Acceptance criteria:

- [ ] The accepted scalar coating integral and the temporary finite-pool comparison remain proof
      evidence until coating deletion authorization; neither becomes a second production path.
- [ ] Localized support and bounded point tiles reproduce the accepted continuous observable within
      frozen tolerance.
- [ ] Peak memory is bounded for the large-forward benchmark.

Verification:

- [ ] Compare equivalent density evaluations, component masses/errors, sampled events, wall time,
      peak memory, and detector output.

Dependencies: the BKI-15 join of accepted Task 1.1 and accepted continuous-coating validation.

## Task 1.3: define the narrow staged numeric boundaries

Description: expose two row-stable numeric batches for the measured continuous hot path without
creating a general backend framework.

```text
density batch input rows:
    incident-state identity and compiled ki state
    family/coating-component identity
    canonical adaptive-node identity and continuous alpha/beta
    phase/parent, exact ragged rod catalog/offsets, and accepted intensity state/seam
    compiled instrument/material state and revision envelope

density batch output rows:
    exact root/L/q/kf and once-only factor ledger
    validity/exclusion/first-failure
    detector-unconditioned density with outgoing propagation/optical validity

sampled-event batch input rows:
    canonical state/draw identity
    selected component and continuous alpha/beta
    physical-keyed rod-choice uniform and exact ragged rod catalog/offsets
    separately aligned assigned draw mass

sampled-event batch output rows:
    exact sampled event geometry and mosaic coordinates
    exact conditional rod identity and contribution
    algebraic-side, intersection-root, coating-component, and physical group provenance
    event-aligned outgoing state and continuous detector hit/contribution
```

The algebraic sphere side, continuous-rod `intersection_root_id`, stable coating-component identity,
and physical fitting `BranchKey.branch_id` are separately typed and never numerically
interchangeable. Tangent algebraic sides coalesce under the accepted boundary policy. After its
physical support is accepted, `m=0` uses its declared intersection root while the selection manifest
retains `branch_id=None/COLLAPSED_00L`. The accepted selection layer derives nonzero physical branch
from exact event geometry and freezes it in its own manifest. Direct, tangent, or numerically
uncertain rows retain the accepted scalar robust classification. Both batches preserve input row
order exactly. Adaptive CDFs/panel trees and executor/device objects remain private orchestration
state and never enter either numeric contract.

Invalidation is explicit:

| Change | Minimum invalidation |
|---|---|
| sample/incident geometry or material/instrument optics | roots, support, outgoing state, affected factors, masses, CDFs |
| detector placement/shape | detector hits, representative labels, deposition/moments; never coating masses/CDFs |
| continuous mosaic parameters/measure | component support, masses, errors, CDFs |
| intensity model/structure | strengths, masses, errors, CDFs; reuse geometry only under its exact revision |
| source/phase/population/optics | every affected once-only factor, mass, and CDF |
| detector-to-angle grid/projector/ROI | downstream T18 projection/profile only; never coating physics |

Acceptance criteria:

- [ ] Scalar, diagnostic, stochastic, and deterministic-moment consumers call the same continuous
      density/root/intensity/transport equations.
- [ ] Canonical component order is state -> family -> coating component; canonical sampled-event
      order is state -> draw. No orientation-node identity enters either order.
- [ ] Density results scatter into canonical component/node slots without depending on how rows were
      batched or where they executed.
- [ ] The sampled-event batch evaluates exact per-rod conditional strengths, performs the keyed rod
      choice, and reconstructs the chosen exact event without a per-draw task.
- [ ] Assigned draw mass is separate from sampled geometry and cannot be reapplied as a reciprocal
      or mosaic weight.
- [ ] Existing incident-state batch and scalar construction agree on identity, `ki`, source weight,
      incident refraction/attenuation, validity, and row order.
- [ ] No public generic backend or arbitrary callback layer is added.

Verification:

- [ ] Compare staged batches and scalar execution for row alignment, component masses/errors,
      exclusions, roots, sampled coordinates, conditional rod frequencies, assigned masses, hits,
      identities, and first divergence.

Dependencies: Task 1.2.

## Task 1.4: implement staged CPU bulk execution

Description: benchmark serial staged batches before adding the smallest justified fused/parallel
CPU execution.

Benchmark order:

1. scalar reference;
2. serial vectorized density and sampled-event batches;
3. the smallest CPU path: bounded complete-component jobs plus one flattened sampled-event path;
4. cross-component adaptive-frontier coalescing only if density integration remains dominant;
5. fused compiled CPU execution only if profiling justifies it; and
6. selected worker and batch sizes under equivalent work.

The first CPU implementation schedules a bounded set of complete components drawn from many fixed
incident states. After convergence, it generates draws in canonical `(state, draw)` order and
flattens them into large exact-event, transport, detector-projection, and deposition batches. Fixed
event tiles or stable segmented pixel records reduce in canonical order. If density integration
remains dominant, component controllers may coalesce pending node rows into shared batches and
scatter results back by canonical `(state, component, node)` identity. There is no task per ray,
draw, rod, node, or adaptive panel; workers never mutate shared CDFs or images, and no unordered
atomic reduction is allowed.

Acceptance criteria:

- [ ] CPU utilization scales across the available state/component/node and sampled-event rows even
      when one or two fitted peaks are active.
- [ ] State-tile, density-batch, event-batch, worker count, and completion order do not change
      component ledgers, sampled identities/coordinates, assigned mass, detector output, or
      expected moments beyond the declared exact/tolerance policy.
- [ ] Peak memory remains under the declared bound for active component controllers/CDFs plus fixed
      density and sampled-event batches and outputs.
- [ ] No task-level parallelism remains inside a batch when vectorized/compiled execution already
      saturates the eligible hardware.

Verification:

- [ ] Use a one-ray inner proof, a 33-ray scaling checkpoint, and one clean 129-ray/full-catalog
      milestone benchmark; measure density/event batch sizes and broad worker/tile sweeps only as
      external artifacts.

Dependencies: Task 1.3.

## Task 1.5: preserve stochastic semantics

Description: key continuous draws to physical identity instead of execution or adaptive order.

The component-choice uniform is keyed by seed, source/incident-state identity, phase/parent, draw
index, purpose tag, and measure/algorithm revision. Conditional alpha, beta, and rod uniforms add
the already selected physical family and coating-component identities. No key depends on
state-tile size, density/event-batch size, worker count, adaptive nodes/panels, completion order,
or backend.

Acceptance criteria:

- [ ] Fixed-seed rendering is invariant to state/density/event batching and CPU worker count.
- [ ] Across CPU/GPU, host-controlled scans and boundary arbitration preserve discrete component,
      root, rod, and draw identities; floating coordinates and observables use frozen tolerances.
- [ ] Component-mass order, continuous inverse-CDF results, and conditional rod draws are stable.
- [ ] A changed sample revision changes provenance explicitly.

Verification:

- [ ] Hash component ledgers, sampled physical identities/coordinates, assigned masses, and final
      pixels for multiple execution layouts.

Dependencies: Tasks 1.3-1.4.

## Task 1.6: expose stochastic rendering and deterministic moments

Description: feed two explicit consumers from the same continuous component measure.

The stochastic renderer samples the complete per-state continuous measure. For state `i`, freeze
integrated total mass `T_i` and draw count `N_draw,i`; every draw receives
`m_id = T_i/N_draw,i`. `N_draw,i` is unrelated to T18's angle-bin normalization field `N_b`.
For a valid positive-mass state, `N_draw,i` must be positive. A zero-support state remains in the
canonical ledger with `T_i=0`, no draws, and a typed invalid/excluded status; no division or silent
row removal occurs. Conservative detector deposition follows.

The deterministic reducer integrates

\[
M=\int d\mu(\xi),
\qquad
\mathbf S=\int \mathbf x(\xi)\,d\mu(\xi),
\qquad
\mathbf H=\int \mathbf x(\xi)\mathbf x(\xi)^{\mathsf T}\,d\mu(\xi).
\]

Each deterministic state tile returns canonical per-state `M`, `S`, and `H` partials; the parent
reduces states in canonical order. This reducer establishes simulation capability; it is not yet a
fitter.

Adaptive acceptance is observable-specific. Stochastic CDF construction declares a mass/CDF error
policy. Detector moments refine the vector integrand `[M, S, H]` until every scaled component meets
its frozen tolerance. A deterministic detector image or T18 profile similarly declares error for
the required detector or active finite-bin `S` contributions. Converged component mass alone never
certifies moment, image, or profile convergence.

Acceptance criteria:

- [ ] Moments equal an independently converged direct continuous integral within frozen integration
      and observable tolerances.
- [ ] High-count stochastic image centroids converge to expected moments.
- [ ] For each incident state, `sum_d m_id = T_i`, and deposited plus edge-clipped plus
      detector-rejected mass equals the assigned sum. Detector misses are retained and never
      resampled; an outgoing-propagation classification disagreement is a typed consistency
      failure.
- [ ] The ledger keeps integrated exclusions, detector consistency failures, edge clipping, and
      deposited mass as separate fields/statuses; none is silently folded into another.
- [ ] Nonzero-`m`, accepted `m=0`, and combined continuous measures are each proven before full
      Phase 1 completion.

Verification:

- [ ] Compare mass, first moment, second moment, centroid, and convergence rate.

Dependencies: Tasks 1.1-1.5.

## Task 1.7: add and evaluate GPU execution

Description: prototype only sufficiently large regular component-point, transport, deposition, or
moment batches identified by CPU profiling.

Requirements:

- keep immutable compiled states resident during repeated work;
- use float64 for proof;
- keep adaptive refinement, robust fallback, and canonical merge under deterministic host control;
- keep component scans, inverse-CDF decisions, rod choice, and all discrete identity decisions on
  the host;
- recompute a GPU row on the CPU whenever its error interval can change validity, refinement,
  compaction, scan ownership, inverse-CDF selection, or rod choice;
- avoid unordered atomic reductions;
- preserve stable compaction and reduction order;
- record compile, transfer, execution, and reduction time separately; and
- define a measured CPU/GPU crossover.

The GPU may evaluate the same regular density and sampled-event rows while the host owns adaptive
control and deterministic reductions. It receives large work-record arrays gathered across many
`ki` and components, not one launch per component or event. The production path may remain
CPU-based when strict-float64 branch-heavy geometry is faster on the CPU. GPU compatibility means
the same continuous work contracts reproduce the accepted observable; it does not require forcing
every workload onto the GPU.

Acceptance criteria:

- [ ] GPU and CPU reproduce the proof observable within frozen tolerance.
- [ ] GPU batching cannot change a discrete identity; decision-boundary rows use the CPU result.
- [ ] Near-boundary cases use the declared robust fallback or satisfy the same classification.
- [ ] A measured crossover determines automatic or explicit backend choice.

Verification:

- [ ] Run small proof, medium forward, large forward, and repeated geometry-fit benchmarks.

Dependencies: Tasks 1.3-1.6.

## Task 1.8: add stable representative detector labels

Description: add one deterministic, zero-mass label for every exact coating component
`(family_id, intersection_branch_id)` that has detector support. This is a compact diagnostic pass
over the accepted continuous coating, not photon provenance and not another scattering path.

Entry gate: the continuous Ewald-coating replacement must first be accepted with its final branch
`0/1/2`, LINE_M0 support, exact event reconstruction, and production replay contracts. This task
consumes those contracts and must not depend on temporary adaptive nodes, CDF storage layout, root
batches, or a finite candidate-pool oracle.

Frozen rules:

- Group by exact canonical `(family_id, intersection_branch_id)`, never floating `m` or `L`.
- Preserve branch `0` for the retained catalog-derived LINE_M0 root, branch `1` for the lower-`L`
  regular nonzero root, and branch `2` for the upper-`L` regular nonzero root.
- Store exact representative `L` as label payload; do not use it as identity.
- Construct one separate nominal incident ray from the declared mean beam position, mean direction,
  and mean wavelength. Never append it to the Monte Carlo source batch.
- Try the exact mosaic mode first. If it is not detector-valid, choose the nearest physically
  supported, detector-valid continuous orientation under the accepted coating and detector
  contracts.
- Resolve exact distance ties by greater coating density, then canonical wrapped azimuth, then
  canonical component identity.
- Emit no row for a component without detector support; keep its typed absence in proof output.
- Label rays carry no assigned mass and never enter deposition, edge-clipped mass, or
  detector-rejected photon mass.
- Reuse the accepted coating, structure-factor, outgoing-transport, detector, batch, and parallel
  implementations. Add no executor, optimizer framework, tag raster, retained event pool, or new
  production module.

### Task 1.8.1: freeze the compact label contract

Files likely touched:

- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`

Exact intended behavior: define the smallest aligned immutable batch with canonical rows and these
fields:

```text
component_id[K]
family_id[K]
m_value[K]
intersection_branch_id[K]
l_coordinate[K]
mosaic_alpha_rad[K]
mosaic_azimuth_rad[K]
detector_column_px[K]
detector_row_px[K]
```

Arrays are aligned, contiguous, read-only, and finite. `(family_id, branch)` is unique; branch is
restricted to `0/1/2`; rows use canonical family-then-branch order. No mass or deposition-weight
field is permitted, and missing components use typed omission rather than sentinel coordinates.

Tests: extend `tests/test_integration.py` with one contract test covering field alignment,
uniqueness, branch range, finite coordinates, canonical order, and absence of photon-mass semantics.

Dependencies: accepted coating boundary and Task 1.3.

Acceptance criteria: one row means one detector-supported component, identity is explicit rather
than positional, and no new module or test file is added.

### Task 1.8.2: construct the nominal incident ray

Files likely touched:

- `scripts/generate_bi2se3_detector_image.py`
- `tests/test_integration.py`

Exact intended behavior: construct a separate one-row nominal `IncidentSampleBatch` from the
script's declared mean source position, direction, and wavelength, with wavelength-matched material
optics. Reuse an existing exact central-ray constructor where possible. Do not change source
weights, Monte Carlo sample count, seed mapping, random sequence, or detector image.

Tests: different seeds produce the same nominal ray; origin/direction/wavelength equal their
declared means; the ordinary Monte Carlo batch and source weights remain array-identical.

Dependencies: Task 1.8.1.

Acceptance criteria: exactly one seed-independent diagnostic incident state exists, has zero
photon mass, and introduces no second source-physics implementation.

### Task 1.8.3: evaluate exact mosaic-center representatives

Files likely touched:

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `tests/test_mosaic_ewald.py`

Exact intended behavior: for every authoritative `(family, branch)`, evaluate
`mosaic_alpha_rad=0` and canonical azimuth `0`; invoke the accepted root evaluator; preserve the
exact branch; reconstruct exact `L`, `q`, and `kf`; and apply the accepted structure strength and
outgoing-propagation classification. Keep a physically valid center candidate, otherwise pass the
component to Task 1.8.4. Do not integrate a mass, sample a CDF, or enumerate individual family rods
for label provenance.

Tests: a compact fixture covers branches `0/1/2`, zero tilt, canonical azimuth, `kf=ki+q`, Ewald
residual, lower/upper `L` ordering for branches `1/2`, and suppressed LINE_M0 direct root.

Dependencies: Task 1.8.2 and the accepted coating root seam.

Acceptance criteria: each center-valid component yields exactly one massless candidate with exact
family/branch identity and no copied coating or structure-factor equation.

### Task 1.8.4: select the nearest supported continuous orientation

Files likely touched:

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `tests/test_mosaic_ewald.py`

Exact intended behavior: for a center-invalid component, minimize spherical mosaic distance to the
mode subject to accepted component support, its exact branch, LINE_M0 boundary semantics, a regular
valid root, finite positive strength, and valid outgoing propagation. Apply the frozen tie order.
Branch `0` never enters nonzero branch-pair logic. Reuse the accepted support/classification
evaluator; do not expose quadrature nodes as candidates, build a painted-sphere grid, invent an
epsilon, or add a general optimizer. If the final coating seam cannot support this without copied
physics, stop and request one narrow interface.

Tests: an analytic nearest-boundary case, a center-valid case, a symmetric tie case, LINE_M0 cutoff
equality, and repeatability. Compare mosaic distance with an independently converged temporary
reference.

Dependencies: Task 1.8.3.

Acceptance criteria: orientation and identities meet frozen tolerances/exactness, no node or random
sample identity affects the result, and physical absence is typed rather than fabricated.

### Task 1.8.5: constrain the representative to detector support

Files likely touched:

- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`
- `src/rasim_next/reciprocal/coating.py` only if one narrow accepted classification seam is needed

Exact intended behavior: project physical candidates through unchanged outgoing transport and
detector geometry. If the physical minimum misses the detector, continue the same minimum-distance
selection with detector validity as a constraint. Use accepted detector-edge inclusivity and
continuous `(column_px, row_px)`. Emit at most one row per component; classify no-hit components as
`NO_DETECTOR_SUPPORT`. Never deposit a label or include it in photon rejection mass.

Tests: center hit, center miss with known nearest detector-valid support, no detector support, exact
winning `L` and identities, and unchanged image/mass ledgers.

Dependencies: Task 1.8.4 and the existing detector projection seam.

Acceptance criteria: every emitted row is detector-valid, no component has more than one row, no
sentinel coordinate is used, and raw image/mass output is bitwise unchanged.

### Task 1.8.6: reuse the staged parallel execution seam

Files likely touched:

- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`

Exact intended behavior: pack requests in canonical `(family, branch)` order and evaluate them
through the existing coating/event batches. If the accepted parallel path owns a persistent worker
pool, use it; otherwise use one vectorized batch rather than adding multiprocessing for this small
table. Workers return aligned candidate rows and the parent merges by canonical identity, never
completion order. Bound output by `K <= 1 + 2*F_nonzero`, independent of rays and draws.

Tests: compare scalar, packed serial, one/two workers, odd tiles, and reversed completion order;
require exact fields and row order. Vary source sample count, draw count, selection seed, and event
tile size without changing labels.

Dependencies: Task 1.8.5 and Tasks 1.3-1.5.

Acceptance criteria: execution layout cannot alter a label, no worker retains photon provenance or
mutates pixels, no new executor is introduced, and workspace is one bounded component tile plus
the compact label table.

### Task 1.8.7: annotate the default detector figure

Files likely touched:

- `scripts/generate_bi2se3_detector_image.py`

Exact intended behavior: calculate labels from the nominal ray, place markers at their continuous
detector coordinates, use stable colors for branches `0/1/2`, and display
`b=<branch>, m=<m>, L=<L>`. If direct text overlaps, use deterministic numeric marker IDs and one
canonical side legend instead of a layout framework. Do not alter raw image values or color scale,
and do not retain another sidecar or generated artifact outside an already accepted deliverable.

Tests: run the retained image script, compare marker and contract coordinates, assert raw
`image_A2` is unchanged, and inspect one temporary external PNG before cleanup.

Dependencies: Task 1.8.6.

Acceptance criteria: every detector-supported component has one readable tag at its exact
coordinate, the scientific image is unchanged, and no tag raster or per-photon identity remains.

### Task 1.8.8: prove, benchmark, document, and clean up

Files likely touched:

- the existing proof owner, preferably `src/rasim_next/pipeline/proof.py`
- `docs/CONTRACTS.md`, `docs/RESULT_MEASURE.md`, `docs/TRACE_SCHEMA.md`, and `docs/EXAMPLES.md`
- existing tests modified above; no new source or test file

Exact intended behavior: report component, center-valid, fallback, detector-supported, and omitted
counts plus coating/root evaluations, detector projections, wall time, and peak workspace. Document
grouping, exact `L` payload, nominal-ray provenance, nearest-mode/tie rules, zero-mass semantics,
and execution-layout invariance. Retain only compact tests protecting the label contract,
center/nearest-support correctness, detector noninterference, and parallel invariance if a distinct
parallel path remains. Delete dense searches, sweeps, visual snapshots, profiling logs, debug
counters, and temporary output.

Tests: run the compact integration/coating proofs and the default script; verify work is independent
of ray/draw count, scales with component count, and scalar/packed results agree. Large timing and
convergence sweeps remain external and temporary.

Dependencies: Task 1.8.7.

Acceptance criteria: label time is at most 10% of the accepted default forward simulation on the
designated host; additional memory is bounded by compact component work; one implementation owns
selection; every retained test protects a unique invariant; and no generated residue, duplicate
helper, unused dependency, or stale per-photon labeling requirement remains.

## Phase 1 checkpoint

- [ ] T07 remains fully passing.
- [ ] The accepted coating cutover is the sole production mosaic--Ewald path; the finite pool is
      proof-only until deletion and is absent afterward.
- [ ] CPU execution covers incident-state/component/node evaluation and sampled-event
      coating-to-detector work through staged bulk batches rather than peak parallelism.
- [ ] RNG, component order, sampled events, assigned masses, and reductions are reproducible.
- [ ] Expected moments match an independent continuous integral and stochastic convergence.
- [ ] The physical `m=0` contract is accepted and proven; otherwise Phase 1 is explicitly partial.
- [ ] Every detector-supported exact `(family, branch)` component has one stable zero-mass
      `(branch, m, L)` label; unsupported components have typed absence and no fabricated point.
- [ ] Representative labels are invariant to seed, ray/draw count, batching, workers, and
      completion order, while raw pixels and all photon-mass ledgers are unchanged.
- [ ] CPU/GPU agreement, crossover, wall time, and peak memory are recorded.

# Phase 2: detector-space geometry fitter

## Fitting gauge ownership required before Phase 2

Phase 2 may begin only after T09--T11 encode and validate the following active-pack contract:

- The LAB beam frame is fixed. One minimal beam-frame rotation may describe the beam relative to
  LAB, but no active pack may expose a compensating common beam/sample pose.
- Beam roll is inactive whenever both the spatial-width pair and divergence-width pair are
  isotropic; an unobservable roll is never retained as a fitted coordinate.
- Each fitted rotation axis has exactly two tangent coordinates. Its pivot has exactly two
  perpendicular components, with no axis-parallel pivot coordinate.
- Exactly one parameter block owns the zero-pose/sample-mount transform; the other representation
  is fixed rather than jointly fitted.
- A finite sample-support extent is inactive until observations reach its edge and uses a positive
  transform when activated. Unbounded support and `sample_from_crystal` translation at the incident
  stage expose no fitted coordinate.
- Detector calibration is a downstream revision: it invalidates projection, selection,
  deposition, and measurement products, never the detector-independent incident `ki` realization.

Future fitting proof must show a full-column-rank Jacobian for every accepted active pack; reject
deliberately redundant packs deterministically; deactivate isotropic beam roll; reconstruct the
two perpendicular pivot components; activate finite support only from edge-reaching observations;
and prove the stated incident-versus-detector invalidation boundary. These are future test
obligations in the owning fitting tasks, not tests or fitting code in this remediation.

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

Description: integrate the accepted continuous coating measure for every locked peak instead of
using stochastic second-stage draws.

When deposition, masks, or clipping matter, accumulate

\[
M_i=\int\sum_pD_p(\xi)\,d\mu_i(\xi),
\]

\[
\mathbf S_i=\int\sum_pD_p(\xi)\mathbf x_p\,d\mu_i(\xi),
\]

\[
\mathbf H_i=\int\sum_pD_p(\xi)\mathbf x_p\mathbf x_p^{\mathsf T}\,d\mu_i(\xi),
\]

and

\[
\boldsymbol\mu_i=\frac{\mathbf S_i}{M_i}.
\]

Use continuous-hit moments only after proving equivalence for unclipped interior bilinear
deposition. Each state tile returns canonical per-state partial `M`, `S`, and `H` for every active
observation, so the number of peaks does not control CPU utilization.

Acceptance criteria:

- [ ] Detector moments match an independently converged direct continuous integral.
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

Phase 3 consumes the accepted T18 immutable detector-to-angle projector, finite-bin `S/N/I`
profiles, masks, losses, identities, and revision envelope. It does not construct another `D -> M`
operator, choose a second grid/seam policy, or rebuild profile normalization.

## Angle-space coordinate contract

Use T18's one exact full-pixel-splitting detector-to-angle transform for angle-space images and
fitting. T18 constructs the sparse splitting operator from the accepted `CompiledInstrument`
physical pixel corners, including rectangular row/column pitch and the declared detector pose.
Simulation-native hits remain in the native detector frame and receive no display rotation.

For each detector pixel, transform all four physical corners into `(2theta, chi_raw)`, unwrap its
angular footprint across the fixed azimuth seam, and distribute its signal and normalization over
the intersected angle-space bins. Sum signal and normalization separately before division. The same
bin edges, seam, mask, corrections, and validity policy serve image construction and fitting.

The current `CompiledInstrument` has no calibrated distortion or non-planar-pixel map. Distortion is
out of scope for this phase. If required later, add one typed
detector-coordinate-to-physical-corner calibration boundary before `M`; never absorb distortion
into beam center, pitch, or Euler angles.

## Task 3.1: consume the accepted T18 angular coordinate contract

Description: validate and freeze the accepted T18 handoff before using its profiles in an
objective. The fitter records the handoff revisions and cannot redefine the transform.

The detailed coordinate requirements below are a consumer-side handoff checklist. T18 owns their
implementation and direct proof; Phase 3 retains only the integration evidence unique to fitting.

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

Dependencies: accepted Phase 2 and accepted T18 coordinate/profile ownership.

## Task 3.2: apply the accepted T18 sparse transfer before reduction

Description: apply T18's immutable sparse detector-to-angle operator to event-specific detector
contributions before any angular reduction; do not construct a fitter-owned operator.

The operator conservation and polygon-enumeration items below are T18 acceptance evidence. Phase 3
checks the accepted proof metadata and keeps one continuous-measure-to-profile integration test.

Let `D_ki` be the same bilinear deposition weight used by the detector renderer from continuous
event contribution `i` to detector pixel `k`. Let `M_bk` be the seam-safe full-pixel-splitting weight
from the four physical corners of detector pixel `k` to angle-space bin `b`. Do not materialize
their Cartesian product. In bounded tiles, the angular footprint of contribution `i` is

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

The authoritative order is event deposition, detector-signal summation, application of `M`,
normalization, and then the matched bin moment. Detector centroids remain detector-space
observables. T18 keys `M` by the complete instrument, nominal angle-space origin,
direct-beam/transverse basis, detector shape, physical pixel-corner calibration, bin edges, seam,
dtype, and summation-engine revisions. If a later fit varies any quantity that changes `M`, it
requests the correctly revised T18 operator/profile or returns an invalid evaluation; the fitter
never silently rebuilds or reuses a stale LUT.

Acceptance criteria:

- [ ] Every angular moment comes from continuous-measure contributions passed through `D` and `M`.
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

Dependencies: Task 3.1, accepted T18 sparse-transfer proof, and the Phase 2
continuous-measure/moment boundary.

## Task 3.3: consume the normalized T18 finite-bin profile

Description: accept T18's one finite-bin `S/N/I` profile and define only the fitter moment that
observation and prediction compare.

The `S/N/I`, mask, clipping, and correction rules below are immutable T18 inputs, not a second
profile implementation in the fitter. T18 constructs expected detector signal

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

Dependencies: Tasks 3.1-3.2 and the accepted T18 finite-bin measure/profile.

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

Description: reuse the Phase 1 device-resident continuous component kernel, the accepted T18 fixed
sparse splitting operator, and deterministic signal/normalization/moment reductions.

For the first angle-space fit, keep `M`, bin centers, masks, normalization state, and immutable
simulation state device-resident across optimizer evaluations. Stream only bounded
component-point/deposition tiles and the small residual vector. Batch independent trial points or
finite-difference/Jacobian columns when memory permits. Do not build a giant
event-by-angle-bin matrix.

Because detector calibration, masks, corrections, grid, and peak supports are frozen, consume
T18's fixed `N_b` once and derive a CSR row view containing only the union of active angle-space fit
bins. Retain the full-field proof path, and permit the active-row path only after it reproduces the same `S`,
`I`, moments, and residuals. This avoids recomputing irrelevant angle-space regions on every trial
while preserving the frozen angle-space measure.

The initial fit freezes detector calibration and the nominal angle-space origin/basis, so `M` is
constant. If a later fit varies that origin/basis, distance, beam center, detector pitch/pose,
physical corners, shape, or another splitting-transform parameter, that trial must use a newly
built or correctly keyed `M` requested from T18. Benchmark that request/rebuild cost separately;
never silently reuse the frozen operator.

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

- [ ] Phase 3 consumes T18's projector/profile/revisions and owns no detector-to-angle producer.
- [ ] Every weighted continuous-measure contribution is transformed before angular reduction.
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
| Tiny scalar | analytic identities and independently converged continuous integration |
| CPU staged batches | scalar proof path |
| CPU parallel | serial staged batches under equivalent work |
| GPU | accepted CPU path, including boundary classifications |
| Detector moments | direct continuous integrals and stochastic convergence |
| Detector fit | hidden known truth and held-out detector observations |
| Angular moments | direct sparse-transfer and normalized-bin enumeration |
| Angle-space fit | same hidden truth used by detector-space fit |
| Binned angle field | independent direct `S`, `N`, valid mask, `I`, axes, and centroid |

Record wall time, peak memory, setup/compile time, transfer time where applicable, incident-state,
component, density-evaluation, adaptive-panel, and sampled-event counts, precision, hardware,
code/data/configuration revisions, and error versus the accepted reference.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CPU parallelism is limited by Python overhead | High | Localize/vectorize first; benchmark fused execution before selecting workers |
| GPU is slow for branch-heavy float64 work | High | Record crossover and retain CPU production path |
| Adaptive or event batches exhaust memory, fragment on ragged rods, or develop a long tail | High | Bound active rows by bytes, use canonical ragged offsets, and finish small tails without a global barrier or nested pool |
| Batching changes refinement, RNG choices, or reductions | Critical | One logical component controller, physical counter keys, canonical slots/ties/scans/reductions, and proof hashes |
| Once-only factors are reapplied or sampled reconstruction disagrees with outgoing propagation | Critical | Separate assigned mass from geometry, prove the output ledger, and treat disagreement as a typed consistency failure |
| Intersection roots, coating components, and physical fitting branches are conflated | Critical | Separate typed identities and explicit mapping proofs |
| Initial nonzero-m coating is mistaken for complete simulation | Critical | Keep Phase 1 partial until a physical m=0 contract passes; never use a hidden legacy fallback |
| Labels are grouped by floating `m` or `L` | Critical | Group by exact family and intersection-branch identity; keep `L` as payload only |
| Diagnostic label rays alter the Monte Carlo image or mass | Critical | Use a separate nominal ray and a contract with no mass field; prove pixels and ledgers unchanged |
| Nearest-label support duplicates coating physics | Critical | Reuse one narrow accepted support evaluator or stop; never add a grid, epsilon, or second solver |
| Parallel completion reorders representative branches | High | Canonical component merge and exact scalar/packed/worker-layout comparisons |
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

- Retain only compact tests protecting unique contracts: once-only continuous mass, deterministic
  physical-keyed sampling, canonical state/component order, the zero-mass representative-label
  contract/noninterference, detector moments, line grouping, angular wrapping, and one
  representative blind recovery per distinct long-term failure mode.
- Keep broad truth matrices, large images, GPU sweeps, convergence studies, and profiling output as
  external proof artifacts.
- Remove temporary benchmarks, exploratory scripts, generated dumps, redundant tests, and unused
  backend experiments before each phase handoff.
- Do not retain a GPU dependency unless it wins a declared workload or is necessary for an accepted
  later fitting workload.

## Final completion gate

- [ ] One authoritative continuous coating implementation serves all paths.
- [ ] Individual images use full CPU capacity through staged incident/component/node and
      sampled-event batch work.
- [ ] GPU acceleration is available only where measured and scientifically equivalent.
- [ ] One stable zero-mass `(branch, m, L)` label exists for every detector-supported exact coating
      component, with canonical branch `0/1/2` identity and no per-photon tag retention.
- [ ] Label generation leaves raw detector pixels, draw counts, and every photon-mass ledger
      unchanged across serial and parallel execution layouts.
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
