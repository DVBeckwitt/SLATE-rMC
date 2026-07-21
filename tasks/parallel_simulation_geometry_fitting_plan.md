# CPU/GPU-compatible simulation, representative detector labels, and staged geometry fitting plan

Status: **HISTORICAL PROPOSAL.** CPU/CUDA source-averaged detector execution is implemented under
contract API v9. Any future fitting work requires a fresh plan against the live continuous APIs;
the sampled-cell, tag-cache, and point-deposition assumptions below are not runtime authority.

Entry gate: shared beam-to-`ki` Checkpoint K passes first. Deterministic DP-02 through DP-06 then
freeze the accepted scalar/cell pushforward, detector-native observable, and component-tag cache
before parallel Task 1.1 or any fitting task begins.

This plan records the agreed path for:

1. making the detector simulation efficient on the CPU and compatible with measured GPU
   acceleration;
2. placing one stable zero-mass `(branch, m, L)` representative label for every coating component
   with detector support;
3. recovering known geometry blindly from detector-native expected peak centroids; and
4. recovering the same geometry from expected angle-space `(2theta, phi)` peak centroids.

The work must preserve one authoritative physics path. CPU rendering, GPU execution,
detector-space fitting, and angle-space fitting consume the same continuous coating measure,
deterministic cell pushforward, detector contributions, and DP-06 tags; none may implement
alternate scattering physics.

## Governing constraints

- Complete and prove the detector-native simulation before fitting.
- Prove detector-native fitting before adding angle-space fitting.
- Parallelize bounded bulk batches of incident states, continuous-component integration records,
  mapped deterministic cells, and pixel partials, not the number of selected peaks, rods, or HKL
  groups.
- Keep one fixed complete source realization and fixed mosaic parameters/revisions during every
  fit.
- Do not use randomly generated geometric truth values, random optimizer starts, or random
  geometric proposals for the current fitting proof.
- Use deliberately chosen geometric combinations whose truth is known but hidden from the fitter.
- Keep rod, reflection-group, branch, and `COLLAPSED_00L` identities frozen inside one optimizer
  run.
- Treat invalid or ambiguous topology as an invalid evaluation; never reassign identities inside
  the objective.
- Use radians internally and canonical detector-native `(column_px, row_px)` coordinates.
- Keep detector-to-angle transfer as a measurement transform. It cannot redefine upstream detector
  orientation, branch identity, coating/detector mass, or normalization.
- Use one exact full-pixel-splitting detector-to-angle transform and one normalized angle-space
  observable.
- Consume representative detector labels only from the deterministic DP-00C/DP-01/DP-06
  component-tag authority. Parallel Task 1.8 may project or display that accepted zero-mass tag
  contract, but it cannot construct another nominal ray or choose another coating representative.
- Generate and hash the complete canonical source realization once, then build the complete
  `IncidentStateBatch` once and serially before private component/cell dispatch. Workers receive
  immutable parent rows selected by explicit `parent_row_index` values plus inherited parent
  revisions; they never
  regenerate source rows, construct or hash public source/incident slices, or sort physical rows
  by `incident_state_id`. That ID remains identity payload, not a reassembly key.
- Do not freeze parallel numeric records or deterministic internal/projection cache keys until
  [Checkpoint K](plan.md#checkpoint-k-shared-beam-to-ki-boundary) accepts the shared beam-to-`ki`
  material, geometry, and revision contracts.
- `geometry/transport.py` is owned by the BKI remediation through BKI-15 and is frozen again after
  that gate; later edits require a separately reviewed scientific need.
- Keep raw azimuth `chi_raw` and fitting/display azimuth `phi` as separately named quantities.
- Select CPU/GPU technology only after the accepted integrated workload is profiled, as required
  by [the performance strategy](../docs/PERFORMANCE.md).

## Deterministic pushforward parallelization amendment

### Decision and ownership

Parallel work consumes the accepted deterministic Ewald-coating pushforward; it does not own or
recreate its source, coating, root, cell-integration, exit, detector, or component-tag equations.
The deterministic DP-02 through DP-05 contracts define the cell/component work and the DP-06 cache
defines tags. Parallelization changes only bounded execution layout.

The controller generates and hashes the complete source once, builds the complete
`IncidentStateBatch` once and serially, and owns its canonical row order. Private downstream
records carry `parent_row_index` into that table. Workers receive immutable indexed inputs and
return aligned results; they never sample a source, call `build_incident_states`, construct or hash
a public slice, sort by `incident_state_id`, deduplicate equal numeric `ki`, mutate a shared image,
or choose a representative tag.

### Smallest useful scope

1. Start only after shared beam-to-`ki` Checkpoint K and deterministic DP-02 through DP-06 are
   accepted.
2. Profile the accepted serial deterministic pushforward before changing execution layout.
3. Batch regular point/cell evaluations across parent states and coating components under bounded
   memory. Keep topology certification, adaptive refinement decisions, cache ownership, and
   canonical reductions under deterministic controller control.
4. Parallelize exact exit/detector mapping and pure cell/pixel partials only when the accepted
   partition and conservation ledger remain unchanged.
5. Use a GPU only for measured regular batches above a declared crossover. Boundary decisions and
   every discrete identity remain controlled by the accepted CPU proof path.
6. Pass accepted DP-06 tag rows to fit/display consumers unchanged. DP-10 remains the sole optional
   rendering owner.

### Private execution records

These are private row layouts, not public contracts or a backend abstraction:

```text
component/cell input:
    parent_row_index
    incident_state_id identity payload
    accepted deterministic component/cell identity
    immutable compiled scientific state and owner revisions

component/cell output:
    parent_row_index
    accepted physical identity, status, value, and certified bound
    deterministic sphere/detector partials or mapped-cell data
```

All outputs preserve input alignment. The controller validates parent indices, scatters through
`parent_row_index`, and then applies the accepted state/family/component/cell reduction order.
`incident_state_id` is verified at the boundary but never used as a row-order key.

### Non-goals

- No outgoing-event RNG, component CDF, inverse-CDF, `T/N`, sampled-event batch, candidate pool,
  point/bilinear deposition, or per-photon tag retention.
- No public packet API, executor framework, scheduler interface, backend registry, compatibility
  facade, duplicate physics helper, or worker-local cache authority.
- No task per state, rod, adaptive node, cell, pixel, or tag row.
- No GPU requirement when strict-float64 branch-heavy work is faster on the CPU.

### Proof cadence

- A one-row proof compares accepted scalar and tiled deterministic component/cell results,
  classifications, certified bounds, ledgers, image, moments, and tags.
- A 33-row checkpoint uses deliberately nonmonotonic/nonconsecutive IDs, contiguous/strided/reverse
  parent packets, odd tiles, one/two workers, and reversed completion order.
- A clean 129-row/full-catalog benchmark records equivalent work, wall time, utilization, and peak
  memory. Broad worker/tile/GPU sweeps remain external artifacts.
- Every optimized path reproduces the accepted deterministic result under frozen exact/tolerance
  policy and leaves source weights, revisions, and all conservation ledgers unchanged.

### Amendment acceptance criteria

- [ ] One complete serial source-to-incident predecessor serves every deterministic parallel row.
- [ ] Reassembly uses `parent_row_index`; layout and completion order cannot change identities,
      revisions, statuses, certified bounds, images, moments, tags, or ledgers.
- [ ] DP-00C/DP-01/DP-06 remain the only nominal-fixture/coating/tag owners, and DP-10 remains the
      only renderer owner.
- [ ] CPU/GPU selection is measurement-based, bounded, and scientifically equivalent.
- [ ] No sampled-outgoing-event or duplicate-physics execution surface remains.

## Dependency graph

```text
shared beam-to-ki Checkpoint K
  -> accepted deterministic DP-02 through DP-06 contracts
  -> profiled serial deterministic pushforward
  -> private parent-indexed component/cell records
  -> bounded CPU execution and canonical reduction
  -> optional measured GPU regular-batch path
  -> detector-space fitting
  -> accepted T18 detector-to-angle transfer
  -> angle-space fitting
```
## Shared simulation boundary

Every execution mode consumes the same authoritative sequence:

```text
complete fixed IncidentSampleBatch
    -> complete serial IncidentStateBatch
    -> deterministic family/root/component cells
    -> exact per-rod coating measure with once-only factors and certified bounds
    -> exact exit/detector mapped cells
    -> conservative detector-pixel mass and deterministic moments
```

The complete continuous measure includes every applicable source, phase/parent,
reciprocal/mosaic/coarea, scattering-strength, optical, polarization, attenuation, population, and
outgoing-propagation factor exactly once. Detector mapping follows the accepted deterministic
coating measure; detector misses retain explicit status/mass and neither detector validity nor
detector solid angle is part of the raw coating mass. Refining or re-tiling a cell cannot reapply a
factor already represented in its measure.

From this boundary, explicit consumers may:

- render the deterministic detector-native image;
- reduce detector expected moments over the same accepted cell measure; or
- pass detector contributions to T18 and consume its deterministic normalized angle-space profile
  and matched expected moments.

No fitting module may reimplement source sampling, continuous coating support, Ewald geometry,
optical transport, detector mapping, conservative accumulation, or coating/detector mass.

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

## Task 1.1: accept the deterministic scalar/cell reference

Description: freeze the accepted DP-02 through DP-06 scientific result before performance work.

Exact intended behavior:

- Consume the accepted one-row source-to-`ki`, coating, sphere, exit, detector, and tag proofs.
- Record stage timings, equivalent work counts, certified bounds, wall time, and peak memory.
- Keep scalar and independently converged latent oracles available as proof authorities; neither
  becomes a second production runtime.

Verification:

- Run the deterministic focused/full proof gates and the numeric-only benchmark twice.
- Require identical scientific arrays, identities, statuses, tags, and ledgers; timing and peak
  memory remain measured metadata.

Dependencies: shared beam-to-`ki` Checkpoint K and accepted deterministic DP-02 through DP-06.

Acceptance criteria:

- The deterministic pushforward is the sole production reference and its frozen tolerances pass.
- No sampled-event comparison code is promoted to an execution seam.

## Task 1.2: reduce deterministic mathematical work

Description: profile the accepted serial path and remove avoidable work before adding executors or
a GPU path.

Exact intended behavior:

1. compile instrument transforms, material state, rod metadata, and invariant reciprocal geometry
   once under their owner-provided revisions;
2. reuse the accepted detector-independent partition and immutable component data only under
   complete deterministic cache keys;
3. batch regular point/cell evaluations in bounded contiguous tiles;
4. stream mapped cells and pixel partials without materializing incident-by-rod-by-cell-by-pixel
   Cartesian products; and
5. profile topology certification, coating evaluation, ordered strength, exit optics, detector
   mapping, conservative accumulation, tag projection, and reduction separately.

Verification:

- Compare equivalent evaluations, classifications, bounds, ledgers, image, moments, wall time, and
  peak memory with Task 1.1.

Dependencies: Task 1.1.

Acceptance criteria:

- Work reduction preserves the accepted deterministic observable and proof bounds.
- Peak memory is bounded by active parent/component/cell tiles and deterministic output partials.

## Task 1.3: freeze the private parent-indexed numeric boundary

Description: define the smallest private row layout needed for measured bulk execution without
adding a public packet or backend API.

```text
input rows:
    parent_row_index into the complete IncidentStateBatch
    incident_state_id identity payload
    accepted component/cell identity and canonical slot
    immutable compiled scientific state and owner revisions

output rows:
    parent_row_index
    accepted physical identities, values, statuses, and certified bounds
    deterministic sphere/detector partials or mapped-cell data
```

Exact intended behavior:

- The controller constructs the complete source and incident table once and serially.
- Validate private parent indices as one-dimensional, in-range, and nonduplicate for state
  partitions; repeated indices are allowed only where multiple component/cell records intentionally
  refer to one state.
- Preserve input alignment in every kernel. Reassemble by `parent_row_index`, then by the accepted
  component/cell slot; never sort by `incident_state_id`.
- Inherit all source/sample/material/incident revisions and proof fields unchanged.
- Keep adaptive state, mapped polygons, work queues, and executor/device objects private.

Verification:

- Compare scalar and private-row execution using nonmonotonic/nonconsecutive IDs, contiguous,
  strided, reverse, odd-sized, one-worker, two-worker, and reversed-completion layouts.
- Reject out-of-range/duplicate state partitions and missing coverage of the controller's declared
  scheduled parent-row set.

Dependencies: Task 1.2 and shared beam-to-`ki` Checkpoint K.

Acceptance criteria:

- Layout cannot change parent order, identity, `ki`, source weight, revisions, status, or result.
- No public beam wrapper, sliced batch, generic callback, or executor abstraction is added.

## Task 1.4: implement bounded CPU bulk execution

Description: benchmark serial vectorized batches, then add only the smallest justified CPU worker
boundary.

Exact intended behavior:

- Schedule bounded complete component/cell tiles drawn from many parent states.
- Workers are pure: they receive immutable indexed data, return aligned partials, and own no source,
  adaptive controller, cache, tag, shared image, or reduction state.
- The controller owns topology/refinement decisions and reduces partials in the accepted canonical
  parent/component/cell/pixel order.
- Use one process/thread level only; do not create nested pools or a task per physical row.

Verification:

- Run the one-row proof, 33-row layout checkpoint, and clean 129-row/full-catalog benchmark.
- Sweep worker/tile sizes only as external evidence and compare equivalent work.

Dependencies: Task 1.3.

Acceptance criteria:

- CPU utilization improves on the measured hot path without changing any discrete identity,
  certified bound, scientific output, or ledger.
- Peak memory stays within the declared active-tile budget.

## Task 1.5: preserve deterministic execution semantics

Description: make worker and tile layout observationally irrelevant.

Exact intended behavior:

- No execution-order or worker identity enters a physical/cache key, adaptive decision, tag
  identity, or reduction order.
- Host/controller arbitration owns every topology, fold, seam, tangent, support, and validity
  decision; uncertain optimized rows replay through the accepted robust CPU path.
- Canonical reductions never use unordered atomics.
- Changing sample/material/source revisions invalidates the exact accepted downstream dependency
  set and is visible in provenance.

Verification:

- Hash canonical identities, statuses, partition/cell ownership, bounds, arrays, tags, and ledgers
  across the layouts from Task 1.3.

Dependencies: Task 1.4.

Acceptance criteria:

- Equivalent layouts reproduce exact discrete state and meet the frozen floating tolerances.
- No outgoing-event RNG, CDF, draw count, selection seed, or sampled-event record exists.

## Task 1.6: expose deterministic images and moments

Description: retain explicit detector image and moment consumers of the same accepted cell measure.

Exact intended behavior:

- Produce detector-native image mass and `M`, `S`, and `H` moment summaries from accepted
  deterministic cell contributions.
- Preserve exhaustive pre-optics exit and post-optics detector status ledgers plus independent
  certified numerical bounds.
- Keep detector misses, clipping, numeric failures, and deposited mass separate; no survivor
  renormalization or hidden balancing residual is allowed.
- A tag is metadata only and cannot contribute to any image, moment, or ledger.

Verification:

- Compare image mass, first/second moments, centroids, status closure, and refinement convergence
  with independently converged latent integration.

Dependencies: Tasks 1.1-1.5.

Acceptance criteria:

- Images and moments satisfy their accepted deterministic tolerance and conservation contracts.
- Parent layout and tags cannot change the numeric outputs.

## Task 1.7: add and evaluate GPU execution

Description: prototype only sufficiently large regular batches identified by CPU profiling.

Exact intended behavior:

- Keep immutable compiled state resident during repeated work and use float64 for proof.
- Keep adaptive refinement, robust classification, cache ownership, and canonical reduction under
  deterministic host control.
- Recompute a GPU row on the CPU whenever its interval can change validity, topology, refinement,
  compaction, cell ownership, or another discrete decision.
- Avoid unordered atomics; record compile, transfer, execution, and reduction time separately.
- Define a measured CPU/GPU crossover and retain the CPU path when it wins.

Verification:

- Compare small proof, medium forward, large forward, and repeated fitting evaluations against the
  accepted CPU path, including boundary classifications and certified bounds.

Dependencies: Tasks 1.3-1.6.

Acceptance criteria:

- GPU batching cannot change a discrete identity and reproduces the accepted observable within
  frozen tolerance.
- No backend framework or mandatory GPU dependency is introduced without measured value.
## Task 1.8: consume accepted deterministic component tags

Description: pass the accepted deterministic `CoatingComponentTagBatch` to fit and display
consumers without adding source, coating, representative, cache, or rendering physics.

Ownership is exclusive: DP-00C owns the sole nominal one-row fixture, DP-01 owns the pointwise
coating equation, DP-06 owns the representative solver/cache/projection contract, and DP-10 owns
the optional rendering tool. This parallel task owns none of those equations or builders.

Exact intended behavior:

- Consume canonical DP-06 rows unchanged; do not add another nominal source/ray, support search,
  tie rule, representative solver, tag type, cache key, or image-script authority.
- If a private consumer tiles tag rows, retain `parent_row_index` and scatter results through that
  index. Verify `incident_state_id` as identity payload; never sort or reassemble by it.
- Preserve the nominal source row's empirical `source_weight == 1`. Only the tag has no assigned
  or deposited event mass, and it cannot enter pixels, assigned mass, rejected mass, clipped mass,
  or any photon ledger.
- Keep invalid exit/detector rows and absent optional projected fields exactly as accepted by DP-06.

Tests and verification:

- Exercise nonmonotonic/nonconsecutive incident IDs, strided/reversed parent-index packets, odd
  tiles, and reversed completion order; require exact tag fields and parent order.
- Prove raw detector pixels, source weights, and every assigned/rejected/clipped/deposited ledger
  remain unchanged.
- Static scans must find no second nominal builder, representative solver, tag contract, cache key,
  or live massless-source description of an `IncidentSampleBatch`.

Dependencies: shared beam-to-`ki` Checkpoint K and accepted deterministic DP-00C, DP-01, DP-06, and
DP-10 for any rendered output.

Acceptance criteria:

- Exactly one fixture, coating equation, representative solver/cache, and renderer owner exists.
- Parallel layout and completion order cannot change tag identity, projection, source mass, or
  photon/detector ledgers.
- No new production module, executor, public packet type, or source physics is added.

## Phase 1 checkpoint

- [ ] T07 remains fully passing.
- [ ] The accepted coating cutover is the sole production mosaic--Ewald path; the finite pool is
      proof-only until deletion and is absent afterward.
- [ ] CPU execution covers parent-indexed incident-state/component/cell evaluation and deterministic
      coating-to-detector pushforward through staged bulk batches rather than peak parallelism.
- [ ] Component order, parent-row reassembly, deterministic partitions, and reductions are
      execution-layout invariant; no outgoing-event RNG or sampled-event record is revived.
- [ ] Expected moments match an independently converged continuous integral.
- [ ] The physical `m=0` contract is accepted and proven; otherwise Phase 1 is explicitly partial.
- [ ] The accepted DP-06 cache supplies one stable no-assigned-mass `(branch, m, L)` tag for every
      supported exact `(family, branch)` component; invalid projections remain typed and no point
      is fabricated.
- [ ] DP-06 tags pass through unchanged under batching, workers, and completion order. Their source
      row retains `source_weight == 1`, while raw pixels and all photon-mass ledgers are unchanged.
- [ ] CPU/GPU agreement, crossover, wall time, and peak memory are recorded.

# Phase 2: detector-space geometry fitter

## Fitting gauge ownership required before Phase 2

Phase 2 acceptance requires the following active-pack contract. T09 and accepted T10 are entry
prerequisites; T11 owns the sample/goniometer active-pack seam implemented within Phase 2.

- T09 compiles material coverage for every exact unique wavelength in the complete parent
  `IncidentSampleBatch`, including rows invalid under baseline geometry, and rejects missing
  wavelengths before objective evaluation. It consumes owner-provided source, sample, and material
  revisions plus the incident model ID and never rehashes that scientific state.
- The LAB beam frame is fixed. One minimal beam-frame rotation may describe the beam relative to
  LAB, but no active pack may expose a compensating common beam/sample pose.
- T10 fixes one named physical source reference plane in LAB—fixed point, accepted nominal LAB
  beam-axis normal, and ordered orthonormal in-plane basis—before exposing any transverse
  position-direction correlation. The active pack has no longitudinal source-origin coordinate;
  trial beam rotation cannot move the plane, and changing reference plane is an explicit
  moment/correlation reparameterization rather than a fit degree of freedom.
- Beam roll is inactive whenever both the spatial-width pair and divergence-width pair are
  isotropic; an unobservable roll is never retained as a fitted coordinate.
- Each fitted rotation axis has exactly two tangent coordinates. Its pivot has exactly two
  perpendicular components, with no axis-parallel pivot coordinate.
- Exactly one parameter block owns the zero-pose/sample-mount transform; the other representation
  is fixed rather than jointly fitted.
- A finite sample-support extent is inactive until observations reach its edge and uses a positive
  transform when activated. Unbounded support exposes exactly one signed plane-normal offset and
  no in-plane translation; `sample_from_crystal` translation remains fixed. Finite-support tangent
  translations activate only when edge-reaching observations identify them.
- Detector calibration is a downstream revision: it invalidates projection, selection,
  deposition, and measurement products, never the detector-independent incident `ki` realization.

Future fitting proof must show a full-column-rank Jacobian for every accepted active pack; reject
deliberately redundant packs deterministically; deactivate isotropic beam roll; reconstruct the
two perpendicular pivot components; cover a baseline-invalid wavelength that becomes valid and a
missing-wavelength preflight failure; prove source-reference-plane shift equivalence; recover the
unbounded signed normal offset while rejecting both tangent translations; contrast finite support
under edge-reaching observations; and prove the stated incident-versus-detector invalidation
boundary. These are future `tests/test_fitting.py` obligations in the owning fitting tasks, not
tests or fitting code in this specification slice.

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

Description: reduce the accepted deterministic cell pushforward for every locked peak.

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

Here `D_p` is the accepted conservative deterministic cell-to-pixel mass allocation, not a point
or bilinear depositor. Each parent-state tile returns canonical partial `M`, `S`, and `H` for every
active observation, so the number of peaks does not control CPU utilization.

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
of parent-state sample intersections and cannot move with a trial. Detector calibration remains fixed
for the first angle-space fit. Azimuth at the direct beam is undefined; a low circular resultant or
direct-beam point is an explicit invalid angular observation. A detector corner exactly at the
angular origin follows one frozen polygon-seam tie policy but does not create a physical azimuth
observation.

Acceptance criteria:

- [ ] Coordinate transforms have explicit frame, unit, revision, axes, and provenance.
- [ ] The nominal angle-space origin and ordered beam basis are immutable, orthonormal, have
      explicitly tested handedness, are versioned, and remain independent of parent-state ray
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

Description: apply T18's immutable sparse detector-to-angle operator to the deterministic
detector-native pixel signal before any angular reduction; do not construct a fitter-owned
operator.

The operator conservation and polygon-enumeration items below are T18 acceptance evidence. Phase 3
checks the accepted proof metadata and keeps one deterministic detector-image-to-profile
integration test.

Let `P_k` be the accepted deterministic detector mass in native pixel `k`, after conservative
cell-to-pixel pushforward and explicit clipping/status accounting. Let `M_bk` be the seam-safe
full-pixel-splitting weight from the four physical corners of detector pixel `k` to angle-space bin
`b`. The angular signal is

\[
S_b=\sum_k M_{bk}P_k.
\]

SLATE-rMC's valid detector support is `[-0.5,W-0.5] x [-0.5,H-0.5]`. Apply the accepted
cell-to-pixel and clipping contract over that entire support, including all four half-pixel edge
strips.

Use increasing uniform bin edges and center-valued axes. Radial bins cover zero through the maximum
detector-corner `2theta`; raw-`chi` bins cover `[-pi,pi)`. Before polygon overlap, locally unwrap a
pixel's four raw-`chi` corners across the seam exactly once so the footprint remains contiguous,
then wrap bin ownership back to the frozen interval. Record boundary/tie policy because exact seam
and beam-center ties can otherwise become order- or dtype-dependent.

The authoritative order is deterministic detector pushforward, detector-signal summation,
application of `M`, normalization, and then the matched bin moment. Detector centroids remain
detector-space observables. T18 keys `M` by the complete detector-causal instrument geometry,
nominal angle-space origin, direct-beam/transverse basis, detector shape, physical pixel-corner
calibration, bin edges, seam, dtype, and summation-engine revisions. If a later fit varies any
quantity that changes `M`, it requests the correctly revised T18 operator/profile or returns an
invalid evaluation; the fitter never silently rebuilds or reuses a stale LUT.

Acceptance criteria:

- [ ] Every angular moment comes from accepted deterministic detector pixels passed through `M`.
- [ ] `M` is a sparse full-pixel-splitting operator built from physical corners with deterministic
      seam handling and canonical bin ordering.
- [ ] For every valid unmasked pixel with full angular support, `sum_b M_bk = 1` within frozen
      tolerance; clipped axes expose `lost_support_k = 1 - sum_b M_bk` and never silently
      renormalize it.
- [ ] A nonlinear counterexample proves the detector signal -> `M` -> `S/N` -> moment order.
- [ ] The angular summary carries its projector, grid, and detector-causal instrument revisions.
- [ ] Operator cache invalidation covers the angle-space origin/basis, center, distance, pitch,
      pose, physical corners, shape, bin edges, seam, dtype, and summation engine.

Verification:

- [ ] Compare direct polygon/bin enumeration, sparse-operator application, and the optimized tiled
      reducer.
- [ ] Prove detector pixel centers/corners, exact shared edges, circular-seam fixtures, the four
      half-pixel edge strips, and direct-beam invalidation by direct enumeration.
- [ ] Prove unity-field, total-signal, and per-pixel column-sum conservation; masks remove identical
      support from `S` and `N`, and angular clipping reports exact lost support.
- [ ] Compare `M` and angle-space fields with an independent direct full-pixel-splitting oracle.

Dependencies: Task 3.1, accepted T18 sparse-transfer proof, and the Phase 2 deterministic
detector-measure/moment boundary.
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
component/cell/pixel-partial tiles and the small residual vector. Batch independent trial points or
finite-difference/Jacobian columns when memory permits. Do not build a giant
cell-by-angle-bin matrix.

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
      of trial parent-state intersections.
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
| Detector moments | direct independently converged continuous integration |
| Detector fit | hidden known truth and held-out detector observations |
| Angular moments | direct sparse-transfer and normalized-bin enumeration |
| Angle-space fit | same hidden truth used by detector-space fit |
| Binned angle field | independent direct `S`, `N`, valid mask, `I`, axes, and centroid |

Record wall time, peak memory, setup/compile time, transfer time where applicable, incident-state,
component, density-evaluation, certified-cell, mapped-cell, and pixel-partial counts, precision,
hardware, code/data/configuration revisions, and error versus the accepted reference.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CPU parallelism is limited by Python overhead | High | Localize/vectorize first; benchmark fused execution before selecting workers |
| GPU is slow for branch-heavy float64 work | High | Record crossover and retain CPU production path |
| Adaptive or mapped-cell batches exhaust memory, fragment on ragged rods, or develop a long tail | High | Bound active rows by bytes, use canonical ragged offsets, and finish small tails without a global barrier or nested pool |
| Batching changes topology, refinement, classification, or reductions | Critical | One logical component controller, canonical parent/component/cell slots and reductions, and proof hashes |
| Once-only factors are reapplied or mapped-cell exit state disagrees with the accepted geometry | Critical | Keep factor ownership explicit, prove the conservation/status ledgers, and treat disagreement as a typed consistency failure |
| Intersection roots, coating components, and physical fitting branches are conflated | Critical | Separate typed identities and explicit mapping proofs |
| Initial nonzero-m coating is mistaken for complete simulation | Critical | Keep Phase 1 partial until a physical m=0 contract passes; never use a hidden legacy fallback |
| Labels are grouped by floating `m` or `L` | Critical | Group by exact family and intersection-branch identity; keep `L` as payload only |
| Component-tag consumption alters source or detector mass | Critical | Consume only DP-06 rows, preserve the DP-00C source weight of one, and prove assigned/deposited ledgers and pixels unchanged |
| Nearest-label support duplicates coating physics | Critical | Reuse one narrow accepted support evaluator or stop; never add a grid, epsilon, or second solver |
| Parallel completion reorders representative branches | High | Reassemble by `parent_row_index` and compare nonmonotonic-ID/reversed-completion layouts exactly |
| Finite source-realization differences destabilize geometry | High | Freeze the complete parent realization per fit and require the denser/different-revision Tier B proof |
| Line residual double-counts centroid data | Medium | Label it dependent guidance, use coordinate-unit scaling, freeze weight, and run point-only audit |
| Trial collapses a line to evade angle penalty | High | Use fixed target span in the half-angle residual |
| m=0 observations acquire fake branches | Critical | Separate typed `COLLAPSED_00L` line contract |
| `chi_raw` is mislabeled as fitting/display `phi` | Critical | Separate names/types and cardinal-direction fixtures |
| Pixel-coordinate handling adds a half-pixel shift | Critical | Direct physical-corner construction and scalar geometry tests |
| RA interior support is imposed on valid SLATE edge strips | High | Domain-specific classification and four edge-strip fixtures |
| A GUI/OSC rotation is applied during detector-to-angle transfer | Critical | Native-frame-only operator; no downstream rotations |
| Angle-space `phi` crosses a moving wrap cut | High | Frozen local unwrap origin per observation and group |
| A stale angle-space LUT is reused after transform geometry changes | Critical | Complete immutable cache key or explicit rebuild per trial |
| Nominal angle-space origin/basis follows per-parent trial geometry | Critical | Freeze and version the nominal origin/basis independently |
| Direct-beam azimuth is treated as physical data | High | Undefined/low-resultant invalidation; radial-only `00L` handling |
| Trial conditioning changes residual length | Critical | Freeze point/line component masks; fixed-schema invalid evaluation |
| Target centroid overwrites a trial prediction | Critical | Immutable observation/prediction types and injected-value regression |
| Synthetic recovery is an inverse crime | High | Add different-sample Tier B and held-out peaks |
| Truth leaks into optimizer state | Critical | Separate truth process/artifact and sanitized-state audit |
| A normalized angle-space field is mistaken for raw detector-pixel mass | Critical | Explicit signal/normalization contract and equivalence proof |
| Signed background-subtracted bins are used as centroid mass | High | Reject or use a separately proven background/signed-data model; never clip |
| Unsupported detector distortion is hidden in rigid geometry | High | Declare out of scope or add a typed physical-corner calibration boundary |

## Permanent proof and cleanup policy

- Retain only compact tests protecting unique contracts: once-only continuous mass, deterministic
  parent-row/component ordering, the no-assigned-mass DP-06 tag contract/noninterference, detector
  moments, line grouping, angular wrapping, and one
  representative blind recovery per distinct long-term failure mode.
- Keep broad truth matrices, large images, GPU sweeps, convergence studies, and profiling output as
  external proof artifacts.
- Remove temporary benchmarks, exploratory scripts, generated dumps, redundant tests, and unused
  backend experiments before each phase handoff.
- Do not retain a GPU dependency unless it wins a declared workload or is necessary for an accepted
  later fitting workload.

## Final completion gate

- [ ] One authoritative continuous coating implementation serves all paths.
- [ ] Individual images use full CPU capacity through staged parent-indexed
      incident/component/cell pushforward work.
- [ ] GPU acceleration is available only where measured and scientifically equivalent.
- [ ] The sole DP-06 cache supplies one stable no-assigned-mass `(branch, m, L)` tag for every
      supported exact coating component, with canonical branch `0/1/2` identity and no per-photon
      tag retention.
- [ ] Parallel consumption of accepted tags leaves the DP-00C source weight, raw detector pixels,
      deterministic output, and every coating/detector ledger unchanged across layouts.
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
