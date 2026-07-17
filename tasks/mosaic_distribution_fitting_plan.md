# Bragg-sphere-informed mosaic distribution fitting plan

Status: PROPOSED. Planning only; no production implementation is authorized by this document.

This is the proposed execution plan for a post-angle mosaic-fit task. Its task ID and owned paths
are intentionally not assigned yet. It begins only after all three phases of
[the simulation and geometry-fitting plan](parallel_simulation_geometry_fitting_plan.md) are
accepted, including its frozen finite-bin angle-space measurement boundary. T07 through T11,
the associated selection audit, and the relevant CPU/GPU proof paths must also be accepted.

The current task graph assigns T12 a detector-native fit, places the angle transform downstream,
and prohibits angle-coordinate code in T12. Before implementation, choose and document exactly one
acyclic resolution:

1. move the accepted angle transform before a formally revised T12 and update its contracts,
   dependencies, prompt, and owned paths; or
2. keep T12 detector-native and assign this plan a separate task after the accepted angle phase.

Under either resolution, this fitter consumes prepared angle profiles. It does not implement
detector-to-angle geometry, redefine branches, or create a second projection path.

## Goal

Recover the three parameters of the shared wrapped mosaic distribution from finite-bin
`I(phi)` profiles:

```text
Gaussian standard deviation       sigma
Lorentzian half-width             Gamma
Lorentzian probability mass       eta
```

The first proof uses genuinely synthetic Bi2Se3 image triplets at exactly 5, 10, and 15 degrees
incidence. It performs three blinded fits:

1. every eligible `(m,L)` reflection group;
2. a frozen nested half of those groups; and
3. a frozen nested quarter of those groups.

Every selected nonzero-`m` incidence-group observation retains both physical branches. An eligible
`m=0` group retains one branchless `COLLAPSED_00L` profile per included incidence dataset. The
calculation reports `(m,L)` for scientific interpretation, but immutable reflection-group and rod
identities remain the computational keys.

The estimator must return `not identifiable` when a requested subset cannot constrain all three
active parameters. Producing three numbers is not a success criterion.

## Non-goals

- Do not vary source, detector, sample, goniometer, lattice, or selection parameters.
- Do not infer or change rod, reflection-group, or branch identity inside an objective.
- Do not reconstruct detector-to-angle coordinates in the fitting package.
- Do not replace the accepted continuous-rod forward calculation with a sphere-only intensity
  model.
- Do not use a free nonparametric distribution as the primary initializer.
- Do not choose truth parameters, optimizer starts, or group subsets randomly.
- Do not treat weighted stochastic detector mass as independent Poisson counts.
- Do not add a generic CPU/GPU backend layer or a new dependency.
- Do not commit generated images, profile banks, broad sweeps, or benchmark output.

## Governing decision

Use a direct response-folded, separable fit. Do not first unfold an arbitrary mosaic curve and then
project it onto three parameters.

For each exact reflection group, incidence angle, physical branch, and `phi` bin, the accepted
forward path produces a finite-bin expected profile. Each Bragg-sphere intersection samples a
different part of the common tilt distribution. The joint objective pieces those sections together
by fitting one shared `(sigma, Gamma, eta)` to all response-folded profiles at once.

Bragg-sphere radii and intersections provide analytic coverage, support, turning-point, and subset
design information. They also provide an independent geometry oracle. They do not own production
intensity, optics, detector deposition, or branch identity.

## Scientific and numerical contract

### Authoritative distribution and measure

Use the authoritative wrapped component densities from `sampling/mosaic.py` without copying their
equations into the fitter. In notation only,

\[
\omega(\delta;\sigma,\Gamma,\eta)
=
(1-\eta)G_{\rm w}(\delta;\sigma)
+\eta L_{\rm w}(\delta;\Gamma),
\]

where `sigma` is a Gaussian standard deviation, `Gamma` is a Lorentzian half-width, and `eta` is
probability mass rather than peak-height fraction.

For folded tilt `alpha` in `[0, pi]` and uniform orientation azimuth `beta`, the accepted measure is

\[
dP
=
2\omega(\alpha)\,d\alpha\,\frac{d\beta}{2\pi}.
\]

No additional `sin(alpha)` factor is applied to this direct-alpha quadrature. The event coarea
Jacobian remains a separate once-only factor. An active zero-width component is a discrete atom at
`alpha=0`, not a continuous-density evaluation.

### Exact pure-component decomposition

The current accepted quadrature uses width-dependent direct-alpha panels. Therefore this plan does
not assume that a parameter-independent latent response matrix already exists.

Instead, for each requested Gaussian width, evaluate a pure-Gaussian conditional expected profile
through the accepted forward path:

\[
\mathbf g_o(\sigma).
\]

For each requested Lorentzian half-width, evaluate a pure-Lorentzian conditional expected profile:

\[
\mathbf l_o(\Gamma).
\]

For observation block `o`, linearity of the probability mixture and conditional expectation gives

\[
\boldsymbol\mu_o(\sigma,\Gamma,\eta)
=
(1-\eta)\mathbf g_o(\sigma)
+\eta\mathbf l_o(\Gamma).
\]

This is exact for the conditional expected signal when source samples, geometry, detector
projection, angle grid, masks, profile support, and normalization are frozen. It is not an identity
between individual stochastic images. A stochastic mixed-truth image must be rendered as a mixed
distribution, not assembled from two realized component images.

### Finite-bin angle-profile observable

For each fixed branch profile and `phi` bin, sum the angle-space signal numerator and normalization
over the frozen `2theta` band before division:

\[
S_{o,k}=\sum_{q\in R_{o,k}} S_q,
\qquad
N_{o,k}=\sum_{q\in R_{o,k}} N_q,
\qquad
I_{o,k}=\frac{S_{o,k}}{N_{o,k}}.
\]

The accepted profile boundary must retain `S`, `N`, the valid mask, and all revisions. Summing an
already divided two-dimensional field is not equivalent when `N` varies.

The pure-component mixture remains linear in `I` only because `N`, the bin grid, and the valid mask
are fixed and mosaic-independent. Unit-area normalization is nonlinear and must not be applied to
each component before mixing. The initial estimator uses an analytic nuisance scale instead, so it
does not need unit-area normalization.

Every immutable profile observation must carry at least:

```text
dataset and incidence-angle identity
exact ReflectionGroupKey and member rod IDs
branch_id 0 or 1, or None with COLLAPSED_00L
phi edges and centers, seam, and local unwrap origin
summed signal and normalization arrays
fixed valid-bin mask and fixed 2theta support
variance, covariance, or declared expected-field weighting
exposure and correction ledger
source, instrument, geometry, selection, projector, grid, and ROI revisions
```

If the preceding phase does not expose this finite-bin contract, implementation stops for the
smallest reviewed upstream contract addition. The fitting module must not reconstruct it privately.

### Nuisance intensity and objective

For the controlled mocks, relative image exposures are known. The preferred scale block is one
exact reflection group containing every available incidence dataset and both nonzero-`m` branches
at each included incidence, or the one branchless `m=0` profile at each included incidence. This
retains cross-angle and cross-branch redistribution.

One shared scale is valid only when the accepted templates include every exact rod contribution and
all relative exposure, optics, and branch-response factors. A display label such as `(m,L)` alone
does not prove that scale sharing is valid. Individual rods remain distinct through projection.

Assemble one globally ordered profile vector for each rendered image/pool, covering every selected
group, branch, and valid bin. Concatenate all image vectors without discarding covariance between
groups or shared-source images. Let `A(theta)` contain one nonnegative nuisance-scale column per exact reflection
group, including known relative exposure factors and zeros outside that group. With a fixed
whitening operator `L`, eliminate all scales jointly:

\[
\mathbf a^*(\theta)
=
\operatorname*{argmin}_{\mathbf a\ge 0}
\left\|
L\left[\mathbf y-A(\theta)\mathbf a\right]
\right\|_2^2.
\]

Use the scalar closed form only after proving the covariance is block diagonal across the proposed
scale blocks. For one such block, let `W_o = L_o.T @ L_o`. With no additive background and a
denominator above its frozen tolerance,

\[
a_o^*
=
\max\!\left(
0,
\frac{\boldsymbol\mu_o^{\mathsf T}W_o\mathbf y_o}
{\boldsymbol\mu_o^{\mathsf T}W_o\boldsymbol\mu_o}
\right).
\]

The profiled objective is

\[
Q(\sigma,\Gamma,\eta)
=
\left\|
L\left[\mathbf y-A(\sigma,\Gamma,\eta)\mathbf a^*\right]
\right\|_2^2.
\]

The primary mock has exactly zero background. If a later dataset requires a frozen linear
background basis, append its columns and solve the small constrained linear nuisance problem
jointly. Do not claim the scalar closed form when cross-group covariance is present, covariance
depends on trial parameters or scale, or the proposed group-by-image factors are bilinear.

Fixed-total sampling can make covariance singular. Freeze its eigenspace threshold before fitting,
remove only declared analytic or numerically converged null modes, and require positive definiteness
on the retained residual subspace. Reject a scale block when its retained
`mu.T @ W @ mu` is below tolerance. Never let a trial change the retained covariance rank.

The expected-field tier may use a predeclared deterministic weighting. The stochastic tier uses a
fixed covariance or whitening operator derived before fitting. Empty and valid zero-signal bins
remain in the residual; masks never change with the trial.

### Deterministic global search

The nonlinear problem has only three parameters. Use a deterministic global template search rather
than relying on one local start:

1. build logarithmic grids in positive `sigma` and `Gamma`;
2. evaluate the accepted pure-component profiles at those exact widths;
3. for every width pair, perform a bounded deterministic search over `eta` in `[0,1]`;
4. retain all near-optimal cells, not just one winner;
5. refine those cells with narrower exact template grids; and
6. optionally polish with exact forward component evaluations after the grid result is proven.

No final accepted value may depend on unproved interpolation between templates. Include the exact
faces `eta=0` and `eta=1`. On either face, the inactive component width is not identifiable and is
reported as inactive rather than recovered.

Report profile likelihoods or equivalent objective slices, active bounds, competing minima,
parameter correlations, Jacobian rank, and condition. A good local residual is insufficient when a
width-mixture ridge remains.

## Observation and subset identity

### Indivisible physical group

The atom used for full/half/quarter selection is one exact reflection group from the union of the
three images. It contains every available incidence-group observation:

```text
exact ReflectionGroupKey
    -> exact member rod IDs preserved
    -> available 5 degree profiles
    -> available 10 degree profiles
    -> available 15 degree profiles
    -> both branches at every included incidence for m != 0
    -> one branchless profile at every included incidence for eligible m = 0
```

A group is eligible when at least one incidence has frozen valid support and exact unambiguous
identity. For a nonzero-`m` incidence-group observation, both branches are indivisible; when either
branch is unavailable, that incidence-group observation is absent rather than half-used. Missing
incidences and detector support are recorded as absent, never as zero intensity. The selected group
sets collectively, rather than every individual group, must cover all three incidence angles.

Each `m=0` profile participates only when `phi` is well-defined away from the direct-beam
singularity, its fixed support excludes or models any incompatible specular/background behavior,
and its nuisance-projected sensitivity is nonzero and adequately conditioned. An ineligible `m=0`
profile is excluded with a recorded reason or retained only as a compatible diagnostic; it never
acquires artificial branches.

### Nested full, half, and quarter sets

Let `G` be the number of eligible physical groups. Freeze

\[
\mathcal G_{1/4}\subset\mathcal G_{1/2}\subset\mathcal G_{\rm full},
\]

with `ceil(G/4)`, `ceil(G/2)`, and `G` groups respectively. Here `full` means the union of every
eligible group seen in any of the three images. Do not enlarge a failed quarter set and still call
it a quarter.

Because the requested full fit uses every eligible group, it has no omitted-group complement. Its
out-of-sample evidence comes from the independent Tier B triplet and from a separate predeclared
leave-group-out audit: choose an auxiliary validation set before mock generation, fit its complement,
and predict it. That auxiliary audit does not redefine the requested full fit. The omitted groups
from the requested half and quarter fits are their direct held-out predictions.

Selection occurs before mock intensities are generated or inspected. It uses only accepted geometry,
response templates over a broad predeclared parameter grid, and uncertainty design values.

For transformed active parameters

\[
(\log\sigma,\log\Gamma,\operatorname{logit}\eta),
\]

form the covariance-whitened sensitivity after projecting out all permitted nuisance columns. A
legal three-parameter subset must have practical rank three on the interior parameter grid. Score
legal subsets by worst-case smallest singular value, then by condition and log determinant.

Require connected coverage of the core, core-to-tail transition, shoulder, and far tail; collective
coverage of all three incidence angles; diverse sphere radii; and both `m=0` and nonzero-`m`
information when the former is valid. Optimize the nested quarter/half pair jointly so a locally
best quarter cannot force a poor half. For small `G`, enumerate all legal nested pairs. For larger
`G`, use deterministic nested greedy construction followed by pair exchange with a frozen tie order.

Boundary faces use their correct reduced active dimension. If the quarter or half set does not pass
rank, coverage, or conditioning, record `not identifiable` and use its profiles only for the
corresponding failure evidence.

## Blinded chosen-truth protocol

No mosaic truth value is generated randomly. The primary mixed case is prescribed as:

```text
gaussian_sigma         = 0.45 degree
lorentzian_half_width  = 2.00 degree
lorentzian_probability = 0.22
```

The values are stored in radians in the numerical configuration. The additional required proof
regimes are prescribed before data generation:

```text
narrow:
    gaussian_sigma         = 0.20 degree
    lorentzian_half_width  = 1.00 degree
    lorentzian_probability = 0.10

tail_dominated:
    gaussian_sigma         = 0.60 degree
    lorentzian_half_width  = 3.00 degree
    lorentzian_probability = 0.70
```

The primary deliverable is the mixed triplet. The additional regimes satisfy the mosaic-model proof;
they do not alter full/half/quarter subset selection.

For every regime:

1. freeze source, geometry, selection, angle grid, profile support, correction ledger, and
   quadrature settings;
2. generate exactly 5, 10, and 15 degree synthetic observations through the accepted forward path;
3. store truth separately from a sanitized fit input;
4. audit the sanitized input recursively for truth parameters or truth-derived start values;
5. run the same deterministic search for full, half, and quarter inputs;
6. freeze all estimates, diagnostics, and held-out predictions; and
7. only then compare with truth.

The tracked Bi2Se3 example currently uses 12 degrees for its third configured incidence despite
the measurement filename. Do not reuse or relabel it. Create a new explicit 5/10/15 synthetic
truth configuration during implementation.

Two recovery tiers are mandatory:

- Tier A uses the same deterministic source and quadrature revision for truth and prediction to
  isolate estimator correctness.
- Tier B uses a denser or independently seeded fixed source/orientation revision for truth so that
  recovery is not merely an identical-discretization inversion.

## Pre-implementation feasibility evidence

The estimator algebra was exercised with finite-bin toy responses before this plan was written.
These checks used no repository renderer and are not production proof.

- A correctly specified branch-split toy with truth `(1.35 deg, 6.5 deg, 0.18)` recovered all
  three parameters from quarter, half, and full nested sets. Width errors were below `0.016 deg`,
  mixture error was below `1.3e-4`, and local conditions were approximately 5.9 to 6.4.
- An independent sphere-intersection toy with truth `(0.45 deg, 2.0 deg, 0.22)` recovered the
  exact solution from all three subset sizes, with conditions approximately 12 to 17.
- A noisy finite-sample toy with truth `(0.010 rad, 0.035 rad, 0.22)` returned approximately
  `(0.009975 rad, 0.03480 rad, 0.22075)`.
- A tail-only subset had rank one and admitted a convincing false three-parameter fit. This is why
  rank and connected-coverage gates are mandatory.
- A 25 percent response-blur mismatch moved the toy solution from `(1.35 deg, 6.5 deg, 0.18)` to
  roughly `(1.78 deg, 6.99 deg, 0.24)`.
- Small incidence-angle errors also biased the recovered widths and mixture. Geometry and source
  revisions therefore remain frozen and must pass held-out checks.
- Trial-dependent removal of active bins changed vector length and failed immediately. Every bin,
  mask, group, and residual slot is frozen before the first evaluation.

The conclusion is conditional: direct response-folded fitting works when the response is correct
and the selected profiles have rank and connected coverage. The first production proof remains a
comparison of direct mixed expected profiles with the pure-component construction.

## Dependency graph

```text
accepted detector-native forward model
    -> accepted CPU ray/candidate execution and deterministic expected profiles
    -> accepted selection identities and fit contracts
    -> accepted detector and sample geometry
    -> accepted detector-to-angle signal/normalization operator
    -> immutable finite-bin angle-profile contract
        -> Phase 0 contract and ownership gate
        -> Phase 1 5/10/15 profile and group manifest
        -> Phase 2 analytic sphere-coverage oracle
        -> Phase 3 pure-component expected-profile bank
        -> Phase 4 profiled objective and deterministic global search
        -> Phase 5 nested information-based subset design
        -> Phase 6 blinded expected-field recovery
        -> Phase 7 stochastic weighted-image recovery
        -> Phase 8 adequacy, performance, cleanup, and handoff
```

## Planned file and ownership strategy

If this work is assigned to the existing mosaic fitting module, the preferred numerical paths are:

```text
src/rasim_next/fitting/mosaic.py
tests/test_fitting.py
the assigned task document and prompt
```

One small chosen-truth configuration is planned:

```text
examples/bi2se3/experiment/mosaic_fit_truth.toml
```

Generated images, extracted profile arrays, template banks, broad subset studies, convergence
sweeps, and performance output remain external proof artifacts.

Before source work, make explicit reviewed dependency requests for:

- the finite-bin angle-profile contract if it is absent from the accepted predecessor;
- the task-index and prompt dependency update needed to consume that boundary; and
- the minimal proof-command registration required by the assigned task if it is not already
  available; and
- ownership of the chosen-truth configuration, or an external truth-manifest policy when example
  ownership is not approved.

Do not silently edit shared contracts, task ordering, example ownership, or proof dispatch from the
fit branch. Add no production module beyond `fitting/mosaic.py` unless a repeated need is
demonstrated and separately approved.

# Phase 0: freeze authority, contracts, and tolerances

## Goal

Prove that the assigned fit task can consume every needed immutable input without owning angle geometry or changing
upstream physics.

## Work

1. Record exact accepted SHAs and revisions for source, instrument, geometry, rod catalog,
   selection, detector response, angle projector, grid, and profile extractor.
2. Confirm the predecessor exposes finite-bin profile signal, normalization, mask, covariance
   metadata, and exact identities.
3. Confirm mosaic changes invalidate only mosaic probability and dependent expected profiles; all
   upstream revisions remain fixed.
4. Freeze width bounds, eta bounds, quadrature stages, profile bins, covariance policy, rank
   threshold, condition threshold, and recovery tolerances before generating fit results.
5. Assign an acyclic task position, ownership, and dependencies in repository planning documents
   before editing source.
6. Verify the requested `mosaic-fit` proof command has an approved minimal registration path.

## Acceptance criteria

- [ ] No fitting code is responsible for detector-to-angle conversion.
- [ ] Every observation and template carries matching immutable revisions.
- [ ] Width conventions and the folded direct-alpha probability measure are explicit.
- [ ] All masks, residual slots, nuisance blocks, and validity rules are trial-independent.
- [ ] Proof tolerances are derived from convergence and sensitivity, then frozen before recovery.
- [ ] Any missing shared field produces one specific dependency request rather than a private copy.

## Verification

- Schema round trip and hash stability.
- Deliberate source, geometry, selection, grid, projector, and ROI revision mismatches are rejected.
- A mosaic-only parameter change leaves every declared upstream revision unchanged.

## Stop conditions

Stop `BLOCKED` if the angle profile cannot be reproduced from accepted `S`, `N`, mask, and
identity metadata, or if task ownership still requires the fitter to implement angle geometry.

# Phase 1: define the 5/10/15 observation and group manifests

## Goal

Define all eligible profiles and indivisible full/half/quarter group atoms before mock intensity is
available.

## Work

1. Add a new Bi2Se3 synthetic case with incidence angles exactly 5, 10, and 15 degrees.
2. Use the accepted geometry and selection manifest to enumerate exact reflection groups and member
   rods visible in the union of the three images.
3. For nonzero `m`, require both physical branches at every included incidence; record an unsupported
   incidence as absent rather than discarding the whole group or using one branch.
4. For `m=0`, retain `branch_id=None` and apply the frozen direct-beam/azimuth conditioning gate.
5. Freeze `2theta` support, `phi` edges, local seam/unwrap origin, normalization support, valid
   bins, and correction ledger for every profile.
6. Define the scale-sharing blocks and prove known relative mock exposures.
7. Freeze the auxiliary leave-group-out validation set and all available-incidence metadata; never
   reserve individual branches from a pair.

## Acceptance criteria

- [ ] The third incidence is numerically 15 degrees, not inferred from a filename.
- [ ] Computational identity uses exact reflection groups and rods rather than floating `Qr` or a
      display label.
- [ ] Every included nonzero-`m` incidence-group observation contains both branches, and the full
      union contains every eligible group seen in any image.
- [ ] Every eligible `m=0` atom remains branchless and has a valid `phi` observable.
- [ ] Each accepted subset collectively contains usable profiles from all three incidence angles.
- [ ] Profile support and residual size cannot change after the manifest is frozen.
- [ ] Auxiliary held-out membership is fixed before truth rendering.

## Verification

- Hash the group/profile manifest under input-order changes.
- Inject a branch swap, rod collapse, missing dataset, moving profile center, and direct-beam
  singularity; each must fail during preparation.

Dependencies: Phase 0.

# Phase 2: build the analytic sphere-coverage oracle

## Goal

Use Bragg-sphere geometry to describe which tilt regions every branch set observes and independently
validate finite-bin support.

## Work

1. From accepted incident wavevectors, reciprocal geometry, sphere radii, and detector geometry,
   determine each group's accessible tilt intervals and physical outgoing branches.
2. Locate tangent points, multiple preimages, seam crossings, detector clipping, and gaps in
   coverage.
3. Integrate simple test densities into finite `phi` bins by independent scalar enumeration.
4. Build a group-by-tilt-band coverage summary for core, transition, shoulder, and far tail.
5. Compare sphere support and turning points with accepted continuous-rod roots and projected
   profile support.

Do not divide measured `I(phi)` pointwise by `dphi/dalpha`. Tangencies and multiple preimages make
that inversion unstable. Integrate the forward finite-bin response instead.

## Acceptance criteria

- [ ] The oracle predicts support, branch count, turning points, and bin ownership independently.
- [ ] All finite-bin preimages are included at tangencies and seam crossings.
- [ ] Sphere roots are not mislabeled as physical branch IDs.
- [ ] The oracle never supplies production optical or intensity factors.
- [ ] Each eligible group has an explicit tilt-coverage record.

## Verification

- Compare analytic intersections with dense scalar enumeration.
- Compare finite-bin support with accepted continuous-rod expected profiles.
- Exercise regular, tangent, no-root, seam-crossing, clipped, and branchless fixtures.

Dependencies: Phase 1.

# Phase 3: compile pure-component expected profiles

## Goal

Create the smallest reusable template bank using the accepted expected forward path.

## Work

1. Choose predeclared logarithmic Gaussian-width and Lorentzian-half-width grids within safe bounds.
2. For each Gaussian width, evaluate all frozen profiles with a pure Gaussian probability model.
3. For each Lorentzian half-width, evaluate all frozen profiles with a pure Lorentzian model.
4. Retain numerator, normalization, derived profile, mass, and conditioning diagnostics for every
   exact observation block.
5. Key every template by width, component, source, geometry, selection, projector, grid, ROI,
   quadrature, precision, and summation revisions.
6. Compare exact convex combinations with independently evaluated mixed expected profiles.
7. Repeat at increasing source count, direct-alpha panel count, azimuth count, and profile-bin
   resolution; freeze the final stage before recovery.

The bank stores only selected finite-bin profiles and small diagnostics. It does not retain a giant
candidate-by-bin array.

## Acceptance criteria

- [ ] Pure templates use the same authoritative component probability and event physics as a mixed
      forward evaluation.
- [ ] `(1-eta) g + eta l` reproduces direct mixed expected `S`, `N`, `I`, and valid masks within
      frozen tolerance.
- [ ] `N` is invariant to mosaic parameters for the frozen observable.
- [ ] No unit-area normalization occurs before component mixing.
- [ ] Width-dependent quadrature remains deterministic and its density changes only between
      declared convergence stages.
- [ ] No final fit relies on unproved width interpolation.

## Verification

- Test interior eta values, `eta=0`, `eta=1`, narrow widths, broad tails, and a zero-width atom
  limit where supported.
- Inject an extra spherical factor, duplicate coarea factor, peak-height mixture, pre-mix profile
  normalization, moving mask, and stale template key; each must fail at its first affected stage.

Dependencies: Phases 1-2 and the accepted expected-profile forward path.

# Phase 4: implement the profiled objective and deterministic search

## Goal

Recover a globally credible initial estimate without random starts or intensity compensation.

## Work

1. Assemble fixed profile vectors in canonical dataset/group/branch/bin order.
2. Build fixed whitening operators from the declared expected-field or stochastic covariance model.
3. Profile all valid nonnegative reflection-group scales jointly by constrained GLS, with zero
   background for the initial mocks; use independent scalar formulas only after block-diagonal
   covariance is proven.
4. Evaluate all coarse width pairs and a bounded deterministic eta search.
5. Retain near-optimal cells and refine exact width templates until the predeclared physical
   resolution is reached.
6. Optionally polish from every retained cell using exact component evaluations; accept polishing
   only when all starts converge to the same basin or all basins are reported.
7. Compute objective slices, nuisance-projected sensitivities, rank, condition, correlations,
   active bounds, and held-out predictions.
8. Handle `eta=0` and `eta=1` as reduced models with one inactive width.

## Acceptance criteria

- [ ] Repeated evaluation is deterministic and fixed-length.
- [ ] Exact nuisance projection matches direct constrained minimization.
- [ ] Covariance null modes, retained rank, and every scale denominator are frozen and valid.
- [ ] No geometry, selection, support, or covariance field is optimized.
- [ ] Global refinement resolves or reports every competitive basin.
- [ ] Boundary models never claim recovery of an inactive width.
- [ ] Invalid or rank-deficient fits return a typed refusal with evidence.

## Verification

- Scalar and joint scale cases, cross-group covariance, singular fixed-total covariance, zeros,
  all-masked rejection, eta faces, width bounds, duplicated profiles, and competing-minimum fixtures.
- Compare the grid/refinement result with an independent dense three-parameter toy objective.

Dependencies: Phase 3 and the T09 fitting foundation.

# Phase 5: freeze information-based full/half/quarter designs

## Goal

Choose nested subsets that maximize recoverable mosaic information without looking at mock
intensities.

## Work

1. Evaluate nuisance-projected, whitened parameter sensitivities for every physical group over a
   broad interior parameter grid.
2. Combine those sensitivities with the Phase 2 connected tilt-coverage constraints.
3. Enumerate all legal nested quarter/half pairs when tractable; otherwise use deterministic nested
   greedy construction and pair exchange under one frozen tie order.
4. Optimize the pair jointly using the predeclared maximin singular-value, condition, coverage, and
   log-determinant ordering; define full as the union of all eligible groups.
5. Freeze exact group lists, ranks, conditions, coverage summaries, and hashes before rendering
   truth.
6. Define the omitted groups from half and quarter as their held-out prediction data, and freeze the
   separate auxiliary leave-group-out audit used to test group generalization for the full model.

## Acceptance criteria

- [ ] `quarter` is a subset of `half`, and `half` is a subset of `full`.
- [ ] Branches and incidence images are never split away from their physical group atom.
- [ ] Every accepted subset collectively covers 5, 10, and 15 degrees.
- [ ] Selection uses geometry/templates/design covariance only, never observed truth intensity.
- [ ] Every accepted three-parameter subset has practical rank three and connected coverage.
- [ ] A failed requested fraction is reported as unidentifiable without changing its size or name.

## Verification

- Exhaustive comparison for a small dummy group pool.
- Deliberately tail-only, core-only, disconnected, duplicate-radius, invalid-`m=0`, and nearly
  collinear sensitivity cases.

Dependencies: Phases 2-4.

# Phase 6: perform blinded expected-field recovery

## Goal

Prove estimator correctness before stochastic image variation is introduced.

## Work

1. Generate direct mixed expected profiles for the prescribed 5/10/15 truth triplet; do not form
   the truth by combining the fitter's stored component profiles.
2. Produce a sanitized fit package with only observations, covariance policy, frozen manifests,
   bounds, and prescribed nominal search configuration.
3. Fit full, half, and quarter independently with the same deterministic algorithm.
4. Freeze estimates, objective surfaces, rank/condition, active bounds, and applicable held-out
   predictions.
5. Reveal truth and compute parameter error in radians and named width conventions.
6. Repeat the primary case under Tier B discretization and run the prescribed narrow and
   tail-dominated proof regimes.
7. Run the predeclared auxiliary leave-group-out fit and predict its untouched groups; keep this
   separate from the requested full fit, which uses every eligible group.

## Acceptance criteria

- [ ] Full expected-field recovery passes the frozen parameter and profile tolerances.
- [ ] Half and quarter either pass their frozen tolerances or were rejected by the pre-fit
      information gate.
- [ ] Applicable held-out group profiles are predicted within frozen final-observable tolerance.
- [ ] The full fit passes independent Tier B prediction, and the auxiliary audit passes untouched
      group prediction without redefining `full`.
- [ ] Tier B demonstrates convergence rather than identical-discretization memorization.
- [ ] Geometry, selection, masks, normalization, and nuisance grouping remain byte-for-byte stable.
- [ ] Truth is absent from fitter inputs and start construction.

## Verification

- Nominal truth replay, deliberately displaced deterministic bounds/grids, independent quadrature,
  held-out groups, and recursive truth-leak audit.

Dependencies: Phases 1-5.

# Phase 7: recover from actual weighted stochastic images

## Goal

Show that the same estimator remains calibrated under both Monte Carlo levels: the initially drawn
source-ray pool and the subsequent complete-pool mosaic/event selections that must hit the detector
before contributing.

## Work

1. Render each mixed 5/10/15 image through complete-pool event selection, candidate-specific hits,
   conservative deposition, and the accepted angle-profile extraction.
2. Condition every individual fit on one frozen source-sample revision, using common random numbers
   across its trial evaluations.
3. For uncertainty and coverage, run a predeclared nested design: independent outer source-ray
   batches and independent inner event-selection batches within each outer batch.
4. Record whether one outer source draw is shared across the three incidence images. Preserve the
   resulting cross-image covariance when it is shared.
5. Derive conditional inner covariance from the actual selection design or estimate it from
   independent inner batches.
6. Retain cross-group, cross-branch, and cross-bin covariance induced by event selection,
   deposition, and angle splitting.
7. Combine outer and inner variation by the law of total covariance, then freeze the resulting
   whitening operator before fitting; do not update it with trial parameters.
8. Fit full, half, and quarter across the predeclared nested replicate set and report bias, spread,
   coverage, failure rate, and held-out prediction.
9. Compare high-ray stochastic means with the conditional expected profiles from Phase 3.

For independent with-replacement inner draws from one complete pool with total mass `T`, candidate
probability `p_i`, and `Ndraw` selected events, let `z_i` be the globally ordered contribution
vector after the fixed deposition, angle-profile operator, and normalization, covering every
selected group, branch, and bin in that pool. Conditional on the frozen source pool, the reference
covariance is

\[
\operatorname{Cov}(Y)
=
\frac{T^2}{N_{\rm draw}}
\left[
\sum_i p_i\mathbf z_i\mathbf z_i^{\mathsf T}
-\bar{\mathbf z}\bar{\mathbf z}^{\mathsf T}
\right],
\qquad
\bar{\mathbf z}=\sum_i p_i\mathbf z_i.
\]

Sum independent pool covariances. If the accepted selector uses stratified or correlated uniforms,
derive that design's second moment or use independent seed batches instead of this formula.

For random source pool `S`, include the outer level:

\[
\operatorname{Cov}(Y)
=
\mathbb E_S[\operatorname{Cov}(Y\mid S)]
+\operatorname{Cov}_S(\mathbb E[Y\mid S]).
\]

Estimate the second term from independent source-pool batches unless an exact source-sampling
second moment is available. Do not report marginal coverage from the conditional inner covariance
alone.

## Acceptance criteria

- [ ] No weighted profile is labeled or fitted as independent Poisson data.
- [ ] Covariance matches both accepted sampling levels and includes cross-profile correlations.
- [ ] Shared-source cross-image covariance is retained when present.
- [ ] Worker count, chunking, and eligible CPU/GPU execution do not change fixed-seed observations.
- [ ] Stochastic means converge to the accepted conditional expectation.
- [ ] Full/half/quarter uncertainty and failure behavior agree with the information audit.
- [ ] Held-out prediction does not worsen while only the training objective improves.

## Verification

- Analytic covariance versus replicate covariance for a tiny conditional independent-draw pool.
- Nested outer-source/inner-event batches for the production selector.
- Increasing ray counts, multiple fixed seeds, CPU chunk layouts, and eligible GPU execution.

Dependencies: accepted Phase 6 and the complete accepted predecessor plan.

# Phase 8: model adequacy, performance, and handoff

## Goal

Demonstrate that the three-parameter model is adequate, retain only useful acceleration, and leave a
compact proven implementation.

## Work

1. Use held-out residual structure as the required model-adequacy check.
2. Only when systematic held-out failure triggers a declared adequacy investigation, optionally fit
   a coarse nonnegative distribution as an external proof study, and only after an accepted fixed
   basis-template response is available without creating a second physics path or hidden contract.
3. If that optional diagnostic is run, require connected response support, choose smoothing from
   held-out groups, and report unresolved relative mass in disconnected sections.
4. Never give the optional diagnostic a production API or permanent test, make it a completion
   blocker, or feed it back into the official initializer.
5. Profile pure-template generation, mixed-profile assembly, nuisance projection, global search,
   covariance work, and held-out prediction separately.
6. Use the predecessor's ray-block/candidate-tile scheduler as the primary parallel axis; do not
   schedule by peak, rod, or HKL count.
7. Treat independent widths only as outer orchestration under one global worker/device budget so
   width batching never nests another pool over the ray scheduler.
8. Batch width pairs and eta values only when this reduces measured time and preserves deterministic
   reduction.
9. Keep templates and small profile arrays resident on a GPU only above the measured crossover.
10. Remove temporary scripts, broad fixtures, generated arrays, benchmark dumps, duplicate tests,
   unused acceleration experiments, and any dependency that did not earn its place.

The three-parameter objective may be faster on the CPU after templates are compiled. GPU
compatibility does not require forcing this small calculation onto a device. The expensive forward
template work and later larger joint fits remain able to reuse the same candidate kernel and
finite-bin profile contract.

A compiled width bank is valid only while every geometry, source, selection, projector, and profile
revision in its key is fixed. A later joint geometry-mosaic fit must invalidate and recompute the
affected incident states, events, hits, projection, and profiles; GPU residency never permits stale
template reuse.

## Acceptance criteria

- [ ] The parametric result passes held-out model-adequacy checks or is explicitly rejected.
- [ ] Any optional nonparametric output is external-only, diagnostic, and carries support limitations.
- [ ] Equivalent CPU/GPU work agrees within frozen final-observable tolerance.
- [ ] Wall time, setup/compile time, transfer time, reuse time, and peak memory are reported.
- [ ] Backend choice follows a measured crossover.
- [ ] Only compact tests protecting unique scientific contracts remain.
- [ ] Repository status contains only intended files and no generated diagnostics.

## Verification

- Run the assigned task's compile, lint, compact fitting tests, proof command, and diff checks.
- Run broad recovery, subset, convergence, stochastic, and performance studies as external proof
  artifacts.
- Run every applicable fitting error injection and record the first failing stage.

Dependencies: Phases 6-7.

## Cross-phase proof matrix

| Claim | Independent evidence | Required failure control |
|---|---|---|
| Direct-alpha probability is correct | normalization and component limits | add an extra spherical factor |
| Components mix by probability mass | direct mixed expected profile | mix peak heights or apply eta twice |
| Sphere geometry describes support | dense scalar intersections | omit a preimage or tangent bin |
| Angle profiles are well-defined | direct `S`, `N`, mask enumeration | divide before summing or move a mask |
| Identities remain physical | selection manifest and branch oracle | swap branches or collapse rods |
| Joint nuisance projection is exact | direct constrained minimization | discard cross-group covariance |
| Global estimate is credible | dense toy surface and retained basins | force one poor local start |
| Subsets are informative | projected sensitivity and held-out data | use a tail-only or disconnected set |
| Expected recovery is not an inverse artifact | Tier B source/quadrature | reuse a stale truth/template revision |
| Stochastic uncertainty is calibrated | nested source/event replicate covariance | omit outer source variation |
| Acceleration is equivalent | stage-by-stage CPU/GPU comparison | reorder nondeterministic reductions |

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Width-dependent quadrature is mistaken for a fixed response matrix | Critical | Use exact pure-component profiles through the accepted forward path |
| Sphere intersections become a competing intensity model | Critical | Restrict them to coverage, support, design, and independent proof |
| `eta` is mixed as peak amplitude | Critical | Use authoritative probability masses and direct mixed-profile proof |
| Profile normalization destroys component linearity | High | Sum `S` and `N`, mix components, and profile proven group scales jointly |
| Independent branch scales erase redistribution | High | Share a proven exact-group scale across branches and known exposures |
| Quarter coverage cannot identify all parameters | High | Pre-fit rank/coverage gate and typed refusal |
| `m=0` azimuth is singular | High | Frozen conditioning gate, exclusion, or compatible diagnostic-only use |
| Geometry/source error biases mosaic widths | Critical | Freeze accepted revisions and require Tier B plus held-out prediction |
| Initial source-ray variation is omitted from uncertainty | Critical | Use outer source batches plus inner event batches |
| Tail bins dominate through noise or zeros are dropped | High | Fixed covariance-aware residual with valid zeros retained |
| A stochastic realization is linearly mixed from component images | Critical | Render the full mixed truth; use linearity only for conditional expectation |
| Trial-dependent masks change objective dimension | Critical | Freeze every profile bin and residual slot before fitting |
| Local search lands on an eta-width ridge | High | Deterministic global grid, retained basins, and profile likelihoods |
| GPU setup exceeds the useful work | Medium | Measure crossover and retain CPU execution for small fits |
| Large candidate response storage causes bloat | High | Store only finite-bin component profiles and stream candidate tiles |

## Final completion gate

- [ ] The work has an assigned acyclic task position and approved owned paths after the accepted
      angle-profile producer.
- [ ] The exact 5/10/15 Bi2Se3 synthetic triplet exists and is not a relabeled 12-degree case.
- [ ] Source, geometry, selection, angle grid, profile support, corrections, and normalization are
      immutable throughout every fit.
- [ ] One accepted forward path produces all pure and direct mixed expected profiles.
- [ ] The Gaussian width, Lorentzian half-width, and mixture probability retain their authoritative
      conventions and folded probability measure.
- [ ] Direct mixed expected profiles equal the convex pure-component construction within frozen
      tolerance.
- [ ] Every included nonzero-`m` incidence-group observation retains both branches; every `m=0`
      observation remains branchless; and full contains every eligible group in the three-image
      union.
- [ ] Quarter is nested in half, half is nested in full, and selection precedes truth intensity.
- [ ] Every accepted three-parameter subset passes rank, conditioning, and connected-coverage gates.
- [ ] Full, half, and quarter recover chosen truth or explicitly report non-identifiability.
- [ ] Half and quarter predict omitted groups; full passes independent Tier B prediction; and the
      separate auxiliary leave-group-out audit predicts untouched groups.
- [ ] Weighted stochastic image recovery includes both source-pool and event-selection variation
      through the actual covariance or nested independent-batch evidence, never a Poisson shortcut.
- [ ] CPU and eligible GPU paths reproduce the same finite-bin observable.
- [ ] Broad studies remain external; only the compact permanent proof suite remains.
- [ ] No detector-to-angle physics, duplicate mosaic equation, generic backend, generated output,
      or unresolved development residue is added to the assigned fit task.
