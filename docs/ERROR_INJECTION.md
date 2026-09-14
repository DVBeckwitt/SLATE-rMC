# Error-injection and tolerance policy

T34 rejection controls cover an insufficient frozen Q/angular envelope, changed spectral
weights including zero-line endpoints, changed detector geometry, changed raw acquisition
hashes, malformed or duplicate worker rows, wrong-N/forward-identity replay, and interrupted
native rendering. Required behavior is direct parity or explicit rejection, never stale
reuse, changed net counts, lost completed predictions or double-counted image batches.
The physical fitting thresholds in PHY-FIT-025 remain unchanged.

A plausible image is not proof. Every scientific proof must detect bounded mistakes at the first
affected stage while the unmodified calculation passes.

## Tolerances

Stage tolerances are declared before inspecting candidate disagreement. For scalar or array `x`,

```text
|x_candidate - x_reference| <= atol_stage + rtol_stage * declared_scale.
```

The scale comes from physical inputs, analytic bounds, or immutable reference values, never from
the candidate magnitude. Exact IDs, order, shape, dtype, status, branch, rod key, and index maps use
exact equality. Near-zero quantities require a positive unit-bearing absolute tolerance. Complex
amplitudes remain complex; intensities are compared before clipping, normalization, or log display.

Convergence is claimed only for a real refinement variable and must be checked at both the local
stage and final observable. An optimized CPU/CUDA path is compared with the readable oracle and an
independent identity where available. A `CORRECTED` legacy comparison is used only through its
declared first-divergence stage.

## Required controls

### Coordinates and geometry

- Rotate OSC data in the wrong direction, transpose it, swap row/column, or shift the pixel-center
  convention.
- Reverse transform order, use the wrong pivot, translate a vector, or alter one transformed
  coordinate after a rigid transform.
- Reverse detector basis handedness or either detector-tilt sign.

Expected detection: OSC mapping, rigid transform, detector ray, or detector round-trip.

### Source and incident transport

- Duplicate or omit a source stratum.
- Uniformize a declared weighted spectral mixture, alter or omit a line mass, couple line identity
  to source geometry, or reverse/swap a declared position--divergence correlation.
- Recompute or slice a parent revision instead of preserving the complete realization.
- Let detector-only state invalidate incident `ki`.
- Accept a changed sample pose through the detector-projection-only rebind, or copy transport arrays
  during a detector-only revision.
- Erase accepted geometry when a later optical stage fails.

Expected detection: exact strata and line masses, configured weighted moments and conditional
means, revision ownership, causal invalidation, or status payload.

### Optics

- Choose the opposite complex-normal branch or propagation sign.
- Substitute the old power-transmittance average for the scalar field amplitude.
- Omit or duplicate entrance/exit transmission or attenuation.
- Use full thickness independently for entrance and exit rather than the uniform-depth average.
- Use a complex wavevector directly in real elastic geometry.
- Omit or duplicate the incident illuminated-path factor, use refracted/detector-ray `z` instead of
  the sampled incident sample-frame `z`, or hide it inside footprint acceptance or depth decay.
- Omit, duplicate, or square external detector-path attenuation; use panel-normal distance; accept
  scalar and table coefficients together; interpolate/fallback an unlisted wavelength; or attach a
  nonzero coefficient to the declared unity medium.

Expected detection: mode dispersion, tangential conservation, amplitude, analytic csc and
Beer--Lambert identities, exact wavelength lookup, once-only CPU/CUDA/oracle parity, or exit
geometry before detector integration.

### Mosaic, rods, and Ewald restriction

- Omit the folded-alpha factor, duplicate a signed beta branch, or independently renormalize mixture
  components.
- Collapse equal-family rods before evaluating their physical structure strengths.
- Use the wrong `u`/`L` scale, omit population, or sum independent rod amplitudes.
- Choose the wrong analytic root, emit a tangent as regular, omit the Ewald coarea factor in the
  latent Ewald-restriction route, or apply it twice.
- Accept an elastic residual outside tolerance.

Expected detection: probability normalization/moments, per-rod sum, quadratic-root oracle, elastic
closure, or latent/coating agreement.

### Intrinsic Ewald outgoing-direction density

- Use the air wavenumber instead of the internal-film magnitude, or omit/apply the `k_film^2`
  solid-angle surface Jacobian twice.
- Apply the forward Ewald coarea factor again after inverse pushforward, or insert an undeclared
  `sin(alpha)` mosaic-measure factor.
- Drop an inverse preimage, retain only one analytic branch, or treat branch identity as the sign of
  sample-frame `Qz`.
- Apply an exit-valid, detector-visibility, positive-`Qz`, or panel mask to the complete intrinsic
  sphere.
- Admit the separately owned `m=0` direct root, hide a positive-numerator `|w*x|=0` caustic,
  replace it with a finite interpolated value, or assign `+inf` to a zero-numerator fold instead of
  the zero a.e. representative.

Expected detection: the independent finite-difference solid-angle Jacobian, known four-preimage
branch sum, positive/negative-`Qz` support, exact Ewald closure, explicit caustic contract, and the
detector-to-sphere change-of-measure identity.

### Ordered, stacking, and reflectivity

- Normalize a reflection to 100, round it, prune a weak rod, or fabricate a fractional reflection.
- Omit occupancy, anomalous scattering, displacement, layer phase, or registry phase.
- Reverse the transition convention, use the wrong finite-layer exponent, or mix parent amplitudes.
- Reuse the 2H same-gauge coefficient for nonzero-epsilon 3R, or change the exact epsilon-zero path.
- Reuse a different complex-normal branch in Parratt or blend outside the named handoff.

Expected detection: raw amplitude, systematic absence, finite-stack/direct enumeration,
normalization, or reflectivity limit.

### Continuous detector and pixels

- Use `Q=ki-kf`, transform Q twice, or use air `kf` where film `kf` is required.
- Drop an inverse latent branch, rod, retained root, source state, or wavelength.
- Apply source, phase, polarization, optical, surface-Jacobian, or Ewald factor zero or two times.
- Apply the illuminated-path or external detector-path factor zero or two times in any NumPy, CPU,
  CUDA, Monte Carlo, sparse-response, or local-`m=0` path.
- Multiply detector solid-angle metadata into the raw field again.
- Swap detector row/column, reverse a tilt, evaluate outside the active panel, or assign intensity to
  an invalid/back-facing ray.
- When deterministic pixel mass is requested, replace the box integral with a center sample,
  undeclared histogram, blur, or point deposit. A separately declared display-only center-density
  sample or API-v12 stochastic estimate is not this error.
- In the stochastic estimator, omit a source/rod/root, divide by accepted rather than attempted
  draws, multiply the natural-proposal mosaic density back in, apply the detector Jacobian or solid
  angle, renormalize after off-panel rejection, swap row/column ownership, or drop regular nonzero
  `m=0`.
- Reuse colliding RNG streams across source indices, make the result depend on worker/chunk order,
  accept a noninteger seed, silently discard nonfinite weights, expose root hits as calibrated
  detector counts, or report ordinary Gaussian standard errors at an unexcluded Ewald fold.
- Make a later Philox draw or source-count request alter an existing source/draw prefix; consume a
  variable number of RNG lanes at Gaussian/Lorentzian mixture limits; or let CPU and CUDA transform
  different latent coordinates.
- Publish a superseded render revision, queue every slider event, restart an accepted prefix for its
  settled stage, reuse a sampler after a failed cancellation reset, or accept a geometry rebind
  after source rows, rods, evaluator topology, detector calibration, sample support, film, or crystal
  mount changed.
- Downsample, crop, transpose, or interpolate the presentation frame; flip detector rows twice or
  not at all; mutate a leased float32 frame before upload completes; or silently change CUDA/OpenGL
  to CPU/Matplotlib while retaining the requested backend identity.
- Hide an unresolved adaptive pixel or a caustic behind display interpolation.

Expected detection: detector ray/Q identity, forward/inverse round-trip, per-rod/source reduction,
constant-field pixel identity, caustic finite-box oracle, quadrature refinement, exact seeded
forward-oracle ownership, refined latent mass/shape comparison, mass conservation, seed-ledger
identity, draw/source-prefix identity, CPU/CUDA and worker parity, latest-only scheduler rejection,
causal-rebind validation, asymmetric full-native texture ownership, fail-closed backend identity, or
invalid-support contract.

### Continuous angle-coordinate measure

- Swap row/column or the canonical `phi` sign, shift detector coordinates by half a pixel, or use a
  detector pose different from the bound detector function.
- Omit the detector-area Jacobian, apply it twice, use a source-state solid angle instead of the
  fixed `AngleFrame` origin, or multiply it into `I` as an acceptance correction.
- Give the exact pole or an invalid/off-panel direction nonzero normalization.
- Average `S/N` point samples before a finite-bin reduction instead of integrating `S` and `N`
  separately.

Expected detection: tilted finite-difference Jacobian, periodic-seam/pole/support checks,
pose-bound instrument comparison, pointwise `S=dJ`, `N=J`, `I=S/N`, and a nonlinear divide-order
oracle.

### Configuration and acceleration

- Accept an unknown, duplicate, aliased, or missing YAML key.
- Change the YAML while reusing stale compiled state.
- Run a staged replay with a numerical package version that differs from the case-bound `uv.lock`.
- Permute packed source/rod indices, change dtype, or omit a factor in the compiled CPU/CUDA kernel.
- Claim CUDA while executing a CPU evaluator or silently fall back when CUDA is unavailable.

Expected detection: strict loader, pre-fit runtime-lock validation, revision/key ownership,
scalar-versus-compiled parity, backend identity, or explicit availability failure.

### Exact-marker geometry fitting

- Replace either callable detector field with a sampled target array or call a pixel integrator.
- Remove the analytic beta-root sign from marker identity or derive it from detector left/right.
- Treat the two beta-root sides as separate Ewald branches, or fabricate +/- branches for m=0.
- Change the m=0 exact-L landmark from the declared minimum-mosaic-tilt solution without changing
  its policy identity.
- Drop either the exact-tag coordinate terms or the paired-root/m=0 line-angle terms.
- Reassign a missing/tangent/branch-changing root inside the residual.
- Reverse either intrinsic correction sign/order, swap detector column/row, or compare caked rather
  than native coordinates.
- Admit an underdetermined raw sample/goniometer pack without a scaled-Jacobian rank gate.
- Fit only the training sites without held-out prediction or the independent outer root audit.

Expected detection: callable/no-pixel spies, duplicate marker identity, topology failure, line-angle
residual-decomposition and independent m=0 constrained-Ewald oracles, synthetic parameter
recovery, held-out native-coordinate error, Jacobian
rank/condition, or post-fit selection classification.

### Multi-OSC shared geometry fitting

- Join images by tuple position, infer the 15-degree angle from legacy 12-degree provenance, use
  degrees as radians, reverse motor sign, or drop one declared image.
- Correct the axis after commanded motion, derive pivot tangents from the nominal rather than the
  corrected axis, add an axis-parallel pivot coordinate, or apply the plane offset along the
  pre-correction normal.
- Use the sample-transform translation rather than the actual nominal incident/sample intersection
  as the selection `AngleFrame` origin, or retain nominal detector axes after a detector-pose
  correction.
- Add detector roll, crystal axial roll, sample tangent translation, detector-center translation,
  or detector distance without an independent calibration owner.
- Give each image its own correction vector, omit the 5 -> 7 -> 9 rank ladder, or accept a deficient
  one/two-image subset through regularization.
- Invoke structure strength, mosaic probability, rasterization, or pixel integration while
  predicting exact tags from the nominal source-center state.
- Replace frozen observations after fitting, or let a newly visible key poison an aggregate track
  and retrospectively delete otherwise unchanged frozen keys.
- Swap the two analytic beta roots while retaining their `(-1,+1)` root-sign tuple, reuse the
  production root solver as its own audit, or make a geometry-dependent fresh cake search the
  frozen-identity acceptance oracle.

Expected detection: strict manifest/image-ID joins, commanded-angle/context validation, independent
nine-coordinate synthetic recovery, arbitrary active-subset ordering with bit-exact fixed-coordinate
preservation, gauge exclusion, active-subset bound-scaled rank/condition gates,
intensity/mosaic/pixel spies, direct fixed-`L` root-coordinate oracle, and corrected-geometry
relabeling of the unchanged selected native candidates.

### Measured selection and indexing

- Supply predicted marker coordinates to global discovery, rotate OSC data twice, swap native
  row/column, or reuse a cake calibration different from the detector-to-`Q` calibration.
- Infer branch identity from detector left/right, collapse equal-family rods, accept tangencies or
  noncoincident symmetry aliases, or relabel an ambiguous assignment.
- Reuse an image, mask, calibration, reciprocal, or policy hash after mutation; accept one-incidence
  tracks, duplicated-image replication, or sites without distinct-incidence support.

Expected detection: position-free discovery boundary, once-only OSC/cake round trip, reciprocal
label and ownership-gate rejection, exact manifest hash, or distinct-incidence and shared-site
replication gates.

### Mosaic response-profile fitting

- Average pointwise `S/N` values instead of integrating finite-bin `S` and `N` separately, or fit a
  detector raster rather than the frozen continuous profiles.
- Drop a detector-visible explicit nonzero branch, fabricate signed duplicates for a collapsed
  `|00L|` profile, omit a frozen admitted raw-supported `m=0` profile, use an off-panel minimum-tilt
  landmark as evidence that its full exact-L circle is absent, or impose a mosaic-tilt cutoff on
  configured support.
- Share one nuisance amplitude across distinct profiles, or use peak heights, cross-profile ratios,
  or planted structure-factor amplitudes as fit weights.
- Reuse a component profile after changing its layout, frame, physical source, backend/device, rod
  catalogue, material, or fixed geometry provenance; substitute a layout hash for the source
  revision or mix source revisions inside one component bank.
- Search eta locally, omit either exact face, miss a second tied interior basin within a coarse
  interval, or accept a nuisance-projected rank-deficient solution.
- Square unnormalized extreme response amplitudes so that per-profile scales underflow or overflow.

Expected detection: integrate-before-divide oracle, explicit branch/cardinality validation,
frozen `m=0` support evidence, per-profile scale invariance, physical-source and profile-revision
rejection, boundary and within-cell global-alias fixtures, extreme-scale invariance, or local
rank/condition failure.

### PbI2 measured rational-landmark admission

- Feed predicted marker coordinates into discovery, use a restricted parent catalogue, or invoke
  the PbI2 catalogue for a different material.
- Admit one root side when its paired visible root is missing, let one peak own two nearby exact
  sites, average competing peaks, ignore covariance, or convert a missing half-/third-order site
  into a zero-valued observation.
- Reuse discovery after changing detector geometry, angle frame, detector shape, source CIF, or
  reciprocal basis.
- Strip the discovery/catalogue identity from an admitted coordinate pack or change the frozen
  catalogue-to-peak row join after admission.

Expected detection: catalogue-free discovery provenance, material/full-parent/revision rejection,
hard-distance and Mahalanobis gates, two-way assignment margins, complete-visible-root admission,
the immutable `Pbi2LayerLPeakAdmission` source join, and the `None` no-observation result.

### PbI2 exact-rational intrinsic SF populations

- Expand one exact parent overlap into one residual per parent, count a signed rod once per
  supporting parent, average the rods, or coherently add parent amplitudes.
- Retain both detector root sides as independent intrinsic structural rows, mask nominally absent
  parent columns, or fabricate a zero-valued row for a missing half-/third-order peak.
- Reorder strengths without their exact landmark identities, change the reciprocal/fixed-state
  sampling revision, change the crystal basis behind a pinned catalogue, drop a contributing rod,
  mix analytic branches across paired source tags, or compile from a restricted-parent catalogue.
- Give every peak an independent scale, permit excluded specimen components to enter, or accept a
  full three-phase fit whose integer-only phase-contrast rank is below two. Drop a nonzero scale
  nuisance merely because its column norm is small relative to another phase response.

Expected detection: exact catalogue-definition, reciprocal-basis, paired-source, and
sampling-revision rejection; structural root-side collapse; explicit unique-rod sum comparison;
nonzero leakage witnesses; scale-normalized nuisance projection; allowed-support NNLS/profile
bounds; held-out prediction; and the integer-only identifiability failure.

### Fixed-position ordered-intensity fitting

- Swap `Qr^2` and `Qz^2`, omit the one-half amplitude exponent, or apply the directional
  Debye-Waller factor twice.
- Recompute detector projection or the full structure factor inside an optimizer iteration instead
  of contracting the frozen six-column occupancy quadratic and cached `Qr/Qz` arrays.
- Prune a weak or baseline-extinct root, omit an admitted `m=0` profile, or weight residuals by
  planted peak intensity.
- Form a scale, normalization, comparison, or residual for each source row before reducing all rows
  into one detector function per incidence.
- Join observations by tuple order, reuse a stale observable-layout revision, or silently bind data
  generated on a different ROI/topology mask.
- Accept the wrong source count or realization revision, mix point-density with ROI-mass records, or
  bypass the interlaced `Uz` response certificate.
- Treat a refined numerical response digest as the observable identity, or accept a coarse response
  without planted-mass and occupancy-basis quadrature convergence.
- Fit all three occupancies together with free image scales, report an arbitrary absolute
  occupancy representative as identified, ignore a rank deficiency, or accept an active bound.

Expected detection: cached-versus-direct structure oracle, asymmetric directional-displacement
regression, zero-strength-versus-CIF marker-identity invariance, explicit all-profile and `m=0`
counts, dataset/observable revision rejection, independent response-order convergence,
explicit source-sum parity and source count/revision rejection, mixed-measure rejection,
common-scale-gauge rejection, and rank/condition/bound diagnostics.

### Portable staged-fit replay

- Change an input byte, decoded OSC value, native dtype/shape, or environment lock; use an absolute
  or repository-escaping case path; add an unknown case/stage key; redirect a nested consumed path
  to a different hash-complete role; mutate a loaded role mapping before execution.
- Drop or relabel one fitted profile or an admitted `m=0` identity; change the source count, seed,
  realization revision, fitted/fixed coordinate lists, active-bound mask, or historical/current
  selection provenance.
- Reorder stages, substitute a non-immediate upstream revision, edit compact state after writing,
  change backend during resume, modify or remove a required stage artifact reference, or return a
  malformed fresh-stage envelope. For Bi2Te3, substitute a self-hashed simulation config, a
  different valid lattice, or a self-consistent position record that did not come from geometry.
- Perturb a fit parameter within and beyond its declared tolerance; change decoded render pixels
  while preserving PNG metadata or container bytes; set a render CUDA chunk/block size nonpositive.

Expected detection: strict case loading, container/native-array hashes, exact identity comparison,
stage-aware tolerance verification, recomputed scientific revisions, envelope/backend/source
validation, and external-artifact content hashes.

### Mixed-chart matched-region fitting

- Assign `m=0` by `Qr/L`, assign a nonzero family by phi/2theta, or apply a downstream OSC flip or
  transpose before membership.
- Give each family or incidence its own structure vector, give each family a separate intensity
  scale, or replace the physical model with a smoothed measured profile.
- Evaluate the model only at pixel centers, rasterize it before fitting, change the measured
  piecewise-constant pixel projector between candidates, discard its cross-row covariance, alter a
  declared chart rectangle, omit its detector-area Jacobian, or apply a detector solid-angle
  correction.
- Reverse the R-centered registry sequence, alias it to 2H, or omit the centering extinction while
  retaining the conventional three-quintuple-layer cell.
- Change the declared fitted rod roster between stages or silently label a fitted-scope result as an
  all-rod/publication result.
- Under v7, skip or reorder A/B/C, substitute a stale predecessor, inject an explicit override, or
  change a child's exact start or frozen coordinates. Under v8, omit or mutate the explicit start,
  alter its hash, or fabricate an initializer predecessor. At the outer-workflow boundary, mutate
  seed-v2 without changing the plan revision. Under either policy, feed anything other than the
  joint result downstream.
- Permute parameter scales, let priors create apparent rank, accept a practically rank-deficient or
  ill-conditioned data Jacobian, or accept a structure coordinate on its active bound.
- Reuse model mass across an implementation/runtime change, alter a profile-checkpoint prefix or
  cubature identity, or report a structure representative different from the simulated vector.
- Promote a lattice candidate with deficient data-only rank/condition or insufficient improvement,
  change its bound position hash, or fail to propagate an accepted full basis through the
  fixed-experiment rebuild into structure physics while independently binding the mosaic state.
- Swap a fixed crystallographic site ADP with the sample-Q envelope, apply either twice, vary a site
  ADP in the five-coordinate joint fit, or omit the event-frame Q rotation before the envelope.
- Reinterpret vacancy as Bi substitution, fit both without an identifiable composition chart, or
  serialize an outer occupancy other than `1-v` or a nonzero antisite in the vacancy model.
- At declared dark scale zero, subtract dark counts, add dark covariance, or accept a dark
  scale/basis/covariance record that does not match the trusted recipe.
- Substitute independent incidence offsets for the shared delta plus zero-sum trims, reorder image
  IDs, or use an unbound mosaic record instead of the strict provided-mosaic checkpoint.

Expected detection: mixed-chart membership invariants, shared-scale synthetic recovery, continuous
chart quadrature and detector-area-Jacobian checks, expanded-CIF/explicit-registry parity including
wrong-hand injection, CPU/CUDA fault-free-parent parity, exact recursive lineage/cache/checkpoint
identities, parameter-scaled data-only sensitivity gates, strict fixed position/mosaic/lattice
handoffs, and separate site-ADP/sample-Q-envelope tests.
Vacancy representative/admission checks and the scale-zero dark invariant detect the added
composition and observation-contract mutations.

## Control record

Each proof mutation records `mutation_id`, fixture, expected first stage, expected metric, observed
first stage, and observed metric. A mutation fails the proof when it is not detected, is detected
only at an unexplained later stage, or triggers an unrelated earlier failure.

Only a minimal representative set belongs in permanent tests. Broad sweeps and one-control-per-test
collections are external proof work and are removed after review.

## Contract-v13 general-CIF controls

- Swap native calibration column/row, translate distance along a non-panel-normal axis, or allow a
  center/distance change without active calibration provenance.
- Exchange generic `repeats` with Bi2X3 `layers`, omit an unknown-U policy, or add a nonzero generic
  stacking epsilon.
- Use reference strength to prune sparse terms, collapse source wavelengths/rows, change signed
  rod identity, apply a candidate with another reciprocal basis, or accept complex/negative
  strength.
- Reuse stale response/parameterization revisions, omit parameter scales, cross dataset rows,
  retain a scale-gauged direction, or accept deficient/over-conditioned sensitivity.
- Treat hexagonal `family_m` as general-cell identity or merge noncoincident metric shells.

Expected first detection is strict config validation, calibration provenance/rank, response basis
and revision checks, strength validation, block ownership, parameterized-fit identifiability, or
the public nonhexagonal indexing regression respectively.

## Continuous-incidence and staged-scan controls

- Reverse traversal while preserving prediction support, reuse a 5--20-degree rule for 5--25
  degrees, change normalized masses, or vary the common calibration, source, or evaluator revision.
- Clip signed contrast, accept a terminal topology/fold/window event, or replace absolute
  covariance-whitened convergence with a relative criterion.
- Evaluate a scan candidate before its fixed-data gate, change either comparison layout, invoke a
  proposal after baseline failure, or exceed the exact-scan bound `1 + 2K`.

Expected detection is acquisition/quadrature revision rejection, immutable adaptive-result
validation, or the fixed-first acceptance policy before any affected result can be admitted.

## Native Bi cell/site controls (v15)

- Wrap the retained surface z lifts, exchange displacement tensor axes, allow an indefinite
  tensor, or substitute a sample-Q intensity envelope for atomic amplitude damping.
- Change occupancy while retaining old optics/transport, change a/c without reciprocal
  rebuilding, omit a potentially elastic rod, or let coherent height exceed the film.
- Duplicate background covariance, move native support, transpose historical guard arrays,
  silently cast complex counts, or profile scale beyond the guard-feasible interval.
- Freeze an optimizer coordinate accidentally or hide a discrete m0 stitch interval change
  in a smooth local sensitivity estimate.

Expected first detection: signed direct-amplitude/isotropic/termination proof, complete-candidate
dependency and native reuse tests, roster bound validation, immutable candidate/observation
validation, correlated GLS/scale oracle, planted all-coordinate recovery, or recorded sensitivity
interval changes. Measured guard failure is retained as rejection evidence, never waived.

## Joint native refinement controls (v16)

- Exchange Pb radial/normal ADPs, wrap iodine lifts, or replace amplitude damping by
  an intensity envelope: the independent finite atomic-path oracle diverges.
- Reuse a response after a source/rigid-pose change, bypass candidate shape/complex
  validation on a cache hit, or retain a cone component after its width/order changes:
  the dependency and fresh-response parity checks fail.
- Count calibration twice or change its coordinate metadata; constrain training with
  full-data historical guards; ignore cross-covariance or duplicate validation groups:
  ownership/leakage/partition checks and the analytic Schur-complement oracle fail.
- Report an inferior converged start as resolved despite a lower unfinished admissible
  point, or let an infeasible low-prior point invalidate a converged guarded solution:
  separate constrained/unconstrained counterexamples detect both errors.
- Hide changed guard feasibility behind unchanged individual free-scale decisions,
  divide control signal by robust-fit weights, or take a zero finite-difference step
  at a narrow search boundary: numerical/control/sensitivity regressions detect these.

Runner numerical records additionally bind the actual target revision and guard policy,
compare fitted center offsets across N, and reject effective no-op refinements. A failing
screen never qualifies a physical fit or automatically enables a new physical term.

## Numerical measures and acceptance (v17)

- Replace the uniform-Q median by a grid-point median: the nonuniform affine-measure
  oracle fails. Omit endpoint handling or zero-segment rejection: compiled endpoint and
  finite-stack-zero regressions detect the error.
- Drop an angular arc's probability fraction: disjoint/wrapped-arc mass and nonseparable
  integral oracles fail. Reuse insufficient frozen Q support: the enclosing-domain check
  raises. Source-count changes must preserve shared axial nodes under a valid frozen domain.
- Restore Q-conditioned arc activation under the fixed-union policy: the public
  observation-boundary continuity regression fails. Incorrect angular measure or weights
  also fail its independently known narrow-Gaussian integral.
- Give a profile center a numerical bias while keeping the initial center exact: its
  objective-contrast gate fails. Perturb only held-out predictions: the training screen
  can pass, but the conditional-validation numerical gate fails.
- Accept a failed initial screen, unresolved N, or an admissible profile minimum below
  the same N's joint fit: runner acceptance must remain null. An execution-only override
  cannot grant numerical agreement or physical identification.
