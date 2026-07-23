# Error-injection and tolerance policy

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
- Recompute or slice a parent revision instead of preserving the complete realization.
- Let detector-only state invalidate incident `ki`.
- Erase accepted geometry when a later optical stage fails.

Expected detection: exact strata, revision ownership, causal invalidation, or status payload.

### Optics

- Choose the opposite complex-normal branch or propagation sign.
- Substitute the old power-transmittance average for the scalar field amplitude.
- Omit or duplicate entrance/exit transmission or attenuation.
- Use full thickness independently for entrance and exit rather than the uniform-depth average.
- Use a complex wavevector directly in real elastic geometry.

Expected detection: mode dispersion, tangential conservation, amplitude, attenuation, or exit
geometry before detector integration.

### Mosaic, rods, and Ewald restriction

- Omit the folded-alpha factor, duplicate a signed beta branch, or independently renormalize mixture
  components.
- Collapse equal-family rods before evaluating their physical structure strengths.
- Use the wrong `u`/`L` scale, omit population, or sum independent rod amplitudes.
- Choose the wrong analytic root, emit a tangent as regular, omit the Ewald coarea factor in the
  intrinsic route, or apply it twice.
- Accept an elastic residual outside tolerance.

Expected detection: probability normalization/moments, per-rod sum, quadratic-root oracle, elastic
closure, or latent/coating agreement.

### Ordered, stacking, and reflectivity

- Normalize a reflection to 100, round it, prune a weak rod, or fabricate a fractional reflection.
- Omit occupancy, anomalous scattering, displacement, layer phase, or registry phase.
- Reverse the transition convention, use the wrong finite-layer exponent, or mix parent amplitudes.
- Reuse a different complex-normal branch in Parratt or blend outside the named handoff.

Expected detection: raw amplitude, systematic absence, finite-stack/direct enumeration,
normalization, or reflectivity limit.

### Continuous detector and pixels

- Use `Q=ki-kf`, transform Q twice, or use air `kf` where film `kf` is required.
- Drop an inverse latent branch, rod, retained root, source state, or wavelength.
- Apply source, phase, polarization, optical, surface-Jacobian, or Ewald factor zero or two times.
- Multiply detector solid-angle metadata into the raw field again.
- Swap detector row/column, reverse a tilt, evaluate outside the active panel, or assign intensity to
  an invalid/back-facing ray.
- Replace the pixel box integral with a center sample, histogram, blur, or point deposit.
- Hide an unresolved adaptive pixel or a caustic behind display interpolation.

Expected detection: detector ray/Q identity, forward/inverse round-trip, per-rod/source reduction,
constant-field pixel identity, caustic finite-box oracle, quadrature refinement, mass/centroid error,
or invalid-support contract.

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
- Permute packed source/rod indices, change dtype, or omit a factor in the compiled CPU/CUDA kernel.
- Claim CUDA while executing a CPU evaluator or silently fall back when CUDA is unavailable.

Expected detection: strict loader, revision/key ownership, scalar-versus-compiled parity, backend
identity, or explicit availability failure.

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

## Control record

Each proof mutation records `mutation_id`, fixture, expected first stage, expected metric, observed
first stage, and observed metric. A mutation fails the proof when it is not detected, is detected
only at an unexplained later stage, or triggers an unrelated earlier failure.

Only a minimal representative set belongs in permanent tests. Broad sweeps and one-control-per-test
collections are external proof work and are removed after review.
