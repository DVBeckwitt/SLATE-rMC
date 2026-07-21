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

### Configuration and acceleration

- Accept an unknown, duplicate, aliased, or missing YAML key.
- Change the YAML while reusing stale compiled state.
- Permute packed source/rod indices, change dtype, or omit a factor in the compiled CPU/CUDA kernel.
- Claim CUDA while executing a CPU evaluator or silently fall back when CUDA is unavailable.

Expected detection: strict loader, revision/key ownership, scalar-versus-compiled parity, backend
identity, or explicit availability failure.

## Control record

Each proof mutation records `mutation_id`, fixture, expected first stage, expected metric, observed
first stage, and observed metric. A mutation fails the proof when it is not detected, is detected
only at an unexplained later stage, or triggers an unrelated earlier failure.

Only a minimal representative set belongs in permanent tests. Broad sweeps and one-control-per-test
collections are external proof work and are removed after review.
