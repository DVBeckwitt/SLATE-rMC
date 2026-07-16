# Error-injection and tolerance policy

Passing a comparison is not enough. The proof system must also demonstrate that it detects the mistakes most likely to survive a visually plausible final image.

## Tolerances are part of the specification

T02--T05 stage tolerances are declared before implementation inspection in `proof/stage_tolerances_v1.json`, verified by its strict loader and canonical hash. The immutable reference pack is not modified. Do not choose a tolerance from observed disagreement.

For a scalar or array quantity `x`, use a declared criterion of the form

\[
|x_{\mathrm{new}}-x_{\mathrm{ref}}|
\le a_{\mathrm{stage}}+r_{\mathrm{stage}}S,
\]

where `S` comes only from declared physical inputs, analytic bounds, or immutable reference values. It never depends on the candidate magnitude and cannot loosen as an error grows.

Rules:

- Exact discrete identities, shapes, status codes, rod IDs, family IDs, branches, event IDs, and index maps require exact equality.
- Unit-bearing absolute tolerances are recorded in the unit of the compared value.
- Dimensionless conservation residuals use explicit dimensionless tolerances.
- Continuous detector coordinates are checked in pixels before any integer rounding or deposition.
- Complex amplitudes are compared as complex values. When a global gauge is physically irrelevant, align the one declared gauge once and then compare both amplitude and phase-sensitive derived quantities.
- Intensities are compared before display normalization, rounding, clipping, or log transformation.
- Near-zero quantities use the artifact's derived positive absolute tolerance. Relative error alone and zero absolute tolerance are invalid near zero.
- Convergence is assessed only for a real approximation variable, at both its stage and the declared final observable.
- Reference-versus-optimized tolerances may differ from analytic tolerances, but both are declared and justified.
- A `CORRECTED` legacy case uses the legacy tolerance only through the declared first-divergence stage. Downstream stages use the independent-oracle tolerance.

The comparator reports maximum error, a robust percentile error, the failing element or event ID, and the first failing stage. It never reports only pass or fail.

## Required negative controls

Each proof command includes bounded error injections. A proof passes only when the intended mutation is detected at the expected stage and the unmutated result still passes. These mutations are proof-only and never remain as production switches.

### Bootstrap and coordinate controls

- apply the OSC rotation in the wrong direction
- transpose instead of rotating
- swap row and column
- shift the pixel-center convention by half a pixel
- compose two rigid transforms in the wrong order
- use the wrong rotation pivot
- apply translation to a vector
- alter one transformed coordinate after the rigid transform

Expected result: the comparator fails at the first affected OSC or geometry stage, not only at the final image.

### Geometry and optics controls

- choose the opposite complex-square-root branch
- reverse the propagation-sign rule
- use the original s/p power-transmittance average instead of the scalar field amplitude
- omit entrance or exit transmission
- apply full film thickness independently to both paths instead of the declared depth average
- omit attenuation
- use a complex wavevector directly in the real geometric Ewald construction
- use a detector normal or basis with the wrong handedness

Expected result: interface, attenuation, dispersion, or detector-coordinate proofs fail before deposition.

### Mosaic and Ewald controls

- remove the declared spherical measure
- normalize the Gaussian and tail components independently but not the mixture
- resample nodes between identical evaluations
- reverse the arc orientation without updating its signed measure
- omit the event Jacobian
- multiply an additional empirical Lorentz factor
- accept an Ewald root whose residual exceeds tolerance

Expected result: normalization, event-mass, repeatability, or convergence proofs fail.

### Ordered and reflectivity controls

- normalize the strongest reflection to 100
- round reflection intensity
- prune a weak but nonzero rod
- collapse equal-`Qr` rods before projection
- replace a systematic absence with a fabricated fractional reflection
- omit occupancy, anomalous scattering, or displacement factors in a fixture that exercises them
- use the wrong reciprocal-cell convention
- use a different complex-`kz` branch in Parratt than in shared optics
- blend Parratt and kinematic outputs outside the declared handoff rule

Expected result: raw-amplitude, rod-identity, symmetry, or reflectivity-limit proofs fail.

### Stacking controls

- transpose or reverse the transition convention
- use matrix powers with the wrong layer-count offset
- normalize per layer when total finite intensity is declared, or the reverse
- combine incoherent parent populations as amplitudes
- omit registry phase
- substitute the stationary limit for a finite-stack case
- perturb one reduced-sector coefficient while leaving the full six-state result unchanged

Expected result: direct-enumeration, full-versus-reduced, normalization, or deterministic-parent proofs fail.

### Integration controls

For the tiny end-to-end case, inject each of the following separately:

- omit one factor from the event-to-pixel ledger
- apply one factor twice
- apply detector solid angle twice
- treat deposition weights as a physical point-spread function
- break event-ID alignment between geometry and intensity
- use the wrong rod or family identity
- rotate the detector image after integration
- deposit with row and column reversed
- discard out-of-frame mass without recording it

Expected result: the first failing factor or identity stage is reported and detector-mass conservation fails when appropriate.

### Selection and fitting controls

These controls are required when T08 through T14 are implemented:

- switch rod or branch association during one objective evaluation
- key a family only by floating-point `Qr`
- define branch from raw detector pixel sign
- allow intensity parameters to repair a geometry displacement
- allow mosaic width to repair source divergence without an identifiability warning
- regenerate stochastic samples between evaluations
- apply a data correction on both the data and model sides
- reuse a compiled state after changing a parameter that invalidates it
- fit independent peak amplitudes before the ordered or stacking model
- accept a fit that improves the training objective while worsening a held-out physical observable

Expected result: the objective or fit validator rejects the run with a specific reason. A generic optimizer failure is not sufficient.

## Sensitivity requirement

A mutation must alter a fixture that is actually sensitive to it. For example, an occupancy mutation is not tested on a unit-occupancy single-species crystal, and an OSC orientation mutation is not tested on a square symmetric image. Each negative control records:

```text
mutation_id
fixture_id
expected_first_stage
expected_failure_metric
observed_first_stage
observed_failure_metric
```

The proof fails when a mutation is not detected, is detected only at an unexplained later stage, or causes an unrelated earlier failure.

## Test-budget rule

Negative controls belong in proof commands and compact proof metadata. Keep only a minimal representative subset in the permanent test suite. Do not create one permanent test function or snapshot per mutation.
