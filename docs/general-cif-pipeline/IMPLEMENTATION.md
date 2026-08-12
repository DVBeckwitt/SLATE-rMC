# General-CIF pipeline implementation

## Slice 1: generic structure strength

- Factor the stable finite periodic repeat from the existing ordered finite-stack equation.
- Add one reciprocal-basis-bound CIF finite-repeat strength with explicit repeat count,
  normalization, and unknown-`Uiso` policy.
- Prove it against direct atom and repeat enumeration.

## Slice 2: shared sparse detector transfer

- Let the low-level detector measure select an explicit incident-state row while preserving the
  singleton default.
- Compile every valid source row into one immutable sparse structure response.
- Apply vectorized strength profiles grouped by physical rod and wavelength, then reduce source
  states incoherently into the existing detector-coordinate measure.
- Prove direct one-state and current Bi2X3 parity before changing any configured caller.

## Slice 3: generic configured simulation and mosaic

- Allow the strict simulation config to select the generic CIF finite-repeat model and declare an
  unknown-isotropic-displacement value only when required.
- Build the same configured input and detector-transfer types for generic and Bi2X3 strengths.
- Feed generic response banks to the existing mosaic fitter; retain the Bi2X3 optimized renderer.

## Slice 4: ordered CIF site basis

- Add one immutable affine site-parameter basis and crystal rebinding function.
- Use the shared detector transfer plus `fit_matched_regions`; retain one scale per dataset and the
  existing covariance/background operator.
- Reject parameter gauges, rank deficiency, stale basis/response revisions, and topology changes.

## Slice 5: geometry calibration

- Add the optional center/distance correction pack to the shared indexed-series fit and downstream
  fixed-position identity.
- Preserve the exact inactive path and bind active calibration into downstream fixed-position
  identity.

## Slice 6: common fitted-structure boundary

- Bind affine CIF, explicit Bi2X3, and PbI2 parent log-ratio parameterizations to the same sparse
  detector-region model and rank-gated matched-region fitter.
- Prove Bi2Se3, Bi2Te3, ordered generic PbI2, and fixed-parent PbI2 through those public types.
- Keep raw-image observation recipes explicit and add no new per-material Python runner. Accepted
  historical Bi2X3 workflow scripts remain compatibility consumers until those recipes are
  declarative.

Each slice follows red/green/refactor and runs its focused proof. The branch lands as one coherent
commit.
