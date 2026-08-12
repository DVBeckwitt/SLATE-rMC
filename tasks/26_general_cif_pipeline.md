# General-CIF staged fitting pipeline

Status: READY_SHARED_NUMERICAL_PIPELINE_MEASURED_PBI2_MODEL_LIMITED
Branch: `codex/general-cif-pipeline`

## Objective

Make geometry, mosaic, and ordered-intensity fitting consume one material-neutral numerical
pipeline. Bi2Se3, Bi2Te3, and PbI2 must differ through CIF data and explicit model declarations,
not through material-name dispatch, imported one-off runners, or copied physics.

The first accepted scope is a layered-film CIF whose first two direct-lattice vectors define the
surface lattice and whose third reciprocal vector defines the continuous rod direction. A CIF does
not supply detector calibration, mounting, finite repeat count, mosaic law, background, source
resolution, fitted site basis, or stacking law; those remain explicit typed inputs.

## Acceptance contract

- Geometry retains its existing exact CIF-derived predictor and optimizer. Optional detector
  reference-column, reference-row, and plane-normal distance corrections are explicit calibration
  coordinates and are inactive by default.
- A generic CIF finite-repeat strength implements the same reciprocal-basis-bound strength
  protocol as `Bi2X3FiniteStackStrength`. The optimized Bi2X3 model remains a compatible
  implementation, not a separate public pipeline.
- The material-neutral mosaic fitter rebinds the detector's mosaic state and evaluates the same
  source/optics/rod detector model into candidate profile banks.
- For structure refinement, one sparse detector-transfer response freezes source, optics, mosaic,
  detector Jacobian, physical rods, inverse roots, and exact wavelength. Applying a candidate
  strength model is the only structure-dependent operation before exact region integration and
  matched-region fitting. Ordered site parameters are declared as a small affine basis over the
  symmetry-expanded CIF; parameter names or element names never select equations.
- Bi2Se3 and Bi2Te3 reproduce the accepted numerical observables within their frozen tolerances.
  PbI2 runs through the same APIs and preserves the existing model-limited or rejected scientific
  classifications when the data do not identify an intrinsic result.
- No production path imports a script, notebook, workstation-local module, or dynamically
  registered material plugin. A new material needs a CIF plus explicit experiment/model data. A
  genuinely different disorder law may add one reusable strength implementation behind the same
  protocol.

## Required proof

1. Generic CIF amplitude times the finite periodic repeat agrees with a direct atom/repeat sum for
   a non-Bi2X3 CIF at integer and noninteger `L` and multiple wavelengths.
2. Sparse source averaging agrees with direct one-state reduction and the existing Bi2X3 compiled
   result on a compact multi-source fixture.
3. Synthetic geometry recovers optional detector-center/distance corrections; an absent correction
   pack is exactly the old path and gauge-deficient requests fail before optimization.
4. The unchanged mosaic fitter recovers a planted generic-CIF response without constructing a
   Bi2X3 model.
5. The matched-region fitter recovers a declared generic CIF site basis with independent dataset
   scales and rejects a gauge- or rank-deficient basis.
6. Focused Bi2Se3, Bi2Te3, and PbI2 integrations use the same public types and preserve the declared
   physical rod identities and measure revisions.

The accepted workflow boundary is typed in-process composition of configured inputs, the shared
sparse detector, mosaic profiles, exact region blocks, provider parameterizations, and the
rank-gated matched-region fitter. Raw OSC observation preparation remains an explicit experiment
recipe rather than a universal CIF-derived operation. No new material-specific runner is part of
this task.

## Stop conditions

Fail closed rather than inferring missing film physics from a CIF. In particular, arbitrary 3-D
single-crystal indexing, powder averaging, specular film/substrate stitching, detector PSF,
background calibration, or a new stacking transition law are separate declared contracts.

## Handoff

- Public seams: optional `DetectorCalibrationCorrections`; `CifFiniteStackStrength` plus affine
  CIF/Bi2X3/PbI2 parameterizations; `SourceAveragedStructureDetector` and its compiled sparse
  response; `ParameterizedStructureRegionModel` with rank-gated matched-region fitting.
- Legacy classification: inactive geometry calibration is bit-exact `MATCH`; sparse Bi2X3
  detector transfer is `MATCH`; generic CIF strength/calibration/Pb fixed-parent composition are
  `NEW` and accepted by independent synthetic oracles. There is no divergent replacement of the
  optimized Bi2X3 renderer.
- Final proof: `407` tests passed in `544.618 s`; changed Python files pass Ruff format/check and
  `git diff --check`. The 250-source/85-rod/five-coordinate benchmark compiled `6,067` terms in
  `15.7920 s`, applied a strength in median `0.243617 s`, and used `11,669,445` traced peak bytes.
- Permanent tests retained: direct non-Bi2X3 atom/repeat enumeration, mixed-wavelength sparse
  parity, calibration recovery/null/gauge/provenance, metric-shell integer indexing, PbI2
  nonzero+001 mosaic recovery, generic affine/gauge/stale fits, Bi2Se3/Bi2Te3 shared-fitter
  recovery, and fixed-parent PbI2 response/log-ratio recovery. Each protects a distinct equation,
  identity, or public join. The PbI2 proof also verifies that an affine iodine-height change moves
  regular 001 intensity while fixed parent-population log ratios remain invariant there.
- Limits: this is the shared typed numerical fitting core, not a universal raw-OSC recipe or
  generic full-image renderer. Rational layer-order admission remains explicitly PbI2/hexagonal;
  generic integer-L identities use the reciprocal metric. Measured mixed/disordered PbI2 remains
  model-limited because its mask/background/source resolution and a free stacking law are not CIF
  data. Regular kinematic 001 is supported, but the current Pb parent-population derivative there
  is exactly zero.
