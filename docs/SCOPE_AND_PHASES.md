# Scope and phases

## Current objective

Maintain the smallest scientifically explicit core that produces detector-native X-ray scattering
from canonical source, sample, material, reciprocal, mosaic, structure, refraction, attenuation, and
detector geometry.

## Implemented core

- Strict YAML configuration and immutable compiled state.
- Source phase-space sampling with direction, position, wavelength, weight, and polarization
  provenance.
- Canonical source/sample intersection and entrance refraction.
- CIF parsing, reciprocal basis, complete physical rod catalogs, and finite ordered/stacked
  intensities.
- Wrapped Gaussian/Lorentzian mosaic probability in continuous latent coordinates.
- Analytic infinite-rod Ewald roots, detector-visible latent coating diagnostics, and an exact
  almost-everywhere full-sphere intrinsic density in internal-film `A2/sr`.
- Continuous detector-coordinate inverse mapping with exit refraction and uniform-depth attenuation.
- Continuous detector-to-`(phi, 2theta)` coordinate pullback with explicit signal and detector-area
  normalization densities before division.
- Prepared finite-bin angle profiles that integrate signal and detector-area normalization
  separately, plus response-bank recovery of Gaussian mosaic width, Lorentzian HWHM, and mixture
  probability with one exact nuisance amplitude per individual profile profiled out.
- Position-free measured-peak discovery in an angle chart, detector-native refinement, reciprocal
  integer-`L`/rod/root labeling, and immutable cross-incidence branch selection.
- Incoherent source/wavelength/phase summation before selected-center comparison, display-only
  center sampling, or deterministic detector box integration.
- Optional streaming Monte Carlo estimation of all-source, all-rod, all-root native-pixel mass by
  weighted forward sampling of the declared mosaic law and exact hard pixel ownership.
- Prefix-stable progressive CPU/CUDA execution through an explicit mutable compiled sampler,
  latest-only cancellation, causal geometry rebinding, and full-native OpenGL or Matplotlib
  presentation without spatial downsampling.
- Explicit detector distance, pitch, beam center, rigid pose, and two intrinsic detector tilts.
- Exact integer-L marker identities and bounded detector-native fitting of the identifiable local
  detector-tilt and effective sample-normal correction pack.
- Strict OSC-series manifests and geometry-only exact-tag contexts for repeatable fitting across
  varying layered-hexagonal materials in separate material/mount groups.
- Shared arbitrary-series decomposition of detector tilt, sample normal, goniometer-axis, signed
  sample-plane offset, and axis-perpendicular pivot offsets, with one common incidence delta,
  optional zero-sum per-image trims, and frozen-key outer audits.
- A separate tightly regularized near-CIF lattice-sensitivity stage with data-only identifiability
  gates and one full-basis handoff into every downstream lattice-dependent calculation.
- A strict fixed-experiment checkpoint that composes position, optional accepted lattice, and
  provided mosaic states while reusing immutable material/source/reciprocal physics across views.
- Material-neutral matched-region observations that combine angular `m=0` and signed-side
  reciprocal `m!=0` charts, continuous chart-region cubature, one scale per image across
  families, an independently frozen radial-background calibration, and staged hash-bound
  fit/profile artifacts. The Bi2X3 layered-quintuple adapter runs modular Wyckoff-z, physical
  outer-site Bi-antisite, and sample-Q intensity-envelope initializer fits followed by one mandatory
  joint five-coordinate refinement. Crystallographic site ADPs remain separate and fixed; only the
  joint result is authoritative.
- Explicit fault-free R-centered three-registry finite stacks in the proof, compiled CPU, and CUDA
  detector paths. This is the tracked R-3m Bi2X3 conventional-cell centering law, not a fitted
  4H/6H mixture or stacking-disorder population.
- A separate synthetic intrinsic-strength PbI2 boundary fits nonnegative populations of five fixed
  2H/4H/6H near-parent responses at exact rational structural landmarks. It has no PbI2
  detector/source/optics forward and does not refine continuous transition-law parameters.
- An optional unified local-lamella `(0,0)` detector field that evaluates the Parratt--kinematic
  strength at internal phase `L` for every actual sampled incident direction and wavelength.
  Interface roughness is physical Nevot--Croce roughness; no auxiliary count scale, model raster,
  smoothing, or detector-resolution convolution is implied.
- NumPy proof, compiled CPU, and CUDA detector evaluators.
- Compact analytic, direct-oracle, mutation, convergence, reference, and integration proofs.

## Current result boundary

The authoritative output is either:

- `raw_detector_coordinate_density_A2_per_px2.v1` at arbitrary floating coordinates; or
- its finite native-pixel integral `raw_detector_pixel_mass_A2.v1`.

`raw_detector_pixel_mass_monte_carlo_estimate_A2.v1` is an optional stochastic estimate of the same
finite native-pixel mass. It is neither a calibrated count image nor a replacement authority for
the continuous field or deterministic integral.

An optional downstream measurement view evaluates
`raw_detector_area_normalized_intensity_A2_per_px2.v1` continuously in `(phi, 2theta)`. It is a
coordinate reparameterization of the detector density, not a replacement raw result or an added
detector-efficiency/solid-angle correction.

The optional reciprocal-space, Ewald-coating, and detector PNGs are display artifacts evaluated from
these functions. They are not stored model state.

`intrinsic_ewald_direction_density_A2_per_sr.v1` is a declared detector-independent diagnostic
callable over internal-film outgoing directions. It is not a replacement authoritative detector
observable, and any sphere mesh or raster is only a display sample of that function.

## Deferred work

- Calibrated detector efficiency, PSF/resolution, masks, beamstop, saturation, and
  acquisition-matched/general backgrounds beyond the implemented frozen radial halo.
- Multiple scattering, extinction, and full distorted-wave off-specular fields.
- Multi-phase optical environments beyond the declared single-film model.
- General-crystal peak identities, multi-axis mechanics, mixed-specimen shared fits, and raw-image
  profile extraction beyond the accepted layered-hexagonal Bi2Se3/Bi2Te3 real-OSC slices.
- General continuous-`S/N` angular-bin products and reciprocal remapping beyond the prepared
  finite-profile fitting boundary.
- Physical/nonnegative mosaic-fit backgrounds and general masks, detector PSF, counting noise,
  covariance/uncertainty intervals, held-out subsets, and intrinsic real-OSC recovery beyond the
  accepted model-limited effective radial estimate.
- General-material structure-parameter bases and profile identity catalogs, per-site anisotropic
  `Uij`, raw-OSC ordered-component extraction/deblending, and ordered-intensity
  noise/background/uncertainty models.
- Structure-parameter bases beyond the tracked Bi2Se3/Bi2Te3 layered-quintuple adapter, with
  user-declared site/occupancy constraints mapped into the generic detector callable.
- Optional bounded approximations for fitting, admitted only with observable error bounds.

## Phase discipline

New work starts from current `main` in an isolated `codex/` worktree. A change owns a narrow set of
contracts, proves its first divergent stage, retains only tests for distinct long-term invariants,
places diagnostics outside the repository, and ends with one clean commit. Historical numbered task
files remain provenance; this document and the live contracts describe the current runtime.
