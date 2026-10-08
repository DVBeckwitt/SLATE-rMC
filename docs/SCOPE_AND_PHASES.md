# Scope and phases

T34 consolidates native preparation, fitting, rendering and recoverable execution for all
six declared acquisitions. All admitted coordinates remain free in the final joint stage.
This delivers reusable infrastructure; it does not qualify the existing provisional fits
or establish their parameter uncertainties. Additional materials supply a scientifically
defined specimen binding, not changes to the shared detector or optimizer.

## Current objective

Maintain the smallest scientifically explicit core that produces detector-native X-ray scattering
from canonical source, sample, material, reciprocal, mosaic, structure, refraction, attenuation, and
detector geometry.

## Implemented core

- Strict YAML configuration and immutable compiled state.
- Weighted Gaussian source phase-space sampling with optional per-axis position--divergence
  correlation, either a legacy Gaussian wavelength law or exactly weighted discrete Gaussian
  spectral lines, and complete direction, position, wavelength, weight, and polarization
  provenance.
- Canonical source/sample intersection and entrance refraction.
- CIF parsing, reciprocal basis, complete physical rod catalogs, and finite ordered/stacked
  intensities.
- Wrapped Gaussian/Lorentzian mosaic probability in continuous latent coordinates.
- Analytic infinite-rod Ewald roots, detector-visible latent coating diagnostics, and an exact
  almost-everywhere full-sphere intrinsic density in internal-film `A2/sr`.
- Continuous detector-coordinate inverse mapping with exit refraction, uniform-depth film
  attenuation, one flat-film illuminated-volume weight `1/|direction_sample,z|`, and optional
  scalar or exact-wavelength Beer--Lambert attenuation along the external detector ray.
- Continuous detector-to-`(phi, 2theta)` coordinate pullback with explicit signal and detector-area
  normalization densities before division.
- Prepared finite-bin angle profiles that integrate signal and detector-area normalization
  separately. G/L fitting uses the shared native-count coordinator; the standalone profile-scale
  optimizer and component response bank are retired.
- Position-free measured-peak discovery in an angle chart, detector-native refinement, reciprocal
  integer-`L`/rod/root labeling, and immutable cross-incidence branch selection.
- Incoherent source/wavelength/phase summation before selected-center comparison, display-only
  center sampling, or deterministic detector box integration.
- Normalized continuous-incidence acquisition and calibrated detector averaging, with a separate
  evaluator-driven adaptive finite-region oracle and bounded fixed-first delayed acceptance.
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
- Material-neutral matched-region observations with continuous chart-region cubature, shared
  specimen parameters, separate exposure scales, and explicit background conditioning. The
  shared native search preserves full declared covariance and data-only rank gates. Optional
  peak aggregation is declared by the caller. Historical v7/v8 five-coordinate orchestration
  is retired; current stages and material limits are documented in `STAGED_FITTING.md`.
- Exact RichEpsilon finite stacks for the R-centered three-registry parent in the NumPy, compiled
  CPU, and CUDA detector paths, including a bit-preserving epsilon-zero fast path. The tracked
  Bi2X3 production model retains `epsilon=0`; this is not a fitted 4H/6H population.
- Fixed five-parent PbI2 strength mixtures through the shared selected-coordinate detector
  transfer, and native Pb transition-law fitting through the same staged search coordinator.
  The separate intrinsic-landmark population optimizer is retired. Declared ordered controls
  and independent parent intensity mixtures retain the canonical recurrence.
- Optional Parratt--kinematic `(0,0)` stitching with an explicit interface convention: a unified
  local-lamella field following the mosaic, or fixed-external-Qz strength on the regular inverse
  map. Both retain wavelength-resolved strength in `A2`; the conventions are not interchangeable.
  Interface roughness is physical Nevot--Croce roughness; neither implies an auxiliary count scale,
  model raster, smoothing, or detector-resolution convolution.
- NumPy, compiled CPU, and CUDA detector evaluators.
- Explicit runtime numerical qualification and input/result validation.

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

- Calibrated detector efficiency, PSF/resolution, masks, beamstop, saturation, and matched-session
  dark/blank backgrounds beyond the implemented frozen radial halo and explicit zero-dark path.
- Multiple scattering, extinction, and full distorted-wave off-specular fields.
- Multi-phase optical environments beyond the declared single-film model.
- Arbitrary 3-D single-crystal or powder peak identities, multi-axis mechanics, mixed-specimen
  shared fits, and automatic raw-image profile extraction beyond the declared layered-film rod
  geometry.
- General continuous-`S/N` angular-bin products and reciprocal remapping beyond the prepared
  finite-profile fitting boundary.
- Physical/nonnegative mosaic-fit backgrounds and general masks, detector PSF, counting noise,
  covariance/uncertainty intervals, held-out subsets, and intrinsic real-OSC recovery beyond the
  accepted model-limited effective radial estimate.
- Species substitution or topology-changing structure bases, per-site anisotropic `Uij`, automatic
  raw-OSC ordered-component extraction/deblending, and ordered-intensity
  noise/background/uncertainty recipes.
- Automatic chemistry constraints, variable cells, and arbitrary stacking-transition laws beyond
  the declared affine CIF, Bi2X3 quintuple, and fixed-parent PbI2 parameterizations.
- Optional bounded approximations for fitting, admitted only with observable error bounds.

## Contract-v13 implemented general-CIF fitting scope

- Complete conventional-CIF unit-cell amplitudes with explicit coherent repeat count, wavelength,
  normalization, and unknown-isotropic-displacement policy.
- One sparse source-averaged detector transfer shared by generic CIF, specialized Bi2X3, and fixed
  five-parent PbI2 strengths at selected continuous coordinates.
- Declarative affine expanded-CIF fractional-coordinate, occupancy, and isotropic-`U` bases, plus
  gauge-free PbI2 parent log ratios, through the shared rank-gated native search.
- Optional native detector reference-center and panel-normal distance calibration with explicit
  fixed-position provenance and an exact inactive legacy path.
- Reciprocal-metric integer-L rod shells for nonhexagonal layered cells; `family_m` remains
  hexagonal display metadata. PbI2 rational layer-order admission remains a specialized optional
  helper rather than a general-cell identity.

A CIF alone still cannot choose the surface mounting, repeat count, termination, mosaic law,
detector mask/PSF/background, fitted chemical constraints, or stacking transition law. Automatic
chemistry inference, species substitution through the affine basis, anisotropic per-site `Uij`, a
generic raw-OSC recipe generator, a generic full-image renderer, arbitrary 3-D single-crystal or
powder indexing, and arbitrary stacking-law inference remain deferred. The shared programmatic
core uses declarative experiment data and the shared prepare/refine/render commands. Historical
per-material workflow scripts are retired; their immutable scientific evidence remains available.

## Phase discipline

Use the current `AGENTS.md`, `WORKTREE_LAUNCH.md` and `docs/VALIDATION.md` for narrow,
reviewable changes. Keep one writer for each checkout and retain only `main` plus the
authorized `codex/desktop-ui-implementation` branch. Historical numbered tasks
are provenance; their retired test/proof campaigns are not current work instructions.
