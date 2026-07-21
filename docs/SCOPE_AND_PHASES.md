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
- Analytic infinite-rod Ewald roots and detector-visible intrinsic coating diagnostics.
- Continuous detector-coordinate inverse mapping with exit refraction and uniform-depth attenuation.
- Incoherent source/wavelength/phase summation before deterministic detector box integration.
- Explicit detector distance, pitch, beam center, rigid pose, and two intrinsic detector tilts.
- Exact integer-L marker identities and bounded detector-native fitting of the identifiable local
  detector-tilt and effective sample-normal correction pack.
- NumPy proof, compiled CPU, and CUDA detector evaluators.
- Compact analytic, direct-oracle, mutation, convergence, reference, and integration proofs.

## Current result boundary

The authoritative output is either:

- `raw_detector_coordinate_density_A2_per_px2.v1` at arbitrary floating coordinates; or
- its finite native-pixel integral `raw_detector_pixel_mass_A2.v1`.

The optional reciprocal-space, Ewald-coating, and detector PNGs are display artifacts evaluated from
these functions. They are not stored model state.

## Deferred work

- Calibrated detector efficiency, PSF/resolution, masks, beamstop, saturation, and background.
- Multiple scattering, extinction, and full distorted-wave off-specular fields.
- Multi-phase optical environments beyond the declared single-film model.
- Experimental peak finding/association, multi-angle decomposition of sample and goniometer
  mechanics, and intensity-profile fitting beyond exact marker positions.
- Caking and reciprocal remapping after detector-native fitting is validated.
- Optional bounded approximations for fitting, admitted only with observable error bounds.

## Phase discipline

New work starts from current `main` in an isolated `codex/` worktree. A change owns a narrow set of
contracts, proves its first divergent stage, retains only tests for distinct long-term invariants,
places diagnostics outside the repository, and ends with one clean commit. Historical numbered task
files remain provenance; this document and the live contracts describe the current runtime.
