# Immutable original-RASIM reference

## Purpose

Original RASIM is used to locate equations, understand intended parameter semantics, reproduce valid behavior, and identify the first divergence where the new model corrects an old error. It is not the numerical authority when it conflicts with analytic physics or the manuscript.

## One serial characterization task

Before the four physics worktrees start, the seed contains one immutable tracked file:

```text
rasim_legacy_v1.ra_ref.npz
```

The file contains numeric arrays and one JSON manifest. It is then made read-only. The repository stores only its SHA-256 and environment metadata.

Physics branches may not regenerate, edit, or append to the pack.

## Pinned environment metadata

Record:

```text
original source archive SHA-256 or Git commit
manuscript archive SHA-256 or Git commit
Python version
operating system
NumPy, SciPy, Numba, Dans_Diffraction, XrayDB, and relevant parser versions
BLAS implementation
thread counts
precision
explicit input arrays
randomness policy
```

Use explicit deterministic source and mosaic arrays rather than relying on old random seeds.

## Required cases

### Coordinates and OSC

- non-square OSC marker payload
- high-range pixel decoding
- raw-to-detector-native mapping
- beam-center mapping

### Geometry and optics

- identity instrument
- tilted detector
- sample offsets and goniometer rotations
- individual sample intersection
- entrance and exit refraction
- critical and evanescent cases
- old Fresnel-power average and selected scalar coefficient values
- old full-thickness attenuation and manuscript uniform-depth result inputs

### Mosaic and reciprocal events

- representative non-specular rod circle
- specular-family case
- tangent and no-root cases
- old density and solve-Q intermediates
- event candidates before resampling and detector deposition

### Ordered and reflectivity

- raw atomic and structure amplitudes before max normalization and rounding
- individual `(h,k)` rods and family metadata
- arbitrary-`Qz` points
- finite-stack output
- Parratt recursion states
- pure kinematic and smooth composite outputs

### Stacking

- registry phases
- six-state and reduced matrices
- direct finite intensity cases
- deterministic parent limits
- rich-epsilon and reduced parameter examples


### Selection and future fitting

Capture these now so later fitting work uses the same immutable evidence:

- hexagonal `m`, `(m,L)`, reported `Qr`, and exact old Q-group keys
- signed-phi branch labels, deadband, and applicable `00L` collapse behavior
- one frozen geometry problem with measured targets, active parameters, old predictions, residuals, and stable numerical metadata
- one mosaic-profile case with ROI support, profile bins, background subtraction, normalized measured shape, and objective values for explicit parameter vectors
- one ordered-intensity case with ROI support, measured mass or pixels, primary/fixed model levels, analytic scale, residual, and objective
- one selected-rod profile case with mask, signal sums, normalization sums, retained bins, and sum-before-division output

Capture detector-native intermediates before caking or display conversion whenever the old code exposes them. When only a caked legacy value exists, record the full coordinate route and transformation metadata.

### Tiny end-to-end

Use a small non-square detector, explicit rays, one or two rods, one wavelength, nonzero refraction, nonzero attenuation, and a tilted detector. Record continuous hits, pixel indices, deposition weights, total event mass, total detector mass, peak pixel, and selected pixel values before display rotation or max normalization.

## Comparison classes

```text
MATCH
    stage-by-stage agreement within tolerance

CORRECTED
    agreement through the first declared divergent stage, followed by independent proof

NO_ORACLE
    analytic or independently converged proof only
```

Known likely `CORRECTED` points include the non-rigid sample-point overwrite, s/p power-transmission average, full-thickness attenuation use, mosaic-measure handling, structure-factor max normalization and rounding, artificial reflection insertion, and global post-intensity Q damping.
