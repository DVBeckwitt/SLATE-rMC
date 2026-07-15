# Scope and phases

This file is the authoritative scope statement.

## Final project goal

The completed system will support a staged detector-native scientific workflow:

1. characterize source size, divergence, wavelength, and their declared correlations
2. calibrate detector pose and beam center from direct-beam and calibrant observations where available
3. index measured sample peaks against stable rod-family, reflection-group, and branch identities
4. fit sample and goniometer geometry with those associations frozen and then revalidate them
5. fit the mosaic orientation distribution with source and geometry fixed
6. fit ordered structural intensities from selected detector regions and `Qr` rod families
7. fit stacking-disorder intensities from selected branches and fixed-`Qr` rod observations

The forward model is built and proved first so every later fit changes only the parameters intended for that stage.

## Phase 0: bootstrap

Included:

- greenfield `src/` package layout
- shared units, frames, coordinate types, event IDs, and validity codes
- shared rigid-transform, complex-normal-wavevector branch selection, and scalar-interface primitives
- exact OSC raw-to-detector-native mapping
- proof CLI and stage-trace schema
- diagnostic writer that rejects repository-local output
- synthetic no-physics pipeline crossing every branch contract
- locked dependency and import policy

## Phase 1: immutable tracked reference verification

Included:

- one pinned original-RASIM environment record
- deterministic explicit inputs
- subsystem and tiny end-to-end intermediate traces
- one tracked read-only `.npz` pack with one JSON manifest
- source and pack hashes committed in `reference/reference_manifest.toml`

No physics branch creates or edits legacy evidence.

## Phase 2: four parallel forward worktrees

Included:

### Geometry and optics

- OSC decoding and high-range pixels
- instrument configuration and rigid poses
- sample and footprint intersection
- detector intersection and continuous detector coordinates
- entrance and exit refraction
- scalar field-amplitude Fresnel factors
- absorption, evanescence, and manuscript uniform-depth attenuation

### Mosaic and Ewald

- source and wavelength samples under declared correlations
- Gaussian-plus-Lorentzian mosaic distribution
- correct spherical probability measure
- reciprocal-event support, Ewald residuals, deterministic quadrature, and event probability mass

### Ordered rods and reflectivity

- CIF parsing, symmetry, occupancy, displacement, and atomic factors
- wavelength-dependent material optics
- complete individual `(h,k)` rod catalog with exact radial-family metadata
- arbitrary-`Qz` complex amplitudes and finite ordered stacks
- pure Parratt, pure kinematic, and named manuscript/legacy composite outputs

### Stacking transition

- `F+` and `F-` consuming contracts
- full six-state and exact reduced transition calculations
- direct finite-sequence and pair-sum oracles
- finite-stack intensity, parent-rich models, and explicit normalization

Excluded from the four worktrees:

- all optimization and fitting
- final rod/branch association
- detector deposition and final image assembly
- caking, `2theta/phi`, and reciprocal-space remapping
- GUI code
- production acceleration frameworks

## Phase 3: review and native-detector integration

Included:

- automated contract and trace review
- read-only scientific review
- one-at-a-time vertical integration
- exact factor ownership and detector solid angle
- non-square detector deposition
- ordered and stacking substitution through one event-aligned interface
- one tiny end-to-end native-detector proof
- integrated profiling and production-path recommendation

## Phase 4: selection and fit foundation

Included only after Phase 3 passes:

- immutable rod, radial-family, reflection-group, and branch identities
- exact hexagonal `m` identity and general-cell reciprocal-metric identity
- physical signed-azimuth branch rule with explicit deadband
- measured peak and ROI association with ambiguity rejection
- versioned selection manifests
- fit parameter, dataset, objective, result, provenance, and invalidation contracts
- deterministic sampling during optimization
- explicit data likelihood, mask, variance, scale, background, and data/model correction semantics

Selection is versioned outside the optimizer. Geometry may trigger a new selection manifest only between optimization runs.

## Phase 5: staged fitting

Included in this order:

1. source and wavelength characterization
2. detector geometry calibration
3. sample and goniometer geometry fit with frozen associations
4. association revalidation
5. mosaic fit with source and geometry fixed
6. ordered intensity fit with geometry and mosaic fixed
7. stacking-disorder fit with the ordered baseline fixed

Initial objectives operate in detector-native coordinates. Compiled forward stages are reused according to an explicit invalidation graph.

## Later phases

Deferred until native-detector fits pass:

- `2theta/phi` caking
- reciprocal-space remapping
- selected `Qz` profiles derived from signal and normalization fields
- optional final joint polish after staged stability
- full multilayer off-specular distorted fields
- graded rough-interface fields
- calibrated absolute flux, detector quantum efficiency, gain, dead time, saturation, global background, beamstop, and bad-pixel response
- multiple scattering and extinction
- vector polarization-resolved distorted-wave calculations
- GUI work
- saved-session migration
- multi-GPU execution

A later caking implementation must consume the same frozen rod and branch identities and must not redefine them.
