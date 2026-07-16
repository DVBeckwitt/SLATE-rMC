# Decision ledger

## D001: Greenfield repository

Original RASIM and the manuscript are archived provenance references. The new repository has no Git or runtime dependency on them.

## D002: Correct result over method parity

Analytic and independently converged results outrank original behavior. Implementation technology is unrestricted.

## D003: One immutable legacy pack

One serial task creates the external tracked reference verification pack. Physics branches cannot create their own legacy evidence.

## D004: Shared serial proof spine

Bootstrap owns units, transforms, OSC mapping, complex normal-wavevector selection, scalar interface amplitude, contracts, trace schema, and synthetic plumbing.

## D005: One OSC orientation conversion

The expected mapping is `np.rot90(osc_raw, -1)` and is confirmed by a non-square fixture. Downstream rotation is prohibited.

## D006: Typed detector coordinates

Array indices are `[row,column]`. Continuous coordinates are `(column_px,row_px)`.

## D007: Scalar field-amplitude Fresnel model

Use `T12=2*k1z/(k1z+k2z)`. Reject the original s/p power-transmittance average in the scalar scattering model.

## D008: Uniform-depth attenuation is the first off-specular reference

Use the manuscript complex-`kz` uniform-depth intensity average with one transmitted incident and one transmitted exit channel. Full multilayer distorted fields are a later named model.

## D009: Parratt is a separate specular calculation

Expose pure Parratt, pure kinematic, and named smooth composite outputs. Do not silently use the composite as a general off-specular optical field.

## D010: Raw amplitudes and complete rods

No normalization to 100, rounding, artificial fractional reflections, or proof-mode pruning.

## D011: Distinct rods, explicit families

Every `(h,k)` rod remains distinct. `Qr` is family metadata and is collapsed only by a declared measurement or fitting selection.

## D012: Event-aligned model contracts

Ordered and stacking models return scattering strengths aligned by `event_id`. Internal grids and interpolation are implementation details with convergence proof.

## D013: One explicit detector measure

For each incident ray and phase/parent, integration selects from one all-rod candidate pool using the full once-only physical mass. Selected events deposit equal `T/N`; source PDF, structure, mosaic, selection probability, and solid angle are not reapplied.

## D014: Branch and Qr selection follow integration

Rod metadata is produced by the ordered subsystem, but measured association and branch selection are defined only after the forward subsystems compose. They are frozen in a selection manifest before one fit run.

## D015: Initial fits use detector-native coordinates

Geometry, mosaic, and ordered intensity fitting do not require `2theta/phi` or caking. Caking is a later measurement transformation.

## D016: Staged fitting

Geometry is fixed before mosaic, mosaic before ordered intensity, and ordered intensity before stacking disorder.

## D017: Frozen association during fitting

Optimizers may not dynamically switch rod or branch identity. Ambiguous or invalid associations are rejected before the fit.

## D018: Reuse is architectural

Immutable compiled states and an explicit invalidation graph are required so repeated intensity fits do not rerun geometry.

## D019: Profile before acceleration

The integrated reference path is profiled before choosing CPU, GPU, or another production implementation.

## D020: Few permanent tests

Keep compact analytic and direct-oracle tests. Large legacy traces, sweeps, images, and benchmarks remain external.

## D021: One external diagnostic file

Persistent diagnostics are one external `.ra_diag.npz` with one JSON manifest and no sidecars.


## D022: Physical branch identity

Branch identity uses signed wrapped reciprocal azimuth in a declared sample/crystal in-plane basis. Raw OSC row/column sign is never the identity. Projected detector side is derived evidence only.

## D023: Exact radial-family identity

Floating `Qr` is a reported value and tolerance check, not the sole key. Hexagonal families retain exact integer `m`; general-cell families retain an exact reciprocal-metric key and reciprocal-cell revision.

## D024: Re-index only between fits

Geometry fits use frozen associations. A changed geometry triggers an explicit outer re-index audit and a new selection-manifest revision, never an identity switch inside an objective.

## D025: Split geometry calibration

Source phase space, detector calibration, and sample/goniometer alignment are separate stages unless data limitations force a declared joint fallback. The fallback must report lost identifiability.

## D026: Explicit data likelihood

Raw counts, dark-subtracted data, and variance-weighted continuous observations are not interchangeable. Every fit records its data model, mask, variance, background, and scale semantics.
