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

## D012: Query-aligned ordered model contracts

Ordered and stacking models return scattering strengths aligned by immutable query IDs. Internal
grids and interpolation are implementation details with convergence proof. The historical
`EventIntensityResult` name denotes this query result, not a sampled scattering-event runtime.

## D013: One explicit detector measure (superseded by D029)

The finite candidate-pool and point-deposition implementation was an integration-stage contract. It
is retired; D029 is the current detector measure.

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

## D027: Owner-derived beam-to-ki revisions

`n_complex` is the sole stored material optical array. `MaterialOptics` derives `material_revision`
under schema `material_optics_revision.v2`, `CompiledInstrument` derives
`sample_geometry_revision` under schema `sample_entrance_revision.v2`, and transport copies both
revisions into the complete incident envelope without rehashing. The v2 digests are
provenance/cache rebaselines rather than numerical corrections; contract API v8 records the
structural field removal.

## D028: Causal geometry identity

An unbounded sample plane is identified by its full orientation and signed LAB normal offset at the
frozen `1e-12 m` position resolution, not by two arbitrary tangent-origin coordinates. Compiled state
retains only transforms with numerical or proof consumers; ordered goniometer composition remains a
local step and no `lab_from_crystal` derivative is stored. Detector-angle fingerprint v2 hashes only
the detector transform, shape, pitches, and reference coordinate, while `AngleFrame` remains a
separate cache-key owner.

## D029: Continuous detector pushforward

Contract API v9 replaces sampled orientations, scattering-event rows, selection, and point
deposition with one continuous latent Bragg measure, analytic Ewald restriction, exact inverse
detector-coordinate pushforward, incoherent source-state sum, and deterministic pixel-box
integration. The raw coordinate density and integrated pixel mass are separately named measures.
There are no compatibility shims for the retired runtime.

## D030: One YAML fixture authority

`configs/bi2se3_simulation.yaml` owns the default source, material, sample, mosaic, finite-2H,
detector, numerical backend, and artifact switches. Specialized diagnostic CLIs inherit omitted
physical values from it and may expose explicit overrides only for controlled studies. Schema v2
makes the execution backend explicit; v1 files are rejected instead of being reinterpreted under a
changed required-field contract.

## D031: Display arrays are not models

Reciprocal-space and Ewald-coating arrays may be sampled for visualization, and detector macrobins
may be integrated for a preview, but no sampled array, sphere object, mesh, texture, or image becomes
the authoritative physical field.

## D032: Complete detector-density reduction precedes pixels

Contract API v10 makes the all-source, all-physical-rod, all-root continuous detector-coordinate
density an explicit public result. Fixed macrobin rendering consumes only that reduced field and
applies one terminal box quadrature; it no longer carries a per-rod pixel tensor. The configured
diagnostic is therefore `rasim-configured-result-v2` and omits the former per-rod image and derived
per-family pixel masses. The detailed per-rod coordinate result remains available for scientific
proof without becoming the production rendering interface. The former source-averaged per-rod
native-pixel integrator fails closed unless a caller explicitly requests
`include_per_rod_evidence=True`; no configured renderer sets that proof-only flag. No v1
compatibility raster is retained.
