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

## D015: Geometry fitting remains detector-native

Geometry fitting uses native detector coordinates. Mosaic fitting may use the continuous
`(2theta,phi)` pullback, and selected-center ordered-intensity fitting may compare continuous angular
signal density. Neither path requires or fits a caked raster; finite caking remains a downstream
measurement product.

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
the authoritative physical field. Native-center density images are likewise display samples of the
completed detector function, not pixel-mass or count observables.

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

Selected-center fitting and native-center display sampling are additional terminal consumers of the
same completed field. Every source intensity is reduced before any dataset scale, normalization,
comparison, or residual. Selected-center density, center-sampled detector density, finite-ROI mass,
and pixel-box mass remain distinct declared measures.

## D033: Multi-OSC geometry uses one geometry-only spine and immutable frozen keys

An OSC geometry fit compiles one ideal source-center, zero-divergence, mean-wavelength material and
reciprocal context, reuses it across declared commanded angles, and performs no structure-strength,
mosaic-probability, raster, or pixel work. Under fixed detector center/distance/pitch and explicit
gauge ownership, the maximum accepted 5/10/15-degree pack has nine shared coordinates: detector
x/y tilt, sample x/y tilt, goniometer-axis pitch/yaw, signed sample-plane normal offset, and two
transported axis-perpendicular pivot offsets. Detector roll, crystal axial roll, sample tangent
translations, and axis-parallel pivot motion remain gauges and are not regularized into the fit.

The optimizer never changes a frozen full `(m,L,analytic branch,root sign,rod)` key. After fitting,
one audit directly brackets the fixed-`L` elastic equation independently of the production root
solver. A second acceptance audit relabels exactly the frozen position-free native candidates under
the corrected geometry and requires coherent unchanged keys. A fresh global cake search is retained
only as an operational diagnostic because chart sampling and same-key lobe ownership are
geometry-dependent. Newly visible keys or alternate lobes cannot delete original observations or
trigger an automatic censor-and-refit cycle.

## D034: Hash-bound portable staged-fit cases

Accepted multi-OSC results are replayed from repository-relative cases that hash both containers
and decoded OSC arrays. Geometry deliberately uses one ideal state; mosaic and ordered intensity
rebuild the same 250-state source realization and reduce it before comparison. Each stage hashes
only compact scientific state and the upstream scientific revision; generated diagnostics and
images remain external. Exact identities and masks are separated from tolerance-bound fit floats,
and the historical Bi2Te3 selection revision remains distinct from the revision recomputed under
the case-bound numerical runtime. The separately frozen catalog row-manifest and catalog-file hash
are two more audit identities, neither a substitute selection revision. Bi2Te3's accepted
250-state mosaic and ordered fits remain current-lock CUDA-qualified and model-limited; its optional
render retains only a historical RTX-3060 pixel oracle pending current-lock requalification. The
runner checks the executing
transitive numerical package versions against the case-bound `uv.lock` before creating an output or
evaluating geometry. The lightweight `packaging` dependency evaluates the lock's PEP 508 markers so
a platform-conditional package is required only where uv installs it. Every stage records the
interpreter, platform, and package identity as non-scientific operational provenance, and resume
requires an exact match. Platform portability is still established by the declared output checks,
not presumed from a matching package lock.

## D035: Optional weighted Monte Carlo detector-pixel estimator

Contract API v11 adds a terminal stochastic estimator without changing D029's continuous detector
measure or deterministic pixel integral. For every fixed source state, it samples the declared
folded-alpha/full-beta mosaic law, enumerates every physical rod and retained analytic Ewald root,
applies the same structure, coarea, optical, source, phase, and polarization factors once, and
hard-bins the resulting forward detector coordinate. Deposits are importance-weighted raw mass in
`A2`; root-hit totals are work diagnostics, not photon counts. Invalid, no-root, and off-panel draws
remain zero and are never resampled or renormalized.

The estimator streams directly into the native image and retains no orientation batch, candidate
pool, event table, hit table, or point-deposition subsystem. Its distinct measure is
`raw_detector_pixel_mass_monte_carlo_estimate_A2.v1`; deterministic
`raw_detector_pixel_mass_A2.v1` remains the quantitative reference. Near visible Ewald folds the
importance weights can have an infinite second moment, so replicate spread is diagnostic rather
than a guaranteed Gaussian error bar. Independent refined latent integration remains the proof
authority.

## D036: Progressive execution is explicit and presentation is not a measure

Contract API v12 keeps D035's estimator and physics unchanged while making reusable execution
state explicit. `CompiledMonteCarloDetectorSampler` is mutable, thread-confined, and owned by one
long-lived render worker; the detector model remains immutable. NumPy Philox reserves a fixed-width
source/draw latent layout so the 1/4/8/requested-draw sequence is a true prefix. CPU execution uses
at most four private full-native blocks with stable reduction. CUDA owns persistent packed state,
deposits roots directly into one raw float64 accumulator, and fails closed when unavailable.

Only unchanged source rows, active evaluator topology, rods, physics, detector calibration, sample
support, film, and crystal mount may use geometry rebinding. A narrower detector-pose rebind also
requires an unchanged sample pose and swaps only the compiled detector covectors, normal, and
per-state ray-origin projection arrays; transport arrays remain resident and unchanged. A
cancellation or execution failure returns no partial scientific result and discards contaminated
mutable execution state before reuse. The interactive scheduler accepts output only from its latest
revision.

`MonteCarloDetectorPresentation` is a leased full-native float32 rendering frame,
valid only until the sampler's next operation. The OpenGL path uploads it to one R32F texture and
performs logarithmic color mapping in a shader. It is neither retained evidence nor a new measure;
the authoritative snapshot remains the validated float64
`raw_detector_pixel_mass_monte_carlo_estimate_A2.v1`. Matplotlib remains an explicit software
choice, and neither execution nor presentation silently falls back.

`PySide6-Essentials` is confined to the optional `visualization` extra. It is the smallest Qt
distribution that supplies `QOpenGLWidget`, OpenGL functions/textures/shaders, and the Shiboken
buffer pointer needed by the persistent R32F presenter. It is imported only when OpenGL
presentation is selected and adds nothing to the core dependency or import path.
