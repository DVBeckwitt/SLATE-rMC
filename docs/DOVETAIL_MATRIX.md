# Dovetail matrix

This matrix records the live production boundaries. Historical task documents describe earlier
phases but do not override these owners.

| Producer | Output | Consumer | Boundary rule |
|---|---|---|---|
| strict YAML loader | `SimulationConfiguration` | configured builder | one schema, no aliases, unknown keys, or implicit backend |
| source sampler | `IncidentSampleBatch` | incident transport | complete rows built once; empirical weights preserved |
| instrument compiler | `CompiledInstrument` | incident and detector geometry | one canonical pose, including both detector tilts |
| incident transport | `IncidentTransportResult` / `IncidentStateBatch` | detector measures | canonical film-phase `ki`, entrance field, attenuation inputs, and failure status |
| CIF/material reader | crystal and `MaterialOptics` | strength and optics | one composition/data source and one material revision |
| reciprocal catalog | physical `Rod` tuple | Bragg space and detector | every `(h,k)` retained; exact family metadata is not identity collapse |
| finite-2H strength | per-rod `S_r(L;K)` | `MosaicBraggSpace` | raw nonnegative strength; no source, mosaic, optics, or detector factor |
| `MosaicBraggSpace` | latent `Q` and Bragg density | Ewald coating and detector inverse map | folded-alpha/full-beta law; rod intensities summed only after per-rod evaluation |
| analytic Ewald solver | roots and coarea | intrinsic coating | stable line/sphere equation; explicit regular/tangent/no-root status |
| detector geometry | arbitrary coordinate rays | detector measure | detector point → front-facing air ray → refracted film `kf` → `Q`; active panel only |
| `DetectorEwaldMeasure` | one-state coordinate density | source average / pixel integration | all inverse branches, optics, source and phase factors exactly once |
| source-averaged measure | summed coordinate density | pixel integrator | independent states and wavelengths add as intensities before one box integral |
| native-pixel integrator | raw pixel mass | rendering / future fitting | deterministic finite box integral; no point deposition or image normalization |
| exact integer-L marker solver | frozen root identities and native coordinates | geometry fitter | every physical rod precedes grouping; analytic root sign is part of identity |
| bound continuous detector function | key-aligned exact-tag predictions | bounded least squares | private geometry-only tag engine; no intensity evaluation or dynamic reassignment inside the objective |
| geometry fitter | pivoted pose corrections plus rank/residual diagnostics | outer marker audit | one fixed LAB sample pivot and a full-rank local pack are required; final visible roots are independently re-enumerated |

## Factor reduction order

```text
atom amplitudes -> per-rod finite-stack strength
per-rod strength * rod population * mosaic density
-> analytic Ewald/inverse-map restriction
-> entrance/exit optical and attenuation factors
-> per-source detector-coordinate density
-> incoherent source/wavelength/phase sum
-> detector-pixel box integral
-> optional separately named measurement transforms
```

Independent rods, source states, wavelengths, phases, and parents sum as intensities. Coherent
atomic/layer contributions sum as amplitudes only inside their owning strength model.

## Deferred consumers

Intensity/profile fitting, masks, background, detector PSF/efficiency, saturation, caking, and
reciprocal remapping consume the accepted detector result later. None may recreate sampled
scattering events or move a factor upstream without a new declared measure and proof. The accepted
exact-marker geometry fit consumes geometry only and therefore does not require detector intensity
or pixel integration.
