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
| source-averaged latent measure | stochastic raw pixel-mass estimate | optional rendering / comparison | fixed-source stratification, exact mosaic proposal, every rod/root, weighted hard-bin ownership; no hit table, count calibration, rejection renormalization, or Gaussian-error claim |
| bound continuous detector function | raw detector-coordinate density | continuous angle measurement | inverse-map `(phi,2theta)` with the corrected owned pose; apply the detector-area coordinate Jacobian once and retain separate `S/N` |
| frozen angle-profile definitions + all-root detector function | `MosaicProfileSet` | mosaic response-bank fitter | integrate finite-bin `S` and `N` separately; preserve indexed nonzero branches, admitted branchless `00L` profiles, frozen geometry-only bin exclusions, and physical source revision; no raster |
| exact Gaussian/Lorentzian component bank | profiled relative-shape residual | mosaic parameter search | one nonnegative nuisance amplitude per individual profile removes absolute and cross-peak intensity assumptions; every profile contributes equal relative shape error; audit both eta faces and a declared finite centered-logit grid |
| source-averaged all-root detector + frozen selected centers | selected-group angular signal density | fixed-position ordered-intensity fitter | sum every source state into one detector function per incidence before rod selection or residuals; keep point density distinct from ROI mass; certify `Uz` interpolation against interlaced full-detector probes |
| detector-native measured image + mask + angle geometry | `MeasuredPeakDiscovery` | reciprocal indexing | global angle-chart discovery accepts no marker catalogue or predicted coordinates; proposals are refined once in native coordinates |
| measured discovery + canonical detector/Ewald geometry | per-image indexing decisions | branch-track selection | infer `m`, integer `L`, Ewald branch, and root sign only from discovered `Q`; deterministically drop ambiguous ownership |
| distinct-incidence image decisions | immutable `MeasuredIndexingResult` | staged fitting | require replicated branch tracks and shared site identities; all image, mask, calibration, reciprocal, and policy hashes remain frozen |
| strict OSC-series manifest | image IDs, paths, commanded angles, and geometry-only contexts | measured indexing / joint fit | join by exact image ID; never infer motor angle from filename or OSC header; one material/mount per fit group |
| exact integer-L marker solver | frozen root identities and native coordinates | geometry fitter | every physical rod precedes grouping; analytic root sign is part of identity |
| bound continuous detector function | key-aligned exact-tag predictions | bounded least squares | private geometry-only tag engine; no intensity evaluation or dynamic reassignment inside the objective |
| geometry fitter | pivoted pose corrections plus rank/residual diagnostics | outer marker audit | the local pack uses one fixed LAB pivot; the shared-series pack fits only its two transported axis-perpendicular offsets; each pack must be full rank |
| frozen observations + per-image exact-tag models | canonical joint residual and nine shared corrections | qualification / downstream staged fits | 5/10/15-degree rank ladder is 5/9 -> 7/9 -> 9/9; detector calibration and gauge coordinates stay fixed |
| corrected geometry + selected position-free native candidates | corrected reciprocal labels on unchanged coordinates | frozen-key acceptance audit | require every original full key and coherent frozen tracks; no cake search, coordinate refinement, dynamic reassignment, or censor/refit |
| corrected geometry + original OSC series | fresh global marker decisions | operational discovery diagnostic | report chart/candidate and same-key lobe changes separately; this geometry-dependent pass cannot replace frozen observations |

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

General raw-image profile extraction beyond the tracked Bi2Se3 and Bi2Te3 series, physical forward-model
masks/background, detector PSF/efficiency, saturation, general finite-bin angle products, and
reciprocal remapping consume the accepted detector result later. The accepted real-OSC mosaic slice
projects exact cropped physical-pixel overlap into frozen finite-bin profiles and applies signed
low-rank detrending; it does not fit a detector raster or claim a physical background. None may
recreate sampled scattering events or move a factor upstream without a new declared measure and
proof. The accepted exact-marker geometry fit consumes geometry only and therefore does not require
detector intensity or pixel integration.
