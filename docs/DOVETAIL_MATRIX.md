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
| finite ordered-parent strength | per-rod `S_r(L;K)` | `MosaicBraggSpace` | raw nonnegative strength; explicit 2H or R-centered 3R parent, no source, mosaic, optics, or detector factor |
| `MosaicBraggSpace` | latent `Q` and Bragg density | Ewald coating and detector inverse map | folded-alpha/full-beta law; rod intensities summed only after per-rod evaluation |
| analytic Ewald solver | roots and coarea | intrinsic coating | stable line/sphere equation; explicit regular/tangent/no-root status |
| `DetectorEwaldMeasure` intrinsic direction evaluator | `EwaldDirectionIntensity` | proof and publication sampling | complete internal-film outgoing sphere in `A2/sr`; all regular inverse preimages; no exit, `Qz`, panel, optical, source, or detector-solid-angle factor; non-specular `m=0` excluded |
| `DetectorEwaldMeasure` detector-visible direction evaluator | `DetectorVisibleEwaldDirectionIntensity` | active-panel Ewald publication patch | configured detector coordinates select top-exit directions; intrinsic `A2/sr` density retains no optical/source/detector Jacobian; all regular inverse preimages include nonzero `m=0`; positive Q gap excludes direct `Q=0` |
| detector geometry | arbitrary coordinate rays | detector measure | detector point → front-facing air ray → refracted film `kf` → `Q`; active panel only |
| `DetectorEwaldMeasure` | one-state coordinate density | source average / pixel integration | all inverse branches, optics, source and phase factors exactly once |
| source-averaged measure | summed coordinate density | pixel integrator | independent states and wavelengths add as intensities before one box integral |
| native-pixel integrator | raw pixel mass | rendering / future fitting | deterministic finite box integral; no point deposition or image normalization |
| source-averaged latent measure | stochastic raw pixel-mass estimate | optional rendering / comparison | fixed-source stratification, exact mosaic proposal, every rod/root, weighted hard-bin ownership; no hit table, count calibration, rejection renormalization, or Gaussian-error claim |
| compiled Monte Carlo sampler | progressive raw accumulator and authoritative float64 snapshots | latest-only interactive render worker | fixed Philox source/draw prefixes; explicit CPU/CUDA provenance; detector-pose rebind changes only four projection arrays, while general geometry rebind requires unchanged source/topology/physics; cancellation returns no partial result |
| full-native presentation lease | contiguous float32 `[row,column]` frame | OpenGL R32F texture | transient until the sampler's next operation; no crop, transpose, or pre-upload spatial resampling; no detector correction or scientific-result substitution |
| bound continuous detector function | raw detector-coordinate density | continuous angle measurement | inverse-map `(phi,2theta)` with the corrected owned pose; apply the detector-area coordinate Jacobian once and retain separate `S/N` |
| declared layered reciprocal frame + immutable reciprocal-profile region | detector-pixel centers or continuous detector coordinates | finite `Qr/L` profile reduction | map through one explicit reciprocal basis/axial axis, retain detector-side identity, assign half-open axial bins, and accumulate signal and measure separately before division; no material-family equation, smoothing, or model raster |
| frozen angle-profile definitions + all-root detector function | `MosaicProfileSet` | mosaic response-bank fitter | integrate finite-bin `S` and `N` separately; preserve indexed nonzero branches, admitted branchless `00L` profiles, frozen geometry-only bin exclusions, and physical source revision; no raster |
| exact Gaussian/Lorentzian component bank | profiled relative-shape residual | mosaic parameter search | one nonnegative nuisance amplitude per individual profile removes absolute and cross-peak intensity assumptions; every profile contributes equal relative shape error; audit both eta faces and a declared finite centered-logit grid |
| source-averaged all-root detector + frozen selected centers | selected-group angular signal density | fixed-position ordered-intensity fitter | sum every source state into one detector function per incidence before rod selection or residuals; keep point density distinct from ROI mass; certify `Uz` interpolation against interlaced full-detector probes |
| detector-native measured image + mask + angle geometry | `MeasuredPeakDiscovery` | reciprocal indexing | global angle-chart discovery accepts no marker catalogue or predicted coordinates; proposals are refined once in native coordinates |
| measured discovery + canonical detector/Ewald geometry | per-image indexing decisions | branch-track selection | infer `m`, integer `L`, Ewald branch, and root sign only from discovered `Q`; deterministically drop ambiguous ownership |
| distinct-incidence image decisions | immutable `MeasuredIndexingResult` | staged fitting | require replicated branch tracks and shared site identities; all image, mask, calibration, reciprocal, and policy hashes remain frozen |
| strict OSC-series manifest | image IDs, paths, commanded angles, and geometry-only contexts | measured indexing / joint fit | join by exact image ID; never infer motor angle from filename or OSC header; one material/mount per fit group |
| exact integer-L marker solver | frozen root identities and native coordinates | geometry fitter | every physical rod precedes grouping; analytic root sign is part of identity |
| exact rational-layer solver + ideal PbI2 parent-support catalogue | one complete `LayerLMarkerObservations` pack | shared geometry fitter | upstream must qualify optional centroids; exact overlaps become one rod-free physical residual with unioned signed-rod provenance; absent optional pack preserves the legacy integer baseline |
| bound continuous detector function | key-aligned exact-tag predictions | bounded least squares | private geometry-only tag engine; no intensity evaluation or dynamic reassignment inside the objective |
| geometry fitter | pivoted pose corrections plus rank/residual diagnostics | outer marker audit | the local pack uses one fixed LAB pivot; the shared-series pack fits only its two transported axis-perpendicular offsets; each pack must be full rank |
| frozen observations + per-image exact-tag models | canonical joint residual, selected shared corrections, one common incidence delta, and optional zero-sum trims | qualification / downstream staged fits | trims use a canonical Helmert basis and cannot replace the shared mean delta; the layered-Bi2X3 pack fixes the sample-x gauge and reports the complete scaled-rank diagnostic |
| accepted position artifact | exact corrections, commanded angles, one common delta, zero-sum trims, effective angles, and scientific revision | conditioned lattice decision | fit only tightly regularized near-CIF hexagonal in-plane/normal strains and serialize the constrained full basis; promote it only with admissible data-only rank, condition, and improvement |
| position + retained/accepted lattice state + strict provided mosaic state | exact fixed experiment series | mixed-chart preparation | consume supplied states explicitly; reuse immutable source/material/reciprocal state across views; rebuild every lattice-dependent object after promotion; never fall back to case-file or nominal geometry |
| verified native-pixel count field + converged continuous-region projector + frozen shared radial background + continuous detector callable | A -> B -> C -> joint matched-region chain | structure/SF fitter | `m=0` is selected and binned in phi/2theta; `m!=0` in signed-side Qr/L; piecewise-constant measured pixels and the unrasterized model use the same rectangles; fractional pixel sharing retains full covariance; one scale per OSC spans every family and one five-coordinate structure spans all OSCs; full outer-site occupancy is `(1-x) X + x Bi`; each child requires its exact predecessor and background SHA; no smoothing or simulated detector raster |
| joint structure | `FIT_CONDITIONED` full-Qz profile diagnostic | Figure-7 renderer | integrate both the piecewise-constant measured field and fitted density over continuous phi/2theta or signed-Qr/L chart rectangles mapped through detector area; display the measured detector pixels and the resulting ROI support; preserve the fitted rod scope and `publication_ready=false` |
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
