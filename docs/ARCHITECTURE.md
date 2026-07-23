# Architecture

## Design rule

The repository has one production path from a configured incident beam to a detector-native
observable. Scientific state is immutable, transformations are explicit, and the public model is a
callable function rather than a sampled cloud or image cache.

```text
strict YAML + CIF
  -> sampled source rows
  -> canonical entrance transport and incident film-phase ki
  -> physical reciprocal rods + continuous mosaic/finite-stack strength
  -> analytic rod/Ewald roots and exit optics
  -> per-source, per-physical-rod continuous detector-coordinate density
  -> all-source, all-rod, all-root intensity sum on the detector function
  -> deterministic detector-pixel box integration
```

No physical Ewald-sphere object is created. A detector coordinate determines an outgoing air ray;
exit refraction determines its film wavevector; the elastic relation and inverse latent map recover
all contributing rod/orientation branches. Only the active detector panel is evaluated.

## Package ownership

### `painted_ewald`

- `types.py`: immutable rods, mosaic parameters, root status, and strength protocols.
- `rotations.py`: column-vector active rotations.
- `mosaic.py`: normalized folded-alpha/full-beta probability law and compact quadrature helpers.
- `rods.py`: detector-independent elastic-reach rod enumeration.
- `bragg.py`: continuous `(alpha, beta, u)` map and per-rod mosaic × population × strength density.
- `ewald.py`: stable analytic line/sphere roots with tangent and no-root status.
- `surface.py`: intrinsic Ewald restriction and its once-only coarea factor.

### `rasim_next`

- `core`: frames, units, immutable contracts, validity, transforms, and trace records.
- `geometry`: instrument compilation, source/sample intersection, incident transport, detector rays,
  and forward detector intersection.
- `materials` and `optics`: CIF-derived material data, shared complex-normal mode selection,
  refraction, Fresnel amplitudes, and uniform-depth attenuation.
- `ordered`, `stacking`, and `reflectivity`: structure amplitudes, finite stacks, stacking models,
  and separately named specular calculations.
- `reciprocal`: reciprocal basis and complete physical rod catalogs.
- `sampling`: deterministic source phase-space sampling only.
- `pipeline/bragg_space.py`: binds CIF-derived finite-2H strength to rods and mosaic geometry.
- `pipeline/continuous_detector.py`: one-incident-state detector pullback and native-pixel
  integration.
- `pipeline/source_averaged_detector.py`: incoherent summation of complete incident-state detector
  fields and compiled CPU/CUDA evaluation.
- `pipeline/configured_simulation.py`: strict YAML boundary, canonical model construction, and
  display-only raster evaluation. Its geometry-only input/context builders stop before structure
  strength or mosaic construction.
- `selection`: position-free angle-chart discovery, detector-native peak refinement, reciprocal
  identity inference, immutable cross-incidence branch manifests, strict OSC-series ingestion, and
  frozen-key post-fit visibility audits.
- `fitting/geometry.py`: callable pose-bound detector fields, their exact-L tagged landmarks,
  the shared public site-plus-line objective diagnostic, detector-coordinate and line-angle pose
  fitting, rank diagnostics, and post-fit root re-enumeration.
- `fitting/indexed_series.py`: exact image-ID joins and one shared nine-coordinate
  detector/sample/axis/pivot correction pack across an arbitrary nonempty commanded-angle series;
  any nonempty coordinate subset may be active while the complement remains exactly fixed.
- `fitting/mosaic.py`: immutable finite-bin profile identities and response banks, exact
  per-profile nuisance-amplitude projection, deterministic width refinement, centered-logit eta
  search, and local/global identifiability diagnostics. Its continuous-profile entry point is a
  fitting-boundary adapter over the canonical measurement transform, not another angle mapping.
- `measurement`: downstream detector-derived observables, including the continuous normalized
  `(phi, 2theta)` coordinate pullback, per-rod all-root angular signal, and the finite-pixel angle
  projector; never part of raw rendering.
- `proof`: compact analytic, reference, mutation, convergence, and benchmark evidence.

## Public runtime layers

1. `MosaicBraggSpace` owns the latent reciprocal measure for every physical rod.
2. `ContinuousEwaldCoating` restricts one rod to one analytic Ewald root for intrinsic diagnostics.
3. `DetectorEwaldMeasure` pulls all inverse branches onto arbitrary detector coordinates and
   integrates one incident state's pixels.
4. `SourceAveragedDetectorEwaldMeasure` sums independent incident-state intensities before the one
   requested detector integration.
5. `configured_simulation` assembles those objects from one validated YAML document.
6. `ContinuousDetectorGeometryModel` binds callable reference and trial fields while reusing packed
   structure/mosaic state; its private exact-tag geometry rebuilds only canonical incident,
   exit-refraction, and detector geometry for each landmark trial.
7. `ContinuousNormalizedAngleFunction` reparameterizes one bound detector field with an explicit
   coordinate Jacobian and separate `S/N` measures; it does not rasterize or alter detector physics.
8. The mosaic-profile adapter evaluates only frozen angular quadrature nodes, integrates `S` and
   `N` before division, and caches exact component responses by width. CUDA performs the expensive
   detector evaluations; the small deterministic profile search remains on the CPU. Explicit
   nonzero profiles use the frozen OSC indexing selection, while branchless `00L` profiles require
   raw-significant observed support and a representable fixed-model landmark. Geometry-audited
   inverse-support boundary bins may be frozen out identically from truth and every component.
9. Measured selection discovers peaks without predicted coordinates, refines them on the native
   detector, infers discrete reciprocal identities, and freezes replicated branch tracks before
   fitting consumes them.
10. The OSC-series boundary joins declared files, motor angles, geometry-only material contexts, and
   frozen observations by image ID. The joint fitter concatenates canonical per-image residual
   blocks while applying one correction vector to every image.
11. Post-fit proof brackets the fixed-`L` elastic equation independently of the production root
    solver, then relabels exactly the selected native candidates under corrected geometry. A full
    corrected-geometry rediscovery is reported separately as a chart/candidate robustness
    diagnostic; it cannot delete or replace accepted observations.

The scalar NumPy path is the readable oracle. Compiled CPU and CUDA kernels reuse immutable packed
state and must reproduce it within the frozen tolerance. Device initialization and caches are never
module-global or import-time side effects.

## Invalidation boundaries

- Source changes rebuild source rows and incident transport.
- Sample entrance geometry or material optics changes rebuild incident transport.
- CIF, finite-stack, mosaic, wavelength, or sample/crystal orientation changes rebuild Bragg and
  detector evaluator state.
- Detector pose, distance, pitch, shape, or either detector tilt rebuilds detector geometry and its
  compiled evaluator, but not detector-independent structure amplitudes.
- The tagged-function fitters never rasterize a detector field. The OSC-series
  `ExactTagGeometryModel` constructs neither mosaic probability nor structure strength. The older
  `ContinuousDetectorGeometryModel` may receive already prepared intensity state, but its exact-tag
  trials evaluate none of it. Geometry-only trials reuse the reciprocal basis and rods, update the
  complete sample and detector rigid transforms, rebuild nominal incident transport, and rebuild
  exit-refraction and detector geometry.
- A measured selection manifest is invalidated by any image, mask, detector/cake calibration,
  reciprocal context, or policy hash change. A fitter never silently relabels a frozen manifest.
- A mosaic response bank is invalidated by any profile layout, angle frame, source, backend/device,
  material/rod catalogue, or fixed geometry revision change. Mosaic width and mixture changes reuse
  the frozen upstream geometry but require their exact component response.
- Display limits and colormaps never invalidate physics.

## Deliberately absent runtime structures

There is no sampled mosaic-orientation batch, sampled scattering-candidate pool, scattering-event
table, outgoing-event table, point depositor, bilinear rasterizer, discrete sphere texture,
compatibility adapter, or parallel legacy simulator. Historical equations remain only in the
immutable reference pack and proof comparisons.
