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
  -> continuous detector-coordinate density
  -> source-state intensity sum
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
  display-only raster evaluation.
- `fitting/geometry.py`: callable pose-bound detector fields, their exact-L tagged landmarks,
  the shared public site-plus-line objective diagnostic, detector-coordinate and line-angle pose fitting, rank diagnostics, and post-fit root
  re-enumeration.
- `measurement`: later detector-derived coordinate transforms; never part of raw rendering.
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
   structure/mosaic state; its associated `IntegerLGeometryModel` rebuilds only canonical incident,
   exit-refraction, and detector geometry for each exact-L landmark trial.

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
- The tagged-function fitter never rasterizes either detector field. It fits exact landmarks and
  their line angles, so it does not need mosaic/SF intensity values, centroids, or detector
  quadrature. Detector/sample pose trials reuse the reciprocal basis, rods, finite-2H model, and
  mosaic contract; sample-normal trials rebuild nominal incident transport, and every trial
  rebuilds exit-refraction/detector geometry.
- Display limits and colormaps never invalidate physics.

## Deliberately absent runtime structures

There is no sampled mosaic-orientation batch, candidate pool, scattering-event table, outgoing-event
table, point depositor, bilinear rasterizer, discrete sphere texture, compatibility adapter, or
parallel legacy simulator. Historical equations remain only in the immutable reference pack and
proof comparisons.
