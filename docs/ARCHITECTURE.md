# Architecture

## Design rule

The repository has one production path from a configured incident beam to a detector-native
observable. Scientific state is immutable, transformations are explicit, and the primary public
model is a callable function rather than a sampled cloud or image cache. Deterministic integration
and weighted Monte Carlo are explicit terminal estimators of that shared physics.

```text
strict YAML + CIF
  -> sampled source rows
  -> canonical entrance transport and incident film-phase ki
  -> physical reciprocal rods + continuous mosaic/finite-stack strength
  -> analytic rod/Ewald roots and exit optics
  -> per-source, per-physical-rod continuous detector-coordinate density
  -> all-source, all-rod, all-root intensity sum on the detector function
  -> selected-center comparison, display-only center sampling, or detector-pixel box integration
  -> optional terminal alternative: weighted forward samples -> exact native-pixel owners
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
  fields, linear rod restriction/physics rebinding, compiled CPU/CUDA evaluation, and the optional
  progressive CPU/CUDA Monte Carlo pixel-mass estimator.
- `pipeline/configured_simulation.py`: strict YAML boundary, canonical model construction, and
  quantitative pixel integration or display-only native-center density sampling. Its geometry-only
  input/context builders stop before structure strength or mosaic construction.
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
- `fitting/ordered_intensity.py`: fixed detector/mosaic sparse ROI-mass responses plus certified
  source-averaged selected-center responses, fixed-position Bi2Se3 occupancy quadratics,
  directional `Qr/Qz` damping, analytic image-scale projection, and structural rank/correlation
  diagnostics. The source-averaged path sums every incident state into one detector function per
  incidence before comparison and certifies its `Uz` interpolation against full-detector probes.
  Dataset-ID-bound observations carry distinct mass/density measures, while numerical-response,
  structure, mosaic, source, instrument, backend, and rod-catalog revisions preserve provenance.
  The full strength model is retained as a proof oracle, not called by optimizer iterations.
- `measurement`: downstream detector-derived observables, including the continuous normalized
  `(phi, 2theta)` coordinate pullback, per-rod all-root angular signal, the full finite-pixel angle
  projector, and an exact cropped physical-pixel projector for fully panel-contained measured
  profiles; never part of raw rendering.
- `proof`: compact analytic, reference, mutation, convergence, and benchmark evidence.

## Public runtime layers

1. `MosaicBraggSpace` owns the latent reciprocal measure for every physical rod.
2. `ContinuousEwaldCoating` restricts one rod to one analytic Ewald root for intrinsic diagnostics.
3. `DetectorEwaldMeasure` pulls all inverse branches onto arbitrary detector coordinates and
   integrates one incident state's pixels.
4. `SourceAveragedDetectorEwaldMeasure` sums independent incident-state intensities before any
   selected-center comparison, display sample, or requested detector integration. Its optional
   stochastic terminal samples the same latent/source physics and streams weighted roots directly
   into half-open native-pixel boxes.
   `CompiledMonteCarloDetectorSampler` is an explicit mutable execution resource, never hidden state
   on that immutable model. It retains a prefix-stable accumulator and either four-or-fewer stable
   CPU blocks or one thread-confined persistent CUDA workspace.
5. `configured_simulation` assembles those objects from one validated YAML document.
6. `ContinuousDetectorGeometryModel` binds callable reference and trial fields while reusing packed
   structure/mosaic state; its private exact-tag geometry rebuilds only canonical incident,
   exit-refraction, and detector geometry for each landmark trial.
7. `ContinuousNormalizedAngleFunction` reparameterizes one bound detector field with an explicit
   coordinate Jacobian and separate `S/N` measures; it does not rasterize or alter detector physics.
8. The mosaic-profile adapter evaluates only frozen angular quadrature nodes, integrates `S` and
   `N` before division, and caches exact component responses by width. CUDA performs the expensive
   detector evaluations; the small deterministic profile search remains on the CPU. The measured
   path projects only cropped raw-OSC pixels into the identical local-bin layout with exact polygon
   overlap. A frozen sideband gate removes whole weak profiles, and explicit policy identities
   remove whole secondary-lobe profiles before the one joint fit; neither operation masks bins by
   their central-profile intensity. Explicit nonzero profiles use the frozen OSC indexing
   selection, while branchless `00L` profiles require raw-significant observed support and a
   representable fixed-model landmark. Geometry-audited inverse-support boundary bins may be frozen
   out identically from every simulated component.
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
12. Ordered-intensity fitting retains the finite-ROI mass path and a certified source-averaged
    selected-center path. The latter produces one combined detector function per incidence before
    any dataset scale or residual, retains every weak nonzero anchor and admitted `m=0` anchor, and
    contracts only occupancy coefficients and directional Debye-Waller factors. Atomic positions,
    geometry, mosaic, lattice, optics, and stacking law are immutable. Multi-incidence observations
    join by dataset ID and exact observable-layout digest rather than tuple position. The active
    Unfiltered Bi2Se3 proof mode uses 250 shared-revision source rows and `88/78/72` centers,
    including six `m=0`; a measured-mosaic handoff instead propagates its exact fit-eligible
    identity set. Both prove synthetic selected-component recovery, not unresolved raw-OSC
    intensity recovery.
13. Native-center rendering samples the final combined detector density once per native pixel. It
    is display-only `A^2/px^2`, not pixel-integrated mass, OSC counts, or a count-calibrated fit.
14. The interactive Monte Carlo viewer owns one latest-only scheduler and one long-lived render
    worker. Detector-only controls compile one batched projection and swap only four projection
    buffers; even after a nonzero sample correction they retain the already-bound incident
    transport. Sample/goniometer controls rebuild incident transport and re-enumerate rods only
    when validity topology changes. CUDA previews normalize the raw float64 accumulator into one
    full-native float32 staging frame, and a persistent OpenGL R32F texture applies only
    masking/log/colormap presentation. Matplotlib is an explicitly selected full-native software
    fallback, never an automatic backend substitution.

The scalar NumPy path is the readable oracle. Compiled CPU and CUDA kernels reuse immutable packed
state and must reproduce it within the frozen tolerance. Device initialization and caches are never
module-global or import-time side effects.

## Invalidation boundaries

- Source changes rebuild source rows and incident transport.
- Sample entrance geometry or material optics changes rebuild incident transport.
- CIF, finite-stack, mosaic, wavelength, or sample/crystal orientation changes rebuild Bragg and
  detector evaluator state.
- Detector rigid pose, distance, or either detector tilt uses the sampler's projection-only rebind
  without rebuilding incident transport or detector-independent structure amplitudes. Detector
  pixel pitch, shape, or reference-coordinate calibration changes invalidate the compiled sampler
  and rebuild detector state.
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
- A measured mosaic observation is additionally invalidated by any OSC bytes, detector-valid mask,
  exact local projector, measured-selection policy, or nuisance-basis profile revision change.
- A source-averaged selected-center response is invalidated by source count/revision, selected-center
  layout, rods, mosaic or structure physics, instrument, backend, or device. Geometry tag/marker
  construction independently rebuilds exactly one ideal source-center, zero-divergence,
  mean-wavelength companion state and never inherits the intensity ensemble size.
- Display limits and colormaps never invalidate physics.

## Deliberately absent runtime structures

There is no retained mosaic-orientation batch, sampled scattering-candidate pool, scattering-event
table, outgoing-event table, hit table, general point depositor, bilinear rasterizer, discrete
sphere texture, compatibility adapter, or parallel legacy simulator. The optional Monte Carlo
terminal generates one fixed-width source/draw latent matrix and streams roots directly into the
final image; it retains no orientation/root/event Cartesian product. Historical equations remain
only in the immutable reference pack and proof comparisons.
