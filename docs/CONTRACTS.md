# Contracts

Contract API version: **9**. Trace schema version: **4**. Reference pack version: **1**.

Production contracts are frozen dataclasses or immutable model objects. Numeric arrays are copied to
contiguous, read-only storage at public boundaries. Shapes, units, frames, measure IDs, validity,
and ordering are validated eagerly.

## Global conventions

- Column vectors and active rotations.
- Radians internally.
- Metres for instrument positions; angstroms for wavelength and crystal lengths; inverse angstroms
  for wavevectors.
- Array access is `[row, column]`; continuous detector coordinates are `(column_px, row_px)`.
- `Q = kf - ki` in the declared frame.
- Invalid rows have explicit status and cannot carry fabricated nonzero intensity.
- IDs prove alignment and provenance; they are not numerical weights or sorting keys.

## Stable core data contracts

| Contract | Owner | Essential payload |
|---|---|---|
| `SourceConfiguration` / `sample_gaussian_source_rays` | configured pipeline / sampling | validated source parameters and immutable sampled source rows |
| `IncidentSampleBatch` | sampling | complete source rows, empirical weights, wavelength, polarization, provenance |
| `InstrumentConfiguration` / `CompiledInstrument` | geometry | canonical transforms, sample support, detector shape/pitch/reference coordinate, revisions |
| `IncidentStateBatch` | geometry | entrance-intersected air and film-phase `ki`, Fresnel amplitude, decay, footprint, source identity |
| `MaterialOptics` | materials | wavelength-aligned complex refractive index and material revision |
| `RodCatalog` | reciprocal | every physical `(h,k)` rod and exact family metadata |
| `RodQueryBatch` | ordered/stacking | rod-aligned `L` queries with stable IDs |
| `EventIntensityResult` | ordered/stacking | query-aligned amplitude, intensity, normalization, and model revision |
| `ParrattResult` / `SpecularResult` | reflectivity | separately named Parratt, kinematic, and composite specular outputs |

`EventIntensityResult` retains “event” in its historical type name, but it is an ordered query result;
it is not a sampled scattering-event runtime. Its strength excludes source probability, rod
population, mosaic probability, optics, polarization, detector Jacobians, and pixel integration.

## Continuous reciprocal contracts

### `MosaicParameters`

Owns Gaussian sigma, Lorentzian HWHM, mixture probability, and deterministic proof quadrature. A
zero width is permitted only for an inactive mixture component. The accepted continuous law is
folded alpha/full beta and integrates to one.

### `Rod`

Owns exact `(h,k)`, family metadata, and population. Rod identity is never a floating radial value.

### `MosaicBraggSpace`

- `map_latent(rod, alpha_rad, beta_rad, u_Ainv)` maps arbitrary broadcast coordinates to sample-frame
  `Q`.
- `evaluate_latent(...)` returns per-rod mosaic density, finite-stack strength, and their product.
- `rod_u_bounds_Ainv(rod)` returns the full elastic-reach axial interval.

The map and density are functions; a quadrature node set is never the model.

### `ContinuousEwaldCoating`

- `evaluate_geometry(rod, branch, alpha_rad, beta_rad)` returns only the analytic Ewald geometry for
  a non-specular rod/root and evaluates no mosaic or structure-factor intensity.
- `evaluate_latent(rod, branch, alpha_rad, beta_rad)` solves the analytic elastic root and returns
  geometry, strength, mosaic density, and the once-only Ewald coarea-weighted coating.
- `evaluate_specular_geometry(...)` exposes retained nonzero branch-0 `m=0` geometry without
  inventing finite direct-beam intensity. The algebraic `Q=0` direct root is suppressed. The
  separate detector-visible all-roots path may include regular nonzero `m=0` support only with a
  positive reciprocal-gap certificate.

## Detector contracts

### `DetectorEwaldMeasure`

Immutable one-incident-state model. It requires the canonical transported film-phase `ki`, matching
air wavelength, strict root classification, material optics, and compiled instrument.

- `evaluate_detector_geometry(column_px, row_px, *, include_surface_jacobian=True)` performs detector
  point → outgoing ray → exit refraction → film `kf` → sample-frame `Q` and reports
  validity and elastic residual. Passing `False` skips derivative work and returns a zero Jacobian
  when only ray validity or `Q` is needed.
- `evaluate_detector_coordinates(column_px, row_px, rods, branch)` returns the almost-everywhere
  `raw_detector_coordinate_density_A2_per_px2.v1`, per-rod contributions, inverse-branch counts, and
  caustic flags.
- `map_latent(...)` is the independent forward route used for proof and diagnostics.
- `map_latent_geometry(...)` is the same canonical forward geometry, exit refraction, and detector
  intersection without mosaic, structure strength, attenuation, or intensity work.
- `integrate_native_pixels(...)` returns deterministic `raw_detector_pixel_mass_A2.v1` with
  convergence, work-count, validity, and per-rod evidence.

### `SourceAveragedDetectorEwaldMeasure`

Owns an ordered tuple of complete incident-state measures. Its arbitrary-coordinate evaluator sums
independent source intensities over all retained roots. Detector-visible `m=0` is admitted only when
every contributing top-exit state proves the positive direct-root gap. Its quantitative native-pixel
integrator is deliberately branch-specific and rejects any model containing `m=0`; the configured
all-root macrobin path is an explicitly nonquantitative display preview. Source state order and
weights are preserved; wavelength-dependent evaluators are never collapsed geometrically.

### Configured simulation

- `load_simulation_config(path, repository_root=...)` accepts one strict
  `rasim-simulation-v2` YAML document. Unknown, duplicate, aliased, or missing fields fail.
- `build_configured_simulation_inputs(config)` creates source rows, canonical incident states,
  material, rods, finite-2H strength, and Bragg space once.
- `sample_configured_source(source, sample_count=...)` is the one mapping from validated configured
  source parameters to the canonical source sampler, including the exact one-row nominal state.
- `build_source_averaged_detector(inputs)` builds the all-state detector model.
- `evaluate_nominal_integer_l_markers(context)` solves exact integer-L intersections on the
  peak-mosaic (`alpha=0`) manifold, applies the canonical nominal-state exit/refraction and active
  detector visibility path, and retains every physical rod before grouping only coincident display
  labels. The seam-safe `root_sign = sign(sin(beta - phase))` distinguishes the analytic
  `phase-delta` and `phase+delta` sites even when `(m,L,branch)` labels coincide. These are
  `peak_mosaic_alpha0_integer_L_center.v2` nominal-source references, not source-averaged raster
  maxima.
- `sample_reciprocal_space`, `evaluate_nominal_ewald_surface`, and
  `integrate_detector_macrobins` generate optional display data without becoming model authority.

## Exact integer-L geometry fitting

`IntegerLGeometryModel` predicts arbitrary frozen non-specular integer-L sites without evaluating
their intensities. `IntegerLMarkerKey` fixes `(m, L, Ewald branch, beta-root sign, representative
physical rod)` throughout the objective; missing, tangent, or branch-changing roots fail instead of
being reassigned. `IntegerLMarkerObservations` stores native `(column_px,row_px)` coordinates and one
positive-definite 2-by-2 covariance per site.

`fit_integer_l_marker_geometry(...)` minimizes covariance-whitened native-coordinate residuals with
bounded TRF least squares. Its accepted single-5-degree parameterization applies active intrinsic
local-x then current-local-y rotations relative to the frozen base poses:

```text
R_detector = R_detector,0 Rx(delta_detector,column) Ry(delta_detector,row)
R_sample   = R_sample,0   Rx(delta_sample-normal,x) Ry(delta_sample-normal,y)
```

The detector limits use RA-SIM's direct `(+/-10 deg, +/-10 deg)` pitch/yaw hard shells. The two
effective sample-normal components conservatively use its `+/-5 deg` angular-correction shell;
RA-SIM has no separately bounded second reduced-normal coordinate. These are not separately
recovered sample-mount and goniometer-axis errors. A preflight bound-scaled Jacobian must have rank
four and acceptable condition. `audit_integer_l_marker_selection(...)` independently re-enumerates
the full visible non-specular catalog after fitting and reports `SAME`, `MISSING`, `CHANGED`, or
`AMBIGUOUS` without reassignment.

One image at one commanded goniometer angle cannot distinguish raw sample-zero tilt, goniometer-axis
yaw/pitch, crystal rotation about the rod axis, detector tangent translation/beam-center shifts, or
sample-normal offset/detector-distance gauges. Those parameters require additional constrained data,
especially multiple commanded goniometer angles; they are not exposed by this fit contract.

## Once-only factor ownership

| Factor | Owner |
|---|---|
| empirical source mass | source sampler |
| sample-footprint acceptance and entrance amplitude | incident transport |
| per-rod population | `MosaicBraggSpace` |
| CIF/finite-stack structure strength | ordered strength model |
| wrapped mosaic probability | `MosaicBraggSpace` |
| Ewald restriction / inverse-map determinant | Ewald or detector pushforward, by declared route |
| exit amplitude and uniform-depth attenuation | detector optical mapping |
| phase and polarization weights | detector measure construction |
| source-state sum | source-averaged detector measure |
| detector box integration | pixel integrator |

The two Ewald routes are equivalent proof views and are never multiplied together.

## Removed API

Contract version 9 intentionally has no compatibility shims for sampled mosaic orientation batches,
candidate selection, scattering-event batches, outgoing-wave batches, detector-hit batches, event
transport, point deposition, discrete Ewald painters, sphere textures, or raster grids. Callers must
use the continuous reciprocal and detector contracts above.
