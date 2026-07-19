# Shared contracts

Bootstrap owns contract API v7. Parallel physics branches treat these contracts as read-only.

## Coordinate and transform types

```python
@dataclass(frozen=True)
class OscRawIndex:
    row: int
    column: int

@dataclass(frozen=True)
class DetectorIndex:
    row: int
    column: int

@dataclass(frozen=True)
class DetectorCoordinate:
    column_px: float
    row_px: float

@dataclass(frozen=True)
class RigidTransform:
    rotation: Float64[3, 3]
    translation_m: Float64[3]
    source_frame: FrameId
    target_frame: FrameId
```

## Instrument configuration

```text
InstrumentConfiguration
    lab axis definitions
    beam origin and direction convention
    goniometer axes, order, and pivots
    commanded angles
    static misalignment rotations
    sample offsets and surface normal
    sample_support_model_id
    sample_width_m and sample_length_m only for finite_rectangle.v1
    sample_from_crystal transform (CRYSTAL -> SAMPLE)
    detector origin, row axis, column axis, normal, and pivot
    detector shape and row/column pitch
    beam center in detector-native continuous coordinates
```

User-facing parameters compile once into explicit transforms. Kernels never reconstruct rotation chains independently.
`CompiledInstrument` stores hashed `lab_from_sample` and internally derives its immutable exact
inverse `sample_from_lab`; callers cannot supply a second inverse authority, and batched incident
construction never inverts or transforms the same direction a second time.

## Source samples

```text
PolarizationState
    model_id
    basis_u_lab[3] float64
    basis_v_lab[3] float64
    stokes[4] float64
    provenance

IncidentSampleBatch
    incident_sample_id[N] int64
    origin_lab_m[N,3] float64
    direction_lab[N,3] float64
    wavelength_A[N] float64
    source_weight[N] float64
    polarization_state_id[N]
    source_sampling_model_id
    source_rng_model_id
    source_seed unsigned 64-bit integer
    source_parameter_provenance canonical text
    source_parameter_revision SHA-256
    source_revision SHA-256
```

A Cartesian product is allowed only when source variables are declared independent.
In both source and incident-state batches, `source_weight` is exactly uniform empirical mass `1/N` and sums to one; it is never the sampled PDF. An incident-state batch covers one source-ray batch for one phase/parent.
The Gaussian source declaration consists of a mean LAB origin in metres, a unit mean LAB direction,
two orthonormal LAB transverse axes, two spatial sigmas in metres, two divergence sigmas in radians,
mean wavelength and wavelength sigma in angstroms, sample count, and polarization-state ID. The
five independent random coordinates are two transverse positions, two tangent-plane divergences,
and wavelength. `independent_gaussian_antithetic_lhs.v2` uses `numpy_pcg64.v1`: every `N`-stratum
is occupied exactly once in every dimension, adjacent rows are exact antithetic complements, and
an odd final row is exactly `0.5`.
`IncidentSampleBatch` derives both revisions through its sole construction path. One parameter
hash covers the
canonical `frames`, `units`, and `values` text for only the declared
means/axes/sigmas/wavelength/count/polarization inputs. Sampling-model ID, RNG-model ID, and seed are
separate and are not duplicated in that parameter text. The realization revision covers those
three fields, the parameter revision,
IDs, rays, wavelengths, weights, and polarization IDs. A merely
well-shaped hexadecimal string cannot be supplied in place of either derived revision. Canonical revision fields are
length-prefixed and typed; numeric arrays use fixed little-endian dtype/rank/shape/C-order bytes.

## Material optics

```text
MaterialOptics
    material_id
    wavelength_A[M] float64
    n_complex[M] complex128
    delta[M] float64
    beta[M] float64
    mu_Ainv[M] float64
    provenance
```

The ordered/materials branch produces this contract. `wavelength_A` is nonempty, finite, positive,
strictly increasing, and exact. The producer applies one sorted exact `unique` operation before
optical evaluation, so repeated requested wavelengths share one authoritative material row.
Manually constructed duplicate or unsorted grids fail. Geometry consumes the contract without
parsing CIF files, interpolation, or tolerance-based matching.

## Sample support

The compiled instrument carries exactly one `sample_support_model_id`. `finite_rectangle.v1`
requires finite positive width and length and retains the closed-edge footprint test.
`unbounded_plane.v1` requires both dimensions to be absent and accepts every unique forward plane
intersection with footprint acceptance one. Zero or placeholder dimensions are invalid.
Every ray with `abs(direction_sample_z) <= 1e-14`, including a coplanar ray, is `PARALLEL` because
the plane intersection is not unique. Such a row has no intersection, footprint mass, or optical
payload.

## Incident states

```text
IncidentStateBatch
    incident_state_id[N] int64
    incident_sample_id[N] int64
    sample_intersection_lab_m[N,3] float64
    direction_sample[N,3] float64
    k_air_sample_Ainv[N,3] float64
    k_film_phase_sample_Ainv[N,3] float64
    kz_film_Ainv[N] complex128
    entrance_amplitude[N] complex128
    footprint_acceptance[N] float64
    source_weight[N] float64
    wavelength_A[N] float64
    polarization_state_id[N]
    status[N] tuple[ValidityCode, ...]
    valid[N] bool
    source_sampling_model_id
    source_rng_model_id
    source_seed unsigned 64-bit integer
    source_parameter_provenance canonical text
    source_parameter_revision SHA-256
    source_revision SHA-256
    sample_geometry_revision SHA-256
    material_revision SHA-256
    incident_model_id
```

For `one_transmitted_channel.v1`, `incident_state_id` and `incident_sample_id` form a one-to-one
identity relation and `valid` equals `status == VALID` exactly. Geometry failures retain source ID,
wavelength, empirical mass, polarization, and every revision but expose zero intersection,
direction, air-side, film-side, amplitude, and footprint payload. Optical failures retain their
accepted intersection, SAMPLE direction, air-side `ki`, footprint, source fields, and revisions,
but expose zero film-side `ki`, complex normal mode, and entrance amplitude. Valid rows enforce
`k_air=(2*pi/wavelength)*direction`, tangential conservation, and equality between the phase-vector
normal component and `real(kz_film)`.

The sample-geometry revision covers only the SAMPLE-to-LAB entrance pose, explicit support model,
finite dimensions when applicable, and intersection model. The material revision covers the exact
canonical optical grid and its provenance. Detector calibration, `sample_from_crystal`, and film
thickness are deliberately excluded. Revisions are computed once for the complete canonical batch;
worker row views inherit parent revisions and canonical row indices and never rehash a slice.

## Rod catalog

```text
RodCatalog
    rod_id[R] int64
    phase_id[R]
    h[R] int32
    k[R] int32
    family_id[R]
    family_key[R]
    qr_Ainv[R] float64
    reciprocal_basis_Ainv[3,3]
    symmetry_metadata[R]
```

Every physical `(h,k)` rod remains separate.

## Pre-selection scattering candidates

```text
ScatteringEventBatch
    event_id[E] int64
    incident_state_id[E] int64
    orientation_id[E] int64
    rod_id[E] int64
    wavelength_A[E] float64
    q_internal_sample_Ainv[E,3] float64
    q_sample_normal_Ainv[E] float64
    l_coordinate[E] float64
    kf_film_phase_sample_Ainv[E,3] float64
    reciprocal_weight[E] float64
    ewald_residual_Ainv[E] float64
    status[E] tuple[ValidityCode, ...]
    valid[E] bool
```

Rows are deterministic/adaptive valid-support candidates. For one incident ray and independent phase/parent, T07 forms one pool across every individual rod and valid mosaic/`Q` solution; each candidate retains its own rod, orientation, `Q`, `kf`, hit, scattering strength, mosaic mass, and other once-only factors. There is no per-reflection normalization or `Qr` collapse, and two-pass/streaming enumeration is preferred.
`reciprocal_weight` is candidate mosaic/Jacobian mass used only in that complete-pool candidate mass, never as a post-selection multiplier. It excludes source, population, scattering strength, optics, solid angle, and deposition.
`orientation_id` is a repeatable foreign key, not an array index. `event_id` maps uniquely to it.
`q_sample_normal_Ainv` equals `q_internal_sample_Ainv[:,2]`, and `valid` is true exactly where `status` is `VALID`; failures retain their exact status.

## Event-aligned rod query

```text
RodQueryBatch
    event_id[E] int64
    rod_id[E] int64
    phase_id[E]
    h[E] int32
    k[E] int32
    q_sample_normal_Ainv[E] float64
    l_coordinate[E] float64
    wavelength_A[E] float64
```

Grid evaluation and interpolation may be internal optimizations. The integration contract is event-aligned.

## Model outputs

```text
LayerAmplitudeResult
    event_id[E] int64
    rod_id[E] int64
    phase_id[E]
    f_plus_e[E] complex128
    f_minus_e[E] complex128 or absent
    normalization = ONE_REGISTRY_FREE_LAYER
    phase_sign = POSITIVE_Q_DOT_R
    gauge_id = pbi2.pb_centered.v1
    layer_normal_crystal[3] float64 unit vector
    layer_repeat_A positive float64

LayerNormalQBatch
    event_id[E] int64
    rod_id[E] int64
    phase_id[E]
    layer_normal_q_Ainv[E] float64
    gauge_id

EventIntensityResult
    event_id[E] int64
    scattering_strength_A2[E] float64
    model_id
    model_component_id
    population_group_id or absent
    normalization = UNIT_CELL | FINITE_TOTAL | FINITE_PER_LAYER

PopulationWeightTable
    population_group_id
    model_component_id
    weight
    semantics = incoherent
    provenance
```

Ordered and stacking models implement the same event-aligned scattering-strength contract.
`LayerNormalQBatch` is produced by future T07 from full event `Q`, `orientation_id`, and T04 layer metadata; T05 requires exact event/rod/phase/gauge alignment and uses `exp(+i Q·R)` with no sample-`Qz` fallback.
`scattering_strength_A2` is unweighted, polarization-neutral `r_e²` times raw electron² in `angstrom²`; it excludes population, optics, polarization, solid angle, and deposition. T04 and T05 use the single core conversion helper exactly once; T07 does not apply `r_e²`.
`PopulationWeightTable` remains a declared incoherent-intensity contract but is deferred to reviewed T07 preparation; T02--T05 do not implement or apply it.

## Outgoing transport and detector hits

```text
OutgoingWaveBatch
    event_id[E] int64
    kf_air_lab_Ainv[E,3] float64
    exit_amplitude[E] complex128
    attenuation_weight[E] float64
    optical_weight[E] float64
    valid[E] bool

DetectorHitBatch
    event_id[H] int64
    column_px[H] float64
    row_px[H] float64
    pixel_solid_angle_sr[H] float64
    valid[H] bool
```

`pixel_solid_angle_sr` is immutable geometry metadata for optional later analysis; it is never an input to raw rendering.

## Pixel contributions

```text
PixelContributionBatch
    event_id[C] int64
    flat_pixel_index[C] int64
    deposition_weight[C] float64
```

For each valid event, deposition weights sum to one unless support falls outside the detector and the declared clipping policy says otherwise.

## Selection identities, post-integration

```text
RadialFamilyKey
    phase_id
    reciprocal_cell_revision
    exact_inplane_key
    qr_Ainv
    rod_ids

ReflectionGroupKey
    radial_family_key
    discrete_out_of_plane_key
    member rod_ids

BranchKey
    reflection_group_key or radial_family_key
    branch_id: 0, 1, or None
    azimuth_frame
    basis_revision
    sign_mapping
    deadband_rad
    rule_version

SelectionManifest
    data_revision
    source_revision
    instrument_revision
    rod_catalog_revision
    event_model_revision
    selected radial families and reflection groups
    selected branches
    measured observations and candidate evidence
    detector-native ROIs
    ambiguity status
    provenance and hash
```

The branch is defined from signed physical reciprocal azimuth in the declared sample/crystal in-plane basis. Detector pixels are used for measured association, not as the branch definition. Associations are immutable during one fit and may change only through a new manifest revision between runs.

## Fit observations, post-integration

```text
DirectBeamObservation
    dataset_id
    detector_distance_or_pose
    detector-native image or centroid/width summary
    covariance or variance

CalibrantObservation
    dataset_id
    calibrant identity and d-spacing provenance
    detector-native ring/peak support
    covariance or variance

PeakObservation
    observation_id
    dataset_id
    rod_id or reflection_group_id
    branch_id
    measured_column_px
    measured_row_px
    covariance_px2[2,2]

ProfileObservation
    observation_id
    dataset_id
    selection_id
    detector-native support
    measured signal
    normalization or variance

IntegratedIntensityObservation
    observation_id
    dataset_id
    selection_id
    measured_mass
    variance or likelihood metadata
```

## Fit contracts, post-integration

```text
ParameterSpec
    name
    value
    unit
    lower and upper bounds
    internal transform
    active flag
    dependency stage

DataCorrectionLedger
    dark subtraction status
    flat-field status
    polarization: model, data-corrected, or declared unity approximation
    solid angle: optional later caking/analysis metadata, never raw-render input
    detector efficiency status
    exposure/flux normalization status
    background policy
    provenance

FitDataset
    detector-native data
    mask
    noise/variance model
    exposure metadata
    preprocessing revision
    correction ledger
    source, instrument, and selection revisions

CompiledFitContext
    immutable forward states
    selected observations and support
    seeded sample revision
    invalidation graph

FitResult
    parameters and units
    objective definition and value
    convergence and invalid evaluations
    uncertainty and identifiability evidence
    held-out results
    provenance and revisions
```

## Trace record

```text
TraceRecord
    case_id
    stage_id
    value
    shape
    dtype
    unit
    frame
    measure
    model_version
    provenance
```

The proof comparator reports the first failing `stage_id`.
