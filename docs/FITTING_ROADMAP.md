# Post-integration selection and staged fitting roadmap

The forward model is not complete merely because it renders an image. The completed project must recover geometry, mosaicity, ordered intensities, and stacking disorder without allowing one stage to compensate for an error in another.

The governing manuscript source is `sections/refinement_workflow.tex:4-59`. Original-RASIM reference locations include:

```text
source branch identity
    ra_sim/utils/calculations.py:48-62

Qr and (m,L) family metadata
    ra_sim/gui/geometry_q_group_manager.py:1197-1329

geometry parameters and bounds
    ra_sim/fitting/geometry_fit_parameters.py

branch-locked geometry objective
    ra_sim/fitting/caked_geometry_objective.py
    ra_sim/fitting/caked_geometry_solver.py

mosaic profile preparation and objective
    ra_sim/fitting/optimization_mosaic_profiles.py

ordered detector-space objective
    ra_sim/gui/ordered_structure_fit.py:53-99,322-540

selected rod profile accumulation
    ra_sim/fitting/rod_profiles.py:91-308

manuscript detector-derived ordered and disorder observables
    2D_Supplemental/SI_failure_modes.tex:691-748
```

These sources define behavior to characterize. They do not dictate the new architecture.

## Prerequisites

Selection and fitting begin only after T07 proves the integrated native-detector forward model. The accepted forward state must expose immutable revisions for the instrument, source samples, material optics, rod catalog, event geometry, detector hits, and detector response.

The initial inverse workflow remains in detector-native coordinates. `2theta/phi`, caking, and reciprocal-space remapping are later measurement transforms, not prerequisites for geometry, mosaic, or ordered-intensity fitting.

## Selection and indexing gate

The selection subsystem is implemented after the four forward parts dovetail. It does not own scattering equations and does not optimize parameters.

### Rod identity

Every individual `(h,k)` rod remains distinct and has an immutable `rod_id`. A radial family records:

```text
phase_id
reciprocal_cell_revision
exact in-plane family key
family_id
Qr for reporting and tolerance checks
member rod_ids
symmetry provenance
```

For a hexagonal cell, the exact family key contains

\[
m=h^2+hk+k^2.
\]

Floating `Qr` alone is not an identity. For a general cell, the exact family key is derived from the declared in-plane reciprocal metric and integer indices. An ordered reflection group additionally records its `L` or other discrete out-of-plane index. Rods sharing `Qr` are not collapsed before their amplitudes, event geometry, and detector projection are evaluated.

### Branch identity

The canonical branch is defined from the sign of a wrapped physical in-plane reciprocal azimuth in the declared sample or crystal basis. It is not defined from raw OSC row or column sign.

```text
branch_id = 0 or 1 for the two signed non-specular branches
branch_id = None with COLLAPSED_00L for applicable specular-family cases
```

The axis basis, angle wrapping, sign-to-label mapping, and deadband are versioned. Original RASIM's signed-phi behavior is characterized and reproduced where it is valid. Detector-native side labels are derived from projected events only for display and association.

### Measured association

A selection run associates measured detector-native peaks or ROIs with predicted rod and branch candidates. It records:

- measured observation and uncertainty
- candidate rod, family, reflection group, and branch IDs
- predicted detector-native coordinates
- residual and gating evidence
- ambiguity status
- instrument, source, rod-catalog, and rule revisions
- an immutable selection-manifest hash

Ambiguous observations are rejected or explicitly marked unused. An optimizer never changes identity.

### Re-indexing rule

A geometry change may invalidate an association. Re-indexing therefore occurs only between optimization runs:

1. create a selection manifest from the current accepted geometry
2. fit with that manifest frozen
3. reproject and audit every association
4. when an identity changes or becomes ambiguous, create a new manifest revision and rerun the fit
5. stop when both geometry and associations are stable

This is an outer discrete loop, not a hidden operation inside the objective.

## Fit foundation

The common fitting layer owns only data, parameters, invalidation, objectives, optimizers, and result provenance. It does not reimplement forward physics.

Required contracts include:

```text
ParameterSpec
    name, value, unit, bounds, transform, active flag, dependency stage

FitDataset
    detector-native data, mask, variance/noise model, exposure metadata,
    preprocessing revision, correction ledger, instrument revision,
    selection-manifest revision

CompiledFitContext
    immutable upstream states, active observation support,
    deterministic samples, explicit invalidation graph

FitResult
    values, uncertainty evidence, objective definition, convergence,
    active bounds, invalid evaluations, identifiability, held-out results,
    code/data/configuration provenance
```

Rules:

- Fixed deterministic source, wavelength, mosaic, and reciprocal nodes are used throughout one optimization.
- Data masks and background treatment are fixed before one objective run.
- A correction ledger states whether dark, flat field, polarization, solid angle, detector efficiency, and exposure normalization are applied to data, applied by the model, or intentionally omitted. No correction occurs on both sides.
- Raw Poisson counts, dark-subtracted data, and variance-weighted continuous data use explicitly different likelihoods or residuals.
- A robust loss is used only under a declared outlier model.
- Linear nuisance scales and simple backgrounds are eliminated analytically where this is exact and numerically stable.
- Parameter scaling, transforms, bounds, units, and active sets are explicit.
- Synthetic recovery, multi-start behavior, conditioning, parameter correlation, held-out prediction, and failure modes are recorded.
- Changing a parameter invalidates only the compiled stages that depend on it.

## Geometry and calibration phase

The manuscript separates incident phase space, detector calibration, and sample alignment because they affect similar detector observables. The new plan retains that separation.

### Stage G0: source and wavelength characterization

Goal: determine the incident phase-space model before mosaicity is fitted.

Primary data:

- direct-beam images at multiple detector distances
- independent wavelength or bandwidth information when available

Candidate parameters:

- beam widths
- angular divergence and correlations
- wavelength distribution and declared correlations with direction or position
- direct-beam normalization nuisance terms

Proof:

- synthetic recovery
- prediction at held-out detector distances
- normalization of the joint source distribution
- identifiability between source size and divergence

The result freezes `CompiledSourceSamples` for downstream geometry and mosaic stages.

### Stage G1: detector geometry calibration

Goal: determine detector mapping independently of sample structure where possible.

Preferred data:

- powder calibrant rings with known spacings
- direct-beam center observations
- multiple detector distances or poses when available

Candidate parameters:

- detector distance
- detector yaw, pitch, and roll or equivalent rigid pose
- detector origin and beam center
- row and column pitch only when not independently calibrated
- static detector pivot or offset terms that are identifiable

The objective is evaluated in detector-native coordinates or directly against calibrated ring geometry. A sample-crystal model is not used to compensate for detector error.

Proof:

- synthetic parameter recovery
- ring or peak residuals in continuous pixels
- held-out rings or distances
- transform and pixel-coordinate invariants
- Jacobian rank, condition, and parameter correlations

When no powder calibrant exists, a sample-based joint detector/alignment stage is an explicit fallback with stronger priors, more datasets, and a report of the lost identifiability. It is not silently treated as equivalent to independent calibration.

### Stage G2: sample and goniometer alignment

Goal: fit sample pose and goniometer corrections after source and detector calibration.

Inputs:

- accepted source and detector results
- frozen peak-to-rod/reflection-group/branch associations
- measured continuous detector centroids and covariance
- fixed lattice, wavelength, and material model
- a narrow declared profile state used only for peak-center prediction

Candidate parameters:

- sample incidence and in-plane orientation
- sample and beam offsets
- crystal-to-sample orientation
- goniometer axis misalignments and pivots
- only explicitly selected detector parameters when a controlled joint polish is justified

For observation `i`, a suitable detector-native residual is

\[
\mathbf r_i=\mathbf L_i^{-1}
\left(\mathbf x_i^{\mathrm{pred}}-\mathbf x_i^{\mathrm{meas}}\right),
\]

where `L_i L_i^T` is the measured centroid covariance.

Rules:

- Associations and branch IDs are frozen during one run.
- Invalid topology is reported, not reassigned.
- Intensity cannot repair peak placement.
- Geometry changes invalidate incident states, reciprocal events, detector hits, and detector response.
- The accepted result must pass the re-indexing audit above.

Proof:

- synthetic recovery across nonzero detector, sample, and goniometer rotations
- held-out peak prediction
- multi-start consistency
- conditioning and active-bound report
- old-RASIM selected-case comparison using identical associations
- first-divergence evidence where corrected transforms change the recovered parameters

## Mosaic phase

### Stage M: mosaic distribution fit

Goal: fit the shared orientation distribution with source and geometry fixed.

Inputs:

- accepted source, detector, and sample geometry
- frozen rod/reflection-group/branch associations
- detector-native local ROIs
- fixed wavelength and beam-divergence model unless a narrowly declared joint stage is separately justified

Observables:

- center-aligned off-specular profile shapes
- local arc widths and tail mass
- specular-family tail checks

Use normalized shape residuals or an exact analytic per-ROI nuisance scale so structural intensity cannot masquerade as mosaic width. Keep Gaussian core width, Lorentzian tail width, and mixture weight as distinct parameters.

Rules:

- Geometry and selection are immutable.
- ROI support and background policy are frozen.
- The same deterministic quadrature is used at every evaluation.
- Quadrature density changes only between optimization stages.
- Report identifiability between beam divergence, bandwidth, Gaussian width, Lorentzian width, and mixture weight.

Proof:

- synthetic recovery for narrow, mixed, and tail-dominated cases
- convergence under quadrature refinement
- held-out rods and specular-tail validation
- comparison with original-RASIM selected profiles under `MATCH` or `CORRECTED`

## Intensity phase

### Stage I1: ordered intensity fit

Goal: fit relative ordered Bragg intensities after source, geometry, and mosaicity are fixed.

Selection:

- detector-native ROIs are tied to immutable rod-family, reflection-group, and branch identities
- individual rods within one `Qr` family remain separate until detector projection and ROI integration
- the exact rods contributing to each observation are recorded
- ROI support, mask, and local background policy are frozen before optimization

Observable:

For observation `i` in image `m`, compare background-subtracted detector mass in the same measured and simulated ROI. Use one analytic nonnegative image scale where appropriate. Structural parameters remain global across images.

Candidate parameters may include:

- constrained atomic coordinates
- occupancies
- isotropic or anisotropic displacement parameters supported by the data
- finite ordered-stack parameters
- declared incoherent phase populations
- explicit per-image scale and permitted local background nuisance terms

Rules:

- Source, geometry, mosaic, event geometry, detector hits, and detector response are reused.
- No maximum normalization, rounding, reflection pruning, fabricated reflections, or independently fitted peak amplitudes.
- Scale and background nuisance terms remain distinct from structural parameters.
- Physical amplitudes and relative intensities retain their declared normalization.

Proof:

- synthetic recovery
- ROI mass conservation
- held-out reflection ratios
- parameter-correlation and identifiability evidence
- original-RASIM raw-intensity comparison before its normalization and rounding

### Stage I2: stacking-disorder intensity fit

Goal: fit transition-matrix disorder after the ordered baseline is fixed.

Selection:

- choose one or more fixed `Qr` families
- choose an explicit branch for the primary fit
- retain the symmetry-related branch for validation when available
- keep selection independent of caking

Initial observations may be detector-native selected regions or event-aligned `Qz` samples generated from the accepted forward model. A later caking layer may form selected fixed-`Qr` profiles, but it must reuse the same rod, reflection-group, and branch identities.

Rules:

- Source, geometry, mosaic, lattice, motif amplitudes, material optics, and ordered baseline are fixed.
- Peak amplitudes are not independently fit before applying the transition model.
- Parent populations combine under declared incoherent semantics.
- Event geometry and detector response are reused.

Proof:

- synthetic recovery for deterministic and faulted stacks
- direct-enumeration validation for short stacks
- held-out branch agreement
- selected-region or profile mass conservation

## Later caking and `2theta/phi`

Caking is a separate measurement transformation, not part of geometry, rod identity, or branch identity. When added:

- consume the canonical detector-native image and accepted instrument revision
- build signal and normalization fields
- sum signal and normalization over a selected rod mask before division
- reuse the frozen selection manifest
- prove non-square coordinate mapping, support, and mass conservation
- never alter an upstream fit result silently

## Mandatory staged order

```text
integrated forward model
    -> selection subsystem and fit foundation
    -> source characterization
    -> detector calibration
    -> initial rod/branch association
    -> sample/goniometer alignment
    -> association revalidation
    -> mosaic fit
    -> ordered intensity fit
    -> stacking intensity fit
```

A later joint polish is optional only after the staged solution is stable. It uses strong bounds or priors, reports parameter compensation, preserves identity semantics, and must not conceal a forward-model error.


## Self-contained fit examples

The first fit implementation uses the tracked Bi2Se3 OSC files and canonical peak CSV. Geometry fitting
operates in detector-native `(column_px, row_px)` and keeps an immutable association manifest during
each optimizer run. Mosaic fitting freezes source and geometry. Ordered-intensity fitting freezes
geometry and mosaic. The PbI2 example supplies the later stacking-disorder recovery case.

Synthetic truth generated by the accepted integrated forward model must be committed only when the
corresponding fit task is implemented. A fit passes only when it recovers held-out synthetic truth, not
merely when it lowers an image residual.
