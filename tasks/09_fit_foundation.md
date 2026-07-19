# T09: fit foundation

Status: FUTURE. Begin only after shared beam-to-`ki` Checkpoint K and the accepted deterministic
detector-native Phase 1 boundary.

Branch: `feat/fit-foundation`

## Goal

Build the smallest reusable fitting infrastructure without reimplementing forward physics or choosing one optimizer as a permanent architecture.

## Owned paths

```text
src/rasim_next/fitting/contracts.py
src/rasim_next/fitting/context.py
src/rasim_next/fitting/invalidation.py
src/rasim_next/fitting/objective.py
src/rasim_next/fitting/result.py
tests/test_fitting.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/fitting/geometry_fit_parameters.py
    ra_sim/fitting/caked_geometry_objective.py:62-138
    ra_sim/fitting/caked_geometry_solver.py
    ra_sim/gui/ordered_structure_fit.py:53-99,322-540

manuscript
    sections/refinement_workflow.tex:4-59
    2D_Supplemental/SI_failure_modes.tex:109-134,691-748
```

## Required work

- typed parameters, units, bounds, transforms, active/fixed status, and dependency stages
- an active-parameter-pack contract that records gauge ownership, excludes fixed or inactive null
  directions, and rejects redundant packs before objective evaluation
- detector-native datasets, masks, variance/noise model, exposure metadata, preprocessing revisions, and a data/model correction ledger
- immutable compiled fit context
- fit-context material coverage for every exact unique wavelength in the complete parent
  `IncidentSampleBatch`, including rows that are invalid under the baseline geometry but may become
  valid in a later trial; each value matches exactly one `MaterialOptics` wavelength row, with no
  tolerance match, interpolation, or geometry-filtered subset
- the immutable owner-provided `source_parameter_revision`, `source_revision`,
  `sample_geometry_revision`, and `material_revision` envelope plus `incident_model_id`; validate
  the context's declared envelope before objective evaluation, require rebuilt trial objects to
  carry owner-derived revisions consistent with the invalidation graph, and never recompute these
  scientific hashes in fitting code
- explicit invalidation graph over compiled forward states
- objective value, invalid-evaluation, convergence, and provenance records
- analytic elimination interface for exact linear nuisance scales/backgrounds
- synthetic objective harness independent of RASIM physics
- separately reviewed dependency request if an optimizer library is needed

## Proof

- parameter transform and bounds round trips
- deterministic repeated objective evaluation
- correct invalidation for geometry, mosaic, intensity, and nuisance changes
- exact analytic scale against a direct numerical minimization
- clear distinction among Poisson, dark-subtracted Gaussian, and variance-weighted data
- rejection of duplicate data/model polarization or solid-angle corrections
- result provenance and hash stability
- no forward equation in the fitting package
- full column rank for each accepted active pack and deterministic rejection of deliberately
  redundant common-pose, isotropic-roll, pivot-axis, zero-pose/mount, and inactive-support packs
- a baseline-invalid source row whose wavelength becomes geometry-valid later has material optics
  available without rebuilding the context
- deterministic preflight rejection when any exact parent wavelength is absent or when an
  owner-provided source/sample/material revision or incident model ID mismatches the compiled
  context; the forward objective is not entered
- geometry-only trials reuse material state, while detector-only trials reuse both material state
  and the detector-independent incident realization

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/fitting tests/test_fitting.py
pytest -q tests/test_fitting.py
python -m rasim_next.proof fit-foundation --json
git diff --check
```

## Execution plan

State: FUTURE

## Handoff

Status:

Commit SHA:

Public APIs:

Invalidation schema:

Proof summary:

Dependency requests:
