# T09: fit foundation

Status: FUTURE. Begin only after T07 passes.

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
- detector-native datasets, masks, variance/noise model, exposure metadata, preprocessing revisions, and a data/model correction ledger
- immutable compiled fit context
- explicit invalidation graph over compiled forward states
- deterministic sample revisions
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
