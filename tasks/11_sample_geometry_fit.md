# T11: sample and goniometer geometry fit

Status: FUTURE. Begin after T08, T09, and accepted T10 calibration.

Branch: `feat/sample-geometry-fit`

## Goal

Fit sample pose, offsets, crystal orientation, and selected goniometer corrections from frozen detector-native peak associations.

## Owned paths

```text
src/rasim_next/fitting/geometry.py
tests/test_fitting.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/fitting/geometry_fit_parameters.py
    ra_sim/fitting/caked_geometry_objective.py
    ra_sim/fitting/caked_geometry_solver.py
    ra_sim/headless_geometry_fit.py
    ra_sim/gui/geometry_q_group_manager.py

manuscript
    sections/refinement_workflow.tex:29-33,53-55
    2D_Supplemental/SI_failure_modes.tex:109-134,226-234
```

## Required work

- consume one immutable `SelectionManifest`
- use measured continuous detector centroids and covariance
- select an explicit active parameter subset
- evaluate only the forward stages invalidated by geometry
- report invalid topology without reassignment
- support an outer selection audit after convergence
- create a new selection revision and rerun only when the audit changes identity
- preserve source and detector calibration unless a separately declared controlled joint polish is requested

## Proof

- synthetic recovery with nonzero detector, sample, and goniometer rotations
- held-out peak prediction
- multi-start consistency
- Jacobian rank, conditioning, bounds, and correlations
- stable selection after the outer audit
- original-RASIM selected-case comparison under identical associations
- first divergence for corrected rigid-transform behavior

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/fitting/geometry.py tests/test_fitting.py
pytest -q tests/test_fitting.py
python -m rasim_next.proof sample-geometry-fit --json
git diff --check
```

## Stop conditions

Stop `BLOCKED` if associations are ambiguous, the active geometry is unidentifiable, or the forward model cannot expose predicted continuous peak coordinates without rendering a full image.

## Execution plan

State: FUTURE

## Handoff

Status:

Commit SHA:

Accepted geometry revision:

Selection revisions used:

Proof summary:

Conditioning and held-out error:
