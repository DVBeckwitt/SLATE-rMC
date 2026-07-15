# T12: mosaic fit

Status: FUTURE. Begin after accepted T11 geometry and stable selection.

Branch: `feat/mosaic-fit`

## Goal

Fit the Gaussian core, Lorentzian tail, and mixture of the orientation distribution from fixed detector-native profile observations without allowing intensity or geometry to compensate.

## Owned paths

```text
src/rasim_next/fitting/mosaic.py
tests/test_fitting.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/fitting/optimization_mosaic_profiles.py
    ra_sim/fitting/optimization.py
    ra_sim/simulation/mosaic_profiles.py

manuscript
    sections/refinement_workflow.tex:34-38,55-57
    sections/mosaicity_texture.tex
    2D_Supplemental/SI_failure_modes.tex:109-134,241-263
```

## Required work

- consume accepted source, detector, sample, and selection revisions
- compile fixed local detector ROIs and profile axes
- use normalized profile shapes or exact analytic nuisance scales
- use fixed deterministic quadrature throughout one run
- change quadrature density only between convergence stages
- keep beam divergence, bandwidth, Gaussian width, Lorentzian width, and mixture semantics distinct
- provide held-out rod and specular-tail validation

## Proof

- synthetic recovery for narrow, mixed, and tail-dominated cases
- quadrature convergence at the final observable
- geometry immutability
- identifiability/correlation report
- held-out profile prediction
- original-RASIM selected-profile `MATCH` or `CORRECTED` evidence

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/fitting/mosaic.py tests/test_fitting.py
pytest -q tests/test_fitting.py
python -m rasim_next.proof mosaic-fit --json
git diff --check
```

## Execution plan

State: FUTURE

## Handoff

Status:

Commit SHA:

Accepted mosaic revision:

Proof summary:

Identifiability:

Held-out performance:
