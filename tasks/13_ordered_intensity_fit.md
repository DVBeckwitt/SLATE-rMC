# T13: ordered intensity fit

Status: FUTURE. Begin after accepted T12 mosaic.

Branch: `feat/ordered-intensity-fit`

## Goal

Fit relative ordered Bragg intensities from immutable detector-native ROI selections while reusing event geometry and detector response.

## Owned paths

```text
src/rasim_next/fitting/ordered_intensity.py
tests/test_fitting.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/gui/ordered_structure_fit.py:53-99,322-540
    ra_sim/fitting/rod_profiles.py

manuscript
    sections/refinement_workflow.tex:39-43,57
    2D_Supplemental/SI_failure_modes.tex:691-703
```

## Required work

- freeze source, detector, sample, mosaic, selection, ROI, mask, and background-policy revisions
- record every individual rod contributing to each ROI
- compare measured and simulated detector mass under one declared noise model
- support exact nonnegative per-image scale where applicable
- keep structural parameters global and nuisance scales/backgrounds separate
- reuse event geometry, continuous hits, and detector response
- fit raw amplitudes/relative intensities without maximum normalization, pruning, or independent peak amplitudes
- expose held-out reflection validation and parameter identifiability

## Proof

- synthetic structural recovery
- ROI mass conservation
- exact scale solution
- held-out reflection ratios
- parameter rank/correlation evidence
- original-RASIM raw-intensity comparison before normalization and rounding

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/fitting/ordered_intensity.py tests/test_fitting.py
pytest -q tests/test_fitting.py
python -m rasim_next.proof ordered-intensity-fit --json
git diff --check
```

## Execution plan

State: FUTURE

## Handoff

Status:

Commit SHA:

Accepted ordered-model revision:

Proof summary:

Held-out ratios:

Identifiability:
