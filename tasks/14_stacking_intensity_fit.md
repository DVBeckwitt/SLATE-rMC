# T14: stacking-disorder intensity fit

Status: FUTURE. Begin after accepted T13 ordered baseline.

Branch: `feat/stacking-intensity-fit`

## Goal

Fit transition-matrix disorder from selected fixed-`Qr` families and explicit branches while all upstream states remain frozen.

## Owned paths

```text
src/rasim_next/fitting/stacking_intensity.py
tests/test_fitting.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/fitting/rod_profiles.py:91-308
    ra_sim/utils/calculations.py:48-62
    ra_sim/utils/stacking_fault.py
    ra_sim/utils/polytype_stacking.py

manuscript
    sections/refinement_workflow.tex:44-48,59
    2D_Supplemental/SI_failure_modes.tex:704-748
```

## Required work

- select one or more immutable radial families and one explicit primary branch
- retain the symmetry-related branch for validation when available
- consume detector-native selected regions or event-aligned `Qz` observations without requiring caking
- freeze source, geometry, mosaic, lattice, motif amplitudes, material optics, and ordered baseline
- reuse event geometry and detector response
- fit typed transition parameters and declared incoherent parent populations
- do not fit independent peak amplitudes before the stacking model

## Proof

- synthetic recovery for deterministic and faulted stacks
- direct-enumeration validation for short stacks
- held-out branch behavior
- selected-region/profile mass conservation
- parameter bounds and identifiability

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/fitting/stacking_intensity.py tests/test_fitting.py
pytest -q tests/test_fitting.py
python -m rasim_next.proof stacking-intensity-fit --json
git diff --check
```

## Execution plan

State: FUTURE

## Handoff

Status:

Commit SHA:

Accepted stacking-fit revision:

Proof summary:

Held-out branch result:

Identifiability:
