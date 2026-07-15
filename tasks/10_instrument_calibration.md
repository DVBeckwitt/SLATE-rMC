# T10: source and detector calibration

Status: FUTURE. Begin after T09. It may be developed while T08 is implemented, but accepted calibration precedes final sample indexing.

Branch: `feat/instrument-calibration`

## Goal

Characterize incident phase space and detector geometry independently of sample structure where the data permit.

## Owned paths

```text
src/rasim_next/fitting/source.py
src/rasim_next/fitting/detector.py
tests/test_fitting.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/hbn_fitter/fitter.py:83-161,244-266
    ra_sim/hbn_geometry.py:1-32
    ra_sim/hbn.py
    ra_sim/gui/_runtime/runtime_session.py:662-684

manuscript
    sections/refinement_workflow.tex:19-28,53-55
    2D_Supplemental/SI_failure_modes.tex:109-134
```

## Required work

### Source stage

- direct-beam observations at several detector distances
- beam size, angular divergence, wavelength/bandwidth, and declared correlation parameters
- normalized source distribution and compiled deterministic source samples
- held-out-distance prediction

### Detector stage

- powder calibrant or other independent geometry observations
- detector distance, pose, origin, and beam center
- row and column pitch only when not independently calibrated
- detector-native residuals or exact ring-distance residuals
- explicit fallback policy for sample-based calibration when no independent calibrant exists

## Rules

- Do not use sample structure intensities to compensate detector error.
- Keep source size and divergence identifiable through multiple distances or priors.
- Report parameter rank, conditioning, active bounds, and correlations.
- A detector calibration change creates a new instrument revision and invalidates downstream selection.

## Proof

- synthetic source recovery
- synthetic detector recovery with non-square pixels and tilted detector
- held-out distance/ring prediction
- direct-beam and calibrant coordinate invariants
- multi-start consistency
- declared failure when the problem is underdetermined

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/fitting/source.py src/rasim_next/fitting/detector.py tests/test_fitting.py
pytest -q tests/test_fitting.py
python -m rasim_next.proof instrument-calibration --json
git diff --check
```

## Execution plan

State: FUTURE

## Handoff

Status:

Commit SHA:

Accepted instrument/source revisions:

Proof summary:

Identifiability:

Known limitations:
