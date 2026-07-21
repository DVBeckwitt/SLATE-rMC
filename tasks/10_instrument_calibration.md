# T10: source and detector calibration

Status: `NEEDS_REPLAN`.

This is a historical design note, not an executable task. Replan it against the contract-v9
continuous detector boundary before implementation.

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
- one named, immutable physical source reference plane in LAB, defined by a fixed point, the
  accepted nominal LAB beam-axis normal, and an ordered orthonormal in-plane basis, before any
  position-direction correlation is exposed; source position has only two in-plane coordinates
  and no fitted longitudinal origin coordinate
- express each accepted position-direction correlation at that plane and prove the deterministic
  transport of its moments to another plane; do not add an unrestricted covariance unless data or
  a prior make every added coordinate identifiable
- one minimal beam-frame rotation relative to the fixed LAB beam frame; beam roll is fixed when both
  the spatial-width pair and divergence-width pair are isotropic
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
- The LAB beam frame is fixed and source-owned. No fit pack may expose a compensating common
  beam/sample pose transform.
- The source reference plane location is fixed for one fit context. A plane shift is a declared
  reparameterization with transformed transverse moments/correlations, not a longitudinal source
  degree of freedom.
- The minimal beam-frame rotation changes the source direction/basis relative to fixed LAB; it does
  not rotate or translate the physical LAB source reference plane or its ordered in-plane basis.
  Keep the fixed reference-plane basis distinct from the trial beam-tangent basis.
- Include the reference-plane declaration and accepted correlation parameters in canonical source
  provenance and its owner-computed revision.
- Keep source size and divergence identifiable through multiple distances or priors.
- Report parameter rank, conditioning, active bounds, and correlations.
- A detector calibration change creates a downstream detector revision. It invalidates detector
  projection, selection, deposition, and measurement products, but does not invalidate an already
  compiled detector-independent incident `ki` realization.

## Proof

- synthetic source recovery
- synthetic detector recovery with non-square pixels and tilted detector
- held-out distance/ring prediction
- direct-beam and calibrant coordinate invariants
- isotropic-width beam-roll deactivation and rejection of a redundant common beam/sample pose pack
- source-reference-plane shift equivalence, full-rank recovery of the accepted transverse
  position/direction correlations, and deterministic rejection of a longitudinal origin coordinate
- deterministic rejection of underdetermined or non-positive-semidefinite correlation declarations
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

State: NEEDS_REPLAN

## Handoff

Status:

Commit SHA:

Accepted instrument/source revisions:

Proof summary:

Identifiability:

Known limitations:
