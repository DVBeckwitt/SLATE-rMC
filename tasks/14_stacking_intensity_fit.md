# T14: stacking-disorder intensity fit

Status: `COMPLETE` under the direct-profile scope revision.

The historical requirements below are superseded by the direct-profile execution plan recorded in
this file; references to detector response do not authorize restoring deleted APIs.

Branch: `codex/minimal-stacking-population-fit`

## Goal

Fit transition-matrix disorder from selected fixed-`Qr` families and explicit branches while all upstream states remain frozen.

## Owned paths

```text
src/rasim_next/fitting/stacking_intensity.py
src/rasim_next/fitting/__init__.py
scripts/recover_pbi2_stacking_population.py
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
- reuse analytic root geometry and continuous detector-coordinate response
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

State: COMPLETE under the user's direct-profile scope revision.

1. Compile the five fixed PbI2 parent strengths at explicit signed `(h,k)`, continuous `L`, and
   wavelength queries by reusing `finite_population_event_intensity`.
2. Fit one nonnegative five-component amount vector, then derive normalized domain fractions and
   the requested 2H/4H/6H phase totals.
3. Reject profile selections that cannot identify all three phase totals and report conditional
   delta-chi-square profile bounds when they are identifiable.
4. Prove exact, noisy, and interleaved-`L` held-out recovery directly in profile space.

The user explicitly excluded detector pixels and 3000 by 3000 images from this revision. Source,
geometry, mosaic, optics, detector quadrature, background, exposure, and CUDA work are therefore
not part of this branch.

## Handoff

Status: READY

Baseline SHA: `8cedf208ad90eb12d22bbd0281f1d4287f5af7c5`

Branch: `codex/minimal-stacking-population-fit`

Public APIs:

- `compile_pbi2_stacking_profile_response`
- `fit_stacking_phase_totals`
- immutable compiled-response, fit-result, and identifiability-error types

Accepted stacking-fit response revision:
`9b6ed9419b405447a0a060d8c58e64d01f14a60887313b148fcb5603c644540f`

Proof summary: six signed first-order training rods, 366 training points, and 360 points on six
distinct second-order held-out rods with interleaved `L` coordinates. The proof uses 50 layers,
planted domain fractions `(0.60, 0.16, 0.09, 0.10, 0.05)`, and planted phase
fractions `(0.60, 0.25, 0.15)`. Noiseless maximum phase error was `4.44e-16`; fixed-noise maximum
phase error was `0.00183`; all planted phase totals were inside their delta-chi-square-one profile
bounds.

Held-out profile result: weighted RMS `1.01388` under fixed Gaussian noise.

Identifiability: the training response had rank five and phase-contrast rank two. An exclusively
`(h + 2k) mod 3 == 0` selection was rejected by the phase-identifiability gate.

Legacy classification: `NO_ORACLE`. This direct signed-rod/`L` observable has no legacy final-image
equivalent; its first divergence is the newly declared pointwise intrinsic profile measure.

Convergence: not applicable. The compiler evaluates exact requested points through the accepted
finite-stack oracle; it performs no pixel or detector quadrature.

Benchmark: `2.15 s` total and `2.30 MiB` peak Python-traced memory for both response compilations,
two recovery fits, held-out prediction, and the identifiability injection.

Permanent tests retained:

- exact full-rank scale/domain/phase recovery and conditional profile bounds;
- phase-total recovery when handedness columns alias;
- five physical PbI2 columns against canonical per-parent calls plus the modulo-three failure
  injection.

Limitations: this revision fits supplied direct profile strengths only. It intentionally does not
map observations to reciprocal coordinates, integrate source or mosaic distributions, or render
detector images. The fixed epsilon, initialization, normalization, registry phase convention,
layer count, crystal revision, and query coordinates are recorded in response provenance.

Minimum integration requests: none.
