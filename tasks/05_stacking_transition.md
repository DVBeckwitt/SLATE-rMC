# T05: stacking transition

Branch: `feat/stacking-transition`

Start from `PROOF_BASE_SHA`.

## Goal

Produce correct finite transition-matrix stacking intensities for the manuscript and legacy parameter families. Prove the public calculation against direct sequence enumeration and direct finite pair sums.

## Owned paths

```text
src/rasim_next/stacking/
tests/test_stacking_transition.py
this task's execution-plan and handoff sections
```

## Read-only inputs

- event-aligned `F_plus`, `F_minus`, rod identity, layer geometry, and population contracts
- immutable tracked reference pack and examples
- in-repository source/equation ledger

## Forbidden

- CIF parsing or atomic factors
- detector geometry or rendering
- mosaic/Ewald events
- interface optics or depth fields
- arbitrary evaluated expression strings
- fitting, selection, or acceleration frameworks
- shared contract and dependency edits

## Source and equation map

Tracked locations:

```text
Original paths resolve under reference/legacy_source/.
Manuscript equation labels resolve under reference/manuscript/.
```


Inventory: `PHY-STK-001` through `PHY-STK-007`, `PHY-STK-009` through `PHY-STK-018`, and `PHY-THK-003`.

Original:

```text
ra_sim/utils/stacking_fault.py:134-424, 763-1607
ra_sim/utils/polytype_stacking.py
ra_sim/stacking/motif_form_factor.py:559-574
```

Manuscript:

```text
eq:si_pbi2_registries
eq:si_pbi2_Fplus and eq:si_pbi2_Fminus
eq:si_pbi2_omega
eq:si_pbi2_T6
eq:si_pbi2_Momega
eq:si_pbi2_P
eq:si_pbi2_pair_kernel
eq:si_pbi2_finite_intensity
eq:si_pbi2_fault_parameterization
eq:si_pbi2_fault_templates
eq:si_pbi2_total_intensity
eq:si_pbi2_m0_laue
```

## Mandatory tasks

### STK-01: typed registry phases and state order

Define the exact state order, transition convention, phase sign, and initial population. Never evaluate user expressions.

### STK-02: full six-state and direct sequence oracle

Build the full transition matrix and directly enumerate short sequences, including start and end effects.

### STK-03: exact reduced calculation

Implement the exact Fourier-sector reduction and prove agreement with the full representation for arbitrary valid parameters and Miller classes.

### STK-04: direct finite pair sum

Implement a transparent self and pair sum for finite `N`. This is the primary numerical oracle.

### STK-05: parent models

Implement deterministic 2H, 4H, 6H, handed variants, the rich-epsilon model, and the reduced `a,b,d` model as separate typed APIs.

### STK-06: public finite intensity

Return event-aligned total and per-layer intensity with explicit normalization. Keep stationary/infinite results separately named and optional.

### STK-07: proof and benchmark

Compare direct enumeration, direct pair sums, full state, reduced state, and any optimized recurrence. Use immutable-pack intermediates and classify old normalization or occupancy differences.

## Required proof

- probability bounds and stochastic sums
- full `6x6` versus reduced result
- direct sequence enumeration for small `N`
- direct finite self/pair result
- `N=1`
- deterministic 2H, 4H, 6H, and handed cases
- representative Miller classes
- `h=k=0`, `F_plus=F_minus` Laue limit
- rich-epsilon and reduced `a,b,d` fixtures
- finite total versus per-layer normalization
- nonnegative real ensemble intensity
- population-order invariance
- legacy matrix and intensity classifications

## Overnight completion rule

`READY` means the mandatory manuscript/reference slice, analytic proof, immutable-pack comparison, convergence, and public contracts all pass. Record unimplemented generalization and acceleration as an explicit extension backlog. Do not mark a scientifically required reference case as optional merely to finish.

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/stacking tests/test_stacking_transition.py
pytest -q tests/test_stacking_transition.py
python -m rasim_next.proof stacking-transition --json
python -m rasim_next.proof references --json
git diff --check
```

## Stop conditions

Stop `BLOCKED` if the layer-amplitude, population, state-order, or event-intensity contract cannot express the direct finite result. Do not modify motifs or optics in this branch.

## Execution plan

State: consumer migration complete in the unstaged pre-review tree based on
`44d3075e0df9aead7eb4a440cf08b0fe7d055e7a`.

1. [x] Preserve the stable full/reduced moment recurrences and near-extinction regression.
2. [x] Consume shared layer-amplitude, layer-normal-Q, event-intensity, and tolerance contracts
   directly; delete the superseded local result and phase-input wrappers.
3. [x] Require exact event/rod/phase/gauge alignment, the one-registry-free-layer amplitude gauge,
   positive-Q-dot-R phase sign, and amplitude-owned layer repeat with no sample-normal fallback.
4. [x] Convert raw electron-squared values through the shared conversion exactly once per public
   output and return explicit finite-total or finite-per-layer angstrom-squared results.
5. [x] Return sorted unweighted population components under fixed model ID `stacking`; T07 retains
   ownership of population mass and all detector/optical/deposition factors.
6. [x] Apply the reviewed versioned tolerance artifact to the compact six-check proof while
   retaining exactly seven calculated negative controls and six permanent tests.
7. [ ] Stage or commit only after PM pre-stage review.

## Handoff

The public functions now return shared `EventIntensityResult` values with fixed
`model_id="stacking"`, caller-selected `FINITE_TOTAL` or `FINITE_PER_LAYER` normalization, and
polarization-neutral `scattering_strength_A2`. Raw electron-squared values remain internal. The
population function returns a sorted tuple of individually identified, unweighted shared results
under one population group; it cannot expose a weighted total.

The compact proof retains stochasticity/state order; direct, pair, full, and reduced finite
agreement including `N=1`; deterministic 2H/4H+/4H-/6H+/6H- across three Miller sectors;
rich-epsilon and reduced-ABD fixtures; the general Laue identity; nonnegativity, normalization,
population-order invariance, and numeric comparison of all 60 probabilities and 165 immutable-pack
curve values. Legacy classifications are `MATCH` for the transition/synthetic finite cases,
`CORRECTED` at named first stages for explicit initial population, normalization, and typed registry
phase, and `NO_ORACLE` for the deferred stationary output.

At the near-extinction counterexample, direct/full/reduced raw electron-squared values remain
`1.7407062383566972e-9`, `1.7407062353385284e-9`, and `1.7407062353385284e-9`; the positive result
is not erased. The proof consumes tolerance artifact `rasim-stage-tolerances-v1`, SHA-256
`d3739963a8decf481fc7ec87723854ef7628e8da02dbcb3e6f7e5bb41522b4b3`, with analytic or immutable
reference-derived scales and no branch-local acceptance thresholds.

Seven proof-only calculated mutations retain their assigned first-stage detections: transition
transpose, layer-count offset, normalization swap, coherent population mixing, omitted registry
phase, stationary-for-finite substitution, and a perturbed reduced coefficient. The one-shot proof
also emits the 48-event by 24-layer timing and peak-memory comparison; exact finite algebra has no
refinement variable, so the `N=1,3,6` record is equality evidence rather than invented convergence.

The six permanent tests protect distinct invariants only: state order/stochasticity; short-stack
direct/full/reduced agreement; the `N=1` initial condition; coherent zeros and nearby positive
intensity; shared event alignment/measure/layer frame; and sorted unweighted populations with
individual initial states. Stationary/infinite output, population weighting, optics, solid angle,
deposition, detector hits/images, and cross-material parity remain outside T05.
