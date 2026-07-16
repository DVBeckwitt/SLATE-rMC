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

State: READY pending final commit.

1. [x] Replace the cancellation-prone finite self/pair subtraction in both public paths with exact
   state-conditioned amplitude mean/variance recurrences. Keep a transparent pair sum only in the
   compact proof for short stacks.
2. [x] Add one permanent coherent-zero neighborhood regression covering both sides of the zero,
   multiple layer counts, both public paths, and an independent direct amplitude sum.
3. [x] Make population components unweighted and give every `StackingPopulation` its own
   `InitialPopulation`. Return branch-local raw electron-squared event results rather than placing
   them in a per-steradian field.
4. [x] Prune broad sweeps, material parity scaffolding, mutation traces, stationary/infinite code,
   retained benchmark code, redundant tests, and stale handoff claims. Keep only the compact T05
   finite-stack proof and six permanent invariants.

## Handoff

Status: READY. Exact commit SHA is recorded after the final commit.

Public APIs retained: `STATE_ORDER`, `StackingState`, `TransitionLaw`, `InitialPopulation`,
`RegistryPhaseModel`, `Parent`, `Handedness`, `RichEpsilonModel`, `ReducedABDModel`,
`StackingPopulation`, `FiniteIntensity`, `StackingEventIntensityResult`,
`PopulationIntensityResult`, `registry_phase`, `full_transition_matrix`,
`reduced_transition_matrix`, `finite_explicit_sequence_intensity`,
`finite_intensity_by_enumeration`, `finite_intensity_full`, `finite_intensity_reduced`,
`finite_event_intensity`, and `finite_population_event_intensity`.

Near-extinction correction: the public full and reduced implementations now propagate conditional
complex-amplitude means and variances. They never form the cancellation-prone difference between
large self and pair terms. For `N=512`, `F+=F-=1`, `omega=1`,
`xi=exp(i*(2*pi/512+1e-9))`, 2H plus-only, direct/full/reduced electron-squared intensities are
`1.7407062383566972e-9`, `1.7407062353385284e-9`, and
`1.7407062372412564e-9`. The maximum direct-versus-public absolute error across exact zeros and
both nearby phases for `N=(8,64,512)` is `9.848353643232856e-18`; nearby positive intensity is
never clipped to zero.

Compact proof: six checks pass. The stochastic row-sum error is zero. At `N=6`, direct enumeration
versus the transparent pair sum, stable full path, and stable reduced path have absolute errors
`1.510e-14`, `1.421e-14`, and `1.599e-14`. The `N=1` error and total/per-layer normalization error
are zero. Per-population components with individual initial states agree with direct enumeration to
`1.776e-15` and are exactly order invariant. The proof JSON validates against
`schemas/proof_result.schema.json`.

Output and population semantics: T05 returns raw finite-stack total and per-layer electron-squared
arrays aligned by exact `event_id`. It does not label either value per steradian and applies no
`r_e^2`, source, optical, geometric, detector, phase-population, or parent-population factor.
`StackingPopulation` contains only its ID, transition law, and initial orientation. Population
results contain unweighted components and no weighted total; T07 owns ID-aligned population mass
and applies it exactly once.

Disposable benchmark: 48 events by 24 layers, minimum of five measured runs. Stable full-state
evaluation took `0.08004 s` with `41,742` peak traced bytes. Stable vectorized reduced evaluation
took `0.01425 s` with `19,786` peak traced bytes (`5.62x` faster). Their maximum absolute difference
was `4.263256414560601e-14`. No benchmark code or artifact is retained.

Permanent tests retained, one distinct invariant each:

1. declared state order and stochastic transition rows;
2. direct enumeration/full/reduced agreement on a short stack;
3. the `N=1` initial-population limit;
4. exact coherent zeros and positive neighborhoods;
5. raw event alignment and explicit total/per-layer normalization;
6. unweighted population components with per-population initial states.

Minimum integration request `T05-T07-INTEGRATION-BOUNDARY`: T07 must accept the raw event-aligned
electron-squared components without routing them through `intensity_per_sr`, provide a separate
ID-aligned population-weight table, and own both population weighting and any `r_e^2`/
per-steradian conversion exactly once. This request does not block the self-contained T05 finite
core. Cross-material T04/T05 A/B/C parity belongs to T06/T07 and is intentionally absent here.

Known limitations: homogeneous first-order transition law and one explicit layer repeat per event
batch. Fitting, detector rendering, optics, material parsing, cross-branch parity, and stationary
infinite-stack evaluation are outside this compact branch.
