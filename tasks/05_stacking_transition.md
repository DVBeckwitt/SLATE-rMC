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

State: NS

## Handoff

Status:

Commit SHA:

Public APIs:

Proof summary:

Legacy classifications:

First divergences:

Convergence:

Benchmark and peak memory:

Known limitations:

Contract requests:
