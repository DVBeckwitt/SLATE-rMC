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

State: BLOCKED after compact local recovery. The finite T05 calculations are locally evidenced;
reviewed shared measure, layer-gauge, and tolerance contracts remain required for integration.

1. [x] Preserve the stable conditional-amplitude recurrence, coherent-zero regression, unweighted
   population components, honest raw electron-squared output, and explicit layer-normal phase input.
2. [x] Restore compact STK-01..07 evidence for finite direct/pair/full/reduced agreement, declared
   parents and models, analytic limits, immutable-pack values, normalization, and event alignment.
3. [x] Detect all seven assigned error injections through recalculated proof-only stages without
   adding mutation tests, stationary production code, or retained artifacts.
4. [x] Record one exact finite-equality record and one disposable 48-event by 24-layer benchmark;
   retain only the six distinct permanent tests.
5. [ ] Consume reviewed shared event-measure/layer-gauge and frozen-tolerance contracts. Until all
   three requests are resolved, numerical acceptance remains `SKIP`/`UNRESOLVED` and T05 is BLOCKED.

## Handoff

Status: BLOCKED. The proof result is schema-v1 valid and reports one exact `PASS` plus five
tolerance-dependent `SKIP` checks; it does not convert finite raw errors into acceptance without a
reviewed tolerance policy. All seven real-calculation mutations are detected at their assigned
first stages. The proof patch has stable patch-id
`9318b7f23e8d10240ece470b49d3d3b0bd3dc0d4` and remains within the +350 ceiling at
`501` additions and `152` deletions relative to `aafee400`.

Scientific evidence: the proof covers stochasticity and declared state order; `N=1,3,6`
direct-sequence, transparent-pair, full, and reduced finite equality; deterministic
2H/4H+/4H-/6H+/6H- across three Miller sectors; rich-epsilon and handed reduced-ABD fixtures; the
general `h=k=0`, `F+=F-` Laue identity; normalization, nonnegativity, population-order invariance,
and all 60 transition probabilities plus 165 immutable-pack curve values. Cross-material parity
and stationary production output remain outside T05.

Near extinction at `N=512`, `F+=F-=1`, `omega=1`,
`xi=exp(i*(2*pi/512+1e-9))`, 2H plus-only gives direct/full/reduced raw electron-squared values
`1.7407062383566972e-9`, `1.7407062353385284e-9`, and
`1.7407062353385284e-9`; nearby positive intensity is not erased. The immutable-pack raw maximum
absolute differences are `2.13623046875e-3` (full/pack), `1.52587890625e-3` (reduced/pack), and
`6.7138671875e-4` (full/reduced). These remain unaccepted numerical evidence pending policy.

Classifications: `stacking.transition_matrix_6` is `MATCH`; legacy total/per-layer normalization is
`CORRECTED` at `stacking.finite_intensity` under `PHY-STK-013`; synthetic finite curves, legacy
initial population, and legacy phase are `UNRESOLVED`; stationary output is `NO_ORACLE` and
deferred. The seven detected mutations cover transpose/reversal, layer-count offset,
total/per-layer swap, coherent population mixing, omitted registry phase, stationary-for-finite
substitution, and a perturbed reduced coefficient.

One-shot 48-event by 24-layer evidence: full evaluation took `0.1051613 s` with `34,748` peak
traced bytes; reduced evaluation took `0.0152844 s` with `19,546` peak bytes (`6.8803x`), with
maximum absolute difference `4.263256414560601e-14`. Exact finite algebra has no numerical
refinement variable; the `N=1,3,6` equality record is reported as equality evidence, not invented
convergence.

The six retained tests each protect one long-term invariant: declared state order/stochasticity;
short-stack direct/full/reduced agreement; the `N=1` initial-population limit; exact and nearby
coherent zeros; event alignment, layer-normal phase, and normalization; and unweighted population
components with individual initial states. No permanent mutation or benchmark framework is kept.

Blocking requests:

1. `T05-SHARED-EVENT-MEASURE`: a shared result capable of unweighted raw electron-squared total
   and per-layer values, with T07 owning population mass and reviewed `r_e^2`/per-steradian use.
2. `T05-LAYER-PHASE-GAUGE`: an exact-ID event handoff for finite `layer_normal_q_Ainv` (or explicit
   phase step) in the declared crystallite/layer gauge, with no sample-frame `qz_Ainv` fallback.
3. `T05-FROZEN-TOLERANCE-PROVENANCE`: a versioned tolerance policy specifying stage, units,
   scale, `atol`, `rtol`, near-zero rationale, and `tolerance_config_sha256`.

Until those contracts are reviewed, raw electron-squared results and unweighted populations remain
honestly branch-local, no PR update or merge is authorized, and overall status is BLOCKED.
