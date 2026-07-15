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

State: READY_NUMERICS / MATERIAL_POLYTYPE_PARITY_PENDING

1. [x] Audit the frozen equations and interfaces.
   Files likely touched: `tests/test_stacking_transition.py`, this execution plan. Intended behavior:
   preserve core-v4, raw electron amplitudes/intensities, exact event alignment, and match
   `main.pdf` Eqs. (37)-(54) after the two-state reduction. Tests: exact `M(omega)` and `P=M(1)`
   assertions. Dependencies: none. Acceptance: phase sign, row/column convention, matrix entries,
   finite normalization, and layer repeat are explicit and unchanged.
2. [x] Correct the revealed full-oracle cancellation error.
   Files likely touched: `src/rasim_next/stacking/finite_intensity.py`,
   `tests/test_stacking_transition.py`. Intended behavior: retain the six-state self/pair equation
   while using accurate finite summation and linear peak scratch. Tests: exact 2H-derived
   cancellation fixture, coherent extinction, overflow rejection. Dependencies: task 1.
   Acceptance: frozen `1e-10*max(1,I_direct)` bound passes; nonfinite results fail clearly; no
   transition/reduction equation changes.
3. [x] Add the deterministic explicit-sequence oracle.
   Files likely touched: `src/rasim_next/stacking/enumeration.py`, package `__init__.py`, compact
   T05 test. Intended behavior: consume event-aligned amplitudes, typed orientation/registry states,
   explicit physical depths/repeat, `h,k,qz`, and finite layer count; return direct raw
   `abs(sum(F*registry_phase*vertical_phase))**2` and `/N`. Tests: direct formula, event alignment,
   state count, depth spacing, and reduced 4H parity. Dependencies: task 1. Acceptance: no CIF,
   atomic-factor, shared-contract, normalization, or downstream-weight logic.
4. [x] Prove deterministic parent and handed sequences.
   Files likely touched: `src/rasim_next/stacking/proof.py`. Intended behavior: compare explicit,
   enumeration, full, and reduced results for 2H, both 4H hands, both 6H hands, one to three periods,
   all `(2h+k) mod 3` sectors, and a physical `qz` grid. Test: stacking proof command. Dependencies:
   tasks 2-3. Acceptance: 2,295 comparisons; maximum scaled error at most `1e-10`.
5. [x] Run final gates, prune residue, and commit.
   Files likely touched: this handoff only. Intended behavior: retain only unique permanent proof
   coverage and report numerical/material states separately. Tests: compile, lint, compact T05 and
   repository pytest, stacking/reference proofs, diff check, status check. Dependencies: tasks 1-4.
   Acceptance: one owned-path commit, measured benchmark/memory, exact legacy classifications and
   first divergences, no temporary files/dependencies, clean Git status.

## Handoff

Status: READY_NUMERICS / MATERIAL_POLYTYPE_PARITY_PENDING

Commit SHA: branch HEAD; exact SHA is reported after the single final commit.

Public APIs: `STATE_ORDER`, `StackingState`, `TransitionLaw`, `InitialPopulation`,
`RegistryPhaseModel`, `Parent`, `Handedness`, `RichEpsilonModel`, `ReducedABDModel`,
`StackingPopulation`, `FiniteIntensity`, `PopulationIntensityResult`, `FiniteNormalization`,
`registry_phase`, `full_transition_matrix`, `reduced_transition_matrix`,
`finite_explicit_sequence_intensity`, `finite_intensity_by_enumeration`,
`finite_intensity_full`, `finite_intensity_reduced`,
`stationary_intensity_reduced`, `finite_event_intensity`, and
`finite_population_event_intensity`.

Proof summary: all thirteen proof checks pass. State order, both nontrivial registry roots,
stochastic rows, and parent vectors are exact. Direct sequence enumeration, the full six-state
pair sum, and the reduced recurrence agree for `N=1..6` and all three Miller sectors with maximum
absolute error `5.862e-14`. Independent deterministic cycles, `N=1`, the `00L` Laue identity, and
the `N=512` coherent extinction pass; the extinction residue is `0`. The immutable pack passes with
tolerance ratio `0.6826`. Population components agree with direct enumeration to `5.773e-15`; the
weighted result is exactly order invariant. A consistent amplitude/registry gauge transformation
has pair-kernel error `2.220e-16` and total-intensity error `0`; the amplitude-only mutation differs
by `0.5601` and first fails at `stacking.pair_kernel`. All eight isolated stage-local mutations fail
at their expected first stage after identical prior trace stages. The T05 suite passes `7/7`; the
compact repository suite passes `12/12`. The proof command reports `READY` for the branch-local
numerical gate and separately reports `READY_NUMERICS` and
`MATERIAL_POLYTYPE_PARITY_PENDING`. `main.pdf` Eqs. (37)-(54) were checked directly: the selected
Fourier channel, exact `[[A,B],[C,A]]` matrix, `P=M(1)` orientation probabilities, pair kernel,
positive vertical phase, and total/per-layer distinction match the implementation. The permanent
2,295-comparison deterministic sweep covers both hands, all three Miller sectors, physical `qz`,
and one to three parent periods. Final enumeration/full/reduced scaled errors are `2.343e-14`,
`2.207e-14`, and `2.243e-14`; the largest repeated review run was `2.60e-14`. A separate
pre-integration challenge using exact 2H-derived amplitudes had zero tolerance failures and maximum
enumeration/full/reduced scaled errors of `5.995e-13`,
`5.944e-11`, and `4.614e-11`. Neither substitutes for final A/B/C material proof.

Layer-amplitude convention: `f_plus` and `f_minus` are raw complex layer structure amplitudes in
electrons, evaluated at exact event coordinates. Occupancy and atomic displacement are included.
No intensity normalization, rounding, registry-translation phase, source/mosaic/optical/detector
factor, or population weight is included. PbI2 uses one motif gauge: in-plane origin at Pb,
layer-center `z=0`, plus/minus motifs share the origin, and the positive structure-factor phase is
retained. T05 applies every interlayer registry phase. Alignment is exclusively exact ordered
`event_id` equality; duplicates, missing/reordered IDs, nonfinite amplitudes, and nonfinite `qz`
are rejected.

Output convention: `FiniteIntensity.intensity_electron2` is raw finite-stack
`|A_stack|^2` in electron^2. `intensity_per_layer_electron2` is exactly that total divided by `N`.
The shared `EventIntensityResult.intensity_per_sr` payload carries this raw model electron^2 value
when total is selected, or the explicitly named per-layer quotient when that normalization is
selected; despite the shared field name, no per-steradian factor is applied here. T05 applies no
`r_e^2`, source, wavelength, mosaic, Jacobian, Fresnel, attenuation, polarization, solid-angle, or deposition factor.
Integration applies those factors once. The layer repeat is the explicit positive `layer_repeat_A`
input and is never inferred.

Population convention: `StackingPopulation` contains only `population_id`, a compiled
`TransitionLaw` model, and an explicit finite nonnegative weight. Population IDs are unique;
weights must sum to one within `1e-12` and are never normalized silently. Components are returned
unweighted for fitting; their explicitly weighted electron^2 total is an incoherent intensity sum.
T05 applies stacking-population weights once. T07 must not reapply them.

Material-polytype parity gate: pending. The expanded tracked structures establish, using labels
anchored to the literal 2H motif, `2H=(O_plus,A)`, `4H=(O_minus,B),(O_plus,C)`, and
`6H=(O_minus,A),(O_minus,C),(O_minus,B)`. T04's current motif labels are globally swapped relative
to those names: T04 `f_plus=O_minus` and `f_minus=O_plus`. The corresponding T05 mappings are 2H
with `TWO_H` and initial minus, supplied 4H with `FOUR_H_PLUS` and initial plus after one global
registry-origin shift, and supplied 6H with `SIX_H_MINUS` and initial plus; the opposite hands use
`FOUR_H_MINUS` and `SIX_H_PLUS`.

Required frozen T04 inputs for integration are exact event-aligned 2H-derived `f_plus/f_minus`,
direct full-CIF 4H/6H structure factors on the same physical events, symmetry-expanded target
coordinates, target lattice/layer-repeat values, and target intralayer heights. Integration must
compare A: an ideal direct atom sum assembled from the exact 2H motif, B: T05 deterministic
transition intensity for the same states and depths, and C: the supplied target-CIF direct atom sum.
A/B must meet the frozen numerical tolerance. C/A residuals are reported without separate curve
normalization. A target-height-relaxed reconstruction must also match the target coordinates after
one declared global origin transform and explicit periodic-image accounting.

Legacy classifications: `stacking.transition_matrix_6`,
`stacking.transition_matrix_reduced`, `stacking.synthetic_finite`,
`stacking.reference_pack_intensity`, and `stacking.incoherent_population_mixture` are `MATCH`.
Explicit initial population, normalization, fixed typed phase models, explicit layer repeat,
epsilon bounds, and raw `a,b,d` validation are `CORRECTED`. No case is `NO_ORACLE`; population
mixing has direct numerical authority.

First divergences: initial population at `stacking.pair_kernel`; total versus per-layer and hidden
legacy area scale at `stacking.finite_intensity`; arbitrary legacy phase expressions at
`stacking.registry_phase`; implicit legacy layer repeat at `stacking.finite_intensity`; epsilon
clipping and hidden `a,b,d` normalization at `stacking.transition_matrix_6`.

Convergence: direct damped lag sums at cutoffs `(4,8,16,32)` converge monotonically to the
separately named stationary solve with errors
`(2.99008,1.224736768,0.205476733,0.005783656)` at correlation decay `0.8`.

Benchmark and peak memory: equivalent work is 48 events by 24 layers. Representative full pair
oracle: `0.13362 s`, `151,616` peak traced bytes. Reduced recurrence: `0.007118 s`, `35,639` peak
traced bytes, `18.77x` faster. Maximum optimized-versus-proof absolute error is `4.263e-14`.

Known limitations: PbI2 A/B/C material-polytype parity remains pending frozen T04 outputs;
homogeneous first-order transition law; one explicit layer repeat per event batch; direct
enumeration limited to ten layers; singular undamped stationary phases are rejected.
Generalized phase expressions, fitting, integration, resolution convolution, and backend
frameworks are intentionally absent.

Permanent test retained: `tests/test_stacking_transition.py` protects six distinct long-term
boundaries: conventions and validation, direct oracles, analytic/deterministic limits and
cancellation, typed parent/event behavior, immutable-pack parity, and incoherent population
components/order. Benchmark, convergence, gauge sweeps, and mutations remain in the explicit proof
command rather than duplicate permanent pytest.

Coordination: the approved amplitude/electron^2 convention was sent to the active T04 task for its
handoff. T04 owns amplitude construction; T05 owns registry phases and stacking populations; T07
owns downstream source, optical, geometric, and detector factors. Core-v4 was not modified. No
serializer or per-array metadata was added. T05 consumes upstream-valid event queries; T03 and T07
retain validity ownership.
