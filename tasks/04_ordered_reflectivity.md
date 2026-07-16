# T04: ordered rods, material optics, and reflectivity

Branch: `feat/ordered-reflectivity`

Start from `PROOF_BASE_SHA`.

## Goal

Produce physically scaled complex ordered amplitudes, a complete individual-rod catalog with `Qr` family metadata, wavelength-dependent material optics, finite ordered-stack intensity, and separate Parratt, kinematic, and composite specular outputs.

## Owned paths

```text
src/rasim_next/materials/
src/rasim_next/reciprocal/lattice.py
src/rasim_next/reciprocal/rods.py
src/rasim_next/ordered/
src/rasim_next/reflectivity/
tests/test_ordered_reflectivity.py
this task's execution-plan and handoff sections
```

## Read-only inputs

- shared wave-mode and interface primitives
- `RodQueryBatch`, `EventIntensityResult`, and material contracts
- immutable tracked reference pack and examples
- in-repository source/equation ledger

## Forbidden

- detector geometry, deposition, or image accumulation
- source/mosaic sampling and Ewald event generation
- transition-matrix disorder
- phase-population assembly
- fitting, branch selection, or acceleration frameworks
- shared contract and dependency edits

## Source and equation map

Tracked locations:

```text
Original paths resolve under reference/legacy_source/.
Manuscript equation labels resolve under reference/manuscript/.
```


Inventory: `PHY-MAT-*`, `PHY-REC-001`, `PHY-REC-010`, `PHY-ORD-*`, `PHY-MOT-*`, `PHY-REF-001` through `PHY-REF-007`, `PHY-THK-003`, `PHY-THK-004`.

Original:

```text
ra_sim/structure_factors/
ra_sim/stacking/motif_form_factor.py
ra_sim/stacking/motif_validation.py
ra_sim/utils/diffraction_tools.py
ra_sim/utils/calculations.py:169-304, 328-803
ra_sim/gui/structure_factor_pruning.py
```

Manuscript:

```text
eq:structure_factor
modelling-methods rod family discussion
Parratt equations eq:si_parratt_kz through eq:si_parratt_reflectivity
finite-stack and handoff equations eq:si_ht_normalized_structure through eq:si_handoff_blend
```

## Mandatory tasks

### ORD-01: crystallographic input

Use Gemmi for CIF parsing and symmetry. Preserve occupancy, special positions, species, cell geometry, and displacement metadata. For the overnight slice, support every feature used by the manuscript reference structures. Apply full anisotropic displacement when implemented; otherwise raise a clear unsupported-input error for affected inputs. Never ignore it silently.

### ORD-02: atomic and material data

Use XrayDB under the documented `q=|Q|/(4*pi)` convention. Define anomalous sign, ionic fallback, density, composition, `n`, `delta`, `beta`, and `mu` once. Produce `MaterialOptics` for geometry and Parratt.

### ORD-03: rod catalog

Construct the general reciprocal basis and every individual `(h,k)` rod. Assign exact family keys, `family_id`, and `Qr`. Keep symmetry/multiplicity provenance explicit. Do not prune or fabricate rods.

### ORD-04: event-aligned structure amplitudes

Evaluate raw complex amplitudes at arbitrary `Qz` or `L` from `RodQueryBatch`. Preserve occupancy, anomalous terms, and displacement factors. Return event-aligned intensity without max normalization or rounding.

### ORD-05: layer motifs and finite ordered stack

Construct validated layer motifs and finite coherent sums. Prove stoichiometry, site coverage, orientation relation, and motif-origin/registry-phase gauge behavior needed by stacking.

### ORD-06: Parratt

Use the shared complex normal-wavevector and interface primitives in a general multilayer bottom-up recursion with explicit semi-infinite media and declared roughness convention.

### ORD-07: specular outputs

Expose pure Parratt, pure raw kinematic, and the manuscript/legacy smooth composite as separate named outputs. The composite is not the off-specular optical model.

### ORD-08: proof and benchmark

Use direct atom sums, direct layer sums, and analytic Parratt limits. Compare raw intermediates from the immutable pack before old normalization, rounding, pruning, or artificial reflections.

## Required proof

- one-atom and two-atom amplitude cases
- occupancy, special position, systematic absence, and non-orthogonal cell
- isotropic displacement and anisotropic metadata behavior
- anomalous sign and optical consistency
- distinct rods sharing one family
- arbitrary noninteger `L`
- no fabricated fractional rods and no proof pruning
- finite-stack analytic limits
- material-optics consistency across wavelength
- Parratt single-interface, zero-thickness, thick-film, roughness, and substrate cases
- pure versus composite API distinction
- legacy raw-amplitude and reflectivity classifications

## Overnight completion rule

`READY` means the mandatory manuscript/reference slice, analytic proof, immutable-pack comparison, convergence, and public contracts all pass. Record unimplemented generalization and acceleration as an explicit extension backlog. Do not mark a scientifically required reference case as optional merely to finish.

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/materials src/rasim_next/reciprocal/lattice.py src/rasim_next/reciprocal/rods.py src/rasim_next/ordered src/rasim_next/reflectivity tests/test_ordered_reflectivity.py
pytest -q tests/test_ordered_reflectivity.py
python -m rasim_next.proof ordered-reflectivity --json
python -m rasim_next.proof references --json
git diff --check
```

## Stop conditions

Stop `BLOCKED` if species, occupancy, displacement, density, or shared wave-mode conventions cannot be represented without silent approximation. Record the smallest required change.

## Execution plan

State: local recovery complete; shared measure, phase/gauge, and tolerance contracts remain
blocking.

1. Keep the strict Gemmi 0.7 CIF validation fix and the compact six-test budget.
2. Treat `l_coordinate` as authoritative for crystal amplitudes; do not equate it to sample `qz`.
3. Preserve the HEAD result types and finite-stack signatures while recording their unresolved
   measure and phase semantics.
4. Correct Parratt for a general lossless incident ambient without changing the vacuum result.
5. Emit schema-v1 proof evidence and leave benchmarks and mutations as disposable command output.
6. Commit only owned-path local fixes; do not publish or rewrite history.

## Handoff

Status: `BLOCKED`

Commit SHA: the local checkpoint commit containing this handoff.

Public signatures and result types: unchanged from
`8954474702009b61e3455318d94bb048d7f1f30c` except deletion of the invalid
`ReciprocalLattice.validate_layer_qz` helper. Existing ordered and finite-stack result types and
call signatures are preserved. `read_crystal` regains strict Gemmi 0.7 validation, crystal
amplitudes use authoritative `L`, and Parratt now handles a general lossless incident ambient.

Proof summary: focused `6/6` and full `11/11` pass with both Gemmi 0.7.1 and 0.7.5. The ordered
proof has six passing checks, validates against `proof_result.schema.json` v1, and reports
`BLOCKED`; core passes `5/5`; references pass `7/7`. Eleven disposable ordered/reflectivity
mutations were detected at their first affected stages.

Benchmark and peak memory: one frozen single-thread run over 256 equivalent amplitudes took
`0.011439099995186552 s` optimized versus `5.972575199994026 s` scalar, with maximum disagreement
`5.275749907936691e-13 e`. A combined 10,000-amplitude and 4,096-point Parratt run took
`0.088571200001752 s` with `8.445280075073242 MiB` traced peak. No benchmark artifact remains.

Legacy classifications:

- `ordered.bi2se3_vesta`: `CORRECTED`. F003 reciprocal d-spacing agrees exactly at the sampled
  prior stage; `ordered.unit_cell_amplitude` is the first divergence from the immutable historical
  amplitude (`9.672786160086654 e`). The scalar atom sum is the downstream authority. The single
  tracked corrected VESTA F003 component check differs by at most `0.0026578549913445215 e`.
- `reflectivity.parratt_three_layer`: `MATCH`; no divergence.
- `reflectivity.manuscript_specular_composite`: `NO_ORACLE`; the immutable pack contains pure
  Parratt arrays but no composite observable.

First divergences: `ordered.unit_cell_amplitude` for the corrected Bi2Se3 legacy amplitude; none
for pure Parratt; no legacy composite oracle exists.

Convergence: not applicable to the retained exact atom, finite-sum, and Parratt kernels; no
approximation/refinement variable is present, so no convergence record is fabricated.

Known limitations: T04 does not implement stacking disorder, cross-material parity, mosaic,
detector geometry/agreement, fitting, CLI, GUI, or acceleration. The protected
`EventIntensityResult.intensity_per_sr` still contains raw electron², the finite-stack phase input
is frame-ambiguous, `LayerAmplitudeResult` lacks gauge/unit metadata, and the local tolerances have
no reviewed version/hash.

Contract requests: `IR-T04-MEASURE-GAUGE` is blocking and owned by a future reviewed shared-contract
change. It must freeze the event measure and once-only `r_e^2`/solid-angle ownership; layer units,
normalization, positive phase sign, Pb-centered origin/gauge, event/rod alignment, and projection/
registry ownership; and a versioned tolerance artifact. T06 records acceptance but does not
silently repair production. Acceptance requires direct T04/T05 substitution without adapters,
false per-steradian labels, hidden scaling, or duplicate factors.

History: after the checkpoint this remains a seven-commit development series from
`812f896fde5b8365ff5c218fc606df674ad7dcad`; integration still needs a squash. No history rewrite
or publication is performed here.
