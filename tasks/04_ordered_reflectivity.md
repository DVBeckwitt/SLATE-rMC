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

State: shared-v5 consumer migration edited; verification pending.

## Handoff

Status: `BLOCKED` pending verification; shared v5 now owns Å² measure, layer metadata, and tolerances without adapters.
