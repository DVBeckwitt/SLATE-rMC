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

## Compact cleanup execution

State: implementation and compact proof complete; final Git handoff supplies the cleanup commit SHA.

The superseding cleanup order narrows permanent proof to distinct long-term scientific boundaries.
It removes the unmerged compatibility expansion and assigns cross-material parity and detector
agreement to T06/T07.

### Retained core scope

- strict CIF ingestion with explicit occupancy, species, displacement, cell, and symmetry handling
- every individual `(h,k)` rod, with `Qr` and symmetry families retained only as metadata
- positive-phase raw complex unit-cell amplitudes in electrons and explicit raw `|F|^2` electron²
- one Bi2Se3 quintuple layer reconstructed by the three exact R-centering translations
- Pb-centered, registry-free PbI2 `F_plus` and `F_minus` layer amplitudes
- explicit and uniform finite ordered stacks
- pure Parratt reflectivity
- the separately named corrected manuscript specular composite

### Deleted scope and residue

- production `bi2se3_whole_cell_compat`, its public facade, cache identities, hard-coded factors,
  17-cell/AREA/pole-clamp observable, and legacy specular stitch
- full VESTA tables, broad material/curve sweeps, deterministic PbI2 parent payloads, generated
  traces, figures, data, mutation frameworks, convergence frameworks, and benchmark frameworks
- `ordered/pbi2_proof.py`; its cross-material/repeat parity belongs to T06/T07
- compatibility and snapshot-style permanent tests

The feature branch ancestry includes quarantined commit `589988d`, but its surviving production
hunks were not accepted: `ordered/motifs.py`, `reciprocal/lattice.py`, and
`reciprocal/rods.py` are restored to their pre-quarantine `920fc59` content. No change was
copied from the separate `wt-post-readiness-quarantine` worktree.

### Compact permanent proof

The ordered proof emits six checks and no artifacts:

1. scalar XrayDB atom sum, tracked F003 boundary, raw event identity, and electron² normalization
2. one five-atom Bi2Se3 QL reconstruction plus distinct same-family rod identities
3. one PbI2 2H motif/layer scalar boundary
4. coherent direct sum, five-repeat off-Bragg enumeration, and seven-repeat Bragg limit
5. three-point scalar Parratt recursion plus immutable-pack sample and zero-thickness collapse
6. separate raw kinematic, pure Parratt, and corrected-composite outputs, including direct
   internal-film phase and normalized `Qz^-2` high-branch equations

Permanent tests retain the same six boundaries. The focused suite passes `6/6`; the full suite
passes `11/11`. The ordered proof is `READY` at `6/6`; core proof passes `5/5`; reference
proof passes `7/7`.

### Numerical evidence

- optimized versus scalar atom sum: maximum `1.9335119931946505e-12 e` over 256 equivalent events
- QL reconstruction: maximum `3.336525901964335e-13 e`; coordinate residual
  `1.1102230246251565e-16` fractional
- PbI2 layer scalar boundary: maximum `7.588599744428075e-15 e`
- uniform finite stack: five-repeat off-Bragg error `1.807312143953211e-15 e`; exact Bragg error
  `0.0 e`
- Parratt scalar stages: maximum `1.3877787807814457e-16`; three-point pack reflectivity maximum
  `4.086730953645201e-13`
- corrected composite direct internal-phase, raw-kinematic, and normalized-high-branch errors:
  `0.0`
- fixed-fit nested composite grids `257/513/1025`: common-node absolute and relative differences
  are `0.0`

Disposable single-thread timing and memory evidence:

- 256 equivalent amplitudes: optimized median `0.01343220000853762 s`; scalar median
  `6.099646199989365 s`
- 10,000 optimized amplitudes: median `0.02356179998605512 s`
- 4,096 three-layer Parratt points: median `0.0014921999827492982 s`
- combined 10,000-amplitude plus 4,096-Parratt traced peak: `8.444900512695312 MiB`

No benchmark script, output artifact, or import-time thread mutation is retained.

### Legacy classifications

- `ordered.bi2se3_vesta`: `CORRECTED`. F003 reciprocal d-spacing agrees exactly at the sampled
  prior stage; `ordered.unit_cell_amplitude` is the first divergence from the immutable historical
  amplitude (`9.672786160086654 e`). The scalar atom sum is the downstream authority. The single
  tracked corrected VESTA F003 component check differs by at most `0.0026578549913445215 e`.
- `reflectivity.parratt_three_layer`: `MATCH`; no divergence.
- `reflectivity.manuscript_specular_composite`: `NO_ORACLE`; the immutable pack contains pure
  Parratt arrays but no composite observable.

### Public APIs

`read_crystal`, `build_rod_catalog`, `unit_cell_amplitude`, `ordered_event_result`,
`extract_pbi2_motifs`, `pbi2_layer_amplitudes`, `coherent_finite_stack`,
`uniform_finite_stack`, `parratt_reflectivity`, and `manuscript_specular_composite`.

### Minimum integration request

`IR-T04-MEASURE-GAUGE`

- Owner: shared-contract/T06 integration.
- Problem: `LayerAmplitudeResult` cannot encode the established Pb-centered, registry-free,
  positive-phase gauge. `EventIntensityResult.intensity_per_sr` currently carries T04's explicitly
  normalized raw electron², not a true per-steradian observable; T05 uses a neutral
  `intensity_electron2` field.
- Decision required: freeze one common raw event-intensity measure, ownership of `r_e^2`, detector
  solid angle, total-versus-per-layer normalization, and the layer gauge convention.
- Acceptance: T04 and T05 event results are interchangeable without a branch-local adapter, false
  per-sr labeling, or hidden scaling. T04 does not change the protected shared contract locally.

### Limitations

T04 does not implement stacking disorder, cross-material parity, mosaic, detector geometry,
detector agreement, fitting, CLI, GUI, or acceleration. Those remain T06/T07 or later work.
