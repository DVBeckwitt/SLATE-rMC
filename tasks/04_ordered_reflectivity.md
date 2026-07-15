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

State: READY

### Execution result

- T04-00 through T04-11 are complete. The branch implements strict tracked-CIF expansion, physical
  reciprocal bases and rods, event-aligned raw amplitudes and intensities, PbI2 layer motifs,
  finite ordered stacks, material optics, pure Parratt recursion, and named specular outputs.
- The immutable ordered case is locally `CORRECTED`: reciprocal `d` values still match through
  `1.3322676295501878e-15 A`, and `ordered.unit_cell_amplitude` is the first evidenced divergence.
  Independent scalar XrayDB atom sums agree with production to `2.0920868753727402e-12 e`.
- The mandatory 2H-derived/target-CIF PbI2 polytype oracle passes at arbitrary event coordinates,
  including exact topology, explicit image unwrapping, both hands, and registry-free `F_plus` and
  `F_minus`; its maximum complex-amplitude discrepancy is `1.596746205780465e-13 e`.
- The mandatory Bi2Se3 R-3m proof expands and deduplicates the corrected CIF, independently parses
  all 15 P1 atoms, derives one complete Se-Bi-Se-Bi-Se quintuple layer (QL), and reconstructs the
  cell only with exact R-centering translations. Direct R-3m/P1, R-3m/QL, and QL/analytic maxima
  are `1.8085231965793619e-12`, `2.5301707487933607e-12`, and
  `1.5023400790586468e-12 e`, respectively.
- All 206 VESTA rows now pass proof-only Waasmaier-Kirfel/Cu Kalpha1 parity, including the nuclear
  Thomson term, while production XrayDB remains separate. Maximum real/imaginary residuals are
  `1.5103053726761573e-5/6.354053656565384e-5 e`; RMS complex residual is
  `2.1300349223246674e-5 e`.
- Pure Parratt remains `MATCH`; the named manuscript composite is `NO_ORACLE` and passes nested-grid
  convergence. All 15 proof checks, 13 original injections, and 10 mandatory Bi2Se3 mutations pass.

### Scope and file policy

Implement the approved reference-first T04 slice only. The owned production directories do not
exist at the proof base, so focused additions are unavoidable. Prefer extending the authoritative
module and the single permanent test module over adding helpers, adapters, fixtures, or extra test
files. Delete or fold any exploratory implementation before handoff. Do not create generic
`tasks/plan.md` or `tasks/todo.md`; this section is the branch-owned plan location.

Fixed decisions:

- Gemmi owns CIF parsing and symmetry expansion. Missing isotropic displacement is preserved as
  `None`; any anisotropic displacement metadata raises an explicit unsupported-input error.
- XrayDB owns `f0`, `f1`, `f2`, atomic mass, and atomic number. Its argument is
  `q=|Q|/(4*pi)`. Ionic `f0` lookup may fall back to the neutral species only with recorded
  provenance; anomalous factors remain neutral-element data.
- Raw complex amplitudes use the declared positive phase sign and electron units. Ordered event
  intensity is `|F|^2` in raw electron squared. No universal scattering-scale factor is applied.
- Hexagonal family identity uses exact integer `m=h^2+h*k+k^2`. General-cell families use a
  canonical exact in-plane symmetry orbit plus the reciprocal-cell provenance; floating `Qr` is
  metadata, never identity.
- `reciprocal_basis_Ainv` has crystal-frame columns `(b1,b2,b3)`, and reciprocal vectors are
  `reciprocal_basis_Ainv @ [h,k,L]`. A rod catalog never stores sample-frame or geometry-dependent
  bases.
- `RodQueryBatch.l_coordinate` drives crystallographic phase. For the c-axis-normal layered
  reference slice, `qz_Ainv` is checked against the reciprocal basis. Arbitrarily oriented
  non-layered event queries remain an extension, not a silent approximation.
- Pure raw kinematic intensity, pure dimensionless Parratt reflectivity, and the named
  dimensionless manuscript composite remain separate results.

Dependency graph:

```text
T04-00 preflight
  -> T04-01 CIF model
      -> T04-02 reciprocal lattice
          -> T04-03 raw structure amplitude
          -> T04-05 rod catalog
      -> T04-04 material optics
      -> T04-07 PbI2 motifs
  T04-03 + T04-05 -> T04-06 event-aligned ordered result
  T04-03 + T04-07 -> T04-08 finite ordered stack
  T04-04 -> T04-09 Parratt
  T04-08 + T04-09 -> T04-10 named specular outputs
  T04-01..T04-10 -> T04-11 proof, mutations, convergence, benchmark
  T04-11 -> T04-12 cleanup, handoff, commit
```

### T04-00: Reproduce the proof-base preflight

**Files likely touched:** none; read-only setup.

**Intended behavior:** Confirm `feat/ordered-reflectivity`, clean status, and common worktree SHA.
Set `PROOF_BASE_SHA` process-locally to that verified SHA, run the frozen environment setup, verify
the seed, then run core and reference proofs before any source edit.

**Tests:**

- `uv sync --frozen --group dev`
- `uv run --frozen --group dev python scripts/verify_seed.py`
- `uv run --frozen --group dev python -m rasim_next.proof core --json`
- `uv run --frozen --group dev python -m rasim_next.proof references --json`

**Dependencies:** none.

**Acceptance criteria:**

- Branch, HEAD, and `PROOF_BASE_SHA` equal the common proof-base SHA; Git status is clean.
- Seed, core proof, reference proof, contract version, trace version, and reference-pack hash pass.
- Any mismatch stops the branch before implementation.

### T04-01: Parse one immutable crystallographic model

**Files likely touched:**

- `src/rasim_next/materials/crystal.py`
- `src/rasim_next/materials/__init__.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Add frozen `CrystalStructure` and expanded-site values. Parse exactly one CIF
structure with Gemmi; validate finite positive cell volume, finite fractional coordinates,
occupancy in `[0,1]`, recognized species, and displacement metadata. Expand symmetry once without
duplicating special positions. Preserve source label, charged species label, neutral element,
occupancy, fractional position, isotropic `U` or explicit unknown state, multiplicity, and
provenance. Reject ambiguous or anisotropic metadata explicitly.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_cif_expansion_occupancy_and_displacement_policy`

The test covers tracked Bi2Se3/PbI2 inputs, symmetric Bi2Se3 versus expanded P1 site equivalence,
a synthetic special position, partial occupancy, malformed occupancy, and anisotropic rejection.

**Dependencies:** T04-00.

**Acceptance criteria:**

- Every tracked reference CIF parses with expected expanded species counts and site multiplicity.
- Symmetry expansion and explicit P1 input represent the same occupied sites modulo lattice images.
- Unsupported displacement or invalid crystallographic metadata fails with an informative error.

### T04-02: Establish the reciprocal-lattice authority

**Files likely touched:**

- `src/rasim_next/reciprocal/lattice.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Construct direct basis vectors as columns and the physical reciprocal basis
as `2*pi*inv(A).T`. Provide batched Miller-to-Cartesian `Q`, the in-plane reciprocal metric,
`Qr`, and layered `L`/`Qz` consistency checks. Keep all values in angstrom and inverse angstrom;
never use a hexagonal shortcut in the general calculation.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_general_reciprocal_basis_and_layer_coordinate`

The oracle directly inverts a non-orthogonal synthetic cell and checks `a_i dot b_j=2*pi*delta_ij`,
then checks the hexagonal manuscript cell and noninteger `L`.

**Dependencies:** T04-01.

**Acceptance criteria:**

- Direct and reciprocal bases satisfy the analytic dual-basis identity at float64 tolerance.
- General-cell `Q`, in-plane metric, and `Qr` agree with direct matrix evaluation.
- Inconsistent layered `L` and `Qz` are rejected rather than silently reconciled.

### Checkpoint A: Crystallographic foundation

- Run the two focused tests from T04-01 and T04-02 together.
- Run Ruff on `materials/`, `reciprocal/lattice.py`, and the permanent test module.
- Confirm no parser alternative, generated CIF, cache, or temporary fixture exists in the tree.

### T04-03: Evaluate raw complex structure amplitude

**Files likely touched:**

- `src/rasim_next/materials/optics.py`
- `src/rasim_next/ordered/amplitudes.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Implement the single XrayDB atomic-factor source and the authoritative
amplitude sum
`sum occupancy * (f0 + f1 + i*f2) * phase * displacement`. Use physical Cartesian `Q`, positive
phase sign, isotropic factor `exp(-Uiso*|Q|^2/2)`, per-event wavelength, and complex128 output.
Expose no normalized, rounded, absolute-value-only, or pruned path.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_raw_structure_amplitude_matches_direct_atom_sum`

The independent oracle contains one-atom, cancelling two-atom, partial-occupancy, isotropic-
displacement, anomalous-sign, systematic-absence, and noninteger-`L` cases.

**Dependencies:** T04-01, T04-02.

**Acceptance criteria:**

- Atomic and unit-cell amplitudes match direct scalar enumeration as complex values.
- XrayDB receives `|Q|/(4*pi)` exactly; ionic fallback is explicit in provenance.
- Near-zero systematic absences use an absolute tolerance and remain unfabricated.

### T04-04: Produce wavelength-consistent material optics

**Files likely touched:**

- `src/rasim_next/materials/optics.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Derive occupied unit-cell composition, mass density, electron/anomalous
forward sums, `delta`, `beta`, `n_complex=1-delta+i*beta`, and `mu_Ainv=4*pi*beta/lambda` from the
same species resolution used by T04-03. Return the shared immutable `MaterialOptics` contract in
the input wavelength order with complete provenance.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_material_optics_matches_atomic_forward_sum`

The test checks density from mass/volume, optical-theorem identities, positive absorption,
wavelength dependence, anomalous sign, and charged/neutral resolution.

**Dependencies:** T04-01, T04-03.

**Acceptance criteria:**

- `n_complex`, `delta`, `beta`, and `mu_Ainv` satisfy their defining identities at every wavelength.
- Composition includes symmetry multiplicity and occupancy exactly once.
- Missing species data or invalid density stops with no fallback material.

### T04-05: Enumerate complete rod identities

**Files likely touched:**

- `src/rasim_next/reciprocal/rods.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Enumerate every integer `(h,k)` in explicit inclusive bounds, sort
deterministically, and assign unique stable `rod_id`. Compute `Qr` from T04-02. Use exact
hexagonal `m` or a canonical general-cell in-plane symmetry orbit for `family_key`/`family_id`.
Record orbit/multiplicity provenance without multiplying intensity, collapsing rods, pruning weak
members, or creating fractional members.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_rod_catalog_preserves_rods_and_exact_families`

The test includes distinct rods sharing one hexagonal family, a general cell, zero-intensity and
weak rods, deterministic repeated construction, and checks that every `(h,k)` remains integral.

**Dependencies:** T04-01, T04-02.

**Acceptance criteria:**

- Catalog size equals the exact Cartesian count of requested integer bounds with unique IDs.
- Equal-family rods remain distinct while exact family metadata and numerical `Qr` agree.
- Pruning, strongest-peak normalization, rounding, and fabricated fractional rods are absent.

### Checkpoint B: Material and rod authority

- Run focused tests T04-03 through T04-05 together plus Checkpoint A tests.
- Compare Bi2Se3 raw complex amplitudes with tracked pack arrays using predeclared branch-local
  tolerances; do not alter tolerances after seeing disagreement.
- Classify any disagreement by the evidence hierarchy. This checkpoint resolved the historical
  arrays as `CORRECTED`; no silent compatibility mode was added.

### T04-06: Align ordered results to event identity

**Files likely touched:**

- `src/rasim_next/ordered/amplitudes.py`
- `src/rasim_next/ordered/__init__.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Validate every `RodQueryBatch` row against its catalog `rod_id`, phase,
`h`, and `k`; evaluate amplitude in event order; preserve `event_id`; and expose raw amplitude plus
`EventIntensityResult` with raw `|F|^2` in electron squared and no source, population, optics,
polarization, solid-angle, deposition, or universal scattering-scale factor.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_event_aligned_ordered_result_preserves_identity_and_scale`

The test uses shuffled events, repeated rods, noninteger `L`, multiple wavelengths, and deliberate
rod/phase/index mismatches.

**Dependencies:** T04-03, T04-05.

**Acceptance criteria:**

- Output order and IDs exactly match the query; identity mismatch raises before evaluation.
- Returned model intensity equals raw `|F|^2` electron squared with no hidden normalization or factor.
- Invalid `L`/`Qz` reference semantics fail explicitly.

### T04-07: Extract validated PbI2 layer motifs

**Files likely touched:**

- `src/rasim_next/ordered/motifs.py`
- `src/rasim_next/ordered/pbi2_proof.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** From expanded tracked PbI2 structures, associate each Pb with two nearest
periodic I images, center each I-Pb-I motif on Pb, require one-Pb/two-I stoichiometry, preserve
occupancy/species, cover each expanded site exactly once, and classify the two orientation-related
motifs after removing registry translation. Evaluate event-aligned `F+` and `F-` through the same
atomic-factor authority and return `LayerAmplitudeResult`. Independently enumerate exact 2H-derived
ideal 4H/6H atoms and tracked target-CIF atoms at arbitrary event coordinates without calling T05;
return both target canonical images and complete-layer coordinates, lattices, repeats, relaxed
heights, both hands, and the explicit Pb-centered motif gauge.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_pbi2_motifs_cover_sites_and_preserve_orientation`
- `uv run --frozen --group dev python -m rasim_next.proof ordered-reflectivity --json`

The compact permanent test covers exact 2H extraction, direct manuscript `F+`/`F-`, ambiguous-
neighbor rejection, stoichiometry, site coverage, and species/occupancy-preserving orientation.
The branch proof expands all tracked 2H/4H/6H CIFs, independently sums every atom, checks exact
topology and image shifts, and emits the larger integration payload without retaining snapshots.

**Dependencies:** T04-01, T04-02, T04-03.

**Acceptance criteria:**

- Every tracked PbI2 site belongs to exactly one validated trilayer with correct stoichiometry.
- `F- (h,k,L) = F+ (h,k,-L)` holds for the reference orientation convention.
- Ambiguous assignment or non-rigid species/occupancy mapping raises an informative error.
- Direct ideal polytype sums factor into registry/depth-phased `F+`/`F-`; target canonical sums
  match production, and complete-layer image shifts are explicitly proven at arbitrary `qz`.

### T04-08: Sum a finite ordered stack coherently

**Files likely touched:**

- `src/rasim_next/ordered/finite_stack.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Sum complex layer or repeat amplitudes at explicit finite depths before
squaring. Provide both a transparent direct sum and a stable uniform-repeat geometric sum, using a
near-Bragg limit that avoids cancellation. Preserve event IDs and expose total finite intensity;
never silently switch to per-layer normalization. Treat motif-origin shifts as a gauge paired with
the corresponding registry/depth phase.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_finite_stack_limits_and_motif_gauge`

The test checks `N=1`, `Qz=0` giving `N^2`, exact Bragg and off-Bragg geometric sums, arbitrary
depths against direct enumeration, total/per-layer distinction, and origin/registry gauge
invariance.

**Dependencies:** T04-03, T04-07.

**Acceptance criteria:**

- Optimized and direct finite sums agree as complex amplitudes across the analytic fixtures.
- Total intensity is nonnegative, finite, and explicitly normalized as a finite total.
- Motif-origin changes leave physical intensity invariant only with the declared phase adjustment.

### Checkpoint C: Ordered vertical slice

- Run all permanent ordered/material/rod tests through T04-08.
- Verify `RodQueryBatch -> raw amplitude -> EventIntensityResult` and
  `RodQueryBatch -> LayerAmplitudeResult` boundaries with exact IDs, dtypes, and read-only arrays.
- Run compileall and Ruff on all current owned modules; remove duplicate direct-sum helpers.

### T04-09: Implement pure Parratt recursion

**Files likely touched:**

- `src/rasim_next/reflectivity/parratt.py`
- `src/rasim_next/reflectivity/__init__.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Accept explicit ambient/interior/substrate indices, `None` thickness for the
two semi-infinite media, nonnegative finite interior thicknesses, and one RMS roughness per interface.
Derive each layer `kz` with the shared complex-normal branch selector; derive the scalar reflection amplitude
from the shared interface amplitude; apply the declared roughness factor; recurse bottom-up; return
layer `kz`, interface amplitudes, recursion amplitude, and pure dimensionless reflectivity.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_parratt_matches_analytic_limits`

The independent oracle covers a single interface, equal media, zero-thickness film, absorbing
thick film, finite substrate, and nonzero roughness.

**Dependencies:** T04-04 and bootstrap shared wave/interface primitives.

**Acceptance criteria:**

- Scalar and batched recursion match analytic/direct scalar results at complex128 tolerance.
- Every layer uses the shared branch selector; no local square-root branch implementation exists.
- Invalid media, thickness, shape, or roughness inputs fail before recursion.

### T04-10: Keep all specular outputs named and separate

**Files likely touched:**

- `src/rasim_next/reflectivity/specular.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Expose raw finite-stack kinematic intensity without normalization. Build the
named manuscript composite separately: map external `Qz` to internal phase `L`, normalize the
finite-stack term at zero, divide by `Qz^2`, fit the positive log-median scale in the declared
overlap, select the widest eligible blend interval with deterministic tie-breaking and `[3,6]`
fallback, then use clipped quintic log blending. Return pure Parratt, pure raw kinematic, scaled
high branch, blend bounds, and composite as distinct fields.

**Tests:**

- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py::test_specular_outputs_and_handoff_are_distinct`

The test checks raw-scale preservation, internal-phase coordinate below/above critical edge,
automatic and fallback windows, exact Parratt equality below the window, exact high-branch equality
above it, endpoint continuity, and no blend outside the rule.

**Dependencies:** T04-08, T04-09.

**Acceptance criteria:**

- Pure outputs remain unchanged and independently accessible after composite construction.
- Composite equals its declared branches exactly outside the selected window and remains finite.
- Nonpositive overlap data fails unless only the documented numerical floor inside a valid blend
  is needed.

### Checkpoint D: Reflectivity vertical slice

- Run all permanent tests through T04-10.
- Compare the three-layer Parratt stages with the immutable pack as `MATCH`.
- Refine nested `Qz/Qc` grids and confirm shared-point Parratt values and selected composite window
  converge under the predeclared criteria.

### T04-11: Consolidate proof, mutations, convergence, and benchmark

**Files likely touched:**

- `src/rasim_next/ordered/proof.py`
- `tests/test_ordered_reflectivity.py`

**Intended behavior:** Add the existing proof-dispatch target. Emit one JSON object containing
analytic/direct-oracle checks, immutable pack hash, `MATCH`/`CORRECTED`/`NO_ORACLE`
classifications, first divergences, branch-local tolerance hash, optimized-versus-proof agreement,
nested-grid convergence, equivalent-work wall time, traced peak memory, limitations, and every
assigned ordered/reflectivity mutation. Mutations remain in-memory proof operations, never
production switches.

**Proof command:**

- `uv run --frozen --group dev python -m rasim_next.proof ordered-reflectivity --json`

The broad convergence sweep and benchmark stay in this explicit handoff command rather than a
permanent pytest case.

Benchmark equivalent work is 10,000 event-aligned structure amplitudes plus 4,096 three-layer
Parratt points, one BLAS/OpenMP thread, warmup followed by repeated timed runs. Compare vectorized
production paths with scalar atom/layer/Parratt oracles and measure peak memory with `tracemalloc`.

**Dependencies:** T04-01 through T04-10.

**Acceptance criteria:**

- Both tracked T04 pack cases pass their immutable classifications; every correction names its
  first divergent shared stage and downstream independent oracle.
- Every assigned mutation is detected at its expected first stage while the unmutated proof passes.
- Convergence, equivalent-work timing, peak memory, and optimized/proof maximum errors are finite
  and present in proof JSON.

### T04-12: Remove residue and complete the handoff

**Files likely touched:**

- `tasks/04_ordered_reflectivity.md`
- Earlier owned files only when deleting or simplifying residue; add no new file.

**Intended behavior:** Review every added function, file, and permanent test. Delete exploratory
scripts, generated data, duplicate helpers, redundant tests, diagnostic output, dead branches,
unused imports, and unused abstractions. Populate the existing handoff with exact evidence, make
one coherent commit, then confirm a clean tree.

**Tests:**

- `uv run --frozen --group dev python -m compileall -q src`
- `uv run --frozen --group dev ruff check src/rasim_next/materials src/rasim_next/reciprocal/lattice.py src/rasim_next/reciprocal/rods.py src/rasim_next/ordered src/rasim_next/reflectivity tests/test_ordered_reflectivity.py`
- `uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py`
- `uv run --frozen --group dev python -m rasim_next.proof ordered-reflectivity --json`
- `uv run --frozen --group dev python -m rasim_next.proof references --json`
- `git diff --check`

**Dependencies:** T04-11.

**Acceptance criteria:**

- All assigned commands pass; optimized and proof paths agree; benchmark and memory are recorded.
- Handoff records commit SHA, public APIs, classifications, first divergences, convergence,
  benchmark, memory, retained tests and rationale, limitations, and minimum contract requests.
- One coherent commit contains only owned-path changes and `git status --short` is empty afterward.

### Risks and stop rules

- If production and the independent scalar XrayDB atom sum disagree beyond the frozen direct-oracle
  tolerance, stop and report the exact first divergence.
- If `RodQueryBatch` cannot represent the layered reference `L`/`Qz` semantics, stop rather than
  editing shared contracts or inferring orientation.
- If Gemmi cannot preserve a reference species, occupancy, special position, or displacement
  field, stop rather than lowering proof coverage.
- If a consumer cannot accept the frozen raw-electron/raw-electron-squared boundary, record the
  smallest integration request and stop before creating an adapter or applying a universal scale.
- Work is sequential for the primary writer. Read-only derivation or evidence review may occur in
  parallel, but no subagent writes branch files.

## Handoff

Status: READY. Final branch-tip SHA is reported externally because a commit cannot embed itself.

Frozen T04/T05 boundary:

- `LayerAmplitudeResult.f_plus` and `f_minus` are raw complex layer structure amplitudes in
  electrons, evaluated at exact `RodQueryBatch` event coordinates. They use one common motif gauge:
  in-plane origin at Pb, layer-center `z=0`, shared plus/minus origin, and positive structure-factor
  phase. Occupancy and atomic displacement are each applied exactly once. Registry-translation
  phase, source/mosaic/optics/detector/population weights, intensity normalization, rounding, and
  universal scattering scale are excluded.
- T05 applies every interlayer registry phase. Ordered and stacking event intensities are raw
  electron squared; neither T04 nor T05 multiplies by `r_e^2`. Integration applies any universal
  scattering-scale factor exactly once.
- Rod identity remains in `RodQueryBatch`, and all alignment is through `event_id`; layer results do
  not duplicate rod metadata.
- T04 manuscript field labels remain literal: expanded 2H is `f_minus`. For coordination wording,
  coordination `+` maps to T04 `minus`, and coordination `-` maps to T04 `plus`.

Public APIs:

- Materials: `read_crystal`, `CrystalSite`, `CrystalStructure`, `MaterialOptics`,
  `material_optics`, `mass_density_g_cm3`.
- Reciprocal/ordered: `ReciprocalLattice`, `RodCatalog`, `build_rod_catalog`,
  `unit_cell_amplitude`, `ordered_event_result`, `pbi2_layer_amplitudes`,
  `coherent_finite_stack`, `uniform_finite_stack` and their immutable result models.
- Reflectivity: `ParrattResult`, `parratt_reflectivity`, `SpecularResult`,
  `manuscript_specular_composite`.

Exact files changed:

- `src/rasim_next/materials/__init__.py`, `crystal.py`, and `optics.py`.
- `src/rasim_next/reciprocal/lattice.py` and `rods.py`.
- `src/rasim_next/ordered/__init__.py`, `amplitudes.py`, `bi2se3_proof.py`, `finite_stack.py`,
  `motifs.py`, `pbi2_proof.py`, and `proof.py`.
- `src/rasim_next/reflectivity/__init__.py`, `parratt.py`, and `specular.py`.
- `tests/test_ordered_reflectivity.py` and `tasks/04_ordered_reflectivity.md`.

Final command results:

- `python -m compileall -q src`: PASS.
- Ruff lint and format checks over all owned Python paths: PASS (`16` files formatted).
- Focused permanent suite: PASS (`10 passed`); full suite: PASS (`15 passed`).
- Ordered proof: READY (`15/15` checks; `13/13` original and `10/10` Bi2Se3 injections); core proof: PASS (`5/5`);
  reference proof: PASS (`7/7`).
- `git diff --check`: PASS. Post-commit `git status --short`: empty.

Permanent proof coverage:

- Tolerance policy `ordered-reflectivity-tolerances-v2` was frozen before the Bi2Se3 module was
  implemented; policy SHA-256 is
  `1a669aa24fbfd26c7dee36091b6a715d318d34c64db8d2f20b6134eb875d7ddf`. The `1e-4 e`
  component and `5e-5 e` RMS gates cover source-rounded VESTA coefficients; `6e-4 e` magnitude and
  `5e-6` relative gates cover the export's six-significant-digit `|F|`; `6e-7 A` covers six-decimal
  d spacing; and `1.2e-4 deg` retains the required fixed wavelength near backscatter.
- Ten focused tests retain one distinct contract/invariant each: CIF expansion/ADP rejection,
  reciprocal/rod identity, raw atom sums and optics, event alignment/raw scale, PbI2 motif gauge,
  finite-stack limits and registry phase, Parratt analytic limits, and named specular separation.
- `python -m rasim_next.proof ordered-reflectivity --json` returns all integration payloads and
  passes 15 checks plus 23/23 total mutations. The immutable reference-pack hash remains
  `e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06`.
- Production/direct atom sums differ by at most `2.0920868753727402e-12 e`. Raw intensity differs
  by at most `4.5838532969355583e-10 electron2` in the equivalent-work benchmark.
- PbI2 proof expands exact tracked 2H/4H/6H CIFs, returns event `h,k,qz,wavelength`, raw `F_plus` and
  `F_minus`, explicit ideal 2H-derived amplitudes for both 4H/6H hands, direct target-CIF amplitudes,
  both canonical and complete-layer coordinates, lattices, repeats, relaxed intralayer heights, and
  integer image shifts. Topologies are 2H `(+ A)`, 4H `(- B),(+ C)`, and 6H
  `(- A),(- C),(- B)` in coordination labels. Maximum oracle error is
  `1.596746205780465e-13 e`; complete-layer image integrality error is
  `2.220446049250313e-16` fractional coordinate.
- Bi2Se3 proof returns the derived QL gauge and coordinates, all expanded R-3m and reconstructed
  coordinates, integer image shifts, `6 Bi + 9 Se` counts, exact R-centering extinctions, explicit
  P1 and continuous-complete-gauge comparisons, four handoff relations, and 10/10 traced mutation
  stages. Its gauge is central Se at `(0,0,0)`, bottom-to-top, xy modulo one, signed z, and positive
  phase. The QL sites are Se2 `(1/3,2/3,-0.12163333333333337)`, Bi
  `(2/3,1/3,-0.06746666666666667)`, Se1 `(0,0,0)`, Bi
  `(1/3,2/3,0.06746666666666656)`, and Se2 `(2/3,1/3,0.12163333333333326)`, all with occupancy
  `1` and `Uiso=0.019 A^2`. The P1 file's 12-decimal coordinates leave its largest cancellation floor,
  `1.617802490138405e-9 e`, at exact absences and a `2.76380278402231e-10 e` continuous-gauge
  discrepancy; both pass analytic coordinate-serialization bounds. Allowed table rows meet the
  frozen A/B gate, while QL/analytic absences are below `2.004728598894011e-13 e`.
- VESTA F(003): direct R-3m `104.9835149797381+14.466756284510423i e`, explicit P1
  `104.98351497973798+14.46675628451043i e`, canonical five-atom QL motif
  `34.9945049932462+4.822252094836864i e`, QL-reconstructed conventional cell
  `104.98351497973862+14.466756284510563i e`, and target
  `104.983515+14.466758i e`. Full-table max real/imaginary/RMS complex residuals are
  `1.5103053726761573e-5/6.354053656565384e-5/2.1300349223246674e-5 e`; maximum relative
  magnitude residual above `1 e` is `4.562285315313092e-6`; d-spacing and finite two-theta maxima
  are `5.229857156230366e-7 A` and `1.1231884087692379e-4 deg`; absence, hkl, and multiplicity
  disagreements are zero.
- Production XrayDB F(003) is `104.98607302019832+14.464100145008656i e`; its full-table
  maximum/RMS complex difference from VESTA is `0.12640464181321473/0.04375278114665144 e`.
- Required Bi2Se3 mutations all fail at their traced first observable stage: omitted R translation,
  duplicated QL, wrong centering translation, premature normalization, and retained special-site
  duplicate at `ordered.unit_cell_amplitude`; wrong reciprocal convention, negative phase, changed
  Se1 occupancy, doubled displacement, and reversed anomalous-imaginary sign at
  `ordered.atomic_amplitude`.
- Parratt direct scalar stage error is `3.036534414019566e-16`; pack reflectivity max/RMS/p95 is
  `2.567342172188347e-12/2.032218994038676e-13/3.3277200439662807e-14`. Zero-thickness collapse is
  `2.0835655249829126e-16`.

Legacy classification and first divergence:

- Material handoff relations are recorded exactly as requested: direct R-3m versus expanded P1 is
  an independent analytic/numerical proof; direct R-3m versus `single_QL reconstruction` is an
  independent material reconstruction proof; `single_QL versus VESTA` is `VESTA_PARITY`; and
  production XrayDB versus VESTA is an independent table comparison. Here `single_QL
  reconstruction` means the 15-atom conventional-cell observable generated only from one canonical
  five-atom QL, not the isolated motif amplitude.
- `ordered.bi2se3_vesta`: `CORRECTED`. Reciprocal d-spacing agrees through
  `1.3322676295501878e-15 A`; first evidenced divergence is `ordered.unit_cell_amplitude`. No
  atomic-stage claim is made because the immutable pack has no atomic trace. Its historical F(003)
  is `95.31913590782447+14.127768516741025i e`. Independent production/direct-XrayDB agreement and
  proof-only legacy-convention QL/VESTA parity are the downstream authorities; production factors
  are not changed to fit VESTA.
- `reflectivity.parratt_three_layer`: `MATCH`; no divergent stage.
- `reflectivity.manuscript_specular_composite`: `NO_ORACLE`; no divergent stage. The pack contains
  no composite arrays, so analytic separation and convergence are the proof.

Convergence and benchmark:

- Nested grids `1025/2049/4097` give Parratt shared-point error `0/0` and composite errors
  `1.2141353628603306e-7` then `5.442864600864461e-10`; blend bounds remain `[3,6]`, and final scale
  relative delta is `9.818399835936908e-6`.
- Equivalent work is 10,000 event-aligned amplitudes plus 4,096 three-layer Parratt points. Current
  production/oracle medians are `0.0154786999919452/0.254853499995079 s` (`16.4648x`). Peak
  traced memory is `9,098,844/3,502,446 bytes`. Thread counts are pinned to one.
- For the separately labeled 6,592-event by 15-atom Bi2Se3 work shape, direct R-3m/P1/
  QL-reconstructed-cell/production-XrayDB medians are
  `0.00427030000719242/0.00423980000778101/0.00443329999689013/0.00847460000659339 s`;
  traced allocation peaks are `4,537,879/4,537,879/4,537,879/5,937,954 bytes`. The first three
  use the VESTA-parity factor convention; production XrayDB is reported separately, not as the same
  observable.

Known limitations and integration requests:

- Layered `L/qz` validation is limited to the declared c-axis-normal crystallographic slice.
  Unknown isotropic displacement requires an explicit calculation value; nonzero anisotropic
  displacement is rejected. VESTA's intensity column is `NO_ORACLE` because all 206 values are
  NaN. Bi2Se3 continuous-L factorization uses explicit complete-image QL coordinates; wrapped
  production unit-cell coordinates differ by up to `205.44156093904647 e` and are not claimed as a
  continuous-L QL oracle. The named composite is specular-only.
- No shared contract change, unresolved interface question, or minimum integration request remains.
