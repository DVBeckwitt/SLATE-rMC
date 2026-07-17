# Continuous Ewald coating replacement plan

Status: proposed. No implementation has started.

## Objective

Replace the discrete mosaic-orientation and finite Ewald-candidate machinery between internal
incident ki and compact outgoing kf events. Preserve incident transport, ordered structure-factor
physics, exit transport, detector projection, and mass-conserving deposition.

The replacement samples one continuous detector-conditioned Ewald measure per incident state. It
retains every physical (h,k) rod, exact m-family metadata, and intersection labels 1 and 2 for
m != 0.

Only one new production file is planned:

    src/rasim_next/reciprocal/coating.py

All other work should modify or delete existing files.

## Smallest useful scope

For incident state i, family m, and branch b, integrate

    rho(i,m,b,alpha,beta)
      = p_alpha(alpha)/(2*pi)
        * sum over rods r in family m of
          S_r(L(i,r,b)) * J(i,r,b) * W_once(i,r,b).

The implementation then:

1. draws family and branch from integrated component masses;
2. draws continuous alpha from the component marginal;
3. draws continuous beta from its conditional distribution;
4. draws the individual rod from its exact conditional contribution;
5. recomputes exact L, q, and kf; and
6. sends the compact event through existing exit transport and deposition.

Rods are summed explicitly inside each family. A faster family-cylinder quotient is allowed only
after a separate measure-preserving proof.

No Ewald matrix, reciprocal voxel field, fixed orientation batch, or retained
incident-by-rod-by-orientation Cartesian product is allowed.

## m = 0 boundary

This work validates and activates only m != 0. A continuous m = 0 component requires a physical
beamstop/support gap or finite-resolution regularization near the direct beam.

- Never silently fall back to the old m = 0 code.
- Report m = 0 as excluded from an m != 0 result.
- Do not apply an arbitrary numerical epsilon.
- Do not claim that m = 0 was validated.

## Non-goals

- Recover complex F, its phase, or Re(F).
- Change source, rod, structure-factor, optics, detector, or deposition equations.
- Add a GPU, backend framework, compatibility facade, or permanent old/new switch.
- Add structure-factor interpolation before exact-L evaluation is proved.
- Retain broad sweeps, generated images, profiling dumps, or adaptive-node tests.

## Dependency order

    T00 baseline
     |-- T01 continuous mosaic density
     \-- T02 nonzero-m coating roots
             |
             v
    T03 joint component density
             |
             v
    T04 adaptive component integration
             |
             v
    T05 continuous sampling
             |
             v
    T06 production cutover
             |
             v
    T07 scientific validation
             |
             v
    T08 delete old runtime modules
             |
             v
    T09-T11 remove legacy contracts and mosaic structures
             |
             v
    T12-T15 update live documents and tasks
             |
             v
    final proof, cleanup, and one coherent commit

## Phase 1: mathematical foundation

### T00: Capture the temporary comparison baseline

Files likely touched:

- No tracked files.
- One temporary external diagnostic and JSON manifest, deleted at handoff.

Intended behavior:

- Run the current scalar enumerator for the fixed 5 degree single-ki Bi2Se3 case.
- Record total, family, branch, rod, detector-valid, and detector-mass results.
- Record the commit SHA and complete configuration.
- Do not alter tracked references.

Tests:

    pytest -q tests/test_mosaic_ewald.py
    python -m rasim_next.proof mosaic-ewald --json

Dependencies: none.

Acceptance:

- The baseline detects total-mass, branch, rod, L, kf, and detector-coordinate changes.
- Repository status remains unchanged.
- Existing untracked strategy and image-script work is preserved.

### T01: Expose the continuous mosaic density

Files likely touched:

- src/rasim_next/sampling/mosaic.py
- tests/test_mosaic_ewald.py

Intended behavior:

- Retain WrappedMosaicParameters.
- Expose folded p_alpha(alpha) = 2*w(alpha) on [0,pi].
- Keep beta measure separate as d beta/(2*pi).
- Add no sin(alpha).
- Keep zero-width components as discrete atoms.
- Temporarily retain old quadrature for comparison.

Tests:

- test_folded_mosaic_density_normalizes_under_declared_measure
- test_zero_width_mosaic_component_remains_discrete

    pytest -q tests/test_mosaic_ewald.py -k folded_mosaic

Dependencies: T00.

Acceptance:

- Continuous probability integrates correctly within 1e-10.
- Gaussian FWHM 2 degrees, Lorentzian 0, eta 0 is represented exactly.
- Existing quadrature remains unchanged until deletion.

### T02: Implement the pure m != 0 coating evaluator

Files likely touched:

- src/rasim_next/reciprocal/coating.py, new.
- tests/test_mosaic_ewald.py

Intended behavior:

- Rotate rod offset and direction with the same accepted rotation.
- Solve both regular nonzero-family roots.
- Order by L and label roots 1 and 2.
- Return L, q, kf, coarea, discriminant, residual, and validity.
- Exclude exact tangencies and no-root points without arbitrary epsilons.
- Implement no m = 0 path.

Tests:

- test_nonzero_coating_roots_match_direct_elastic_equation
- test_nonzero_coating_labels_roots_by_increasing_l
- test_nonzero_coarea_uses_exactly_one_ewald_derivative

Dependencies: T00.

Acceptance:

- Roots and residuals meet frozen reciprocal tolerances.
- Coarea matches the analytic derivative.
- The module imports no ordered, optics, detector, render, or pipeline code.
- No rod-by-orientation matrix is created.

### Checkpoint A

    python -m compileall -q src
    ruff check src/rasim_next/sampling/mosaic.py src/rasim_next/reciprocal/coating.py
    pytest -q tests/test_mosaic_ewald.py

## Phase 2: continuous joint measure

### T03: Evaluate one detector-conditioned component density

Files likely touched:

- src/rasim_next/pipeline/simulate.py
- tests/test_mosaic_ewald.py

Intended behavior:

- Evaluate one incident state, family, branch, and alpha/beta batch.
- Use T02 roots and exact RodQueryBatch rows.
- Call existing ordered strength, outgoing transport, and detector projection.
- Apply source, mosaic, coarea, population, structure, optics, footprint, polarization, support,
  and detector validity exactly once.
- Sum rods as intensities and retain per-rod conditional contributions.

Tests:

- test_component_density_matches_direct_once_only_factor_product
- test_detector_invalid_points_have_zero_component_density
- test_family_density_is_sum_of_distinct_rod_intensities

Dependencies: T01 and T02.

Acceptance:

- Detector validity and optical weight affect the measure before sampling.
- Rods are never summed coherently.
- No family multiplicity is reapplied.
- Factor omission or duplication is detected.

### T04: Integrate component masses adaptively

Files likely touched:

- src/rasim_next/pipeline/simulate.py
- tests/test_mosaic_ewald.py

Intended behavior:

- Integrate each incident/family/branch component without a surface matrix.
- Use conditional beta integration inside the alpha marginal.
- Split support at detected root, tangent, and detector-validity boundaries.
- Return mass, error, evaluation count, and reusable continuous CDF support.
- Keep adaptive nodes private.

Tests:

- test_adaptive_component_mass_matches_separable_analytic_integral
- test_adaptive_component_mass_converges_near_support_boundary
- test_component_error_bound_tightens_under_refinement

Dependencies: T03.

Acceptance:

- Reported error is at most atol plus 1e-8 times component mass.
- Mass is finite and nonnegative.
- Refinement reduces observed error.
- Memory is bounded by component tiles.

### T05: Sample component, orientation, and rod continuously

Files likely touched:

- src/rasim_next/pipeline/simulate.py
- tests/test_mosaic_ewald.py

Intended behavior:

- Draw family/branch from component masses.
- Invert continuous alpha and conditional beta CDFs.
- Draw one rod from exact conditional contributions.
- Recompute exact root, L, q, and kf.
- Assign T_i/N to every draw.
- Key randomness by incident-state and draw identity.

Tests:

- test_continuous_draws_are_not_snapped_to_integration_nodes
- test_seeded_continuous_sampling_is_repeatable
- test_conditional_rod_frequencies_follow_exact_contributions
- test_sampled_event_mass_is_total_mass_over_draw_count

Dependencies: T04.

Acceptance:

- Samples are continuous and reproducible.
- Every draw has one rod and branch 1 or 2.
- Conditional probabilities use exact sampled-L strengths.
- Assigned mass sums to total state mass within floating-point tolerance.

### Checkpoint B

    python -m compileall -q src
    ruff check src/rasim_next/reciprocal/coating.py src/rasim_next/pipeline/simulate.py
    pytest -q tests/test_mosaic_ewald.py

## Phase 3: production cutover

### T06: Replace the ki-to-kf section of simulate_ordered

Files likely touched:

- src/rasim_next/pipeline/simulate.py
- scripts/generate_bi2se3_detector_image.py
- tests/test_integration.py

Intended behavior:

- Accept WrappedMosaicParameters instead of MosaicOrientationBatch.
- Remove alpha-cell and azimuth-cell inputs.
- Invoke continuous sampling immediately after incident states and rods are built.
- Send only sampled exact events through existing outgoing transport.
- Deposit T_i/N through existing bilinear deposition.
- Report m = 0 exclusion.
- Keep old modules callable only by the temporary T07 proof.

Tests:

- test_continuous_simulation_crosses_ki_to_kf_seam
- test_continuous_simulation_conserves_deposited_and_clipped_mass
- test_continuous_simulation_reports_m0_exclusion

Dependencies: T05.

Acceptance:

- Production imports no pipeline.intersections or pipeline.selection.
- Exit transport and deposition equations are unchanged.
- The image script supplies one mosaic distribution.
- Output is explicitly m != 0 only.

### T07: Validate the 5 degree Bi2Se3 fixture

Files likely touched:

- src/rasim_next/reciprocal/proof.py
- tests/test_mosaic_ewald.py
- Temporary external proof files only.

Intended behavior:

- Use one internal ki at 5 degree incidence.
- Use wavelength 1.540592925 angstrom.
- Use Gaussian FWHM 2 degrees, Lorentzian 0, eta 0.
- Build 121 rods from h,k in [-5,5].
- Exclude (0,0) and validate 120 m != 0 rods.
- Compare with the temporary old enumerator and an independent integral.
- Recover the observable S_hk(L) = r_e^2 times abs(F_hk(L)) squared.
- Record equivalent-work wall time and peak memory.

Tests:

- test_one_nonzero_component_round_trips_observable_strength
- Proof-only 120-rod sweep.
- Proof-only mass, family, branch, rod-fraction, detector-coordinate, and Monte Carlo checks.

    pytest -q tests/test_mosaic_ewald.py
    python -m rasim_next.proof mosaic-ewald --json

Dependencies: T06.

Acceptance:

- Every rod/root is RECOVERED, NO_SUPPORT, or TANGENT_BOUNDARY.
- Strengths and residuals meet frozen tolerances.
- Total, family, and branch masses converge.
- Seeded frequencies and moments lie within four expected standard errors.
- No numerical gate is applied to m = 0.

### Checkpoint C: deletion authorization

The old implementation may be deleted only after T07 passes completely.

## Phase 4: delete obsolete machinery

### T08: Delete discrete intersection and candidate-pool modules

Files deleted:

- src/rasim_next/reciprocal/ewald.py
- src/rasim_next/reciprocal/events.py
- src/rasim_next/pipeline/intersections.py
- src/rasim_next/pipeline/selection.py

Intended behavior:

- Remove all imports before deletion.
- Retain no fallback, facade, registry, or feature flag.

Tests:

    python -m compileall -q src
    pytest -q
    rg "reciprocal.ewald|reciprocal.events|pipeline.intersections|pipeline.selection" src tests scripts

Dependencies: T07.

Acceptance:

- All four files are absent.
- The search finds no runtime or test imports.
- Full tests pass without the old oracle.

### T09: Remove legacy fields from ScatteringEventBatch

Files likely touched:

- src/rasim_next/core/contracts.py
- src/rasim_next/pipeline/simulate.py
- src/rasim_next/proof/core.py
- tests/test_core_coordinates.py

Intended behavior:

- Remove orientation_id and reciprocal_weight.
- Add mosaic_alpha_rad, mosaic_azimuth_rad, and intersection_branch_id.
- Keep assigned sampled mass outside the event geometry contract.
- Remove candidate_row alignment.

Tests:

- test_scattering_event_contract_carries_continuous_mosaic_identity
- test_sampled_event_mass_is_not_a_reapplicable_event_weight

Dependencies: T08.

Acceptance:

- Removed fields no longer exist.
- Branch values are restricted to 0, 1, or 2.
- No placeholders preserve obsolete semantics.
- Core proof passes.

### T10: Adapt geometry fixtures

Files likely touched:

- src/rasim_next/geometry/proof.py
- tests/test_geometry_optics.py

Intended behavior:

- Construct the revised event contract in geometry fixtures.
- Geometry continues consuming only kf, wavelength, incident identity, and validity.
- Change no geometry or optics equation.

Tests:

    pytest -q tests/test_geometry_optics.py
    python -m rasim_next.proof geometry-optics --json

Dependencies: T09.

Acceptance:

- Existing geometry observables remain within frozen tolerances.
- Geometry does not depend on mosaic coordinates or branch.
- Event identity and status remain aligned.

### T11: Delete discrete mosaic-orientation construction

Files likely touched:

- src/rasim_next/sampling/mosaic.py
- tests/test_mosaic_ewald.py

Intended behavior:

- Delete MosaicOrientationBatch.
- Delete manuscript_axisymmetric_v1_orientation_quadrature.
- Delete their panel, azimuth-node, rotation-batch, and orientation-ID helpers.
- Retain only parameters and continuous probability density.

Tests:

    pytest -q tests/test_mosaic_ewald.py
    rg "MosaicOrientationBatch|manuscript_axisymmetric_v1_orientation_quadrature|alpha_cell_count|azimuth_cell_count" src tests scripts

Dependencies: T10.

Acceptance:

- The search returns no matches.
- Mosaic normalization and zero-width behavior remain proven.
- No fixed orientation array remains.

### Checkpoint D

    python -m compileall -q src
    ruff check src tests scripts
    pytest -q
    python -m rasim_next.proof core --json
    python -m rasim_next.proof geometry-optics --json
    python -m rasim_next.proof mosaic-ewald --json

## Phase 5: remove live specification residue

### T12: Update architecture and contracts

Files likely touched:

- docs/CONTRACTS.md
- docs/RESULT_MEASURE.md
- docs/DOVETAIL_MATRIX.md
- docs/ARCHITECTURE.md
- docs/TRACE_SCHEMA.md

Intended behavior:

- Replace discrete candidate-pool language with continuous detector-conditioned sampling.
- Document continuous event fields and equal T/N mass.
- Assign root/coating ownership to reciprocal/coating.py.
- Remove or rename obsolete trace stages.

Tests:

- Search documents for obsolete symbols.
- Compare the documented factor ledger with production code.

Dependencies: T11.

Acceptance:

- Live documents describe one implementation.
- Every once-only factor has one owner.
- No document instructs callers to build an orientation batch or candidate pool.

### T13: Update strategy and validation

Files likely touched:

- docs/CONTINUOUS_EWAL_COATING_STRATEGY.md
- docs/VALIDATION.md
- docs/PHYSICS_LEDGER.md
- docs/DECISIONS.md
- docs/SCOPE_AND_PHASES.md

Intended behavior:

- Mark continuous coating implemented for m != 0.
- Keep m = 0 deferred pending physical regularization.
- Replace old equation-authority paths with reciprocal/coating.py.
- Record proof results and deletion classification.

Tests:

- Cross-document search for module names, branches, factor ownership, and m = 0 status.

Dependencies: T12.

Acceptance:

- Documents agree on observable, measure, identities, and support.
- No document claims complex F recovery.
- No document claims m = 0 validation.

### T14: Delete retired T03 instructions

Files likely touched:

- tasks/03_mosaic_ewald.md, delete.
- tasks/prompts/mosaic_ewald.md, delete.
- tasks/index.yaml
- tasks/06_parallel_review.md
- tasks/07_integration.md

Intended behavior:

- Remove obsolete implementation instructions and branch entries.
- Update review and integration tasks for continuous sampling.

Tests:

- Inspect tasks/index.yaml.
- Search live task instructions for deleted APIs.

Dependencies: T13.

Acceptance:

- No active task directs work toward the retired branch or APIs.
- Historical archive prose may retain provenance but no executable instruction does.

### T15: Update active simulation and fitting plans

Files likely touched:

- tasks/parallel_simulation_geometry_fitting_plan.md
- tasks/mosaic_distribution_fitting_plan.md
- tasks/OVERNIGHT_RUNBOOK.md
- tasks/02_geometry_optics.md

Intended behavior:

- Replace finite candidate-pool assumptions with reusable continuous component masses and CDFs.
- Preserve incident-ray block parallelization.
- Record invalidation:
  - geometry changes invalidate roots and hits;
  - mosaic changes invalidate component measures;
  - intensity changes invalidate strengths, masses, and CDFs.
- Remove fixed orientation-node and finite-candidate-order requirements.

Tests:

- Search for obsolete APIs and finite-pool assumptions.
- Review the invalidation graph against pipeline/simulate.py.

Dependencies: T14.

Acceptance:

- Future parallelization consumes the continuous seam without changing downstream geometry.
- No active plan requires a candidate-by-orientation matrix.
- No acceleration dependency or backend abstraction is introduced.

## Final verification

    python -m compileall -q src
    ruff check src tests scripts
    pytest -q
    python -m rasim_next.proof core --json
    python -m rasim_next.proof geometry-optics --json
    python -m rasim_next.proof mosaic-ewald --json
    python -m rasim_next.proof ordered-reflectivity --json
    git diff --check
    rg "MosaicOrientationBatch|manuscript_axisymmetric_v1_orientation_quadrature|build_scattering_events|solve_continuous_rod_ewald|IntersectionSupport|CandidatePool|SelectedCandidateBatch|alpha_cell_count|azimuth_cell_count" src tests scripts docs tasks

Final acceptance:

- The legacy-symbol search returns no matches from live files.
- reciprocal/coating.py is the only new production module.
- Four obsolete runtime modules and two obsolete task files are deleted.
- Temporary comparisons, broad sweeps, diagnostics, benchmarks, and generated images are absent.
- Every retained test protects a unique scientific invariant or integration boundary.
- Optimized and proof paths agree within frozen tolerances.
- Handoff reports proof state, benchmark, peak memory, m = 0 limitation, retained tests, and
  deletion inventory.
- Git status contains only intended changes before one coherent commit and is clean afterward.

## Execution policy

- T00 through T11 are sequential.
- T12 through T15 may receive parallel read-only review after contracts freeze.
- The main workbranch is the only writer.
- Do not commit an intermediate compatibility path.
- Do not delete the old enumerator until T07 passes.
- Do not retain temporary tests, scripts, figures, or profiling output.
