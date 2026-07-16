# Recovery plan: statistical native-detector integration

## Goal

Reconcile T02–T05 on the corrected raw-image measure, merge them locally, implement minimal T07, match authoritative high-sample raw images, and demonstrate speed suitable for later fitting.

## Authority and decisions

- Authoritative implementation folder: `C:\Users\Kenpo\Downloads\New folder\ra_sim`.
- Production source rays are seeded randomized Latin-hypercube Gaussian samples with independent dimensions, antithetic pairs, an odd central ray, and equal empirical mass.
- T03 owns source sampling and reciprocal candidate construction; T07 owns physical-factor assembly, weighted selection, equal selected-event mass, deposition, and end-image statistical proofs.
- Raw native images exclude pixel solid angle; deterministic Gauss–Hermite and all-candidate raster paths are oracle/diagnostic only.

## Dependency order

1. PM approves this plan and the exact eight T07 proofs.
2. Make one reviewed follow-up correction to the superseded result measure, contracts, factor ledger, and names.
3. Recover T03 onto that base: equal-mass seeded source samples and physical reciprocal candidates/masses, with no selection or pixels.
4. Reconcile T02, T04, and T05 to the corrected shared base; rerun T06 and merge approved SHAs locally in dependency order.
5. Implement T07 with one integrated source sampler, one complete streaming candidate pool per ray+phase across all rods/solutions, one weighted selector, and one conservative bilinear depositor.
6. Match authoritative high-sample raw images through the named stage and profile full rendering plus fixed-geometry reevaluation.

## Acceptance checkpoints

- Measure: no sampled source/mosaic probability is multiplied again, selected events receive `total_candidate_mass / event_count`, and solid-angle metadata cannot affect raw pixels.
- Merge: each T02–T05 proof passes on the corrected base and T06 records exact approved SHAs and factor ownership.
- Integration: all eight required proofs pass, edge clipping is explicit, authoritative high-sample agreement is recorded, and fitting-relevant timings are reported.

## Boundaries

- No new framework, PSF, caking, fitting, backend, generic RNG layer, or proof infrastructure.
- Broad sweeps remain external artifacts; permanent tests protect only distinct estimator, measure, and integration failures.
