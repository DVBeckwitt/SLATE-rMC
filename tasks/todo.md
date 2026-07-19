# Beam-to-`ki` Audit Follow-up Checklist

Status: **PROPOSED — NO IMPLEMENTATION STARTS BEFORE HUMAN APPROVAL**

Detailed behavior, file ownership, tests, and acceptance criteria are in
[the implementation plan](plan.md). Completed BKI remediation history remains in commit
`d5eed25` and `docs/VALIDATION.md`.

## Phase 0: plan ownership

- [ ] **PLAN-01** — Reconcile deterministic/parallel ownership, serial incident compilation,
      parent-row identity, and nominal source/tag mass semantics. Dependencies: none.
- [ ] **PLAN-02** — Add full-source wavelength coverage, fixed source reference plane, and
      unbounded-plane translation rules to T09-T11. Dependencies: PLAN-01.

### Checkpoint P

- [ ] Docs, links, and focused stale-claim scans pass.
- [ ] Human approves execution and fitting ownership.
- [ ] No production file has changed.

## Phase 1: material authority

- [ ] **MAT-01** — Make `MaterialOptics` own a v2 revision, reject inconsistent
      `n/delta/beta/mu`, and remove transport-time hashing. Dependencies: Checkpoint P.
- [ ] **MAT-02** — Delete stored/constructor `delta`, `beta`, and `mu_Ainv`; retain
      `n_complex` as the sole authority. Dependencies: MAT-01.

## Phase 2: sample entrance authority

- [ ] **GEO-01** — Make `CompiledInstrument` own the v2 sample-entrance revision; canonicalize
      unbounded support to orientation plus signed normal offset. Dependencies: MAT-02.
- [ ] **GEO-02** — After a fresh consumer scan, delete stored `lab_from_goniometer` and
      `lab_from_crystal`, retain the ordered local compilation step, and narrow the angle
      fingerprint to detector-causal inputs. Dependencies: GEO-01.

### Checkpoint K0

- [ ] Focused material/geometry/reciprocal/integration tests and all scientific proofs pass.
- [ ] Numeric source-to-`ki` fields remain accepted; only named revision digests rebaseline.
- [ ] Consumer and stale-symbol scans confirm the intended deletion set.

- [ ] **SYNC-01** — Synchronize contracts, architecture, decisions, dovetail, and trace documents.
      Dependencies: GEO-02 and Checkpoint K0.
- [ ] **SYNC-02** — Record compact validation, error-injection, performance, and task evidence.
      Dependencies: SYNC-01 and Checkpoint K0.
- [ ] **MANIFEST-01** — Refresh `FILE_MANIFEST.json` once and run the seed verifier.
      Dependencies: SYNC-02.

### Checkpoint K

- [ ] Focused material/geometry/integration tests and all scientific proofs pass.
- [ ] Source-to-`ki` numeric results remain accepted; only named API/revision digests rebaseline.
- [ ] Deleted-field, dead-transform, transport-hash, raw-source-rejoin, and duplicate-equation scans
      are clean.
- [ ] Human approves the shared contract before downstream records freeze.

## Phase 3: downstream owners

- [ ] **PAR-01** — Implement private `parent_row_index` partitions and canonical scatter/merge;
      never sort rows by state ID or hash public slices. Dependencies: Checkpoint K and accepted
      staged numeric boundary.
- [ ] **FIT-01** — Require material coverage for all unique parent-source wavelengths before any
      geometry objective evaluation. Dependencies: Checkpoint K and T09 base contracts.
- [ ] **SRC-01** — Freeze a physical source reference plane before position-direction correlations.
      Dependencies: FIT-01 and T10 shared-path review.
- [ ] **GEO-FIT-01** — Expose only signed normal translation for unbounded support; reject tangent
      coordinates. Dependencies: GEO-01, FIT-01, accepted T10.
- [ ] **NOM-01** — Implement DP-00C's sole nominal incident fixture with fixed seed, exact
      means/material, `source_weight == 1`, and a massless tag contract. Dependencies:
      Checkpoint K and deterministic Checkpoint 0/DP-00A; precedes DP-01 and DP-06.
- [ ] **PERF-01** — Profile the incident seam; record `NO_CHANGE` unless the optional four-file
      allocation cleanup is measurably justified. Dependencies: Checkpoint K; nonblocking.

### Checkpoint D: parallel and nominal

- [ ] Scalar, packed, and worker layouts agree in parent order for nonmonotonic IDs.
- [ ] Nominal tags cannot affect detector pixels or any photon-mass ledger.
- [ ] No worker RNG, slice hash, public packet batch, ID-sort merge, or duplicate tag physics exists.

### Checkpoint F: fitting

- [ ] A baseline-invalid wavelength can become valid without material lookup failure.
- [ ] Source correlation is defined at one physical plane and has a full-rank accepted pack.
- [ ] Unbounded tangent translations are rejected; finite translations activate only from
      edge-sensitive observations.
- [ ] Detector-only changes do not invalidate incident material or `ki`.

## Final branch gate

- [ ] Every completed task meets its detailed acceptance criteria.
- [ ] Retained tests each protect a unique long-term invariant.
- [ ] Compileall, Ruff lint/format, docs/links, compact full tests, all proof commands, manifest,
      and assigned mutations pass.
- [ ] `examples/` and `reference/` are unchanged.
- [ ] No temporary test, benchmark, diagnostic, cache, generated file, compatibility shim, or
      unused dependency remains.
- [ ] Each implementation worktree ends in one coherent commit and a clean status.
