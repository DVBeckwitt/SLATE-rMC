# T06: read-only parallel review

Branch: `review/parallel-proof`

## Goal

Determine whether each physics branch is correct, path-compliant, independently proved, and composable under the common contracts. Do not merge or modify physics source.

## Inputs

```text
PROOF_BASE_SHA
four branch SHAs
four task handoffs
four proof JSON outputs
immutable legacy-pack path and SHA-256
```

## Per-branch review

1. Verify ancestry from `PROOF_BASE_SHA`.
2. Verify only owned paths changed.
3. Verify no unreviewed dependency, shared-contract, root-guidance, or reference-manifest changes.
4. Rerun the exact proof and permanent test commands.
5. Unset original source and manuscript variables, then rerun permanent tests.
6. Inspect the independent oracle and ensure it is not a wrapper around the public calculation.
7. Verify stage metadata, units, frames, measures, model versions, and pack hash.
8. Verify every legacy case is classified and every correction has a first divergence.
9. Verify convergence, equivalent-work benchmark, and peak memory.
10. Verify no generated or diagnostic files exist inside the repository.
11. Separate extension backlog from any missing manuscript/reference requirement.

## Cross-branch review

Check:

- shared complex wave and scalar interface functions are used, not copied
- material optics output matches geometry input
- real phase wavevectors and decay components are not mixed
- rod IDs, phase IDs, family IDs, `Qz`, `L`, and wavelength semantics align
- T03 source samples have equal empirical mass and its candidates suffice for ordered and stacking queries
- ordered and stacking return compatible physical intensity components under the corrected result measure
- outgoing film wavevectors can be consumed by geometry exit transport
- source/mosaic probability acts only through frequency; physical factors act once; raw pixels exclude solid angle
- T07 alone owns weighted selection, equal selected-event mass, clipping reports, and deposition
- no branch assumes a square detector
- no branch assigns final branch or `Qr` selections prematurely

## Output

Classify each branch:

```text
APPROVE
APPROVE_WITH_INTEGRATION_ACTIONS
BLOCK
```

List exact SHAs, evidence, failed cases, extension backlogs, and minimum contract actions. Shared changes are separate reviewed commits before integration.

If persistent evidence is needed, write exactly one external `parallel_review.ra_diag.npz` and no sidecars.

## Execution plan

State: READY

Evidence: the reviewed candidate commits below are exact ancestors of clean integrated `main` at
`caf7acd649a27dc66c6c0b73a2f66dcd520389f9`. The shared sampled-result correction is
`7cc0620fa826ed1b17e9f589991a3cf77e161374`.

Commands run: branch-specific Ruff, focused tests, subsystem proof commands, full pytest, ancestry,
owned-path, and clean-tree checks were completed during recovery. The post-merge external locked
environment passed Ruff, 29/29 permanent tests, and the core proof.

Remaining work: T07 only. T02--T05 worktrees, local feature branches, and pinned tasks are retired.

Contract or dependency issue: none blocks the planned one-phase/one-component T07 vertical slice.

## Handoff

Status: READY

Reviewed SHAs:

```text
T02 geometry/optics        8536008f61cede815d0cfe6b77b3dc844b8b6706
T03 mosaic/Ewald           f6ba4c44b223122ee9fa04e39eb0bc935864ee4a
T04 ordered/reflectivity   50713f9b0e9c649b6e98e38088b604826255e548
T05 stacking/transition    3bbc2ca2913e32004ebf4b75470923e6363ce740
```

Classifications: APPROVE after the shared sampled-result correction and branch-local contract
adoptions.

Required integration actions: preserve complete source batches; form one all-rod candidate pool per
incident ray and caller phase/component; add only weighted selection, equal `T/N_event`, and
conservative raw deposition.

Required shared-contract changes: documentation clarifications only; no contract API or trace-schema
version change is required by the MVP.

External diagnostic path and hash: none retained.
