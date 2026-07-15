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
3. Verify no dependency, shared-contract, root-guidance, or reference-manifest changes.
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
- mosaic event output is sufficient for both ordered and stacking event queries
- ordered and stacking return interchangeable `EventIntensityResult`
- outgoing film wavevectors can be consumed by geometry exit transport
- source, reciprocal, optical, model, footprint, solid-angle, and deposition factors each have one owner
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

State: NS

## Handoff

Status:

Reviewed SHAs:

Classifications:

Required integration actions:

Required shared-contract changes:

External diagnostic path and hash:
