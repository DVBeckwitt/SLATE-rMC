# Codex prompt: native-detector integration

## Goal

Implement T07 from approved local branch SHAs. Produce the seeded statistical raw native-detector image and reusable fixed-geometry states.

## Context

Read all shared documents, T07, the T06 review, and every branch handoff. Use the exact accepted SHAs. Do not duplicate subsystem equations in pipeline code.

## Constraints

- Integrate through the mandatory vertical sequence.
- Rerun the smallest proof after each merge.
- Integrate the one T03 source sampler; for each ray+phase stream one complete pool across every individual rod and valid mosaic/Q solution, never one pool per reflection.
- Add one seeded inverse-CDF selector (configurable count, legacy default 50) and one depositor; preserve each selected candidate's own `Q`, `kf`, rod, and hit.
- Source/mosaic probability acts through frequency only; selected events receive equal `total_pool_mass / selected_count` and are never candidate-weighted again.
- Raw pixels exclude solid angle; keep it metadata-only and report bilinear edge clipping.
- Do not implement PSF, fitting, caking, `2theta/phi`, GUI, backend, generic RNG, or proof frameworks.
- Profile before recommending an acceleration method.

## Verify

Run exactly the eight T07 proof obligations in the task, the compact permanent suite, subsystem proofs, clean-tree checks, and fitting-relevant performance workloads. Keep broad sweeps external.

## Done when

Make one coherent integration commit series or one documented integration branch history, complete T07 handoff, and end exactly `READY`. End exactly `BLOCKED` at the first incompatible boundary with the smallest required fix.
