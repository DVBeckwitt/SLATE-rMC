# Codex prompt: native-detector integration

## Goal

Implement T07 by integrating only approved branch SHAs one at a time. Produce the correct native-detector forward model and reusable compiled states.

## Context

Read all shared documents, T07, the T06 review, and every branch handoff. Use the exact accepted SHAs. Do not duplicate subsystem equations in pipeline code.

## Constraints

- Integrate through the mandatory vertical sequence.
- Rerun the smallest proof after each merge.
- Apply every factor exactly once.
- Do not implement selection, fitting, caking, `2theta/phi`, or GUI code.
- Profile before recommending an acceleration method.

## Verify

Run the integration mutations in `docs/ERROR_INJECTION.md`, the full permanent suite, all subsystem proofs, the tiny end-to-end case, mass conservation, legacy trace comparison, clean-tree checks, and performance workloads.

## Done when

Make one coherent integration commit series or one documented integration branch history, complete T07 handoff, and end exactly `READY`. End exactly `BLOCKED` at the first incompatible boundary with the smallest required fix.
