# Codex prompt: read-only parallel review

## Goal

Review the four physics branches against `PROOF_BASE_SHA`, the immutable reference pack, shared contracts, and T06. Do not merge or modify physics source.

## Inputs

Provide the four branch SHAs and handoffs. Verify the legacy-pack SHA-256 before review.

## Work

Use read-only subagents when useful to review each branch or one cross-branch concern. Rerun every proof and its required negative controls. Inspect oracles, trace metadata, first divergences, convergence, path ownership, and factor ownership. Test composability using the synthetic contract path or temporary review-only checkouts without committing integration code.

## Output

Classify each branch `APPROVE`, `APPROVE_WITH_INTEGRATION_ACTIONS`, or `BLOCK`. State exact required actions and minimum shared changes. Write at most one external review diagnostic. Do not merge.

## Done when

Complete the T06 handoff and end exactly `READY`. End exactly `BLOCKED` if required branch evidence or the immutable pack is unavailable.
