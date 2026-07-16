# T16 checklist: minimum cohesive hBN ring fitter

Status: PLANNED, `BLOCKED` at Checkpoint A pending accepted T10. The detailed behavior, files,
tests, dependencies, and acceptance criteria are in `tasks/hbn_ring_fitter_plan.md`.

Shared `tasks/todo.md` remains generic; this checklist owns T16 execution state.

## Prerequisites

- [x] Human approves `docs/HBN_RING_FITTER_MINIMUM_SPEC.md` and the T16 plan.
- [ ] T07 and T09 are accepted.
- [ ] T10 detector calibration, result writer/reader, predicted ring support, and proof are accepted.
- [ ] Clean `feat/hbn-ring-fitter` is created from approved main, not from the copied branch.

## Documentation gate

- [ ] Task 1: publish the optional-tool boundary.
- [ ] Checkpoint A: documentation, T10, branch, and clean-tree preconditions pass.

## Headless frontend

- [ ] Task 2: define the fixed hBN ring model.
- [ ] Task 3: prepare one canonical hBN image.
- [ ] Task 4: add deterministic observation state.
- [ ] Task 5: adapt hBN points to T10.
- [ ] Checkpoint B: two focused tests, T10 proof, import boundary, and scientific review pass.

## Optional GUI

- [ ] Task 6: add one optional launcher.
- [ ] Task 7: show the minimal image workspace.
- [ ] Task 8: wire precision placement.
- [ ] Checkpoint C: observation workflow, latency, test-count, and LoC trajectory pass.
- [ ] Task 9: connect the T10 solve.
- [ ] Task 10: render and export the accepted result.
- [ ] Checkpoint D: complete user workflow and human acceptance pass.

## Handoff

- [ ] Task 11: document only the shipped surface.
- [ ] Task 12: run the final proof and footprint gate; reopen owning tasks for any edits.
- [ ] Exactly two production modules, at most 800 hBN Python lines, two permanent hBN tests, and one optional dependency remain.
- [ ] Full tests, T10 proof, docs, build, CLI help, manual smoke, injections, benchmark, memory, and clean-tree gates pass.
- [ ] One coherent commit and complete handoff are recorded.
