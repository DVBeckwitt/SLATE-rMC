# Overnight four-worktree runbook

The overnight run begins only after T00 and T01 have completed and `PROOF_BASE_SHA` is recorded. The objective is four trustworthy, composable reference subsystems for the manuscript cases. It is not four production accelerators or complete generalized libraries.

## Preflight in the main checkout

```bash
test -n "$PROOF_BASE_SHA"
uv sync --frozen --group dev
python scripts/verify_seed.py
python -m rasim_next.proof references --json
python -m rasim_next.proof core --json
git status --short
```

Require a clean tree and matching tracked manifests. No original-repository or manuscript path
is required. Use a separate `.venv` per worktree and a shared external `UV_CACHE_DIR`.

Recommended process environment:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
XDG_CACHE_HOME=<external cache root>
PYTHONPYCACHEPREFIX=<external cache root>/pycache
```

Raise thread counts only for a measured branch-local benchmark.

## Launch

Create four worktrees from the exact same `PROOF_BASE_SHA`:

```text
feat/geometry-optics
feat/mosaic-ewald
feat/ordered-reflectivity
feat/stacking-transition
```

Launch one Codex task per worktree with the matching prompt. The main agent is the sole writer in each worktree. Read-only subagents may inspect source, derive equations, or challenge proofs.

## Required overnight result

Each branch must complete its mandatory reference slice:

### T02 geometry and optics

- OSC reader and canonical mapping
- rigid instrument transforms
- sample/footprint and detector intersections
- single-interface entrance/exit refraction
- scalar field amplitudes and uniform-depth attenuation
- individual and batched trace output

### T03 mosaic and Ewald

- fixed-seed randomized equal-mass source and wavelength samples
- normalized wrapped Gaussian-plus-Lorentzian orientation model
- transparent dense event oracle
- deterministic/adaptive valid-candidate support and physical reciprocal candidate mass; selection and rasterization belong to T07
- event contract and convergence evidence

A faster localized/adaptive solver is optional overnight. A dense correct implementation is acceptable as the reference authority.

### T04 ordered and reflectivity

- validated manuscript-material CIF path or equivalent explicit reference structures
- raw complex atomic/unit-cell/layer amplitudes
- complete individual rods and exact family metadata
- finite ordered stack
- material optics
- pure Parratt, pure kinematic, and named composite outputs

Full support for every CIF edge case is optional overnight. Unsupported crystallographic metadata must fail explicitly rather than be ignored.

### T05 stacking transition

- typed state/phase conventions
- direct short-sequence and finite-pair oracles
- full six-state and exact reduced calculations
- finite intensity and manuscript parent/fault parameterization cases
- explicit normalization

## Branch completion evidence

Each branch produces:

- one coherent commit
- completed task plan and handoff
- proof JSON on standard output
- legacy classifications and first divergences
- convergence result
- equivalent-work wall time and peak memory
- explicit extension backlog
- final response ending `READY` or `BLOCKED`

No branch merges overnight. No branch regenerates legacy evidence. No branch edits shared contracts, dependencies, root documents, or another branch's paths.

## Diagnostics

Persistent diagnostics are normally disabled. A branch may use external temporary scratch during proof, but it must remove it before handoff. When retained evidence is necessary, T06 consolidates selected arrays from all branches into exactly one external `overnight_proof.ra_diag.npz` with one JSON manifest, then deletes all fragments. The repository and worktrees remain free of diagnostics.

## Morning review

Run the read-only review prompt with exact branch SHAs. T06 checks:

- ancestry and owned paths
- reproducible proof commands
- independent oracles
- reference-pack hash and classifications
- convergence and error-injection sensitivity
- contract and factor compatibility
- extension backlogs that block or do not block the manuscript vertical slice

A branch may be `APPROVE_WITH_INTEGRATION_ACTIONS` when its mandatory slice is correct and its generalization backlog does not affect the reference materials or tiny end-to-end case.

## Failure handling

A branch stops `BLOCKED` for a missing shared field, unresolved equation/convention, unavailable required reference, or failed independent proof. It does not weaken tolerances, alter the reference pack, add a silent approximation, or edit shared contracts to force completion.
