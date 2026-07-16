# Project instructions

## Objective

Build the smallest cohesive numerical core that produces scientifically correct detector-native
results and supports staged geometry, mosaic, ordered-intensity, and stacking-disorder fitting
without duplicating physics.

Correct declared outputs come first. Subject to correctness, the code must be Pythonic,
lightweight, optimized, and free of development residue.

## Non-negotiable code qualities

### 1. Pythonic

- Write clear, conventional Python that another scientist can read without reverse engineering.
- Prefer small pure functions, explicit data flow, typed dataclasses, narrow protocols, `pathlib`,
  context managers, and informative exceptions.
- Use NumPy idioms for array work. Keep scalar equations readable and batch them without changing
  their meaning.
- Prefer composition over inheritance. Avoid metaprogramming, dynamic registries, magic dispatch,
  clever decorators, hidden mutation, and unnecessary class hierarchies.
- Use explicit names that include frame, unit, measure, or ordering where ambiguity is possible.
- Do not hide scientific state in closures, singletons, module globals, or implicit object mutation.

### 2. Lightweight

- Implement the smallest API and the fewest modules needed for the accepted result.
- Add a dependency only when it removes more code and risk than it introduces. Record why it is
  needed. Do not add frameworks for one feature.
- Do not add plugin systems, generic backend layers, service containers, registries, compatibility
  facades, serialization frameworks, or abstraction layers without a demonstrated repeated need.
- Avoid one-line wrapper functions and classes that only rename another object.
- Keep optional functionality out of the import path and out of the core dependency set.
- One equation, convention, or transformation has one authoritative implementation.

### 3. Optimized

- Optimize the mathematical work before optimizing syntax or choosing a processor.
- Avoid unnecessary searches, repeated transforms, repeated structure calculations, redundant
  interpolation, and materialization of large Cartesian products.
- Batch and vectorize regular work, minimize allocations and copies, use contiguous arrays where
  useful, and reuse immutable compiled state.
- Keep a transparent proof path. An optimized path must reproduce the accepted observable within
  the frozen tolerance and must be benchmarked against equivalent work.
- Profile before adding specialized acceleration. Optimize measured bottlenecks, not presumed ones.
- Do not sacrifice deterministic proof, numerical stability, or debuggability for an unmeasured
  micro-optimization.

### 4. No bloat or leftovers

- Production modules contain no embedded tests, `__main__` demos, scratch harnesses, debug prints,
  diagnostic writers, commented-out alternatives, abandoned implementations, or generated output.
- Before handoff, delete temporary scripts, exploratory notebooks, one-off fixtures, duplicate
  helpers, obsolete adapters, benchmark dumps, and tests that no longer protect a unique behavior.
- Permanent tests are retained only when they protect a distinct scientific invariant, public
  contract, accepted legacy comparison, or end-to-end integration boundary.
- Do not keep tests of private implementation details, duplicate parameterizations of the same
  equation, or large snapshot collections. Replace obsolete tests when behavior is replaced.
- Temporary tests may be used while developing, but they must be removed before the branch moves on
  unless they meet the permanent-test rule above.
- Every committed file must be required by the production package, the compact permanent proof
  suite, the tracked examples, or the project documentation.
- No unresolved `TODO`, `FIXME`, temporary feature flag, dead branch, or unused dependency may
  remain in touched code at handoff.

"No leftover tests" does not mean removing the minimum permanent regression suite. It means removing
exploratory, redundant, obsolete, and implementation-detail tests once their purpose is complete.

## Read before editing

Read `docs/SCOPE_AND_PHASES.md`, `docs/CONVENTIONS.md`, `docs/OSC_COORDINATES.md`,
`docs/RESULT_MEASURE.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`,
`docs/DOVETAIL_MATRIX.md`, the assigned task, its referenced rows in `docs/PHYSICS_LEDGER.md`,
`docs/TRACE_SCHEMA.md`, `docs/VALIDATION.md`, `docs/ERROR_INJECTION.md`, `docs/EXAMPLES.md`, and
`reference/README.md`.

For Codex worktrees, also read `WORKTREE_LAUNCH.md`, `docs/CODEX_EXECUTION.md`,
`tasks/OVERNIGHT_RUNBOOK.md`, and the assigned prompt.

## Numerical authority

Use this order when evidence disagrees:

1. analytic identities, conservation laws, direct enumeration, and independently converged results
2. manuscript equations after assumptions, measure, and conventions are explicit
3. the immutable tracked reference pack and example references
4. original-RASIM final images

Classify legacy comparisons as `MATCH`, `CORRECTED`, or `NO_ORACLE`. A corrected case must agree
through a named first divergent stage, then follow an independent oracle. A final image is never
sufficient proof.

## Working philosophy

- Prefer the smallest correct implementation and one public implementation of each equation.
- Keep an independent oracle only where it can detect a plausible mistake.
- Optimize work count, memory traffic, conditioning, and reuse before choosing CPU or GPU.
- Use `float64` and `complex128` for proof unless final-observable error is explicitly bounded.
- Make units, frames, probability measure, normalization, coherence, shape, and validity explicit.
- Keep models immutable, modules narrow, imports one-way, and public APIs small.
- No mutable globals, hidden caches, import-time computation, or import-time device setup.
- No hidden normalization, reflection pruning, fabricated reflections, sentinel overloading, or
  silent fallbacks.
- Never import or execute the tracked original-RASIM snapshot from production code or permanent
  tests.
- Never copy legacy modules wholesale. Reimplement the selected equations behind the new contracts.
- Prefer deleting unnecessary code over maintaining it.

## Coordinates and OSC data

- Radians internally.
- Instrument positions in metres.
- Wavelengths and crystal lengths in angstroms.
- Wavevectors in inverse angstroms.
- Column vectors and active rotations.
- Continuous detector coordinates are `(column_px, row_px)`.
- Arrays are indexed `[row, column]`.
- OSC raw indices, detector-native indices, and continuous coordinates are distinct types.
- Convert OSC orientation exactly once at the I/O boundary using the accepted clockwise mapping.
- No downstream rotation, flip, or transpose option is permitted.
- Never alter one coordinate after a rigid transform.
- Legacy `x`/`y` names in the supplied state are provenance only: legacy `x` was native row and
  legacy `y` was native column.

## First validated physics model

- Conserve tangential wavevector at planar interfaces.
- Use one shared complex-normal-wavevector branch selector in refraction, attenuation, and Parratt.
- Use scalar field amplitude `t12 = 2*k1z/(k1z+k2z)` and `|t_in*t_out|^2`.
- Do not implement the old 50/50 s/p power-transmittance average.
- Use one transmitted incident channel, one transmitted exit channel, and the manuscript
  uniform-depth attenuation average for the first off-specular model.
- Keep Parratt, kinematic, and named composite specular outputs separate.
- Use real phase wavevectors for elastic geometry and imaginary normal components for decay.
- Integrate mosaic probability with its declared spherical measure.
- Preserve raw complex amplitudes and every individual `(h,k)` rod.
- Treat Qr as family metadata, not rod identity.
- Sum coherent contributions as amplitudes, and independent source/wavelength/mosaic/phase/
  parent contributions as intensities.
- Apply each applicable source, event, optical, polarization, and deposition factor exactly once.
  Detector solid angle is excluded from the raw image and applies only in an explicitly requested
  later analysis correction.

## Worktree isolation

T02--T05 were created from one clean `PROOF_BASE_SHA`, reviewed, merged into `main` at
`caf7acd649a27dc66c6c0b73a2f66dcd520389f9`, and retired. Do not recreate or resume those
worktrees or feature branches. New write-heavy phases start from the approved current `main` in
their own worktree and retain the same owned-path and read-only-contract discipline.

The main agent is the only writer. Subagents may be used for bounded read-heavy exploration,
derivation challenge, test review, or log analysis. Stop `BLOCKED` when a shared contract is
insufficient rather than editing protected files or weakening proof.

## Proof budget

Every branch provides analytic or invariant proof, an independent oracle where specified, tracked
reference comparison, first-divergence evidence for corrections, convergence, wall time, peak
memory, and assigned error-injection detection.

Keep the permanent suite compact. Large sweeps, full images, profiling runs, generated diagnostics,
and legacy traces are proof artifacts, not permanent tests. Before handoff, review every test added
by the branch and retain only those that protect a unique long-term failure mode.

## Diagnostics

Diagnostics are disabled by default. No diagnostic output may be written under the repository
root. A retained diagnostic is exactly one external `.ra_diag.npz` containing numeric arrays and
one JSON manifest. No sidecars or diagnostic directories.

## Handoff gate

Before committing the branch result:

- Remove temporary tests, scratch files, debug output, dead code, commented alternatives, unused
  imports, unused dependencies, and generated files.
- Confirm every retained test protects a unique long-term invariant or interface.
- Run formatting, linting, type checks where configured, the compact permanent test suite, assigned
  proof commands, and the branch benchmark.
- Confirm the optimized and proof paths agree within the frozen tolerance.
- Confirm `git status --short` contains only the intended branch changes before the final commit and
  is clean afterward.

End a branch with one coherent commit and a handoff containing commit SHA, proof state, public APIs,
legacy classifications, first divergences, convergence, benchmark, peak memory, limitations, the
permanent tests retained and why, and minimum integration requests. End the Codex response with
exactly `READY` or `BLOCKED`.
