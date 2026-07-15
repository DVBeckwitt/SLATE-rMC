# Project instructions

## Objective

Build the smallest cohesive numerical core that produces scientifically correct detector-native
results and supports staged geometry, mosaic, ordered-intensity, and stacking-disorder fitting
without duplicating physics.

Correct declared outputs outrank legacy parity, implementation style, language, or processor.

## Read before editing

Read `docs/SCOPE_AND_PHASES.md`, `docs/CONVENTIONS.md`, `docs/OSC_COORDINATES.md`,
`docs/RESULT_MEASURE.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`,
`docs/DOVETAIL_MATRIX.md`, the assigned task, its referenced rows in `docs/PHYSICS_LEDGER.md`,
`docs/TRACE_SCHEMA.md`, `docs/VALIDATION.md`, `docs/ERROR_INJECTION.md`, `docs/EXAMPLES.md`, and `reference/README.md`.

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
- Never import or execute the tracked original-RASIM snapshot from production code or permanent tests.
- Never copy legacy modules wholesale. Reimplement the selected equations behind the new contracts.
- Do not add backend, plugin, registry, compatibility, or abstraction frameworks without a
  measured need.
- Production modules contain no embedded tests, demo harnesses, or debug output.

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
- Apply each source, event, optical, polarization, solid-angle, and deposition factor exactly once.

## Worktree isolation

Run bootstrap and reference verification serially. Create all four physics worktrees from the same
clean `PROOF_BASE_SHA`. Each worktree edits only owned paths, treats shared contracts, examples,
dependencies, and reference files as read-only, and never merges another physics branch.

The main agent is the only writer. Subagents may be used for bounded read-heavy exploration,
derivation challenge, test review, or log analysis. Stop `BLOCKED` when a shared contract is
insufficient rather than editing protected files or weakening proof.

## Proof budget

Every branch provides analytic/invariant proof, an independent oracle where specified, tracked
reference comparison, first-divergence evidence for corrections, convergence, wall time, peak
memory, and assigned error-injection detection. Keep the permanent suite compact. Large sweeps,
images, profiling, and legacy traces are not permanent tests.

## Diagnostics

Diagnostics are disabled by default. No diagnostic output may be written under the repository
root. A retained diagnostic is exactly one external `.ra_diag.npz` containing numeric arrays and
one JSON manifest. No sidecars or diagnostic directories.

## Handoff

End a branch with one coherent commit and a handoff containing commit SHA, proof state, public
APIs, legacy classifications, first divergences, convergence, benchmark, peak memory, limitations,
and minimum integration requests. End the Codex response with exactly `READY` or `BLOCKED`.
