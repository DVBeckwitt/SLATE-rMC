# T01: verify the tracked proof reference

Branch: `chore/reference-verification`

## Goal

Prove that a clean clone can read and validate the immutable reference pack and all tracked example
inputs without original RASIM, manuscript files, absolute user paths, or repository-local output.

## Owned paths

```text
src/rasim_next/reference/
src/rasim_next/proof/reference.py
tests/test_core_coordinates.py
this task's execution-plan and handoff sections
```

Reference and example data are read-only.

## Tasks

1. Validate `reference/reference_manifest.toml`, pack SHA-256, embedded JSON, array metadata, and
   case classifications.
2. Validate `examples/MANIFEST.toml` and every tracked file hash.
3. Stream all gzip OSC files without materializing them under the repository.
4. Prove both synthetic endian cases, high-range decoding, non-square clockwise orientation, inverse
   mapping, and pixel-center mapping.
5. Prove the Bi2Se3 legacy `x/y` mapping is converted to canonical `column_px/row_px` exactly once.
6. Make `python -m rasim_next.proof references --json` deterministic and read-only.
7. Run comparator negative controls by altering a copy in memory, never the tracked pack.

## Done when

The reference proof passes from a clean installation with no external physics paths, all assigned
error injections fail at the expected stage, no tracked input changes, and the worktree is clean.
