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

## Execution plan

State: READY

Source audit:

- The read-only original checkout was at Git commit
  `235085a4a5b3e66f907041578b4e115347c4de80`. Because that checkout had unrelated local changes,
  every inspection used committed `HEAD` objects through `git show`/`git grep`, never working-tree
  content.
- `ra_sim/io/osc_reader.py:10-11,18-66` (Git blob
  `9dc7ccda07accb87a81c5b58477eea1e11cdfe05`) defines the RAXIS signature, 6000-byte header,
  version/endian rule, dimensions, payload, and high-range decoding. This blob exactly matches the
  tracked source snapshot.
- `ra_sim/gui/_runtime/runtime_session.py:558-565` (Git blob
  `eba763dc44be42c453dc3e01d862f1213d6d86c8`) declares the one-clockwise measured-image mapping.
- `ra_sim/gui/background.py:219-223` (Git blob
  `6b3e438d3d66b00891aa318fbb2ace317deb1f52`) applies that clockwise conversion.
- `docs/simulation_and_fitting.md:219-226` (Git blob
  `5308c9fa97af8a430f1758b5f7fadcf733ffda91`) records legacy `x=row`, `y=column` semantics.
- The immutable tracked snapshot and reference pack retain original archive SHA-256
  `41327341317c45bddb4edbac8ac9954f8ec500a57a3349db22854e703322defa`.

Commands run:

- `uv run --frozen --group dev python -m rasim_next.proof references --json` twice with exact output
  comparison
- `uv run --frozen --group dev ruff check src/rasim_next/proof/reference.py`
- `git diff --exit-code -- reference examples`

## Handoff

Status: READY

Commit SHA: this handoff is committed with T01; the exact immutable proof-base SHA is recorded after
the commit.

Proof summary:

- Verified all 29 tracked source-snapshot hashes and the exact OSC/orientation citations.
- Verified reference-pack SHA-256
  `e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06`, its embedded manifest,
  73 nonempty numeric arrays, unique ownership by 12 cases, and 8 `MATCH`/4 `CORRECTED`
  classifications.
- Verified all 21 example-input hashes and complete non-document input coverage.
- Streamed all five gzip OSC files without writing decompressed files. The three Bi2Se3
  decompressed hashes, six independently defined summary values, selected raw/native pixels, and
  raw/native argmax positions match the pack.
- Verified big- and little-endian synthetic decoding, `0x8002 -> 64`, `0xffff -> 1048544`, the
  7-by-11 clockwise mapping and inverse, and raw marker `(4,3)` to native coordinate `(2,3)`.
- Verified the native Bi2Se3 beam center and all 82 legacy coordinate rows through exactly one
  orientation conversion (maximum CSV serialization residual `2e-13` pixel).
- All four in-memory negative controls fail at their expected first stage. Repeated proof JSON is
  byte-for-byte deterministic, and `reference/` plus `examples/` remain unchanged.
- One proof run took 0.632206 s with 10,904,349 bytes traced peak memory.

Known limitations:

- The v1 pack authenticates arrays through its whole-file SHA-256 but does not declare per-array
  hashes, tolerances, stage metadata, or ledger IDs. Its undocumented seventh OSC-summary column is
  authenticated but was not assigned a guessed meaning.
- The available Git checkout is a different provenance object from the tracked source archive;
  its OSC blob matches exactly, while its runtime-session blob differs although the cited
  orientation lines agree.
- The task file names `chore/reference-verification`; `tasks/index.yaml` names
  `chore/original-characterization`. T01 ran serially on `main`, so neither branch name was used.

Contract requests: none. No T00 convention changed and no external path is required at runtime.
