# Data provenance

The OSC, CIF, PONI, and calibration inputs in this directory were supplied by the project owner
for this repository seed. Absolute legacy paths and GUI state were removed. SHA-256 values are
recorded in `examples/MANIFEST.toml` and `FILE_MANIFEST.json`.

The three files under `pbi2/benchmark/` are the exception: they are generated, explicit-P1
scientific validation fixtures idealized from the SHA-pinned native 2H metric and layer height,
with exact rational registry coordinates. Their CIF comments and `pbi2/README.md` distinguish them
from the supplied relaxed structures.

`Bi2Se3_legacy.cif` is the CIF referenced by the supplied saved state. `Bi2Se3_vesta.cif` differs
only in textual occupancy formatting for Se1 and is paired with the VESTA export. The expanded P1
file is an independent symmetry-expansion fixture.

The legacy peak CSV is provenance evidence. For this CSV specifically, after the one clockwise OSC
conversion at the I/O boundary, `legacy_raw_x` is the accepted native detector column and
`legacy_raw_y` is the accepted native detector row. These fields are distinct provenance from the
supplied-state legacy `x`/`y` variables. `observed_column_px = columns - 1 - legacy_raw_x`
contains an additional legacy horizontal reflection and is not a numerical oracle;
`observed_row_px` duplicates `legacy_raw_y`.

The tracked Bi2Te3 replay inputs were supplied by the project owner and are preserved without user
paths. `structures/Bi2Te3_cod_9011962.cif` is the exact 731-byte fit input with SHA-256
`e1e5f42082bf0699b1a674b19fbf74cf1163b5c18dcb7d00142dd0b2dc4b1fe3`.
The deterministic gzip OSC containers decode to the original detector arrays; the original
uncompressed file SHA-256 values were:

- 5 degrees: `6f00b27802e6419ad79d9ec38441f3cd051b0becaba399f351caeb9a0e451c26`
- 10 degrees: `519c4fbf05bed80abcce287423bc271a06b4e349d2fbc778dcb62b5c5d779212`
- 15 degrees: `70f12525554b5223341f3987545665bcd4b2ea35e24fe565548aff4557329355`

`bi2te3/observations/indexed_catalog.json` is an earlier frozen position-free audit catalog, not the
exact observation handoff used by the accepted final geometry fit and not a substitute fit input.
The replay records its catalog manifest, the accepted historical selection revision, the selection
revision recomputed under the case-bound numerical runtime, and the tracked catalog-file SHA-256
separately. The catalog row-manifest is `sha256-4f3755ac...`; the file SHA-256 is `84cca62c...`.
The decoded detector arrays are exact, while the selection manifest also hashes floating-point
context tokens that vary with the numerical runtime. The Bi2Se3/HBN
`calibration/hbn/darkImg.osc.gz` is reused as the exact dark input; it is not duplicated under
Bi2Te3.
