# Data provenance

The OSC, CIF, PONI, and calibration inputs in this directory were supplied by the project owner
for this repository seed. Absolute legacy paths and GUI state were removed. SHA-256 values are
recorded in `examples/MANIFEST.toml` and `FILE_MANIFEST.json`.

`Bi2Se3_legacy.cif` is the CIF referenced by the supplied saved state. `Bi2Se3_vesta.cif` differs
only in textual occupancy formatting for Se1 and is paired with the VESTA export. The expanded P1
file is an independent symmetry-expansion fixture.

The legacy peak CSV is provenance evidence. For this CSV specifically, after the one clockwise OSC
conversion at the I/O boundary, `legacy_raw_x` is the accepted native detector column and
`legacy_raw_y` is the accepted native detector row. These fields are distinct provenance from the
supplied-state legacy `x`/`y` variables. `observed_column_px = columns - 1 - legacy_raw_x`
contains an additional legacy horizontal reflection and is not a numerical oracle;
`observed_row_px` duplicates `legacy_raw_y`.
