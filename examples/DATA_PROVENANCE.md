# Data provenance

The OSC, CIF, PONI, and calibration inputs in this directory were supplied by the project owner
for this repository seed. Absolute legacy paths and GUI state were removed. SHA-256 values are
recorded in `examples/MANIFEST.toml` and `FILE_MANIFEST.json`.

`Bi2Se3_legacy.cif` is the CIF referenced by the supplied saved state. `Bi2Se3_vesta.cif` differs
only in textual occupancy formatting for Se1 and is paired with the VESTA export. The expanded P1
file is an independent symmetry-expansion fixture.

The legacy peak CSV is provenance evidence. Its canonical columns are `observed_column_px` and
`observed_row_px`. The retained `legacy_raw_x` and `legacy_raw_y` columns are not canonical axes.
