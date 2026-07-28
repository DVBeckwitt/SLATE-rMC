# Bi2Te3 staged-fit example

`structures/Bi2Te3_cod_9011962.cif` is the full-occupancy archived COD 9011962
structure used by `configs/bi2te3_simulation.yaml`. The archived source omitted
`_atom_site_type_symbol`; this tracked normalization adds the explicit `Bi`,
`Te`, and `Te` symbols required by the strict CIF boundary and otherwise keeps
the lattice, fractional coordinates, displacement values, and occupancies
unchanged.

The tracked CIF is the exact 731-byte input used by the accepted fit. The three detector-native
OSC arrays are stored as deterministic gzip containers under `osc/`; `observations/indexed_catalog.json`
retains the historical catalog as an audit input.

`experiment/staged_fit_replay.toml` reproduces the one-state position fit and the joint 250-state
mosaic and relative ordered-intensity fits. Detector tilts, beam center, lattice constants, atomic
positions, geometry during mosaic/SF fitting, and mosaic during SF fitting remain frozen. The
ordered occupancies are Te1/Bi and Te2/Bi ratios, not selenium parameters. The measured fit is
model-limited and has no independent structure oracle or count calibration. The accepted Bi2Te3
250-state mosaic and ordered stages are CUDA-qualified under the current lock. The optional native
image hashes retain a historical RTX-3060 CUDA oracle and require an explicit `--through render`
requalification. Invoke the case with `uv run --frozen python scripts/replay_staged_fit.py` so the
tracked dependency lock participates in the replay.
