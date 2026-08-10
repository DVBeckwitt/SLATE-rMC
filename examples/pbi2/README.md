# PbI2 stacking example

The three CIFs under `structures/` are the supplied, separately relaxed 2H, 4H, and 6H structures.
They support native structure and topology checks, but their different metrics and intralayer
coordinates make them unsuitable as exact common-motif intensity oracles.

The three explicit-P1 CIFs under `benchmark/` are synthetic validation inputs idealized from the
SHA-pinned native 2H metric and layer height, with registry coordinates rationalized to exact
thirds. They encode the exact manuscript cycles `0F+`, `0F+ -> 1F-`, and
`0F+ -> 1F+ -> 2F+`, with one common layer repeat and zero displacement. Run their independent
whole-CIF Bragg-sum comparison against the pure transition parents with:

```powershell
python -m rasim_next.proof pbi2-polytype-bragg --json
```

The benchmark 6H+ hand is intentionally opposite the supplied relaxed R-3m 6H- setting. The
benchmark validates ideal stacking topology and raw intensity assembly; it is not another material
structure determination. `stacking_truth.toml` remains a synthetic disorder input. Bi2Se3 is the
main detector example, but it is not a sufficient material fixture for the PbI2 stacking model.
