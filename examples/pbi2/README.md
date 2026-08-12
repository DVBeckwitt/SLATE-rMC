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
structure determination.

The optional-polytype geometry validation uses `structures/PbI2_2H.cif` as one declared
single-trilayer metric and admits already-qualified exact half-/third-order landmarks:

```powershell
uv run --frozen pytest -q tests/test_fitting.py `
  -k optional_pbi2_polytype_landmarks_strengthen
```

It is synthetic and geometry-only: no measured PbI2 OSC data, separate detector calibrations, or
population/intensity fit are supplied here.

The same test contains a separate synthetic mosaic response-bank proof. It converts the qualified
integer/half/third landmarks to exact typed profile identities, omits unavailable peaks, keeps one
profile at exact overlaps, and recovers a shared planted mosaic distribution while profiling out
population-weighted amplitudes. The fitted shapes are analytic and predate the contract-v13 sparse
detector response. This example still contains no measured mosaic-profile observation/background
data and is not a detector-folded fit.

`stacking_truth.toml` remains a synthetic disorder input. Bi2Se3 is the main detector example, but
it is not a sufficient material fixture for the PbI2 stacking model.
