# SLATE-rMC

SLATE-rMC is a compact detector-native X-ray scattering core. It maps a configured incident beam,
CIF-derived reciprocal rods, mosaic probability, finite-stack structure strength, refraction, and
attenuation directly to a continuous detector-coordinate density and then integrates that density
over detector pixels.

The production path is continuous and deterministic. It does not construct an Ewald-sphere mesh,
sample an orientation cloud, create scattering-event rows, resample candidates, or deposit points
onto pixels. The Ewald sphere appears only through the elastic equation, solved analytically for
each rod and incident state.

## Quick start

Python 3.12 or 3.13 is required.

```powershell
uv sync --frozen --group dev --extra visualization
uv run python scripts/verify_seed.py
uv run pytest
```

The authoritative Bi2Se3 input is
[`configs/bi2se3_simulation.yaml`](configs/bi2se3_simulation.yaml). It controls the source, sample,
detector pose and both detector tilts, mosaic, finite 2H structure model, numerical backend, and
which external figures are rendered.

```powershell
uv run python scripts/run_configured_simulation.py configs/bi2se3_simulation.yaml
```

Generated figures and diagnostics must resolve outside the repository. Override the configured
destination with `--output-dir` when needed. Select `cpu` or `cuda` explicitly in the YAML; CUDA
fails clearly when no compatible device is available.

For a one-incident-state, native-pixel convergence diagnostic, use
`scripts/generate_bi2se3_continuous_detector.py`. Its omitted physical options inherit the same YAML
fixture, so there is only one default authority.

## Scientific contracts

- Internal angles are radians; distances are metres; wavelengths and crystal lengths are
  angstroms; wavevectors are inverse angstroms.
- Arrays are `[row, column]`; continuous detector coordinates are `(column_px, row_px)`.
- Every physical `(h,k)` rod is evaluated independently before an exact-family intensity sum.
- Source, phase, polarization, structure, mosaic, optical, Jacobian, and pixel-integration factors
  each have one owner and are applied once.
- Invalid rays carry explicit status and zero intensity.
- The pre-binned detector field is callable at arbitrary floating-point detector coordinates.
- Pixel values are deterministic box integrals, never display interpolation.

Read `AGENTS.md`, `docs/CONVENTIONS.md`, `docs/RESULT_MEASURE.md`, `docs/ARCHITECTURE.md`, and
`docs/CONTRACTS.md` before changing the numerical core. Compact proof commands and accepted
classifications are recorded in `docs/VALIDATION.md` and `docs/PHYSICS_LEDGER.md`.
