# SLATE-rMC

SLATE-rMC is a compact detector-native X-ray scattering core. It maps a configured incident beam,
CIF-derived reciprocal rods, mosaic probability, finite-stack structure strength, refraction, and
attenuation directly to a continuous detector-coordinate density and then integrates that density
over detector pixels.

The deterministic production path does not construct an Ewald-sphere mesh,
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

For fitting at selected continuous detector coordinates, contract v13 also accepts a general
layered CIF through `cif_conventional_cell_finite_repeat.v1` and the shared
`build_source_averaged_structure_detector(...)` boundary. Bi2Se3, Bi2Te3, ordered generic CIFs,
and the fixed five-parent PbI2 provider use the same sparse detector and rank-gated region fitter.
The ordinary `run_configured_simulation.py` command below remains the optimized Bi2X3 full-image
renderer; a CIF alone does not define observation regions, fit coordinates, background, mosaic, or
stacking law. See `docs/EXAMPLES.md` for the shared API and limitations.

Generic configured strength declaration:

```yaml
structure_factor:
  model_id: cif_conventional_cell_finite_repeat.v1
  repeats: 5
  normalization: FINITE_TOTAL
  unknown_u_iso_A2: 0.0
```

```powershell
uv run python scripts/run_configured_simulation.py configs/bi2se3_simulation.yaml
```

Generated figures and diagnostics must resolve outside the repository. Override the configured
destination with `--output-dir` when needed. Select `cpu` or `cuda` explicitly in the YAML; CUDA
fails clearly when no compatible device is available.

For a one-incident-state, native-pixel convergence diagnostic, use
`scripts/generate_bi2se3_continuous_detector.py`. Its omitted physical options inherit the same YAML
fixture, so there is only one default authority.

The current fitting workflow uses frozen detector-native observations and one shared
material-independent search, response and rendering path. Built-in specimen bindings cover
Bi2Se3, Bi2Te3 and the four declared PbI2 acquisitions. New materials supply an explicit
`NativeRefinementModel`; their symmetry and parameter rules do not enter the detector core.

```powershell
uv run --frozen python scripts/prepare_native.py `
  --observations C:\external\original\sample_observations.json `
  --output-directory C:\external\prepared

uv run --frozen python scripts/refine_native.py `
  --physics C:\external\prepared\PHYSICS.json `
  --observations C:\external\prepared\sample_observations.json `
  --plan C:\external\fit_plan.json --output C:\external\fit.ra_diag.npz
```

Preparation verifies and relocates an existing calibrated experiment; it preserves measured
support, background and covariance. Use the hash-prefixed physics filename referenced by the
prepared descriptor. `--resume` reuses exact completed predictions and restarts the public
optimizer. `scripts/render_native.py` renders a saved selection; unqualified candidates require
`--candidate`. Add `--full-image` for continuous integration over every native pixel.
See [native refinement](docs/NATIVE_REFINEMENT.md) for complete commands, parameter ownership,
reuse validity, recovery and scientific qualifications. Historical orchestration is archived
at Git revision `349524d24f960198d75df8def104adbca204944a`.

## Interactive tools

All supported live viewers are collected in [`interactive/`](interactive/README.md):

```powershell
uv run --extra visualization python interactive/detector_viewer.py
uv run --extra visualization python interactive/ewald_sphere_viewer.py
```

The detector viewer provides continuous full-native Monte Carlo updates while parameters change.
The Ewald viewer synchronizes Bi2Se3 and Bi2Te3 Ewald spheres and detector planes at 5, 10, and 15
degrees. Their controls, backend choices, and scientific qualifications are documented beside the
entry points and in `docs/EXAMPLES.md`.

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
