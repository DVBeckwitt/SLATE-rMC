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

The accepted Bi2Se3 and Bi2Te3 5/10/15-degree staged fits are portable, hash-bound examples. Check
their inputs without fitting, or replay through each currently qualified checkpoint into an
external directory:

```powershell
uv run --frozen python scripts/replay_staged_fit.py examples/bi2se3/experiment/staged_fit_replay.toml --inputs-only
uv run --frozen python scripts/replay_staged_fit.py examples/bi2te3/experiment/staged_fit_replay.toml --inputs-only
uv run --frozen python scripts/replay_staged_fit.py examples/bi2se3/experiment/staged_fit_replay.toml `
  --output-directory C:\path\outside\the\repository\bi2se3-replay --backend cuda --through ordered_intensity
uv run --frozen python scripts/replay_staged_fit.py examples/bi2te3/experiment/staged_fit_replay.toml `
  --output-directory C:\path\outside\the\repository\bi2te3-replay --backend cuda
```

Geometry uses one ideal source state. Mosaic and ordered intensity reduce all 250 source states
into one detector function per incidence before comparison. See `docs/EXAMPLES.md` for result and
render qualifications.

Use `--frozen` for these replays: each case hashes `uv.lock`, and the frozen invocation installs
and executes its complete locked numerical dependency closure. Every stage records that execution
runtime and resume requires an exact runtime match; fit values remain tolerance-based across fresh
interpreter, operating-system, and CUDA executions. The optional historical render oracle has a
separate qualification boundary described in `docs/VALIDATION.md`.

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
