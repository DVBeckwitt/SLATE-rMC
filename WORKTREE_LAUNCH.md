# Worktree launch

## 1. Initialize the greenfield repository

```bash
unzip RASIM_Next_Repo_Seed.zip
cd rasim-next
git init -b main
git add .
git commit -m "chore: seed greenfield RASIM Next repository"
uv sync --frozen --group dev
python scripts/verify_seed.py
```

## 2. Run the serial spine

Run `tasks/prompts/bootstrap.md` in the main checkout. Commit its accepted result. Then run
`tasks/prompts/reference_verification.md`. Commit any required proof-base changes and record:

```bash
PROOF_BASE_SHA=$(git rev-parse HEAD)
```

The tracked `reference/rasim_reference_v1.npz` and `examples/` inputs are immutable after this
point. The external archives are not required for any worktree proof. The exact cited manuscript extracts and legacy-source snapshot are tracked under `reference/`.

## 3. Create four worktrees

Manual Git worktrees:

```bash
git worktree add -b feat/geometry-optics ../wt-geometry "$PROOF_BASE_SHA"
git worktree add -b feat/mosaic-ewald ../wt-mosaic "$PROOF_BASE_SHA"
git worktree add -b feat/ordered-reflectivity ../wt-ordered "$PROOF_BASE_SHA"
git worktree add -b feat/stacking-transition ../wt-stacking "$PROOF_BASE_SHA"
```

Start one Codex task in each worktree with the matching prompt under `tasks/prompts/`.

Codex-managed worktrees start detached. The prompt may create its feature branch after confirming
that `HEAD` is the selected proof-base commit. Tracked files are present automatically. No ignored
environment file is required.

Use one separate `.venv` per worktree and one shared external package cache:

```bash
export UV_CACHE_DIR="$HOME/.cache/uv"
uv sync --frozen --group dev
```

Limit each concurrent branch to one BLAS/OpenMP thread unless its own benchmark intentionally
measures parallel scaling.

## 4. Review and integrate

Do not merge overnight. Run the read-only review prompt with the four exact SHAs. Integrate one
approved branch at a time through T07, rerunning the smallest vertical proof after each merge.
Selection and fitting begin only after the detector-native forward model is accepted.
