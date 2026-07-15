# Self-contained reference material

This directory makes every bootstrap, proof, physics, integration, selection, and fitting worktree
independent of external paths.

- `rasim_reference_v1.npz` is immutable subsystem comparison evidence.
- `reference_manifest.toml` records the pack hash and schema.
- `manuscript/` contains the exact TeX sources for cited equations and refinement requirements.
- `legacy_source/` contains only the original-RASIM files cited by the task documents.

The proof hierarchy remains:

1. analytic identities, conservation laws, direct enumeration, and independently converged results
2. tracked manuscript equations with explicit assumptions and measure
3. immutable reference-pack intermediates
4. final original-RASIM images or display values

The tracked legacy source is evidence, not architecture. Never import it from production or tests,
copy modules wholesale, or execute it as the proof oracle. `MATCH` cases reproduce valid legacy
intermediates. `CORRECTED` cases agree until the named first divergence and then use an independent
oracle. Physics worktrees may read but never modify any file under `reference/`.
