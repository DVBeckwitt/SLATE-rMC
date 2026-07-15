# RASIM Next repository seed

This archive seeds a new greenfield repository for correct native-detector X-ray scattering and
staged refinement. It contains the project philosophy, mathematical conventions, physics/source
ledger, cross-branch contracts, proof protocol, worktree prompts, task graph, immutable compact
reference data, and self-contained Bi2Se3, HBN, OSC, and PbI2 examples.

The controlling result is a declared detector-native observable. Architecture and acceleration are
subordinate to correctness, convergence, composability, wall time, and memory.

Development order:

1. Run the serial bootstrap task to implement shared units, frames, OSC conversion, complex-kz
   primitives, contracts, trace schema, reference reader, and synthetic plumbing.
2. Run the serial reference-verification task and commit the common proof base.
3. Create four worktrees from that exact commit for geometry/optics, mosaic/Ewald, ordered
   rods/reflectivity, and stacking transition.
4. Review all four branches read-only, then integrate them one at a time through detector-native
   vertical slices.
5. Only after integration, implement stable Qr-family, rod, reflection-group, and physical-branch
   selection.
6. Add fitting in stages: detector/source calibration, sample/goniometer geometry, mosaic shape,
   ordered intensities, then stacking-disorder intensities.
7. Add caking and `2theta/phi` only later as a measurement transformation that reuses accepted
   identities and geometry.

No GUI work is included. Persistent diagnostics are external, disabled by default, and exactly one
`.ra_diag.npz` file when requested.

Start with `AGENTS.md`, `WORKTREE_LAUNCH.md`, and `tasks/OVERNIGHT_RUNBOOK.md`. Verify the archive
with `uv sync --frozen --group dev` followed by `python scripts/verify_seed.py`.

## In-repository scientific evidence

The seed includes the selected manuscript TeX sources, the exact original-RASIM files cited by the
tasks, a compact immutable numerical reference pack, three Bi2Se3 OSC images, HBN calibration and
dark images, and Bi2Se3/PbI2 crystallographic examples. Physics branches require no external data.
