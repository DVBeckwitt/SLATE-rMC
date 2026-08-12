# General-CIF pipeline progress

- [x] Read the mandatory contracts, measures, ledgers, validation record, examples, and worktree
  instructions.
- [x] Freeze scope: one layered-film CIF pipeline, explicit non-CIF experiment/model inputs, no
  material-name dispatch or plugin registry.
- [x] Create clean branch `codex/general-cif-pipeline` from accepted `main` `5c365e5`.
- [x] Slice 1: generic CIF finite-repeat strength and direct triclinic atom/repeat oracle.
- [x] Slice 2: material-neutral source-averaged sparse detector transfer and Bi2X3 parity.
- [x] Slice 3: configured generic-CIF construction and mosaic recovery.
- [x] Slice 4: affine CIF site basis and ordered matched-region recovery.
- [x] Slice 5: optional detector center/distance geometry calibration.
- [x] Slice 6: one typed structure-region boundary and Bi2Se3/Bi2Te3/PbI2 integrations, including
  fixed-parent PbI2 and regular kinematic `00L`.
- [x] Final proof, adversarial review, cleanup, and one coherent handoff commit.

The reusable numerical stages are shared. Raw-OSC discovery, mask/background preparation, and the
choice of fitted coordinates remain explicit experiment data; no new per-material runner or
plugin was introduced. Measured mixed/disordered PbI2 remains model-limited rather than being
promoted by synthetic plumbing tests.
