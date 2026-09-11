# Smooth beam-position detector viewer

Base: `19063b07b54bbc14fdcb0b6aeb267bb111fae0dc` (`main`).

The user requests a more efficient qualitative detector viewer which integrates
beam position smoothly while sampling divergence, bandwidth and mosaic. Root is
the only writer; derivation and final review are read-only delegated work.

## Plan before implementation

1. Retain the existing source angular/wavelength realization and spectral masses,
   but replace positional draws by their conditional means. Retain the residual
   Gaussian spatial covariance, including the zero-divergence limit.
2. Reuse canonical forward roots, structure and optics once per scattering
   contribution. Integrate the projected conditional Gaussian into native pixel
   boxes, retaining off-panel influx. The affine plane-intersection derivation in
   `codex/source-position-kernel` is prior evidence; do not import its unrelated
   fitting quadrature or all-pairs density evaluation.
3. Add an explicit smooth-position viewer mode and preserve sampled-position mode
   for comparison and unsupported geometries. Smooth integration requires an
   unbounded planar sample and zero external-path absorption. Report the named
   position-integrated estimate separately from the existing empirical-source
   hard-bin estimate. Use an explicit Gaussian tail budget and no survivor
   normalization. Retain every rod/root, the full detector and latest-only
   progressive rendering. CPU and CUDA use the same positional quadrature.
4. Improve source-count limits/labels, full render timing and preview allocation
   where supported by inspection and measurement. Keep the existing scheduler.
5. Prove conditional moments and direct ray geometry, Gaussian box mass and
   off-panel tails, source spectral weights, zero-width limits, CPU/CUDA parity
   and geometry invalidation. Measure fixed-work latency and quality at equal
   time; do not infer speedup merely from removing two random coordinates.
6. Run focused/full tests, Ruff/format, registered proofs, documentation and
   manifest/seed checks, and independent read-only review. Retain only distinct
   scientific/interface regressions and external benchmark evidence. End with
   one coherent commit and a clean worktree.

## Owned scope

`interactive/detector_viewer.py`, `interactive/README.md`, source sampling and
configured-source construction, the existing CPU/CUDA forward terminal and
sampler, at most one narrow spatial-probability helper, focused permanent tests,
and the corresponding active contract/measure/architecture/example/validation/
ledger/task/file-manifest entries. Reference bytes and unrelated fitting code
are read-only. No dependency additions are planned.

Relevant ledger: PHY-SRC-001/002/003/004/005/006/008,
PHY-GEO-005/008/009/011/012, PHY-MEA-005A/013 and the existing optical factors.
Analytic conditional probability and independent ray/box integrals are the
primary authority. Existing hard-bin results must remain MATCH; position
marginalization first changes the source position integral and has NO_ORACLE in
the immutable legacy pack. It does not change the selected mosaic physics.

## Execution

Implemented conditional source means and a revision-bound residual Gaussian profile; the CPU and
CUDA terminals share the scalar pixel integral and existing scattering physics. The viewer now
defaults to smooth profiles, 128 source states and 8 mosaic draws, with explicit sampled mode.
Short CUDA chunks improve cancellation responsiveness; previews use the full-native float32
presentation lease and settled results retain float64. Source limits respect spectral support,
color ranges agree across presenters, and render timing includes bundle construction.

The user explicitly shortened the completion scope: "focus less on provign it works and just
finishing now". Accordingly, extended convergence sweeps, the full suite, registered proof runs
and peak-memory qualification were not completed. No equal-quality speedup or convergence claim
is made. Exploratory RTX 3060 timings before the final cancellation-chunk adjustment were about
39--42 ms for smooth 128-state/one-draw accumulation and 133--135 ms for four draws, after warmup.
At fixed work, spatial integration costs more than point deposition; its intended benefit is a
smooth qualitative footprint with fewer stochastic coordinates and smaller viewer budgets.

Essential validation: 20 focused beam/viewer/Monte Carlo tests pass, including CPU/CUDA parity,
prefix/reset/cancel/rebind behavior and the unchanged zero-width point limit. Four retained tests
protect distinct contracts: conditional source probability and spectral masses; direct affine ray
transport and invalid-use guards; independent pixel-box integrals and boundary mass; and backend
integration lifecycle. Read-only final review found no remaining correctness blockers.

Public additions: `ConditionalBeamPosition`, conditional-position source construction, optional
`beam_position` sampler argument, and explicit position-model/result-measure identities. Existing
sampled-position behavior remains covered by its regression comparisons (MATCH). The new spatial
integral has NO_ORACLE in the tracked legacy pack; its first changed stage is source-position
integration. It requires unbounded sample support, zero external-path absorption and sufficient
forward-flight margin. Six-sigma spatial cutoffs omit at most approximately 5.92e-9 probability,
plus numerical quadrature error. No reference bytes or dependencies changed.
