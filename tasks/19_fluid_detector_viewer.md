# T19 — Fluid full-native detector viewer

Status: `READY`

## Activation and objective

This task is activated by the user request to make every parameter control in
`interactive/detector_viewer.py` update the complete detector image continuously, without
spatial downsampling, and to implement the approved scheduler, invalidation, progressive-sampling,
CPU, CUDA, and presentation improvements as one cohesive slice.

The authoritative observable remains the T07 weighted forward Monte Carlo estimate of raw native
detector-pixel mass. Interactive previews reduce only the number of prefix-stable mosaic draws;
they never reduce the 3,000 × 3,000 detector grid, source-state catalog selected by the user, rod
catalog, roots, or physics factors. The final settled stage uses the exact requested draw count.

## Frozen scientific and execution semantics

- The image is `float64`, shape `[detector row, detector column]`, and has measure
  `raw_detector_pixel_mass_monte_carlo_estimate_A2.v1`.
- Every valid source state remains an independent stratum. Every sampled orientation enumerates
  every reachable physical rod and retained root and deposits into the exact hard native-pixel bin.
- A fixed-width NumPy Philox key is derived from the detector seed, each draw owns a disjoint counter
  interval, and canonical source-state rows consume fixed eight-lane blocks within that interval.
  The first `n` draws and source rows are identical for every larger request; CPU and CUDA consume
  the same host-generated latent coordinates.
- Progressive stages are the deduplicated prefix `(1, 4, 8, requested_draw_count)`. A newer
  revision cancels and supersedes all unfinished work. Settled output is published only for the
  latest revision.
- The CPU implementation is the float64 proof oracle. CPU state blocks use private full-native
  accumulators, a stable reduction order, and at most four workers; worker count cannot change the
  declared result beyond the frozen accumulation tolerance.
- CUDA is an explicitly requested forward-Monte-Carlo backend with its own backend and device
  identity. Its sampler owns persistent device copies of compiled rod, structure, source, and
  geometry state plus reusable image/ledger scratch. Geometry rebinding updates only causally
  changed device arrays. CUDA unavailability fails closed; it never silently executes the CPU path.
- The CUDA kernel streams roots directly into the native image and draw ledger. No orientation,
  candidate, root, event, hit, or point-deposition table is retained.
- GPU presentation uploads every native pixel to one single-channel texture and applies the log
  transform in its fragment shader. A contiguous `float32` copy is presentation-only; the
  authoritative `float64` result is unchanged. The explicitly reported software-presentation
  fallback also keeps the full native grid.
- The six detector-pose controls invalidate detector projection only. The eight goniometer/sample
  controls rebuild incident transport and re-enumerate rods only if valid-state topology changes.
  The incidence control displays absolute `theta_i` from 0 to 20 degrees and converts it once to the
  configured-pose delta consumed by the existing geometry path.
  Draw count invalidates sampling only; source-state count invalidates the source bundle and is
  committed on release rather than rebuilt at drag cadence.

The authoritative equations are PHY-MEA-005A in `docs/PHYSICS_LEDGER.md`, the forward-root map in
`_continuous_detector_kernel.py`, the optical/structure equations it calls, and exact pixel
ownership `floor(coordinate_px + 0.5)`. This task adds no new scattering equation or normalization.

## Owned paths

- `src/rasim_next/pipeline/source_averaged_detector.py`
- `src/rasim_next/pipeline/_continuous_detector_kernel.py`
- one narrow forward-CUDA module under `src/rasim_next/pipeline/` if separation is smaller than
  extending the inverse-coordinate CUDA module
- `interactive/detector_viewer.py` and at most one narrow presentation helper under
  `src/rasim_next/visualization/`
- `tests/test_integration.py`
- `pyproject.toml` and `uv.lock` only if the measured GPU presentation requires one optional
  dependency
- T19 entries in the contract, measure, architecture, dependency, example, validation,
  performance, error-injection, task-index, and final manifest documentation

Forbidden paths and retained absences are the original-RASIM snapshot, reference-pack bytes,
unrelated fitting modules, a general point depositor, retained orientation/event/hit tables,
module-global caches, import-time CUDA or GUI initialization, a second physics implementation, and
any diagnostic or generated image beneath the repository root.

## Incremental slices

1. Extract a deterministic latest-only progressive scheduler, cancellation token, and dependency
   table; prove stale work cannot publish and detector-only changes reuse incident transport.
2. Add fixed-width Philox latent generation, progressive prefix proof, cooperative CPU block
   cancellation, four-worker private accumulation, and stable reduction against the scalar oracle.
3. Add the persistent forward CUDA sampler, streamed atomic deposition, reusable scratch,
   geometry-only updates, explicit backend/device provenance, parity, and fail-closed behavior.
4. Replace the detector raster presentation with a full-native GPU texture/log shader while keeping
   controls responsive and an explicit full-native software fallback.
5. Remove temporary probes, freeze tolerances from independent evidence, update contracts, run the
   complete proof and benchmark gates, refresh the one manifest, and make one coherent commit.

## Permanent proof and error injections

- The existing tiny direct latent oracle remains independent of production latent generation and
  checks m=0, both nonzero roots, optics, weighting, hard bins, mass conservation, and work ledgers.
- A prefix/worker test checks draw-prefix identity and stable CPU reduction. An already-cancelled
  request must raise the named cancellation exception without returning a partial result.
- A deterministic scheduler test injects revisions A/B/C and proves B never starts, A cannot
  publish, C alone refines, and superseding C removes its remaining stages.
- The invalidation table test makes incident rebuilding fail during detector-only changes; the
  differential test retains a nonzero sample correction across the next detector-only revision;
  the existing changed-validity test proves the complementary rebuild and rod-reenumeration path.
- CUDA uses the same tiny fixture. Occupied bins and integer ledgers match CPU exactly; masses match
  the predeclared float64 tolerance. Projection-only output matches a complete fresh rebind, its
  transport buffers retain identity, and an injected staged-transfer failure cannot swap active
  projection buffers and poisons the sampler. Patched device absence must fail at backend selection.
- A presentation-boundary test uses a non-square sentinel and proves unchanged shape, orientation,
  element count, and authoritative `float64` bytes. GPU-context and frame-time checks remain
  one-shot external evidence rather than flaky permanent tests.
- Assigned mutations are stale-publication acceptance, reversed detector row/column texture
  ownership, removal of one source/root factor, non-prefix latent layout, worker-dependent reduction,
  and silent CUDA fallback. Each must fail first at its named scheduler, presentation, direct-oracle,
  prefix, worker-parity, or backend-identity gate.

Legacy viewer latency and raster presentation are `NO_ORACLE`; they are operational behavior, not
scientific authority. For an identical explicit latent block, the CPU forward observable is
`MATCH`. CUDA is qualified against that CPU oracle. The changed Philox realization is not compared
pixel-for-pixel with the retired PCG64 realization; both estimate the same declared measure, and
the direct oracle plus conservation checks are authoritative.

## Benchmark and done conditions

The acceptance workload is the tracked Bi2Se3 configuration: 1,000 source states, 85 physical
rods, 49 settled draws per state, and a 3,000 × 3,000 native detector on the available RTX 3060.
Record cold setup/JIT separately from warm interaction. Compare equivalent CPU serial, bounded CPU,
and CUDA work; record scheduler-to-display latency for 1/4/8/49 draws, stale-cancellation latency,
texture upload/draw time, total wall time, device identity, and peak host/device memory. No benchmark
may downsample the detector or prune source states, rods, or roots.

Done requires latest-only continuous geometry updates at a 5 ms scheduling cadence, visible
progressive full-native frames, a full requested-draw settled frame, a materially faster measured
warm path than the baseline, explicit fallback status, CPU/oracle and CPU/CUDA parity, clean imports,
Ruff/format, the compact permanent suite, all registered proofs, documentation links, seed/manifest
verification, mutation evidence, no residue, one coherent commit, and a clean worktree. If full-native
GPU presentation or forward CUDA cannot be made explicit and parity-qualified on the available
device, stop `BLOCKED` rather than silently weakening the requested scope.

## Execution state

State: `READY`

Evidence: The latest-only scheduler coalesces at 5 ms, rejects stale completions, retries failed
settled work, and restarts a canceled same-request final stage. Progressive stages are exact Philox
prefixes at 1/4/8/requested draws. CPU one/four-worker output is stable, CUDA occupied bins and work
ledgers match CPU exactly, and the full 1,000-state/85-rod parity probe differed by at most
`1.11e-16 A2`. Cancellation returns within `11.594 ms` median and `28.935 ms` maximum after the
signal. Reset, general geometry rebind, and projection-only rebind failures poison the sampler and
cannot expose partially reset or staged device state.

The full 3,000 × 3,000 RTX 3060 workload measured `16.494/16.823/28.666/244.746 ms` median for the
warm 1/add-to-4/add-to-8/add-to-49 CUDA stages. The projection-only detector path retained the four
transport buffers by identity and measured `28.074 ms` median (`26.474--30.986 ms`) for pitch and
`28.342 ms` (`25.798--30.850 ms`) for translation through full-native texture preparation; all 20
warmed revisions were below 31 ms. Persistent R32F upload plus shader draw measured `11.691 ms`
median (`14.936 ms` p95). The workspace owned `105.464 MiB` of device memory plus one reused
`34.332 MiB` pinned frame. No benchmark reduced source states, rods, roots, selected-stage draws, or
pixels.

The permanent suite covers the independent scalar Philox/root/weight oracle, prefix and worker
parity, CPU/CUDA parity, cancellation cleanup, general and projection-only fresh-rebind parity,
fast/full buffer sequencing, changed-topology rejection, failure-atomic poisoning, mixed nonzero
sample plus detector invalidation, A-to-canceled-B-to-A binding state, latest-only scheduling, and
non-square full-native texture orientation. The complete pytest suite, Ruff lint/format, syntax,
documentation, TOML/YAML, and lock checks pass. All seven registered scientific proof commands pass
their numerical checks; clean-tree wrappers are rerun on the final commit. Independent final code
and CUDA state-machine audits report `READY`.

Commands run: the required scope/convention/architecture/contract/ledger/trace/validation/error/
example/reference/worktree/runbook documents; focused and complete pytest with CUDA; Ruff check and
format check; compileall; `tools/check_docs.py`; `uv lock --check`; all seven
`python -m rasim_next.proof ... --json` commands; full-native CPU/CUDA, cancellation, memory,
projection-only, and fenced OpenGL benchmarks; seed and file-manifest verification.

The final clean references gate also replaced an inherited fixed five-OSC inventory assertion with
named immutable-pack membership while continuing to stream and shape-check all eight declared OSC
inputs; Python example tools remain covered by `FILE_MANIFEST.json`, not the scientific-input
manifest.

Remaining work: none.

Contract or dependency issue: none. `PySide6-Essentials` is confined to the optional
`visualization` extra and the core import path remains unchanged.
