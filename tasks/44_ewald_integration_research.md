# T44: Accurate native images at lower computational cost

Branch: `codex/ewald-integration-research`, based on `4cf84de`.
The user explicitly requested this research branch. Main remains the comparison baseline.
Root owns all repository edits; reviewers are read-only. This file is the project plan and
checklist; existing historical plans and incomplete scientific tasks remain intact.

## Question and hypotheses

Can pixel-error-driven angular quadrature produce the same complete native image more cheaply
than geometry-driven panels, after both implementations receive comparable optimization?
Can a physically complete detector-coordinate formulation improve on the optimized angular route?

The first hypothesis is that eliminating repeated full-image allocation and reusing evaluated
children will reduce angular preparation cost. The second is that detector coordinates can avoid
some pixel-boundary work, but its inverse folds and conditional source-position integral may erase
that advantage. Both are hypotheses, not established speedups.

## Frozen scientific comparison

Start with the archived Bi2Se3 independent-basal native model: 85 signed physical rods, all 32
original source rows, both signed structure-factor sheets, the original local-lamella m0 stitch,
and 3000 by 3000 native pixels. Source, geometry, lattice, optical factors and float64 arithmetic
stay fixed. Eight axial nodes are a mechanism screen only; they cannot qualify a full image.
Keep every physical factor exactly once and retain off-panel Gaussian centers whose tails hit
the panel. No intensity threshold, cropped image, survivor normalization or fitted image scaling
may improve a comparison score.

For a completed image I and independently refined reference R, report E1 = sum(abs(I-R))/sum(R),
flux error, error by fixed detector tile, maximum absolute pixel error, runtime and memory.
The target is 1% global relative L1, with reference uncertainty allowance U_R at most 0.2%.
A conservative 1% claim requires (E1 + U_R)/(1 - U_R) <= 0.01 when U_R bounds
sum(abs(R-true))/sum(R); empirical refinement supplies an allowance, not a rigorous certificate.
Report E1 and U_R separately. Global L1 does not guarantee 1% in each dim pixel.
Fitting additionally requires the existing whitened prediction, parameter-contrast and objective
gates in `docs/NATIVE_REFINEMENT.md`; image agreement alone cannot qualify inference.

The production baseline currently fails its angular node budget before producing an image.
There is consequently no qualified baseline runtime or full-image speedup ratio yet.

## Ordered work packages and acceptance

1. **Exact pixel projection with bounded storage** (first implementation).
   - [x] Extend the single Gaussian rectangle owner to native row/column windows and explicit
     caller-owned accumulation; preserve its current support and probability arithmetic.
   - [x] Verify narrow, broad, correlated, off-panel and empty-support kernels against the parent
     revision on identical nodes, including windows reassembled into a complete native image.
   - [x] Measure first-call and warmed costs separately. Accuracy is a gate before timing.
   Files: `source_spatial.py`, this plan, contracts. No new physics or dependencies.
2. **Angular preparation and child reuse** (depends on 1).
   - [ ] Compare the archived dense streaming prototype with compact pixel windows on the same
     frozen nodes, then reuse each evaluated child as its next parent and accepted contribution.
   - [ ] Keep support seeds, axial arrays and signed strengths identical; bound pending storage
     explicitly. Use the canonical rectangle arithmetic, not copied Gaussian code.
   - [ ] Retain accepted meshes, empirical error indicators, exact inputs and failure ledgers in
     one external diagnostic. Check independent angular subdivision and order refinement.
   Temporary research drivers are external and removed after their sources are archived.
3. **Complete image qualification** (depends on 2).
   - [ ] Complete all 192 reachable source/group channels, then independently refine axial,
     angular, cone, spatial and source quadrature. Failed or partial images cannot pass.
   - [ ] Check the coupled ladder I(a,b), I(a+,b), I(a,b+), I(a+,b+) for axial/angular
     resolution, including its mixed difference I(a+,b+)-I(a+,b)-I(a,b+)+I(a,b).
     Compare the joint fine result with another independently refined/seeded level. Seed known
     narrow features and investigate joint adaptivity to reduce expensive evaluations.
   - [ ] Bind mesh reuse to actual parameter stencils and measure preparation plus a fixed
     sequence of 20 evaluations; retain separate cold-image and repeated-evaluation results.
     Predeclare that physical sequence and invalidate/reprepare when geometry, support or the
     mesh's parameter envelope changes. Source convergence is a separate study of the continuous
     source law; compare methods on identical refined source inputs at each level.
   - [ ] Extend qualification to Bi2Te3 and predeclared narrow/broad mosaic, footprint and
     near-fold cases. These cases must use verified inputs rather than invented parameters.
4. **Detector-coordinate feasibility** (depends on the common pixel and physics contracts).
   - [ ] Implement independent-basal inverse intensity through shared geometry, signed strengths,
     cone probability and optics. Treat local-air m0 separately and regularize each source fold.
   - [ ] Select one equivalent spatial formulation: expanded mean-hit coordinates with
     direction-dependent Gaussian pixel probabilities, or actual pixel coordinates with explicit
     conditional-position averaging. Integrate source position once and qualify its numerical
     error; a single global convolution is insufficient.
   - [ ] Establish local physics/support parity before optimization; permit batching, factor reuse
     and local coordinates even if the first full image cannot finish. Give both routes access to
     the same optimized projector. Rank methods only on completed images at the same accuracy gate.
   The existing tied-rotation inverse renderer is a different physical model and cannot substitute.
5. **Decision map and adoption** (depends on 3 and 4).
   - [ ] Compare at least five eligible warmed repetitions per scenario in alternating method
     order, plus separate cold starts. Record median, spread, peak memory, work and failures.
     Work includes unique scattering/structure/cone queries, Gaussian rectangle evaluations,
     accepted/rejected panels and reused children. Distinguish allocation savings from reduced
     physical evaluation counts; retain preparation and accepted-mesh replay separately.
   - [ ] Map tolerance and scenario to the fastest *qualified* method; mark unresolved cells.
     Record preparation break-even counts for repeated use rather than hiding setup cost.
   - [ ] Review the physical and numerical evidence before any default change or main merge.

## First campaign: bounded implementation and measurement

The first campaign has 1800 seconds of numerical/evaluator work, including failed attempts and
refinements. Sub-budgets: 300 seconds for exact projection checks/timing, 900 for optimized angular
preparation, 480 for independent angular replay, 120 reserve. Stop any allocation exceeding the
declared scratch-memory cap; stop a channel at its node/depth/time limit. Retain complete-channel
and partial-current-channel mass separately. No background continuation or automatic merge.

A two-candidate maximum discovery/replay cycle may compare projection implementations against a
frozen same-node evaluator with a valid baseline. Runtime is ranked only after exact support and
float64 parity gates; it is not a scientific accuracy score. Full-image research has no valid
scalar score until a reference qualifies. No benchmark harness is retained in this repository.

If the campaign reaches a limit, record the measured bottleneck and the next specific repair.
Do not equate an immature implementation failure with a mathematical rejection of its method.

## Evidence and status

Prior immutable evidence lives under the task's external visualization root:
`angular_pixel_investigation_20261003.ra_diag.npz` (SHA256
`c8db491ae9e68139b8ac4cfb8e8f4e449daa862204500fa90ff9e590b1036227`).
It contains exact prototype sources, inputs, channel ledgers and independent local angular audits.
The first regular channel took about 415 s to prepare versus 46 s to replay accepted nodes.
These are one-channel measurements at eight axial nodes, not complete-image performance.

## First campaign results (closed, 2026-10-03)

The branch implements native column/row windows, explicit accumulation and conservative visited-pixel
bounds in the existing Gaussian owner. Default calls remain bitwise equal in the frozen checks.
No new adaptive renderer or alternative physical model is installed as a production default.

The retained diagnostic is `ewald_research_branch_20261003.ra_diag.npz` in the same external root,
174861477 bytes, SHA256
`f6da0a5fd0300b2accc1f23019ac54955faf5af41ed5f4d45bba7fb65a3a01b0`.
It contains numeric results, accepted meshes, partial-channel separation, input hashes, exact
temporary sources, matched timing repetitions and failures. Its `temporary_sources` mapping preserves
the runnable research implementations; the temporary files were removed after sealing. Production
code and active documentation remain in this branch, with no retained benchmark harness.

| Measurement | Result | Scope |
| --- | --- | --- |
| Eight-kernel projection batches, five warm repeats | median 0.346 s to 0.142 s, 2.43x | Same 57 physical kernels on the complete native panel; storage optimization only. |
| One-kernel projection batches, five warm repeats | median 1.812 s to 0.119 s, 15.2x | Same work and support; not a simulation speedup. |
| Pixel parity | Full, cropped and tiled outputs bitwise equal in the checked cases | Includes narrow, broad, highly correlated, off-panel and zero-mass examples. |
| Complete-support angular attempt | 34/192 channels complete at 900.029 s | Eight axial nodes; unfinished channel separate; no full-image estimate. |
| First regular channel, compact windows plus child reuse | 202.035 s | Earlier dense prototype took 415.138 s, but adaptive meshes differ. |
| Same channel, joint child batches | 191.185 s | Accepted mesh identical to compact run; one paired mechanism comparison. |
| Spatial projection within that channel | 173.012 s, about 90.5% of channel time | Remaining time includes mass contraction, geometry, refinement and overhead. |
| Tracked live panel arrays | at most 6.51 MB; batched candidate 5.82 MB | Explicit cap 256 MiB; excludes final images, Python/JIT and other process storage. |
| Observed process peak during complete-support attempt | 555511808 bytes | Last process observation before completion; not a guaranteed final process peak. |

Against the saved independently h-refined angular images at identical eight axial nodes,
relative L1 differences are 0.0265647% for local m0/source0 and 0.00141034% for the first
regular group/source0. The saved h/p reference differences are 0.00841841% and 0.000327722%.
These are conditional angular checks with fixed cone/spatial/source rules. They do not qualify
the total physical image. Joint batching preserves the accepted meshes exactly and changes the
two native channel images by relative L1 below 1e-16.

A final frozen-kernel spatial-order screen used 57 actual regular-channel kernels on all native
pixels. Order 64 versus 96 differed by relative L1 1.39e-15. Order 8 differed from order 96 by
1.40e-7 (0.0000140%), with median 0.103 s versus order 16's 0.122 s. Order 4 differed by
0.0610%; order 2 differed by 2.087% and failed the predeclared 0.1% conditional screen. These
results nominate order 8 for a broader spatial qualification; the production order remains 16.
This sample cannot establish spatial convergence across all channels or parameter cases.

Projection checks passed both the historical comparison runtime (NumPy 2.2.6, Numba 0.61.2,
SciPy 1.15.1) and the supported repository runtime (NumPy 2.4.6, Numba 0.66.0, SciPy 1.18.0).
The angular prototype timings used the historical runtime to preserve comparison context.
Ruff checks, formatting, whitespace checks and an offline wheel build passed. The wheel contains
the exact checked production source. No type-check command is configured in this project.

Measured numerical/evaluator work was 1201.822 s. Conservatively charging the entire 390 s
discovery-cycle allowance plus import/build reserve gives 1596.848 s of the 1800 s budget.
The root-writer rule restricted discovery workers to read-only investigation; both timed out.
The completed replay/policy cycle supplied no validated algorithmic improvement and did not
promote code. Its scheduling-policy result is not evidence about numerical-method quality.

## Next experiment and decision

Continue the angular research: exact window storage removed a demonstrated overhead and bounded
the pending panel arrays. Further small batching changes have limited potential on the measured
regular channel because spatial projection now dominates. Next, qualify spatial order 8 on entire
accepted channels and difficult footprint cases, measure rectangle work, and investigate reducing
unnecessary evaluations through global error allocation and coupled axial/angular adaptation.
The production adaptive API should use an explicit shared per-channel transfer seam; the external
research monkeypatch is not an acceptable product interface.

Then require complete images and the coupled refinement ladder before selecting any default.
The detector-coordinate feasibility work remains open and gets comparable optimization after local
physics parity. No method wins the full-image or quantitative-fitting comparison yet. The first
campaign is closed; its completed checks do not mark the remaining work packages complete.
