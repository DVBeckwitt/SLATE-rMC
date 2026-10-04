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

## Second campaign in progress

The user authorized continuation toward the full target, with a stated 7200-second numerical
work budget including failed attempts. After the initial checks, the user requested a separate
GPT-6.1 Sol task at extra-high reasoning for implementation and diagnostics, with this chat
providing scientific oversight and milestone review. That execution task is the sole writer
from handoff onward; main remains unchanged. Conservatively reserve 360 seconds already used.

Initial mechanism results:
- Rolling integrated-corner and marginal-tail reuse passed frozen-kernel comparisons with
  relative L1 about 1.06e-14. Five single-call 57-kernel timings gave a 1.17x median improvement.
- A direct conditional-CDF-only alternative was slower on those kernels and is not selected.
- Complete accepted-node replays for channels 0 and 32 compared spatial orders 8, 16 and 32.
  Order8 versus32 relative L1 was 1.66e-10 and 4.54e-8; order16 versus32 was 3.99e-17 and
  4.14e-15. Total run256.411s. These use eight fixed axial nodes and qualify only that conditional
  spatial comparison. The run imported the rolling-only source preserved externally as
  `source_spatial_rolling.py`, before the complementary candidate was added on disk.
- A complementary near-unit-correlation candidate is currently uncommitted and experimental.
  Its first frozen-kernel comparison gave relative L1 8.41e-15. Timings overlapped the channel
  replay and require a controlled rerun. It has not received full independent corner/region
  qualification or code review. In particular, audit cache representation when a rectangle
  switches between complementary and Plackett formulas at the endpoint guard.

Temporary sources/results remain in the external task root for the execution handoff. They
must be archived in one diagnostic and removed before final handoff. The next milestone is
independent numerical review and measurement of the spatial candidate, followed by axial/angular
convergence work; no new accuracy or whole-image speed claim is established.

## First delegated spatial milestone (complete, accepted by originating chat)

Retain complementary residuals plus rolling integrated-corner/tail reuse on this research
branch. The endpoint guard now returns direct conditional integration inside the
kernel-stable complementary branch, before any cache access. It therefore cannot mix a
Plackett correction with a complementary residual at a shared corner. No new adaptive
production API is added; spatial default 16 and every physical/qualification gate remain.

The independent oracle integrates the conditional normal CDF with physical transition
panels and GL128/GL256, without production interior-CDF shortcuts. Across 159 difficult
rectangles, its maximum refinement difference and candidate16 absolute error are both
3.3306690738754696e-16. Cases include both correlation signs, the 0.925 transition, exact
and crossed endpoint 12 guards, near-degenerate conditional ratios down to 1e-13,
X-tail clipping and far Y endpoints. Orders 4/8/16/32 have maximum absolute errors
8.54e-6/4.20e-10/3.33e-16/3.33e-16; requested-order refinement remains effective.

Shared-corner guard regressions were evaluated in both visitation orders and signs.
The unrepaired implementation erred by up to 0.1083122747 probability; the repair's
maximum error was 1.11e-16. A decisive native-region case sharing central corner (1,0)
showed unrepaired joint/isolated discrepancies 0.079327627/0.079279821 for negative/positive
correlation. Repaired joint and isolated region projections agree exactly and differ from
the independent reference by at most 5.55e-17. Weighted overlapping regions, native
clipping and off-panel means were separately checked. Tiled native pixels are bitwise
equal to the repaired complete panel; cropped probabilities and accumulation agree
within the recorded float64 roundoff. No probabilities were clipped or normalized.

| Same-work spatial measurement | Parent | Rolling alone | Complementary + rolling |
| --- | --- | --- | --- |
| Frozen 57 kernels, one call, six warmed alternating repeats, median seconds | 0.128566 | 0.109284 | 0.096565 |
| Frozen 57 kernels, batches 8, six warmed alternating repeats, median seconds | 0.277035 | 0.257480 | 0.246986 |
| Complete local-m0/source0 accepted nodes, spatial deposition seconds | -- | 8.057012 | 6.184646 |
| Complete first regular/source0 accepted nodes, spatial deposition seconds | -- | 65.069488 | 49.465508 |

The frozen single-call speedup is 1.331x over parent and 1.132x over rolling alone.
Complete-channel values are spatial deposition on identical accepted nodes, one matched
replay, not angular preparation or full simulation. Local-m0 includes fresh-process
compilation; the regular channel uses warmed kernels. The shared generation plus both
projections took 16.617659 s and 115.065981 s. Kernel byte digests are identical per channel
between methods. Accepted-node counts are 16112/62384, with maximum streamed
mean/factor/mass storage 114688 bytes per batch.

Complementary order16 versus saved rolling order32 channel images differs by relative
L1=5.769253061153425e-17 (local m0) and 3.87465635406183e-15 (regular). Paired complementary
versus rolling order16 differences are 2.1236271664609792e-17/2.9124943509325627e-15.
Frozen 57 versus parent relative L1 is 6.158976919843851e-15; flux error is 1.24e-16.
These are conditional spatial checks at the original eight axial nodes and archived
accepted angular meshes. They do not establish the 1% full-image gate or reference
allowance <=0.2%, axial/angular/source/cone convergence, or fitting qualification.

The new evidence is external ewald_spatial_milestone_20261003.ra_diag.npz, containing
numeric arrays and one JSON manifest with exact temporary/evaluated/final sources,
input/evidence hashes, versions, failed-candidate probabilities, timing repeats,
independent references and software checks. Prior immutable diagnostics are unchanged.
The overwritten rolling-only corner-check output was unavailable at handoff; no
reconstruction of it is claimed. Its earlier facts remain historical plan evidence.
Temporary sources/results were sealed and removed, including obsolete restored sources.
No test/benchmark harness or generated output is retained in the repository.

Finite image/temporary-array allocation is bounded by fixed 3000x3000 buffers and streamed
kernel batches; a conservative owned-array bound is 768 MiB, excluding Python/JIT and
loaded libraries. The phase driver's generic 'three images' description was too narrow
because completed channel arrays remain retained for sealing; the manifest records this
correction. No process peak measurement is claimed. Evidence sealing streams 8 MiB chunks.

Measured additional numerical phase work is 173.438874 s. Charge 300 s conservatively
for this entire milestone including imports, historical replay, software checks and
sealing, plus the prior 360 s reserve: campaign total 660/7200 s, remaining 6540 s.
This is below the delegated 900 s additional allowance. All launched numerical processes
finished, and a process inventory found no research Python/evaluator process remaining.

Dream-rsi history was consulted. In-place bounded comparison evaluated parent, rolling
and complementary+rolling on the frozen evaluator with correctness gates before timing.
Historical completed-world replay and policy selection retained the incumbent; scheduling
reward is not numerical-method quality. A fresh runner cycle would create additional
detached checkouts, contrary to this assignment's sole-checkout instruction, so no new
runner cycle or candidate was fabricated. The old two read-only worker timeouts remain
recorded failures, not scientific rejections. Root remains the sole coding writer.

Ruff lint/format, import/default checks, code-AST parity after formatting/docstring edits,
whitespace checks and an offline wheel build passed. The wheel contains the exact final
production spatial source. These are software checks, not physical adequacy evidence.
No type-check command is configured. Main stays clean at 4cf84de; no merge or push.

### Next experiment proposed for scientific review

Reduce costly native-response evaluations along physical axial panels before another
Gaussian micro-optimization. Screen response-hat product integration
I_p=sum_sign integral S_sign(u) A_sign,p(u)du: interpolate only costly native A with
nonnegative hats and integrate sharp cheap canonical S accurately into effective weights.
Compare against independently refined physical-panel quadrature, include midpoint/native
image checks and both signed strength sheets, then run the coupled axial/angular ladder.
GL8 physical panels and configured Sobol 2^12 are numerical choices, not convergence
certificates; geometry may still require dense u. Do not start this experiment until
the originating chat reviews this milestone. No full image, fit or later campaign ran.

Milestone evidence SHA256: d1904a56b4b639351a3e6b5ae438be1b8fa81100b96db415d6a8ace8d823f9f0 (247712585 bytes).

The direct-environment build first failed because hatchling was absent. The configured
uv isolated offline build succeeded with cached dependencies and exact wheel/source
parity; main's environment was unchanged. The initial failure/source is in the manifest.
After sealing, cleanup encountered uv's output-directory .gitignore; that verified
metadata file and the formatting cache were removed. Final read-only review found no
remaining adoption-blocking spatial code issue. All temporary sources/results are gone.

## Second delegated milestone: coupled integration diagnosis (complete, accepted by originating chat)

The originating chat accepted the spatial milestone and authorized up to 1800 additional
numerical/evaluator seconds to identify the remaining integration error before choosing a new
axial algorithm. Product changes remain limited to the existing research checkout. Normal
external detached read-only dream-rsi checkouts were explicitly permitted for this milestone.
Main, fitting qualification, spatial default 16 and the adaptive production API are unchanged.

### Frozen conditional ladder and provenance

Evaluate source 0/channel 0 local lamella m0 and source 0/channel 32, the first regular radius
1.7511941725650184 A^-1, separately. The regular group retains (-1,0), (-1,1), (0,-1),
(0,1), (1,-1), (1,0), each population 1, and both signed strength sheets. The original roster
of 85 rods and 32 source rows remains in the iterator, preserving group index and original
Sobol seeds 1009/66546. Pooled axial support from all source rows is unchanged:
[0,5.419055674241323] external q for local m0 and [0,5.127512118222034] phase axial u
for the regular group. These are two conditional contributions, not the full 192-channel image.

Exact input bytes, current production source hashes, runtime, original 8 axial coordinates,
PDFs and half weights were checked. Runtime is the requested main environment: Python 3.13.13,
NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0, with research src first. Signed strength prefixes agree
exactly for both groups. The accepted 8 angular meshes come from the immutable branch archive;
their current-runtime spatial 16 images come from the accepted spatial milestone archive.
The prior angular h images used an older runtime and were not reused.

The four complete 3000x3000 native images per group are I8,accepted, I8,h, I16,accepted and
I16,h. Each h mesh independently subdivides every accepted physical angle/Cauchy-CDF panel
into two children and retains GL8. This is a further h indicator of the same formulation,
not an independently converged reference. No old mesh is moved to a new axial coordinate.
The appended eight Sobol positions receive fresh angular preparation. With unchanged g,
I16 = 0.5 I8 + sum_new J(u)/(16g(u)); each old node/PDF/weight prefix is checked exactly.
The appended angular preparation uses half the old channel absolute budget, retaining its
empirical 0.001 relative ledger. No intensity normalization or solid-angle correction is applied.

### Native observables and empirical differences

L1 is sum(abs(fine-coarse))/sum(fine); flux delta is sum(fine-coarse)/sum(fine).
For the mixed difference, D=I16,h-I16,accepted-I8,h+I8,accepted and the denominator is I16,h.
All absolute values below use the unchanged raw native observable. Neither two-level Sobol
differences nor angular coarse/fine indicators are rigorous error bounds or replicate uncertainty.

| Conditional group / difference | Relative L1 | Relative flux delta | Maximum absolute pixel difference | Maximum 64x64 tile absolute sum |
| --- | ---: | ---: | ---: | ---: |
| Local m0: angular at 8 | 3.643869e-4 | -2.979777e-4 | 1.543768e-8 | 1.755862e-7 |
| Local m0: angular at 16 | 5.308798e-4 | -2.947058e-4 | 7.718842e-9 | 8.779308e-8 |
| Local m0: axial at accepted | 0.999999973 | -0.977851028 | 2.426074e-5 | 3.623147e-4 |
| Local m0: axial at h | 0.999999970 | -0.977844558 | 2.426074e-5 | 3.623147e-4 |
| Local m0: mixed D | 5.308798e-4 | 2.946477e-4 | 7.718842e-9 | 8.779308e-8 |
| Regular: angular at 8 | 1.217991e-5 | -1.667868e-8 | 7.822690e-11 | 3.955986e-9 |
| Regular: angular at 16 | 4.537363e-5 | -1.266355e-8 | 6.829472e-11 | 1.345912e-8 |
| Regular: axial at accepted | 0.999805906 | -0.666558260 | 9.848973e-7 | 2.428017e-4 |
| Regular: axial at h | 0.999805897 | -0.666558254 | 9.848973e-7 | 2.428017e-4 |
| Regular: mixed D | 4.537469e-5 | 1.513245e-8 | 6.829472e-11 | 1.345912e-8 |

All fixed tile absolute-error sums are retained as 47x47 arrays with edges 0,64,...,2944,3000.
Their sums reproduce the corresponding full-panel absolute L1 numerators. Largest axial
tiles have zero-based [tile_row,tile_column]=[22,22] for local m0 and [21,28] for regular;
largest local angular/mixed tiles are [24,22], regular angular 8 is [21,17] and angular 16/mixed
is [11,28]. No observable is cropped to these tiles.

The h-refined fluxes are 0.0011760737319706659 ->0.0005946239440219335 for local m0
and 0.0019084585545165915 ->0.0011451496221475909 for regular. Relative to the old 8 estimate,
flux drops about 49.44%/40.00%; this differs from the fine-denominator flux-delta column above.
The appended 8 accepted contributions are 6.5870952957658884e-6/1.9092034347566005e-4.
Read-only review found no missing 1/N factor or duplicated physical contraction. Both conditional
images therefore fail axial stability decisively despite small angular indicators. Neither
group qualifies the 1% full-image target or <=0.2% reference allowance.

### Cheap strength and response inspection

Canonical signed group strengths were sampled over each full support on 4097 points, preserving
the individual raw per-rod sheets. Their grids identify 317/288 local maxima per sign, with
spacing 0.00132301/0.00125183 A^-1. This coarse peak finder can miss narrow features and removes
no support. Independent physical uniform 256-panel scalar GL8/GL16 strength integration is a
diagnostic, not an image oracle: the regular positive-sheet integrals are 0.007507893452724/
0.007507893452757 (relative difference about 4.4e-12); local m0 gives 0.005891990284875/
0.005883211598509 (about 0.15%). The negative sheets remain separately recorded.

Positive-sheet strength-only Sobol 8/16/32 estimates are 0.00310137/0.00227589/0.00129196 for
local m0 and 0.02842027/0.01647853/0.01324790 for regular. This supports severe undersampling
of structured strengths at these sparse levels. It does not determine the detector integral:
unit response, visibility, mosaic density and spatial transport also vary strongly with u.

Interior unit-strength signed response probes retain canonical source, optics, phase population,
polarization, envelope and mosaic factors. At one declared angular proposal center, 159/160
local and 119/129 regular coordinates survive physical transport. Maximum adjacent mean-hit
motion is 771.14/698.66 pixels. Requested/retained indices and full means/covariance factors are saved.
These sparse motion probes seed refinement; they give no interpolation or image-error bound.

Canonical local stitch compilation gives qc=0.05065406047700051/0.05065545795587627 A^-1
at wavelengths 1.540592925/1.544427 A. Blend endpoints are 3qc and 6qc, about 0.151962/0.303924
and 0.151966/0.303933 A^-1, for each of the three incoherent structure components.
No direct q=0 unit response was evaluated. Local geometry contains exactly 1/q, while low-q
stitched strength behaves as q^2 R(q). A product rule must establish a one-sided limit for
B=qA and integrate S/q, or use separately resolved interior low-q quadrature. Assigning an
arbitrary finite or zero A(0) would be invalid.

### Work, discovery, evidence and next decision

Local h8/new 16 preparation/new 16 h replay evaluated 32224/129504/135264 incoming angular
quadrature nodes in 14.0866/68.9024/62.1469 s. Regular evaluated 124768/97488/100480 nodes in
105.7270/101.3492/90.2776 s. These include rejected preparation nodes; transported kernels
are a subset. Complete processes including saving took 152.8666/305.3656 s; the support
probe including imports/saving took 4.6910 s. No same-work speedup is inferred from these
different meshes. Spatial deposition dominates recorded stages. Pending patch storage keeps
the inherited 256 MiB cap; no process peak-memory measurement is claimed.

The new bounded dream-rsi direction narrowed the prior timeout-prone read-only investigation
to two named code sections. Its worker completed in 71.5129 s and returned no_change; baseline
feasibility evaluation took 5.2528 s. Online replay/policy improvement retained the incumbent
with feasibility_only_no_quality_signal. No changed candidate or numerical-method improvement
was accepted. The worker's hypothetical quarter-weight statement does not apply to physical
angular h subdivisions here; its possible phase-peak seeding hazard remains unverified.

The new immutable external evidence is ewald_coupled_ladder_20261003.ra_diag.npz, 343847792 bytes,
SHA256 6603bd44995d7557392c806de7b077fcd8b1010bc6c632857025d475339f3c14. It contains 100 numeric
arrays and one JSON manifest: all four images per group, appended contributions, meshes,
prefix strengths/PDFs, tile differences, support/response probes, exact runnable source bytes
and raw/canonical-text hashes, immutable input/archive hashes, versions, failures and budgets.
Previous archives retain their recorded hashes.

Initial helper restoration failed before writes because the old manifest normalized source
newlines but hashed raw bytes. Uniform CRLF restoration matched each original helper hash;
the next manifest stores raw decoded UTF8 and separately records canonical-LF hashes.
The first sealing attempt failed on an unavailable private NumPy header reader, before any
array was copied. Its verified empty 22-byte ZIP was removed; public header readers succeeded.
Exact failed source and the repair explanation are retained. No scientific data changed.
The probe's initial inventory exception path was protected before execution.

Charge 900 s conservatively for this entire milestone, including the full 150 s discovery
allowance, failed attempts, imports, checks and sealing. Campaign total 1560/7200 s, remaining
5640 s, within the delegated 1800 s additional ceiling and required >=4740 s remainder.
All scientific phases completed; neither channel has an unfinished partial result.

Recommend a bounded physically seeded axial-panel/product-integration prototype next, after
originating-chat review. Keep separate S_sign(u) and costly native A_sign,p(u), nonnegative
effective weights, every rod/source/sign and all declared support. Seed endpoint, critical,
stitch, elastic/source-region and actual finite-stack fringe coordinates in their own frames;
resolve both signed strengths accurately. Then use complete native midpoint response images
and coupled physical-panel refinement to control interpolation, including tile/flux checks.
Mean/covariance motion alone is insufficient. First test local endpoint factoring and regular
panel geometry in small conditional cases; do not launch a large product campaign yet.

Only this current plan changes in the repository. Formatting/lint/whitespace and clean-state
checks are software checks; no production build change or numerical acceptance is claimed.
Temporary external scripts/results are sealed and removed at handoff. Main remains clean at
4cf84defb0fd1326dd98714544dcfda44049976d; no merge, push, full image or fit ran.

## Third delegated milestone: positive product integration (complete, accepted by originating chat)

The originating chat accepted the coupled diagnosis and authorized at most 1800 additional
numerical/evaluator seconds for a positive strength/response prototype. Its review then requested
an independent direct reference on the same selected interval, capped at 1000 additional seconds
inside that allocation. This extension is included below. No new production implementation,
full-support native image, full-source campaign or fit was authorized or launched.

### Canonical separation and cheap positive weights

For source 0/group 32 retain the same six individual rods, populations, both signed sheets,
original 85-rod/32-source iterator and complete 3000x3000 native pixels. Write the conditional
intensity as sum_sign integral S_sign(u) A_sign,p(u) du. S is the canonical strength table,
including rod populations and incoherent structure probabilities once. A uses unit strength for
one sign and zero for the other, axial weight 1, physical angular weights, original optical/source/
polarization/envelope coefficients and spatial order 16. No division by a small S, Sobol PDF,
1/N factor, image normalization or detector solid angle enters this product rule.

The actual three incoherent components are CifFiniteStackStrength with 13 full-cell repeats.
The normal cell length is c=28.636 A, highest repeat extent bound 13c=372.268 A. Physical fringe
coordinates follow L and signed u=+/-[b3 L + rod normal offset], with repeat-factor zeros at
L=m+k/13. A cheap strength-panel cap pi/(2*13c)=0.004219530893858448 A^-1, both signed fringe
knots, response-hat breakpoints and support endpoints resolve full support [0,5.127512118222034].
Expensive response grids and cheap strength grids are separate.

For response grids of 65/129 nodes, W_i,sign=integral S_sign H_i du is nonnegative and preserves
partition mass and first moment without renormalization. GL8 base, same-panel GL16 and half-cap
GL8 weights agree within 8.1e-15 in signed relative weight L1. Base/p/h cheap node counts are
11160/22320/20880 for 65 responses and 11672/23344/21392 for 129. Signed full-support masses
are [0.0075078934527568,0.0075070106468658]. These are scalar strength results; neither whole
response grid has a native image accuracy result.

The native pilot interval is [1.0415258990138507,1.12164327586107], width
0.0801173768472192 A^-1. A physically resolved strength peak times the earlier explicitly
limited single-center response proxy selected it. This choice is a useful mechanism pilot,
not evidence that the interval represents all support. Interval hats use GL16 on half-cap
cheap panels and preserve both signed mass/first moments.

Fresh angular preparation at each of nine response coordinates uses a sign-weight envelope
covering the 2/3/5/9-node hats, relative empirical indicator 1e-4 and absolute budget 1e-14.
Independent angular h subdivision follows preparation at each unchanged u. Actual product
weights revalidate the signed preparation and h triangle ledgers. Canonical event mass equals
S_positive*A_positive+S_negative*A_negative at every evaluated batch within 4.4e-16 relative.
One same-node native angular panel per evaluation agrees within 1.5e-16 relative L1; that
bounded check is not an independent full-point native parity proof.

The h-phase 2->3, 3->5 and 5->9 image differences are 45.7675%, 16.0744% and 2.8940% L1,
while their flux changes are only 5.11e-5,1.35e-5,2.55e-6 relative. Corresponding angular L1
differences are 2.45e-7,2.30e-7,4.01e-7,3.14e-7 for 2/3/5/9 nodes; mixed axial/angular
L1 differences are 2.07e-7,3.40e-7,1.87e-7. These are differences between estimates.
They reject the coarse response grids, but do not establish the true error or excessive cost
of a refined product method. The originating review required the direct comparison below.

### Independent physical-panel interval reference

Before evaluation, freeze four equal physical u panels, GL4 base (16 nodes), same-panel GL8
p refinement (32 nodes), and eight-panel GL4 h refinement (32 nodes). Each base panel spans
at most about 1.19 highest-repeat intensity fringes and two nine-node response spacings.
This resolves physical strength and moving pixel response together, without assigning GL8
to every cheap scalar panel. Both signed canonical strengths enter direct event masses;
the signed masses combine before projection, as in current production. Stream patches into
complete native images; no panel by nine-million-pixel array is materialized.

Each direct image has a whole-interval physical angular ledger: relative empirical indicator
5e-5, absolute budget 1e-15. Empty or failed paths cannot qualify a reference. Saved direct
nodes, physical weights, both canonical strengths and accepted angular meshes permit replay.
The reference R is the p-refined image. All L1 denominators below are sum(R), except the
separately stored base->h refinement, whose denominator is sum(H).

| Direct comparison | Relative L1 | Relative flux delta | Max absolute pixel | Max 64x64 tile absolute sum |
| --- | ---: | ---: | ---: | ---: |
| Base16 -> P32 | 1.9857515e-4 | -5.0717314e-5 | 6.1044656e-11 | 1.6860411e-8 |
| Base16 -> H32 | 1.9864601e-4 | -5.0819197e-5 | 6.1082516e-11 | 1.6867380e-8 |
| H32 -> P32 | 1.4400368e-7 | 1.0187807e-7 | 2.5042383e-13 | 8.9593504e-12 |

Reference flux is 0.00018445193163813144. The frozen maximum empirical allowance is
4.9999956e-5. Use the stricter additive allowance U_R=(||P-H||1+Eang_P+Eang_H)/sum(P)
=1.0014372734663848e-4 (0.0100144%), below the 0.2% ceiling. Report the observed p/h
difference separately from this allowance. Neither the difference nor the preparation
indicators rigorously bound true error. Direct quadrature is independent of response-hat
interpolation; both paths share accepted physical kernels and spatial 16.

### Product error against the direct reference and response count

Eight additional midpoint responses complete 17 nodes. Their old-node hats are componentwise
bounded by the previously used envelope and old coordinates agree exactly. The actual 17-weight
preparation ledger is 9.9477183e-5 relative; angular h L1 is 2.2326610e-7 and its signed triangle
indicator is 3.0297259e-7. Preserve accepted/h signed images separately. The 9->17 h difference
is 0.00753956943 relative L1, with flux delta 5.9457703e-7, maximum absolute pixel
2.7000765e-9 and maximum tile absolute sum 6.6727166e-7.

E1=sum(abs(product_h-R))/sum(R). Combined=(E1+U_R)/(1-U_R), using the additive allowance.

| Product nodes | E1 | Relative flux delta | Max absolute pixel | Max tile absolute sum | Combined | Empirical interval gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.603524336 | -6.7972532e-5 | 1.8211994e-7 | 5.2823054e-5 | 0.603684935 | FAIL |
| 3 | 0.197617598 | -1.6916323e-5 | 6.8631818e-8 | 1.7452196e-5 | 0.197737544 | FAIL |
| 5 | 0.038899584 | -3.4181244e-6 | 1.3768459e-8 | 3.4374647e-6 | 0.039003634 | FAIL |
| 9 | 0.010072676 | -8.6719500e-7 | 3.6118197e-9 | 8.9151302e-7 | 0.010173839 | FAIL |
| 17 | 0.002533143 | -2.7261813e-7 | 9.1174323e-10 | 2.2424245e-7 | 0.002633551 | PASS |

Seventeen is the first tested passing product count on this interval, not the minimum possible
count. Nine narrowly fails the combined 1% gate; seventeen passes at 0.263355%. Direct Base16
also passes against R, combined 0.000298748798 (0.0298749%). Thus this interval demonstrates
no cold expensive-response-count reduction for product integration versus the 16-node direct
rule. Flux agreement alone would have missed the product shape errors. All fixed tile arrays
are retained; the largest direct/product error tile is [21,28] (native rows 1344:1408,
columns 1792:1856), with no crop of any observable.

### Work, endpoint restriction, discovery and handoff

| Evaluation | Axial coordinates | Incoming angular nodes | Accepted angular nodes | Preparation seconds | Strength/event/spatial seconds |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct Base | 16 | 202976 | 104560 | 108.4483 | 0.1003 / 9.8258 / 76.2074 |
| Direct P | 32 | 425056 | 218672 | 209.2398 | 0.1137 / 20.4844 / 151.4077 |
| Direct H | 32 | 428768 | 220528 | 211.7715 | 0.1154 / 20.6171 / 153.4694 |
| Unit-response old9 prepare | 9 | 108896 | saved by point | 102.4088 | saved by point |
| Unit-response new8 prepare | 8 | 97920 | saved by point | 91.4053 | 0.1300 / 5.4876 / 72.1944 |
| Unit-response new8 angular h | 8 | 100992 | 100992 | 74.4458 | 0.0996 / 0.4279 / 72.5531 |

Old9 angular h adds 112352 nodes/84.1753 s. Direct evaluation combines the current signed
physical masses before deposition. This reusable unit-response experiment renders two signs
separately, then renders them again for angular h. Different angular tolerances, accepted meshes,
work and setup are included above; their timings are not a method speed ranking. Highest retained
live direct panel-array count is 430072 bytes under the inherited 256 MiB cap. Full native buffers
are separate from that cap; no process peak-memory measurement is claimed.

A cheap local scalar endpoint audit, completed before the interval extension, checks strictly
interior q at both actual wavelengths. S/q^2 approaches about 107.6956101/108.2819576 and
S/q vanishes linearly. Positive GL16/32/64 integration of S/q over [0,qc] refines to
0.112803645284293/0.113350854878139 per sign. This does not establish the native B=qA limit.
The future local route is separately resolved interior low-q physical quadrature; no arbitrary
A(0), native endpoint response or local image was evaluated in this milestone.

The bounded dream-rsi audit uses the actual frozen scalar-weight evaluator and feasibility-only
score. Its baseline passes in 0.25755 s; the sole read-only worker times out at 45.47312 s.
Replay/policy improvement retains the incumbent with no quality signal or changed candidate.
The executor timeout is not scientific rejection. Initial configuration used an unsupported
key; correcting evaluator_timeout_s and the constraints type enabled the sole retry. Preserve
both exact configs and logs. Prototype empty-support/partial-failure bookkeeping was repaired
in the new midpoint helper before reuse; original completed nine-point evidence is unchanged.
A wholly empty direct interval would be marked incomplete by its writer's missing strength cache;
this nonempty completed interval is unaffected. These temporary APIs are not promoted.

Immutable evidence: ewald_product_pilot_20261003.ra_diag.npz, 112131639 bytes, 235 numeric arrays
and one JSON manifest, SHA256 51385ca2564b14a44da5c7bd85aa6e61ac449e19a1d7d165497d43a0a1a3c0c3.
It preserves signed native levels, direct images, physical/scalar/angular meshes, tile measures,
exact runnable evaluator versions including the initial comparison, inputs, production hashes,
failed configuration, read-only discovery records and the frozen reference declaration. Prior
immutable archive hashes are unchanged. Sealing and integrity checks took 31.2656 s.

Measured launcher phases total 984.7668 s, including imports, saving and the cycle; the requested
interval extension uses 719.0495 s, below its 1000 s cap. Charge this complete milestone 1500 s
conservatively, including the full 90 s discovery allowance, failed attempts and closure reserve.
Campaign charge is 3060/7200 s, with 4140 s remaining; the 1800 s milestone ceiling and required
>=3840 s campaign remainder are preserved. Allocated charge is not measured wall time.

Next decision for originating review: keep the physically resolved direct rule as the cold
interval baseline. Positive hats are viable at 17 responses here but have no demonstrated cold
count advantage. Any further product optimization needs a concrete reuse benefit and an equal-work
comparison, potentially combining weighted signs before deposition while preserving reusable
geometry. Neither interval result selects a whole-support grid or qualifies a full detector-source
image or fit. No additional campaign is started at handoff.

Only this plan changes in the repository: production +0/-0, retired development infrastructure
+0/-0. Formatting/lint/whitespace, source/input hash and clean-checkout checks are scoped to this
change and evidence integrity. Software checks do not establish scientific adequacy. Temporary
external sources/results are sealed and removed. Main remains clean at 4cf84defb0fd1326dd98714544dcfda44049976d;
research receives one coherent documentation checkpoint, with no merge or push.

## Fourth delegated milestone: positive strength-weighted Gauss rule (complete, accepted by originating chat)

The originating chat accepted the interval comparison and allocated at most 900 further
numerical/evaluator seconds, including discovery, failures and closure, with >=3240 s campaign
remainder. This is a short discriminating screen on the same regular interval/source0/group32.
The full 85-rod/32-source detector target remains open. Main, defaults, physical factors, cone
and spatial rules, qualification states and earlier immutable evidence remain unchanged.

### Positive measure, stable recurrence and source support

Use W(u)=S_positive(u)+S_negative(u) and F_p(u)=sum_sign [S_sign(u)/W(u)] A_unit_sign,p(u).
Then I_p=integral W F_p du. Evaluate both ratios explicitly from canonical signed strengths;
obtain A through the existing unit-strength event path and combine the signed weighted masses
before one native projection. A zero/nonfinite W or unsafe generated-node ratio fails; no floor,
silent mass removal or division of an existing image by strength is permitted. The original
source/group iterator, six individual rods, native pixels and all optical/source/envelope factors
are preserved. At all completed batches the explicit unit-path contraction agrees with the
canonical ratio-table event mass within 4.32e-16 relative.

The construction follows [Gautschi's discretized Stieltjes procedure, section 2.2](https://www.cs.purdue.edu/homes/wxg/selected_works/section_04/081.pdf)
and the [NIST DLMF Jacobi eigenvalue/weight construction](https://dlmf.nist.gov/3.5#vi).
These sources support positive-measure quadrature and stable recurrence construction; the
following detector results are independent measured evidence for this interval.

Scale the finite interval to [-1,1]. The cheap canonical discretization already includes
physical du, so scaling adds no second Jacobian. Lanczos multiplication by the scaled
coordinate uses two explicit reorthogonalization passes. A symmetric tridiagonal eigensolve
returns nodes and first-eigenvector-component squares. The recurrence alone uses mass-normalized
discrete weights for conditioning; final Gaussian weights multiply the exact canonical physical
mass back. The native observable receives no normalization.

Fringe/cap-seeded cheap scalar base/p/h rules use 43 panels GL16 (688 nodes), the same panels
GL32 (1376), and half-cap 81 panels GL16 (1296). The base cap is pi/(4*13c)
=0.002109765446929224 A^-1. Total physical mass is 0.005087096187815536, signed masses
[0.002543836828335796,0.0025432593594797385]. Genuine zero discrete masses can remain;
no positive mass is discarded. Generated rules at n=4/8/12/16 all have finite positive weights,
interior nodes, correct total mass and Legendre-basis exactness through degree2n-1 within
1.20e-15 normalized to physical mass (frozen gate 5e-13). Basis orthogonality errors remain
below 6.4e-16. Across independently p/h-refined scalar measures, maximum physical node shift
is 4.44e-16 A^-1 and relative weight L1 is below 2.50e-14. Minimum generated W for n4/n8
is 0.000310988491/0.001266417892; both signed ratios remain finite and in [0,1].

### Native screen and separate scalar contribution

Reuse the reviewed direct P/H reference after archive, source, input and exact runtime checks:
Python 3.13.13, NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0, main's requested interpreter with research
src first. Every new native image is complete 3000x3000 for interval
[1.0415258990138507,1.12164327586107]. Its physical quadrature weights enter the angular nodes' integrated mass once;
canonical signed ratios combine unit masses before spatial16 deposition. Angular preparation
uses current-weight whole-interval empirical 5e-5 relative and 1e-15 absolute ledgers.

| Image against reviewed R | Relative L1 | Relative flux delta | Max absolute pixel | Max64tile absolute sum |
| --- | ---: | ---: | ---: | ---: |
| Gauss4 prepared | 5.0502971e-5 | -1.6283172e-8 | 1.9674430e-11 | 4.7309282e-9 |
| Gauss8 prepared order screen | 1.4482586e-7 | 2.3436847e-8 | 7.8279330e-13 | 1.4073224e-11 |
| Gauss4 angularh | 5.0514326e-5 | -7.4285809e-9 | 1.9674693e-11 | 4.7309308e-9 |
| Gauss4 scalar-p, fresh preparation | 5.0502971e-5 | -1.6283157e-8 | 1.9674430e-11 | 4.7309282e-9 |

The prepared4->8 order difference is 5.0583481e-5 relative L1, normalized to R. Independent
angularh subdivides the four-node accepted physical/Cauchy-CDF panels at unchanged u. Its
full-native L1 difference from preparation is 2.8143746e-8, flux delta 8.8545914e-9, max pixel
1.1138930e-13 and max tile 2.1073710e-12. The four-node physical preparation ledger is
4.9987702e-5 normalized to R. Eight is an order screen only; it receives no separate final
angularh acceptance. Twelve/sixteen pass scalar construction only and have no native images.

Scalar-p generated nodes differ by at most one physical ulp; fresh angular preparation and
projection measure scalar-p versus base image sensitivity at 1.5661718e-14 relative L1,
max pixel 5.4527746e-21 and max tile 1.3601740e-18. Keep this observed sensitivity separate
from any uncertainty allowance. Scalar-h at n4 has exactly identical physical nodes, canonical
signed strengths and ratios. For the same fixed angularh response, positive intensities imply
||I_hscalar-I_base||1 <= eta*flux_base, where eta=max_i abs(delta_w_i/w_i)
=7.8907725e-15. This supplies a normalized native perturbation bound 7.8907724e-15. Exact
identity and maximum relative weight perturbation are checked; tiny node shifts or relative
weight L1 alone would not justify this bound.

Use a conservative separate scalar allowance: U_scalar=max(observed scalar-p native L1 plus
both base/p empirical angular ledgers, identical-node scalar-h weight bound) plus the observed
candidate angularh difference. This is 1.0000354802099959e-4, normalized to R. It is an
empirical continuous-measure allowance; only the fixed-response weight perturbation bound
above is analytic. The reviewed reference U_R remains 1.0014372734663848e-4, below 0.002.
It is not inflated to hide scalar construction error.

For chosen I=Gauss4 angularh, E1=5.05143257232967e-5 and
(E1+U_R+U_scalar)/(1-U_R)=0.00025068670579204906 (0.0250687%), passing the empirical 1% interval
gate. Four is the first tested passing common response count. Direct16 already passed the
same reviewed reference, so this demonstrates a fourfold reduction in expensive axial response
count relative to the tested direct16 baseline on this interval. The smallest adequate ordinary
rule count is not established. No runtime ratio or full-image error bound follows.
Fixed 47x47 tile arrays and all signed strengths/ratios/weights/meshes are retained. Largest
Gauss4/reference error tile remains [21,28]; no observable is cropped to a selected tile.

### Work, audit, evidence and decision

| Stage | Common response nodes | Incoming angular nodes | Accepted nodes | Seconds | Strength/event/spatial seconds |
| --- | ---: | ---: | ---: | ---: | --- |
| Scalar construction, all rules | cheap only | none | none | 0.3122 | not a native evaluation |
| Gauss4 preparation | 4 | 53312 | 27424 | 34.5676 | 0.1750 / 2.9991 / 22.4277 |
| Gauss8 preparation | 8 | 107360 | 55216 | 61.7517 | 0.3036 / 5.9719 / 41.7621 |
| Gauss4 angularh replay | 4, same nodes | 54848 | 54848 | 26.4202 | 0.0557 / 0.2422 / 23.4837 |
| Gauss4 scalar-p preparation | 4 | 53312 | 27424 | 34.2375 | 0.1719 / 2.9728 / 22.1633 |

Construction, preparation and replay are separated; stages include first-call setup/JIT.
Both cold Gauss and reviewed direct rules use the existing batching/spatial16 and combine
signed weighted masses before deposition. Different u locations lead to different accepted
angular work. No repeated matched timing benchmark ran, so these times are observations,
not a method speed ranking. No process peak-memory claim or whole-support count extrapolation
is made. The physical measure is parameter-dependent; fixed-geometry strength sweeps remain
unqualified.

The single <=60 s read-only dream-rsi cycle uses frozen scalar feasibility, with no detector
quality score. Baseline passes in 0.20877 s; the narrowed worker still times out at 35.46243 s.
Replay/policy improvement retains the incumbent with no quality signal or changed candidate.
This executor failure does not reject the measured rule. Initialization itself succeeded, but
a consumer interpreted structured JSON output as a plain path. Parse the recorded experiment
field and recover that same experiment; no duplicate initialization or new worker retry ran.
Exact failed caller, recovery source, structured result and cycle logs are archived.

New immutable evidence: ewald_strength_gauss_20261003.ra_diag.npz, 9558108 bytes, 130 numeric
arrays plus one JSON manifest, SHA256 f65ab765a94d9dc9c6c0d78ef3a32dc333d854a94fda801e44b32864438f0124.
It includes exact sources, all scalar rules and canonical sheets, four completed native images,
three unchanged reviewed direct images, angular meshes, tile measures, independent sensitivities,
inputs/source/runtime identities, frozen declaration, failures and budget. Earlier archives
retain their hashes. Sealing and integrity checks took 4.0536 s.

Measured launcher phases total 207.8071 s; discovery initialization/recovery/cycle total 40.1693 s,
within 60. Charge 450 s conservatively for this milestone, including full 60 s discovery allowance,
failed parsing, imports and 150 s closure reserve. Campaign cumulative 3510/7200 s, remaining 3690 s,
within the 900 s milestone ceiling and required >=3240 s remainder. This charge is an allocation,
not actual wall time.

Decision: GO for a next bounded optimization experiment using the positive common-measure rule,
based on verified 4-versus-16 response count. The next review must choose where reuse and matched
cold work justify implementation. This milestone promotes no production rule and starts no
full-support, local endpoint, full-image or fitting campaign. Conditional agreement fixes
cone/spatial/source rules; the original 85-rod/32-source scientific target remains open.

Only this plan changes: production +0/-0 and retired development infrastructure +0/-0. Relevant
formatting/lint/whitespace and evidence-integrity checks pass; they do not establish scientific
adequacy. Temporary external files are sealed and removed before handoff. Main remains clean
at 4cf84defb0fd1326dd98714544dcfda44049976d; research receives one coherent documentation checkpoint,
with no merge or push.
