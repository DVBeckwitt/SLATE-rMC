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

## Second delegated milestone: coupled integration diagnosis (complete, awaiting review)

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
