# Native angular response panels

## Authorized plan

Implement the demonstrated GD1 angular-resolution repair in an isolated branch
from clean main `4ea34f5`, carrying the previously audited numerical prerequisites
through `1747f22`. No fitting, check-ins, old queue restarts or promotion of old
predictions. Keep all failed evidence unchanged.

The local RCA `gd1_parent19_root_20260918.ra_diag.npz` demonstrates that actual
native-bin deposition is correct at three contributing kernels but the whole-arc
angular rule undersamples the response. Physical-angle panels agree with two
independently refined angular references. This is not full-fit qualification.

1. Add a public analytic adjacent-bin Gaussian-crossing regression before edits.
2. Add immutable physical-angle panel edges, separate from support intervals.
   Intersect the fixed partition with the original merged support, integrating
   every interval exactly once with physical-angle Gauss weights. No CDF/proposal
   normalization is introduced in this explicit path. Preserve the old path when
   no mesh is supplied. Angular power is the order per panel, never silently capped.
3. Add deterministic geometry-bound seeding and explicit panel/node work limits.
   This prepares a candidate mesh, not a numerical certificate or an automatic
   optimal/adaptive algorithm. A common frozen support envelope remains necessary
   for identical angular nodes across changing geometry/source candidates.
4. Bind the mesh to native overrides, revisions and cache identities. Reject
   unsupported or ineffective combinations. Preserve observations, masks, sources,
   rods, axial coordinates, local-m0 conventions, physical factors and covariance.
5. Test the actual GD1 axial1067 with all original observations in bounded serial
   blocks, comparing a generic mesh to an independently refined physical-angle
   reference. No parent-specific production constants. Check cost and peak memory.
6. Review independently, run focused and compact software/proof gates, retain one
   external NPZ, remove temporary harnesses, and commit one coherent repair.

Owned paths: `pipeline/fiber_detector.py`, `pipeline/conditional_detector.py`,
`fitting/native_workflow.py`, focused integration and
native-workflow tests, `docs/NATIVE_REFINEMENT.md`, this task, task index and file
inventory. Ledger: PHY-FIT-025. No reference/equation/tolerance edits.

Acceptance is the supported numerical mechanism and its focused proofs. Complete
source/family, all-sample, endpoint/stencil and optimizer qualification remain
separate. No production-speedup claim follows from a one-row pilot.

## Implementation and first divergence

The physical-angle path bypasses the proposal angular CDF/PDF and uses the
original axial measure times each clipped panel's physical Gauss weight. The
original support merge is shared, not duplicated. Immutable edges bind to the
native revision, serialized integration rule and response-cache identity. Default
`None` retains the old proposal algorithm and revision fields. Node counting
precedes angular allocation; work-limit failure never returns a coarse fallback.

The correction is at angular integration of the native-bin response, after
pointwise deposition agreed with an independent pixel-probability oracle in the
sealed RCA. The new analytic adjacent-bin regression originally failed: the old
power7 integral disagreed and the explicit-panel API was absent. The permanent
test asserts each child's analytic mass, not just their cancelling parent total.
It also protects wrapped/overlapping support, seed boundaries, invalid-input and
work-limit rejection, and proposal-independent physical nodes. Existing joint
measure and native parity tests were extended to protect CDF-axial weights and
revision/list-to-tuple binding. No test requires the old algorithm to stay wrong.

Legacy classifications remain unchanged; the immutable reference pack is not an
oracle for this observed GD1 angular-resolution failure. This local numerical
comparison is CORRECTED by analytic bin integrals and independently refined
physical-angle integration. It does not establish an independent complete
physics/model oracle.

## Bounded real-data proof and cost

The original GD1 family1/source8 axial1067 contribution uses N72, all six signed
rods, the exact saved axial weight, the complete original angular union and all
1,226 original observations. Geometry seeding includes every observation, without
parent-specific constants, source renormalization, mask changes or bin grouping.

The public sampler's width 0.001 candidate has 33,864 nodes; width 0.0005 has 56,968.
Independent full-union physical GL8 references use 47,840/95,680 nodes at
width 0.0005/0.00025. Relative discrepancy is maximum absolute vector difference
divided by the finer reference peak, not an elementwise relative or fit-gate RMS:

| Comparison against finer reference | Relative discrepancy |
| --- | ---: |
| Seeded width 0.001 | 4.799896364567665e-13 |
| Seeded width 0.0005 | 1.5208799030084124e-15 |
| Independent width 0.0005 | 3.1197536471967433e-15 |

Original fixed-scale maximum discrepancies are 1.8773e-9, 5.9484e-12 and 1.2202e-11
counts respectively. The original parent19 reference is reproduced as a subset,
while the acceptance comparison retains all 1,226 outputs. Angular mass and sine
moment invariants pass independently.

Supervised serial blocks finish and are reaped in 15.7519294 and 18.2219985 seconds
(33.9739279 total), below separate 115-second caps. Peak process working set is
450633728 bytes. Candidate/reference evaluation timings include different JIT
warmup states and concurrent software tests; they are not a speedup benchmark.
Both candidate and oracle evaluate the same observable to the declared local
1e-8 reference-peak tolerance, with their measured node/time/memory costs retained.
The whole-call node budget may reject a fine mesh spanning a complete family's
axial domain. No full-family resource bound or fit speedup has been demonstrated.

Evidence: external `native_angular_panel_repair_20260918.ra_diag.npz`. Retained
arrays include all original output rows, full vectors, nodes, physical weights,
seed bounds and independent-reference results. Source/input hashes, executor,
timings, shutdown, reviews and software gates accompany the arrays.

This is one axial/source/family contribution. Full source/family sums, other
failed samples, all controls, actual endpoint/stencils and optimizer convergence
remain unqualified. No numerical width is declared universally sufficient. Fits
and check-ins remain stopped; production inputs and all prior evidence stay intact.

## Software handoff

All 380 permanent tests pass with normal Windows process access; 14 expected
CUDA low-occupancy warnings remain. Ruff and formatting pass for all 167 Python
files. No type checker is configured. Six precommit registered proofs pass; the
core proof alone required a temporary-directory permission retry. Its one empty
scratch directory was inspected and removed. The 394-file/12-reference-case
inventory passes. Clean-commit geometry/optics and mosaic/Ewald proof records
are appended post-commit to the external evidence.

Two independent code reviews clear the patch after allocation-preflight and
input-validation fixes. An independent saved-array audit reproduces every metric
exactly and verifies all three evidence-input and 123 current-source hashes.
The two in-memory mutations (ignore angular mesh; double physical angular mass)
are both detected at the individual adjacent-bin analytic comparison.

Public additions are the optional `angular_panel_edges_rad` and
`maximum_angular_panel_nodes` rule/sampler fields, plus `seed_angular_panel_edges`.
Only one new permanent test is retained; two existing tests gain distinct branch
and native-binding coverage. No dependency, extra production module, diagnostic
writer, hidden normalization or tolerance change is introduced.

Temporary external executors are embedded in the single retained NPZ, then
removed as loose files. Integration requires this repair and its audited
prerequisites from the isolated branch; no old failed executor is resumed or
result relabeled. Scientific acceptance remains limited to the scope above.
