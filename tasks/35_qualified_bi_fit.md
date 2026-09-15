# Qualified Bi2Se3 fixed/free structure-factor comparison

Base: `05de593df436f9f9d0ea8d6482f96acb041bf1d7`. One writer; agents read-only.
User authorizes completing the full39 fit, fixed13 control, N alternatives,
ambiguity profiles and detector comparison. Relevant ledger: PHY-FIT-023/024/025.

## Plan before implementation

1. Replay the saved full endpoint through consolidated production code. Recenter
   the numerical mosaic proposal on the saved physical mosaic, freeze it across
   actual Lorentzian-width optimizer steps, and refine angular order independently.
   Preserve observations, covariance, source, N, full rods, normalized proposal,
   training split and existing numerical gates. Stop broad fitting while they fail.
2. Close two demonstrated public-runner gaps: an explicit fixed-parameter control
   roster and a local-m0 axial panel cap. Preserve ordinary full-release validation;
   conditional controls retain fixed values throughout fitting and qualification.
   Identification profiles remain the responsibility of the released full model.
3. Resolve remaining numerical components at the saved endpoint, then qualify the
   actual optimizer stencil. Refit fixed/free models under identical accepted rules,
   compare neighboring N, profile weak coordinates, and render accepted candidates.

No physical equations, thresholds, observations, or dependencies change. Numerical
proposal adjustment is not qualification by itself. No new optimization project.
Permanent regressions cover distinct public boundaries: channel-specific integration
equals explicit additive partitions; fixed-control declarations preserve fixed values
and cannot bypass full-release validation. External numerical artifacts retain input,
code and candidate identities, once-profiled scales, timing, memory and gate results.
Run compact suite, formatting/lint, registered proofs and inventory before handoff.

## Implemented boundary and measured outcome

The public runner now supports an explicit fixed-parameter control roster and a
local-m0-only axial panel cap. The latter reproduces the previously split raw
response through one evaluator: relative differences below 2.7e-17 at the saved
full candidate and two gamma probes. No numerical threshold or physics changed.
The compact suite passes 366 tests; formatting/lint and the six scientific proofs
that permit an uncommitted checkout pass. The two clean-checkout wrappers and
inventory are final handoff gates, not fitting qualification.

The consolidated old-rule replay agrees with the saved raw prediction to 4.0e-17
relative norm. Its angular proposal Lorentzian width was approximately 585 times
the fitted width. Recentring the numerical proposal is necessary but insufficient:
successive angular g5/g6, g6/g7 and g7/g8 comparisons have endpoint disagreements
6.80, 1.69 and 0.548 sigma. At g7/g8 the gamma objective-contrast error is 3.95,
above the existing 0.5 gate. The isolated local-m0 contribution passes that last
comparison; other rods do not. Axial and source qualification remain unresolved.
The previous gamma step is about 8.8% of its fitted width and also needs a smaller
step and a step-halving check before derivative-based termination is credible.

An independent fixed-Q/source slice identifies narrow detector-acceptance windows
undersampled by the global angular proposal. Physical midpoint integration at
32768/65536 angles agrees to 2.1e-10 sigma over all 1250 observation rows. Retaining
all region endpoints, tighter geometric boxes or uniformly finer physical panels
does not establish a lower-cost replacement. A temporary adaptive GL8 experiment
uses 920 retained nodes versus 2048 for CDF g10, with 1.56e-5 versus 2.60e-5 sigma
slice error. However, adaptive construction costs 4968 evaluations and 0.567 s;
the 0.100 versus 0.269 s frozen projection timings are not optimizer speedups.
The production fitter already caches those projections. No adaptive engine,
alternative physical model, fitted result or qualified uncertainty is introduced.

User priority: lightweight, fast full-SF optimization. Preserve the persistent
evaluator for response-only blocks and use existing staged fitting, finishing
with all admitted coordinates released. Parallel prediction groups create new
evaluators and must not be presumed faster than persistent serial evaluation.
Any future integration change must beat equivalent response construction and
actual cached fitting work, including memory, before adoption. Avoid further
blind global order sweeps or a new adaptive framework based on one slice.

The measured full/free and fixed-SF fits, N alternatives, ambiguity profiles and
accepted detector comparison are not completed by this software boundary. Failed
numerical qualification remains explicit; neither old candidate is promoted.

Evidence resides outside the repository under
`C:/Users/Kenpo/.codex/visualizations/2026/09/11/01a09194-718e-7e83-ab46-0fc959c9d928`:

- `bi2se3_current_proposal_qualification_20260914.ra_diag.npz`: complete predictions,
  actual probes, partition attribution, timing and 5.03 GB peak working set.
- `bi2se3_qualified_runner_boundary.ra_diag.npz`: unified/split public-boundary parity;
  three evaluations take 90.34 s with concurrent work, not a speedup measurement.
- `bi2se3_first_family_current_proposal_20260914.ra_diag.npz` and
  `bi2se3_first_family_angular_slice_20260914.ra_diag.npz`: source/axial attribution
  and independently converged angular slice.
- `bi2se3_angular_slice_cost_screen_20260914.ra_diag.npz`,
  `bi2se3_angular_all_regions_cost_screen_20260914.ra_diag.npz` and
  `bi2se3_angular_parent_panels_cost_screen_20260914.ra_diag.npz`: rejected production
  candidates and the distinction between construction and cached evaluation cost.
- `qualified_bi_software_proofs_20260914.ra_diag.npz`: software checks and final
  handoff evidence. Each artifact embeds its own numerical arrays and JSON manifest.

Permanent coverage added: one fixed-control public-CLI boundary test. The existing
native-response test additionally protects local-m0 partition additivity and
numerical-revision invalidation. Existing legacy classifications remain unchanged;
these new boundaries are NO_ORACLE against legacy and use exact same-core parity.
