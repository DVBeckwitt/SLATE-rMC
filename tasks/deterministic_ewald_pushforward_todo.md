# Deterministic Ewald pushforward checklist

Status: **CORRECTED PLANNING ONLY — re-baselined after the completed BKI merge; implementation is
blocked pending human re-review and the post-merge reconciliation/base/certification gates below.**

Authoritative specification: `tasks/deterministic_ewald_pushforward_plan.md`.

This tracked checklist is not edited after every code task: doing so would violate the plan's exact
<=5-file lists. Record interim progress in the external run log/commit handoff. Synchronize this
file only in DP-06J and DP-15, and check an item only when its named evidence passed.

## Base, BKI, versions, and proof authority

- [ ] Record that merged `main` `3af2f4f61d3bc73d8d54891eb867a964240f46cd` incorporates completed
      BKI implementation `d5eed2524a636c7c190b8f9be300d1d73728884e`; never claim this correction
      preceded BKI-15.
- [ ] Record planning-only beam findings commits
      `af456b5988076c400eea972fb4f2ac2d647891dc` and
      `0953cab668b489fd4c4d5a1d4de070fcab165295` (or their approved replacement), and reconcile their
      committed BKI/deterministic/parallel-plan paths plus the observed dirty T09/T10/T11/parallel-
      plan patch by exact hash; never overwrite the sibling worktree.
- [ ] DP-00R: preserve BKI-00--BKI-17 completion history, replace stale parallel/coating successor
      edges with the sole overlapping sequence “implemented/accepted shared Checkpoint K, then this
      deterministic plan,” and record the exact corrected-planning and approved rebase/merge hashes.
- [ ] In DP-00R, refresh/review `FILE_MANIFEST.json` for its exact three-file task and rerun the
      affected BKI-15 documentation, all-six-proof, seed-integrity, diff, and status commands.
      Require docs/seed and non-baseline proof stages to pass; record only the expected reciprocal/
      stacking proof-base-contract failures for DP-00B.
- [ ] Record BKI-04A/BKI-14 as historical >5-file process deviations with owner-approved audit
      dispositions and independent revalidation; do not call either task <=5-file compliant.
- [ ] Reconcile unrelated dirty/untracked/status-warning paths without overwriting user work;
      require warning-free status with every untracked path visible before DP-00.
- [ ] Preserve the unique coating-validation ref/commit until its owner approves a port/archive
      disposition; never reuse or silently delete it.
- [ ] Require the shared beam-to-`ki` branch to implement and accept Checkpoint K on approved main;
      a planning commit is not acceptance. Build/hash the complete source once, build the complete
      incident table once and serially, retain `parent_row_index`, never ID-sort/rejoin or hash a
      public slice, and preserve `source_weight=1` for the nominal one-row source.
- [ ] DP-00: record `PUSHFORWARD_BASE_SHA`, reference hashes, current API/trace/reference versions,
      accepted Checkpoint K commit/API/revision hashes, accepted-domain inventory inputs, and a clean exclusive worktree on
      `codex/deterministic-ewald-pushforward`.
- [ ] Re-audit the exact post-BKI modules, imports, tests, scripts, task files, fixture data, and
      public domains; any mismatch gets a new <=5-file plan amendment before implementation.
- [ ] DP-00A: preserve `stage_tolerances_v1.json` byte/hash exactly; add schema-distinct v2 with
      scalar/global-L1, tangent, topology, indicator, ledger, and `L_peak` stages.
- [ ] Make v2 a strict schema superset whose inherited v1 stages are numerically identical. Add
      named v1/v2 loaders and SHA constants; keep the no-argument loader temporarily fixed to v1.
- [ ] Prove v2 uses candidate-independent scales, rejects elementwise misuse of global L1, rejects
      uncertified estimates and v1/v2 mixing. Freeze separate legacy/reserved trace constants;
      activate the reserved trace exactly once in DP-05V1. Existing proofs remain v1 until
      DP-05V1/V2; new deterministic stages explicitly use v2.
- [ ] Update validation atomically so v1 remains historical and v2 is the current authority before
      observing coating output.
- [ ] DP-00B: migrate reciprocal/stacking proof-base ownership without changing scientific
      oracles; keep the dead `integration` dispatcher absent.
- [ ] Checkpoint 0: approve the tolerance artifact/hash and run the six existing proof commands.
- [ ] DP-00C: install the sole immutable 5-degree fixture authority and migrate the old script's
      permanent caller before that script is eligible for deletion.
- [ ] DP-00D: demonstrate executable outward-rounded/interval enclosures for topology, fused
      tangent, ordered strength, exit/attenuation, status boundaries, and latent indicators.
- [ ] Stop before production code if the certification method/dependency is unresolved or cannot
      contract below v2; never relabel refinement differences as certified bounds.

## Frozen Bi2Se3 proof fixture

- [ ] Freeze source origin `(0,-0.020,0) m`, direction `(0,1,0)`, transverse axes
      `((1,0,0),(0,0,1))`, zero spreads, wavelength `1.540592925 A`, source weight one, seed 1729,
      and both `UNITY_APPROXIMATION` polarization provenance fields.
- [ ] Derive the single incident row through source/intersection/entrance refraction and verify
      `IncidentStateBatch.k_film_phase_sample_Ainv =
      (0,4.062900581047559,-0.3545543022596421)` within v2; never inject it by hand.
- [ ] Freeze +5 degree LAB-x sample rotation, unbounded support with absent dimensions, 500 A film,
      population one, the VESTA CIF/SHA, XrayDB/material values/provenance, and canonical detector.
- [ ] Freeze the pure wrapped Gaussian as FWHM 2 degrees, sigma
      `0.014823461823991656 rad` (`0x1.e5bc35dfafaccp-7`), Lorentzian probability/width both zero,
      and no atom.
- [ ] Build one explicit `RodCatalog` with inclusive `[-5,5]^2`, h outer/k inner order, 121 total,
      120 non-specular, integer-table SHA256
      `8f3ee59954e719159164d93bc52d3828d14317021644eaf3e1c51bdac2ec0143`
      for literal LF bytes, and canonical `rod_catalog_revision`
      `70c4b94a7bfac0749f35052bd0335f25db43698431922846cfd97c7b11e4d60d`.
- [ ] Prove detector pose/extent cannot discover, complete, prune, or reorder rods.
- [ ] Freeze detector shape `(3000,3000)`, 1e-4 m pitches, reference `(1453.12,1596.422)` in
      `(column,row)`, declared transform/translation, exact pixel boxes, and 512x1024 sphere raster.

## Point physics and internal coating

- [ ] DP-01: add folded `p_alpha(alpha)=w_signed(alpha)+w_signed(-alpha)` with respect to
      `d alpha`; preserve the existing signed-density trace as a separate quantity and add no
      `sin(alpha)`.
- [ ] Consume exactly `ordered_event_result(...).scattering_strength_A2` with
      `ordered/raw_unit_cell/UNIT_CELL` metadata; apply no second `r_e^2`.
- [ ] Classify 120 rod identities and all 240 ordered `(rod_id,root_label)` slots with one Ewald
      coarea factor; prove roots, residuals, normalization, beta periodicity, and universal exact
      `(h,k)=(0,0)` specular-intensity exclusion.
- [ ] Mutate omitted/doubled folding, coarea, and electron-to-area conversion at their owner.
- [ ] DP-01A: add finite nonnegative ordered-strength enclosures over closed `L` intervals and
      compare them with analytic extrema/dense direct enumeration.
- [ ] DP-02A: certify every periodic `D>0/D=0/D<0` component and tangent chart; sampled coverage is
      never an absence proof.
- [ ] Evaluate/bound the fused tangent continuation
      `2*|ki|/abs(det d(D,eta)/d(alpha,beta))` through `s=0`; never form `0*inf`.
- [ ] Stop a vanishing gradient, unresolved chart denominator, or unresolved topology with a typed
      failure; prove disconnected islands and tangent endpoint mutations.
- [ ] DP-02B: integrate the complete nonnegative latent integrand using factor enclosures owned by
      mosaic/reciprocal/strength modules; stream canonical tiles and retain no Cartesian product.
- [ ] DP-02C: integrate sphere-bin indicators in latent coordinates. Whole-inside cells contribute
      full latent mass, disjoint cells zero, crossings split, and unresolved full mass bounds the
      global-L1 error as `U_true,C + M_nom,C`. Use `2*U_C` only after proving positive quadrature
      and `M_nom,C<=U_C`; sum resolved scalar errors and unresolved allocation errors once per cell.
- [ ] Use mapped polygons only for candidate-bin topology; never apportion physical mass by mapped
      area or multiply a sphere/plane determinant.
- [ ] Prove poles, seam/handedness, exact edges, all preimages, varying-map-determinant mutation,
      rod/root/family/incident/mosaic-kind/phase/parent/strength ledgers, moments, bound coverage,
      refinement, and detector invariance. Every axis reduces the same contribution once.
- [ ] Checkpoint A: focused tests/proof and the internal coating scientific summary are accepted.

## Sole top exit and detector pushforward

- [ ] DP-03: introduce only a private/unexported deterministic internal-wave seam while retaining
      the old event seam only for the sampled runtime until cutover; migrate geometry proof/tests
      off `ScatteringEventBatch`/`transport_scattering_events`, and prove
      `CONTRACT_API_VERSION` and public exports are unchanged.
- [ ] Freeze the sole exit as top plane `z_sample=0`, film `z<=0`, ambient `z>0`, normal `+z`.
      Certified-positive mass is eligible; certified-negative is pre-optics `BACKWARD`; exact
      analytic zero is `PARALLEL`; every other zero-containing enclosure splits/refines or remains
      unresolved. A numerical tolerance never creates a physical dead band.
- [ ] Send only eligible mass through the shared refraction branch/top attenuation. Never use an
      implicit bottom surface or same-top-surface attenuation for inward-going waves.
- [ ] Prove tangential conservation, external dispersion/statuses, one SAMPLE-to-LAB transform,
      actual ray origin, rigid covariance, and detector round trips.
- [ ] DP-03A: add shared-equation exit-amplitude/attenuation enclosures; split every
      top-exit/critical/evanescent boundary, use the manuscript uniform-depth average with
      `|t_in*t_out|^2` once, and give rejected mass no optical weight.
- [ ] DP-04A: certify/split top-exit and every detector `ValidityCode` boundary; unresolved full
      mass remains bounded and `NUMERIC_FAILURE` never balances a ledger.
- [ ] Keep detector point/enclosure equations in `geometry/detector.py`: denominator, ray time,
      continuous `(column,row)`, support, and statuses. Render owns only latent-indicator reduction.
- [ ] DP-04B: certify folds, curved mapped boundaries, self-intersections, and all preimages in
      detector-native `(column,row)` coordinates; polygon area owns no mass.
- [ ] DP-04C: evaluate `integral f_postopt(z)*indicator[Phi(z) in pixel] dz` with exact half-open
      boxes; unresolved cells supply `U_true,C + M_nom,C` to global L1. Use `2*U_C` only with the
      proved positive-quadrature premise, and sum each resolved/unresolved cell exactly once.
- [ ] Prove row/column/corner ties, non-square pixels, independent tiny latent oracle, strongly
      varying map determinant, exhaustive pre-/post-optics ledgers, no solid-angle/Jacobian factor,
      bound coverage, supplemental convergence, and centroid stability.

## End-to-end path, versions, and seventh proof

- [ ] DP-05: compose the temporary deterministic path beside the old runtime; consume the explicit
      incident batch and catalog, never resample a source or derive rods from the detector.
- [ ] Move accepted orchestration to the sole pipeline owner and delete non-oracle temporary
      harnesses. Mark non-fixture public domains `NOT_YET_MIGRATED`, not deprecated.
- [ ] Add a Matplotlib-free numeric-only CLI that creates no file and separates canonical
      `scientific_summary` from volatile `operational_summary`.
- [ ] Run numeric-only twice: arrays and scientific summary are bitwise/canonically equal; wall
      time, RSS, process/platform, and cache metadata are excluded from equality and type checked.
- [ ] Meet the full fixture `<=15 min`/`<=1.5 GiB` gate and classify every first divergence from
      Monte Carlo/bilinear output using independent oracles, not final images.
- [ ] DP-05V1: activate the reserved trace version once, update `docs/TRACE_SCHEMA.md`, and migrate
      core/geometry/reciprocal producers to the shared trace constant and explicit v2 loader.
- [ ] DP-05V2: migrate ordered/stacking/reference producers and their tests; retain immutable
      reference-pack v4 as historical and reject stale/mixed traces.
- [ ] DP-05V3: migrate the final test/validation caller, remove only the ambiguous no-argument
      tolerance-loader alias, and retain named v1 historical access plus explicit current v2.
- [ ] DP-05P: register real `deterministic-pushforward` proof only after production exists; compare
      with a compact analytic/direct-enumeration latent-to-pixel oracle and prove all three ledgers.
- [ ] Run all seven commands: core, geometry-optics, mosaic-ewald, ordered-reflectivity, references,
      stacking-transition, and deterministic-pushforward.

## Peak-mosaic representative and cache

- [ ] DP-06I: for every nonempty closed
      `(incident_state_id,phase_geometry_id,family_id,branch)` support, emit one internal metadata-
      only row with no assigned/deposited mass field whose representative maximizes only the
      complete folded mosaic density.
- [ ] Enumerate/bound interior, seam, and tangent-boundary candidates; use stable log density and
      the declared dimension-aware density/tie order, never adaptive discovery order.
- [ ] Preserve every rod/orientation in physical mass, ignore off-peak `L`, and retain exact
      `(L_peak,m,branch)` for hexagonal display. General-family parity later uses `m=None`.
- [ ] Keep `intensity_status` distinct from root/exit/detector status so
      `SPECULAR_ROOT_COALESCENCE` and `SPECULAR_INTENSITY_EXCLUDED` serialize together without
      overloading either field.
- [ ] For exact `(h,k)=(0,0)` in every lattice family, remove the direct `u=0` root before branch-0
      closure; evaluate no physical mass and mark `SPECULAR_INTENSITY_EXCLUDED`, including a
      limiting signed-`a=0` `SPECULAR_ROOT_COALESCENCE` winner. This metadata-only exception uses no
      physical tangent chart. Hexagonal `m=0` is display provenance only.
- [ ] Add an explicit immutable in/out `PushforwardCache` with canonical SHA256 internal key, one
      complete internal batch, full replacement on miss, and no hidden/global mutation.
- [ ] Put `cache: PushforwardCache | None` and returned `next_cache: PushforwardCache` in the formal
      interface. Keep reciprocal `family_id` crystal/family-only; add a validated canonical
      `phase_geometry_id` to phase inputs, tag rows/IDs, order, and internal keys. Parent/population/
      provider identities remain physical ledger provenance and do not move a shared-geometry tag.
- [ ] DP-06P: project tags through the exact exit/frame/detector seams, retain invalid rows with
      `None` fields, and add the canonical projection key/one-batch replacement.
- [ ] Prove internal miss replaces both batches, projection-only miss reuses internal rows, partial
      or stale hits fail, storage is two tag batches, and cache status cannot change arrays/IDs/order.

## Current-domain parity and planned dovetail support before deletion

- [ ] Checkpoint B0: persist in `docs/VALIDATION.md` every exact post-BKI public domain and planned
      dovetail seam, classified as current parity or first implementation, with owner, task, proof,
      and disposition; an unlisted current domain blocks cutover.
- [ ] DP-06C: preserve multi-row source/incident/wavelength identities, weights, tags, ledgers, and
      canonical incoherent reduction without pipeline resampling. Build the complete incident table
      once and serially; private rows retain `parent_row_index`, scatter through it, never sort by
      state ID, hash a public slice, or regenerate source rows, and pass every parent revision/proof
      field unchanged. Preserve both
      `finite_rectangle.v1` footprint/status behavior, `unbounded_plane.v1` absent dimensions, and
      the current single-phase positive population scalar (including non-unit values) exactly once.
- [ ] Entrance-invalid or zero-source-weight rows emit no coating/tag. Keep rejected positive source
      probability in a separate dimensionless incident-status ledger, never in area-valued mass.
- [ ] DP-06D: integrate the complete smooth Gaussian/Lorentzian mixture and maximize its complete
      folded density; remove the temporary Lorentzian rejection.
- [ ] DP-06E1: integrate `P_atom*delta_0(d alpha)*d beta/(2*pi)` as a one-dimensional periodic beta
      pushforward. Never collapse beta or double count the continuous endpoint.
- [ ] Use ordinary coarea for atom `D>0`; at transverse beta `D=0`, use only
      `2*|ki|/abs(dD/d beta)`. A derivative not certified nonzero is a typed unresolved/degenerate
      stop; the two-dimensional tangent chart is forbidden for the atom.
- [ ] DP-06E2: compose atom mass once through arrays/ledgers and implement supported-beta tag/cache
      semantics. Dimension one outranks dimension two; compare density only within a dimension,
      choose the smallest supported beta, and emit no zero-probability/unsupported row.
- [ ] DP-06F: add the first deterministic general-cell family path. Retain exact `family_id`, every
      rod, and `rod_catalog_revision`; hexagonal rows have integer `m`, general rows `m=None`, and
      nonhexagonal `(0,0)` gets the universal specular exclusion.
- [ ] DP-06G: add the first point-and-enclosure ordered/stacking strength-provider seam with
      once-only area conversion and explicit normalization metadata.
- [ ] DP-06H: add the first phase/parent intensity composition with population ownership once,
      physical parent/provider ledgers, geometric tags, same-family/different-geometry identity
      tests, and no duplicated pushforward runtime; this adds multiphase/multiparent composition,
      not the already-current single-phase population scalar.
- [ ] Checkpoint B: all current-parity and required dovetail rows, full/focused tests, seven proofs,
      certified bounds, canonical summaries, runtime, and memory pass; obtain human review but do
      not cut over until docs pass.
- [ ] DP-06U: factor the unchanged sampled implementation into private
      `_simulate_sampled_comparison`, keep the old public signature/behavior as an exact delegate,
      and migrate the old image script to the private comparator before API activation.
- [ ] DP-06V: after Checkpoint B and DP-06U, promote the complete final phase/cache/result/tag/status contract,
      advance `CONTRACT_API_VERSION` exactly once, change public `simulate_ordered` to its final
      deterministic signature, remove the obsolete event-transport geometry exports, and update
      `docs/CONTRACTS.md` in the same exact five-file task.
- [ ] Prove no earlier private task changed the public version/exports and no later task changes the
      final public schema without a separately approved version increment.
- [ ] DP-06V2: migrate the seventh proof and numeric-only CLI to public `simulate_ordered`, prove
      exact public/private-core equivalence, then forbid direct private-core callers outside the
      pipeline owner. Run P7 and numeric-only twice before documentation/cutover gates.

## Synchronize every live instruction before cutover

- [ ] DP-06A: update architecture/contracts/dovetail/result/trace authorities for latent indicator
      measures, versioned APIs, ledgers, certified bounds, and explicit cache.
- [ ] DP-06B: update scope/physics/validation/error-injection/decisions for the folded measure,
      ordered-strength ownership, fused tangent, sole top exit, and scientific correction.
- [ ] DP-06J: update coating strategy, parallel/mosaic-fitting plans, runbook, and this checklist.
- [ ] DP-06K: update prompts/task README/BKI routing; confirm DP-00R's deterministic-only successor
      and completed BKI-16/BKI-17 disposition. Retain BKI stubs if any obligation or Markdown link
      remains.
- [ ] DP-06L: rewrite T03/T07 as indexed routing/history stubs; do not delete numbered task files.
      Update bootstrap, index, and parallel review in the same exact five-file task.
- [ ] DP-06M: update root README/worktree launch/performance/fitting roadmap for the seven-proof,
      cache, resource, and command contracts.
- [ ] Checkpoint D: full tests, seven proofs, docs/index/import/format checks, accepted-domain
      inventory, and scans over root Markdown/docs/tasks/prompts/YAML/src/tests/scripts are green.
- [ ] If a post-BKI live file remains, amend this plan with another <=5-file task; do not repair it
      after cutover or opportunistically exceed atomicity.

## Delete the old runtime only after Checkpoint D

- [ ] DP-07A: public `simulate_ordered` is already deterministic after DP-06V; delete the private
      sampled comparator, selector/intersection/old image script, and dynamic
      `_detector_complete_rods`/`_symmetric_hk_bounds`; remove DP-06U's temporary comparison
      assertions from `tests/test_integration.py` and consume only the explicit catalog.
- [ ] DP-07B: migrate unique reciprocal proof invariants, then delete `reciprocal/events.py`.
- [ ] DP-08: delete `ScatteringEventBatch` and the now-private event transport in the exact
      four-file task; DP-06V already removed its package exports, and no compatibility facade may
      remain. Preserve `IncidentSampleBatch`/`IncidentStateBatch` and their complete source-to-`ki`
      proof/provenance fields.
- [ ] DP-09: delete fixed orientation construction, bilinear deposition, and obsolete implementation-
      detail tests while retaining unique continuous/root/pushforward invariants.
- [ ] After every deletion, run full tests, all seven proofs, imports, formatting, and symbol scans;
      every intermediate tree must remain green.

## Two required external images

- [ ] DP-10: lock Matplotlib in optional `image` only; numeric-only still creates no file and core/
      dev/permanent tests never import Matplotlib. Extend the already public-API numeric CLI rather
      than changing its scientific call path.
- [ ] Require a resolved plain external writable parent with no reparse-point ancestor and a
      collision-safe target leaf that does not exist; create it once, refuse overwrite/rerun, and
      leave exactly the two declared PNG filenames and no others.
- [ ] Render the sphere PNG from exact 512x1024 bins on an exact decoded 2400x1200 canvas with
      <=0.1 display-pixel chord error, flat bin values, exact seam/poles, no smoothing/resampling,
      explicit zero, and one absolute positive log scale.
- [ ] Render detector data from the exact native `(3000,3000)` `[row,column]` array with
      `origin="upper"`, nearest/no smoothing, absolute units, reference point, and valid tag overlays.
      The decorated detector PNG canvas is explicitly not the scientific array-size contract.
- [ ] Match numeric array/scientific-summary hashes to the prior numeric-only run; type/range-check
      volatile fields; verify titles/provenance/branch-0 exclusion; use neither image as proof.
- [ ] Checkpoint C: fresh leaf contains exactly two PNGs, sphere is exactly 2400x1200, detector
      source array is exactly 3000x3000, repository has no output, and all seven proofs pass.

## Retire residual plans and scan

- [ ] DP-13: delete only the unnumbered superseded coating plan after all unique content is
      owner-reconciled/ported and its resolvable Markdown inbound-link scan is empty; code-span
      historical/deletion/mutation path mentions are not links. Retain indexed T03/T07 stubs.
- [ ] DP-13A: delete BKI plan/checklist only if completed BKI-16/17 obligations are closed/migrated
      and there are zero resolvable Markdown inbound links; otherwise retain DP-06K stubs as a no-op.
- [ ] DP-13B: scan production/tests/scripts for zero removed symbols and scan root/docs/tasks/
      prompts/YAML with only non-imperative historical/deletion/mutation explanations allowlisted.
- [ ] Include sampled-event contracts/transport, selectors, dynamic rod discovery, orientation
      batches, deposition, commands, imports, and successor edges; import every public initializer.

## Final gate

- [ ] Use one explicitly resolved, pre-existing external scratch parent; reject the repository,
      descendants, filesystem roots, and every reparse-point path/ancestor. Use a unique create-new
      probe and create one collision-safe task-owned scratch leaf. Apply the same parent/type/probe
      checks to images, quote both pytest paths, and set `TEMP`, `TMP`, uv, pycache, pytest, Ruff,
      and Matplotlib locations beneath the task-owned scratch leaf.
- [ ] DP-15: compile, lint, format-check, run compact full tests, all seven proofs, docs/index/import
      scans, numeric-only twice, exact image command once, benchmark, peak memory, and clean-tree gate.
- [ ] Use the specification's exact per-task/checkpoint command matrix. Checkpoint 0/A invoke only
      then-existing artifacts; B/D run seven proofs and numeric-only twice; C inspects the DP-10
      image leaf; only DP-00R and final DP-15 run seed verification after their manifest refresh.
- [ ] Detect every folded-measure/area/coarea/fused-tangent/topology/exit/frame/indicator/catalog/
      cache/version/current-domain/dovetail mutation at its owning stage.
- [ ] Meet scalar/global-L1 certified bounds, supplemental convergence/centroid limits, runtime,
      memory, and zero accepted unresolved/numeric-failure mass. Images/Monte Carlo are non-oracles.
- [ ] Audit retained tests; remove temporary comparisons, grids, harnesses, diagnostics, profiles,
      generated files, dead code, stale prompts, and unused dependencies.
- [ ] Regenerate `FILE_MANIFEST.json` with the exact `MANIFEST` command—sorted tracked paths,
      manifest itself excluded, working-tree byte hashes/sizes—only after final validation/file-set
      changes; review its diff and then pass `verify_seed.py`.
- [ ] Run working/cached/base-range diff checks and require warning-free status; squash to one
      coherent commit and provide the complete scientific handoff plus both external PNGs.
