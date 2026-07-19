# Specification and atomic plan: deterministic Ewald-coating pushforward

Status: **CORRECTED PLAN — re-baselined after the completed BKI merge against the current branch
and working tree. Human re-review is required, and implementation remains blocked until the
post-merge reconciliation/base gates below are satisfied. No implementation has started.**

This is the authoritative replacement proposal for the Monte Carlo coating and detector-integration
framework. It supersedes the production direction in:

- `tasks/continuous_ewald_coating_replacement_plan.md`;
- the selection/deposition portions of `tasks/07_integration.md`;
- Sections 4–7 of `docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`; and
- the sampled-event portions of `tasks/parallel_simulation_geometry_fitting_plan.md`.

The named coating/parallel documents are tracked legacy planning artifacts. The implementation
workbranch retires or reconciles all four superseded documents in explicit tasks below. The
companion execution checklist is `tasks/deterministic_ewald_pushforward_todo.md`.

This correction was revalidated on 2026-07-18 against `main` merge
`3af2f4f61d3bc73d8d54891eb867a964240f46cd`, which incorporates the completed BKI implementation
commit `d5eed2524a636c7c190b8f9be300d1d73728884e`. The checked-in BKI plan and checklist mark
BKI-00--BKI-17 complete, so the earlier requirement to import this correction before BKI-15 is
historically impossible and is not retained. This planning commit is a post-merge addendum.

The completed BKI files still route from BKI-15 to parallel Task 1.1 and continuous-coating
validation. Before a deterministic implementation branch is created, DP-00R must reconcile that
stale successor text to the sole overlapping sequence “shared beam-to-`ki` Checkpoint K, then the
deterministic plan,” record the exact BKI
merge/implementation/corrected-planning hashes, refresh the manifest for its exact file set, and
rerun the affected BKI-15 documentation, proof, and seed-integrity commands. Documentation/seed
must pass; any existing reciprocal/stacking proof-base-only failure is recorded exactly and remains
owned by DP-00B rather than being misreported as a new scientific regression. It must preserve BKI
completion evidence as history rather than pretending the reconciliation occurred before BKI-15.
BKI-16 and BKI-17 are already landed; their remaining obligations, if any, must be recorded rather
than projected as future prerequisites.

The historical BKI-04A and BKI-14 changes exceeded the repository's <=5-file task rule. They cannot
be made compliant after merge by splitting a completed commit without rewriting history. DP-00R
must record an owner-approved audit disposition for both deviations and independently rerun the
full affected validation; neither an exception nor a successful rerun may describe those historical
tasks as <=5-file compliant. Every deterministic task below remains <=5 files.

At revalidation, local `main` was four commits ahead of `origin/main` (`063b034`), so DP-00 must
record an owner-approved local-versus-remote authority/synchronization disposition rather than
assuming either ref is canonical. Main also showed owner-controlled untracked `examples.zip` and
permission-denied paths `pytest-of-Kenpo/`, `tmp0p_i8jt4/`, and `tmpjc1vjoa6/`. This planning commit
must not stage, overwrite, delete, or imply disposition of them.

At final plan-freeze revalidation, active sibling worktree
`../SLATE-rMC-beam-ki-findings-plan` was on `codex/beam-ki-findings-plan` at planning-only commit
`0953cab668b489fd4c4d5a1d4de070fcab165295`, a descendant of current `main` not yet incorporated
there. Its parent `af456b5988076c400eea972fb4f2ac2d647891dc` changes `FILE_MANIFEST.json`,
`tasks/plan.md`, and `tasks/todo.md`; `0953cab` changes this deterministic plan/checklist and
`tasks/parallel_simulation_geometry_fitting_plan.md`. The worktree then had additional uncommitted
changes to `tasks/09_fit_foundation.md`, `tasks/10_instrument_calibration.md`,
`tasks/11_sample_geometry_fit.md`, and the parallel plan. Those paths directly overlap DP-00R or
later instruction synchronization. DP-00R cannot start until the owner lands, transfers by exact
patch hash, or explicitly supersedes both commits and the remaining dirty changes; it must
reconcile rather than overwrite them and must record the approved rebase/merge order. A separate clean detached worktree
`C:/Users/Kenpo/.codex/worktrees/791a/SLATE-rMC` at
`55dc336ec0d65d22c49cf7ee9bb96d2105405a82` also requires an owner disposition in the active-
worktree audit, but must not be pruned merely to manufacture cleanliness. DP-00 requires a warning-
free status in which every untracked path is visible before recording the implementation base.

The `af456b5..0953cab` planning sequence defines a future shared beam-to-`ki` Checkpoint K; it does not implement or accept
that checkpoint. The deterministic workbranch must start only after the owner-reconciled shared
contract branch lands and Checkpoint K passes on approved `main`. The accepted boundary constructs
and hashes one complete `IncidentSampleBatch`, builds its complete `IncidentStateBatch` once and
serially, and then tiles downstream work. Private rows retain explicit `parent_row_index`; reassembly
scatters through that index, while `incident_state_id` remains verified identity payload and is
never a sorting key. Workers never regenerate source rows or construct/hash public batch slices.
The one-row nominal source retains empirical `source_weight=1`; only tag metadata has no assigned
or deposited event mass. DP-00 records the accepted Checkpoint K commit/API/revision hashes and
revalidates every frozen fixture field before branch creation.

Any existing `codex/continuous-ewald-coating-validation` ref remains
owner-controlled provenance unless its unique work is explicitly ported or archived; it is never
reused, resumed, or silently deleted. At revalidation it pointed to unique unmerged commit
`8681b01aa476a96ad6e53529c3ff0a249a6720f5`; DP-00 must verify that ref or record an owner-approved
successor disposition rather than assuming it disappeared.

## Problem statement

How might we replace random mosaic-event selection and point deposition with one deterministic,
mass-conserving pushforward from the accepted Bragg-rod/mosaic measure to the actual detector,
while preserving the validated source, refraction, structure-factor, coordinate, and detector
physics and making the result inspectable at every stage?

## Recommended direction

Integrate the accepted continuous mosaic coordinates directly. For every explicit rod and Ewald
root, adaptively partition the latent `(alpha, beta)` domain, evaluate the exact root, ordered
strength, exit transport, and detector map, and conservatively accumulate each cell into:

1. equal-solid-angle bins on the **internal** Ewald sphere for the requested coating diagnostic;
2. exact detector-native pixel boxes after exit refraction and the sample-to-lab transform, using a
   tolerance-bounded conservative cell approximation; and
3. a separate metadata-only cache containing exactly one peak-mosaic
   `(L_peak,m,intersection_branch_id)` representative for each nonempty supported family/root
   closure, with its exit and detector status and its detector position when valid; tag rows have
   no assigned/deposited event-mass field and cannot enter a physical ledger.

There is no Monte Carlo selection, no sampled outgoing-event mass, no intermediate `kf` plane, no
point/bilinear deposition, and no retained incident-by-rod-by-orientation Cartesian product. The
necessary Ewald coarea factor is applied once. A sphere-to-plane determinant is not multiplied into
the raw image because forward cell integration already performs the change of variables.

## Objectives

- Produce a deterministic detector-native `angstrom^2/pixel` image for one centered incident ray.
- Produce a deterministic image of the unsampled intensity coating on the internal Ewald sphere.
- Derive the incident state through the existing source/intersection/entrance-refraction boundary.
- Exit every internal `kf` through the sample surface with the accepted refractive branch and then
  transform it once into the lab frame before detector projection.
- Preserve every physical `(h,k)` rod and Ewald-root identity until incoherent mass reduction.
- Cache exactly one internal peak-mosaic representative for each nonempty supported exact
  phase-geometry/family and Ewald intersection root, ignoring the many off-peak `L` values
  contributed by the mosaic distribution; expose exact `m` only for hexagonal display and keep the
  row even when its exit or detector projection is invalid.
- Replace the old runtime atomically and delete each superseded module, API, test, script, and live
  instruction after its replacement passes its gate.
- Leave one small production path, one compact permanent proof suite, and no compatibility switch.
- Generate exactly two external PNG deliverables from one frozen Bi2Se3 case at handoff.
- Preserve every accepted post-BKI production domain before deleting the sampled runtime: incident
  batches and wavelength rows, finite/unbounded sample support, continuous mixtures, and the
  zero-width probability atom, including the current single-phase population weight. Also
  implement the already declared general-family,
  ordered/stacking, and phase/parent dovetail seams before cutover without mislabelling them as
  current-runtime parity.

## Non-goals

- No exact `(h,k)=(0,0)` coating intensity in any lattice family until a physical beamstop, support
  gap, or finite-resolution model is declared. The hexagonal display calls this `m=0`; branch-0
  geometry may be cached as metadata only, with no assigned/deposited mass field.
- The DP-00C proof fixture and the DP-01--DP-06 narrow milestone use one incident row, one
  wavelength, one hexagonal phase, a positive-width pure Gaussian, no atom, and ordered strength.
  Those are fixture restrictions, not final sole-runtime exclusions. DP-00 records the exact
  accepted post-BKI domain inventory, and DP-07 deletion is forbidden until every accepted domain
  has deterministic parity or an explicit, separately reviewed deprecation/migration.
- No new fitting, caking, angle-space remapping, PSF,
  background, efficiency, exposure, gain, or absolute photon-count model.
- No bottom-interface transmission in this first optical model. The only exit surface is
  `z_sample=0`, film occupies `z_sample <= 0`, ambient occupies `z_sample > 0`, and the outward
  normal is `+z_sample`. Waves propagating away from that surface are classified as pre-optics
  rejected mass, never transmitted through an implicit second surface.
- No GPU, backend layer, executor framework, plugin, registry, compatibility facade, or permanent
  old/new feature flag.
- No family-cylinder shortcut, coherent sum between rods, floating-`Qr` identity, reflection
  pruning, interpolated structure factors, or fabricated fractional reflections.
- No intermediate plane normal to the mean `kf`, inverse detector rasterization, or pointwise
  detector-density Jacobian in the production path.
- No physical pixel-solid-angle multiplier. Pixel solid angle remains immutable geometry metadata.
- No committed PNG, image snapshot, broad convergence grid, profiling dump, or repository-local
  diagnostic.

## Key assumptions to validate before cutover

- [ ] Adaptive forward cells can meet the frozen sphere/detector error bounds without retaining a
      dense rod-by-orientation-by-pixel product.
- [ ] Analytic or interval branch-and-bound can certify the complete topology of Ewald support,
      sphere/detector folds, and validity regions; adaptive sampling alone is not an absence proof.
- [ ] Maximization on the closure of regular root support, including tangent-boundary semantics and
      a continuous periodic-`beta` tie-break, yields exactly one
      lattice-aware `(L_peak,optional m,family_id,intersection_branch_id)` tag per nonempty
      supported degenerate set.
- [ ] The existing entrance/exit transport and detector equations can be consumed in batches
      without duplicating or weakening their scalar proof path.
- [ ] Tolerance-bounded conservative integration into exact detector-pixel boxes is the intended
      corrected observable, even though it intentionally diverges from the current bilinear
      ensemble mean at that stage.
- [ ] The frozen coating-validation mosaic is the pure 2-degree Gaussian case. The tracked
      1-degree forward-case mosaic and a Gaussian/Lorentzian mixture are not silently substituted.
- [ ] Every accepted post-BKI domain has deterministic parity before cutover, and every listed
      future dovetail seam has its independently proved first implementation; a narrow milestone
      rejection is not evidence that an existing public domain may be removed.
- [ ] Interval/outward-rounded enclosures can certify topology and the complete physical integrand
      before production orchestration is written; inability to build those enclosures is a
      pre-code feasibility stop, not permission to relabel refinement differences as bounds.

DP-01/DP-01A test point physics and factor bounds; DP-02A--DP-04C test numerical feasibility;
DP-06I/DP-06P test tag/cache semantics; DP-06C--DP-06E2 prove current-domain parity and
DP-06F--DP-06H prove the required first dovetail implementations. A failed
dealbreaker stops the workbranch for review rather than weakening certification or restoring Monte
Carlo.

## Technology, style, and boundaries

### Technology

- Python `>=3.12,<3.14`, NumPy float64/complex128, and the existing SciPy/Gemmi/XrayDB stack.
- No new production dependency is planned.
- Matplotlib is locked in a separate `image` dependency group for the two requested PNGs. It is
  never a core/dev dependency and is never imported by `rasim_next` production modules, the frozen
  fixture builder, or permanent tests.

### Code style

- Small pure batched functions, immutable typed results, explicit SAMPLE/LAB frames, explicit
  angstrom/metre/pixel units, canonical row order, and bounded NumPy tiles.
- One implementation of each physical equation. Scalar functions remain independent proof
  authorities; optimized batches call equivalent primitives rather than copying equations.
- Adaptive state and mapped simple polygons are private. Public results expose physical arrays, error
  estimates, and conservation ledgers, not implementation work queues.

### Boundaries

- **Always:** preserve individual rod identities; apply every factor once; propagate explicit
  validity; run the focused gate after each task; keep artifacts external; delete replaced code at
  its named gate.
- **Ask first:** change the frozen fixture, sphere raster, result measure, tolerance, public
  frame/unit contract,
  dependency set, exact `(h,k)=(0,0)` physical support, or workbranch base SHA.
- **Never:** edit immutable `examples/` or `reference/`; import legacy source; add a silent
  approximation/fallback; normalize images to their maximum; keep both runtimes at handoff; or
  weaken a tolerance after observing disagreement.

## Frozen smallest useful scope

“Default Bi2Se3” for this branch means the already proposed 5-degree coating-validation fixture,
combined with the canonical detector geometry. It intentionally does **not** mean the current
12-degree, 20-ray, 40-draw image script.

### Source and incident state

- Mean source origin in LAB: `(0.0, -0.020, 0.0) m`.
- Mean source direction in LAB: `(0.0, 1.0, 0.0)`.
- Source position: the exact center only; spatial spread is absent.
- Divergence: exactly zero.
- Source transverse axes in LAB are exactly
  `((1.0,0.0,0.0),(0.0,0.0,1.0))` in that order.
- Wavelength: exactly `1.540592925 angstrom`; bandwidth is absent.
- Source weight: exactly `1.0`.
- Polarization: `polarization_state_id="UNITY_APPROXIMATION"` on the source row and
  `polarization_policy_id="UNITY_APPROXIMATION"` at the pipeline boundary; the numeric factor is
  exactly one and both provenance fields are retained in the result.
- Construct the row through `sample_gaussian_source_rays` with `sample_count=1`, all spatial and
  divergence sigmas zero, wavelength sigma zero, and the existing source seed `1729`. Its odd
  center row is exactly the requested mean and draws no stochastic coordinate.
- Sample motion: one active `+5 degree` rotation around `+LAB x` at the origin and no other sample,
  goniometer, or crystal-mount rotation.
- The one-row source batch must pass through `build_incident_states`; production must not inject an
  internal `ki` directly.
- The resulting accepted `IncidentStateBatch.k_film_phase_sample_Ainv` row is
  `(0.0, 4.062900581047559, -0.3545543022596421)` within the frozen stage tolerance. The shorter
  `ki_film_sample_Ainv` notation in equations denotes that field; it is not another contract or an
  alternate source of truth.

### Crystal, rods, and mosaic

- Structure: `examples/bi2se3/structures/Bi2Se3_vesta.cif`.
- Structure SHA256:
  `2a02dfceb6b302810229076479c54515382b3adcd558e3d9f874192dcd6b539f`.
- Phase identifier: `bi2se3`; parsed space group: `R -3 m:H`. The parsed lattice and occupied sites
  are authoritative.
- At the frozen wavelength, the locked XrayDB `4.5.8` database revision `9.2` and the tracked CIF
  must reproduce `n_film = 0.9999807177019893 + 1.5800513246371106e-6j`,
  `delta = 1.9282298010759586e-5`, `beta = 1.5800513246371106e-6`, and
  `mu = 1.2888226482734933e-5 angstrom^-1`. The full returned material provenance is hashed.
- Rod window for this proof: every `(h,k)` in inclusive `[-5,5] x [-5,5]`, built once through
  `build_rod_catalog(crystal, h_bounds=(-5,5), k_bounds=(-5,5))` and passed explicitly to the
  pipeline. Canonical order is h-major, then k-major within h (h outer/k inner).
- Catalog count: `121`; active non-specular count: `120` after excluding `(0,0)` intensity.
- The ASCII audit serialization `"{h},{k}\n"` using a literal LF byte (`0x0a`, never a
  platform newline) in canonical order has SHA256
  `8f3ee59954e719159164d93bc52d3828d14317021644eaf3e1c51bdac2ec0143`. The fixture also records
  `rod_catalog_revision=70c4b94a7bfac0749f35052bd0335f25db43698431922846cfd97c7b11e4d60d`,
  computed by `reciprocal.rods.rod_catalog_revision` with the accepted canonical typed SHA256 v1
  over `rod_id`, `phase_id`, `h`, `k`, `family_id`, `family_key`, `qr_Ainv`,
  `reciprocal_basis_Ainv`, and `symmetry_metadata`. Detector pose/extent never discovers,
  completes, prunes, or reorders the catalog.
- Every rod remains explicit even when several rods share the same integer `m`.
- Mosaic: pure wrapped Gaussian with FWHM `2 degree`, sigma exactly computed as
  `radians(2)/(2*sqrt(2*log(2))) = 0.014823461823991656 rad`
  (`0x1.e5bc35dfafaccp-7`), `lorentzian_probability=0.0`,
  `lorentzian_half_width_rad=0.0`, and no zero-width atom.
- Mosaic measure: folded `p_alpha(alpha) d alpha` and `d beta/(2*pi)`; no extra `sin(alpha)`.
- Nonzero-`m` roots are labelled `1` and `2` by increasing `L`.
- The non-direct exact `(h,k)=(0,0)` geometry uses label `0`; in this hexagonal fixture it displays
  `m=0`, but its coating mass is excluded with `SPECULAR_INTENSITY_EXCLUDED`. No epsilon or fallback
  is permitted.

### Detector and sample

- Detector-native shape: `(3000 rows, 3000 columns)`.
- Pixel pitch: `1.0e-4 m` in both row and column.
- Detector reference coordinate: `(column,row) = (1453.12, 1596.422) px`.
- Detector-to-LAB active rotation:

  ```text
  [[1,  0, 0],
   [0,  0, 1],
   [0, -1, 0]]
  ```

  and translation `(0.0, 0.075, 0.0) m`. Detector local column is `+LAB x`, local row is
  `-LAB z`, and the detector normal is `+LAB y`; there is no additional detector-face tilt.
- Sample support is explicit `unbounded_plane.v1` with absent width and length, matching the
  accepted BKI correction of the legacy zero-disabled footprint. The centered ray must have
  footprint weight one; no replacement finite dimensions are permitted.
- Film thickness: `500 angstrom`.
- Population weight: `1.0`.

The exact source, instrument, crystal/material, mosaic, and rod-window literals have one authority:
`src/rasim_next/proof/bi2se3_pushforward_fixture.py`. Proofs, permanent tests, and the optional
image script consume that pure builder; none copies its literal block. The superseded 12-degree
script remains temporary legacy comparison evidence only until DP-07A deletes it.

### Exact detector-pixel ownership

Continuous detector coordinates remain `(column_px,row_px)`. Pixel `(row=r,column=c)` owns
`[c-0.5,c+0.5) x [r-0.5,r+0.5)`. The final column closes its outer `c+0.5` edge and the final row
closes its outer `r+0.5` edge, so the full detector support is exactly
`[-0.5,n_columns-0.5] x [-0.5,n_rows-0.5]` without gaps or double ownership. Exact internal shared
edges go to the pixel on their left-closed side.

The detector plane is normal to the mean LAB beam in this fixture and is tilted relative to the
sample solely because the sample is at 5 degrees. Adding an independent detector tilt is outside
this branch.

### Frozen Ewald-sphere raster

- Bin the unit internal outgoing direction
  `s_hat = kf_film_phase_sample_Ainv/|k_film_phase_sample_Ainv|` in
  `mu = s_hat_z` and `phi = atan2(s_hat_y,s_hat_x)`.
- Use `512` uniform bins in `mu` over `[-1,1]`: bins `0..510` are left-closed/right-open and bin
  `511` is closed at `mu=+1`. Use `1024` uniform left-closed/right-open bins in `phi` over
  `[-pi,pi)`. Clamp only a roundoff excursion within the frozen unit-vector tolerance; assign exact
  north and south poles to `phi=0`, and wrap the `-pi/+pi` seam once.
- Every bin has declared full solid angle `DeltaOmega = DeltaMu*DeltaPhi`. Overlapping rods, roots,
  and latent preimages add mass in the same bin.
- Report `rho_B = mu_coat(B)/DeltaOmega` in `angstrom^2/sr`. The raster sum uses the full declared
  bin area, never a preimage-covered fraction.
- Render front (`s_hat_y >= 0`) and back (`s_hat_y < 0`) orthographic panels in SAMPLE axes with
  horizontal coordinate `s_hat_x` increasing left-to-right and vertical coordinate `s_hat_z`
  increasing bottom-to-top. The back panel is coordinate-fixed, not viewer-mirrored: `+SAMPLE x`
  points right and `+SAMPLE z` points up in both panels. Test both poles, the seam, and handedness.
  The orthographic view is only a display of equal-solid-angle data; it is not itself an equal-area
  projection or a physical transform.

## Declared measures and factor ownership

Let `w_signed(theta)` denote the sum of the positive-width wrapped signed-angle components with
their declared mixture weights, with respect to `d theta` on `[-pi,pi)`. When no atom is active,
this is the existing signed-density result. When an atom is active, the equation owner evaluates
the continuous part separately without changing the existing public trace semantics. The
continuous positive-tilt density is a distinct folded quantity:

```text
p_alpha(alpha) = w_signed(alpha) + w_signed(-alpha),  alpha in [0,pi].
```

It satisfies `integral_0^pi p_alpha(alpha) d_alpha = P_continuous`, where
`P_continuous=1-P_atom`; the frozen fixture has `P_atom=0` and integral one. For a symmetric
component this is `2*w_signed(alpha)` almost everywhere. Endpoint values carry no atom and do not
change the measure. The accepted zero-width component is the separate singular measure

```text
P_atom * delta_0(d alpha) * d beta/(2*pi),  beta on S^1.
```

It is a one-dimensional beta pushforward, not a beta-collapsed point: the accepted
`R(0,beta)=R_c*(beta)` rotation still traces the raw family cylinder. It is evaluated by its own
path and never double counts the continuous endpoint. The existing signed-density trace retains
its current meaning; folded density gets its own trace field and is never substituted into that
legacy field.

For incident state `i`, rod `r`, regular Ewald intersection root `b`, and mosaic coordinate
`z=(alpha,beta)`, first define the reciprocal kernel

```text
d_nu_recip(i,r,b,z)
  = p_alpha(alpha)/(2*pi)
    * J_ewald(i,r,b,z)
    * d_alpha*d_beta.
```

`reciprocal/coating.py` owns only the rotation, root/status, `L`, `J_ewald`, and this kernel. The
pipeline composes the physical internal coating measure exactly once:

```text
d_mu_coat(i,r,b,z)
  = source_weight(i)
    * population_weight
    * footprint_weight(i)
    * S_r(L_b(z))
    * d_nu_recip(i,r,b,z).
```

`S_r(L)` is exactly the area-converted
`ordered_event_result(...).scattering_strength_A2` value, including its declared model and
normalization metadata. `ordered/amplitudes.py` owns the electron-to-area conversion and applies
`r_e^2` once. Neither the reciprocal kernel nor the pipeline reapplies that conversion. For the
frozen case the model is `ordered`, the normalization is `raw_unit_cell`, and the area unit is
`UNIT_CELL`. The ordered or stacking strength owner must expose the same narrow point-and-certified-
enclosure seam before cutover. Further,

```text
J_ewald = |ki| / abs(dot(rod_direction_hat, kf_film))
```

is the one Ewald-root coarea factor. The Ewald image bins `d_mu_coat` before exit optics. A fixed
sphere bin `B` displays `mu_coat(B)/DeltaOmega_B` in `angstrom^2/sr`; the conversion must sum back
to the same coating mass and can never be reused as a physical weight.

The raw detector pixel mass is

```text
M[pixel]
  = sum(i,r,b) integral_{z in mosaic support}(
        d_mu_coat(i,r,b,z)
        * W_entrance_exit_and_attenuation_average(z)
        * W_polarization(z)
        * indicator[Phi(i,r,b,z) lies inside pixel]
    )
```

Here `z=(alpha,beta)` remains the mosaic coordinate and the outer integral is with respect to its
declared continuous/atom measure. The separate depth coordinate is `zeta_sample`. The optical
factor contains the accepted manuscript uniform-depth average
`|t_in*t_out|^2*(1/thickness)*integral_{-thickness}^0 decay(z,zeta_sample) d zeta_sample`, not a
fitted or sampled depth distribution. Its owner applies the one incident transmitted channel, one
eligible top-exit transmitted channel, amplitudes, and incident/exit decay exactly once using the
shared complex-normal-wavevector branch selector. Rejected top-exit mass receives no exit amplitude
or depth weight.

`Phi` is the direct composition

```text
internal kf in SAMPLE
  -> classify eligibility for the sole top interface with
       g_exit = dot(kf_film_phase_sample_Ainv,+z_sample)
       certified lower(g_exit) > 0: eligible
       certified upper(g_exit) < 0: pre-optics BACKWARD
       exact analytic g_exit == 0: pre-optics PARALLEL
       otherwise (the enclosure contains zero): split/refine or remain unresolved
  -> exit refraction at sample-surface coordinate z_sample=0 with normal +z_sample
  -> external kf in SAMPLE
  -> lab_from_sample.apply_vector
  -> ray from the actual sample intersection in LAB
  -> tilted detector-plane intersection
  -> detector-native (column_px,row_px)
```

No second Lorentz factor, exit-refraction determinant, plane determinant, or pixel-solid-angle
multiplier appears. A pointwise plane density would need a coordinate Jacobian, but the chosen
forward mass integral does not evaluate that singular density.

Only the eligible positive-outward branch enters exit refraction, top-surface attenuation,
SAMPLE-to-LAB transformation, and detector projection. `BACKWARD` here means propagation away from
the one declared exit interface and is ledger-distinct from a backward detector ray time. A
bottom-interface model would require a second plane, opposite normal, material, ray origin, and
attenuation derivation and is outside this plan.

Near a certified transverse tangent, use the unscaled Ewald radicand `D`, for which every regular
root satisfies `abs(dot(rod_direction_hat,kf_film))=sqrt(D)`. For a local nonsingular chart
`(D,eta)` and the one-sided substitution `D=s^2`, evaluate and bound the fused reciprocal/chart
kernel

```text
J_ewald * abs(det d(alpha,beta)/d(s,eta))
  = (|ki|/abs(s))
    * (2*abs(s)/abs(det d(D,eta)/d(alpha,beta)))
  = 2*|ki|/abs(det d(D,eta)/d(alpha,beta)).
```

The right-hand expression is the continuous one-sided endpoint value at `s=0`. Production and
certified-bound code never form or bound `1/abs(s)` and `abs(s)` separately at or across a tangent.
Away from the endpoint the proof compares the fused value with the separated regular-root factors.
The ledger still attributes one physical Ewald/coarea factor and one numerical chart Jacobian;
fusion is only their stable evaluation. The implementation records chart orientation and refuses a
chart whose denominator cannot be bounded away from zero.

For the one-dimensional zero-tilt atom, `D<0` has no root and `D>0` uses the ordinary Ewald coarea
factor under `d beta/(2*pi)`. At a transverse beta tangent, `D=0` with
`abs(dD/d beta)` certified away from zero, the one-sided substitution `D=s^2` owns the distinct
one-dimensional fused continuation

```text
J_ewald * abs(d beta/ds) = 2*|ki|/abs(dD/d beta).
```

The two-dimensional `(D,eta)` chart is never borrowed for the atom. If `dD/d beta` cannot be
certified nonzero, the atom tangent is `DEGENERATE_ATOM_TANGENCY`/unresolved and the branch stops;
it is neither silently discarded nor assigned finite mass by a heuristic.

### Conservation ledger

The implementation first reports a dimensionless incident/source-probability identity, then three
separate area-measure identities:

```text
P_source
  = sum(source_weight[entrance-valid])
  + sum_by_incident_status(source_weight[entrance-rejected])
  + r_incident_probability

M_coat
  = sphere-bin mass + r_sphere

M_coat
  = exit-supported pre-optics coating mass
  + exit-rejected pre-optics coating mass
  + r_exit_classification

M_postopt
  = detector-status mass[VALID and deposited]
  + detector-status mass[OUTSIDE_SUPPORT]
  + detector-status mass[PARALLEL]
  + detector-status mass[BACKWARD]
  + detector-status mass[NO_SOLUTION]
  + detector-status mass[RESIDUAL_EXCEEDED]
  + detector-status mass[NUMERIC_FAILURE]
  + r_pushforward
```

`P_source` and `r_incident_probability` are dimensionless and never enter an area ledger. A zero-
weight row contributes exact zero; an entrance-invalid positive-weight row remains visible under its
incident status but produces no Ewald support, coating tag, or `M_coat` contribution.
`M_postopt` integrates `d_mu_coat * W_optical * W_polarization` only over exit-supported points;
exit-rejected coating mass is never called optically weighted. These are scalar-field intensity
measures, not a power-flux balance. Every residual is the independently computed algebraic closure
difference and is accompanied by a separately certified quadrature/mapping upper bound; any
heuristic refinement estimate is separately named and is never promoted to a bound. No residual
may be assigned as a balancing plug. Rod, family, root, incident-state, mosaic-measure-kind
(`CONTINUOUS`/`ZERO_TILT_ATOM`), phase, parent/population-group, and strength-model/component
subtotals must each reduce exactly once to their parent total. These axes are simultaneous
reductions of the same contributions, not extra factors. Exit classification and detector
classification each expose an
exhaustive mass mapping keyed by every `ValidityCode`; codes inapplicable at a stage have exact zero mass.
`NON_PROPAGATING` belongs to the pre-optics exit-rejected mapping and never enters `M_postopt`.
Any nonzero `NUMERIC_FAILURE` mass above the frozen absolute floor fails acceptance and is never
hidden in `r_pushforward`. Component-tag rows carry no assigned/deposited physical-mass field and
cannot change any source, coating, optical, or detector ledger.

## Peak-mosaic component-tag cache

The tag pass is deterministic metadata derived from the same equations, not a second forward
model. It records one representative of a family/root coating component, not a Bragg peak and not
an adaptive integration node. In this section, `intersection_branch_id` is only the Ewald-root
label `0`, `1`, or `2`; it is not a fitting or Git branch.

Let `P_continuous=1-P_atom`. For incident state `i`, immutable phase geometry `g`, exact reciprocal
family `f`, and regular-root label `b in {1,2}`, define two discriminated probability-measure
supports:

```text
Z_cont(i,g,f,b) = closure({(CONTINUOUS,r,alpha,beta):
                          incident_valid(i), source_weight(i) > 0,
                          P_continuous > 0,
                          r belongs to family f,
                          p_alpha(alpha) > 0,
                          alpha in [0,pi], beta on S^1,
                          discriminant D(alpha,beta) > 0,
                          Ewald root label is b})

Z_atom(i,g,f,b) = closure({(ZERO_TILT_ATOM,r,beta):
                          incident_valid(i), source_weight(i) > 0,
                          P_atom > 0,
                          r belongs to family f,
                          alpha = 0, beta on S^1,
                          discriminant D(0,beta) > 0,
                          Ewald root label is b})

Z_total(i,g,f,b) = discriminated union(Z_cont,Z_atom).
```

Both supports are empty for an entrance-invalid or zero-source-weight row; such a row emits no
coating mass or tag. Its source probability/status remains in the separate dimensionless incident
ledger and is never balanced into area-valued `M_coat`. `Z_cont` is empty when
`P_continuous=0`; `Z_atom` is empty when `P_atom=0`. Thus geometric Ewald support with zero accepted
product probability never emits a row. Continuous closure uses
`[0,pi] x S^1`; atom closure uses the one-dimensional beta circle because the accepted rotation
`R(0,beta)=R_c*(beta)` still traces the raw family cylinder. A half-open beta interval is only the
canonical serialization and never duplicates the seam. At a transverse `D=0` boundary the common
double root carries `TANGENT_BOUNDARY` and its limiting label; branches 1/2 may share geometry while
remaining distinct. An atom boundary enters this closure only when its beta transversality is
certified; a zero/unresolved `dD/d beta` is a typed branch stop, not a tag candidate.

1. Retain every underlying rod and orientation separately in the physical mass calculation.
2. For each nonempty `Z_cont`, globally maximize exactly the declared mosaic
   probability density, the complete folded `p_alpha(alpha)/(2*pi)`. Ordered strength, coarea, exit optics,
   polarization, detector validity,
   and pixel position do not participate. The first deterministic milestone accepts only the frozen positive-width
   pure Gaussian, so its fixture-only certified comparison is equivalently the global minimum of
   `alpha` on the closed support. Mixture parity later maximizes the complete accepted folded
   mixture rather than retaining this shortcut. Enumerate and bound every interior, seam, and
   tangent-boundary candidate; a
   local optimizer or adaptive-node search is not an authority.
   For nonempty `Z_atom`, the density with respect to its one-dimensional reference measure is the
   constant `P_atom/(2*pi)` over supported beta. Do not compare that `rad^-1` line density
   numerically with the continuous `rad^-2` density. Use the small-ball mode order: measure
   dimension one (atom line) outranks dimension two (continuous surface); compare density only
   within the same dimension. If `Z_atom` is empty, fall back to `Z_cont`; if both are empty, emit
   no row.
3. Resolve candidates by the dimension-aware total order
   `(measure_dimension, -log_density_within_dimension, alpha, abs(wrapped_beta), wrapped_beta,
   rod_id)`, never by adaptive-node/discovery order. A uniform atom does not force `beta=0`; the
   smallest supported beta under the tie order wins. This remains defined at the seam and avoids a
   false far-tail plateau from underflow.
4. Store the winner's exact continuous `L`, exact family identity, optional exact hexagonal `m`,
   and Ewald-root label. The frozen hexagonal display is
   `(L_peak,m,intersection_branch_id)`; a general family has `m=None`. Do not round `L`, search for
   an integer `L`, or tag any off-mode `L`.
5. Emit exactly one internal metadata-only row for every nonempty `Z_total`. The row has no
   assigned/deposited event-mass field. Then pass its
   internal `kf` through the same exit-refraction, SAMPLE-to-LAB, and detector mapping used by the
   mass integral. Retain the row when exit or detector projection fails, with explicit status and
   absent optional external vector/pixel fields; never substitute an off-peak representative.
6. Detector pose and exit validity cannot change which internal representative wins. Only
   detector-valid rows are overlaid on the detector PNG.

Define the universal specular rod by exact `(h,k)=(0,0)`, independently of optional hexagonal `m`.
Remove its direct `u=0` root before forming branch 0 and consider only the retained non-direct root.
For the hexagonal fixture this is the requested `m=0` exclusion; a general-cell row retains
`m=None`. It inherits the same `incident_valid(i)` and positive-source-weight prerequisites. This is
a separate metadata-only closure, not `Z_cont`/`Z_atom` physical regular-root topology. With
unit rod direction `d_hat`, define the signed scalar
`a=dot(ki,d_hat)` and retained root `u_non_direct=-2*a`. Form its measure-supported closure only
after applying

```text
(h,k) == (0,0) and u_non_direct != 0 and the direct u == 0 root is absent.
```

The signed-`a` closure is taken after the direct root has been removed. A limiting point `a=0`,
where the retained root coalesces with the removed direct root, may win and is recorded as
`root_status=SPECULAR_ROOT_COALESCENCE` with `intersection_branch_id=0`. It is not called a regular
`D=0` tangent and does not invoke the continuous or atom tangent chart/transversality rule. Evaluate
no specular strength, coarea mass, optical mass, or detector mass for this set. Its row has no mass-
valued field and carries `intensity_status=SPECULAR_INTENSITY_EXCLUDED`; invalid exit/projection status is
retained rather than deleting the representative.

Each immutable row contains at least:

```text
component_tag_id
incident_state_id
phase_geometry_id
family_id, canonical representative rod_id, h, k
m = h^2 + h*k + k^2 for a hexagonal family; otherwise None
intersection_branch_id in {0,1,2}
L_peak
mosaic_alpha_rad, mosaic_beta_rad
mosaic_measure_kind in {CONTINUOUS, ZERO_TILT_ATOM}
mosaic_measure_dimension in {1,2}
optional atom_probability_mass and atom_line_density_rad_inv
q_internal_sample_Ainv
kf_film_phase_sample_Ainv
optional kf_air_sample_Ainv
optional kf_air_lab_Ainv
optional (column_px,row_px)
root_status, exit_status, detector_status
intensity_status in {INCLUDED, SPECULAR_INTENSITY_EXCLUDED}
```

For general-cell catalogs, exact `family_id` remains the primary identity and `m=None`; no integer
is fabricated. The frozen fixture therefore exposes the requested
`(L_peak,m,intersection_branch_id)` display tuple, while a general row exposes
`(L_peak,family_id,intersection_branch_id)` plus optional `m`. Neither tuple is rod identity. The public
`CoatingComponentTagBatch` wraps immutable typed rows so invalid optional fields are `None`, not
numeric sentinels. It has exactly one row for each nonempty supported
`(incident_state_id,phase_geometry_id,family_id,intersection_branch_id)` closure and records the canonical supporting
rod for provenance. Branch `0` rows are allowed only for exact `(h,k)=(0,0)` with
`SPECULAR_INTENSITY_EXCLUDED`; no tag row has an assigned/deposited event-mass field.

`component_tag_id` and canonical row order derive only from exact
`(incident_state_id,phase_geometry_id,family_id,intersection_branch_id)` identity, never from `L`,
floating density, or discovery order. The internal representative key explicitly contains the
validated `phase_geometry_id`, immutable incident
batch/revision and internal `ki`, `sample_from_crystal.rotation`, reciprocal basis and full rod
catalog revision, internal-medium revision, complete mosaic parameters, support/root solver
revision, tag tie-order revision, and only the topology/root/closure tolerances used to classify
internal membership. A separate projection key contains the actual sample intersection,
`lab_from_sample`, wavelength and material exit-optics revision, the sole-top-interface exit
policy, exit solver and optical tolerances, the `component_tag_id` plus an exact hash of the
internal representative payload including `kf_film_phase_sample_Ainv`, detector transform, shape,
row/column pitches, reference coordinate, support-edge convention, detector solver revision, and
only the enclosure/refinement/classification tolerances owned by exit, optics, support-edge, and
detector projection. Both keys include the public API, trace schema, and applicable canonical-
serialization revisions; neither key may absorb the other stage's policy or tolerances.
Film thickness and ordered-intensity-only changes do not move a peak-mosaic component tag because
neither changes its stored geometry. Detector, exit-material/policy, LAB transform, actual-
intersection, or projection-tolerance changes invalidate only projected fields and never internal
membership. Incident, crystal/reciprocal, `sample_from_crystal`, internal-medium, mosaic, internal-
root, or internal-tolerance changes invalidate the full row.

`family_id` retains the current `RodCatalog` meaning, `crystal.phase_id:family_key`; it is never
overloaded with mosaic or mount state. `phase_geometry_id` is a separate canonical typed SHA256
fingerprint of the internal tag geometry: crystal/reciprocal catalog revision, internal-medium
revision, mosaic law, sample/crystal mount, and solver/tie convention. `PhasePushforwardInput`
carries this ID and validates it against that payload. Distinct internal phase geometries therefore
cannot collide even when they share a reciprocal family. Parent/population/provider/model-component
identity is physical-ledger provenance only: contributions sharing exact phase geometry share one
geometric tag, and changing only a parent weight or strength model neither duplicates nor moves it.
If a nominally parent-specific input changes mosaic, mount, crystal, or internal medium, it is a new
`phase_geometry_id` and invalidates the internal key. Parent/provider IDs are never added ad hoc to
tag IDs or projection keys.

Cache ownership is explicit and purely functional. `simulate_ordered` accepts an optional immutable
`PushforwardCache` value and returns the next cache value; no module global, closure, singleton, or
hidden object mutation is permitted. The cache contains at most one complete internal-tag batch and
one complete projection batch for one revision, never adaptive cells, images, or strength arrays.
Keys are SHA256 fingerprints of canonical typed byte payloads, never Python hashes or repr strings.
An internal-key miss recomputes and replaces both batches. A projection-only miss reuses the exact
internal batch, recomputes all projected fields, and replaces only the projection batch. Partial
row hits, tolerance-relaxed hits, and stale-status reuse are forbidden. Storage is therefore at
most two immutable tag batches, `O(number of incident/family/branch rows)`. Hit/miss/replacement
provenance is operational metadata and cannot affect scientific arrays, tag IDs, or row order.

## Directions considered

| Direction | Decision | Reason |
|---|---|---|
| Materialize an Ewald raster, flatten it to a `kf` plane, then warp the plane to the detector | Reject | Adds two discretizations, an unnecessary plane, and ambiguous Jacobian ownership |
| Invert every detector pixel back to coating coordinates | Reject for first scope | Multiple preimages, folds, and critical-angle boundaries make the inverse expensive and fragile |
| Deterministic quadrature nodes through the existing bilinear point depositor | Reject | Removes RNG but preserves the point-deposition approximation the strategy is replacing |
| Direct adaptive latent-cell pushforward to sphere bins and exact detector pixel boxes | **Choose** | One measure, tolerance-bounded forward conservation, natural treatment of multiple preimages, and no singular plane-density division |
| Keep Monte Carlo as a second production backend | Reject | Doubles the physics surface and leaves permanent legacy code; it may be used only as disposable comparison evidence before deletion |

## Target interfaces

Only narrow typed results are public. Adaptive cells, clipping polygons, quadrature nodes, and
work queues remain private implementation details.

### `reciprocal/coating.py`

Owns batched continuous mosaic rotation, regular-root evaluation, exact root identity, one Ewald
coarea factor, reciprocal-kernel evaluation, component status, and internal peak-mosaic candidate
solving through the `kf_film_phase_sample_Ainv` payload. It imports no ordered, optics, detector, render, or pipeline
module.

### `render/pushforward.py`

Owns exact sphere-bin and detector-pixel box ownership, mapped-boundary candidate discovery, and
deterministic reduction of certified latent indicator integrals. Polygon clipping may identify
candidate boxes and boundary crossings but never allocates physical mass by mapped area. Sphere-bin
and detector-pixel indicator errors are separate. It contains no adaptive physics orchestration,
crystallography, Ewald root, structure-factor, refraction, or detector-frame equation.

### Certified enclosure ownership

Point values and certified enclosures share the same equation owner:

- `sampling/mosaic.py` owns point and outward-rounded interval bounds for `p_alpha`;
- `reciprocal/coating.py` owns `D`, root/`L`, regular coarea, and fused tangent-kernel enclosures;
- `ordered/amplitudes.py` owns point and certified interval bounds for `S_r(L)`;
- `optics/refraction.py` and `optics/attenuation.py` own exit classification, exit-amplitude, and
  attenuation/optical-weight enclosures using their shared branch selector; and
- `geometry/detector.py` owns point values and outward-rounded enclosures for the detector-plane
  denominator, forward ray time, continuous `(column_px,row_px)`, exact support boundaries, and
  every analytic detector `ValidityCode`; and
- in the frozen fixture, source, population, footprint, and unity-polarization factors are exact
  degenerate intervals.

The private pre-DP-05 orchestrator, and later the pipeline, multiply these nonnegative factor
enclosures exactly once. They may not duplicate ordered or optical equations.
`render/pushforward.py` receives evaluated/enclosed latent mass and indicator information; it owns
no physics-factor enclosure. A refinement difference is never an enclosure.

### Revised `simulate_ordered(...)`

The pipeline owns cross-domain orchestration: it adaptively partitions latent cells, calls the
reciprocal kernel, composes source/population/footprint and ordered strength, calls the accepted
exit/detector seams, and passes already evaluated cells to the pure accumulators. This makes factor
ownership explicit without a stateful callback or a second physics implementation.

It validates one complete `IncidentSampleBatch`, builds the complete `IncidentStateBatch` once and
serially, and only then tiles deterministic component work. Private work rows carry
`parent_row_index` into that immutable parent and preserve input alignment; the controller scatters
results through this index before applying declared state/family/component order. It verifies
`incident_state_id` at boundaries but never sorts or rejoins by that ID, never regenerates source
rows, and never constructs or hashes a sliced public incident batch. Intersection, SAMPLE
direction, air/film wavevectors, complex film-normal component, entrance amplitude, footprint,
wavelength, polarization, source weight, status/validity, IDs/model IDs, and source/sample/material/
incident revisions pass through unchanged to their distinct proof and provenance consumers.

DP-02C freezes one canonical detector-independent internal partition, reduction order, and
`EwaldCoatingResult`. DP-04A--DP-04C may refine descendants for exit/detector classification
and pixel allocation, but those descendants cannot be reduced back into or otherwise change the
accepted sphere result. Detector pose and detector tolerances therefore cannot perturb the
upstream coating diagnostic.

The final target inputs preserve current validated domains and add the declared dovetail seams:

```text
one explicit IncidentSampleBatch whose rows retain source/wavelength identity
one or more immutable PhasePushforwardInput values, each containing its crystal/material optics,
  explicit detector-independent complete RodCatalog in canonical rod_id order and
  `rod_catalog_revision`, validated canonical `phase_geometry_id`,
  WrappedMosaicParameters, population/parent identity, and strength provider
compiled instrument
polarization policy and provenance
explicit versioned frozen pushforward tolerances
cache: PushforwardCache | None
```

The DP-01--DP-06 milestone may raise an informative unsupported-model error before work for domains
outside the frozen proof fixture. Such rejection is temporary and cannot survive sole-runtime
cutover for a domain accepted by the recorded post-BKI public API. DP-06C--DP-06E2 provide current
parity, and DP-06F--DP-06H provide the required first dovetail
implementations. Otherwise a human-approved deprecation/migration is required. No component is
silently dropped or approximated.

Removed inputs:

```text
MosaicOrientationBatch
alpha/azimuth cell counts
selection_seed
draw_count
candidate chunk size
```

The result contains or references the exact inputs:

```text
phase-indexed incident transport
the exact complete RodCatalog identity for each phase
EwaldCoatingResult
DetectorPushforwardResult
CoatingComponentTagBatch
SPECULAR_INTENSITY_EXCLUDED status for exact (h,k)=(0,0)
next_cache: PushforwardCache
```

`EwaldCoatingResult` contains equal-solid-angle sphere-bin mass/density, per-rod/root/family totals,
status counts, moment summaries, certified numerical upper bounds, and separately labelled
refinement heuristics. `DetectorPushforwardResult` replaces `DepositionResult` and contains the
native image, separate pre-optics exit classification, post-optics detector categories, closure
residuals, independently certified numerical upper bounds, and separately labelled refinement
heuristics.
`CoatingComponentTagBatch` is immutable; its rows have no assigned/deposited event-mass field and
cannot alter source weights, photon/coating ledgers, or detector pixels.

No compatibility aliases are retained for the removed runtime contracts at handoff; the temporary
v1-fixed tolerance-loader alias exists only through DP-05V2 and is removed by DP-05V3.

DP-00 records the exact post-BKI `CONTRACT_API_VERSION`, trace schema version, and immutable
reference-pack version before code changes. `stage_tolerances_v1.json` remains byte-for-byte and
hash-identical historical T02--T05 evidence. DP-00A adds a schema-distinct v2 tolerance artifact;
named loaders require explicit v1/v2 selection and reject field mixing, while the pre-existing no-
argument name temporarily resolves to v1 to keep unmigrated callers green. Existing proof producers
remain v1/legacy-trace consumers until DP-05V1/V2, while every new deterministic stage uses v2/the
reserved new trace version. No single trace mixes versions. Pre-cutover deterministic helper types
remain private and unexported. DP-06V advances the recorded public API version exactly once only
after the complete final signature/types exist. DP-05V1
activates the reserved trace version exactly once, DP-05V1/V2 migrate producers, and DP-05V3 removes
the ambiguous tolerance alias before the new dispatcher is accepted.
Immutable reference-pack v4 checks remain explicitly historical and are never relabelled as the
current trace contract.

## Affected production and proof files

### Retain as equation authorities

- `src/rasim_next/core/frames.py`
- `src/rasim_next/core/transforms.py`
- `src/rasim_next/core/interfaces.py`
- `src/rasim_next/core/scattering.py`
- `src/rasim_next/sampling/source.py` — retained source boundary; the fixture takes only its center.
- `src/rasim_next/reciprocal/ewald.py` — analytic root and scalar coarea oracle.
- `src/rasim_next/reciprocal/rods.py`
- `src/rasim_next/ordered/amplitudes.py`
- `src/rasim_next/materials/crystal.py`
- `src/rasim_next/materials/optics.py`
- `src/rasim_next/optics/refraction.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/geometry/detector.py`

### Add

- `src/rasim_next/proof/bi2se3_pushforward_fixture.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py`
- `src/rasim_next/pipeline/proof.py`
- `src/rasim_next/proof/stage_tolerances_v2.json`
- `scripts/generate_bi2se3_pushforward_images.py`

### Rewrite

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/core/contracts.py`
- `src/rasim_next/reciprocal/rods.py`
- `src/rasim_next/ordered/amplitudes.py`
- `src/rasim_next/ordered/proof.py`
- `src/rasim_next/stacking/finite_intensity.py`
- `src/rasim_next/geometry/__init__.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/proof/__main__.py`
- `src/rasim_next/proof/core.py`
- `src/rasim_next/proof/reference.py`
- `src/rasim_next/proof/tolerances.py`
- `src/rasim_next/proof/traces.py`
- `src/rasim_next/reciprocal/proof.py`
- `src/rasim_next/stacking/proof.py`
- `src/rasim_next/optics/attenuation.py`
- `src/rasim_next/geometry/proof.py`
- `pyproject.toml` and `uv.lock` for the isolated locked image dependency group.
- `FILE_MANIFEST.json` in this corrected-planning commit, DP-00R's exact planning reconciliation,
  and DP-15 after the final implementation file set is known; no ordinary intermediate code task
  rewrites it.
- the existing compact test modules and live contracts named by the tasks below.

### Delete after the named replacement gate

- `src/rasim_next/reciprocal/events.py`
- `src/rasim_next/pipeline/intersections.py`
- `src/rasim_next/pipeline/selection.py`
- `src/rasim_next/render/deposition.py`
- `scripts/generate_bi2se3_detector_image.py`

Also delete the retired orientation-batch constructors, selected-candidate contracts, RNG selection
arguments, bilinear-deposition tests, Monte Carlo frequency tests, and imperative legacy task
instructions. Indexed numbered tasks are rewritten as routing/history stubs rather than deleted. Do
not delete `reciprocal/ewald.py`, source/incident transport, exit refraction, detector geometry, or
the independent scalar proofs; they are validated authorities, not legacy residue.

Current `main` still carries the completed BKI `tasks/plan.md` and `tasks/todo.md` with a stale
successor graph; the unmerged planning sequence through `0953cab` proposes their shared beam-to-
`ki` replacement and deterministic/parallel ownership amendments. DP-00R
reconciles both histories without rewriting completion evidence or treating planning as an accepted
implementation. All unrelated dirty/untracked files remain owner-controlled. DP-00 is blocked
until DP-00R lands, shared Checkpoint K is implemented and accepted on `main`, its affected gates
pass, status is warning-free, and no parallel/coating writer owns overlapping paths. Later
retirement tasks reconcile remaining live text only on that approved base.

## Dependency graph

```text
committed corrected plan on merged BKI main
  -> DP-00R post-merge BKI/beam-plan routing and manifest reconciliation
  -> shared beam-to-ki implementation + accepted Checkpoint K on main
  -> DP-00 isolated workbranch from warning-free reconciled main
  -> DP-00A versioned tolerance contract
  -> DP-00B proof-base repair + Checkpoint 0 approval
  -> DP-00C single frozen-fixture authority
  -> DP-00D certified-enclosure feasibility gate
  -> DP-01 frozen fixture + pointwise coating
  -> DP-01A ordered-strength enclosure
  -> DP-02A topology/tangent certificates
  -> DP-02B bounded scalar latent integration
  -> DP-02C sphere-indicator integration + Checkpoint A
  -> DP-03 sole-top-interface exit and private deterministic seam
  -> DP-03A exit/optical enclosures
  -> DP-04A exit/status boundaries
  -> DP-04B detector-map topology
  -> DP-04C detector-pixel indicator integration
  -> DP-05 end-to-end deterministic path
  -> DP-05V1/DP-05V2 trace-version migration
  -> DP-05V3 remove ambiguous tolerance-loader alias
  -> DP-05P seventh proof dispatcher
  -> DP-06I internal representative/cache
  -> DP-06P exit/detector projection cache
  -> Checkpoint B0 accepted-domain inventory
  -> DP-06C incident/source/wavelength parity
  -> DP-06D smooth-mixture parity
  -> DP-06E1 zero-width-atom beta pushforward
  -> DP-06E2 atom pipeline/tag/cache integration
  -> DP-06F first general-family deterministic path
  -> DP-06G first ordered/stacking strength-provider seam
  -> DP-06H first phase/parent composition path
  -> Checkpoint B: implementation and resource acceptance
  -> DP-06U factor the sampled comparator without changing the public API
  -> DP-06V one-time final public API activation
  -> DP-06V2 migrate numeric CLI/seventh proof to public API
  -> DP-06A/DP-06B architecture/scientific-document synchronization
  -> DP-06J/DP-06K/DP-06L/DP-06M all remaining live instructions
  -> Checkpoint D: pre-cutover live-instruction and parity acceptance
  -> DP-07A private sampled-comparison + selector/intersection deletion
  -> DP-07B reciprocal proof migration + event-builder deletion
  -> DP-08 sampled-event contract deletion
  -> DP-09 orientation/deposition deletion
  -> DP-10 locked two-image tool
  -> Checkpoint C: residue-free production path
  -> DP-13 unnumbered-plan retirement
  -> DP-13A conditional BKI-plan retirement
  -> DP-13B repository-wide residual scan
  -> DP-15 final proof, two external images, cleanup, one coherent commit
```

Tasks are sequential under one writer. Read-only derivation, test, and performance review may run
in parallel. Each task changes no more than five files and must leave its focused verification
green. Temporary coexistence in DP-05 is allowed only in the unmerged workbranch and must be
removed immediately by DP-07A/DP-07B after Checkpoint D. Every file list below is exact, not
illustrative. If the post-BKI audit discovers another required file, the plan must be amended into
another <=5-file green task before implementation touches it.

The companion checklist is frozen during ordinary code tasks because it is not an undeclared sixth
file. Between synchronization points, progress/evidence lives in the external run log and task
commit/handoff text. DP-06J is the only pre-cutover checklist synchronization task; DP-15 performs
the final status/check reconciliation. No other task ticks boxes opportunistically.

## Atomic implementation tasks

### DP-00R: Reconcile the completed BKI plan with the post-merge correction

**Files (3):**

- `tasks/plan.md`
- `tasks/todo.md`
- `FILE_MANIFEST.json`

**Work:** Preserve BKI-00--BKI-17 as completed historical work. Reconcile the stale BKI-15 successor
edges with the planning-only `af456b5..0953cab` beam findings and this deterministic correction. The sole
active overlapping sequence is shared beam-to-`ki` implementation through accepted Checkpoint K,
then this deterministic implementation; do not claim either correction preceded BKI-15 or that a
planning commit satisfies Checkpoint K. Record merge
`3af2f4f61d3bc73d8d54891eb867a964240f46cd`, incorporated implementation
`d5eed2524a636c7c190b8f9be300d1d73728884e`, beam-planning commit
`af456b5988076c400eea972fb4f2ac2d647891dc`, ownership-amendment commit
`0953cab668b489fd4c4d5a1d4de070fcab165295` (or their approved replacement), this corrected
planning commit's exact hash, and the approved rebase/merge order. Record BKI-04A/BKI-14 as historical >5-
file process deviations with owner-approved audit dispositions; a successful rerun is revalidation,
not retroactive atomicity compliance. Recompute the exact manifest entries for the two changed
planning documents and the already tracked corrected plan/checklist, then review the complete
manifest delta.

**Verify:** exact successor scan, preserved completion/evidence text, recorded hashes, affected
BKI-15 documentation/proof commands, `tools/check_docs.py`, `scripts/verify_seed.py`, working/cached
diff checks, and warning-free status with every untracked path visible. Require docs/seed and all
non-baseline proof stages to pass; record the exact expected reciprocal/stacking proof-base contract
failure for DP-00B. Confirm the `codex/beam-ki-findings-plan` owner disposition covers the committed
`af456b5` and `0953cab` deltas plus the observed dirty T09/T10/T11/parallel-plan delta by exact hash,
and no parallel or old coating-validation writer owns an overlapping path.

**Acceptance:** the repository tells one truthful post-merge history, routes overlapping work only
through Checkpoint K and then the deterministic successor, passes docs/seed and every unaffected
BKI integrity gate, carries only the explicit DP-00B proof-base repair, and does not describe BKI-
04A or BKI-14 as <=5-file compliant.

**Dependencies:** this committed corrected plan/checklist and owner disposition of unrelated
working-tree/status-warning paths, including exact reconciliation of the overlapping beam-findings
worktree's commit and dirty planning patch.

### DP-00: Create the isolated workbranch

**Files:** none.

**Work:** After DP-00R lands, shared Checkpoint K is implemented/accepted, and the owner resolves
unrelated working-tree/status-warning paths, record the approved reconciled main SHA as
`PUSHFORWARD_BASE_SHA`. Record the accepted Checkpoint K commit plus contract/material/sample/
incident revision hashes and independently revalidate its source-to-`ki` invariants. Confirm BKI-
00--BKI-17 remain
complete, their hashes/evidence are recorded truthfully, and any still-live obligation is assigned
to a retained owner before BKI files can retire. Parallel Task 1.1 and old coating-validation work
remain inactive while the deterministic successor owns overlapping paths. Resolve every permission
warning before calling the base clean. Confirm DP-00R replaced stale successor routing, refreshed
the manifest, reran affected BKI-15 gates, and recorded BKI-04A/BKI-14 as historical >5-file
deviations with approved audit dispositions and independent revalidation. Perform a fresh read-only
active-worktree audit, record the owner disposition of detached `55dc336`, and record whether the
approved local main (observed four commits ahead) must synchronize with `origin/main`. Then perform
a fresh read-only API/import/fixture audit
against that exact SHA; any material mismatch with this plan returns to human review before branch
creation. Create a separate worktree and branch `codex/deterministic-ewald-pushforward`.
Never reuse the existing `codex/continuous-ewald-coating-validation` ref or delete it without an
owner-approved disposition of its unique commit.

Planned PowerShell commands, after resolving the target path and confirming it does not exist:

```powershell
$pushforwardBaseSha = git rev-parse main
git worktree add -b codex/deterministic-ewald-pushforward `
  ..\SLATE-rMC-wt-deterministic-ewald $pushforwardBaseSha
```

**Verify:** corrected plan/checklist object hashes, DP-00R handoff and manifest delta, corrected
Checkpoint-K-then-deterministic routing, accepted Checkpoint K evidence, completed BKI-16/BKI-17,
no active retired/parallel/follow-up writer,
BKI-04A/BKI-14 historical-deviation disposition and independent revalidation, explicit
preservation/port/archive decision for the retired validation ref, base SHA, reference-pack hash,
local-versus-remote and every active/detached worktree disposition, warning-free clean status with
all untracked paths visible, dependency sync, fresh post-BKI audit,
and current full tests. Run every
currently runnable proof with its legacy environment and record the known reciprocal/stacking
baseline-contract failures; do not claim a six-proof green baseline before DP-00B repairs them.

**Acceptance:** Checkpoint K and the reconciled post-merge docs/tests/status gates are green and
only the explicitly carried DP-00B proof-base failures remain; the approved corrected plan is present on the base; no
user change or unique retired-branch commit is moved, overwritten, or deleted; status has no
permission warning; and the new writer exclusively owns the new worktree and every overlapping
path.

**Dependencies:** DP-00R, accepted shared beam-to-`ki` Checkpoint K, warning-free approved
reconciled main, and no overlapping writer.

### DP-00A: Freeze the new tolerance contract

**Files (5):**

- `src/rasim_next/proof/stage_tolerances_v2.json` (new)
- `src/rasim_next/proof/tolerances.py`
- `src/rasim_next/proof/traces.py`
- `tests/test_core_coordinates.py`
- `docs/VALIDATION.md`

**Work:** Before implementing coating code, add approved stages for reciprocal coating mass,
unit-direction moments, Ewald-bin mass, detector-pixel mass, pushforward-ledger closure, and
`L_peak`. Scalar totals, individual unit-direction moment components, and ledger closure use

```text
abs(candidate-reference) <= 1.1284538007351267e-23 angstrom^2
                             + 1e-8*reference_scale,
```

where `reference_scale=max(abs(independent_reference),r_e^2)`. Sphere-bin and detector-pixel arrays
use one global norm, not a per-element total-mass allowance:

```text
sum(abs(candidate_array-reference_array))
  <= 1.1284538007351267e-23 angstrom^2
     + 1e-8*array_reference_scale.
```

Here `array_reference_scale=sum(abs(reference_array))` when a full array oracle exists. Otherwise,
the candidate-independent scale is the corresponding independently certified nonnegative scalar
total: `M_coat_reference` for the sphere and `M_deposited_postopt_reference` for the detector. If
neither a full array oracle nor that scalar oracle is available, the gate cannot be evaluated and
fails; candidate totals are never substituted. The artifact/parser records the scalar versus
`global_l1` comparison kind and scale provenance. A certified scalar or array numerical bound must
itself be at most the corresponding frozen tolerance; an embedded-pair or two-refinement difference
alone is not a bound.
Freeze separate stages for tangent-chart inversion/Jacobian, sphere-bin indicator allocation,
detector-pixel indicator allocation, and topology-certification residuals. Freeze `L_peak` with
`atol=1.4210854715202206e-14`, `rtol=2.2737367544328376e-13`, and
`scale=max(abs(L_reference),1)`. Keep the existing pixel-coordinate tolerance. Record the new
discriminant and angular-gradient stages in `angstrom^-2` and `angstrom^-2/rad` with
`atol=1.4210854715202206e-14`, `rtol=4.547473508866709e-13`, and reference scale `|ki|^2`;
transversality requires the gradient norm to exceed that full bound. Record the new artifact
SHA256 and prohibit a candidate-dependent scale.

Keep `stage_tolerances_v1.json` byte-for-byte and hash-identical. V2 is a strict schema superset: it
copies every v1 stage with numerically identical comparison semantics before adding
`comparison_kind`, candidate-independent scale provenance, the certified-bound rule, and
pushforward stages. Add named `load_stage_tolerances_v1()` and `load_stage_tolerances_v2()` loaders
plus immutable v1/v2 SHA constants. During migration only, retain the existing no-argument
`load_stage_tolerances()` as an explicitly deprecated alias fixed to v1 so every intermediate tree
stays green; new deterministic code calls the v2 loader explicitly, and old callers migrate in
DP-05V1/DP-05V2. DP-05V3 removes the ambiguous alias after its final test caller migrates. Every
loader rejects schema/version mixing.

Record the post-BKI contract and trace versions before editing; reserve one deterministic API
increment and one deterministic trace increment instead of hardcoding the pre-BKI numbers.
DP-00A freezes separate `LEGACY_TRACE_SCHEMA_VERSION` and
`RESERVED_PUSHFORWARD_TRACE_SCHEMA_VERSION` constants while `CURRENT_TRACE_SCHEMA_VERSION` remains
the legacy value. New deterministic traces explicitly use the reserved value. DP-05V1 activates the
reserved value as current exactly once, and DP-05V1/DP-05V2 migrate all producers. Immutable
reference-pack v4 remains separately named historical evidence.

Update `docs/VALIDATION.md` in the same task: retain the previous hash only as explicitly historical
T02--T05 evidence and name the new artifact hash as the current authority before any coating code
uses it. DP-06B later adds accepted pushforward results without deferring this authority change.

Do not alter any tolerance after observing a coating result.

**Verify:** v1 byte/hash identity, v2 artifact schema/hash, exact numerical identity of every
inherited v1 stage, named-loader selection, temporary no-argument-to-v1 behavior, exact stage
lookup, rejection of mixed schemas, scalar/global-L1 norm behavior and scale
provenance, rejection of a candidate-dependent scale, rejection of a reported error estimate that
is not a certified upper bound below its frozen stage tolerance, current-versus-historical hash
wording in `docs/VALIDATION.md`, a mutation interpreting global L1 as elementwise tolerance, and
`tools/check_docs.py`.

**Acceptance:** v1 is unchanged; v2 is a strict numerical superset; the v2 stage IDs, comparison
norms, absolute floors, scale rules, version migration, and artifact hash are frozen without
observing new coating output; validation names v2 as the current authority; no scientific code has
been added.

**Dependencies:** DP-00.

### DP-00B: Repair proof baseline ownership

**Files (4):**

- `src/rasim_next/reciprocal/proof.py`
- `src/rasim_next/stacking/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_stacking_transition.py`

**Work:** BKI-13 must already have removed and tested the dead `integration` registration; fail the
preflight rather than editing or recreating that completed work. Migrate both the reciprocal
hard-coded T03 `PROOF_BASE_SHA` gate and the stacking dependency on retired
`origin/codex/proof-base` to the recorded `PUSHFORWARD_BASE_SHA` contract, without changing their
scientific oracles. Add negative tests for a missing/wrong base SHA. The six
accepted commands are `core`, `geometry-optics`, `mosaic-ewald`, `ordered-reflectivity`,
`references`, and `stacking-transition`.

**Verify:** proof-command registry importability, missing/wrong base rejection, correct-base
acceptance, and all six proof commands from the clean worktree.

**Acceptance:** all six documented proofs are green with the approved base SHA; no proof depends
on a retired branch/ref, and the already-correct BKI dispatcher remains unchanged and importable.

**Dependencies:** DP-00A.

### Checkpoint 0: tolerance and proof authority

- A human approves the immutable tolerance artifact and its hash.
- Every documented proof command is runnable from the clean worktree with the declared external
  cache environment.
- Failure stops the branch before any new coating implementation.

### DP-00C: Install the single frozen-fixture authority

**Files (4):**

- `src/rasim_next/proof/bi2se3_pushforward_fixture.py` (new)
- `src/rasim_next/reciprocal/rods.py`
- `tests/test_integration.py`
- `scripts/generate_bi2se3_detector_image.py`

**Work:** Add one pure, immutable builder for the complete frozen 5-degree source, unbounded sample,
instrument, crystal/material, pure 2-degree Gaussian mosaic, polarization declaration, and the
explicit inclusive `[-5,5]` `RodCatalog`. Build it only with `build_rod_catalog`, freeze h-major/
k-minor order, the literal-LF integer-table hash, 121/120 counts, and the named canonical
`rod_catalog_revision` helper/hash above.
Preserve the BKI source model/RNG/provenance assertions while
changing the diagnostic case to the exact one-row center with empirical `source_weight=1.0`.
Construct the complete source once and derive its complete `IncidentStateBatch` through the
accepted Checkpoint K boundary; never hand-author `ki`, hash a public slice, or create a competing
nominal ray. The module imports no Matplotlib and no
new coating implementation. Migrate the post-BKI permanent default-builder invariant away from the
old image script to this authority before that script can be deleted. Rename/relabel the old
12-degree builder as legacy comparison state, not a second canonical fixture; its module text and
API must no longer claim to be the canonical or single authority.

**Verify:** exact literals and provenance, count-one center behavior with `source_weight=1.0`,
unchanged source/photon ledgers, unbounded support with absent dimensions, detector/frame contracts,
CIF/material/accepted-Checkpoint-K revision hashes, and importability without Matplotlib.

**Acceptance:** every subsequent proof, test, and image tool names this builder; no deterministic
fixture literal block is copied elsewhere, and the old script has no permanent-test caller.

**Dependencies:** Checkpoint 0.

### DP-00D: Prove certified-enclosure feasibility before production code

**Files (5):**

- `docs/VALIDATION.md`
- `src/rasim_next/reciprocal/proof.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_geometry_optics.py`

**Work:** Build the smallest executable proof-only prototypes needed to demonstrate outward-rounded
or interval enclosures for Ewald support/topology, the fused tangent kernel, ordered-strength
composition, sole-top-interface classification, exit/attenuation weight, detector status
boundaries, and latent indicator uncertainty. Compare every enclosure with dense direct
enumeration and analytic identities. Record the chosen rounding/interval method and its dependency
impact. If the current stack cannot provide rigorous outward rounding without a new dependency or
the bounds cannot contract below v2 tolerances on representative hard cases, stop for human
approval before DP-01; do not implement an adaptive heuristic and call it certified.

**Verify:** exact topology cases, disconnected-support island, interval spanning `s=0`, optical
critical boundary, detector fold/status boundary, enclosure coverage, and monotone contraction.
Run both focused test files, both existing proof commands, and `tools/check_docs.py`.

**Acceptance:** the certification strategy is executable and dependency-complete before production
orchestration exists; all claimed bounds enclose independent values and have a demonstrated path
below frozen tolerances. Only distinct independent-oracle/mutation code remains in the listed proof
and test files; exploratory scaffolding is removed before DP-00D finishes.

**Dependencies:** DP-00C.

### DP-01: Freeze the one-ray fixture and pointwise coating equation

**Files (4):**

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/reciprocal/coating.py` (new)
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`

**Work:** Add the folded continuous mosaic density as a new quantity without changing the existing
signed-density trace, and add a pure batched reciprocal-kernel evaluator
for explicit rod/root `(alpha,beta)` points. Consume the DP-00C fixture authority—not production
`simulate_ordered`—and its explicit one-row `IncidentSampleBatch`. Assert the CIF/material hashes
and values, exact source-derived internal `ki`, 121/120 rod counts, 240 ordered nonzero-`m`
`(rod_id,root_label)` slots, root ordering, residuals, and one coarea factor. Consume exactly
`ordered_event_result(...).scattering_strength_A2` with its `ordered/raw_unit_cell/UNIT_CELL`
metadata; do not multiply by `r_e^2` again. Keep the old enumerator only as a temporary oracle.

**Verify:** analytic roots, direct delta/coarea evaluation, folded-continuous normalization, signed-
density trace preservation, beta periodicity, tangent/no-root classification, and mutation
detection for omitted/doubled folding, omitted/doubled coarea, omitted/doubled electron-to-area
conversion, and added `sin(alpha)`.

**Acceptance:** pointwise values are finite/nonnegative on regular support; every rod/root identity
is exact; no ordered, optics, detector, render, or pipeline import enters `coating.py`.

**Dependencies:** DP-00D.

### DP-01A: Add ordered-strength enclosures

**Files (3):**

- `src/rasim_next/ordered/amplitudes.py`
- `src/rasim_next/ordered/proof.py`
- `tests/test_ordered_reflectivity.py`

**Work:** Add a certified nonnegative enclosure for
`ordered_event_result(...).scattering_strength_A2` over a closed `L` interval. It shares the point
evaluator's normalization/model metadata and once-only electron-to-area conversion. It may use
analytic extrema, outward-rounded subintervals, or direct finite enumeration where exact; it may
not duplicate the structure-factor equation in the pipeline.

**Verify:** analytic constant/linear cases, dense direct enumeration, an interior extremum, bound
contraction, and mutations for omitted/doubled `r_e^2` or wrong normalization. Run the focused
ordered test and `ordered-reflectivity` proof.

**Acceptance:** every ordered strength point on the tested closed interval lies inside a finite
nonnegative enclosure that can meet v2 tolerance; point and enclosure APIs report identical
ownership metadata.

**Dependencies:** DP-01.

### DP-02A: Certify Ewald topology and stable tangent charts

**Files (4):**

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Use analytic identities plus domain-covering interval branch-and-bound to certify every
connected `D>0`, `D=0`, and `D<0` region on the periodic latent domain. A sampled sign grid is never
an absence proof. Split one-sided support at every certified `D=0` boundary. At a regular tangent,
choose a nonsingular `(D,eta)` chart, use `D=s^2`, and evaluate the fused continuation
`2*|ki|/abs(det d(D,eta)/d(alpha,beta))`, including at `s=0`; never form `0*inf`. A zero gradient,
unbounded chart denominator, or unresolved topology is a typed failure and stops the component.

**Verify:** transverse and degenerate tangencies, exact tangent endpoint, near-tangent equality to
separated regular factors, intervals spanning `s=0`, periodic seam, a disconnected support island,
and mutations for `0*inf`, omitted/doubled chart factor, or sampled-only topology. Run focused
mosaic/integration tests and the existing `mosaic-ewald` proof.

**Acceptance:** every root-support component has a complete certified topology and stable chart or
an explicit blocking status; no physical mass or sphere allocation is added yet.

**Dependencies:** DP-01A.

### DP-02B: Add bounded scalar latent-cell integration

**Files (5):**

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py` (new)
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Integrate one rod/root at a time using the DP-02A support partition and the complete
nonnegative factor enclosures owned by mosaic, reciprocal, and ordered modules. An embedded pair
may guide refinement, but the reported upper bound comes only from outward-rounded full-integrand
enclosures. Stream canonical component tiles; retain no rod-by-cell Cartesian product. A private
preproduction orchestrator may exist only in `tests/test_integration.py`, which DP-05 also owns;
DP-05 removes it after installing production orchestration. `reciprocal/proof.py` retains only
independent equations/invariants, not a second orchestration path.

**Verify:** constant/polynomial integrands, tangent cells, direct high-order enumeration, once-only
factor mutations, certified-bound coverage/contraction, and canonical/alternate tile order.

**Acceptance:** scalar coating totals and moments satisfy v2 tolerances with a certified bound;
heuristic differences are separately labelled and no mapping code affects the result.

**Dependencies:** DP-02A.

### DP-02C: Integrate latent indicators into the Ewald-sphere raster

**Files (4):**

- `src/rasim_next/render/pushforward.py`
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Certify the periodic seam, poles, bin boundaries, and sphere-map folds. For every bin `B`
evaluate `integral_C f_coat(z)*indicator[Psi(z) in B] dz` in latent coordinates. A wholly-inside
cell contributes its full latent integral; a disjoint cell contributes zero; a crossing cell is
split. Nominal indicator quadrature assigns mass exhaustively. Moving an unresolved cell's nominal
mass between bins has certified array-L1 error
`E_C <= U_true,C + M_nom,C`, where the true and nominal cell masses are separately proven
nonnegative, `U_true,C` encloses the true mass, and `M_nom,C` is the actual nominal allocated mass.
The shorthand `2*U_C` is allowed only after proving `M_nom,C <= U_C` with positive quadrature
weights and the same outward-rounded full-integrand enclosure. Resolved cells contribute their
certified scalar quadrature error; unresolved cells contribute the allocation expression above,
and the global bound sums each cell exactly once. A tighter certified indicator-vector enclosure
may replace this bound. Mapped polygons only enumerate
candidate bins and locate crossings; mapped area never allocates physical mass. Freeze the
detector-independent partition and reduction order.

**Verify:** poles, seam/handedness, exact edge ownership, all preimages, a crossing with no initial
node, a strongly varying map determinant mutation, rod/root/family ledgers, global-L1 bound,
refinement contraction, and invariance to detector pose/tolerance.

**Acceptance:** sphere totals/moments and the certified global-L1 bound meet v2. Finest-refinement
normalized L1 change `<=1e-3` is supplemental only. No sphere/plane Jacobian or area fraction is a
physical weight.

**Dependencies:** DP-02B.

### Checkpoint A: internal coating

- Focused mosaic/Ewald tests and proof pass.
- The first requested image can be rendered from deterministic numeric bins, although no PNG is
  retained yet.
- The old runtime has not been deleted because detector replacement is not yet proven.

### DP-03: Freeze the sole-top-interface exit and private deterministic-wave seam

**Files (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`

**Work:** Introduce a narrow private, unexported deterministic internal-wave batch without changing
`CONTRACT_API_VERSION` or the accepted public `simulate_ordered` signature. Only the private seams
needed by deterministic cells may use it; validated scalar refraction/detector equations remain
authoritative. DP-06V later promotes the complete final superset atomically after all result/cache/
phase-input fields exist. For each regular
`kf_film_phase_sample_Ainv`, classify the declared top surface first with a certified sign
enclosure: positive lower bound is eligible, negative upper bound is pre-optics `BACKWARD`, exact
analytic zero is pre-optics `PARALLEL`, and any other enclosure containing zero must split/refine
or remain explicitly unresolved. A numerical tolerance controls enclosure termination only and
never creates a physical dead band. Only eligible mass calls the accepted non-growing
refraction branch and top-surface attenuation. Do not reinterpret rejected mass as an undeclared
bottom exit. Keep the old event-shaped entry point only until cutover so this intermediate tree is
green for the sampled runtime, but migrate `geometry/proof.py` and `test_geometry_optics.py` in this
same task from `ScatteringEventBatch`/`transport_scattering_events` to the narrow private
deterministic-wave seam. No proof or geometry test may retain an event-shaped import after DP-03.
Transform an accepted external vector exactly
once as

```text
r_hat_lab = lab_from_sample.apply_vector(kf_air_sample) / k0
```

and project a ray beginning at the actual sample intersection. The pipeline, not
`reciprocal/coating.py`, attaches exit/detector fields to internal tag candidates.

**Verify:** exhaustive positive/negative/zero normal support; tangential conservation; external
dispersion `|kf_air|=2*pi/lambda`; evanescent/critical statuses; `n -> 1`; SAMPLE-to-LAB round trip;
global rigid-rotation covariance; detector ray/pixel round trip; private-schema invariants and proof
that `CONTRACT_API_VERSION`/public exports are unchanged;
zero event-shaped imports/usages in `geometry/proof.py` and `test_geometry_optics.py`; and mutations
for wrong side, frame, normal sign, direction rule, or ray origin.

**Acceptance:** the frozen 5-degree case reproduces accepted frame/optical tolerances, proof/tests
use only the private deterministic-wave seam, and no downstream code reconstructs the exit branch
or rotation.

**Dependencies:** Checkpoint A.

### DP-03A: Add exit-amplitude and attenuation enclosures

**Files (4):**

- `src/rasim_next/optics/refraction.py`
- `src/rasim_next/optics/attenuation.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`

**Work:** Add certified exit-amplitude, manuscript uniform-depth attenuation-average, and combined
optical-weight enclosures over a top-surface-eligible latent cell. Use the shared complex-normal-
wavevector branch selector and apply `|t_in*t_out|^2` once. An
interval crossing a top-exit, critical, evanescent, or attenuation-status boundary must split and
cannot receive one optical enclosure. Point and enclosure paths share the authoritative equations.

**Verify:** analytic constant cells, critical-boundary splits, direct dense enumeration, monotone
attenuation limits, nonnegative finite weights, enclosure contraction, and mutations for a second
amplitude factor, wrong branch, wrong thickness direction, or same-surface treatment of negative
outward propagation. Run the geometry test and `geometry-optics` proof.

**Acceptance:** every optically supported point is enclosed, every rejected point receives no
optical weight, and the bound can meet v2 without duplicating optical equations.

**Dependencies:** DP-03.

### DP-04A: Certify top-exit and detector-status boundaries

**Files (5):**

- `src/rasim_next/render/pushforward.py`
- `src/rasim_next/geometry/detector.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`
- `tests/test_integration.py`

**Work:** Refine descendants of frozen DP-02C cells without changing the accepted sphere result.
Add detector point/enclosure APIs to the existing geometry owner; render consumes those status and
coordinate enclosures and owns only latent-indicator reduction. Certify/split `g_exit=0`, the exit
critical boundary, detector parallel denominator,
forward/backward ray time, support boundary, and every analytic `ValidityCode` transition. An
interval crossing a boundary remains unresolved and is refined. `NUMERIC_FAILURE` is never a
geometric category; nonzero failed mass above the absolute floor aborts.

**Verify:** all top-exit and detector statuses, narrow status islands, exact status-boundary ties,
wrong-side mutations, exhaustive pre-/post-optics ledgers, and detector-change invariance of the
sphere. Run both focused tests plus existing geometry/mosaic proofs.

**Acceptance:** every accepted descendant has exactly one certified stage-appropriate status, and
the full upper mass of unresolved cells is explicitly bounded rather than balanced into a ledger.

**Dependencies:** DP-02C and DP-03A.

### DP-04B: Certify detector-map topology and all preimages

**Files (5):**

- `src/rasim_next/render/pushforward.py`
- `src/rasim_next/geometry/detector.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`
- `tests/test_integration.py`

**Work:** Extend the same geometry-owned detector equations to certify and split every detector-map
fold and recursively enclose mapped boundaries in detector-native coordinates. Preserve all
preimages. Render must not duplicate the detector denominator, ray-time, coordinate, support, or
status equations. Mapped simple polygons may enumerate
candidate pixels and locate crossings, but self-intersections or uncertified curvature force
refinement and no polygon area carries mass.

**Verify:** identity/tilted maps, non-square pixels, certified zero-determinant folds with multiple
preimages, curved boundaries, self-intersection rejection, row/column orientation, and a narrow
fold missed by initial nodes.

**Acceptance:** every retained cell has certified map topology/candidate boxes or an explicit
unresolved bound below tolerance; no inverse rasterizer or map-determinant weight exists.

**Dependencies:** DP-04A.

### DP-04C: Integrate latent indicators into exact detector pixels

**Files (3):**

- `src/rasim_next/render/pushforward.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_integration.py`

**Work:** For each exact half-open pixel `P`, evaluate
`integral_C f_postopt(z)*indicator[Phi(z) in P] dz` in latent coordinates. Certified wholly-inside
cells contribute all latent mass; disjoint cells contribute zero; crossings split. Nominal point
indicators assign every quadrature contribution to exactly one status/pixel. Each unresolved
boundary cell contributes `U_true,C + M_nom,C` to array-global-L1 error after proving both masses
nonnegative and `U_true,C` valid. It may contribute `2*U_C` only when positive quadrature weights
and the same full-integrand enclosure prove `M_nom,C <= U_C`. Resolved cells contribute certified
scalar quadrature error, unresolved cells contribute the allocation expression, and every cell is
summed exactly once unless a tighter certified indicator-vector enclosure replaces it. Polygon
clipping is only a candidate/topology accelerator. Never apportion by mapped area, call
`deposit_bilinear`, or use a pointwise map determinant as a physical factor.

**Verify:** exact edge/corner ties, non-square pixels, multiple preimages, independent high-order
latent quadrature on a tiny detector, a constant latent density under strongly varying map
determinant, exhaustive ledgers separate from accuracy, solid-angle-factor mutation, bound coverage,
and refinement contraction.

**Acceptance:** analytic/full-fixture ledgers and certified global-L1 detector bound meet v2;
normalized L1 refinement change `<=1e-3` and centroid shift `<=0.05 px` are supplemental only; all
pixels are finite/nonnegative.

**Dependencies:** DP-04B.

### DP-05: Compose a temporary deterministic end-to-end path

**Files (5):**

- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py`
- `tests/test_integration.py`
- `scripts/generate_bi2se3_pushforward_images.py` (new numeric-only CLI skeleton)

**Work:** Add one named private `_simulate_deterministic_core` path beside the old path only long
enough to compare factor ledgers. Consume and validate the explicit one-row `IncidentSampleBatch`
from the sole DP-00C fixture authority, build its complete `IncidentStateBatch` once and serially,
and retain `parent_row_index` in every private component row. Then consume and validate its explicit `RodCatalog`, exact ordered strengths,
internal coating, exit transport, and detector image. Rods are never built or discovered from the
detector. The private deterministic core does not resample or construct a source; public
`simulate_ordered` remains the sampled entry until DP-06V, so no API cutover occurs here. Move the
accepted adaptive orchestration out of temporary proof/test helpers into this sole production
owner; delete temporary orchestration that is not an independent oracle.
Temporary milestone rejection of non-fixture domains occurs before allocating cells and is
explicitly marked `NOT_YET_MIGRATED`; DP-06C--DP-06H must remove those rejections before cutover.

Add the final script path initially as a Matplotlib-free `--numeric-only --json` CLI over the sole
fixture and private deterministic core. Time exactly `_simulate_deterministic_core` with `perf_counter` and
report `wall_time_s` plus process `peak_rss_bytes`; on the named Windows handoff machine, peak RSS
comes from `GetProcessMemoryInfo(GetCurrentProcess()).PeakWorkingSetSize` through a small private
`ctypes` helper. The JSON separates `scientific_summary` (API/tolerance/config hashes, array hashes,
component counts, and ledgers) from `operational_summary` (wall time, RSS, process, cache,
platform, and Python metadata). Numeric-only mode writes no file under any circumstances.

**Verify:** factor omission/duplication controls; exact classification of 120 rod identities and all
240 `(rod_id,root_label)` slots as empty/nonempty/failed support; integration of every supported
regular component; totals and moments; deterministic repeatability; alternate tile order;
equivalent-work wall time/peak RSS for the complete fixture using the exact numeric-only command;
and comparison with an independently converged latent oracle. Run the command twice and require
bitwise-identical numeric arrays and canonical `scientific_summary`; timing, RSS, process/platform,
and cache metadata are excluded from equality. Tag/cache repeatability is tested after DP-06I/P.
The old Monte Carlo mean may be used only as low-authority disposable evidence.

**Acceptance:** all deterministic gates pass; the complete fixture meets `<=15 min` and
`<=1.5 GiB` on the named handoff machine before deletion is authorized; and every difference from
bilinear/Monte Carlo output is classified at the intended first divergent stage.

**Dependencies:** DP-04C.

### DP-05V1: Migrate the first trace-producer group

**Files (5):**

- `src/rasim_next/proof/traces.py`
- `src/rasim_next/proof/core.py`
- `src/rasim_next/geometry/proof.py`
- `src/rasim_next/reciprocal/proof.py`
- `docs/TRACE_SCHEMA.md`

**Work:** Activate `RESERVED_PUSHFORWARD_TRACE_SCHEMA_VERSION` as
`CURRENT_TRACE_SCHEMA_VERSION` exactly once, replace hard-coded current-version literals in this
producer group with the shared constant, call the explicit v2 tolerance loader, and update the live
trace authority in the same task. Preserve separately named immutable reference-pack versions and
add explicit stale/mixed-version rejection.

**Verify:** core, geometry, and reciprocal tests/proofs; stale-current-trace rejection; immutable
reference v4 acceptance; imports and formatting.

**Acceptance:** these producers emit one coherent current trace version without relabelling
historical artifacts.

**Dependencies:** DP-05.

### DP-05V2: Migrate the remaining trace producers

**Files (5):**

- `src/rasim_next/ordered/proof.py`
- `src/rasim_next/stacking/proof.py`
- `src/rasim_next/proof/reference.py`
- `tests/test_ordered_reflectivity.py`
- `tests/test_stacking_transition.py`

**Work:** Migrate the remaining current-trace producers/consumers to the shared version and explicit
v2 tolerance loader while keeping immutable reference manifests historical. Do not change their
scientific oracles.

**Verify:** ordered, stacking, and references proofs; stale/mixed-version negative tests; imports
and formatting.

**Acceptance:** all six existing proof commands use the coherent current trace contract and remain
green before the new proof command is registered.

**Dependencies:** DP-05V1.

### DP-05V3: Remove the ambiguous tolerance-loader alias

**Files (3):**

- `src/rasim_next/proof/tolerances.py`
- `tests/test_core_coordinates.py`
- `docs/VALIDATION.md`

**Work:** Migrate the final test/validation caller to an explicit named loader, then remove the
temporary no-argument `load_stage_tolerances()` alias. Retain the named v1 loader and immutable v1
artifact as historical evidence; v2 is the explicit current authority. Do not remove or relabel
historical trace/reference evidence.

**Verify:** exact v1/v2 hashes, numerical identity of every inherited v1 stage, explicit current-v2
selection, rejection of an omitted/unknown/mixed version, core-coordinate tests, all six existing
proof commands, documentation checks, imports, and formatting.

**Acceptance:** no production, proof, test, or live-document caller relies on an ambiguous tolerance
loader, and every intermediate/current version claim is explicit.

**Dependencies:** DP-05V2.

### DP-05P: Add the compact deterministic-pushforward proof dispatcher

**Files (3):**

- `src/rasim_next/pipeline/proof.py` (new)
- `src/rasim_next/proof/__main__.py`
- `tests/test_integration.py`

**Work:** Register `deterministic-pushforward` only now that production exists. Compare production
with a tiny analytic/direct-enumeration latent-cell-to-pixel oracle; prove sphere, exit, and
detector ledgers, exact status/edge ownership, and named first divergence from bilinear deposition.
Keep full images/full-fixture sweeps out of this compact proof. The dead old `integration`
dispatcher remains deleted.

**Verify:** dispatcher help/registry, JSON schema/version, pass case, required mutations, the exact
new proof command, and all six existing commands.

**Acceptance:** seven real proof commands are green; the new command independently protects the
deterministic observable rather than merely invoking an integration test.

**Dependencies:** DP-05V3.

### DP-06I: Add the internal peak-mosaic representative and immutable cache

**Files (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`

**Work:** Implement the closed-support representative solver and the explicit immutable internal
cache. Add and validate the canonical `phase_geometry_id` fingerprint for the current single-phase
input; DP-06G later embeds the same contract in `PhasePushforwardInput`. Predeclare the discriminated
continuous/atom row schema, including measure kind, measure dimension, optional atom probability/
line density, and a distinct lattice-neutral `intensity_status` that is never overloaded into root,
exit, or detector status. Enumerate/bound all interior, seam, and tangent-boundary candidates for the accepted pure Gaussian; compare
stable log density and apply the complete dimension-aware order. Keep all degenerate rods/orientations in the
physical sum, but emit exactly one metadata-only, mass-field-free
`(L_peak,m,intersection_branch_id)` display row per nonempty
`(incident_state_id,phase_geometry_id,family_id,intersection_branch_id)` closure using the declared continuous
tie-break. For the universal exact `(h,k)=(0,0)` specular rod, remove the direct `u=0` root before
forming its branch-0 closure independently of optional `m`. Implement only the
canonical SHA256 internal key and one-batch replacement semantics here; projected fields remain
absent until DP-06P.

**Verify:** analytic/global mode cases for branches 0/1/2; rejection of an unsupported mixture; a
far-tail probability-underflow mutation; a regular branch-1/2 maximum attained only at a certified
`D=0` tangent closure; branches 1/2 sharing boundary geometry but retaining identity; a branch-0
maximum at signed-`a=0` direct-root coalescence without a physical tangent chart; exact `m`; exact continuous
`L_peak`; periodic-beta tie order independent of cells; multiple off-peak `L` values ignored;
duplicate same-`m` rods retained in mass but collapsed only in tag metadata; exact component ID/
canonical order; mutation of every internal key field; full replacement on a key miss; removal of
the universal specular direct root; distinct phase geometry with the same reciprocal `family_id`
producing distinct tag identity; `SPECULAR_ROOT_COALESCENCE` root status coexisting with
`SPECULAR_INTENSITY_EXCLUDED` intensity status; bounded storage; and proof that enabling internal
tags leaves both numeric images bitwise unchanged.

**Acceptance:** every nonempty closed degenerate set has exactly one internal row maximizing only
the declared mosaic density; specular branch-0 rows use only the non-direct root, evaluate no
physical mass, and carry `intensity_status=SPECULAR_INTENSITY_EXCLUDED`; no adaptive node is tagged
merely because it was sampled; cache storage is one internal batch and has no hidden mutable/global
owner.

**Dependencies:** DP-05P.

### DP-06P: Add the exit/detector projection cache

**Files (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_integration.py`

**Work:** Project every internal representative through the exact DP-03/DP-04 seams. Retain rows
with invalid exit/detector status and `None` projected fields. Add the canonical projection
fingerprint, one-batch replacement semantics, and projection-only reuse of an exact internal
batch. Include the top-interface policy, only exit/optics/support-edge/detector tolerances and
versions, full representative payload, actual intersection, transforms, optics, detector geometry,
and support-edge convention. Cache metadata remains outside `scientific_summary`.

**Verify:** valid ray/pixel round trip; invalid row retention; mutation of every projection-key
field; a detector-tolerance mutation producing a projection miss with exact internal reuse;
projection-only invalidation; internal miss invalidating both batches; no partial/stale hit;
two-batch memory bound; bitwise numeric/tag repeatability; and invariant internal winners under
detector/optics-only changes.

**Acceptance:** the explicit returned cache contains at most one internal and one projection batch;
hit/miss status cannot change arrays, IDs, order, or representative membership.

**Dependencies:** DP-06I.

### Checkpoint B0: Freeze the accepted-domain inventory

**Files (1):**

- `docs/VALIDATION.md`

**Work:** Compare the exact post-BKI public contracts, current tests, and live callers with the deterministic
milestone and persist the reviewed inventory. At minimum record incident/source/wavelength batch
behavior, `finite_rectangle.v1` and `unbounded_plane.v1` support/footprint statuses, smooth
Gaussian/Lorentzian mixtures, zero-width probability atoms, current single-phase population
weight, general-cell rod/family identity, phase/parent incoherent composition, and ordered/stacking
strength substitution. Each row records
whether it is a currently accepted public domain or a future dovetail seam, factor/measure owner,
implementation task, independent proof, and disposition. Current `simulate_ordered` accepts the
batch/support/mixture/atom groups and one positive phase-population scalar; its exact-hex
intersection requirement means general-family deterministic support is new DP-06F work, while
stacking and multiphase/multiparent composition are first DP-06G/DP-06H
dovetail implementations rather than preserved runtime parity. A one-ray fixture does not
authorize removal of any current domain. Any unlisted current domain blocks deletion until this
plan receives a new <=5-file task or an explicit human-approved deprecation/migration.

**Verify:** exact public signature/import scan, current compact tests, all seven proofs, and a
reviewed one-to-one mapping from every current-domain row to a parity task/proof and every dovetail
row to a first-implementation task/proof. Run `tools/check_docs.py`.

**Acceptance:** the durable table is complete, source-cited, classified, and has no unowned current
domain or falsely labelled parity row.

**Dependencies:** DP-06P.

### DP-06C: Preserve incident/source/wavelength batch parity

**Files (5):**

- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/proof.py`
- `tests/test_integration.py`
- `tests/test_mosaic_ewald.py`

**Work:** Accept every recorded incident row and wavelength identity. Validate the complete parent
source batch, build the complete incident table once and serially, and tile only downstream work.
Private rows retain in-range `parent_row_index`; reassembly scatters through those indices before
canonical component order and never sorts/rejoins by `incident_state_id`, hashes a public slice, or
regenerates a source row. Integrate each row's continuous measure independently, apply its source
weight once, retain stable incident IDs/tags and every parent proof/revision field unchanged, and
reduce independent rows as intensities in canonical order. Preserve both accepted sample
support models: exact finite-rectangle footprint/status ownership and unbounded-plane unit
footprint with absent dimensions. Source characterization remains at the existing boundary;
production never resamples inside the pipeline. Entrance-invalid or zero-source-weight rows emit no
coating/tag; invalid positive source probability remains only in the dimensionless incident-status
ledger. Preserve the current single-phase positive population factor, including non-unit values,
once here; multi-phase/multi-parent composition remains DP-06H's new dovetail path.

**Verify:** two-row/two-wavelength direct enumeration, nonmonotonic IDs, row-order permutation,
reversed tile completion, invalid/duplicate/out-of-range parent indices, no ID-sorted merge/public-
slice hash/worker source regeneration, exact parent provenance/revision pass-through, zero source
weight, source-factor mutation, finite-rectangle inside/edge/outside rays, unbounded-plane absent-
dimension behavior, invalid/zero-weight no-tag behavior, separate source-probability versus area-
ledger units, non-unit/invalid single-phase population factors, footprint/status-factor mutations,
per-row ledgers/tags, and compact dispatcher parity.

**Acceptance:** every accepted batch row contributes once and only once; the one-row fixture is the
exact special case and temporary batch rejection is removed.

**Dependencies:** Checkpoint B0.

### DP-06D: Preserve smooth Gaussian/Lorentzian-mixture parity

**Files (5):**

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Integrate the complete folded positive-tilt continuous mixture and globally maximize its
complete log density for representative tags. Preserve the signed-density trace separately.
Component probability ownership remains in `WrappedMosaicParameters`; no component is renormalized
or dropped because its active width is small.

**Verify:** pure-Gaussian limit, pure-Lorentzian positive-width limit, mixed direct quadrature,
density-mode shift, far-tail underflow, mixture-weight mutations, and normalization/ledger closure.

**Acceptance:** every accepted smooth mixture is deterministic and certified; the temporary
nonzero-Lorentzian rejection is removed.

**Dependencies:** DP-06C.

### DP-06E1: Integrate the zero-width atom as a one-dimensional beta pushforward

**Files (5):**

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Integrate the accepted singular measure
`P_atom*delta_0(d alpha)*d beta/(2*pi)` on the periodic beta circle. Add no `d alpha`, keep it
distinct from continuous endpoint values, and apply atom probability/line density once. Certify all
`D>0/D=0/D<0` beta components. Regular points use the ordinary Ewald coarea; a transverse beta
tangent uses only the fused one-dimensional continuation `2*|ki|/abs(dD/d beta)`. A derivative
that cannot be certified nonzero is `DEGENERATE_ATOM_TANGENCY`/unresolved and stops. Never use the
two-dimensional `(D,eta)` chart for an atom.

**Verify:** pure-atom beta variation, atom-plus-continuous direct quadrature, periodic seam, no
continuous-endpoint double count, regular root enumeration, supported/unsupported branches,
transverse beta tangent, exact degenerate-tangent failure, weight limits, and omitted/doubled atom,
beta-collapse, and wrong-chart mutations.

**Acceptance:** the accepted atom has exact one-dimensional probability/mass ownership and a
certified regular/tangent pushforward; no beta-collapsed shortcut remains.

**Dependencies:** DP-06D.

### DP-06E2: Integrate atom orchestration, tags, and cache semantics

**Files (5):**

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/pipeline/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Remove the temporary active-atom rejection and compose the DP-06E1 atom line measure once
through coating, exit, detector, arrays, and every declared ledger axis. Extend tag selection with
the dimensionally valid small-ball order: a supported dimension-one atom mode outranks a
dimension-two continuous mode; compare densities only within the same dimension. The uniform atom
chooses the smallest *supported* beta under the canonical periodic tie order, not unconditionally
zero. Emit no row for a zero-probability or unsupported branch. Record measure kind/dimension,
atom probability, and line density in rows and internal keys.

**Verify:** pure atom and mixed end-to-end oracles, continuous/atom/parent ledger reductions,
atom-supported versus unsupported tags, beta-zero unsupported tie case, transverse tangent winner,
cache invalidation by atom parameters, and mutations comparing atom mass numerically with
continuous density or omitting/duplicating an atom contribution.

**Acceptance:** atom arrays, summaries, ledgers, tags, and cache semantics are deterministic and
certified, and the temporary active-atom rejection is removed.

**Dependencies:** DP-06E1.

### DP-06F: Add the first deterministic general-cell family path

**Files (5):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/reciprocal/rods.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`

**Work:** Keep exact `family_id` as the universal cache/tag identity. Hexagonal rows preserve exact
integer `m`; general-cell rows carry `m=None`, never a fabricated integer. Retain every explicit
rod and `rod_catalog_revision`; family metadata collapses only metadata tag display rows and never
physical mass.

**Verify:** one compact nonhexagonal catalog, its exact `(h,k)=(0,0)` direct-root exclusion with
`m=None` and `SPECULAR_INTENSITY_EXCLUDED`, same-family duplicate rods, catalog-order/content
mutations, detector-pose independence, optional-`m` serialization, and family/root ledger sums.

**Acceptance:** the planned general-family dovetail seam has one deterministic identity contract and
the temporary nonhexagonal rejection is removed without claiming it was current runtime parity.

**Dependencies:** DP-06E2.

### DP-06G: Add the first deterministic ordered/stacking strength-provider seam

**Files (5):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/ordered/amplitudes.py`
- `src/rasim_next/stacking/finite_intensity.py`
- `tests/test_ordered_reflectivity.py`
- `tests/test_stacking_transition.py`

**Work:** Freeze one narrow immutable point-and-enclosure strength-provider contract implemented by
the ordered and stacking equation owners. Both return area-valued strength with explicit model/
normalization metadata and own the electron-to-area conversion once. Complete the private immutable
`PhasePushforwardInput` representation here so each phase owns its material, explicit catalog/
fingerprint, mosaic, validated canonical `phase_geometry_id`, population/parent identity, and
provider; DP-06V alone promotes it publicly. Keep `RodCatalog.family_id` crystal/family-only. No
provider imports coating, render, or pipeline code.

**Verify:** ordered identity, stacking finite-intensity cases, enclosure coverage, provider
substitution, unit/normalization mismatch rejection, and omitted/doubled-area mutations.

**Acceptance:** both accepted strength models satisfy the same seam without duplicated coating
physics or a backend registry.

**Dependencies:** DP-06F.

### DP-06H: Add the first deterministic phase/parent incoherent composition

**Files (5):**

- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/pipeline/proof.py`
- `tests/test_integration.py`
- `tests/test_ordered_reflectivity.py`
- `tests/test_stacking_transition.py`

**Work:** Compose the declared phase/parent contributions as intensities, apply each population weight
once, retain phase/parent/provider IDs in physical ledgers, and consume either DP-06G strength
provider. Tags remain geometric: parent/population/provider-only changes neither duplicate nor move
a tag, while a distinct phase geometry has a distinct `phase_geometry_id` even when reciprocal
`family_id` is equal. Share one
coating/pushforward implementation; do not add per-model runtime copies.

**Verify:** two-phase and two-parent direct sums, zero/renormalized population rejection as defined
by the declared contracts, ordered/stacking substitution, factor mutations, canonical reduction,
parent/provider-only changes leaving shared-geometry tag IDs/positions unchanged, distinct phase
geometry producing distinct `phase_geometry_id` with unchanged reciprocal family semantics, and
compact dispatcher coverage.

**Acceptance:** all accepted phase/parent and strength-provider inventory rows have deterministic
implementations or a separately approved migration; no temporary milestone rejection remains on
the public cutover path, and the handoff does not mislabel future dovetail support as preserved
post-BKI runtime parity.

**Dependencies:** DP-06G.

### Checkpoint B: accept implementation and authorize public API activation

- DP-01 through DP-06H pass focused and full tests; all seven proof commands are green.
- Internal coating, external outgoing rays, detector mass, and component tags each have independent
  proof and mutation sensitivity.
- Every Checkpoint B0 current-domain row has parity or a separately reviewed deprecation/migration,
  and every required dovetail row has its accepted first implementation/proof.
- The complete 120-rod/240-root-slot run meets `<=15 min` and `<=1.5 GiB` before any old runtime
  file is deleted.
- A human reviews canonical scientific summaries and enclosure evidence before any old runtime is
  removed. Runtime deletion is still withheld until all live instructions are synchronized at
  Checkpoint D.

### DP-06U: Factor the sampled comparator before public activation

**Files (3):**

- `src/rasim_next/pipeline/simulate.py`
- `scripts/generate_bi2se3_detector_image.py`
- `tests/test_integration.py`

**Work:** Factor the unchanged sampled implementation behind private
`_simulate_sampled_comparison`. For this one transition task, the old public `simulate_ordered`
delegates to that private function with its existing signature and exact behavior, while the old
image script calls the private comparator explicitly. Do not change `CONTRACT_API_VERSION`, public
imports, serialization, accepted input, output, RNG consumption, or numerical results.

**Verify:** exact old-public/private-comparator equivalence for values, arrays, statuses, ledgers,
trace, and seeded RNG behavior; the old image script remains runnable; integration tests, full
tests, all seven proofs, imports, and formatting.

**Acceptance:** the sampled path has one private implementation and every retained sampled caller
is isolated before the atomic public activation; the public contract and behavior are unchanged.

**Dependencies:** Checkpoint B.

### DP-06V: Activate the complete final public API exactly once

**Files (5):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/geometry/__init__.py`
- `tests/test_integration.py`
- `docs/CONTRACTS.md`

**Work:** Promote the accepted private deterministic types only now that their complete final
superset exists: `PhasePushforwardInput` with validated `phase_geometry_id`, explicit
`PushforwardCache` input/`next_cache` output, coating/detector results, typed component-tag rows and
separate statuses, ledgers, and certified errors. Advance `CONTRACT_API_VERSION` exactly once and
change public `simulate_ordered` to the final deterministic signature. Keep the sampled comparison
path as private `_simulate_sampled_comparison` only until DP-07A deletes it; DP-06U already migrated
the retained old image script to that private name so the intermediate tree remains runnable.
Remove `EventTransportResult` and `transport_scattering_events` from the public `geometry` imports
and `__all__` in this same versioned activation; their implementations remain private transition
residue until DP-08. Public
`simulate_ordered` validates the final contract and orchestrates the existing
`_simulate_deterministic_core`; it is not a duplicate physics implementation or compatibility
facade. Update the live contract authority in the same task. Later tasks may implement no
structural public-schema/signature change without a separately approved version increment.

**Verify:** old-version and partial-superset rejection; exact final signature/field/unit/frame/
measure/status serialization; public import surface; cache in/out contract; phase-geometry/family
identity; all current-domain and dovetail compact cases; full tests, all seven proofs,
`tools/check_docs.py`, absence of the two obsolete public geometry exports, imports, and formatting.

**Acceptance:** one recorded API increment exposes the complete accepted deterministic contract;
no earlier task changed the public version, no final field is deferred, and the remaining sampled
code is private comparison residue scheduled for DP-07A deletion.

**Dependencies:** DP-06U.

### DP-06V2: Migrate proof and numeric callers to the activated public API

**Files (3):**

- `src/rasim_next/pipeline/proof.py`
- `scripts/generate_bi2se3_pushforward_images.py`
- `tests/test_integration.py`

**Work:** Switch the seventh proof and Matplotlib-free numeric CLI from direct private-core calls to
the activated public `simulate_ordered` contract. During this task only, compare public output with
the same `_simulate_deterministic_core` implementation to prove exact arrays, scientific summary,
tag/cache rows, and ledgers. Remove every direct private-core import from proof/test/script callers
before acceptance; afterward only public `simulate_ordered` orchestrates that private implementation.

**Verify:** integration tests, all seven proofs, numeric-only twice, exact public/private equivalence,
no direct private-core caller outside `pipeline/simulate.py`, zero created files, imports, and
formatting.

**Acceptance:** every pre-cutover scientific/proof/resource gate exercises the final public API;
the private core remains one internal implementation detail rather than an alternate entry point.

**Dependencies:** DP-06V.

### DP-06A: Synchronize live architecture and result contracts before cutover

**Files (5):**

- `docs/ARCHITECTURE.md`
- `docs/CONTRACTS.md`
- `docs/DOVETAIL_MATRIX.md`
- `docs/RESULT_MEASURE.md`
- `docs/TRACE_SCHEMA.md`

**Work:** Replace sampled-candidate/selection/deposition authority with the accepted deterministic
latent indicator measure, result interfaces, versioned factor/rejection ledgers, certified-error contract,
single-fixture authority, and component-tag cache. Mark the old runtime as temporary comparison
code scheduled for immediate DP-07 deletion. Remove obsolete trace stages and add stable
coating/pushforward/tag stages without renumbering unrelated stages.

**Verify:** `tools/check_docs.py`, contract/import scans, and comparison of every documented field
to the accepted DP-05/DP-06 result types.

**Acceptance:** these five live authorities describe the accepted implementation before the old
private comparison runtime is deleted and none directs a new consumer to sampled-event APIs. DP-06J--DP-06M
and Checkpoint D, not this five-file task, own the repository-wide synchronization claim.

**Dependencies:** DP-06V2.

### DP-06B: Record the scientific correction and validation rules before cutover

**Files (5):**

- `docs/SCOPE_AND_PHASES.md`
- `docs/PHYSICS_LEDGER.md`
- `docs/VALIDATION.md`
- `docs/ERROR_INJECTION.md`
- `docs/DECISIONS.md`

**Work:** Replace D013 and relevant `PHY-REC`/`PHY-MEA` rows. Classify deterministic coating as
`NEW`, removal of inverse-CDF selection as `CORRECTED`, and certified latent pixel-indicator
integration as `CORRECTED` first at `measurement.detector_pixel_mass`. Record the fused tangent
Jacobian, topology certificate, two-image diagnostic, and peak-mosaic component-tag requirements
without treating convergence or images as proof.

**Verify:** proof hierarchy, tolerances, mutations, result measure, universal exact-specular
limitation, and
`tools/check_docs.py` agree with DP-00A through DP-06.

**Acceptance:** scientific and error-injection authorities are current before deletion; no
intermediate cutover tree relies on stale sampled-event documentation.

**Dependencies:** DP-06A.

### DP-06J: Reconcile downstream strategy and fitting plans before cutover

**Files (5):**

- `docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`
- `tasks/parallel_simulation_geometry_fitting_plan.md`
- `tasks/mosaic_distribution_fitting_plan.md`
- `tasks/OVERNIGHT_RUNBOOK.md`
- `tasks/deterministic_ewald_pushforward_todo.md`

**Work:** Preserve valuable derivations while replacing sampled coating/deposition directions with
the accepted deterministic measure, explicit catalog, strength-provider seam, tag cache, and
invalidation rules. Remove executable directions to use outgoing-event RNG/candidate pools. Keep
source sampling only at its declared upstream boundary.

**Verify:** `tools/check_docs.py`, task/link scan, and field/API comparison with production.

**Acceptance:** these live plans cannot direct a concurrent or downstream worker back to sampled
events and contain no link scheduled to break later.

**Dependencies:** DP-06B.

### DP-06K: Reconcile executable prompts and BKI routing before cutover

**Files (5):**

- `tasks/prompts/integration.md`
- `tasks/prompts/mosaic_ewald.md`
- `tasks/README.md`
- `tasks/plan.md`
- `tasks/todo.md`

**Work:** Rewrite obsolete prompts as deterministic routing/history stubs or retire them only when
inbound-link checks permit. Confirm DP-00R's deterministic-only successor and remove any remaining
sampled-event instruction. Record the landed BKI-16/BKI-17 disposition. If BKI obligations or inbound links
remain, retain and rewrite `tasks/plan.md`/`tasks/todo.md`; never delete them into an orphaned tree.

**Verify:** exact inbound-link scan before/after, task coverage, BKI obligation map,
`tools/check_docs.py`, and the repository live-instruction scan.

**Acceptance:** no executable prompt or BKI route can recreate the old framework, and no BKI-17
obligation is lost.

**Dependencies:** DP-06J.

### DP-06L: Reconcile numbered task and bootstrap contracts before cutover

**Files (5):**

- `tasks/00_bootstrap.md`
- `tasks/03_mosaic_ewald.md`
- `tasks/07_integration.md`
- `tasks/index.yaml`
- `tasks/06_parallel_review.md`

**Work:** Rewrite T03 and T07 as concise indexed historical/routing stubs; do not delete numbered
tasks because `tools/check_docs.py` requires indexed numbered files. Replace bootstrap/parallel
sampled-event dataflow with the deterministic successor and preserve unique historical facts in
validation/decision ledgers.

**Verify:** task-index coverage, exact inbound links, `tools/check_docs.py`, and imperative-symbol
scan.

**Acceptance:** every numbered task remains indexed, no live numbered instruction builds the old
runtime, and every intermediate tree stays documentation-green.

**Dependencies:** DP-06K.

### DP-06M: Reconcile root and performance/fitting instructions before cutover

**Files (4):**

- `README.md`
- `WORKTREE_LAUNCH.md`
- `docs/PERFORMANCE.md`
- `docs/FITTING_ROADMAP.md`

**Work:** Update all current root entry points and performance/fitting guidance to the accepted
deterministic command, seven-proof gate, explicit cache invalidation, and resource contract. Remove
stale task/branch links and sampled-runtime examples.

**Verify:** `tools/check_docs.py`, root/task/doc link scan, command/import scan, and exact proof-list
comparison.

**Acceptance:** all currently known live root instructions are synchronized before cutover. The
Checkpoint D scan blocks if the post-BKI audit found another live file; implementation must add a
separate <=5-file plan amendment rather than editing it opportunistically.

**Dependencies:** DP-06L.

### Checkpoint D: Authorize sole-runtime cutover

- All seven proof commands, the full compact suite, docs checker, imports, and formatting are green.
- Checkpoint B0 has no unresolved accepted-domain row.
- Repository-wide scans over root Markdown, `docs`, `tasks`, prompts, YAML, `src`, `tests`, and
  `scripts` find no live imperative sampled-event direction and no inbound link to a deletion
  target.
- A human approves cutover; otherwise the old runtime remains temporarily reachable and DP-07 does
  not start.

### DP-07A: Delete the private sampled comparison and selector/intersection runtime

**Files (5):**

- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/pipeline/intersections.py` (delete)
- `src/rasim_next/pipeline/selection.py` (delete)
- `scripts/generate_bi2se3_detector_image.py` (delete)
- `tests/test_integration.py`

**Work:** Public `simulate_ordered` is already deterministic after DP-06V. Delete the private
`_simulate_sampled_comparison` and
`_detector_complete_rods`, `_symmetric_hk_bounds`, and every detector-derived rod-discovery path
with `pipeline/intersections.py`. Consume only the explicit catalog. Remove selection RNG,
selected-event compaction,
candidate pools, draw counts, and imports of the two deleted pipeline modules. Delete the old image
script in the same cutover; remove the temporary DP-06U sampled-comparator/script equivalence
assertions from the integration test while retaining every distinct deterministic invariant.
DP-00C already migrated its permanent test/canonical fixture caller, so no checked-in caller is
left broken. Leave
`reciprocal/events.py` temporarily only for its still-live reciprocal proof/test consumers.

**Verify:** compile, full tests, all proof commands, and static scans for selector/intersection
modules and symbols.

**Acceptance:** the deterministic core is the sole simulation implementation; no sampled comparison,
runtime fallback, feature flag, compatibility alias, candidate pool, or outgoing-event RNG
selection remains; temporary `events.py` is unreachable from production.

**Dependencies:** Checkpoint D.

### DP-07B: Migrate reciprocal proof and delete the event builder

**Files (3):**

- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`
- `src/rasim_next/reciprocal/events.py` (delete)

**Work:** Replace the proof/test use of `build_scattering_events` with the accepted pointwise and
adaptive coating authorities, preserving only distinct root/coarea/tangent invariants. Delete the
event builder after its last import is gone.

**Verify:** focused mosaic/Ewald tests, full tests, all proof commands, and import/symbol scans.

**Acceptance:** no production, proof, test, or script imports `reciprocal/events.py`; the file is
deleted without losing an independent scientific invariant.

**Dependencies:** DP-07A.

### DP-08: Delete the sampled-event contract and repair its final consumers

**Files (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/proof/core.py`
- `tests/test_core_coordinates.py`

**Work:** Delete `ScatteringEventBatch` and its candidate-row, sampled-orientation, reciprocal
selection-weight, selected-mass, and point-contribution fields. DP-03 has already migrated geometry
transport proof/tests to the narrow deterministic internal-wave contract; DP-07B has removed the
event builder. DP-06V already removed the obsolete public geometry exports during the versioned API
activation. Delete the now-unused private event-shaped transport entry point and update the final
core proof and constructor tests atomically. Do not delete or collapse `IncidentSampleBatch` or
`IncidentStateBatch`: retain their intersection, SAMPLE direction, air/film wavevectors, complex
film-normal component, entrance amplitude, footprint, wavelength, polarization, source weight,
status/validity, IDs/model IDs, and source/sample/material/incident revision envelope for their
distinct proof and provenance consumers.

**Verify:** core contract/frame proof, full tests, all proof commands, a static scan for the removed
class/fields, and exact before/after comparison of the protected source/incident evidence.

**Acceptance:** no placeholder, deprecated property, or compatibility contract preserves sampled
event semantics, while every distinct source/incident proof and provenance field remains.

**Dependencies:** DP-07B.

### DP-09: Delete discrete orientation construction and point deposition

**Files (4):**

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/render/deposition.py` (delete)
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Delete `MosaicOrientationBatch`, fixed orientation-panel/quadrature construction,
alpha/azimuth cell-count APIs, bilinear deposition, Monte Carlo frequency tests, and point-event
clipping tests. Retain only unique continuous-density, root, cell-conservation, and
detector-pushforward invariants. DP-07B has already migrated reciprocal event tests; remove the
remaining fixed-orientation tests here.

**Verify:** full tests/proofs and a static scan for orientation-batch, old-script, and deposition
symbols.

**Acceptance:** the repository contains one continuous mosaic law and one detector integrator; no
checked-in script is left broken between tasks.

**Dependencies:** DP-08.

### DP-10: Add the locked replacement image tool and generate both views from one run

**Files (4):**

- `scripts/generate_bi2se3_pushforward_images.py`
- `tests/test_integration.py`
- `pyproject.toml`
- `uv.lock`

**Work:** Add a separate locked `image` dependency group containing Matplotlib; it is not part of
core or dev. Extend the already public-API DP-06V2 numeric-only script with an image mode that
consumes the same sole DP-00C frozen-fixture builder, imports Matplotlib only after CLI entry, and calls
`matplotlib.use("Agg", force=True)` before importing `pyplot`. Numeric-only mode creates no file.
Image mode requires a resolved external parent and a target `--output-dir` leaf that does not yet
exist, creates that leaf, refuses overwrite/rerun into it, computes one shared result, and writes
exactly:

```text
bi2se3_5deg_ewald_coating.png
bi2se3_5deg_detector_pushforward.png
```

The sphere PNG uses a fixed `2400 x 1200` pixel canvas with fixed panel axes and renders each exact
numeric `(mu,phi)` bin as one flat-colour front/back orthographic curvilinear patch. Fixed-`phi`
curves are adaptively and deterministically tessellated until their maximum projected chord error
is at most `0.1` output pixel; the tessellation changes display geometry only and never resamples,
interpolates, or subdivides the source-bin value. Antialiasing/interpolation are disabled. The
`phi=0` panel edge, periodic seam, and pole vertices are emitted once under the frozen ownership
rules. Exact zero bins use one explicit under/transparent color; positive bins share
`vmin=min(positive)` and `vmax=max(positive)` across both panels under one absolute log scale
labelled `angstrom^2/sr`. An all-zero image is a typed failure, not an invented log floor. The
panels include internal `ki` and metadata-only component-tag overlays.
The detector PNG uses the native `(3000,3000)` `[row,column]` array, `origin="upper"`, nearest/no
smoothing, the same explicit zero/log rule, absolute `angstrom^2/pixel`, the detector reference
coordinate, and the same detector-valid component-tag overlays. The decorated detector PNG canvas
is explicitly not the scientific 3000x3000 array contract and need not be one display pixel per
detector pixel; only its deterministic layout under the locked image environment is required.

Titles state `5 degree`, the wavelength, `2 degree Gaussian`, exact `(h,k)!=(0,0) intensity`
(`m!=0` for this hexagonal fixture), and `deterministic`. Branch-0 tags state that intensity is
excluded. The script prints Git/CIF/
tolerance hashes, dependency versions, numeric array hashes, configuration, convergence, and mass
ledger as JSON to stdout. It writes no numeric sidecar by default.

**Verify:** the permanent integration test exercises only the DP-00C fixture/result/filename
contract under core+dev and proves Matplotlib is not imported. Run two numeric simulations with
bitwise-identical arrays/canonical scientific summaries and zero created files. Then run the exact locked command
`uv run --frozen --group dev --group image python scripts/generate_bi2se3_pushforward_images.py
--output-dir <resolved-external-directory>` with external `MPLCONFIGDIR` and `MPLBACKEND=Agg`;
record the locked Matplotlib version; require the exact two filenames and no other directory entry;
decode the sphere PNG as exactly `2400x1200`; assert the detector source array is exactly
`(3000,3000)` in native orientation; match numeric-array/scientific-summary hashes to the prior
numeric-only run; type/range-check volatile performance fields; reject an existing output leaf;
confirm no repository-local output; and perform visual review. Do not skip the images, move
Matplotlib into core/dev, or compare PNG byte hashes across dependency versions.

**Acceptance:** a fresh external leaf contains exactly two PNGs; the coating PNG is exactly
2400x1200 and has the tessellation bound; the detector numeric source is exactly 3000x3000; neither
image is max-normalized, rotated, flipped, transposed, randomly speckled, overwritten, or used as
the scientific proof oracle.

**Dependencies:** DP-09.

### Checkpoint C: residue-free production path

- The deterministic coating/simulation/render import path contains no `np.random`, candidate
  selection, sampled outgoing event, orientation batch, or bilinear depositor.
- The upstream source module remains because it owns source characterization; the accepted fixture
  uses its one-row center semantics only and draws no random source coordinate.
- The old image script is gone, the replacement script is runnable, and the full suite plus all seven
  proof commands pass.

### DP-13: Retire the unnumbered superseded coating plan

**Files (1):**

- `tasks/continuous_ewald_coating_replacement_plan.md` (delete only after unique dirty work is ported)

**Work:** Delete only this unnumbered superseded plan after its owner has reconciled all unique
content, all retained material has moved to live authorities, and the exact repository-wide inbound-
link scan is empty. Here and in DP-13A, an `inbound link` means a resolvable Markdown link whose
target is the candidate file; code-span path mentions retained solely as explicit historical,
deletion, or mutation evidence are not links. Retain indexed T03/T07 routing/history stubs. If
ownership or a Markdown inbound link remains, stop; do not delete or overwrite it.

**Verify:** owner/hash disposition, before/after inbound-link scan, task-index coverage,
`tools/check_docs.py`, and seven proofs.

**Acceptance:** the unnumbered plan is gone without losing user work or breaking a link; numbered
task files remain indexed and green.

**Dependencies:** Checkpoint C.

### DP-13A: Conditionally retire the completed BKI plan/checklist

**Files (2, conditional):**

- `tasks/plan.md`
- `tasks/todo.md`

**Work:** Delete these files only if completed BKI-16/BKI-17 obligations are closed or migrated to a
retained owner/task, both files are cleanly reconciled with their historical evidence, and both
have zero retained Markdown inbound links under DP-13's definition. Otherwise retain the
deterministic routing/history stubs
written in DP-06K and record this task as a no-op. Never use deletion to manufacture completion.

**Verify:** BKI obligation ledger, owner/hash disposition, before/after inbound-link scan,
`tools/check_docs.py`, task-index coverage, and live-instruction scan.

**Acceptance:** completed BKI files are either safely retired or deliberately retained; no
outstanding follow-up, user edit, or local link is orphaned.

**Dependencies:** DP-13.

### DP-13B: Run the repository-wide residual-symbol and instruction scan

**Files:** none.

**Work:** DP-06L already repaired bootstrap before cutover. Run two final scans: (1) zero removed
modules/classes/functions in `src`, `tests`, and `scripts`; and (2) root Markdown, docs, tasks,
prompts, and YAML, whose only allowlisted occurrences are non-imperative historical/deletion/
mutation explanations in this accepted plan/checklist and validation/decision ledgers. Include
`ScatteringEventBatch`, `EventTransportResult`, `transport_scattering_events`, dynamic detector rod
helpers, selectors, orientation batches, and deposition symbols. An allowlist never permits an
import, command, successor edge, or implementation instruction.

**Verify:** exact scans, `tools/check_docs.py`, task-index coverage, all public package initializer
imports, full tests, and all seven proofs.

**Acceptance:** no live production/test/script/task instruction or package export can recreate the
sampled-event path; every residual text occurrence has an explicit historical allowlist reason.

**Dependencies:** DP-13A.

### DP-15: Final proof, cleanup, images, and handoff

**Files (4):**

- `docs/VALIDATION.md`
- `tasks/deterministic_ewald_pushforward_plan.md`
- `tasks/deterministic_ewald_pushforward_todo.md`
- `FILE_MANIFEST.json`

**Work:** Run the complete gate, render the two external PNGs into the Codex visualization or
another explicitly external directory and inspect them. Remove only external/untracked temporary
comparisons and proof artifacts here. If any tracked code/test/doc residue remains outside these
four files, stop and add or return to a named <=5-file cleanup task; DP-15 never edits an undeclared
tracked path. Audit retained tests, benchmark equivalent work, and report peak memory. Record the
mandatory compact durable proof/resource/handoff evidence in `docs/VALIDATION.md`, set this plan's
status to complete, and reconcile checklist boxes only to evidence that actually passed. After that
text and the complete file set are fixed, regenerate and review `FILE_MANIFEST.json`, then run
`verify_seed.py`. Before committing, run working-tree and cached diff checks; after the one coherent
commit, run `git diff --check "${env:PUSHFORWARD_BASE_SHA}..HEAD"` and require warning-free empty
status. Squash temporary task commits as required by project policy.

**Acceptance:** every final criterion below passes, the two PNGs are shown to the user, the branch
is clean after one commit, the plan/checklist no longer claim work is pending, and no generated
artifact is committed.

**Dependencies:** DP-13B.

## Testing strategy

### Permanent tests retained

- `tests/test_mosaic_ewald.py`: continuous mosaic normalization, analytic root/coarea identity,
  tangent/no-root behavior, and one compact sphere-conservation case.
- `tests/test_geometry_optics.py`: exit surface branch, tangential conservation, SAMPLE/LAB frame
  transform, actual ray origin, detector mapping, and existing optical invariants.
- `tests/test_integration.py`: one tiny independent cell-to-pixel oracle, full-ledger conservation,
  deterministic repeatability, explicit-catalog/batch/domain parity, and peak-mosaic cache behavior.
- `tests/test_core_coordinates.py`: revised contract/frame/index invariants only.
- `tests/test_ordered_reflectivity.py`: ordered strength point/enclosure and provider-unit ownership.
- `tests/test_stacking_transition.py`: stacking strength-provider substitution seam.

Do not add a new permanent test module. Temporary full-catalog sweeps, image arrays, broad
refinement grids, performance harnesses, and mutation variants are deleted before handoff.

### Frozen numerical acceptance

- Exact 120 rod identities, 240 ordered nonzero-`m` `(rod_id,root_label)` classifications,
  branch-0 geometry classification, statuses/family IDs/root labels, and valid/invalid tag-row
  counts: exact equality.
- Ewald residual and detector coordinates: approved frozen stage tolerances.
- Regular component mass and unit-direction first/second moments versus an independent oracle:
  `abs(error) <= 1.1284538007351267e-23 angstrom^2 + 1e-8*reference_scale`.
- Whenever an independent sphere/detector array oracle is available, compare with one global `L1`
  bound `1.1284538007351267e-23 angstrom^2 + 1e-8*sum(abs(reference))`; never grant that total-mass
  scale separately to every bin or pixel.
- Sphere, pre-optics exit, and post-optics detector ledger closures satisfy the same frozen
  absolute-plus-relative mass rule. Every reported topology/chart/quadrature/mapping/indicator
  upper bound must cover observed oracle error and itself be below the corresponding frozen
  tolerance; embedded-pair differences are never called bounds.
- Two-finest-refinement normalized `L1` change: `<=1e-3` for both sphere and detector arrays as a
  convergence gate only, never as an accuracy oracle.
- Detector centroid change: `<=0.05 px` between the two finest refinements.
- Identical inputs and canonical tile order: bitwise identical numeric arrays and cache rows.
- Canonical `scientific_summary` equality excludes wall time, RSS, process/platform, dependency
  runtime details, and cache hit/miss metadata; those operational fields are type/range checked.
- Alternate tile/worker order: identities exact and floating results within the declared reduction
  tolerance; canonical production reduction restores repeatable output.
- Multi-row/wavelength, smooth-mixture, zero-tilt atom, general-family, phase/parent, and
  ordered/stacking provider compact oracles pass before runtime deletion.
- Full 120-rod/240-root-slot fixture target on the handoff machine: both numeric arrays and
  component tags complete in `<=15 min` with peak RSS `<=1.5 GiB`. This passes before DP-07A and at
  handoff; failure triggers design review, not a tolerance waiver.

### Required error injections

- omit or double `J_ewald`, folded-density mass, or the electron-to-area conversion;
- add an erroneous `sin(alpha)` or overwrite the signed-density trace with the folded density;
- evaluate the tangent kernel as `0*inf`, omit/double the chart factor, use a singular chart, or apply `D=s^2` at a
  non-transverse tangent;
- hide a narrow disconnected support/fold/status island between sample nodes;
- assign a sphere-bin crossing or pixel-edge tie from quadrature-node ownership alone or apportion
  physical mass by mapped polygon area instead of certified latent indicators;
- swap roots 1 and 2;
- collapse same-`m` rods;
- drop one preimage at a fold;
- use the wrong exit square-root branch/surface normal, send negative-outward mass through the top
  interface, or apply same-surface attenuation to it;
- skip SAMPLE-to-LAB transformation or use the wrong ray origin;
- multiply pixel solid angle or a sphere/plane Jacobian into raw mass;
- swap detector row/column or rotate the image;
- discard outside/exit-rejected mass;
- let component-tag rows contribute mass;
- choose a tag by integration-node/local-optimizer order, create a false underflow density tie,
  omit a tangent-closure maximum, or emit other than one row for a nonempty family/root closure;
- let detector validity change the internal representative; or
- let detector geometry/refinement change the frozen sphere result;
- retain the direct `u=0` root of exact `(h,k)=(0,0)` in any lattice family or evaluate any
  specular branch-0 physical mass;
- silently drop a source/wavelength row, zero-width atom, nonzero Lorentzian mixture, phase/parent,
  or stacking provider; compute hexagonal `m` for a nonhexagonal catalog; discover rods from the
  detector; mix v1/v2 tolerance or old/new trace schemas; or
- apply the total-mass tolerance independently to every pixel, compare volatile summaries as
  scientific state, or overwrite an existing image output.

Each mutation must fail first at its owning stage. Only a minimal representative subset remains in
permanent tests.

## Commands

Use external uv, Python-bytecode, pytest, and image-output locations as required by repository
policy. After every code task, run the exact focused bundle assigned below, compile/import the
touched package, and run Ruff on touched Python; after every deletion/version task, run the full
suite and every then-existing proof. Documentation-only tasks run `tools/check_docs.py` and their
exact scans. A checkpoint runs only its declared then-existing matrix: Checkpoint 0/A cannot invoke
the seventh proof or image script, B/D run all seven proofs and numeric repeatability after DP-05P,
C adds the image command, DP-00R performs the one reconciled-planning manifest/seed refresh, and
only DP-15 regenerates the final implementation manifest and runs the final seed gate. The exact
external parent paths must already exist; they are resolved, type-checked, rejected
if they or any ancestor are reparse points, and write-probed with collision-safe task-owned names.
A fresh task-owned scratch leaf is then created without overwrite before running:

```powershell
$repoRoot = (Get-Item -LiteralPath (Resolve-Path -LiteralPath ".").Path).FullName.TrimEnd("\", "/")
$repoRootCase = $repoRoot.ToUpperInvariant()

function Resolve-PlainExternalDirectory {
    param([string]$Candidate, [string]$Label)

    if (-not (Test-Path -LiteralPath $Candidate -PathType Container)) {
        throw "$Label must be a pre-existing directory"
    }
    $item = Get-Item -LiteralPath (Resolve-Path -LiteralPath $Candidate).Path
    if (-not $item.PSIsContainer) {
        throw "$Label must be a directory"
    }
    $full = $item.FullName.TrimEnd("\", "/")
    $fullCase = $full.ToUpperInvariant()
    $driveRoot = $item.PSDrive.Root.TrimEnd("\", "/")
    if (
        $fullCase -eq $repoRootCase -or
        $fullCase.StartsWith($repoRootCase + "\") -or
        $full -ieq $driveRoot
    ) {
        throw "$Label must be outside the repository and may not be a filesystem root"
    }
    $cursor = $item
    while ($null -ne $cursor) {
        if ($cursor.Attributes -match "ReparsePoint") {
            throw "$Label may not use a reparse-point path or ancestor"
        }
        $cursor = $cursor.Parent
    }
    $probe = Join-Path $full (".ra-write-probe-" + ((New-Guid).Guid -replace "-", ""))
    $probeCreated = $false
    try {
        New-Item -ItemType File -Path $probe -ErrorAction Stop | Out-Null
        $probeCreated = $true
    }
    finally {
        if ($probeCreated) {
            Remove-Item -LiteralPath $probe -Force
        }
    }
    return $full
}

$scratchParent = Resolve-PlainExternalDirectory "<resolved writable external scratch parent>" "scratch parent"
$pushforwardScratch = Join-Path $scratchParent ("ra-pushforward-" + ((New-Guid).Guid -replace "-", ""))
New-Item -ItemType Directory -Path $pushforwardScratch -ErrorAction Stop | Out-Null

$env:TEMP = $pushforwardScratch
$env:TMP = $pushforwardScratch
$env:UV_CACHE_DIR = Join-Path $pushforwardScratch "uv-cache"
$env:PYTHONPYCACHEPREFIX = Join-Path $pushforwardScratch "pycache"
$pytestBase = Join-Path $pushforwardScratch "pytest"
$pytestCache = Join-Path $pushforwardScratch "pytest-cache"
$env:PYTEST_ADDOPTS = "--basetemp=`"$pytestBase`" -o cache_dir=`"$pytestCache`""
$env:RUFF_CACHE_DIR = Join-Path $pushforwardScratch "ruff-cache"
$env:MPLCONFIGDIR = Join-Path $pushforwardScratch "matplotlib"
$env:MPLBACKEND = "Agg"
$env:PUSHFORWARD_BASE_SHA = "<approved reconciled post-merge main SHA>"
$imageParent = Resolve-PlainExternalDirectory "<resolved writable external image parent>" "image parent"
$env:PUSHFORWARD_IMAGE_DIR = Join-Path $imageParent ("bi2se3-pushforward-" + ((New-Guid).Guid -replace "-", ""))
if (Test-Path -LiteralPath $env:PUSHFORWARD_IMAGE_DIR) {
    throw "fresh image output leaf already exists"
}

uv run --frozen --group dev python -m compileall -q src
uv run --frozen --group dev ruff check src tests scripts
uv run --frozen --group dev ruff format --check src tests scripts
uv run --frozen --group dev pytest -q
uv run --frozen --group dev python -m rasim_next.proof core --json
uv run --frozen --group dev python -m rasim_next.proof geometry-optics --json
uv run --frozen --group dev python -m rasim_next.proof mosaic-ewald --json
uv run --frozen --group dev python -m rasim_next.proof ordered-reflectivity --json
uv run --frozen --group dev python -m rasim_next.proof references --json
uv run --frozen --group dev python -m rasim_next.proof stacking-transition --json
uv run --frozen --group dev python -m rasim_next.proof deterministic-pushforward --json
uv run --frozen --group dev python tools/check_docs.py
uv run --frozen --group dev python scripts/generate_bi2se3_pushforward_images.py --numeric-only --json
uv run --frozen --group dev python scripts/generate_bi2se3_pushforward_images.py --numeric-only --json
uv run --frozen --group dev --group image python scripts/generate_bi2se3_pushforward_images.py --output-dir $env:PUSHFORWARD_IMAGE_DIR
# MANIFEST: DP-00R after its planning refresh, or DP-15 after final text/file-set freeze; no other task.
# The tracked set is sorted, FILE_MANIFEST.json excludes itself, and hashes cover working-tree bytes.
uv run --frozen --group dev python -c 'import hashlib, json, subprocess; from pathlib import Path; root = Path.cwd(); paths = sorted(raw.decode("utf-8") for raw in subprocess.check_output(["git", "-c", "core.quotepath=false", "ls-files", "-z"]).split(b"\0") if raw and raw != b"FILE_MANIFEST.json"); files = [{"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(), "size_bytes": (root / path).stat().st_size} for path in paths]; (root / "FILE_MANIFEST.json").write_text(json.dumps({"file_count": len(files), "files": files}, indent=2) + "\n", encoding="utf-8", newline="\n")'
# SEED: only after MANIFEST and an explicit manifest diff review.
uv run --frozen --group dev python scripts/verify_seed.py
git diff --check
git diff --cached --check
# after the coherent commit:
git diff --check "${env:PUSHFORWARD_BASE_SHA}..HEAD"
git status --short
```

The compact task matrix below uses these exact bundles; `STATIC` is the three compile/Ruff commands
above, `DOC` is the exact `tools/check_docs.py` command above, `FULL` is the exact full-pytest
command above, `P6` is the six proof commands through `stacking-transition`, `P7` is `P6` plus
`deterministic-pushforward`, `N2` is the two consecutive identical numeric-only commands, `IMG1` is
the one locked image command into a newly resolved nonexistent leaf, and `SEED` is the exact
`verify_seed.py` command. `MANIFEST` is the exact tracked-set/hash regeneration command immediately
above and always requires review before `SEED`. Focused test bundles are literal:

```powershell
# Tcore
uv run --frozen --group dev pytest -q tests/test_core_coordinates.py
# Tgeometry
uv run --frozen --group dev pytest -q tests/test_geometry_optics.py
# Tmosaic
uv run --frozen --group dev pytest -q tests/test_mosaic_ewald.py
# Tordered
uv run --frozen --group dev pytest -q tests/test_ordered_reflectivity.py
# Tstacking
uv run --frozen --group dev pytest -q tests/test_stacking_transition.py
# Tintegration
uv run --frozen --group dev pytest -q tests/test_integration.py
```

Every Python task runs `STATIC` in addition to the listed row. A comma means run every named bundle,
not choose one:

| Task or checkpoint | Exact required bundles after the task's own named mutation/link assertions |
|---|---|
| DP-00R | `DOC`, run all `P6` and record only the expected proof-base-contract failures, `MANIFEST`, manifest diff review, `SEED`, working/cached diff checks, status |
| DP-00 | `FULL`, run all `P6` and record only the expected proof-base-contract failures, `DOC`, import/base/status audit |
| DP-00A | `Tcore`, `DOC` |
| DP-00B / Checkpoint 0 | `Tmosaic`, `Tstacking`, `P6`, `DOC` |
| DP-00C | `Tcore`, `Tintegration` |
| DP-00D | `Tmosaic`, `Tgeometry`, `mosaic-ewald`, `geometry-optics`, `DOC` |
| DP-01 | `Tmosaic`, `mosaic-ewald` |
| DP-01A | `Tmosaic`, `Tordered`, `mosaic-ewald`, `ordered-reflectivity` |
| DP-02A / DP-02B / DP-02C | `Tmosaic`, `Tintegration`, `mosaic-ewald` |
| Checkpoint A | `Tmosaic`, `Tintegration`, `mosaic-ewald`, `DOC` |
| DP-03 | `Tcore`, `Tgeometry`, `geometry-optics` |
| DP-03A | `Tgeometry`, `geometry-optics` |
| DP-04A / DP-04B / DP-04C | `Tgeometry`, `Tintegration`, `geometry-optics`, `mosaic-ewald` |
| DP-05 | `Tintegration`, `P6`, `N2` |
| DP-05V1 | `Tcore`, `Tgeometry`, `Tmosaic`, `core`, `geometry-optics`, `mosaic-ewald`, `DOC` |
| DP-05V2 | `Tordered`, `Tstacking`, `ordered-reflectivity`, `references`, `stacking-transition` |
| DP-05V3 | `Tcore`, `P6`, `DOC` |
| DP-05P | `Tintegration`, `P7` |
| DP-06I / DP-06P | `Tcore`, `Tintegration`, `P7` |
| Checkpoint B0 | `FULL`, `P7`, `DOC` |
| DP-06C / DP-06D / DP-06E1 / DP-06E2 | `Tmosaic`, `Tintegration`, `P7` |
| DP-06F | `Tintegration`, `P7` |
| DP-06G | `Tordered`, `Tstacking`, `P7` |
| DP-06H | `Tintegration`, `Tordered`, `Tstacking`, `P7` |
| Checkpoint B | `FULL`, `P7`, `DOC`, `N2` (the exact benchmark/peak-RSS gate) |
| DP-06U | `Tintegration`, `FULL`, `P7`, exact sampled public/private/RNG/script equivalence assertions |
| DP-06V | `Tcore`, `Tintegration`, `FULL`, `P7`, `DOC`, public-import/signature/version assertions |
| DP-06V2 | `Tintegration`, `P7`, `N2`, public/private equivalence and private-caller scan |
| DP-06A / DP-06B / DP-06J / DP-06K / DP-06L / DP-06M | `DOC`, `P7` |
| Checkpoint D | `FULL`, `P7`, `DOC`, `N2`, pre-cutover live-instruction/inbound-link scans only; the future-deletion production zero-match pattern is not yet applicable |
| DP-07A / DP-07B / DP-08 | `FULL`, `P7`, each task's exact owned-symbol/import subset scans; do not run the full future-deletion zero-match pattern early |
| DP-09 | `FULL`, `P7`, both exact residual scans below |
| DP-10 | `Tintegration`, `FULL`, `P7`, `N2`, `IMG1`, exact two-file/dimension/hash assertions |
| Checkpoint C | `FULL`, `P7`, `DOC`, both residual scans, inspect DP-10's recorded image leaf |
| DP-13 / DP-13A | `DOC`, `P7` |
| DP-13B | `FULL`, `P7`, `DOC`, both exact residual scans |
| DP-15 | `STATIC`, `FULL`, `P7`, `DOC`, `N2` (benchmark/RSS), `IMG1` into a new final leaf, both residual scans, `MANIFEST`, final manifest diff review, `SEED`, all diff/status gates |

The bare proof names in the table mean the exact
`uv run --frozen --group dev python -m rasim_next.proof <name> --json` form shown above. `N2` is
required twice verbatim with no intervening scientific-state change. DP-10 uses a temporary fresh
external validation leaf; DP-15 uses a different fresh final-deliverable leaf. No intermediate task
other than DP-00R runs `SEED`, because only DP-00R deliberately refreshes the reconciled planning
manifest; DP-15 owns the implementation branch's final manifest regeneration and seed gate.

BKI-13 deletes the stale `integration` dispatcher registration; DP-00B verifies it remains absent.
`all` and `tiny-end-to-end` are not accepted commands and must not be claimed as gates unless a
later separately approved plan adds and proves real implementations.

The final production scan over `src`, `tests`, and `scripts` must return zero occurrences of:

```text
MosaicOrientationBatch
manuscript_axisymmetric_v1_orientation_quadrature
CandidatePool
CandidateMassSummary
SelectedCandidateBatch
ScatteringEventBatch
EventTransportResult
transport_scattering_events
build_scattering_events
_detector_complete_rods
_symmetric_hk_bounds
select_candidates
selection_seed
draw_count
deposit_bilinear
alpha_cell_count
azimuth_cell_count
```

The exact zero-match production command is:

```powershell
$removedSymbolPattern = 'MosaicOrientationBatch|manuscript_axisymmetric_v1_orientation_quadrature|CandidatePool|CandidateMassSummary|SelectedCandidateBatch|ScatteringEventBatch|EventTransportResult|transport_scattering_events|build_scattering_events|_detector_complete_rods|_symmetric_hk_bounds|select_candidates|selection_seed|draw_count|deposit_bilinear|alpha_cell_count|azimuth_cell_count'
rg -n --glob '*.py' $removedSymbolPattern src tests scripts
if ($LASTEXITCODE -eq 0) { throw "removed production symbol remains" }
if ($LASTEXITCODE -ne 1) { throw "production residual scan failed" }
```

Run a second scan over root Markdown, `tasks`, `docs`, prompts, and YAML. Its reviewed allowlist is limited to historical
explanation or deletion/mutation statements in this accepted plan/checklist and the validation or
decision ledgers. Executable prompts, package exports, commands, successor edges, and any imperative
instruction/import are never allowlisted. Record both exact commands and reviewed residual rows in
the handoff. The exact residual-producing command is:

```powershell
rg -n --glob '*.md' --glob '*.yaml' --glob '*.yml' $removedSymbolPattern .
if ($LASTEXITCODE -gt 1) { throw "documentation residual scan failed" }
```

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Ewald tangency gives an integrable coarea singularity | High | Certify topology, split one-sided support, evaluate/bound the fused finite coarea-chart continuation through `s=0`, and never form `0*inf` or cap it |
| A located tangent is non-transverse or its chart is unresolved | High | Classify `DEGENERATE_TANGENCY` and stop that branch for a new derivation; never apply `D=s^2` without gradient/chart proofs |
| Sampling misses a disconnected support, fold, or status island | High | Use domain-covering analytic/interval branch-and-bound; unresolved cells fail and images/convergence are not absence proofs |
| Exit critical angle or detector fold makes the composite map singular/non-injective | High | Forward-integrate latent cells, retain all preimages, subdivide folds, and avoid pointwise determinant division |
| A cell crosses a sphere-bin/exit/detector boundary | High | Integrate certified latent indicators, split boundaries, bound the full unresolved mass, expose every category, and fail nonzero numeric-failure mass |
| Mapped-polygon area gives closure but the wrong pixel distribution | High | Use polygons only for candidate topology; prove pixel arrays with latent indicator integrals and a varying-map-determinant mutation |
| Conservative pixel-box pushforward changes the current bilinear ensemble mean | High | Declare `CORRECTED` at detector pixel integration, compare through the first divergent stage, then use conservation, certified-bound coverage, and an independent latent-indicator oracle |
| The mosaic produces many `L` values for one family/root | High | Maximize the declared mosaic density on closed support, store only `(L_peak,m,intersection_branch_id)`, and never tag off-peak cells |
| A component maximum lies only at Ewald tangency | High | Optimize on the closure, retain `TANGENT_BOUNDARY`, permit roots 1/2 to share geometry, and test the limiting labels |
| Several rods represent the same exact-`m` peak solution | Medium | Keep every rod in the mass sum, use the complete analytic alpha/beta/rod total order for metadata, and test it independently of cells |
| The exact `(h,k)=(0,0)` coating is logarithmically divergent without physical support | High | Exclude its intensity for every lattice family; permit only mass-field-free metadata branch-0 tags with `SPECULAR_INTENSITY_EXCLUDED` |
| Current batch/support/mixture/atom domains regress, or planned general-family/phase/stacking seams are omitted, at cutover | High | Persist the classified inventory and complete DP-06C--DP-06H before deletion; the atom is a separate beta-line measure and general families use `m=None` |
| Same top surface is applied to inward-going waves | High | Freeze one top interface, classify the outward normal sign before optics, and keep rejected mass separate; a bottom interface requires a new derivation |
| A claimed certified bound is only a refinement heuristic | Blocker | DP-00D proves outward-rounded full-integrand enclosures before production code; inability to contract stops the branch |
| Tolerance/API/trace migration corrupts accepted evidence | High | Preserve v1 bytes/hash, add schema-distinct v2, advance recorded versions once, and reject mixed schemas while keeping reference v4 historical |
| Cache reuse silently crosses a geometry/tolerance change | High | Use explicit immutable in/out cache values, canonical SHA256 keys, full/projection replacement semantics, and a two-batch memory bound |
| Adaptive boundaries make later fit objectives nonsmooth | Medium | Freeze/reuse accepted cells while only intensity parameters change; geometry/mosaic changes explicitly invalidate them |
| 3000x3000 pixel clipping is too slow or memory-heavy | Medium | Stream sparse mapped cell footprints, tile pixels, profile before acceleration, and enforce the branch resource gate |
| Sphere display accidentally becomes a physical correction | Medium | Keep sphere density in a diagnostic result, prove its mass sum, and prohibit it as detector input |
| Temporary old/new coexistence survives | High | Allow it only through pre-cutover parity/docs work, require Checkpoint D, delete it in DP-07A/DP-07B, and enforce final scans |
| “Default” fixture drifts between 1-degree and 2-degree mosaic cases | Medium | Freeze the 2-degree validation fixture in code/test metadata before implementation; any change requires spec approval |
| Stale BKI/parallel routing or another writer overlaps the deterministic branch | High | Land DP-00R, preserve completed BKI hashes, reconcile status/ownership, and route only to the deterministic successor; one writer owns overlapping paths |
| Optional rendering overwrites or weakens an image contract | Medium | Lock Matplotlib, require a new external leaf, refuse overwrite, assert exactly two files and the 2400x1200 sphere canvas, and keep PNGs non-proof |

## Final acceptance criteria

- [ ] A new `codex/deterministic-ewald-pushforward` workbranch starts from warning-free approved
      post-merge main containing this committed plan/checklist and completed DP-00R reconciliation;
      retired/parallel/BKI writers are reconciled, completed BKI hashes are recorded, and the fresh
      API/import/fixture audit passes without status warnings.
- [ ] Historical BKI-04A/BKI-14 >5-file deviations have explicit owner-approved audit dispositions
      and independent revalidation without a false atomicity-compliance claim; every deterministic
      task is <=5 files. V1 tolerance evidence is byte/hash unchanged, v2 is a strict numerical
      superset, and explicit tolerance/API/trace migrations reject mixed versions.
- [ ] The centered, zero-divergence, monochromatic source boundary derives and verifies the 5-degree
      internal `ki`; no internal wavevector is injected by hand.
- [ ] All 120 explicit `m!=0` rods and all 240 ordered `(rod_id,root_label)` slots are classified
      from the explicit h-major/k-minor `RodCatalog` without Monte Carlo or detector-derived rod
      discovery, and every supported regular component is integrated.
- [ ] Complete support/fold/status topology is analytically or interval certified. Every regular
      Ewald tangent has a transverse nonsingular chart and a finite fused continuation through the
      endpoint; injected degenerate/unresolved tangencies stop and `0*inf` is never formed.
- [ ] The Ewald coating applies one folded mosaic measure, exactly one area-valued ordered/stacking
      strength, and one coarea factor; signed-density trace and `r_e^2` ownership remain intact.
- [ ] Every optically weighted outgoing beam is eligible for the sole top surface, is correct in
      SAMPLE and LAB,
      and begins detector projection at the actual sample intersection.
- [ ] Detector pixels contain certified latent-indicator mass for exact boxes, with
      separately closed exhaustive pre-optics/post-optics `ValidityCode` ledgers, no accepted cell
      crossing an unresolved status boundary, no mapped-area allocation, and no pixel-solid-angle
      or plane-Jacobian multiplier.
- [ ] Sphere bins use exact seam-safe `(mu,phi)` box ownership and certified allocation bounds; the
      frozen internal coating result is invariant to detector geometry/refinement.
- [ ] Exactly one internal peak-mosaic representative is cached for every nonempty degenerate
      `(incident_state_id,phase_geometry_id,family_id,intersection_branch_id)` closure, with exact
      `(L_peak,m,intersection_branch_id)` display metadata and no assigned/deposited mass field;
      invalid projected rows
      remain cached and off-peak mosaic `L` values are never tagged. Cache ownership is explicit,
      immutable, canonical-fingerprinted, bounded to two batches, and obeys full/projection misses.
- [ ] Exact `(h,k)=(0,0)` intensity remains explicitly excluded in every lattice family with no
      epsilon or legacy fallback; specular branch-0 tags use only the non-direct root, carry
      `SPECULAR_INTENSITY_EXCLUDED`, and are geometry-only.
- [ ] Every accepted post-BKI incident/wavelength batch, finite/unbounded support, smooth mixture,
      zero-tilt beta-line atom, and current positive single-phase population scalar—including
      non-unit values—has deterministic parity before deletion. Zero-source-weight and entrance-
      invalid rows emit no coating/tag, while invalid positive source probability remains in the
      dimensionless incident-status ledger. General-cell, phase/parent, and ordered/stacking
      dovetail seams have independently proved first deterministic implementations; fixture-only
      rejections do not survive cutover.
- [ ] The candidate selector, sampled-event RNG, discrete orientation batch, point depositor, old
      image script, obsolete tests, and live imperative legacy instructions are deleted. Indexed
      T03/T07 remain routing/history stubs; conditional BKI files remain when obligations/links do.
- [ ] No compatibility facade, alternate backend, active stale/overlapping worktree, unresolved
      retired-branch disposition, TODO, temporary switch, or generated repository artifact remains.
- [ ] Analytic, independent-oracle, convergence, error-injection, full-suite, lint, documentation,
      all seven proof, benchmark, memory, and clean-tree gates pass.
- [ ] One external `bi2se3_5deg_ewald_coating.png` and one external
      `bi2se3_5deg_detector_pushforward.png` are generated from the same accepted run and shown to
      the user in a fresh external leaf containing no other file; the sphere PNG decodes exactly
      2400x1200 and the detector numeric source is native `(3000,3000)`.
- [ ] `FILE_MANIFEST.json` is regenerated only after the final file set and validation text, and
      `verify_seed.py`, cached/working/base-range diff checks, and warning-free clean status pass.
- [ ] The final workbranch contains one coherent commit and a complete scientific handoff.

## Review gate

Implementation must not start until the user approves this corrected specification, especially the
unbounded one-ray/pure-2-degree-Gaussian proof fixture, accepted-domain parity before cutover,
versioned certified sphere/pixel indicator ownership, sole-top-interface policy, fused tangent and
topology proofs, universal exact `(h,k)=(0,0)` exclusion, and immutable closed-support peak-mosaic
cache definition.
