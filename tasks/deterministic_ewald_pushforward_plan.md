# Specification and atomic plan: deterministic Ewald-coating pushforward

Status: **PROPOSED — human review required; no implementation has started.**

This is the authoritative replacement proposal for the Monte Carlo coating and detector-integration
framework. It supersedes the production direction in:

- `tasks/continuous_ewald_coating_replacement_plan.md`;
- the selection/deposition portions of `tasks/07_integration.md`;
- Sections 4–7 of `docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`; and
- the sampled-event portions of `tasks/parallel_simulation_geometry_fitting_plan.md`.

The two named coating/parallel plan files contain existing uncommitted work, and this planning turn
does not overwrite any existing file. The implementation workbranch retires or reconciles all four
superseded documents in explicit deletion tasks below. The companion execution checklist is
`tasks/deterministic_ewald_pushforward_todo.md`.

On approval, this specification immediately supersedes the BKI-15 successor edge that still names
`codex/continuous-ewald-coating-validation` in the untracked `tasks/plan.md`/`tasks/todo.md`. The
BKI writer must reconcile that documentation before executing BKI-15 and must route its accepted
handoff only to `codex/deterministic-ewald-pushforward`; the retired branch must not be created.
This turn leaves those pre-existing uncommitted BKI files untouched.

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
3. a separate zero-mass cache containing exactly one peak-mosaic
   `(L_peak,m,intersection_branch_id)` representative for each nonempty supported family/root
   closure, with its exit and detector status and its detector position when valid.

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
- Cache exactly one internal peak-mosaic representative for each nonempty supported exact-`m`
  family and Ewald intersection root, ignoring the many off-peak `L` values contributed by the
  mosaic distribution; keep the row even when its exit or detector projection is invalid.
- Replace the old runtime atomically and delete each superseded module, API, test, script, and live
  instruction after its replacement passes its gate.
- Leave one small production path, one compact permanent proof suite, and no compatibility switch.
- Generate exactly two external PNG deliverables from one frozen Bi2Se3 case at handoff.

## Non-goals

- No `m=0` coating intensity until a physical beamstop, support gap, or finite-resolution model is
  declared. Branch-0 geometry may be cached as zero-mass metadata only.
- No lower-dimensional zero-width mosaic atom. This slice requires
  `zero_tilt_probability_mass == 0` and rejects an active zero-width Gaussian/Lorentzian component
  before allocating work; a separate one-dimensional atom integrator is future scope.
- No nonhexagonal crystal in this first replacement slice. Exact integer `m=h^2+h*k+k^2` is part
  of the required tag contract, so the production entry point rejects a rod catalog whose
  `family_key` is not `hex:m=...`; general-cell family tags require a later contract.
- No beam-size, divergence, wavelength-bandwidth, or multi-source integration in this branch.
  The source boundary is retained, but the fixture uses exactly its center row.
- No stacking-disorder substitution, multiple phases, fitting, caking, angle-space remapping, PSF,
  background, efficiency, exposure, gain, or absolute photon-count model.
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
- [ ] Maximization on the closure of regular root support, including tangent-boundary semantics and
      a continuous periodic-`beta` tie-break, yields exactly one
      `(L_peak,m,intersection_branch_id)` tag per nonempty supported degenerate set.
- [ ] The existing entrance/exit transport and detector equations can be consumed in batches
      without duplicating or weakening their scalar proof path.
- [ ] Tolerance-bounded conservative integration into exact detector-pixel boxes is the intended
      corrected observable, even though it intentionally diverges from the current bilinear
      ensemble mean at that stage.
- [ ] The frozen coating-validation mosaic is the 2-degree Gaussian case. The tracked 1-degree
      forward-case mosaic is not silently substituted.
- [ ] The first replacement API may explicitly reject nonhexagonal catalogs and active zero-width
      mosaic atoms rather than expanding this branch to new tag identities or lower-dimensional
      measures.

DP-01 tests the physics assumptions; DP-02 through DP-04 test numerical feasibility; DP-06 tests
the tag definition. A failed dealbreaker stops the workbranch for review rather than expanding the
scope or restoring Monte Carlo.

## Technology, style, and boundaries

### Technology

- Python `>=3.12,<3.14`, NumPy float64/complex128, and the existing SciPy/Gemmi/XrayDB stack.
- No new production dependency is planned.
- Matplotlib remains an optional script-only dependency for the two requested PNGs and is never
  imported by `rasim_next` production modules or permanent tests.

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
  dependency set, `m=0` physical support, or workbranch base SHA.
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
- Wavelength: exactly `1.540592925 angstrom`; bandwidth is absent.
- Source weight: exactly `1.0`.
- Polarization: explicit `UNITY_APPROXIMATION` for this diagnostic fixture.
- Construct the row through `sample_gaussian_source_rays` with `sample_count=1`, all spatial and
  divergence sigmas zero, wavelength sigma zero, and the existing source seed `1729`. Its odd
  center row is exactly the requested mean and draws no stochastic coordinate.
- Sample motion: one active `+5 degree` rotation around `+LAB x` at the origin and no other sample,
  goniometer, or crystal-mount rotation.
- The one-row source batch must pass through `build_incident_states`; production must not inject an
  internal `ki` directly.
- The resulting accepted internal reference is
  `ki_film_sample_Ainv = (0.0, 4.062900581047559, -0.3545543022596421)` within the frozen stage
  tolerance. This is an assertion on the input boundary, not an alternate source of truth.

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
- Rod window for this proof: every `(h,k)` in inclusive `[-5,5] x [-5,5]`.
- Catalog count: `121`; active non-specular count: `120` after excluding `(0,0)` intensity.
- Every rod remains explicit even when several rods share the same integer `m`.
- Mosaic: pure wrapped Gaussian with FWHM `2 degree`, sigma
  `0.01482346182399166 rad`, zero Lorentzian mass, and no zero-width atom.
- Mosaic measure: folded `p_alpha(alpha) d alpha` and `d beta/(2*pi)`; no extra `sin(alpha)`.
- Nonzero-`m` roots are labelled `1` and `2` by increasing `L`.
- The non-direct `m=0` geometry uses label `0`, but its coating mass is excluded with an explicit
  status; no epsilon or fallback is permitted.

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
- Sample support is the finite centered rectangle `width = 2.0e-4 m` along SAMPLE `x` and
  `length = 5.0e-4 m` along SAMPLE `y`; the centered ray must have footprint weight one.
- Film thickness: `500 angstrom`.
- Population weight: `1.0`.

The detector plane is normal to the mean LAB beam in this fixture and is tilted relative to the
sample solely because the sample is at 5 degrees. Adding an independent detector tilt is outside
this branch.

### Frozen Ewald-sphere raster

- Bin the unit internal outgoing direction `s_hat = kf_film_sample/|ki_film_sample|` in
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

Here `S_r = r_e^2 * |F_r|^2` for the ordered model and

```text
J_ewald = |ki| / abs(dot(rod_direction_hat, kf_film))
```

is the one Ewald-root coarea factor. The Ewald image bins `d_mu_coat` before exit optics. A fixed
sphere bin `B` displays `mu_coat(B)/DeltaOmega_B` in `angstrom^2/sr`; the conversion must sum back
to the same coating mass and can never be reused as a physical weight.

The raw detector pixel mass is

```text
M[pixel]
  = sum(i,r,b) integral_z(
        d_mu_coat
        * W_entrance_exit_and_attenuation(z)
        * W_polarization(z)
        * indicator[Phi(i,r,b,z) lies inside pixel]
    )
```

`Phi` is the direct composition

```text
internal kf in SAMPLE
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

### Conservation ledger

The implementation reports three separate equal-measure identities:

```text
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

`M_postopt` integrates `d_mu_coat * W_optical * W_polarization` only over exit-supported points;
exit-rejected coating mass is never called optically weighted. These are scalar-field intensity
measures, not a power-flux balance. Every residual is the independently computed algebraic closure
difference and is accompanied by a separately estimated quadrature/mapping bound; no residual may
be assigned as a balancing plug. Rod, family, root, and incident-state subtotals must each reduce
to their parent total. Exit classification and detector classification each expose an exhaustive
mass mapping keyed by every `ValidityCode`; codes inapplicable at a stage have exact zero mass.
`NON_PROPAGATING` belongs to the pre-optics exit-rejected mapping and never enters `M_postopt`.
Any nonzero `NUMERIC_FAILURE` mass above the frozen absolute floor fails acceptance and is never
hidden in `r_pushforward`. Component-tag rows carry exactly zero mass and cannot change any ledger.

## Peak-mosaic component-tag cache

The tag pass is deterministic metadata derived from the same equations, not a second forward
model. It records one representative of a family/root coating component, not a Bragg peak and not
an adaptive integration node. In this section, `intersection_branch_id` is only the Ewald-root
label `0`, `1`, or `2`; it is not a fitting or Git branch.

For incident state `i`, exact hexagonal family `m`, and regular-root label `b in {1,2}`, define

```text
Z(i,m,b) = closure({(r,alpha,beta):
                     r belongs to family m,
                     alpha in [0,pi], beta on S^1 represented by [-pi,pi),
                     discriminant D > 0,
                     Ewald root label is b}).
```

The closure is taken on the compact domain `[0,pi] x S^1`; the half-open interval is only the
canonical serialized representation and does not duplicate the seam. The closure is essential:
regular support can open at `D=0`, where the mosaic-density supremum
would otherwise not be attained. A boundary winner uses the common double root, carries explicit
`TANGENT_BOUNDARY` status, and retains its limiting root label. Branches `1` and `2` may therefore
have identical tag geometry while remaining distinct components.

1. Retain every underlying rod and orientation separately in the physical mass calculation.
2. For each nonempty `Z(i,m,b)`, maximize exactly the declared mosaic probability density
   `p_alpha(alpha)/(2*pi)`. Ordered strength, coarea, exit optics, polarization, detector validity,
   and pixel position do not participate. For the frozen Gaussian this is equivalent to minimizing
   `alpha` on the closed support.
3. Resolve a density tie analytically, never by adaptive-node order: first minimize
   `abs(wrap_to_[-pi,pi)(beta))`, then the wrapped signed `beta`, then stable `rod_id`. This total
   order chooses `beta=0` when it is in the tied support and is defined at the periodic seam.
4. Store the winner's exact continuous `L`, exact integer `m`, and Ewald-root label as
   `(L_peak,m,intersection_branch_id)`. Do not round `L`, search for an integer `L`, or create tags
   for any other `L` generated away from the mosaic-density maximum.
5. Emit exactly one internal zero-mass row for every nonempty supported closure. Then pass its
   internal `kf` through the same exit-refraction, SAMPLE-to-LAB, and detector mapping used by the
   mass integral. Retain the row when exit or detector projection fails, with explicit status and
   absent optional external vector/pixel fields; never substitute an off-peak representative.
6. Detector pose and exit validity cannot change which internal representative wins. Only
   detector-valid rows are overlaid on the detector PNG.

For `m=0`, remove the direct `u=0` root before forming the branch-0 tag set and consider only the
retained non-direct root. Formally,

```text
Z(i,0,0) = closure({(r_00,alpha,beta):
                     alpha in [0,pi], beta on S^1,
                     the analytic non-direct root exists and u_non_direct != 0}).
```

The closure is taken after the direct root has been removed. A limiting point where the retained
root coalesces with `u=0` may win and is recorded as `TANGENT_BOUNDARY` with limiting
`intersection_branch_id=0`. Evaluate no `S_0`, coarea mass, optical mass, or detector mass for this
set. Its row is zero-mass metadata with `M0_INTENSITY_EXCLUDED`; invalid exit/projection status is
retained rather than deleting the representative.

Each immutable row contains at least:

```text
component_tag_id
incident_state_id
family_id, canonical representative rod_id, h, k
m = h^2 + h*k + k^2
intersection_branch_id in {0,1,2}
L_peak
mosaic_alpha_rad, mosaic_beta_rad
q_internal_sample_Ainv
kf_film_sample_Ainv
optional kf_air_sample_Ainv
optional kf_air_lab_Ainv
optional (column_px,row_px)
root_status, exit_status, detector_status, and zero-mass declaration
```

`(L_peak,m,intersection_branch_id)` is display metadata, not rod identity. The public
`CoatingComponentTagBatch` wraps immutable typed rows so invalid optional fields are `None`, not
numeric sentinels. It has exactly one row for each nonempty supported
`(incident_state_id,family_id,intersection_branch_id)` closure and records the canonical supporting
rod for provenance. Branch `0` rows are allowed only with `M0_INTENSITY_EXCLUDED` and zero mass.

The internal representative key includes source/incident, crystal/rod catalog, sample pose,
wavelength/internal-medium, mosaic model, and solver revision. A separate projection key includes
exit transport and detector revisions. Detector-pose changes invalidate only projected fields and
never internal tag membership; incident/sample/internal-medium changes invalidate the full row.
Ordered-intensity-only changes do not move a peak-mosaic component tag.

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
solving through `kf_film_sample`. It imports no ordered, optics, detector, render, or pipeline
module.

### `render/pushforward.py`

Owns pure equal-solid-angle sphere accumulation, simple-polygon clipping against exact
detector-pixel boxes, conservative cell-mass allocation, and numerical error/ledger helpers. It
contains no adaptive physics orchestration, crystallography, Ewald root, structure-factor,
refraction, or detector-frame equation.

### Revised `simulate_ordered(...)`

The pipeline owns cross-domain orchestration: it adaptively partitions latent cells, calls the
reciprocal kernel, composes source/population/footprint and ordered strength, calls the accepted
exit/detector seams, and passes already evaluated cells to the pure accumulators. This makes factor
ownership explicit without a stateful callback or a second physics implementation.

Inputs retain the existing validated domains:

```text
one hexagonal crystal/phase with exact hex:m family metadata
one explicit IncidentSampleBatch
material optics
compiled instrument
WrappedMosaicParameters with zero_tilt_probability_mass == 0
phase population
polarization policy and provenance
frozen pushforward tolerances
```

This first replacement API raises an informative unsupported-model error before work for an active
zero-width mosaic atom, nonhexagonal catalog, or multiple phases. It never silently drops those
components.

Removed inputs:

```text
MosaicOrientationBatch
alpha/azimuth cell counts
selection_seed
draw_count
candidate chunk size
```

The result contains:

```text
incident transport
complete RodCatalog
EwaldCoatingResult
DetectorPushforwardResult
CoatingComponentTagBatch
m=0 exclusion status
```

`EwaldCoatingResult` contains equal-solid-angle sphere-bin mass/density, per-rod/root/family totals,
status counts, moment summaries, and error estimates. `DetectorPushforwardResult` replaces
`DepositionResult` and contains the native image, separate pre-optics exit classification,
post-optics detector categories, closure residuals, and independently estimated numerical bounds.
`CoatingComponentTagBatch` is immutable and every row has zero mass.

No compatibility aliases are retained for the removed contracts.

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

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py`
- `scripts/generate_bi2se3_pushforward_images.py`

### Rewrite

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/core/contracts.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/proof/core.py`
- `src/rasim_next/proof/__main__.py`
- `src/rasim_next/proof/stage_tolerances_v1.json`
- `src/rasim_next/reciprocal/proof.py`
- `src/rasim_next/stacking/proof.py`
- `src/rasim_next/geometry/proof.py`
- the existing compact test modules and live contracts named by the tasks below.

### Delete after the named replacement gate

- `src/rasim_next/reciprocal/events.py`
- `src/rasim_next/pipeline/intersections.py`
- `src/rasim_next/pipeline/selection.py`
- `src/rasim_next/render/deposition.py`
- `scripts/generate_bi2se3_detector_image.py`

Also delete the retired orientation-batch constructors, selected-candidate contracts, RNG selection
arguments, bilinear-deposition tests, Monte Carlo frequency tests, and live task instructions. Do
not delete `reciprocal/ewald.py`, source/incident transport, exit refraction, detector geometry, or
the independent scalar proofs; they are validated authorities, not legacy residue.

The currently untracked BKI `tasks/plan.md` and `tasks/todo.md`, the two already modified plan
files, and all other pre-existing dirty files remain untouched by this planning turn. DP-00 is
blocked until their BKI owner corrects the successor edge before BKI-15, BKI-15 lands, and those
files have a single owner; later retirement tasks reconcile any remaining live text on the approved
base.

## Dependency graph

```text
approved plan + completed BKI-15 + clean approved main
  -> DP-00 isolated workbranch
  -> DP-00A tolerance contract
  -> DP-00B proof-dispatch/base repair + approval checkpoint
  -> DP-01 frozen fixture + pointwise coating
  -> DP-02 adaptive coating + sphere conservation
  -> DP-03 exit-frame correctness
  -> DP-04 detector-cell pushforward
  -> DP-05 end-to-end deterministic path
  -> DP-06 peak-mosaic component-tag cache
  -> Checkpoint B: deletion authorization
  -> DP-07A production cutover + selector/intersection deletion
  -> DP-07B reciprocal proof migration + event-builder deletion
  -> DP-08 sampled-event contract deletion
  -> DP-09 orientation/deposition deletion
  -> DP-10 two-image tool replacement
  -> Checkpoint C: residue-free production path
  -> DP-11..DP-14A live-document/task retirement
  -> DP-15 final proof, two external images, cleanup, one coherent commit
```

Tasks are sequential under one writer. Read-only derivation, test, and performance review may run
in parallel. Each task changes no more than five files and must leave its focused verification
green. Temporary coexistence in DP-05 is allowed only in the unmerged workbranch and must be
removed immediately by DP-07A/DP-07B after Checkpoint B.

## Atomic implementation tasks

### DP-00: Create the isolated workbranch

**Files:** none.

**Work:** After this plan is approved, BKI-00 through BKI-15 have completed successfully, and the
current dirty main checkout is resolved by its owner, record the approved post-BKI main SHA as
`PUSHFORWARD_BASE_SHA`. Create a separate worktree and branch
`codex/deterministic-ewald-pushforward`. Do not reuse
`codex/continuous-ewald-coating-validation`.

Planned PowerShell commands, after resolving the target path and confirming it does not exist:

```powershell
$pushforwardBaseSha = git rev-parse main
git worktree add -b codex/deterministic-ewald-pushforward `
  ..\SLATE-rMC-wt-deterministic-ewald $pushforwardBaseSha
```

**Verify:** BKI-15 handoff, corrected successor routing, absence of the retired validation branch,
base SHA, reference-pack hash, clean worktree, dependency sync, and current full tests. Run every
currently runnable proof with its legacy environment and record the known reciprocal/stacking
baseline-contract failures; do not claim a six-proof green baseline before DP-00B repairs them.

**Acceptance:** BKI-15 is green; no user change is moved or overwritten; the new writer owns only
the new worktree.

### DP-00A: Freeze the new tolerance contract

**Files likely touched (3):**

- `src/rasim_next/proof/stage_tolerances_v1.json`
- `src/rasim_next/proof/tolerances.py`
- `tests/test_core_coordinates.py`

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
     + 1e-8*sum(abs(reference_array)).
```

The artifact/parser records the scalar versus `global_l1` comparison kind. Freeze `L_peak` with
`atol=1.4210854715202206e-14`, `rtol=2.2737367544328376e-13`, and
`scale=max(abs(L_reference),1)`. Keep the existing pixel-coordinate tolerance. Record the new
discriminant and angular-gradient stages in `angstrom^-2` and `angstrom^-2/rad` with
`atol=1.4210854715202206e-14`, `rtol=4.547473508866709e-13`, and reference scale `|ki|^2`;
transversality requires the gradient norm to exceed that full bound. Record the new artifact
SHA256 and prohibit a candidate-dependent scale.

Do not alter any tolerance after observing a coating result.

**Verify:** artifact schema/hash, exact stage lookup, scalar/global-L1 norm behavior, and rejection
of a candidate-dependent scale.

**Acceptance:** the new stage IDs, comparison norms, absolute floors, scale rules, and artifact hash
are frozen without observing new coating output; no scientific code has been added.

**Dependencies:** DP-00.

### DP-00B: Repair proof dispatch and baseline ownership

**Files likely touched (5):**

- `src/rasim_next/proof/__main__.py`
- `src/rasim_next/reciprocal/proof.py`
- `src/rasim_next/stacking/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_stacking_transition.py`

**Work:** Remove the dead `integration` proof registration. Migrate both the reciprocal hard-coded
T03 `PROOF_BASE_SHA` gate and the stacking dependency on retired
`origin/codex/proof-base` to the recorded `PUSHFORWARD_BASE_SHA` contract, without changing their
scientific oracles. Add negative tests for a missing/wrong base SHA and registry targets. The six
accepted commands are `core`, `geometry-optics`, `mosaic-ewald`, `ordered-reflectivity`,
`references`, and `stacking-transition`.

**Verify:** proof-command registry importability, missing/wrong base rejection, correct-base
acceptance, and all six proof commands from the clean worktree.

**Acceptance:** all six documented proofs are green with the approved base SHA; no proof depends
on a retired branch/ref, and the dead dispatcher entry is gone.

**Dependencies:** DP-00A.

### Checkpoint 0: tolerance and proof authority

- A human approves the immutable tolerance artifact and its hash.
- Every documented proof command is runnable from the clean worktree with the declared external
  cache environment.
- Failure stops the branch before any new coating implementation.

### DP-01: Freeze the one-ray fixture and pointwise coating equation

**Files likely touched (4):**

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/reciprocal/coating.py` (new)
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`

**Work:** Add the folded continuous mosaic density and a pure batched reciprocal-kernel evaluator
for explicit rod/root `(alpha,beta)` points. The test fixture—not production `simulate_ordered`—
builds the frozen source/instrument and supplies its explicit one-row `IncidentSampleBatch`. Assert
the CIF/material hashes and values, exact source-derived internal `ki`, 121/120 rod counts, root
ordering, residuals, and one coarea factor. Keep the old enumerator only as a temporary oracle.

**Verify:** analytic roots, direct delta/coarea evaluation, mosaic normalization, beta periodicity,
tangent/no-root classification, and mutation detection for missing/doubled coarea and added
`sin(alpha)`.

**Acceptance:** pointwise values are finite/nonnegative on regular support; every rod/root identity
is exact; no ordered, optics, detector, render, or pipeline import enters `coating.py`.

**Dependencies:** Checkpoint 0.

### DP-02: Add bounded adaptive cells and the Ewald-sphere accumulator

**Files likely touched (5):**

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py` (new)
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Integrate one rod/root at a time over periodic latent cells. Locate every discriminant
boundary `D=0`, split one-sided support there, and use a square-root regularizing coordinate
`D=s^2` near a regular tangent only after proving
`norm(gradient_(alpha,beta) D) > tolerance` on that boundary segment. A point with `D=0` and
vanishing gradient is `DEGENERATE_TANGENCY`; it stops the component/branch for review rather than
using the regular substitution. Split the periodic beta seam and all detected zero-determinant
sphere-map folds. Use an embedded deterministic quadrature pair to estimate cell mass error,
accumulate into the frozen equal-solid-angle raster, and stream component tiles. No
self-intersecting mapped cell is accepted as one polygon.

At this stage, a private proof/test harness orchestrates the kernel and accumulator; it is not a
second production implementation. DP-05 moves the accepted orchestration into `pipeline/simulate.py`
and deletes any temporary non-oracle harness code.

**Verify:** constant-density and analytic polynomial surfaces, a square-root tangent case with an
explicitly located transverse boundary, injected degenerate tangency that stops with the named
status, north/south poles, sphere seam/handedness, sphere mass conservation, rod/root/family sums,
moments, refinement contraction, error-bound coverage, and tile-order parity.

**Acceptance:** total and unit-direction moment errors satisfy the frozen mass tolerance, including
its positive absolute floor; the sphere normalized `L1` change between the two finest accepted
refinements is at most `1e-3` as a convergence gate, not an accuracy oracle; no full rod-by-cell
matrix is retained.

**Dependencies:** DP-01.

### Checkpoint A: internal coating

- Focused mosaic/Ewald tests and proof pass.
- The first requested image can be rendered from deterministic numeric bins, although no PNG is
  retained yet.
- The old runtime has not been deleted because detector replacement is not yet proven.

### DP-03: Prove outgoing-wave correctness at the sample surface and in LAB

**Files likely touched (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`

**Work:** Introduce the narrow deterministic internal-wave batch and expose only the accepted
batched seams needed by deterministic cells; leave the validated scalar refraction and detector
equations authoritative. For every regular internal `kf`, conserve the component tangent to
`z_sample=0` with normal `+z_sample`, apply the propagation-direction/non-growing branch rule
already declared by `optics/refraction.py`, and retain exit amplitude and attenuation separately.
Keep the old event-shaped entry point only until the production cutover so the full suite remains
green between tasks. Do not add an unreviewed blanket `kf_z > 0` rule. Transform the accepted
external vector exactly
once as

```text
r_hat_lab = lab_from_sample.apply_vector(kf_air_sample) / k0
```

and project a ray beginning at the actual sample intersection. The pipeline, not
`reciprocal/coating.py`, attaches exit/detector fields to internal tag candidates.

**Verify:** tangential conservation; external dispersion `|kf_air|=2*pi/lambda`; accepted
propagation-direction status; evanescent/critical statuses; `n -> 1`; SAMPLE-to-LAB round trip;
global rigid-rotation covariance; detector ray/pixel round trip; and a mutation for wrong frame,
wrong normal sign, direction rule, or origin.

**Acceptance:** the frozen 5-degree case reproduces accepted frame/optical tolerances and no
downstream code reconstructs the exit branch or rotation.

**Dependencies:** DP-01.

### DP-04: Conservatively integrate mapped cells into exact detector boxes

**Files likely touched (3):**

- `src/rasim_next/render/pushforward.py`
- `tests/test_integration.py`
- `src/rasim_next/geometry/proof.py`

**Work:** Map adaptive cell boundaries through DP-03. Locate and split map folds, recursively bound
boundary-chord curvature in detector coordinates, and reject every self-intersecting polygon for
further subdivision. Also locate and split the exit critical boundary
`k0^2 - norm(k_parallel)^2 = 0`, detector parallel denominator, forward/backward ray-time boundary,
detector support edges, and every analytic status transition. If an analytic boundary cannot be
located, use an independently error-bounded indicator quadrature and refuse the cell until its
classification error is below the frozen bound. `NUMERIC_FAILURE` is not a geometric category: any
nonzero failed mass above the absolute floor aborts acceptance.

Compute each latent-cell mass with the embedded quadrature from DP-02. On an accepted simple mapped
polygon with one status, use a nonnegative cell-constant density only to apportion that already
computed mass by polygon area clipped against exact pixel boxes; parent/child refinement bounds
density, curvature, status-boundary, and pixel-allocation error. Exactness applies to pixel boxes
and bookkeeping, not to the finite curved-cell approximation. Track the exhaustive pre-optics exit
and post-optics detector `ValidityCode` mappings, closure residuals, and independent error bounds.
Do not call `deposit_bilinear` and do not divide by a pointwise map determinant.

As in DP-02, a private proof/test harness performs orchestration until DP-05 installs the sole
production orchestration in `pipeline/simulate.py`.

**Verify:** identity and tilted analytic polygons, non-square pixels, curved-boundary refinement,
edge clipping, a located zero-determinant fold with multiple preimages, rejection of a
self-intersecting polygon, exit-critical/detector-status boundary splits, exact row/column
orientation, independent high-order latent quadrature on a tiny detector, exhaustive per-status
mass assignment, algebraic ledger closure separate from image accuracy, estimator coverage, and
solid-angle-invariance mutation.

**Acceptance:** analytic and full-fixture ledgers satisfy their frozen absolute-plus-relative mass
tolerance and every independently estimated numeric bound covers its observed oracle error;
detector normalized `L1` refinement change is at most `1e-3` as a convergence gate, detector
centroid shift at most `0.05 px`, and all pixels are finite/nonnegative.

**Dependencies:** DP-02 and DP-03.

### DP-05: Compose a temporary deterministic end-to-end path

**Files likely touched (4):**

- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py`
- `tests/test_integration.py`

**Work:** Add the deterministic path beside the old path only long enough to compare factor
ledgers. Consume and validate the explicit one-row `IncidentSampleBatch` built by the test/image
fixture, then build rods, exact ordered strengths, internal coating, exit transport, and detector
image. Production `simulate_ordered` does not resample or construct a source. No production API
cutover occurs yet. Move the accepted adaptive orchestration out of temporary proof/test helpers
into this sole production owner; delete temporary orchestration that is not an independent oracle.
Reject nonhexagonal catalogs and active zero-width mosaic atoms before allocating cells.

**Verify:** factor omission/duplication controls, classification of all 120 rods/root attempts,
integration of every supported regular component, totals and moments, deterministic repeatability,
alternate tile order, and comparison with an independently converged latent oracle.
The old Monte Carlo mean may be used only as low-authority disposable evidence.

**Acceptance:** all deterministic gates pass and every difference from bilinear/Monte Carlo output
is classified at the intended first divergent stage.

**Dependencies:** DP-04.

### DP-06: Add the zero-mass peak-mosaic component-tag cache

**Files likely touched (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`

**Work:** Implement the closed-support cache contract and analytic peak-mosaic-density
representative solver. `reciprocal/coating.py` returns the internal candidate; the pipeline reuses
the exact exit, frame, and detector seams to attach optional projected fields. Keep all degenerate
rods/orientations in the physical sum, but emit exactly one zero-mass
`(L_peak,m,intersection_branch_id)` display row per nonempty
`(incident_state_id,family_id,intersection_branch_id)` closure using the declared continuous
tie-break.

**Verify:** analytic mode cases for branches 0/1/2; a maximum attained only at a `D=0` tangent
closure; branches 1/2 sharing boundary geometry but retaining identity; exact `m`; exact continuous
`L_peak`; periodic-beta tie order independent of cells; multiple off-peak `L` values ignored;
duplicate same-`m` rods retained in mass but collapsed only in tag metadata; invalid exit/detector
rows retained with absent projected fields; projection-only cache invalidation; removal of the
`m=0` direct root; and proof that enabling tags leaves both numeric images bitwise unchanged.

**Acceptance:** every nonempty closed degenerate set has exactly one internal row maximizing only
the declared mosaic density; every detector-valid row round-trips its coordinate to the outgoing
LAB ray; invalid rows remain cached; branch-0 rows use only the non-direct root, evaluate no
physical mass, and carry `M0_INTENSITY_EXCLUDED`; no adaptive integration node is tagged merely
because it was sampled by the quadrature.

**Dependencies:** DP-05.

### Checkpoint B: authorize runtime deletion

- DP-01 through DP-06 pass focused and full tests.
- Internal coating, external outgoing rays, detector mass, and component tags each have independent
  proof and mutation sensitivity.
- A human reviews numeric summaries before the old runtime is removed.

### DP-07A: Cut over and delete selector/intersection runtime

**Files likely touched (5):**

- `src/rasim_next/reciprocal/rods.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/pipeline/intersections.py` (delete)
- `src/rasim_next/pipeline/selection.py` (delete)
- `scripts/generate_bi2se3_detector_image.py` (delete)

**Work:** Move only the still-required complete rod-bound helper to `reciprocal/rods.py`; make the
deterministic path the sole `simulate_ordered`; remove selection RNG, selected-event compaction,
candidate pools, draw counts, and imports of the two deleted pipeline modules. Delete the old image
script in the same cutover so no checked-in caller of the removed API is left broken. Leave
`reciprocal/events.py` temporarily only for its still-live reciprocal proof/test consumers.

**Verify:** compile, full tests, all proof commands, and static scans for selector/intersection
modules and symbols.

**Acceptance:** no production runtime fallback, feature flag, compatibility alias, candidate pool,
or outgoing-event RNG selection remains; temporary `events.py` is unreachable from production.

**Dependencies:** Checkpoint B.

### DP-07B: Migrate reciprocal proof and delete the event builder

**Files likely touched (3):**

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

**Files likely touched (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/proof/core.py`
- `tests/test_core_coordinates.py`

**Work:** Delete `ScatteringEventBatch` and its candidate-row, sampled-orientation, reciprocal
selection-weight, selected-mass, and point-contribution fields. DP-03 has already migrated geometry
transport proof/tests to the narrow deterministic internal-wave contract; DP-07B has removed the
event builder. Delete the now-unused event-shaped transport entry point and update the final core
proof and constructor tests atomically.

**Verify:** core contract/frame proof, full tests, all proof commands, and a static scan for the
removed class/fields.

**Acceptance:** no placeholder, deprecated property, or compatibility contract preserves sampled
event semantics.

**Dependencies:** DP-07B.

### DP-09: Delete discrete orientation construction and point deposition

**Files likely touched (4):**

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

### DP-10: Add the replacement image script and generate both views from one run

**Files likely touched (2):**

- `scripts/generate_bi2se3_pushforward_images.py` (new)
- `tests/test_integration.py`

**Work:** Add one optional-Matplotlib script with a pure frozen-fixture builder and an external
`--output-dir`. One call computes one shared result and writes exactly:

```text
bi2se3_5deg_ewald_coating.png
bi2se3_5deg_detector_pushforward.png
```

The sphere PNG uses the frozen front/back orthographic view of equal-solid-angle bins, one absolute
log color scale labelled `angstrom^2/sr`, the internal `ki`, and zero-mass component-tag overlays.
The detector PNG uses the native 3000x3000 `[row,column]` array, `origin="upper"`, nearest/no
smoothing, absolute `angstrom^2/pixel`, the detector reference coordinate, and the same
detector-valid component-tag overlays.

Titles state `5 degree`, the wavelength, `2 degree Gaussian`, `m!=0 intensity`, and
  `deterministic`. Branch-0 tags state that intensity is excluded. The script prints Git/CIF/
tolerance hashes, dependency versions, numeric array hashes, configuration, convergence, and mass
ledger as JSON to stdout. It writes no numeric sidecar by default.

**Verify:** the permanent integration test exercises only the pure fixture/result/filename contract
and never imports Matplotlib. Run two numeric simulations with identical arrays and summaries.
When the optional Matplotlib dependency is present in the handoff environment, run the script
manually, check both images exist with nonzero dimensions, confirm no repository-local output, and
perform visual review. Do not skip the two requested handoff images, add Matplotlib to core/test
dependencies, or compare PNG byte hashes across Matplotlib versions.

**Acceptance:** exactly two external PNGs are produced; neither image is max-normalized, rotated,
flipped, transposed, randomly speckled, or used as the scientific proof oracle.

**Dependencies:** DP-09.

### Checkpoint C: residue-free production path

- The deterministic coating/simulation/render import path contains no `np.random`, candidate
  selection, sampled outgoing event, orientation batch, or bilinear depositor.
- The upstream source module remains because it owns source characterization; the accepted fixture
  uses its one-row center semantics only and draws no random source coordinate.
- The old image script is gone, the replacement script is runnable, and the full suite plus all six
  proof commands pass.

### DP-11: Synchronize the live architecture and result contracts

**Files likely touched (5):**

- `docs/ARCHITECTURE.md`
- `docs/CONTRACTS.md`
- `docs/DOVETAIL_MATRIX.md`
- `docs/RESULT_MEASURE.md`
- `docs/TRACE_SCHEMA.md`

**Work:** Replace sampled-candidate/selection/deposition language with the deterministic cell
measure, result interfaces, factor ledger, rejection ledger, and component-tag cache. Remove
obsolete trace stages; add stable coating/pushforward/tag stages without renumbering unrelated
stages.

**Acceptance:** documents describe exactly one live implementation and one owner per factor.

**Dependencies:** Checkpoint C.

### DP-12: Record the scientific correction and validation rules

**Files likely touched (5):**

- `docs/SCOPE_AND_PHASES.md`
- `docs/PHYSICS_LEDGER.md`
- `docs/VALIDATION.md`
- `docs/ERROR_INJECTION.md`
- `docs/DECISIONS.md`

**Work:** Replace D013 and relevant `PHY-REC`/`PHY-MEA` rows. Classify deterministic coating as
`NEW`, removal of inverse-CDF selection as `CORRECTED`, and tolerance-bounded conservative
box-pixel integration as `CORRECTED` first at `measurement.detector_pixel_mass`. Record the
two-image diagnostic and peak-mosaic component-tag requirements without treating images as proof.

**Acceptance:** proof hierarchy, tolerances, mutations, result measure, and `m=0` limitation agree.

**Dependencies:** DP-11.

### DP-13: Retire the old coating/integration instructions

**Files likely touched (5):**

- `tasks/03_mosaic_ewald.md` (delete after historical facts move to validation)
- `tasks/07_integration.md` (delete)
- `tasks/continuous_ewald_coating_replacement_plan.md` (delete)
- `tasks/index.yaml`
- `tasks/06_parallel_review.md`

**Work:** Point the task index at this accepted plan/branch, retire the already completed historical
instructions, and remove every live direction to build candidate pools, sample CDFs, assign `T/N`,
or deposit point events.

**Acceptance:** no active task can recreate the deleted framework; required historical proof facts
remain in `docs/VALIDATION.md` and Git history.

**Dependencies:** DP-12.

### DP-13A: Retire remaining prompt and BKI successor instructions

**Files likely touched (5, only if still live on the approved post-BKI base):**

- `tasks/prompts/integration.md`
- `tasks/prompts/mosaic_ewald.md`
- `tasks/README.md`
- `tasks/plan.md`
- `tasks/todo.md`

**Work:** Delete obsolete executable prompts or rewrite the minimum task routing needed to name
this accepted deterministic successor. Remove any remaining instruction to call
`build_scattering_events`, branch from BKI-15 into the retired coating-validation branch, construct
candidate CDFs, or preserve bilinear deposition. If BKI-15 already retired a listed file, record it
as an explicit no-op rather than recreating it.

**Acceptance:** repository-wide scans find no live prompt/checklist that can recreate the old
framework, and no pre-existing BKI user work was overwritten outside its accepted successor edit.

**Dependencies:** DP-13.

### DP-14: Reconcile downstream plans and the pure strategy document

**Files likely touched (5):**

- `docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`
- `tasks/parallel_simulation_geometry_fitting_plan.md`
- `tasks/mosaic_distribution_fitting_plan.md`
- `tasks/OVERNIGHT_RUNBOOK.md`
- `tasks/deterministic_ewald_pushforward_todo.md`

**Work:** Keep the valuable Bragg-space, rods, mosaic, structure-factor, and `m=0/m!=0` derivation,
but replace the continuous sampler with direct deterministic pushforward. Update invalidation and
fitting reuse around deterministic cells and the component-tag cache. Remove outgoing-event RNG,
sampled-event batching, and candidate assumptions from active plans; retain the upstream source
sampler and its accepted center-row semantics.

**Acceptance:** all live docs agree; historical Git history is the only remaining source of the
retired strategy.

**Dependencies:** DP-13A.

### DP-15: Final proof, cleanup, images, and handoff

**Files likely touched:** `docs/VALIDATION.md` only if compact durable proof results are recorded;
otherwise none.

**Work:** Run the complete gate, render the two external PNGs into the Codex visualization or
another explicitly external directory, inspect them, delete all temporary comparisons and proof
artifacts, audit retained tests, benchmark equivalent work, and report peak memory. Squash any
temporary task commits into one coherent branch commit as required by project policy.

**Acceptance:** every final criterion below passes, the two PNGs are shown to the user, the branch
is clean after one commit, and no generated artifact is committed.

**Dependencies:** DP-14.

## Testing strategy

### Permanent tests retained

- `tests/test_mosaic_ewald.py`: continuous mosaic normalization, analytic root/coarea identity,
  tangent/no-root behavior, and one compact sphere-conservation case.
- `tests/test_geometry_optics.py`: exit surface branch, tangential conservation, SAMPLE/LAB frame
  transform, actual ray origin, detector mapping, and existing optical invariants.
- `tests/test_integration.py`: one tiny independent cell-to-pixel oracle, full-ledger conservation,
  deterministic repeatability, and peak-mosaic component-tag cache behavior.
- `tests/test_core_coordinates.py`: revised contract/frame/index invariants only.

Do not add a new permanent test module. Temporary full-catalog sweeps, image arrays, broad
refinement grids, performance harnesses, and mutation variants are deleted before handoff.

### Frozen numerical acceptance

- Exact identities/statuses/rod IDs/family IDs/root labels and the count of all valid and invalid
  tag rows: exact equality.
- Ewald residual and detector coordinates: approved frozen stage tolerances.
- Regular component mass and unit-direction first/second moments versus an independent oracle:
  `abs(error) <= 1.1284538007351267e-23 angstrom^2 + 1e-8*reference_scale`.
- Whenever an independent sphere/detector array oracle is available, compare with one global `L1`
  bound `1.1284538007351267e-23 angstrom^2 + 1e-8*sum(abs(reference))`; never grant that total-mass
  scale separately to every bin or pixel.
- Sphere, pre-optics exit, and post-optics detector ledger closures satisfy the same frozen
  absolute-plus-relative mass rule; independently estimated quadrature/mapping bounds must also
  cover observed oracle error.
- Two-finest-refinement normalized `L1` change: `<=1e-3` for both sphere and detector arrays as a
  convergence gate only, never as an accuracy oracle.
- Detector centroid change: `<=0.05 px` between the two finest refinements.
- Identical inputs and canonical tile order: bitwise identical numeric arrays and cache rows.
- Alternate tile/worker order: identities exact and floating results within the declared reduction
  tolerance; canonical production reduction restores repeatable output.
- Full 120-rod fixture target on the handoff machine: both numeric arrays and component tags complete in
  `<=15 min` with peak RSS `<=1.5 GiB`. Failure triggers design review, not a tolerance waiver.

### Required error injections

- omit or double `J_ewald`;
- add an erroneous `sin(alpha)`;
- apply `D=s^2` at a non-transverse tangent or cross an unresolved status boundary;
- swap roots 1 and 2;
- collapse same-`m` rods;
- drop one preimage at a fold;
- use the wrong exit square-root branch or surface normal;
- skip SAMPLE-to-LAB transformation or use the wrong ray origin;
- multiply pixel solid angle or a sphere/plane Jacobian into raw mass;
- swap detector row/column or rotate the image;
- discard outside/exit-rejected mass;
- let component-tag rows contribute mass;
- choose a tag by integration-node order, omit a tangent-closure maximum, or emit other than one
  row for a nonempty family/root closure;
- let detector validity change the internal representative; or
- retain the direct `m=0` root or evaluate any branch-0 physical mass;
- silently drop a zero-width atom or compute hexagonal `m` for a nonhexagonal catalog; or
- apply the total-mass tolerance independently to every pixel.

Each mutation must fail first at its owning stage. Only a minimal representative subset remains in
permanent tests.

## Commands

Use external uv, Python-bytecode, pytest, and image-output locations as required by repository
policy. The exact external paths are resolved and checked before running; an illustrative
PowerShell preamble is:

```powershell
$env:UV_CACHE_DIR = "$env:TEMP\rasim-next-uv-cache"
$env:PYTHONPYCACHEPREFIX = "$env:TEMP\rasim-next-pycache"
$env:PYTEST_ADDOPTS = "--basetemp=$env:TEMP\rasim-next-pytest -o cache_dir=$env:TEMP\rasim-next-pytest-cache"
$env:RUFF_CACHE_DIR = "$env:TEMP\rasim-next-ruff-cache"
$env:MPLCONFIGDIR = "$env:TEMP\rasim-next-matplotlib"
$env:PUSHFORWARD_BASE_SHA = "<approved post-BKI main SHA>"

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
uv run --frozen --group dev python tools/check_docs.py
uv run --frozen --group dev python scripts/verify_seed.py
git diff --check
```

DP-00A deletes the stale `integration` dispatcher registration. `all` and `tiny-end-to-end` are not
accepted commands and must not be claimed as gates unless a later separately approved plan adds
and proves real implementations.

The final static scan must find no live production/test/script/task use of:

```text
MosaicOrientationBatch
manuscript_axisymmetric_v1_orientation_quadrature
CandidatePool
CandidateMassSummary
SelectedCandidateBatch
build_scattering_events
select_candidates
selection_seed
draw_count
deposit_bilinear
alpha_cell_count
azimuth_cell_count
```

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Ewald tangency gives an integrable coarea singularity | High | Detect topology, split one-sided support, use a regularizing coordinate, and prove convergence; never cap it |
| A located tangent is non-transverse | High | Classify `DEGENERATE_TANGENCY` and stop that branch for a new derivation; never apply `D=s^2` without the gradient proof |
| Exit critical angle or detector fold makes the composite map singular/non-injective | High | Forward-integrate latent cells, retain all preimages, subdivide folds, and avoid pointwise determinant division |
| A cell crosses an exit/detector status boundary | High | Locate and split analytic boundaries or use independently bounded indicator quadrature; expose every `ValidityCode` mass and fail nonzero numeric-failure mass |
| Conservative pixel-box pushforward changes the current bilinear ensemble mean | High | Declare `CORRECTED` at detector pixel integration, compare through the first divergent stage, then use conservation, estimator coverage, and an independent box oracle |
| The mosaic produces many `L` values for one family/root | High | Maximize the declared mosaic density on closed support, store only `(L_peak,m,intersection_branch_id)`, and never tag off-peak cells |
| A component maximum lies only at Ewald tangency | High | Optimize on the closure, retain `TANGENT_BOUNDARY`, permit roots 1/2 to share geometry, and test the limiting labels |
| Several rods represent the same exact-`m` peak solution | Medium | Keep every rod in the mass sum, use the analytic beta/rod total order for metadata, and test it independently of cells |
| `m=0` coating is logarithmically divergent without physical support | High | Exclude its intensity; permit only zero-mass branch-0 component tags with an explicit status |
| Zero-width mosaic atoms or nonhexagonal families need different measures/identity | Medium | Reject them explicitly in this smallest slice and plan their one-dimensional/general-family contracts separately |
| Adaptive boundaries make later fit objectives nonsmooth | Medium | Freeze/reuse accepted cells while only intensity parameters change; geometry/mosaic changes explicitly invalidate them |
| 3000x3000 pixel clipping is too slow or memory-heavy | Medium | Stream sparse mapped cell footprints, tile pixels, profile before acceleration, and enforce the branch resource gate |
| Sphere display accidentally becomes a physical correction | Medium | Keep sphere density in a diagnostic result, prove its mass sum, and prohibit it as detector input |
| Temporary old/new coexistence survives | High | Allow it only through DP-05, require Checkpoint B, delete it in DP-07A/DP-07B, and enforce final symbol scans |
| “Default” fixture drifts between 1-degree and 2-degree mosaic cases | Medium | Freeze the 2-degree validation fixture in code/test metadata before implementation; any change requires spec approval |

## Final acceptance criteria

- [ ] A new `codex/deterministic-ewald-pushforward` workbranch starts from the approved post-BKI-15
      main, with the tolerance/proof artifact approved before physics work.
- [ ] The centered, zero-divergence, monochromatic source boundary derives and verifies the 5-degree
      internal `ki`; no internal wavevector is injected by hand.
- [ ] All 120 explicit `m!=0` rods/root attempts are classified without Monte Carlo, and every
      supported regular component is integrated.
- [ ] Every regular Ewald tangent is transverse under the frozen bound; an injected degenerate
      tangent stops with `DEGENERATE_TANGENCY` rather than using the square-root substitution.
- [ ] The Ewald coating applies one mosaic measure, one structure strength, and one coarea factor.
- [ ] Every outgoing beam exits through the accepted surface branch, is correct in SAMPLE and LAB,
      and begins detector projection at the actual sample intersection.
- [ ] Detector pixels contain tolerance-bounded conservative mass integrated into exact boxes, with
      separately closed exhaustive pre-optics/post-optics `ValidityCode` ledgers, no accepted cell
      crossing an unresolved status boundary, and no pixel-solid-angle or plane-Jacobian multiplier.
- [ ] Exactly one internal peak-mosaic representative is cached for every nonempty degenerate
      `(incident_state_id,family_id,intersection_branch_id)` closure, with exact
      `(L_peak,m,intersection_branch_id)` display metadata and zero mass; invalid projected rows
      remain cached and off-peak mosaic `L` values are never tagged.
- [ ] `m=0` intensity remains explicitly excluded with no epsilon or legacy fallback; branch-0
      component tags use only the non-direct root and are geometry-only.
- [ ] Nonhexagonal catalogs and active zero-width mosaic atoms fail explicitly before work; neither
      is silently dropped or approximated in the 2-D integrator.
- [ ] The candidate selector, sampled-event RNG, discrete orientation batch, point depositor, old
      image script, obsolete tests, and live legacy task instructions are deleted.
- [ ] No compatibility facade, alternate backend, dead branch, TODO, temporary switch, or generated
      repository artifact remains.
- [ ] Analytic, independent-oracle, convergence, error-injection, full-suite, lint, documentation,
      benchmark, memory, and clean-tree gates pass.
- [ ] One external `bi2se3_5deg_ewald_coating.png` and one external
      `bi2se3_5deg_detector_pushforward.png` are generated from the same accepted run and shown to
      the user.
- [ ] The final workbranch contains one coherent commit and a complete scientific handoff.

## Review gate

Implementation must not start until the user approves this specification, especially the frozen
2-degree mosaic fixture, frozen sphere raster, tolerance-bounded box-pixel observable, `m=0`
exclusion, and closed-support peak-mosaic component-tag definition.
