# Specification and atomic plan: deterministic Ewald-coating pushforward

Status: **PROPOSED — corrected after independent validation; human re-review required; no
implementation has started.**

This is the authoritative replacement proposal for the Monte Carlo coating and detector-integration
framework. It supersedes the production direction in:

- `tasks/continuous_ewald_coating_replacement_plan.md`;
- the selection/deposition portions of `tasks/07_integration.md`;
- Sections 4–7 of `docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`; and
- the sampled-event portions of `tasks/parallel_simulation_geometry_fitting_plan.md`.

The two named coating/parallel plan files contain existing uncommitted work. The implementation
workbranch retires or reconciles all four superseded documents in explicit tasks below. The
companion execution checklist is `tasks/deterministic_ewald_pushforward_todo.md`.

Before BKI-15 executes, this specification and checklist must be committed on the approved base or
transferred by an owner-approved, hash-recorded handoff. The BKI successor graph must route only to
`codex/deterministic-ewald-pushforward`: stale parallel Task 1.1 and the old coating-validation
successor must not start concurrently. BKI-16 and BKI-17 must either land before the pushforward
base is recorded or remain explicitly paused until the deterministic handoff. Any existing
`codex/continuous-ewald-coating-validation` ref is preserved as owner-controlled provenance unless
its unique work is explicitly ported or archived; it is never reused, resumed, or silently deleted.
At this audit, that ref points to unique unmerged commit
`8681b01aa476a96ad6e53529c3ff0a249a6720f5`; DP-00 must verify the ref or record an owner-approved
successor disposition rather than assuming it disappeared.

Before DP-00C freezes the nominal fixture, DP-01 freezes pointwise coating records, or DP-06 freezes
internal/projection cache keys, the shared beam-to-`ki`
[Checkpoint K](plan.md#checkpoint-k-shared-beam-to-ki-boundary) must be accepted. The complete
source realization and its `IncidentStateBatch` are built once and serially before any downstream
component tiling. Any later private parallel record retains `parent_row_index` into that immutable
parent; `incident_state_id` remains verified identity payload and is never a sorting key. Workers
never regenerate source rows or construct/hash public batch slices.

Because these two plan files were absent from BKI-15's precomputed manifest delta and its old
successor graph, the BKI owner must import the corrected planning commit, replace both stale
successor edges, and recompute the declared file count, path set, hashes, and manifest delta before
BKI-15 runs. The deterministic branch may not treat the current unchecked BKI-15 declaration as a
completed prerequisite.

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
- No Gaussian/Lorentzian mixture in the first replacement API. The accepted tag solver and full
  end-to-end fixture are restricted to one positive-width pure wrapped Gaussian; unsupported
  mixtures fail before cell allocation rather than using an uncertified representative.
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
- [ ] Analytic or interval branch-and-bound can certify the complete topology of Ewald support,
      sphere/detector folds, and validity regions; adaptive sampling alone is not an absence proof.
- [ ] Maximization on the closure of regular root support, including tangent-boundary semantics and
      a continuous periodic-`beta` tie-break, yields exactly one
      `(L_peak,m,intersection_branch_id)` tag per nonempty supported degenerate set.
- [ ] The existing entrance/exit transport and detector equations can be consumed in batches
      without duplicating or weakening their scalar proof path.
- [ ] Tolerance-bounded conservative integration into exact detector-pixel boxes is the intended
      corrected observable, even though it intentionally diverges from the current bilinear
      ensemble mean at that stage.
- [ ] The frozen coating-validation mosaic is the pure 2-degree Gaussian case. The tracked
      1-degree forward-case mosaic and a Gaussian/Lorentzian mixture are not silently substituted.
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
- The one-row `IncidentSampleBatch` is ordinary empirical source mass with
  `source_weight == 1.0`. Only its component-tag output is zero-assigned-mass metadata: it has no
  assigned or deposited mass and cannot enter any photon or detector ledger.
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

Near a certified transverse tangent, the adaptive integrator—not `reciprocal/coating.py`—owns a
purely numerical coordinate-chart Jacobian. For a local nonsingular chart `(D,eta)` and the
one-sided substitution `D=s^2`,

```text
abs(det d(alpha,beta)/d(s,eta))
  = 2*abs(s) / abs(det d(D,eta)/d(alpha,beta)).
```

This chart factor is applied exactly once with the latent integration measure and is not a second
physical Ewald/coarea factor. The implementation records its chart orientation and refuses a chart
whose denominator cannot be bounded away from zero.

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
difference and is accompanied by a separately certified quadrature/mapping upper bound; any
heuristic refinement estimate is separately named and is never promoted to a bound. No residual
may be assigned as a balancing plug. Rod, family, root, and incident-state subtotals must each
reduce to their parent total. Exit classification and detector classification each expose an
exhaustive mass mapping keyed by every `ValidityCode`; codes inapplicable at a stage have exact zero mass.
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
2. For each nonempty `Z(i,m,b)`, globally maximize exactly the declared mosaic probability density
   `p_alpha(alpha)/(2*pi)`. Ordered strength, coarea, exit optics, polarization, detector validity,
   and pixel position do not participate. The first API accepts only the frozen positive-width
   pure Gaussian, so the certified comparison is equivalently the global minimum of `alpha` on the
   closed support. Enumerate and bound every interior, seam, and tangent-boundary candidate; a
   local optimizer or adaptive-node search is not an authority.
3. Compare density in a stable analytic/log-density representation and resolve every tie by the
   total order `(-density, alpha, abs(wrapped_beta), wrapped_beta, rod_id)`, never by adaptive-node
   or discovery order. This chooses `beta=0` when it is in tied support, remains defined at the
   periodic seam, and cannot acquire a false far-tail plateau from probability underflow.
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

`component_tag_id` and canonical row order derive only from exact
`(incident_state_id,family_id,intersection_branch_id)` identity, never from `L`, floating density,
or discovery order. The internal representative key explicitly contains the immutable incident
batch/revision and internal `ki`, `sample_from_crystal.rotation`, reciprocal basis and full rod
catalog revision, wavelength/internal-medium revision, complete mosaic parameters, support/root
solver revision, tag tie-order revision, and every tolerance used to classify the closure. A
separate projection key contains the actual sample intersection, `lab_from_sample`, wavelength and
material exit-optics revision, exit solver revision, the `component_tag_id` plus an exact hash of
the internal representative payload including `kf_film_sample_Ainv`, detector transform, shape,
row/column pitches, reference coordinate, support-edge convention, and detector solver revision.
Film thickness and ordered-intensity-only changes do not move a peak-mosaic component tag because
neither changes its stored geometry. Detector changes invalidate only projected fields and never
internal membership; incident, crystal/sample, reciprocal, internal-medium, mosaic, or solver
changes invalidate the full row.

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
`(mu,phi)` bin boxes and exact detector-pixel boxes, conservative cell-mass allocation, and
numerical error/ledger helpers. Sphere-bin and detector-pixel indicator errors are separate. It
contains no adaptive physics orchestration, crystallography, Ewald root, structure-factor,
refraction, or detector-frame equation.

### Revised `simulate_ordered(...)`

The pipeline owns cross-domain orchestration: it adaptively partitions latent cells, calls the
reciprocal kernel, composes source/population/footprint and ordered strength, calls the accepted
exit/detector seams, and passes already evaluated cells to the pure accumulators. This makes factor
ownership explicit without a stateful callback or a second physics implementation.

The pipeline validates one complete `IncidentSampleBatch`, builds its complete
`IncidentStateBatch` once and serially, and then tiles deterministic component work. Private rows
carry `parent_row_index` and preserve input alignment; reassembly scatters through that index,
never an `incident_state_id` sort. Every source/incident proof field and owner-provided revision
passes through unchanged.

DP-02 freezes one canonical detector-independent internal partition, reduction order, and
`EwaldCoatingResult`. DP-04 may refine descendants of those cells for exit/detector classification
and pixel allocation, but those descendants cannot be reduced back into or otherwise change the
accepted sphere result. Detector pose and detector tolerances therefore cannot perturb the
upstream coating diagnostic.

Inputs retain the existing validated domains:

```text
one hexagonal crystal/phase with exact hex:m family metadata
one explicit IncidentSampleBatch
material optics
compiled instrument
WrappedMosaicParameters describing one positive-width pure Gaussian,
  lorentzian_probability == 0, and zero_tilt_probability_mass == 0
phase population
polarization policy and provenance
frozen pushforward tolerances
```

This first replacement API raises an informative unsupported-model error before work for an active
zero-width mosaic atom, nonzero Lorentzian mixture, nonhexagonal catalog, or multiple phases. It
never silently drops or approximates those components.

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
status counts, moment summaries, certified numerical upper bounds, and separately labelled
refinement heuristics. `DetectorPushforwardResult` replaces `DepositionResult` and contains the
native image, separate pre-optics exit classification, post-optics detector categories, closure
residuals, independently certified numerical upper bounds, and separately labelled refinement
heuristics.
`CoatingComponentTagBatch` is immutable. Its rows have no assigned or deposited event mass and do
not alter `source_weight` or any photon/detector ledger.

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

- `src/rasim_next/proof/bi2se3_pushforward_fixture.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py`
- `scripts/generate_bi2se3_pushforward_images.py`

### Rewrite

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/__init__.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/proof/core.py`
- `src/rasim_next/proof/stage_tolerances_v1.json`
- `src/rasim_next/reciprocal/proof.py`
- `src/rasim_next/stacking/proof.py`
- `src/rasim_next/geometry/proof.py`
- `pyproject.toml` and `uv.lock` for the isolated locked image dependency group.
- `FILE_MANIFEST.json` after the final file set is known.
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

The BKI `tasks/plan.md` and `tasks/todo.md`, the two already modified plan files, and all other
pre-existing dirty files remain owner-controlled. DP-00 is blocked until their owner corrects the
successor edge before BKI-15, BKI-15 lands, and all overlapping post-BKI work is either landed or
paused. Later retirement tasks reconcile remaining live text only on that approved base.

## Dependency graph

```text
committed approved plan + completed BKI-15 + warning-free clean approved main
  -> DP-00 isolated workbranch
  -> DP-00A tolerance contract
  -> DP-00B proof-base repair + Checkpoint 0 approval
  -> shared beam-to-ki Checkpoint K
  -> DP-00C single frozen-fixture authority
  -> DP-01 frozen fixture + pointwise coating
  -> DP-02 adaptive coating + sphere conservation
  -> DP-03 exit-frame correctness
  -> DP-04 detector-cell pushforward
  -> DP-05 end-to-end deterministic path
  -> DP-06 peak-mosaic component-tag cache
  -> Checkpoint B: deletion authorization
  -> DP-06A/DP-06B live contract + scientific-document synchronization
  -> DP-07A production cutover + selector/intersection deletion
  -> DP-07B reciprocal proof migration + event-builder deletion
  -> DP-08 sampled-event contract deletion
  -> DP-09 orientation/deposition deletion
  -> DP-10 locked two-image tool
  -> Checkpoint C: residue-free production path
  -> DP-14 downstream-link/strategy reconciliation
  -> DP-13/DP-13A/DP-13B old-task and residual-symbol retirement
  -> DP-15 final proof, two external images, cleanup, one coherent commit
```

Tasks are sequential under one writer. Read-only derivation, test, and performance review may run
in parallel. Each task changes no more than five files and must leave its focused verification
green. Temporary coexistence in DP-05 is allowed only in the unmerged workbranch and must be
removed immediately by DP-07A/DP-07B after Checkpoint B.

## Atomic implementation tasks

### DP-00: Create the isolated workbranch

**Files:** none.

**Work:** After this plan is approved and committed, BKI-00 through BKI-15 have completed
successfully, and the current dirty main checkout is resolved by its owner, record the approved
post-BKI main SHA as `PUSHFORWARD_BASE_SHA`. BKI-16/BKI-17 and parallel Task 1.1 must be either
landed in the recorded SHA or explicitly paused with no overlapping worktree. Resolve every
permission warning before calling the base clean. Confirm the BKI owner imported this corrected
plan/checklist, replaced its stale successor graph, and recomputed BKI-15's declared manifest file
count, paths, hashes, and delta before BKI-15 ran. Perform a fresh read-only API/import/fixture audit
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

**Verify:** committed plan/checklist object hashes, BKI-15 handoff and recomputed manifest delta,
corrected single-successor routing, no active retired/parallel/follow-up writer, explicit
preservation/port/archive decision for the retired validation ref, base SHA, reference-pack hash,
warning-free clean status with all untracked paths visible, dependency sync, fresh post-BKI audit,
and current full tests. Run every
currently runnable proof with its legacy environment and record the known reciprocal/stacking
baseline-contract failures; do not claim a six-proof green baseline before DP-00B repairs them.

**Acceptance:** BKI-15 is green; the approved committed plan is present on the base; no user change
or unique retired-branch commit is moved, overwritten, or deleted; status has no permission
warning; and the new writer exclusively owns the new worktree and every overlapping path.

### DP-00A: Freeze the new tolerance contract

**Files likely touched (4):**

- `src/rasim_next/proof/stage_tolerances_v1.json`
- `src/rasim_next/proof/tolerances.py`
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

Update `docs/VALIDATION.md` in the same task: retain the previous hash only as explicitly historical
T02--T05 evidence and name the new artifact hash as the current authority before any coating code
uses it. DP-06B later adds accepted pushforward results without deferring this authority change.

Do not alter any tolerance after observing a coating result.

**Verify:** artifact schema/hash, exact stage lookup, scalar/global-L1 norm behavior and scale
provenance, rejection of a candidate-dependent scale, rejection of a reported error estimate that
is not a certified upper bound below its frozen stage tolerance, current-versus-historical hash
wording in `docs/VALIDATION.md`, and `tools/check_docs.py`.

**Acceptance:** the new stage IDs, comparison norms, absolute floors, scale rules, and artifact hash
are frozen without observing new coating output; validation names the new current authority; no
scientific code has been added.

**Dependencies:** DP-00.

### DP-00B: Repair proof baseline ownership

**Files likely touched (4):**

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

**Files likely touched (3):**

- `src/rasim_next/proof/bi2se3_pushforward_fixture.py` (new)
- `tests/test_integration.py`
- `scripts/generate_bi2se3_detector_image.py`

**Work:** Add one pure, immutable builder for the complete frozen 5-degree source, unbounded sample,
instrument, crystal/material, pure 2-degree Gaussian mosaic, polarization declaration, and
inclusive `[-5,5]` rod window. Preserve the BKI source model/RNG/provenance assertions while
changing the diagnostic case to the exact one-row center. The module imports no Matplotlib and no
new coating implementation. Migrate the post-BKI permanent default-builder invariant away from the
old image script to this authority before that script can be deleted. Rename/relabel the old
12-degree builder as legacy comparison state, not a second canonical fixture; its module text and
API must no longer claim to be the canonical or single authority.

**Verify:** exact literals and provenance, count-one center behavior, unbounded support with absent
dimensions, detector/frame contracts, CIF/material hashes, and importability without Matplotlib.

**Acceptance:** every subsequent proof, test, and image tool names this builder; no deterministic
fixture literal block is copied elsewhere, and the old script has no permanent-test caller.

**Dependencies:** Checkpoint 0 and shared beam-to-`ki` Checkpoint K.

### DP-01: Freeze the one-ray fixture and pointwise coating equation

**Files likely touched (4):**

- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/reciprocal/coating.py` (new)
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`

**Work:** Add the folded continuous mosaic density and a pure batched reciprocal-kernel evaluator
for explicit rod/root `(alpha,beta)` points. Consume the DP-00C fixture authority—not production
`simulate_ordered`—and its explicit one-row `IncidentSampleBatch`. Assert the CIF/material hashes
and values, exact source-derived internal `ki`, 121/120 rod counts, 240 ordered nonzero-`m`
`(rod_id,root_label)` slots, root ordering, residuals, and one coarea factor. Keep the old enumerator
only as a temporary oracle.

Construct the one-row source and complete `IncidentStateBatch` once and serially before coating
records are tiled. DP-01 consumes the sole DP-00C/NOM-01 fixture authority; no script, parallel
task, worker, or coating kernel may construct a competing nominal ray.

**Verify:** analytic roots, direct delta/coarea evaluation, mosaic normalization, beta periodicity,
tangent/no-root classification, and mutation detection for missing/doubled coarea and added
`sin(alpha)`.

**Acceptance:** pointwise values are finite/nonnegative on regular support; every rod/root identity
is exact; no ordered, optics, detector, render, or pipeline import enters `coating.py`.

**Dependencies:** DP-00C and shared beam-to-`ki` Checkpoint K.

### DP-02: Add bounded adaptive cells and the Ewald-sphere accumulator

**Files likely touched (5):**

- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py` (new)
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`
- `tests/test_integration.py`

**Work:** Integrate one rod/root at a time over periodic latent cells. Use analytic identities plus
domain-covering interval branch-and-bound to certify every connected `D>0`, `D=0`, and `D<0`
region; a sampled sign grid is never an absence proof. Split one-sided support at every certified
`D=0` boundary. Near a regular tangent, choose and record a nonsingular local `(D,eta)` chart and
use `D=s^2` only after bounding `norm(gradient_(alpha,beta) D)` above the frozen transversality
tolerance on the complete boundary segment. Apply the declared
`2*abs(s)/abs(det d(D,eta)/d(alpha,beta))` chart Jacobian exactly once. A zero gradient, unresolved
chart denominator, or uncertified topology is `DEGENERATE_TANGENCY`/typed unresolved failure and
stops that component for review.

Certify and split the periodic beta seam, poles, and every zero-determinant sphere-map fold; an
unresolved cell is never accepted. Map accepted simple cells into equal-area `(mu,phi)`, clip them
against the exact frozen bin rectangles, and apportion already integrated nonnegative cell mass
with a certified bin-indicator/allocation bound. Retain every preimage and reject every
self-intersecting mapped cell for refinement. An embedded deterministic quadrature pair may drive
refinement, but only analytic/interval density, chart, curvature, and indicator enclosures are
reported as upper bounds. Stream component tiles and freeze the canonical detector-independent
partition and reduction order; no full rod-by-cell matrix is retained.

At this stage, a private proof/test harness orchestrates the kernel and accumulator; it is not a
second production implementation. DP-05 moves the accepted orchestration into `pipeline/simulate.py`
and deletes any temporary non-oracle harness code.

**Verify:** constant-density and analytic polynomial surfaces; a square-root tangent with the full
chart Jacobian; omit/double-chart-Jacobian mutations; certified transverse and injected degenerate
tangencies; a narrow disconnected support island missed by sample nodes but found by the topology
certificate; north/south poles; sphere seam/handedness; exact bin-edge ownership; a mapped bin
crossing containing no quadrature node; sphere mass conservation; rod/root/family sums; moments;
certified error-bound coverage; refinement contraction; and tile-order parity.

**Acceptance:** total/moment errors and the certified global-L1 sphere bound satisfy the frozen
tolerances, including positive absolute floors, even without a full-array oracle. The sphere
normalized `L1` change between the two finest accepted refinements is at most `1e-3` only as a
supplemental convergence gate. The frozen sphere result is invariant to detector geometry and
detector-only tolerances.

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

**Dependencies:** Checkpoint A.

### DP-04: Conservatively integrate mapped cells into exact detector boxes

**Files likely touched (3):**

- `src/rasim_next/render/pushforward.py`
- `tests/test_integration.py`
- `src/rasim_next/geometry/proof.py`

**Work:** Refine descendants of the frozen DP-02 cells through DP-03 without changing the accepted
sphere result. Use analytic identities plus domain-covering interval branch-and-bound to certify
and split every map fold, recursively bound boundary-chord curvature in detector coordinates, and
reject every self-intersecting polygon for further subdivision. Also certify and split the exit
critical boundary
`k0^2 - norm(k_parallel)^2 = 0`, detector parallel denominator, forward/backward ray-time boundary,
the exact half-open detector support/pixel edges, and every analytic status transition. If an
analytic boundary cannot be isolated, use validated interval indicator bounds and refuse the cell
until its classification uncertainty is below the frozen bound. `NUMERIC_FAILURE` is not a
geometric category: any nonzero failed mass above the absolute floor aborts acceptance.

Compute each latent-cell mass with the accepted DP-02 integrator. On an accepted simple mapped
polygon with one status, use a nonnegative cell-constant target density only to apportion that
already computed mass by polygon area clipped against exact half-open pixel boxes; validated
density, curvature, status-boundary, and pixel-allocation enclosures provide the reported bound.
Embedded-pair and parent/child differences are refinement heuristics, not promoted to bounds.
Exactness applies to pixel boxes and bookkeeping, not to the finite curved-cell approximation.
Track the exhaustive pre-optics exit and post-optics detector `ValidityCode` mappings, closure
residuals, and independent certified bounds. Do not call `deposit_bilinear` and do not divide by a
pointwise map determinant.

As in DP-02, a private proof/test harness performs orchestration until DP-05 installs the sole
production orchestration in `pipeline/simulate.py`.

**Verify:** identity and tilted analytic polygons, non-square pixels, curved-boundary refinement,
edge clipping, a certified zero-determinant fold with multiple preimages, rejection of a
self-intersecting polygon, exit-critical/detector-status boundary splits, exact row/column
orientation and half-open edge ties, an injected narrow fold/status island, independent high-order
latent quadrature on a tiny detector, exhaustive per-status mass assignment, algebraic ledger
closure separate from image accuracy, certified-bound coverage, detector-change invariance of the
sphere result, and solid-angle-invariance mutation.

**Acceptance:** analytic and full-fixture ledgers satisfy their frozen absolute-plus-relative mass
tolerance; every certified numeric bound covers observed oracle error and is itself below its
frozen tolerance. Detector normalized `L1` refinement change is at most `1e-3` as a supplemental
convergence gate, detector centroid shift at most `0.05 px`, and all pixels are finite/nonnegative.

**Dependencies:** DP-02 and DP-03.

### DP-05: Compose a temporary deterministic end-to-end path

**Files likely touched (5):**

- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/render/pushforward.py`
- `tests/test_integration.py`
- `scripts/generate_bi2se3_pushforward_images.py` (new numeric-only CLI skeleton)

**Work:** Add the deterministic path beside the old path only long enough to compare factor
ledgers. Consume and validate the explicit one-row `IncidentSampleBatch` from the sole DP-00C
fixture authority, then build rods, exact ordered strengths, internal coating, exit transport, and detector
image. Production `simulate_ordered` does not resample or construct a source. No production API
cutover occurs yet. Move the accepted adaptive orchestration out of temporary proof/test helpers
into this sole production owner; delete temporary orchestration that is not an independent oracle.
Reject nonhexagonal catalogs, a nonzero Lorentzian mixture, and active zero-width mosaic atoms
before allocating cells.

The cutover preserves the complete incident predecessor rather than replacing it with sampled-event
payload: intersection points, SAMPLE directions, air and film wavevectors, complex film normal
components, entrance amplitudes, footprint factors, wavelength, polarization, source weights,
status/valid flags, IDs, model IDs, and source/sample/material/incident revisions remain available
to their distinct proof and provenance consumers.

Add the final script path initially as a Matplotlib-free `--numeric-only --json` CLI over the sole
fixture and production pipeline. Time exactly the `simulate_ordered` call with `perf_counter` and
report `wall_time_s` plus process `peak_rss_bytes`; on the named Windows handoff machine, peak RSS
comes from `GetProcessMemoryInfo(GetCurrentProcess()).PeakWorkingSetSize` through a small private
`ctypes` helper. The JSON also records platform, Python, Git/config hashes, array hashes, component
counts, and ledgers. It writes no file unless stdout is explicitly redirected to an external path.

**Verify:** factor omission/duplication controls; exact classification of 120 rod identities and all
240 `(rod_id,root_label)` slots as empty/nonempty/failed support; integration of every supported
regular component; totals and moments; deterministic repeatability; alternate tile order;
equivalent-work wall time/peak RSS for the complete fixture using the exact numeric-only command;
and comparison with an independently converged latent oracle. Run the command twice and require
identical scientific arrays/tags while treating timing and peak RSS as measured metadata.
The old Monte Carlo mean may be used only as low-authority disposable evidence.

**Acceptance:** all deterministic gates pass; the complete fixture meets `<=15 min` and
`<=1.5 GiB` on the named handoff machine before deletion is authorized; and every difference from
bilinear/Monte Carlo output is classified at the intended first divergent stage.

**Dependencies:** DP-04.

### DP-06: Add the zero-mass peak-mosaic component-tag cache

**Files likely touched (4):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/reciprocal/coating.py`
- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`

**Work:** Implement the closed-support cache contract and a globally certified
peak-mosaic-density representative solver. Enumerate/bound all interior, seam, and tangent-boundary
candidates for the accepted pure Gaussian; compare stable log density and apply the complete
`(-density,alpha,abs(beta),beta,rod_id)` order. `reciprocal/coating.py` returns the internal candidate; the pipeline reuses
the exact exit, frame, and detector seams to attach optional projected fields. Keep all degenerate
rods/orientations in the physical sum, but emit exactly one zero-mass
`(L_peak,m,intersection_branch_id)` display row per nonempty
`(incident_state_id,family_id,intersection_branch_id)` closure using the declared continuous
tie-break.

**Verify:** analytic/global mode cases for branches 0/1/2; rejection of an unsupported mixture; a
far-tail probability-underflow mutation; a maximum attained only at a `D=0` tangent
closure; branches 1/2 sharing boundary geometry but retaining identity; exact `m`; exact continuous
`L_peak`; periodic-beta tie order independent of cells; multiple off-peak `L` values ignored;
duplicate same-`m` rods retained in mass but collapsed only in tag metadata; invalid exit/detector
rows retained with absent projected fields; exact component ID/canonical order; mutation of every
internal/projection cache-key field; projection-only cache invalidation; removal of the `m=0`
direct root; and proof that enabling tags leaves both numeric images bitwise unchanged.

**Acceptance:** every nonempty closed degenerate set has exactly one internal row maximizing only
the declared mosaic density; every detector-valid row round-trips its coordinate to the outgoing
LAB ray; invalid rows remain cached; branch-0 rows use only the non-direct root, evaluate no
physical mass, and carry `M0_INTENSITY_EXCLUDED`; no adaptive integration node is tagged merely
because it was sampled by the quadrature.

**Dependencies:** DP-05 and shared beam-to-`ki` Checkpoint K. DP-06 is the sole representative-tag
owner; parallel Task 1.8 may consume its accepted rows but cannot implement another nominal source
or representative solver.

### Checkpoint B: authorize runtime deletion

- DP-01 through DP-06 pass focused and full tests.
- Internal coating, external outgoing rays, detector mass, and component tags each have independent
  proof and mutation sensitivity.
- The complete 120-rod/240-root-slot run meets `<=15 min` and `<=1.5 GiB` before any old runtime
  file is deleted.
- A human reviews numeric summaries before the old runtime is removed.

### DP-06A: Synchronize live architecture and result contracts before cutover

**Files likely touched (5):**

- `docs/ARCHITECTURE.md`
- `docs/CONTRACTS.md`
- `docs/DOVETAIL_MATRIX.md`
- `docs/RESULT_MEASURE.md`
- `docs/TRACE_SCHEMA.md`

**Work:** Replace sampled-candidate/selection/deposition authority with the accepted deterministic
cell measure, result interfaces, factor ledger, rejection ledger, certified-error contract,
single-fixture authority, and component-tag cache. Mark the old runtime as temporary comparison
code scheduled for immediate DP-07 deletion. Remove obsolete trace stages and add stable
coating/pushforward/tag stages without renumbering unrelated stages.

**Verify:** `tools/check_docs.py`, contract/import scans, and comparison of every documented field
to the accepted DP-05/DP-06 result types.

**Acceptance:** the live architecture describes the accepted implementation before the old public
runtime is removed; no document directs a new consumer to sampled-event APIs.

**Dependencies:** Checkpoint B.

### DP-06B: Record the scientific correction and validation rules before cutover

**Files likely touched (5):**

- `docs/SCOPE_AND_PHASES.md`
- `docs/PHYSICS_LEDGER.md`
- `docs/VALIDATION.md`
- `docs/ERROR_INJECTION.md`
- `docs/DECISIONS.md`

**Work:** Replace D013 and relevant `PHY-REC`/`PHY-MEA` rows. Classify deterministic coating as
`NEW`, removal of inverse-CDF selection as `CORRECTED`, and certified conservative box-pixel
integration as `CORRECTED` first at `measurement.detector_pixel_mass`. Record the tangent-chart
Jacobian, topology certificate, two-image diagnostic, and peak-mosaic component-tag requirements
without treating convergence or images as proof.

**Verify:** proof hierarchy, tolerances, mutations, result measure, `m=0` limitation, and
`tools/check_docs.py` agree with DP-00A through DP-06.

**Acceptance:** scientific and error-injection authorities are current before deletion; no
intermediate cutover tree relies on stale sampled-event documentation.

**Dependencies:** DP-06A.

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
script in the same cutover; DP-00C already migrated its permanent test/canonical fixture caller, so
no checked-in caller is left broken. Leave
`reciprocal/events.py` temporarily only for its still-live reciprocal proof/test consumers.

**Verify:** compile, full tests, all proof commands, and static scans for selector/intersection
modules and symbols.

**Acceptance:** no production runtime fallback, feature flag, compatibility alias, candidate pool,
or outgoing-event RNG selection remains; temporary `events.py` is unreachable from production.

**Dependencies:** DP-06B.

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

**Files likely touched (5):**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/geometry/__init__.py`
- `src/rasim_next/proof/core.py`
- `tests/test_core_coordinates.py`

**Work:** Delete `ScatteringEventBatch` and its candidate-row, sampled-orientation, reciprocal
selection-weight, selected-mass, and point-contribution fields. DP-03 has already migrated geometry
transport proof/tests to the narrow deterministic internal-wave contract; DP-07B has removed the
event builder. Delete the now-unused event-shaped transport entry point, remove its public package
imports/`__all__` entries, and update the final core proof and constructor tests atomically.

Do not delete or collapse the upstream source/incident proof surface with the sampled-event
contract. Retain `IncidentSampleBatch` and the incident intersection, SAMPLE direction, air and
film wavevectors, complex film normal component, entrance amplitude, footprint, wavelength,
polarization, source weight, status/valid flags, IDs, model IDs, and revision envelope.

**Verify:** core contract/frame proof, full tests, all proof commands, a static scan for the removed
class/fields, and exact before/after comparison of the protected source/incident evidence.

**Acceptance:** no placeholder, deprecated property, or compatibility contract preserves sampled
event semantics, while every distinct source/incident proof and provenance field remains.

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

### DP-10: Add the locked replacement image tool and generate both views from one run

**Files likely touched (4):**

- `scripts/generate_bi2se3_pushforward_images.py`
- `tests/test_integration.py`
- `pyproject.toml`
- `uv.lock`

**Work:** Add a separate locked `image` dependency group containing Matplotlib; it is not part of
core or dev. Extend the DP-05 numeric-only script with an image mode that consumes the same sole
DP-00C frozen-fixture builder, imports Matplotlib only after CLI entry, calls
`matplotlib.use("Agg", force=True)` before importing `pyplot`, requires an external `--output-dir`,
computes one shared result, and writes exactly:

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
panels include internal `ki` and zero-mass component-tag overlays.
The detector PNG uses the native 3000x3000 `[row,column]` array, `origin="upper"`, nearest/no
smoothing, the same explicit zero/log rule, absolute `angstrom^2/pixel`, the detector reference
coordinate, and the same detector-valid component-tag overlays.

Titles state `5 degree`, the wavelength, `2 degree Gaussian`, `m!=0 intensity`, and
  `deterministic`. Branch-0 tags state that intensity is excluded. The script prints Git/CIF/
tolerance hashes, dependency versions, numeric array hashes, configuration, convergence, and mass
ledger as JSON to stdout. It writes no numeric sidecar by default.

**Verify:** the permanent integration test exercises only the DP-00C fixture/result/filename
contract under core+dev and proves Matplotlib is not imported. Run two numeric simulations with
identical arrays and summaries. Then run the exact locked command
`uv run --frozen --group dev --group image python scripts/generate_bi2se3_pushforward_images.py
--output-dir <resolved-external-directory>` with external `MPLCONFIGDIR` and `MPLBACKEND=Agg`;
record the locked Matplotlib version, check both images have nonzero dimensions, confirm no
repository-local output, and perform visual review. Do not skip the two images, move Matplotlib into
core/dev, or compare PNG byte hashes across dependency versions.

**Acceptance:** exactly two external PNGs are produced; the coating PNG has the frozen canvas and
tessellation bound; neither image is max-normalized, rotated, flipped, transposed, randomly
speckled, or used as the scientific proof oracle.

**Dependencies:** DP-09.

### Checkpoint C: residue-free production path

- The deterministic coating/simulation/render import path contains no `np.random`, candidate
  selection, sampled outgoing event, orientation batch, or bilinear depositor.
- The upstream source module remains because it owns source characterization; the accepted fixture
  uses its one-row center semantics only and draws no random source coordinate.
- The old image script is gone, the replacement script is runnable, and the full suite plus all six
  proof commands pass.

### DP-14: Reconcile downstream plans and every inbound link before deletion

**Files likely touched (5):**

- `docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`
- `tasks/parallel_simulation_geometry_fitting_plan.md`
- `tasks/mosaic_distribution_fitting_plan.md`
- `tasks/OVERNIGHT_RUNBOOK.md`
- `tasks/deterministic_ewald_pushforward_todo.md`

**Work:** Keep the valuable Bragg-space, rods, mosaic, structure-factor, and `m=0/m!=0` derivation,
but replace the continuous sampler with direct deterministic pushforward. Update invalidation and
fitting reuse around canonical internal cells and the component-tag cache. Remove outgoing-event
RNG, sampled-event batching, candidate assumptions, and every Markdown link to files DP-13 will
delete; retain the upstream source sampler, its accepted center-row semantics, and the complete
incident proof/provenance schema consumed at the reciprocal boundary.

**Verify:** `tools/check_docs.py` passes before deletion, and an inbound-link scan proves no retained
document points at `03_mosaic_ewald.md`, `07_integration.md`, or
`continuous_ewald_coating_replacement_plan.md`.

**Acceptance:** all retained live documents agree and deletion of the DP-13 targets cannot create a
broken link.

**Dependencies:** Checkpoint C.

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

**Verify:** task-index coverage, `tools/check_docs.py`, and the exact inbound-link scan remain green
immediately after deletion.

**Acceptance:** no active task can recreate the deleted framework, no local link is broken, and
required historical proof facts remain in `docs/VALIDATION.md` and Git history.

**Dependencies:** DP-14.

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

**Verify:** before and immediately after each conditional deletion/rewrite, run the exact inbound-
link scan for all five targets, `tools/check_docs.py`, task-index coverage, and the live-instruction
scan. A target with a retained inbound reference is rewritten or retained until its owner moves the
reference; it is never deleted into a broken intermediate tree.

**Acceptance:** repository-wide scans find no live prompt/checklist that can recreate the old
framework, and no pre-existing BKI user work was overwritten outside its accepted successor edit.

**Dependencies:** DP-13.

### DP-13B: Remove residual live sampled-event symbols and freeze scan scope

**Files likely touched (1):**

- `tasks/00_bootstrap.md`

**Work:** Replace the stale bootstrap dataflow that still names `ScatteringEventBatch`. Define two
exact final scans: (1) zero occurrences of removed modules/classes/functions in `src`, `tests`, and
`scripts`; and (2) a reviewed task/doc scan whose only allowlisted occurrences are historical
explanation or deletion/mutation statements in this accepted plan/checklist and validation/decision
ledgers. Add `ScatteringEventBatch`, `EventTransportResult`, and
`transport_scattering_events` to the removed-symbol list. A historical allowlist never permits an
imperative instruction or import.

**Verify:** run the exact scans, `tools/check_docs.py`, and import every public package initializer.

**Acceptance:** no live production/test/script/task instruction or package export can recreate the
sampled-event path; every residual text occurrence has an explicit historical allowlist reason.

**Dependencies:** DP-13A.

### DP-15: Final proof, cleanup, images, and handoff

**Files likely touched (2):**

- `docs/VALIDATION.md` if compact durable proof results are recorded
- `FILE_MANIFEST.json`

**Work:** Run the complete gate, render the two external PNGs into the Codex visualization or
another explicitly external directory, inspect them, delete all temporary comparisons and proof
artifacts, audit retained tests, benchmark equivalent work, and report peak memory. After the final
validation text and complete file set are fixed, regenerate and review `FILE_MANIFEST.json`, then
run `verify_seed.py`. Before committing, run working-tree and cached diff checks; after the one
coherent commit, run `git diff --check "${env:PUSHFORWARD_BASE_SHA}..HEAD"` and require
warning-free empty status. Squash temporary task commits as required by project policy.

**Acceptance:** every final criterion below passes, the two PNGs are shown to the user, the branch
is clean after one commit, and no generated artifact is committed.

**Dependencies:** DP-13B.

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
- Alternate tile/worker order: identities exact and floating results within the declared reduction
  tolerance; canonical production reduction restores repeatable output.
- Full 120-rod/240-root-slot fixture target on the handoff machine: both numeric arrays and
  component tags complete in `<=15 min` with peak RSS `<=1.5 GiB`. This passes before DP-07A and at
  handoff; failure triggers design review, not a tolerance waiver.

### Required error injections

- omit or double `J_ewald`;
- add an erroneous `sin(alpha)`;
- omit or double the tangent-chart Jacobian, use a singular chart, or apply `D=s^2` at a
  non-transverse tangent;
- hide a narrow disconnected support/fold/status island between sample nodes;
- assign a sphere-bin crossing or pixel-edge tie from quadrature-node ownership instead of exact
  clipped indicator ownership;
- swap roots 1 and 2;
- collapse same-`m` rods;
- drop one preimage at a fold;
- use the wrong exit square-root branch or surface normal;
- skip SAMPLE-to-LAB transformation or use the wrong ray origin;
- multiply pixel solid angle or a sphere/plane Jacobian into raw mass;
- swap detector row/column or rotate the image;
- discard outside/exit-rejected mass;
- let component-tag rows contribute mass;
- choose a tag by integration-node/local-optimizer order, create a false underflow density tie,
  omit a tangent-closure maximum, or emit other than one row for a nonempty family/root closure;
- let detector validity change the internal representative; or
- let detector geometry/refinement change the frozen sphere result;
- retain the direct `m=0` root or evaluate any branch-0 physical mass;
- silently drop a zero-width atom/nonzero Lorentzian mixture or compute hexagonal `m` for a
  nonhexagonal catalog; or
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
$env:MPLBACKEND = "Agg"
$env:PUSHFORWARD_BASE_SHA = "<approved post-BKI main SHA>"
$env:PUSHFORWARD_IMAGE_DIR = "<resolved external image directory>"

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
uv run --frozen --group dev python scripts/generate_bi2se3_pushforward_images.py --numeric-only --json
uv run --frozen --group dev --group image python scripts/generate_bi2se3_pushforward_images.py --output-dir $env:PUSHFORWARD_IMAGE_DIR
uv run --frozen --group dev python scripts/verify_seed.py
git diff --check
git diff --cached --check
# after the coherent commit:
git diff --check "${env:PUSHFORWARD_BASE_SHA}..HEAD"
git status --short
```

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
select_candidates
selection_seed
draw_count
deposit_bilinear
alpha_cell_count
azimuth_cell_count
```

Run a second scan over `tasks` and `docs`. Its reviewed allowlist is limited to historical
explanation or deletion/mutation statements in this accepted plan/checklist and the validation or
decision ledgers. `tasks/00_bootstrap.md`, executable prompts, package exports, and any imperative
instruction/import are never allowlisted. Record both exact commands and the reviewed residual
rows in the handoff.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Ewald tangency gives an integrable coarea singularity | High | Certify topology, split one-sided support, apply the full nonsingular chart Jacobian once, and prove the bound; never cap it |
| A located tangent is non-transverse or its chart is unresolved | High | Classify `DEGENERATE_TANGENCY` and stop that branch for a new derivation; never apply `D=s^2` without gradient/chart proofs |
| Sampling misses a disconnected support, fold, or status island | High | Use domain-covering analytic/interval branch-and-bound; unresolved cells fail and images/convergence are not absence proofs |
| Exit critical angle or detector fold makes the composite map singular/non-injective | High | Forward-integrate latent cells, retain all preimages, subdivide folds, and avoid pointwise determinant division |
| A cell crosses a sphere-bin/exit/detector boundary | High | Certify and split boundaries or use validated indicator bounds; expose every mass category and fail unresolved/nonzero numeric-failure mass |
| Conservative pixel-box pushforward changes the current bilinear ensemble mean | High | Declare `CORRECTED` at detector pixel integration, compare through the first divergent stage, then use conservation, certified-bound coverage, and an independent box oracle |
| The mosaic produces many `L` values for one family/root | High | Maximize the declared mosaic density on closed support, store only `(L_peak,m,intersection_branch_id)`, and never tag off-peak cells |
| A component maximum lies only at Ewald tangency | High | Optimize on the closure, retain `TANGENT_BOUNDARY`, permit roots 1/2 to share geometry, and test the limiting labels |
| Several rods represent the same exact-`m` peak solution | Medium | Keep every rod in the mass sum, use the complete analytic alpha/beta/rod total order for metadata, and test it independently of cells |
| `m=0` coating is logarithmically divergent without physical support | High | Exclude its intensity; permit only zero-mass branch-0 component tags with an explicit status |
| Mixed/zero-width mosaic or nonhexagonal families need different measure/identity proofs | Medium | Reject them explicitly in this smallest slice and plan their mixed, one-dimensional, and general-family contracts separately |
| Adaptive boundaries make later fit objectives nonsmooth | Medium | Freeze/reuse accepted cells while only intensity parameters change; geometry/mosaic changes explicitly invalidate them |
| 3000x3000 pixel clipping is too slow or memory-heavy | Medium | Stream sparse mapped cell footprints, tile pixels, profile before acceleration, and enforce the branch resource gate |
| Sphere display accidentally becomes a physical correction | Medium | Keep sphere density in a diagnostic result, prove its mass sum, and prohibit it as detector input |
| Temporary old/new coexistence survives | High | Allow it only through DP-05, require Checkpoint B, delete it in DP-07A/DP-07B, and enforce final symbol scans |
| “Default” fixture drifts between 1-degree and 2-degree mosaic cases | Medium | Freeze the 2-degree validation fixture in code/test metadata before implementation; any change requires spec approval |
| BKI/parallel follow-up writers overlap the deterministic branch | High | Land them before `PUSHFORWARD_BASE_SHA` or pause them explicitly; one writer owns all overlapping paths |
| Optional rendering cannot run reproducibly | Medium | Lock Matplotlib in the isolated image group and run the exact external-output command; never make PNGs proof |

## Final acceptance criteria

- [ ] A new `codex/deterministic-ewald-pushforward` workbranch starts from warning-free approved
      post-BKI-15 main containing this committed plan/checklist; retired/parallel/BKI follow-up
      writers have an explicit landed-or-paused disposition and the post-BKI API re-audit passes.
- [ ] The centered, zero-divergence, monochromatic source boundary derives and verifies the 5-degree
      internal `ki`; no internal wavevector is injected by hand.
- [ ] All 120 explicit `m!=0` rods and all 240 ordered `(rod_id,root_label)` slots are classified
      without Monte Carlo, and every supported regular component is integrated.
- [ ] Complete support/fold/status topology is analytically or interval certified. Every regular
      Ewald tangent has a transverse nonsingular chart and applies its coordinate Jacobian once; an
      injected degenerate or unresolved tangent stops instead of using the substitution.
- [ ] The Ewald coating applies one mosaic measure, one structure strength, and one coarea factor.
- [ ] Every outgoing beam exits through the accepted surface branch, is correct in SAMPLE and LAB,
      and begins detector projection at the actual sample intersection.
- [ ] Detector pixels contain tolerance-bounded conservative mass integrated into exact boxes, with
      separately closed exhaustive pre-optics/post-optics `ValidityCode` ledgers, no accepted cell
      crossing an unresolved status boundary, and no pixel-solid-angle or plane-Jacobian multiplier.
- [ ] Sphere bins use exact seam-safe `(mu,phi)` box ownership and certified allocation bounds; the
      frozen internal coating result is invariant to detector geometry/refinement.
- [ ] Exactly one internal peak-mosaic representative is cached for every nonempty degenerate
      `(incident_state_id,family_id,intersection_branch_id)` closure, with exact
      `(L_peak,m,intersection_branch_id)` display metadata and zero mass; invalid projected rows
      remain cached and off-peak mosaic `L` values are never tagged.
- [ ] `m=0` intensity remains explicitly excluded with no epsilon or legacy fallback; branch-0
      component tags use only the non-direct root and are geometry-only.
- [ ] Nonhexagonal catalogs, nonzero Lorentzian mixtures, and active zero-width mosaic atoms fail
      explicitly before work; none is silently dropped or approximated in the 2-D integrator.
- [ ] The candidate selector, sampled-event RNG, discrete orientation batch, point depositor, old
      image script, obsolete tests, and live legacy task instructions are deleted.
- [ ] No compatibility facade, alternate backend, active stale/overlapping worktree, unresolved
      retired-branch disposition, TODO, temporary switch, or generated repository artifact remains.
- [ ] Analytic, independent-oracle, convergence, error-injection, full-suite, lint, documentation,
      benchmark, memory, and clean-tree gates pass.
- [ ] One external `bi2se3_5deg_ewald_coating.png` and one external
      `bi2se3_5deg_detector_pushforward.png` are generated from the same accepted run and shown to
      the user.
- [ ] `FILE_MANIFEST.json` is regenerated only after the final file set and validation text, and
      `verify_seed.py`, cached/working/base-range diff checks, and warning-free clean status pass.
- [ ] The final workbranch contains one coherent commit and a complete scientific handoff.

## Review gate

Implementation must not start until the user approves this corrected specification, especially the
unbounded one-ray fixture, pure 2-degree Gaussian scope, exact/certified sphere and pixel ownership,
tangent-chart/topology proofs, `m=0` exclusion, and globally certified closed-support peak-mosaic
component-tag definition.
