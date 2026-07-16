# Validation and proof

Tolerance selection and required negative controls are authoritative in [ERROR_INJECTION.md](ERROR_INJECTION.md).

## Proof hierarchy

1. Analytic identities and limiting cases.
2. Independent numerical oracles that do not call the public algorithm.
3. Direct manuscript-equation evaluation.
4. Shared immutable original-RASIM traces.
5. Tiny end-to-end detector result.

## Completed cross-repository audits

The mosaic and initial ordered audits below started from SLATE-rMC baseline
`caf7acd649a27dc66c6c0b73a2f66dcd520389f9`; the later PbI2 polytype proof started from
`7bccbd328220345b1a62a588d3b418bf82c1f0a9`. External repositories were consulted read-only;
`ra_sim` execution was isolated in a separate process, while the `2D_Mosaic_Sim` peak-shape
equation was evaluated independently in a proof-only harness. Neither production code nor
permanent tests import either repository. These conclusions apply to the named observable and
declared measure or atomic-factor model, not to display-normalized arrays or unrelated downstream
physics.

### Mosaic cap/ring shape and probability measure

The existing wrapped, probability-normalized mosaic density was evaluated without changing the
production model. The audit used `(0,0,3)` (`m=0`) for the cap and `(1,0,0)` and `(1,0,3)`
(`m!=0`) for rings, with pure Gaussian, mixed, and pure Lorentzian profiles. After recentering by
the reciprocal-metric Bragg polar angle, the largest cap/ring line-shape difference was
`1.9206858326015208e-14`. Profiles were scaled to unit peak only for this shape comparison;
conservation used the raw probability-normalized density. Integrating that density with the
declared spherical measure `G^2 sin(theta) dtheta dphi` returned both test intensities, `I0=1`
and `I0=7.25`, with maximum error `1.7763568394002505e-15`. The nonzero-`m` peak locus closed
through `2*pi`; the `m=0` locus remained a cap.

Four targeted mutations changed the cap/ring width, omitted the spherical measure, summed raw
surface density, or reversed the family-topology dispatch. Each failed at its intended density,
quadrature, or event-weight stage. Together with the seven existing T03 controls, all eleven
mutations were detected in the one-shot audit. Main retains the seven T03 controls after the four
temporary cap/ring controls are retired. This audit did not newly compare Ewald/Bragg intersection
loci or detector projection with `2D_Mosaic_Sim`; Ewald correctness remains supported by the
existing tracked-legacy `MATCH`, independent dense oracle, and elastic-residual proofs. Those
proofs cover narrow, broad, tail, tangent/no-root, bandwidth, and specular cases; the largest
regular-case elastic residual was `8.881784197001252e-16 A^-1`. The audit does not create a second
Ewald or mosaic implementation.

Against `2D_Mosaic_Sim` commit `5efb3233d60843f3fd4e0e3b5b73536f05c035e8`, the pure-Gaussian
peak-normalized shape is `MATCH`. Mixed and Lorentzian shapes are `CORRECTED` first at
`mosaic.wrapped_line_density`: that program mixes unit-peak components and uses an unwrapped
Lorentzian tail, whereas SLATE-rMC mixes probability mass in a wrapped density. The external
program therefore is not an oracle for absolute orientation probability. The accepted conclusion
is that the original SLATE-rMC mosaic model correctly shares one recentered shape between caps and
rings and conserves intensity under the project's declared measure; no production change is
required.

### Raw complex ordered structure factor

The structure-factor audit ran the clean `ra_sim` commit
`8fb1415e8e4695aa2ce8ec7f576b575264d4b328` in a separate Python process and compared raw complex
amplitudes in electron units before squaring, rounding, pruning, or normalization. The first
default-to-default divergence was the atomic-factor data source: SLATE-rMC uses Waasmaier--Kirfel
`f0` with XrayDB/Chantler anomalous terms, while that `ra_sim` revision uses ITC-1992 `f0` with
Henke anomalous terms. Phase sign, reciprocal coordinates, site expansion, occupancy, and
displacement factors agreed.

Holding the atomic-factor table equal to the legacy oracle produced `MATCH` for the PbI2 2H and
Bi2Se3 whole-cell amplitudes and for both physical PbI2 layer orientations. An independent audit
exercised 51,200 atomic factors and 70,400 whole-cell and layer amplitudes across
`7.92068--8.17898 keV`; the atomic factors matched exactly and the largest complex-amplitude error
was `7.7276e-13 e`, below the audit's declared `2e-12 e` absolute acceptance bound. The matched
legacy table was the `ra_sim` package default supplied by Dans_Diffraction 3.3.3. PbI2's absent
isotropic displacement value was supplied explicitly as `unknown_u_iso_A2=0.0`. The two
repositories' plus/minus layer labels are reversed, so layer results were aligned by physical
orientation rather than label.

This isolates the legacy numerical mismatch to the declared atomic-factor choice and proves the
ordered structure-factor equation and conventions. The accepted scientific default remains the
existing XrayDB/Chantler path, so no legacy atomic-factor compatibility mode or production change
is required; the temporary matched-table comparison path is not retained. That earlier audit
covers ordered PbI2/Bi2Se3 structure factors only; the separate proof below establishes the PbI2
stacking-disorder, 4H/6H transition, and phase-mixture claims.

### PbI2 2H/4H/6H structure-factor and stacking parity

The polytype proof kept three quantities distinct: file-native whole-cell `F_cell`, physical
one-layer `F_plus`/`F_minus` derived only from `PbI2_2H.cif`, and finite-stack `I_stack`. This is the
same 2H motif authority used by the historical stacking path. The 4H and 6H CIFs validate their own
expanded whole-cell structures and stacking topology; they do not replace the 2H layer pair in the
homogenized transition model. Legacy plus/minus strings were aligned by physical layer orientation.

The direct three-atom 2H sum and production layer amplitudes differed by at most
`2.4457182613372133e-14 e`. Sixty direct-sequence checks across 2H, both 4H hands, both 6H hands,
three period multiples, and four physical-Q events had maximum amplitude error
`1.2397319609735734e-12 e`. The native 2H/4H/6H files expanded to site/motif counts `3/1`, `6/2`,
and `9/3`; twelve direct whole-cell sums differed from `unit_cell_amplitude` by at most
`1.563597684016243e-13 e`. The strict legacy `N=50` finite-per-layer comparison differed by
`3.2862601528904634e-14 e^2/layer` after removing legacy `AREA` and aligning phase,
normalization, and initial-state conventions.

An independent full-six-state/direct oracle covered all three parent types at `epsilon` values
`{0, 0.01, 0.1, 0.5, 0.99}` plus 18 convex binary/ternary mixtures. The largest component and
mixture errors were both `1.8917489796876907e-10 e^2`; transition-mass error was
`2.220446049250313e-16`, and linearity and zero-weight errors were exactly zero. The structure-factor
and aggregate evidence SHA-256 values are respectively
`6379e6fa3ed3e3ab9da97fc21f9b4d3b07a15e9d8f7f2f34cb41925d96247f8e` and
`0da8be40686da52c179d8febf1791d332bcb143f12517713218e332ddf028b7f`.

The equations and convention-matched legacy comparisons are `MATCH`. Default atomic-factor
differences remain intentionally `CORRECTED` first at `ordered.atomic_amplitude`, and the accepted
registry convention is `CORRECTED` first at `stacking.registry_phase`. Comparing relaxed native
4H/6H cells directly with ideal 2H-derived parents is `NO_ORACLE` because they are different
structural models. All assigned mutations were detected, all 29 permanent tests and focused proof
gates passed, and no production code, API, dependency, CLI, or example changed. The disposable
proof design, tolerances, exact legacy anchors, performance, and branch disposition are recorded in
[WORKBRANCH_ARCHIVE_2026-07-16.md](WORKBRANCH_ARCHIVE_2026-07-16.md).

## Permanent suite

Keep a small permanent suite:

```text
test_core_coordinates.py
test_geometry_optics.py
test_mosaic_ewald.py
test_ordered_reflectivity.py
test_stacking_transition.py
test_integration.py
```

Post-integration work adds only:

```text
test_selection.py
test_fitting.py
```

The sequential fitting tasks extend `test_fitting.py`; they do not leave one permanent module per fit stage. Do not create large per-feature suites.

Permanent tests:

- use analytic and tiny direct cases
- require no original-RASIM or manuscript access
- commit no large images or experimental data
- write no diagnostics
- exclude benchmarks and broad convergence sweeps

## Branch proof gates

Every branch must pass:

- analytic or invariant checks
- independent oracle where required
- shared-pack legacy classification
- first-divergence record for each correction
- convergence only when the calculation has a real numerical refinement variable
- equivalent-work benchmark and peak memory
- clean import and dependency check
- assigned error-injection controls fail at the expected first stage
- clean Git tree

## Core analytic checks

### Shared and geometry

- transform inverse and composition
- orthogonal rotations with determinant `+1`
- pixel-to-ray-to-pixel round trip
- direct beam and detector-plane intersection
- global rigid-rotation covariance
- OSC forward and inverse marker mapping

### Optics

- tangential momentum conservation
- dispersion relation
- correct propagating direction
- decaying evanescent branch
- `n -> 1` limit
- scalar interface coefficient limits
- uniform-depth attenuation limit at zero decay and large thickness

### Mosaic and Ewald

- probability normalization
- azimuthal periodicity
- zero-width limit
- pole behavior
- tangent and no-root status
- Ewald residual
- integrated event mass convergence

### Ordered and reflectivity

- crystallographic symmetry and occupancy
- systematic absences from amplitude
- general-cell reciprocal basis
- distinct rods with shared family metadata
- finite-stack limiting cases
- Parratt Fresnel and single-interface limits
- composite equality to declared branches outside the blend window

### Stacking

- direct sequence enumeration for small `N`
- full six-state versus exact reduced result
- `N=1`
- deterministic parent limits
- nonnegative real ensemble intensity
- `h=k=0`, `F+=F-` Laue limit
- finite total versus per-layer normalization

## Cross-branch review gates

Before merging, an automated review checks:

```text
contract API version
trace schema version
legacy-pack hash
units and frames
array shapes
source and event ID preservation
amplitude versus intensity declarations
probability density versus probability mass
model versions
owned-path compliance
mandatory proof records
```

A read-only scientific review then checks that:

- no factor is omitted or applied twice
- the same shared equation is not reimplemented inconsistently
- material optics and complex-wave branches agree
- rod and family identities agree
- event `Qz`, `L`, and wavelength semantics agree
- ordered and stacking return interchangeable event-aligned intensities
- outgoing film wavevectors can be consumed by geometry
- OSC and simulation coordinates meet at exactly one boundary

## Integration sequence

Integrate through vertical slices:

1. synthetic source to synthetic incident state
2. ordered rod catalog to mosaic with synthetic incident states
3. geometry incident states to mosaic event generation
4. event-aligned queries to ordered intensities
5. event outgoing wavevectors to exit refraction and continuous detector hits
6. hits to integration-owned detector deposition
7. substitute stacking strength for ordered strength under the same contract
8. run the tiny end-to-end detector case

## Later fitting proof

Each fit stage must first recover parameters from synthetic data generated by the accepted forward model.

- Source fit: recover size, divergence, and declared correlations from multiple-distance direct-beam data.
- Detector fit: recover pose and beam center from calibrant or independently constrained observations.
- Sample geometry fit: recover sample/goniometer parameters from frozen peak associations and pass an outer re-index audit.
- Mosaic fit: recover core width, tail width, and mixture from fixed profiles with source and geometry frozen.
- Ordered intensity fit: recover structural parameters and image scales from fixed selected regions.
- Stacking fit: recover transition parameters from fixed rods and branches with upstream states frozen.

Measured-data improvement is not proof without synthetic recovery and parameter-identifiability diagnostics.

## Tolerance freeze and proof sensitivity

Before T02--T05 compare with the shared pack, they load `proof/stage_tolerances_v1.json` through the strict loader and record canonical SHA-256 `d3739963a8decf481fc7ec87723854ef7628e8da02dbcb3e6f7e5bb41522b4b3`. Tolerances may change only through reviewed proof-base work, never after seeing branch error. Exact calculations do not invent convergence; broad sweeps, mutations, and benchmarks remain one-shot evidence rather than permanent frameworks.

A branch is not proven when the fixture is insensitive to likely mistakes. Run the workstream mutations from [ERROR_INJECTION.md](ERROR_INJECTION.md) and record the expected and observed first failing stage. The integration proof runs the factor-omission, factor-duplication, event-identity, OSC-orientation, and row-column mutations.
