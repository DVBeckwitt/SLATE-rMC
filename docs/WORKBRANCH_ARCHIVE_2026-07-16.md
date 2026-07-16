# Workbranch archive — 2026-07-16

This file records the short-lived SLATE-rMC workbranches that were reviewed before retirement.
It preserves their purpose, files, method, numerical status, and limitations without retaining
their worktrees. A recorded proof result is not the same as merged code: unless stated otherwise,
the retired branch commit was not merged into `main`.

The initial proof batch used baseline `caf7acd649a27dc66c6c0b73a2f66dcd520389f9` and was
retired when `main` was `46f5b75b8b0a55f1445177313eb1357fd3f357e8`. The later PbI2
polytype proof used baseline `7bccbd328220345b1a62a588d3b418bf82c1f0a9`.

## Lifecycle summary

| Workbranch | Tip or state | Final status | Disposition |
|---|---|---|---|
| `codex/mosaic-equivalence` | `97a33743bc88dfd2703f00ff975492bdce4efdec` | Accepted proof; code/tests not merged | Already deleted; conclusions retained on `main` |
| `codex/sf-equivalence` | `971317426071ca7985483354f8281e309539caec` | Accepted proof; compatibility code/tests not merged | Already deleted; conclusions retained on `main` |
| `codex/parratt-proof` | `6db62f766c17a62924ddf1e7b1478d7f1122d83f` | Clean `READY` proof commit; not merged or pushed | Retired and deleted |
| `codex/stacking-disorder-parity` | `e831c67cba8a8b7662991e9c7433f183f02e36c3` | Clean proof commit; handoff omitted explicit final-gate/`READY` statement | Retired and deleted |
| `codex/pbi2-polytype-sf-proof` | `a652fc315f107f038f4de284044e63c1452dd9d8` | Clean `READY`; no production change; one test assertion not merged | Retired and deleted after proof archival |
| `fix/complete-degenerate-rod-families` | branch at `caf7acd`; dirty working copy | Unfinished, unverified prototype | Working copy discarded and branch deleted |
| `codex/ray-geometry-lab` | branch at `caf7acd`; no commit | Useful disposable lab run, no durable branch result | Residual working copy and branch deleted |
| `feat/hbn-ring-fitter` | `18c6da69067aa05d9814f4a5f0281c3cc27e89a2` | Clean and accepted by the project owner | **Protected; retained untouched** |
| stitch worktree `379c` | clean detached `46f5b75` | Active scope/design task | **Protected; retained untouched** |

`main` is retained. Its unrelated pre-existing working changes were not staged or modified by this
cleanup.

## Mosaic cap/ring equivalence

### Goal and files

Prove that the existing wrapped mosaic density uses one recentered line shape for polar caps and
off-axis rings, assigns the correct topology, and conserves intensity under the declared spherical
measure. The branch changed three files (`+285/-6`):

- `src/rasim_next/reciprocal/proof.py` (`+197/-2`)
- `tasks/03_mosaic_ewald.md` (`+39/-4`)
- `tests/test_mosaic_ewald.py` (`+49/-0`)

### Method and result

The proof evaluated `(003)` as the `m=0` cap and `(100)`/`(103)` as `m!=0` rings for Gaussian,
mixed, and Lorentzian profiles. It recentered by the reciprocal-metric Bragg polar angle,
peak-normalized only for shape comparison, and integrated the raw probability density with
`G^2 sin(theta) dtheta dphi` for `I0=1` and `I0=7.25`. Four mutations exercised width mismatch,
omitted spherical measure, raw surface summation, and reversed topology.

The read-only external shape comparator was `2D_Mosaic_Sim` commit
`5efb3233d60843f3fd4e0e3b5b73536f05c035e8`; it was independently evaluated and never imported by
production or permanent tests.

- Maximum cap/ring shape error: `1.9206858326015208e-14`.
- Maximum collapsed-intensity error: `1.7763568394002505e-15`.
- All four new mutations and all seven existing T03 controls were detected.
- Pure-Gaussian peak shape: `MATCH`.
- Mixed and Lorentzian shapes: `CORRECTED` first at `mosaic.wrapped_line_density`, because the
  external program mixes unit-peak components and uses an unwrapped tail rather than the project's
  wrapped probability-mass semantics.
- Existing Ewald intersection evidence remains `MATCH`; seeded LHS source and continuous-rod
  candidates remain `NO_ORACLE` outside analytic/dense-oracle proof.

The branch and its added parameterized test were not merged. `main` commit `46f5b75` retains the
accepted narrative in `docs/VALIDATION.md`; main keeps the original seven T03 controls, not the four
disposable cap/ring controls. The audit did not newly compare detector projection with
`2D_Mosaic_Sim`.

## Ordered structure-factor equivalence

### Goal and files

Isolate the RA-SIM ordered-amplitude mismatch by holding the atomic-factor table equal while
leaving SLATE-rMC's accepted XrayDB/Chantler default unchanged. The branch changed four files
(`+168/-13`):

- `src/rasim_next/ordered/__init__.py` (`+2/-0`)
- `src/rasim_next/ordered/amplitudes.py` (`+92/-12`)
- `src/rasim_next/ordered/motifs.py` (`+4/-1`)
- `tests/test_ordered_reflectivity.py` (`+70/-0`)

### Method and result

Clean `ra_sim` commit `8fb1415e8e4695aa2ce8ec7f576b575264d4b328` ran in a separate process
using Dans_Diffraction 3.3.3. Raw complex electron-unit amplitudes were compared before squaring,
rounding, pruning, or normalization. The temporary branch model matched RA-SIM's ITC-1992 `f0`
and Henke anomalous data for neutral Bi/I/Pb/Se around Cu K-alpha; the project default remained
Waasmaier-Kirfel plus XrayDB/Chantler. PbI2's missing `Uiso` was supplied explicitly as zero, and
layer labels were aligned by physical orientation.

- `51,200` atomic factors and `70,400` whole-cell/layer amplitudes were checked from
  `7.92068` to `8.17898 keV`.
- Atomic factors matched exactly; maximum complex-amplitude error was `7.7276e-13 e`, below the
  `2e-12 e` bound.
- PbI2 2H and Bi2Se3 whole-cell amplitudes and both physical PbI2 layer orientations were `MATCH`
  when the factor table was equal.
- The default-to-default first divergence was the declared atomic-factor data source; phase sign,
  reciprocal coordinates, site expansion, occupancy, and displacement handling agreed.

The narrow compatibility implementation and its two regressions were deliberately not merged.
Only the conclusion survives in `docs/VALIDATION.md` at `46f5b75`. That retired branch alone was
scoped to ordered PbI2/Bi2Se3; the later polytype proof below establishes the separate
stacking-disorder, 4H/6H, and mixture claims.

## Parratt reflectivity proof

### Goal and files

Complete the T04 Parratt proof without changing production physics or public APIs. Commit
`6db62f7` changed three files (`+146/-19`):

- `src/rasim_next/ordered/proof.py` (`+90/-16`)
- `tasks/04_ordered_reflectivity.md` (`+54/-2`)
- `tests/test_ordered_reflectivity.py` (`+2/-1`)

It used `reference/rasim_reference_v1.npz`, the Bi2Se3 VESTA CIF, shared production/tolerance code,
and live RA-SIM `8fb1415` (`ra_sim/utils/calculations.py`). A scalar recursion independent of the
public Parratt routine supplied stage-level evidence.

### Method and result

- All 257 immutable points passed; maximum `R` error was `2.567342172188347e-12`.
- On 323 well-conditioned GUI-default points, maximum errors were `4.129628320646766e-14 A^-1`
  in layer `kz`, `1.5077277754161236e-12` in interface amplitude,
  `1.588900070922836e-12` in recursion amplitude, and `4.423926502905573e-13` in `R`.
- Canonical and GUI-default 600-point legacy curves matched within `4.353628568765089e-12` and
  `1.059091703226045e-11` absolute error, respectively.
- Ultra-low `Q=1e-8 A^-1` was `CORRECTED` first at `reflectivity.layer_kz`: legacy cancellation
  gives `R=1`; target `R=0.9999999675481261` agrees with the 80-digit oracle.
- A wrong film square-root branch failed first at `reflectivity.layer_kz` with error
  `1.7545286931907171` versus limit `2.1376818629568278e-13`.
- Production took `0.000512249 s/call` and `120250 B` peak traced memory for 257 points; the scalar
  oracle took `0.011484484 s/call` and `155192 B`.

Only one low-Q row/assertion was added to the existing scalar-recursion test. Compile, Ruff, six
focused tests, ordered/reference proofs, and diff checks were recorded passing. The exact lossless
film critical point still raises at a removable `0/0`; its analytic limit is
`R=0.9671418660002263369`, while legacy `R=1` is also wrong. The branch was clean and independently
approved, but was not merged or pushed. Its ignored cache/bytecode residue was removed with the
worktree.

## Stacking-disorder parity

### Goal and files

Prove epsilon as transition-probability mass for every parent while preserving the existing finite
stacking implementation and archiving external PbI2/Bi2Se3 parity. Commit `e831c67` changed only:

- `src/rasim_next/stacking/proof.py` (`+28/-1`)
- `tasks/05_stacking_transition.md` (`+20/-3`)

No production solver, material adapter, test, dependency, or ordered file changed. The external
oracle was `ra_sim/utils/stacking_fault.py` at `8fb1415`; its file SHA-256 was
`8cb903e191318975223161172812b3c3e94738fde4e5a2b13e6af7245880050f`.

### Method and result

The compact proof compared direct sequence enumeration, direct pair sums, full six-state moments,
and the reduced recurrence; retained `N=1`, Laue, near-extinction, deterministic 2H/4H±/6H±,
rich-epsilon/reduced-ABD, alignment, measure, normalization, and population checks. Twenty new
frozen cases cover five parents across four epsilon values: parent mass is `1-epsilon`, the other
four channels sum to `epsilon`, and each is `epsilon/4`. Seven assigned mutations were detected.

After matching legacy `AREA`, finite-per-layer normalization, first-layer convention, registry
phase, and vertical divisor:

- 12 PbI2 parent/epsilon curves agreed within `1.134e-12 electron^2/layer`.
- Default 2H at epsilon `0.01` agreed within `2.203e-13`.
- Binary/ternary incoherent mixtures agreed within `3.376e-14`.
- A disposable 63-event Bi2Se3 `legacy_2H` sweep agreed with RA-SIM within `6.003e-10` and with an
  analytic geometric sum within `8.732e-11 electron^2/layer`.

Transition/synthetic finite cases were `MATCH`; explicit initial population, normalization, and
registry-phase differences were `CORRECTED` at their named stages; stationary output remained
`NO_ORACLE`. No permanent test changed. The clean commit was not merged or pushed. Although every
execution-plan step was checked, its handoff omitted an explicit final `READY` line and final gate
command list. A canonical Bi2Se3 quintuple-layer adapter remains the minimum integration request.

## PbI2 2H/4H/6H structure-factor and stacking proof

### Goal and retained change

The later `codex/pbi2-polytype-sf-proof` workbranch proved the PbI2 path from the files historically
used for 2H, 4H, and 6H without conflating three observables:

- `F_cell`: the raw complex whole-cell amplitude native to one CIF;
- physical `F_plus` and `F_minus`: the registry-free amplitudes of one Pb-centered I--Pb--I layer,
  both derived from the tracked 2H CIF; and
- `I_stack`: deterministic or disordered finite-stack intensity assembled from that 2H layer pair.

The broad oracle, sweeps, traces, and mutation harness were disposable and were removed. Commit
`a652fc315f107f038f4de284044e63c1452dd9d8` retained exactly one assertion in
`tests/test_ordered_reflectivity.py`: at signed reciprocal events, the physical negative-orientation
amplitude at `+L` equals the positive-orientation amplitude at `-L`. No production module, public
API, dependency, CLI, or example changed. The test-only commit was not merged into `main`; this
archive retains its conclusion before branch retirement.

### Proof construction and numerical evidence

The layer proof used `examples/pbi2/structures/PbI2_2H.cif` as the single motif authority, matching
the historical stacking path. A direct three-atom scalar sum was compared with
`pbi2_layer_amplitudes` at identical Cartesian `Q`, wavelength, occupancy, displacement, and atomic
factors. The maximum complex error was `2.4457182613372133e-14 e` against a
`3.3244476461733e-10 e` limit; signed-reflection error was exactly zero and the largest Laue error
was `1.0658141036401503e-14 e`. Legacy layer names are reversed relative to the manuscript names,
so comparisons aligned physical orientations rather than the strings `plus` and `minus`.

With the retired ITC-1992/Henke factor table, the exact legacy anchors were:

- whole 2H cell: `72.03163488540059 + 10.486067181091277i e`;
- physical `F_plus`: `62.02309184119507 + 7.503644555681163i e`; and
- physical `F_minus`: `-4.469134181424458 - 2.2961006486888493i e`.

The 2H, 4H, and 6H CIFs were also checked independently as whole-cell fixtures. Their exact
site/motif counts were `3/1`, `6/2`, and `9/3`, with layer repeats `c`, `c/2`, and `c/3`.
Twelve file-native direct atom sums matched `unit_cell_amplitude` with maximum complex error
`1.563597684016243e-13 e` against `1.0039447668149997e-9 e`; integer-image gauge error was
`1.4228607195221903e-13 e`. The complete structure-factor table has SHA-256
`6379e6fa3ed3e3ab9da97fc21f9b4d3b07a15e9d8f7f2f34cb41925d96247f8e`.

Five 2H-derived ideal parents (`2H`, both `4H` hands, and both `6H` hands), three period multiples,
and four physical-Q events gave 60 direct-sequence comparisons. Maximum amplitude error was
`1.2397319609735734e-12 e` against `3.9891680662876425e-8 e`; maximum intensity error was
`1.3460521586239338e-10 e^2` against `8.748511945050639e-4 e^2`. The strict `N=50` legacy
finite-per-layer path matched after removing legacy `AREA`, with maximum error
`3.2862601528904634e-14 e^2/layer` against `2.137399519846063e-9`.

The generalized disorder oracle then covered 15 component cases--each of 2H, 4H, and 6H at
`epsilon` in `{0, 0.01, 0.1, 0.5, 0.99}`--and 18 convex binary/ternary mixtures. Both the largest
component error and largest mixture error were `1.8917489796876907e-10 e^2` against
`7.322474510325397e-3 e^2`. Transition-mass error was `2.220446049250313e-16`; linearity and
zero-weight errors were zero; permutation error was `1.4551915228366852e-11 e^2`; invalid weights
were rejected. The aggregate evidence SHA-256 is
`0da8be40686da52c179d8febf1791d332bcb143f12517713218e332ddf028b7f`.

### Classification, gates, and limits

- `MATCH`: direct CIF sums, the 2H physical layer pair after label alignment, ideal-parent sequence
  sums, finite legacy curves after declared convention alignment, epsilon mass, and incoherent
  population mixing.
- `CORRECTED`: current Waasmaier--Kirfel/XrayDB-Chantler factors intentionally first diverge from
  ITC-1992/Henke at `ordered.atomic_amplitude`; the accepted `h+2k` registry convention first
  diverges from the legacy convention at `stacking.registry_phase`.
- `NO_ORACLE`: relaxed, file-native 4H/6H whole cells are not required to equal ideal 4H/6H parents
  homogenized from the 2H motif. Those are different structural models; native `F_cell` must not be
  substituted for 2H-derived `F_plus` or `F_minus`.

The legacy default 2H `epsilon=0.01` witness is symmetry-insensitive because its four alternative
channels are equal, which is why the asymmetric multi-parent sweep is required. The arbitrary
epsilon cases are independent full-six-state/direct-oracle evidence, not claimed captured legacy
outputs. Existing seven stacking controls plus reversed layer labels, registry omission or
conjugation, wrong repeat, nominal-`L` comparison, implicit `Uiso`, mixed factor tables, maximum
normalization, and coherent population mixing all failed at their named stages.

All 29 tests, focused formatting and lint, compile, ordered, stacking, and reference proof gates
passed. Full/reduced equivalent work agreed within `4.263256414560601e-14 e^2`; the reduced path
was `5.75x` faster (`0.01193 s` versus `0.06863 s`) and used `19,758 B` versus `33,880 B` peak
traced memory. The known stale seed-manifest warning and repository-wide Ruff findings inside the
immutable legacy snapshot were unchanged baseline exceptions; scoped gates passed.

## Complete degenerate rod families prototype

The branch ref never moved from `caf7acd`; there was no unique commit. Its dirty working copy had
three modified files (`+88/-30`):

- `src/rasim_next/ordered/bi2se3_proof.py` (`+5/-2`)
- `src/rasim_next/reciprocal/rods.py` (`+33/-21`)
- `tests/test_ordered_reflectivity.py` (`+50/-7`)

The prototype replaced rectangular `h_bounds`/`k_bounds` with a physical
`maximum_qr_Ainv` cutoff. It validated a finite nonnegative scalar, used the smallest eigenvalue of
the in-plane reciprocal metric to construct an exhaustive integer envelope, evaluated candidate
`q^2` values with `einsum`, and included boundary rods with a 64-epsilon tolerance. The strengthened
Bi2Se3 test expected all 13 rods through hexagonal `m=3`, all six degenerate `m=3` members, distinct
rod identities, and no `m=4` family.

No durable tests, lint, proof, benchmark, review, handoff, or commit existed. The public API change
also lacked invalid-cutoff, skew/nonhex boundary, and candidate-growth evidence. This was an
unfinished prototype; its uncommitted working copy was intentionally discarded during retirement.

## Ray geometry lab

The local branch never moved from `caf7acd` and therefore contains no committed ray-lab files or
proof. Its former worktree registration had already been removed. The residual working copy showed
the disposable implementation in:

- `README.md`
- `src/rasim_next/geometry/ray_lab.py`
- `src/rasim_next/ray_geometry_lab/{__init__.py,__main__.py,server.py}`
- `src/rasim_next/ray_geometry_lab/static/{app.js,index.html,lab.css}`
- `tests/test_geometry_optics.py`

It implemented an isolated loopback browser lab based on the old Specular Geometry view, using the
project's names and shared rigid instrument/sample/detector geometry. The final footprint reported
about 783 production lines and 96 added test lines. Development evidence included 33 passing
repository tests, all 24 controls rendered, responsive browser checks, an independent analytic
full detector-frame oracle for nonzero `gamma/Gamma`, exact signed `zS` translation, and HTTP/1.0
request handling. Final repository gates and a clean commit were never completed before the task
was stopped. The branch result was therefore disposable evidence, not a merge candidate.

## Protected hBN ring fitter

Branch `feat/hbn-ring-fitter` is intentionally retained at `18c6da6` and was not modified during
cleanup. It is one commit above `caf7acd`, adding an isolated GPL nested project that preserves the
legacy GUI and `hbn-fit` CLI around the `hbn_ellipse_bundle.npz` boundary.

Changed files versus `caf7acd` were `.gitattributes`, `README.md`, `RUN_HBN_RING_FITTER.bat`,
`docs/HBN_RING_FITTER_SIMPLIFICATION_SPEC.md`, `tools/check_docs.py`, and these nested-project files:

- `tools/hbn-ring-fitter/{LICENSE,README.md,pyproject.toml,uv.lock}`
- `tools/hbn-ring-fitter/src/ra_sim/{SOURCE_PROVENANCE.toml,__init__.py,__main__.py,hbn.py,hbn_cli.py,hbn_geometry.py}`
- `tools/hbn-ring-fitter/src/ra_sim/config/{__init__.py,hbn_paths.example.yaml,instrument.yaml,loader.py}`
- `tools/hbn-ring-fitter/src/ra_sim/hbn_fitter/{__init__.py,fitter.py}`
- `tools/hbn-ring-fitter/src/ra_sim/io/{__init__.py,osc_reader.py}`
- `tools/hbn-ring-fitter/src/ra_sim/utils/{__init__.py,calculations.py}`
- `tools/hbn-ring-fitter/tests/test_hbn_ring_fitter.py`

The final simplification reduced production Python from 6,984 to 6,648 lines (`-336`), limited the
permanent test delta to `+116` lines, removed both temporary task files and `user_paths.py`, added no
production file/dependency during simplification, and corrected source provenance. Reported final
gates were 11 hBN tests, 29 repository tests, references, core, docs, and Ruff, all passing. The
worktree was tracked/untracked clean. It remains standalone, requires a future importer into
`rasim_next`, retains GPL-3.0-only/legacy-namespace constraints, and uses its pinned Python/Tkinter
stack. No cleanup action, including ignored-cache removal, was performed there.

## Protected stitch worktree

Worktree `C:/Users/Kenpo/.codex/worktrees/379c/SLATE-rMC` was clean and detached at `46f5b75` when
this archive was written. Its active task is defining the smallest 20-ray Bi2Se3 stitching slice:
RA-SIM-default source position/wavelength/divergence, mosaic sampling, `m=0` versus two-branch
`m!=0` handling, degenerate branch accounting, 40 intersection samples per ray, detector output,
and recovery of the nominal structure-factor total across branches. It was still in scope/design
state with no Git changes. It was not moved, edited, archived, or deleted.
