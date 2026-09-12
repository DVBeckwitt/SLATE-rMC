# Physics ledger

Treatments:

- `MATCH`: reproduce original-RASIM results where they satisfy the new specification
- `CORRECTED`: capture the original result, then use a better result with first-divergence proof
- `NEW`: no adequate original result, prove independently
- `DEFERRED`: intentionally outside current scope

Owners are `bootstrap`, `characterization`, `geometry`, `mosaic`, `ordered`, `stacking`, `integration`, `analysis`, `selection`, or `fitting`.

## I/O, coordinates, and source

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-IO-001 | OSC signature, endian, dimensions, payload | `ra_sim/io/osc_reader.py:18-60` | n/a | geometry | MATCH |
| PHY-IO-002 | OSC high-range pixels | `ra_sim/io/osc_reader.py:61-65` | n/a | geometry | MATCH |
| PHY-IO-003 | 90-degree measured-image conversion | `runtime_session.py:558-565`; `gui/background.py` | experimental convention | bootstrap/geometry | CORRECTED, centralized |
| PHY-IO-004 | raw OSC versus detector indices | distributed | n/a | bootstrap | CORRECTED |
| PHY-IO-005 | continuous detector coordinate | `diffraction.py:2302-2306` | geometry figures | bootstrap/geometry | CORRECTED |
| PHY-SRC-001 | spatial beam distribution | `simulation/mosaic_profiles.py:15-63` | Methods line 18 | mosaic | CORRECTED, seeded empirical; optional conditional Gaussian pixel integration is NEW / NO_ORACLE in the legacy pack, with explicit tail cutoff and source correlations |
| PHY-SRC-002 | divergence distribution | same | Methods line 18 | mosaic | CORRECTED, normalized |
| PHY-SRC-003 | wavelength distribution | same and GUI bandwidth settings | `eq:detector_sum_lambda_main` | mosaic | CORRECTED active; either the legacy Gaussian law or a declared weighted discrete-Gaussian line mixture |
| PHY-SRC-004 | independent wavelength intensity sum | downstream sample loop | `eq:detector_sum_lambda_main` | mosaic/integration | MATCH |
| PHY-SRC-005 | stable sample identity and weights | distributed runtime arrays | n/a | bootstrap/mosaic | NEW |
| PHY-SRC-006 | joint source-variable correlations | separate old arrays imply factorization | source phase-space definition | mosaic | NEW active; a declared nonzero correlation is moment-matched to the full four-dimensional position/divergence covariance when every line has at least eight rows; zero-correlation LHS grids and smaller correlated grids retain finite-quadrature semantics, while equal per-line grids keep geometry independent of line identity |
| PHY-SRC-007 | endpoint-safe independent Gaussian antithetic N-stratum realization | prior pair-stratified sampler | source phase-space definition | mosaic | CORRECTED v2; first divergence `geometry.lab_ray` |
| PHY-SRC-008 | exactly weighted discrete source-line mixture | GUI bandwidth scalar | `eq:detector_sum_lambda_main` | mosaic/integration | CORRECTED active; line masses equal their declared probabilities, equal per-line row counts use the same geometry grid, geometry is conditionally moment-matched within each line when rank permits, unequal finite row counts are explicitly labeled an approximation, and line contributions sum incoherently |

## Geometry

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-GEO-001 | rigid frame composition | `diffraction.py:1682-1742` | Methods geometry figures | geometry | CORRECTED |
| PHY-GEO-002 | sample pivot and offsets | same | sample geometry figure | geometry | CORRECTED |
| PHY-GEO-003 | remove `P0_rot[0]=0` | `diffraction.py:1740-1741` | none | geometry | CORRECTED |
| PHY-GEO-004 | detector plane and basis | `diffraction.py`; `intersection_analysis.py:80-122` | detector geometry figure | geometry | CORRECTED, single source |
| PHY-GEO-005 | unique ray-plane intersection; reject parallel and coplanar rays | `diffraction.py:285-325` | geometric construction | geometry | CORRECTED for coplanar; first divergence `geometry.sample_intersection` |
| PHY-GEO-006 | sample footprint clipping | `diffraction.py:1807-1874` | Methods line 18; SI line 525 | geometry | MATCH as accepted beam mass |
| PHY-GEO-007 | internal-to-lab outgoing vector | `diffraction.py:2260-2264` | Methods | geometry | CORRECTED |
| PHY-GEO-008 | front-facing detector intersection | `diffraction.py:2266-2299` | Methods | geometry | CORRECTED, signed active-face validity after frame correction |
| PHY-GEO-009 | forward ray-to-pixel | `diffraction.py:2294-2306` | Methods | geometry | CORRECTED typed coordinates |
| PHY-GEO-010 | inverse pixel-to-ray | `intersection_analysis.py:506-615` | n/a | geometry | CORRECTED shared inverse |
| PHY-GEO-011 | rectangular detector and anisotropic pitch | old core assumes square | n/a | geometry | NEW |
| PHY-GEO-012 | global rigid-rotation covariance | no explicit proof | geometric invariant | geometry | NEW |

## Material optics and interfaces

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-MAT-001 | CIF composition and density | `utils/calculations.py:169-227` | `eq:delta_beta_lambda_main` | ordered | CORRECTED |
| PHY-MAT-002 | wavelength-dependent `f1`, absorptive `f2` | mixed structure-factor paths | `eq:structure_factor` | ordered | CORRECTED |
| PHY-MAT-003 | refractive index from one consistent data source | `calculations.py:228-304` | `eq:n_complex_lambda_main` | ordered | CORRECTED |
| PHY-MAT-004 | anomalous-sign conversion and optical theorem | mixed sign helpers | n/a | ordered | NEW |
| PHY-OPT-001 | tangential-wavevector conservation | `diffraction.py:1917-1928` | SI refraction equations | geometry | CORRECTED vector form |
| PHY-OPT-002 | complex normal mode and branch | `diffraction.py:339-344`; `calculations.py:328-347` | `eq:si_ktz_solution_lambda` | geometry | CORRECTED shared representation |
| PHY-OPT-003 | entrance refraction | `diffraction.py:1917-1928` | SI refraction equations | geometry | CORRECTED |
| PHY-OPT-004 | exit refraction | `diffraction.py:415-436`, `2223-2241` | `eq:si_exit_af_lambda` to `eq:si_exit_angle_lambda` | geometry | CORRECTED |
| PHY-OPT-005 | scalar entrance and exit field amplitudes | old power average at `1930-1936`, `2242-2248` | `eq:si_scalar_transmission`, `eq:si_entry_exit_transmission` | geometry | CORRECTED |
| PHY-OPT-006 | propagating and evanescent validity | same | `eq:si_kappa_if_lambda` | geometry | CORRECTED |
| PHY-OPT-007 | incident and exit decay constants from complex normal wavevectors | post-intensity attenuation at `2250-2256` | `eq:si_kappa_if_lambda`, `eq:si_imkz_pathlength_weight` | geometry | CORRECTED explicit |
| PHY-OPT-008 | uniform-depth attenuation average | old path applies full thickness separately to entrance and exit | `eq:si_abs_complex_kz` | geometry | CORRECTED, first off-specular reference |
| PHY-OPT-009 | reciprocity and lossless energy checks | no complete proof | interface physics | geometry | NEW |
| PHY-OPT-010 | phase wavevector versus decay component | old paths mix real `kz` geometry with complex attenuation implicitly | absorbing-wave theory | geometry/mosaic | CORRECTED explicit |
| PHY-OPT-011 | multilayer reciprocal exit-field normalization | no complete old equivalent | future distorted-wave model | none | DEFERRED |
| PHY-OPT-012 | external sample-to-detector Beer--Lambert intensity attenuation | absent | detector-path attenuation | geometry/integration | NEW optional; one scalar or an exact wavelength-keyed table supplies `exp(-mu(lambda) ell_ext)` once, unlisted wavelengths fail closed, and zero coefficient is exactly unity |

## Mosaic and reciprocal events

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-MOS-001 | wrapped Gaussian tilt density | old unwrapped code `diffraction.py:245-271` | `eq:mosaic_two_component_maintext`; SI lines 241-263 | mosaic | CORRECTED |
| PHY-MOS-002 | wrapped Lorentzian tail | same | same | mosaic | CORRECTED |
| PHY-MOS-003 | independent widths and mixture | old pseudo-Voigt arguments | manuscript explicitly independent | mosaic | CORRECTED |
| PHY-MOS-004 | spherical orientation probability measure | old factor removed at `268-271` | SI orientation density | mosaic | CORRECTED |
| PHY-MOS-005 | random in-plane powder azimuth | circle construction and rod groups | Methods lines 64-78 | mosaic | CORRECTED explicit measure |
| PHY-REC-001 | reciprocal basis and general in-plane metric | several hexagonal helpers | Methods lines 64-78 | mosaic/ordered | CORRECTED |
| PHY-REC-002 | elastic Ewald residual | `diffraction.py:1524-1668` | Methods lines 6-16 | mosaic | MATCH equation |
| PHY-REC-003 | Bragg-sphere circle construction | same | Methods line 16 | mosaic | MATCH as reference option |
| PHY-REC-004 | continuous rod/Ewald roots | old code discretizes/uses spheres | rod model in Methods | mosaic | NEW recommended |
| PHY-REC-005 | tangent and no-root status | `solve_q` statuses | elastic geometry | mosaic | CORRECTED |
| PHY-REC-006 | continuous per-rod mosaic/population/strength density | implicit `I_Q` | SI line 525 | mosaic | CORRECTED explicit latent measure |
| PHY-REC-007 | analytic Ewald roots, support, coarea, and convergence | old uniform/adaptive scan | numerical requirement | mosaic/integration | CORRECTED analytic |
| PHY-REC-008 | complete-pool inverse-CDF event selection | event resampling paths | n/a | none | RETIRED by contract-v9 continuous pushforward |
| PHY-REC-009 | external versus internal Q | inconsistent helper paths | refraction section | bootstrap/mosaic | CORRECTED |
| PHY-REC-011 | exact intrinsic Ewald solid-angle inverse pushforward | absent; legacy paints sampled intersections | continuous change of variables | mosaic/integration | NO_ORACLE; NEW API-v12 `k_film^2 / |J_latent|`, all regular non-specular preimages, no repeated coarea |
| PHY-REC-012 | detector-visible `m=0` field | absent | continuous change of variables plus canonical detector support | mosaic/integration | NO_ORACLE; plain kinematic mode has a positive Q-gap, while optional local-lamella Parratt--kinematic mode owns all nonzero Q and declares gap zero; direct `Q=0` remains excluded |

## Ordered structure and rods

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-ORD-001 | CIF parsing and symmetry | motif and GUI structure paths | `eq:structure_factor` | ordered | CORRECTED, one parser |
| PHY-ORD-002 | special positions and multiplicity | structure expansion paths | `eq:structure_factor` | ordered | NEW proof |
| PHY-ORD-003 | occupancy | motif path; ignored in some diffuse paths | `eq:structure_factor` | ordered | CORRECTED mandatory |
| PHY-ORD-004 | general direct and reciprocal cell | motif helpers | Methods lines 64-83 | ordered | CORRECTED |
| PHY-ORD-005 | atomic `f0(Q)` | structure-factor modules | `eq:structure_factor` | ordered | MATCH after data decision |
| PHY-ORD-006 | anomalous `f' + i f''` | VESTA/package modes | `eq:structure_factor` | ordered | CORRECTED |
| PHY-ORD-007 | isotropic displacement | motif form factor | `eq:structure_factor` | ordered | MATCH |
| PHY-ORD-008 | anisotropic `Uij` | incomplete old support | `eq:structure_factor` | ordered | PARTIAL active shared and site-resolved transverse-isotropic `Ur/Uz` profiles; general per-site `Uij` DEFERRED |
| PHY-ORD-009 | complex structure amplitude | `motif_form_factor.py:473-543` | `eq:structure_factor` | ordered | MATCH |
| PHY-ORD-010 | remove normalization to 100 and rounding | `diffraction_tools.py:135-207` | none | ordered | CORRECTED |
| PHY-ORD-011 | distinct `(h,k)` rods with family metadata | Miller/grouping paths | Methods lines 64-78 | ordered | CORRECTED |
| PHY-ORD-012 | continuous arbitrary-Qz rod amplitude | old cached grids | rod construction | ordered | CORRECTED |
| PHY-ORD-013 | finite ordered stack | structure and stacking helpers | finite thickness discussion | ordered | MATCH/analytic |
| PHY-ORD-014 | arbitrary complex depth-field weights in ordered amplitudes | no exact old equivalent | future distorted-wave model | none | DEFERRED |
| PHY-ORD-015 | systematic absences from amplitude | old pruning/generation | crystallographic invariant | ordered | NEW proof |
| PHY-ORD-016 | global anisotropic event-intensity envelope `exp(-U_r Q_r^2-U_z Q_z^2)` in fixed sample Q after mosaic rotation, distinct from crystallite-local site ADPs inside amplitude | `diffraction.py:2314-2315` | not derived | detector integration | CORRECTED, off by default |

## Reflectivity

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-REF-001 | general multilayer normal modes | `calculations.py:328-480` | `eq:si_parratt_kz` | ordered | CORRECTED shared mode |
| PHY-REF-002 | bottom-up Parratt recursion | `calculations.py:403-429` | `eq:si_parratt_recursion` | ordered | MATCH |
| PHY-REF-003 | interface roughness | `calculations.py:419-423` | `eq:si_parratt_roughness` | ordered | MATCH convention |
| PHY-REF-004 | substrate and finite film | `calculations.py:432-480` | Parratt section | ordered | MATCH/generalized |
| PHY-REF-005 | external Qz, internal phase Qz, and L | `calculations.py:462-478` | `eq:si_internal_phase_coordinate` | ordered | CORRECTED named |
| PHY-REF-006 | pure kinematic specular rod | helper internals | `eq:si_ht_normalized_structure` | ordered | CORRECTED raw |
| PHY-REF-007 | empirical smooth handoff inside the continuous `(0,0)` strength | `calculations.py:591-803` | `eq:si_handoff_x_l`, `eq:si_handoff_unscaled_structure`, `eq:si_handoff_scale`, `eq:si_handoff_scaled_structure`, `eq:si_handoff_log_ratio`, `eq:si_handoff_smoothstep`, and `eq:si_handoff_blend` | ordered/detector | MATCH as named compatibility; the local-lamella field converts Parratt reflectivity to finite-stack `A2`, evaluates that strength at internal phase `L`, and recovers the kinematic high branch exactly |
| PHY-REF-008 | scalar multilayer incident and reciprocal-exit field profiles | UI says DWBA but event path is single-pass | future distorted-wave model | none | DEFERRED |
| PHY-REF-009 | amplitude-level optical weighting inside ordered and stacking sums | no old equivalent | future distorted-wave model | none | DEFERRED |
| PHY-REF-010 | single-pass scalar entrance/exit field weighting | `diffraction.py:1917-1946`, `2223-2316` | `eq:si_scalar_transmission` through `eq:si_full_optical_weight_lambda` | geometry/integration | CORRECTED, current reference |
| PHY-REF-011 | roughness-consistent off-specular local fields | Parratt roughness exists, event local fields do not | future distorted-wave model | none | DEFERRED |
| PHY-REF-012 | unified source-averaged local-lamella `(0,0)` detector density | no declared detector-measure equivalent | specular reflection change of variables | ordered/detector | NEW; actual sampled incident directions and wavelengths sum as intensities, the stitched strength is one function of internal phase `L`, nonzero rods are unchanged, and there is no separate detector curve, scale, shift, raster, or smoothing |

## Stacking disorder

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-STK-001 | `F+`, `F-` motif amplitudes | `motif_form_factor.py:559-574` | `eq:si_pbi2_Fplus`, `Fminus` | ordered/stacking | MATCH |
| PHY-STK-002 | registry phase `omega(h,k)` | stacking helpers | `eq:si_pbi2_omega` | stacking | MATCH |
| PHY-STK-003 | six-state matrix `T6` | stacking utilities | `eq:si_pbi2_T6` | stacking | MATCH |
| PHY-STK-004 | exact 2x2 Fourier block | `stacking_fault.py:287-401`; `polytype_stacking.py:208-374` | `eq:si_pbi2_Momega` | stacking | MATCH |
| PHY-STK-005 | orientation population `P` | same | `eq:si_pbi2_P` | stacking | MATCH |
| PHY-STK-006 | finite self and pair sums | same | `eq:si_pbi2_finite_intensity` | stacking | MATCH/direct |
| PHY-STK-007 | initial orientation and end effects | partially implicit | same | stacking | CORRECTED explicit |
| PHY-STK-008 | arbitrary complex depth-field weights in stacking correlations | no exact old equivalent | future distorted-wave model | none | DEFERRED |
| PHY-STK-009 | deterministic 2H, 4H, 6H limits | stacking utilities | `eq:si_pbi2_parent_vectors` | stacking | MATCH |
| PHY-STK-010 | parent-rich fault templates | stacking utilities | `eq:si_pbi2_fault_templates` | stacking | MATCH |
| PHY-STK-011 | incoherent population mixture | `polytype_stacking.py` | `eq:si_pbi2_total_intensity` | stacking | MATCH |
| PHY-STK-012 | `h=k=0` Laue limit | implicit | `eq:si_pbi2_m0_laue` | stacking | NEW analytic proof |
| PHY-STK-013 | normalization per layer versus total | mixed APIs | finite intensity equation | stacking | CORRECTED |
| PHY-STK-014 | infinite-stack HT path | old utilities | not required for finite manuscript result | stacking | DEFERRED unless fixture requires |

## Measurement and rendering

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-MEA-001 | continuous raw detector-coordinate density | absent | SI lines 523-527 | integration | NEW explicit `A2/px2` measure |
| PHY-MEA-002 | solid-angle correction for later caking/analysis | caking helper `exact_cake_portable.py:866-874` | SI lines 523-527 | analysis | CORRECTED, excluded from raw rendering |
| PHY-MEA-003 | scattering polarization distinct from Fresnel fields | no consistent native path | SI preprocessing and interface notes | integration | CORRECTED active; the unpolarized Thomson event factor `(1 + (ki_hat_air dot kf_hat_air)^2) / 2` is applied once, separately from Fresnel fields |
| PHY-MEA-003A | reciprocal event Jacobian versus separate Lorentz factor | implicit old `I_Q` and powder paths | SI event measure | mosaic/integration | CORRECTED, no duplicate factor |
| PHY-MEA-004 | deterministic detector pixel-box integration with grouped exact-`x` inverse-fold replacement | point/bilinear events | detector measurement | integration | CORRECTED with regular/fold convergence proof |
| PHY-MEA-005 | point/bilinear event deposition | diffraction accumulation helpers | numerical | none | RETIRED by pixel-box integration |
| PHY-MEA-005A | weighted forward Monte Carlo detector-pixel mass estimate | no accepted equivalent | latent pushforward and detector measurement | integration | NEW optional API-v11 estimator, execution-extended in API v12; exact hard bins, no retained events or calibrated counts; deterministic latent oracle required |
| PHY-MEA-006 | detector PSF/resolution | bilinear was not a PSF | ordered-results resolution discussion | none | DEFERRED normalized operator |
| PHY-MEA-007 | detector efficiency | not explicit | absolute-count requirement | none | DEFERRED unless calibrated |
| PHY-MEA-008 | masks, beamstop, saturation, bad pixels | GUI/data paths | experimental handling | none | DEFERRED from forward core |
| PHY-MEA-009 | empirical shared radial detector background | GUI/fitting paths | later comparison | fitting/measurement | CORRECTED active for matched-region fits; a declared scaled dark is applied first, then a rise-times-decay shape shared across OSCs with per-OSC amplitude/pedestal is fitted on background-only azimuth holdouts and frozen before SF |
| PHY-MEA-009A | declared scaled dark correction and covariance | ad hoc image subtraction | detector preprocessing | measurement/fitting | CORRECTED active; raw and dark are identically oriented and projected, signed `raw - s_dark dark` mass is preserved, and dark covariance scales as `s_dark^2`; the current Bi2Se3 run uses `s_dark=0`, `no_acquisition_matched_dark.v1`, and exactly zero shared-dark covariance |
| PHY-MEA-010 | multiple scattering and extinction | absent | not claimed | none | DEFERRED |
| PHY-MEA-011 | internal-film outgoing-direction Ewald density | no declared equivalent | Ewald surface measure | mosaic/integration | NO_ORACLE; NEW `intrinsic_ewald_direction_density_A2_per_sr.v1`; full sphere diagnostic, not detector solid angle |
| PHY-MEA-012 | configured detector-visible internal-film Ewald direction density | no declared equivalent | Ewald surface measure with named support | mosaic/integration | NO_ORACLE; NEW `detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1`; regular `m=0` included, detector optics/Jacobian absent |
| PHY-MEA-013 | flat-film illuminated incident-path weight | absent from the integrated-incidence observable | thin-film illuminated-volume measure | integration | CORRECTED active; `1/|direction_sample,z|` is applied once in source-phase construction, remains separate from finite footprint acceptance and the depth attenuation exponent, and first diverges in the typed source-phase weight before `measurement.total_detector_mass` |
| PHY-MEA-014 | normalized continuous-incidence probability average | post-hoc 5--20-degree scan | continuous acquisition measure | integration | CORRECTED active; acquisition-bound 5--25-degree support, calibrated deterministic nodes, original source/exposure masses, and no survivor renormalization |
| PHY-MAP-001A | continuous detector-density pullback to canonical `2theta/phi` with separate `S/N` | `exact_cake_portable.py` continuous geometry | coordinate-measure identity | analysis | MATCH coordinate convention, NEW generalized pose/J/`S/N` under T17; no finite caking claim |

Every non-deferred row must have one proof case or a documented reason that it is covered by a shared proof.

## Additional audited behavior

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-PHA-001 | multi-CIF phase mixture | `gui/controllers.py:425-443` | independent phase populations | integration | CORRECTED |
| PHY-PHA-002 | phase population semantics | same | intensity additivity | bootstrap/integration | NEW explicit |
| PHY-PHA-003 | remove per-phase and combined max normalization | same | none | integration | CORRECTED |
| PHY-PHA-004 | phase-mixture optical environment | primary-CIF refractive-index paths plus secondary-CIF intensity mix | optical boundary assumptions | ordered/integration | CORRECTED explicit |
| PHY-REC-010 | symmetry multiplicity semantics | rod `deg` fields and grouping paths | powder/domain sum | mosaic/ordered/integration | CORRECTED explicit |
| PHY-ORD-017 | remove injected fractional reflections | `utils/diffraction_tools.py:210-228` | continuous rods | ordered | CORRECTED |
| PHY-ORD-018 | reflection pruning disabled in proof | `gui/structure_factor_pruning.py`; `controllers.py:691-891` | none | ordered/integration | CORRECTED |
| PHY-ORD-019 | production pruning with detector-error bound | same | numerical approximation | integration | DEFERRED until profiled |
| PHY-ORD-020 | explicit R-centered three-registry finite parent | implicit conventional-cell centering and legacy stacking paths | `eq:structure_factor` plus crystallographic centering | ordered/stacking | CORRECTED active CPU/CUDA; exact `(0F+,2F+,1F+)` zero-disorder sequence, `2h+k+L=3n` extinction oracle, and bounded RichEpsilon departure law |
| PHY-ORD-021 | outer-chalcogen vacancy coordinate | full-occupancy substitution path | `eq:structure_factor` occupancy | ordered/fitting | CORRECTED active; vacancy `v` maps to outer-chalcogen occupancy `1-v`, outer-site Bi substitution is fixed to zero in the vacancy model, and Bi substitution is screened only as a separate discrete competitor |
| PHY-MAT-005 | charged and neutral species resolution | `utils/calculations.py` label parsing | `eq:structure_factor` | ordered | CORRECTED explicit |
| PHY-MOT-001 | layer/block extraction from expanded CIF | `stacking/motif_validation.py` | layer-amplitude construction | ordered | CORRECTED explicit |
| PHY-MOT-002 | stoichiometry and complete site coverage | same | motif definition | ordered | NEW proof |
| PHY-MOT-003 | species/occupancy-preserving orientation relation | same | `F+`, `F-` construction | ordered | CORRECTED explicit |
| PHY-MOT-004 | motif-origin and registry-phase gauge invariance | `motif_form_factor.py` plus stacking phase paths | Fourier convention | ordered/stacking | NEW proof |
| PHY-THK-001 | explicit finite rectangle or legacy-unbounded sample support | instrument configuration | geometry | geometry | CORRECTED; first possible divergence `geometry.footprint_acceptance` |
| PHY-THK-002 | optical film thickness | instrument configuration and attenuation | optics section | geometry | CORRECTED typed |
| PHY-THK-003 | coherent layer depths and repeat | `gui/controllers.py:336-351` | finite-stack equations | ordered/stacking | CORRECTED explicit |
| PHY-THK-004 | Parratt layer thickness and substrate infinity | `utils/calculations.py:328-480` | Parratt equations | ordered | CORRECTED typed |
| PHY-STK-015 | rich-parent epsilon parameterization | `utils/stacking_fault.py:134-424` | fault-template equations | stacking | MATCH/CORRECTED active; native 2H and 3R parents use the exact reduced recurrence, direct finite-stack enumeration is the oracle, CPU/CUDA agree, and epsilon zero retains the exact fast path |
| PHY-STK-016 | reduced `a,b,d` parameterization | `utils/polytype_stacking.py:38-414` | transition equations | stacking | MATCH after direct proof |
| PHY-STK-017 | typed registry-phase models | `utils/stacking_fault.py:763-904` | `eq:si_pbi2_omega` | stacking | CORRECTED |
| PHY-STK-018 | stationary or infinite-stack limit | stacking utilities | limiting theory | stacking | MATCH as separate output |
| PHY-CAL-001 | hBN geometry calibration behavior and coordinate signs | `hbn_geometry.py`; `hbn_fitter/fitter.py` | refinement step 2 | fitting | FUTURE T10 reference/specialization |
| PHY-CAL-002 | hBN ring and distance fitting | `hbn_fitter/fitter.py:83-161` | refinement step 2 | fitting | FUTURE T10, reimplemented without GUI |

## Selection and staged fitting

Position-free discovery, shared-incidence geometry, measured mosaic, fixed-position ordered
response, and the detector-native mixed-chart layered-Bi2X3 adapter for Bi2Se3 and Bi2Te3 are
active. Arbitrary-material raw-OSC structure recovery and raw-OSC stacking-disorder fitting remain
deferred.

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-SEL-001 | stable exact rod identity | distributed Miller and hit tables | Methods rod-family discussion | selection | NEW explicit |
| PHY-SEL-002 | hexagonal `m` and general-metric `Qr` family identity | `gui/geometry_q_group_manager.py:1197-1329` | Methods lines 64-78 | selection | CORRECTED |
| PHY-SEL-003 | deterministic physical signed-azimuth branch identity | `utils/calculations.py:48-62` | SI selected-branch discussion | selection | CORRECTED active analytic beta-root sign |
| PHY-SEL-004 | collapsed `00L` branch status | `utils/calculations.py:90-117` | specular-family semantics | selection | MATCH/CORRECTED typed |
| PHY-SEL-005 | measured peak to rod/branch association | geometry Q-group and peak-selection paths | refinement workflow | selection | CORRECTED frozen association |
| PHY-SEL-006 | detector-native ROI and selected-rod manifests | GUI selection managers | ordered and diffuse objectives | selection | NEW immutable manifest |
| PHY-FIT-000 | source size, divergence, wavelength, and correlation characterization | distributed beam setup | refinement step 1 | fitting | NEW staged result |
| PHY-FIT-001 | fit parameter, bounds, units, and dependency metadata | `fitting/geometry_fit_parameters.py` and GUI runtime | refinement workflow | fitting | CORRECTED typed; active four-angle local pack, selectable subsets of the nine shared series coordinates, and one optional common additive incidence delta; the overlapping sample-x gauge is rejected |
| PHY-FIT-002 | independent detector geometry calibration | calibrant and geometry paths | refinement step 2 | fitting | NEW/CORRECTED detector-native |
| PHY-FIT-002A | detector-native sample/goniometer residual | caked geometry objective and solver | refinement step 3 | fitting | CORRECTED active native marker coordinates; shared detector/sample/axis/pivot series pack under fixed calibration |
| PHY-FIT-002B | continuous detector-function exact-tag map from one nominal companion incident state | no single legacy owner | alignment stage | fitting | NEW active; source center, zero divergence, mean wavelength; shared geometry authorities construct no strength/mosaic and exclude the companion from empirical source mass |
| PHY-FIT-002C | paired tag-branch half-angle and m=0 increasing-L TLS-line residuals | caked line/peak geometry objectives | alignment stage | fitting | CORRECTED active simultaneous coordinate-plus-angle objective |
| PHY-FIT-003 | fixed branch/rod association during geometry optimization | `caked_geometry_objective.py` locked targets | alignment stage | fitting | MATCH active principle, analytic root sign added |
| PHY-FIT-003A | explicit outer re-index audit after geometry changes | distributed GUI selection behavior | indexing requirement | fitting/selection | NEW active direct fixed-L root oracle plus corrected-geometry relabeling of unchanged selected native candidates; fresh global search is diagnostic only |
| PHY-FIT-003B | exact-L m=0 minimum-tilt geometry landmark (`tag_branch=0`) with signed inverse orientations collapsed by positive `|L|` | no legacy point identity for the m=0 orientation curve | constrained Ewald geometry | fitting/mosaic | NEW active; deterministic geometry landmark, not claimed intensity maximum or exhaustive visible-circle scan |
| PHY-FIT-003C | strict arbitrary-length OSC-series schema and provenance-bound image-ID join | hard-coded GUI/script image lists | repeatable staged fitting | fitting/selection | NEW active; explicit commanded motor angles, one optional shared delta plus zero-sum Helmert trims, and one material/mount per fit group |
| PHY-FIT-003D | exact commensurate layer-coordinate landmarks with rod-free detector-locus identity and signed-rod provenance | integer-only layered peak labels | `eq:si_pbi2_parent_vectors` | fitting | NEW active additive boundary; reduced rational `L`, denominator-one legacy parity, exact overlap deduplication, independent fixed-L root audit, and post-discovery measured PbI2 admission with covariance/ambiguity/complete-root gates; the shared numerical detector profile response is active, while measured observation/background admission remains DEFERRED |
| PHY-FIT-004 | geometry synthetic recovery and held-out peaks | no compact old proof | refinement workflow | fitting | NEW active arbitrary-series proof; scaled-rank diagnostics cover the selected shared corrections, common delta, and trim contrasts without granting independent per-image offsets |
| PHY-FIT-004A | tightly conditioned lattice sensitivity after accepted position | no isolated legacy stage | refinement workflow | fitting/materials | NEW active; log-scaled in-plane/normal candidate, separate data/posterior sensitivity, data-only promotion gate, otherwise exact CIF retention |
| PHY-FIT-004B | optional PbI2 half-/third-order landmarks strengthen shared geometry information when observed | no compact old proof | pure-parent stacking periodicity | fitting | NEW active declared-single-trilayer-metric synthetic proof; 2H-only fallback and 2H/4H/6H augmentation recover one planted nine-coordinate pose; relaxed native cells and measured PbI2 data remain `NO_ORACLE` |
| PHY-FIT-005 | finite-bin mosaic-profile objective with integrate-`S/N`-before-divide and exact per-profile nuisance amplitudes | `optimization_mosaic_profiles.py` | refinement step 4 | fitting | CORRECTED active deterministic synthetic proof and model-limited three-OSC estimate; absolute and cross-peak intensities excluded from weighting |
| PHY-FIT-005A | detector-native OSC projection by exact cropped physical-pixel overlap plus frozen whole-profile weak/secondary selection | no single legacy owner | refinement step 4 | fitting/measurement | RETAINED legacy proof/input preparation; current mixed-chart fitting consumes a strict provided-mosaic checkpoint |
| PHY-FIT-005B | optional exact-rational PbI2 half-/third-order profiles add nuisance-projected shared-mosaic information | no compact old proof | pure-parent stacking periodicity + refinement step 4 | fitting | NEW active synthetic profile-response boundary; exact overlap deduplication, absent-site omission, population-scale invariance, and independent projected-Fisher gain; the shared detector response is active, while a measured PbI2 fit remains `NO_ORACLE` without admitted observations/background |
| PHY-FIT-006 | separate Gaussian width, Lorentzian width, and mixture with exact faces and finite centered-logit audit | old pseudo-Voigt workflows | `eq:mosaic_two_component_maintext` | fitting | CORRECTED active deterministic synthetic proof and model-limited three-OSC estimate |
| PHY-FIT-007 | frozen geometry/source/material/profile revisions during mosaic fitting | staged old workflow | refinement workflow lines 53-59 | fitting | MATCH principle; active deterministic synthetic proof and model-limited three-OSC estimate consume an explicit position revision, corrections, common incidence delta, and effective angles |
| PHY-FIT-008 | detector-native ordered selected-group ROI component-mass objective | `gui/ordered_structure_fit.py:322-540` | refinement step 5 and SI detector objectives | fitting | CORRECTED active deterministic 5/10/15-degree slice; raw-OSC component extraction DEFERRED |
| PHY-FIT-008A | source-averaged selected-group peak-center angular-density objective | no single legacy owner | refinement step 5 accelerator | fitting | NEW active synthetic 5/10/15-degree proof; every incident state is summed before selection/comparison; raw-OSC intensity recovery remains DEFERRED |
| PHY-FIT-009 | analytic nonnegative per-image scale and explicit common-occupancy gauge | `ordered_structure_fit.py:53-99` | nuisance image scale | fitting | MATCH principle, active exact profiling and ratio recovery |
| PHY-FIT-010 | fixed detector-response reuse for intensity fits | old code rerenders broadly | performance requirement | fitting | NEW active cached occupancy quadratic plus `Qr/Qz` damping; no detector reprojection per trial |
| PHY-FIT-011 | selected `Qr` and branch stacking objective | rod-profile and branch-selection paths | SI lines 704-749 | fitting | CORRECTED |
| PHY-FIT-011B | exact-rational PbI2 landmark responses constrain fixed 2H/4H/6H populations with one residual per physical overlap | no compact old proof | pure-parent periodicity + refinement step 6 | fitting | NEW active synthetic intrinsic-A2 boundary; detector roots collapse, unique signed rods sum once, missing optional sites add no row, active phase rank is enforced, and measured detector-folded PbI2 fitting remains `NO_ORACLE` |
| PHY-FIT-012 | signal and normalization summed before division in future caking | `fitting/rod_profiles.py:91-308` | SI selected-rod profile equation | fitting | MATCH when caking is added |
| PHY-FIT-013 | upstream parameters frozen before stacking fit | staged runtime | refinement step 6 | fitting | MATCH principle |
| PHY-FIT-014 | stage-specific synthetic parameter recovery | absent as one system | scientific validation | fitting | NEW active for geometry, mosaic, and fixed-position ordered intensity |
| PHY-FIT-017 | mixed-chart detector-native structure fit across three OSCs | separate caked/specular and reciprocal profile paths | refinement step 5 | fitting/measurement | CORRECTED active; `m=0` phi/2theta and `m!=0` signed-side Qr/L regions, one shared structure vector, one scale per OSC across all families, continuous chart-region model integration with no model raster, and covariance-aware contraction to one measured/model mass per complete peak rectangle |
| PHY-FIT-017A | policy-declared five-coordinate structure refinement | staged old workflow | refinement step 5 | fitting/materials | CORRECTED active; v7 isolates Wyckoff z, outer-chalcogen vacancy fraction, and sample-Q envelope through the exact A/B/C predecessor chain, while v8 `seeded_joint_only.v1` requires an explicit numeric start and exposes only the all-active joint stage; the outer workflow separately hashes seed-v2 when it supplies that start; vacancy uses outer occupancy `1-v` and zero Bi antisite, crystallographic site ADPs remain fixed, the frozen background/state is enforced, and only a gate-passing joint result may become authoritative |
| PHY-FIT-018 | fitted-scope full-Qz profile | legacy pruning/grouping paths | numerical approximation | fitting/integration | NEW active `FIT_CONDITIONED` terminal; exact fitted rod roster and cubature are retained, no all-rod or publication claim is made, and `publication_ready=false` |
| PHY-FIT-014A | explicit likelihood, variance, mask, background, scale, and data/model correction ledger | distributed fit paths | SI lines 109-134 and detector-derived objectives | fitting | PARTIAL active for matched regions: declared dark scale/basis/covariance, full piecewise-constant native-count projection covariance, propagated frozen radial-background covariance, identical peak aggregation, and one OSC scale; the Bi2Se3 remediation uses dark scale zero because no acquisition-matched dark exists; matched blank, PSF, and full detector-noise calibration remain DEFERRED |
| PHY-FIT-015 | dependency-aware cache invalidation | distributed runtime caches | performance requirement | fitting | NEW active response/observable/structure/mosaic/source/material revision boundaries |
| PHY-FIT-016 | optional global polish across geometry, lattice, mosaic, and structure stages | global old optimization paths | refinement discussion | fitting | FUTURE with safeguards; distinct from the mandatory within-structure five-coordinate joint stage |
| PHY-MAP-001B | finite `2theta/phi` caking and reciprocal remapping | exact-cake and exact-qspace modules | SI selected-profile workflow | measurement/fitting | PARTIAL selected continuous mixed-chart pullbacks are active through PHY-FIT-017; general or full-image caking remains DEFERRED |

## Coverage rule

Every current-phase `MATCH`, `CORRECTED`, or `NEW` row must be named by at least one task and one proof record. Every future or deferred row remains visible and may not be silently implemented under another name. Selection and fitting rows become active only after the integration merge gate.

## Contract-v13 general-CIF additions

| ID | Quantity | Legacy owner | Authority | Stage | Status |
|---|---|---|---|---|---|
| PHY-ORD-017 | complete conventional-CIF cell amplitude times coherent finite repeat | no generic owner | direct atom and repeat enumeration | ordered | NEW; exact mixed `(h,k,L,wavelength)` provider, explicit `repeats`, normalization, and unknown-U policy |
| PHY-FIT-019 | structure-independent source-averaged sparse detector transfer | Bi2X3 compiled renderer | source/rod/root direct reduction plus Bi2X3 parity | fitting/detector | NEW; selected coordinates, physical signed rods, exact wavelengths, regular kinematic `00L`, no generic raster or Parratt stitch |
| PHY-FIT-019A | affine expanded-CIF and explicit provider parameterizations through matched regions | Bi2X3-only ordered adapter | planted multi-dataset recovery and rank/gauge/revision rejection | fitting/materials | NEW; fixed cell/species/topology and isotropic U only |
| PHY-FIT-019B | five fixed PbI2 parent strengths through the shared detector | intrinsic-A2 parent compiler | detector linearity and log-ratio recovery | fitting/stacking | NEW model-limited provider; fixed `epsilon=0.001`, not arbitrary transition-law inference; `00L` is parent-invariant |
| PHY-FIT-002D | optional native detector reference-center and panel-normal distance corrections | calibration-owned fixed inputs | independent synthetic recovery and legacy-null parity | geometry/fitting | NEW; explicit downstream calibration provenance and ordinary data-rank gate |
| PHY-FIT-014B | Bi2Se3, Bi2Te3, ordered generic PbI2, and fixed-parent PbI2 share public sparse response/fitter types | separate material scripts | compact three-material synthetic integration | fitting | NEW numerical core; measured mixed/disordered PbI2 remains `NO_ORACLE` |
| PHY-FIT-020 | covariance-whitened adaptive finite-region angular oracle | no accepted equivalent | deterministic cubature qualification | fitting | NEW material-neutral core; caller-owned region evaluator, frozen source/evaluation identity, signed contrast, and fail-closed terminal events |
| PHY-FIT-021 | fixed-first bounded delayed acceptance | ad hoc staged scripts | staged inverse-problem qualification | fitting | NEW material-neutral core; fixed gate precedes scan work and exact scan scores are bounded by `1 + 2K` |
| PHY-ORD-018 | generic finite-CIF per-site crystal-frame displacement tensors | site isotropic path | direct signed complex atom/repeat enumeration and isotropic reduction | ordered | NEW additive v15; complete finite surface rows retained; tensor equation shared with unit-cell amplitude |
| PHY-FIT-022 | native Bi 13-coordinate cell/site plus morphology refinement | nominal native fixed-structure fits | full native replay, complete dependency/reuse invariants, GLS/guard oracle and planted 21-coordinate recovery | fitting | NEW v15 branch core; all occupancies update optics, cell updates geometry, N uses independent integer refits; measured exploration is not an accepted or qualified fit |
| PHY-ORD-019 | signed finite Pb motifs with orbit radial/normal ADPs | isotropic finite Pb endpoint path | independent explicit atomic enumeration over all two-layer state paths and three terminations | ordered/stacking | NEW v16; canonical tensor amplitude and existing recurrence, FINITE_TOTAL/PER_LAYER retained |
| PHY-FIT-023 | all admitted Bi/Pb specimen and acquisition coordinates through one native response | fixed-source native recipes | all-six typed bundles, Pb isotropic reduction, source/pose dependency and fresh-response parity | fitting | NEW v16 implementation; 39/39/39/39/45/43 continuous coordinates plus discrete N; measured numerical acceptance unresolved |
| PHY-FIT-024 | multistart, full nuisance/discrete profiles, calibration ownership and conditional validation | ad hoc fit workflows | analytic nuisance/scale optimum, correlated Schur-complement oracle, leakage and unresolved-minimum counterexamples | fitting | NEW v16; raw profiles have no automatic confidence interpretation; calibration covariance absent from current six acquisitions |
| PHY-FIT-025 | observable numerical contrasts, control contamination and explicit cone reuse | repeated response evaluation | identical-rule parity, guard-feasibility counterexample, real-forward recovery and all-six convergence screens | fitting/integration | NEW v16 diagnostics; current axial/angular/source rules fail declared tolerances; no unsupported extra blur admitted |
