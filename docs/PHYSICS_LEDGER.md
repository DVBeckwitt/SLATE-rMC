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
| PHY-SRC-001 | spatial beam distribution | `simulation/mosaic_profiles.py:15-63` | Methods line 18 | mosaic | CORRECTED, seeded empirical |
| PHY-SRC-002 | divergence distribution | same | Methods line 18 | mosaic | CORRECTED, normalized |
| PHY-SRC-003 | wavelength distribution | same and GUI bandwidth settings | `eq:detector_sum_lambda_main` | mosaic | CORRECTED, generic spectrum |
| PHY-SRC-004 | independent wavelength intensity sum | downstream sample loop | `eq:detector_sum_lambda_main` | mosaic/integration | MATCH |
| PHY-SRC-005 | stable sample identity and weights | distributed runtime arrays | n/a | bootstrap/mosaic | NEW |
| PHY-SRC-006 | joint source-variable correlations | separate old arrays imply factorization | source phase-space definition | mosaic | NEW explicit |

## Geometry

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-GEO-001 | rigid frame composition | `diffraction.py:1682-1742` | Methods geometry figures | geometry | CORRECTED |
| PHY-GEO-002 | sample pivot and offsets | same | sample geometry figure | geometry | CORRECTED |
| PHY-GEO-003 | remove `P0_rot[0]=0` | `diffraction.py:1740-1741` | none | geometry | CORRECTED |
| PHY-GEO-004 | detector plane and basis | `diffraction.py`; `intersection_analysis.py:80-122` | detector geometry figure | geometry | CORRECTED, single source |
| PHY-GEO-005 | ray-plane intersection | `diffraction.py:285-325` | geometric construction | geometry | MATCH/conditioned |
| PHY-GEO-006 | sample footprint clipping | `diffraction.py:1807-1874` | Methods line 18; SI line 525 | geometry | MATCH as accepted beam mass |
| PHY-GEO-007 | internal-to-lab outgoing vector | `diffraction.py:2260-2264` | Methods | geometry | CORRECTED |
| PHY-GEO-008 | detector intersection | `diffraction.py:2266-2299` | Methods | geometry | MATCH after frame correction |
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
| PHY-REC-006 | candidate mosaic/Jacobian mass | implicit `I_Q` | SI line 525 | mosaic | CORRECTED explicit |
| PHY-REC-007 | valid-support construction and convergence | old uniform/adaptive scan | numerical requirement | mosaic | NEW |
| PHY-REC-008 | complete-pool inverse-CDF selection without double weighting | event resampling paths | n/a | integration | CORRECTED |
| PHY-REC-009 | external versus internal Q | inconsistent helper paths | refraction section | bootstrap/mosaic | CORRECTED |

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
| PHY-ORD-008 | anisotropic `Uij` | incomplete old support | `eq:structure_factor` | ordered | NEW required |
| PHY-ORD-009 | complex structure amplitude | `motif_form_factor.py:473-543` | `eq:structure_factor` | ordered | MATCH |
| PHY-ORD-010 | remove normalization to 100 and rounding | `diffraction_tools.py:135-207` | none | ordered | CORRECTED |
| PHY-ORD-011 | distinct `(h,k)` rods with family metadata | Miller/grouping paths | Methods lines 64-78 | ordered | CORRECTED |
| PHY-ORD-012 | continuous arbitrary-Qz rod amplitude | old cached grids | rod construction | ordered | CORRECTED |
| PHY-ORD-013 | finite ordered stack | structure and stacking helpers | finite thickness discussion | ordered | MATCH/analytic |
| PHY-ORD-014 | arbitrary complex depth-field weights in ordered amplitudes | no exact old equivalent | future distorted-wave model | none | DEFERRED |
| PHY-ORD-015 | systematic absences from amplitude | old pruning/generation | crystallographic invariant | ordered | NEW proof |
| PHY-ORD-016 | global anisotropic Q damping | `diffraction.py:2314-2315` | not derived | integration | CORRECTED, off by default |

## Reflectivity

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-REF-001 | general multilayer normal modes | `calculations.py:328-480` | `eq:si_parratt_kz` | ordered | CORRECTED shared mode |
| PHY-REF-002 | bottom-up Parratt recursion | `calculations.py:403-429` | `eq:si_parratt_recursion` | ordered | MATCH |
| PHY-REF-003 | interface roughness | `calculations.py:419-423` | `eq:si_parratt_roughness` | ordered | MATCH convention |
| PHY-REF-004 | substrate and finite film | `calculations.py:432-480` | Parratt section | ordered | MATCH/generalized |
| PHY-REF-005 | external Qz, internal phase Qz, and L | `calculations.py:462-478` | `eq:si_internal_phase_coordinate` | ordered | CORRECTED named |
| PHY-REF-006 | pure kinematic specular rod | helper internals | `eq:si_ht_normalized_structure` | ordered | CORRECTED raw |
| PHY-REF-007 | empirical smooth handoff | `calculations.py:591-803` | `eq:si_handoff_x_l`, `eq:si_handoff_unscaled_structure`, `eq:si_handoff_scale`, `eq:si_handoff_scaled_structure`, `eq:si_handoff_log_ratio`, `eq:si_handoff_smoothstep`, and `eq:si_handoff_blend` | ordered | MATCH as named compatibility |
| PHY-REF-008 | scalar multilayer incident and reciprocal-exit field profiles | UI says DWBA but event path is single-pass | future distorted-wave model | none | DEFERRED |
| PHY-REF-009 | amplitude-level optical weighting inside ordered and stacking sums | no old equivalent | future distorted-wave model | none | DEFERRED |
| PHY-REF-010 | single-pass scalar entrance/exit field weighting | `diffraction.py:1917-1946`, `2223-2316` | `eq:si_scalar_transmission` through `eq:si_full_optical_weight_lambda` | geometry/integration | CORRECTED, current reference |
| PHY-REF-011 | roughness-consistent off-specular local fields | Parratt roughness exists, event local fields do not | future distorted-wave model | none | DEFERRED |

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
| PHY-MEA-001 | sampled raw-detector mass declaration | absent | SI lines 523-527 | bootstrap | NEW |
| PHY-MEA-002 | solid-angle correction for later caking/analysis | caking helper `exact_cake_portable.py:866-874` | SI lines 523-527 | analysis | CORRECTED, excluded from raw rendering |
| PHY-MEA-003 | scattering polarization distinct from Fresnel fields | no consistent native path | SI preprocessing and interface notes | integration | NEW explicit model, data-corrected unity, or declared approximation |
| PHY-MEA-003A | reciprocal event Jacobian versus separate Lorentz factor | implicit old `I_Q` and powder paths | SI event measure | mosaic/integration | CORRECTED, no duplicate factor |
| PHY-MEA-004 | detector pixel integration | point/bilinear events | detector measurement | integration | NEW convergence proof |
| PHY-MEA-005 | mass-conserving deposition | diffraction accumulation helpers | numerical | integration | MATCH/prove |
| PHY-MEA-006 | detector PSF/resolution | bilinear only | ordered-results resolution discussion | integration | NEW normalized operator |
| PHY-MEA-007 | detector efficiency | not explicit | absolute-count requirement | none | DEFERRED unless calibrated |
| PHY-MEA-008 | masks, beamstop, saturation, bad pixels | GUI/data paths | experimental handling | none | DEFERRED from forward core |
| PHY-MEA-009 | background | GUI/fitting paths | later comparison | none | DEFERRED |
| PHY-MEA-010 | multiple scattering and extinction | absent | not claimed | none | DEFERRED |

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
| PHY-MAT-005 | charged and neutral species resolution | `utils/calculations.py` label parsing | `eq:structure_factor` | ordered | CORRECTED explicit |
| PHY-MOT-001 | layer/block extraction from expanded CIF | `stacking/motif_validation.py` | layer-amplitude construction | ordered | CORRECTED explicit |
| PHY-MOT-002 | stoichiometry and complete site coverage | same | motif definition | ordered | NEW proof |
| PHY-MOT-003 | species/occupancy-preserving orientation relation | same | `F+`, `F-` construction | ordered | CORRECTED explicit |
| PHY-MOT-004 | motif-origin and registry-phase gauge invariance | `motif_form_factor.py` plus stacking phase paths | Fourier convention | ordered/stacking | NEW proof |
| PHY-THK-001 | sample footprint dimensions | instrument configuration | geometry | geometry | CORRECTED typed |
| PHY-THK-002 | optical film thickness | instrument configuration and attenuation | optics section | geometry | CORRECTED typed |
| PHY-THK-003 | coherent layer depths and repeat | `gui/controllers.py:336-351` | finite-stack equations | ordered/stacking | CORRECTED explicit |
| PHY-THK-004 | Parratt layer thickness and substrate infinity | `utils/calculations.py:328-480` | Parratt equations | ordered | CORRECTED typed |
| PHY-STK-015 | rich-parent epsilon parameterization | `utils/stacking_fault.py:134-424` | fault-template equations | stacking | MATCH after direct proof |
| PHY-STK-016 | reduced `a,b,d` parameterization | `utils/polytype_stacking.py:38-414` | transition equations | stacking | MATCH after direct proof |
| PHY-STK-017 | typed registry-phase models | `utils/stacking_fault.py:763-904` | `eq:si_pbi2_omega` | stacking | CORRECTED |
| PHY-STK-018 | stationary or infinite-stack limit | stacking utilities | limiting theory | stacking | MATCH as separate output |
| PHY-CAL-001 | hBN geometry calibration behavior and coordinate signs | `hbn_geometry.py`; `hbn_fitter/fitter.py` | refinement step 2 | fitting | FUTURE T10 reference/specialization |
| PHY-CAL-002 | hBN ring and distance fitting | `hbn_fitter/fitter.py:83-161` | refinement step 2 | fitting | FUTURE T10, reimplemented without GUI |

## Selection and staged fitting

These rows are future work and begin only after native-detector integration passes.

| ID | Operation | Original RASIM source | Manuscript source | Owner | Treatment |
|---|---|---|---|---|---|
| PHY-SEL-001 | stable exact rod identity | distributed Miller and hit tables | Methods rod-family discussion | selection | NEW explicit |
| PHY-SEL-002 | hexagonal `m` and general-metric `Qr` family identity | `gui/geometry_q_group_manager.py:1197-1329` | Methods lines 64-78 | selection | CORRECTED |
| PHY-SEL-003 | deterministic physical signed-azimuth branch identity | `utils/calculations.py:48-62` | SI selected-branch discussion | selection | CORRECTED, declared sample/crystal basis |
| PHY-SEL-004 | collapsed `00L` branch status | `utils/calculations.py:90-117` | specular-family semantics | selection | MATCH/CORRECTED typed |
| PHY-SEL-005 | measured peak to rod/branch association | geometry Q-group and peak-selection paths | refinement workflow | selection | CORRECTED frozen association |
| PHY-SEL-006 | detector-native ROI and selected-rod manifests | GUI selection managers | ordered and diffuse objectives | selection | NEW immutable manifest |
| PHY-FIT-000 | source size, divergence, wavelength, and correlation characterization | distributed beam setup | refinement step 1 | fitting | NEW staged result |
| PHY-FIT-001 | fit parameter, bounds, units, and dependency metadata | `fitting/geometry_fit_parameters.py` and GUI runtime | refinement workflow | fitting | CORRECTED typed |
| PHY-FIT-002 | independent detector geometry calibration | calibrant and geometry paths | refinement step 2 | fitting | NEW/CORRECTED detector-native |
| PHY-FIT-002A | detector-native sample/goniometer residual | caked geometry objective and solver | refinement step 3 | fitting | CORRECTED, no caking required |
| PHY-FIT-003 | fixed branch/rod association during geometry optimization | `caked_geometry_objective.py` locked targets | alignment stage | fitting | MATCH principle |
| PHY-FIT-003A | explicit outer re-index audit after geometry changes | distributed GUI selection behavior | indexing requirement | fitting/selection | NEW required |
| PHY-FIT-004 | geometry synthetic recovery and held-out peaks | no compact old proof | refinement workflow | fitting | NEW required |
| PHY-FIT-005 | normalized local mosaic-profile objective | `optimization_mosaic_profiles.py` | refinement step 4 | fitting | CORRECTED |
| PHY-FIT-006 | separate Gaussian width, Lorentzian width, and mixture | old pseudo-Voigt workflows | `eq:mosaic_two_component_maintext` | fitting | CORRECTED |
| PHY-FIT-007 | frozen geometry during mosaic fitting | staged old workflow | refinement workflow lines 53-59 | fitting | MATCH principle |
| PHY-FIT-008 | detector-native ordered ROI mass objective | `gui/ordered_structure_fit.py:322-540` | refinement step 5 and SI detector objectives | fitting | CORRECTED |
| PHY-FIT-009 | analytic nonnegative per-image scale | `ordered_structure_fit.py:53-99` | nuisance image scale | fitting | MATCH/verify |
| PHY-FIT-010 | event and detector-response reuse for intensity fits | old code rerenders broadly | future performance requirement | fitting | NEW |
| PHY-FIT-011 | selected `Qr` and branch stacking objective | rod-profile and branch-selection paths | SI lines 704-749 | fitting | CORRECTED |
| PHY-FIT-012 | signal and normalization summed before division in future caking | `fitting/rod_profiles.py:91-308` | SI selected-rod profile equation | fitting | MATCH when caking is added |
| PHY-FIT-013 | upstream parameters frozen before stacking fit | staged runtime | refinement step 6 | fitting | MATCH principle |
| PHY-FIT-014 | stage-specific synthetic parameter recovery | absent as one system | scientific validation | fitting | NEW required |
| PHY-FIT-014A | explicit likelihood, variance, mask, background, scale, and data/model correction ledger | distributed fit paths | SI lines 109-134 and detector-derived objectives | fitting | CORRECTED explicit |
| PHY-FIT-015 | dependency-aware cache invalidation | distributed runtime caches | performance requirement | fitting | NEW |
| PHY-FIT-016 | optional final joint polish after staged stability | global old optimization paths | refinement discussion | fitting | FUTURE with safeguards |
| PHY-MAP-001 | `2theta/phi` caking and reciprocal remapping | exact-cake and exact-qspace modules | SI selected-profile workflow | none | DEFERRED until native fits pass |

## Coverage rule

Every current-phase `MATCH`, `CORRECTED`, or `NEW` row must be named by at least one task and one proof record. Every future or deferred row remains visible and may not be silently implemented under another name. Selection and fitting rows become active only after the integration merge gate.
