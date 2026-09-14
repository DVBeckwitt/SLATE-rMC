from __future__ import annotations

import hashlib
import json
import math
import runpy
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from painted_ewald import ContinuousEwaldCoating, Rod
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    incidence_scan_calibration_binding_revision,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting.geometry import (
    ContinuousDetectorGeometryModel,
    ExactTagGeometryModel,
    GeometryCorrections,
)
from rasim_next.fitting.matched_regions import (
    FixedMatchedRegionBackground,
    MatchedRegionObservations,
    ParameterizedStructureRegionModel,
    StructureRegionIdentifiabilityError,
    StructureRegionResponseBlock,
    fit_parameterized_matched_regions,
)
from rasim_next.fitting.stacking_intensity import (
    Pbi2ParentLogRatioParameterization,
    Pbi2ParentMixtureStrength,
)
from rasim_next.geometry import (
    AngleFrame,
    CompiledInstrument,
    InstrumentConfiguration,
    angles_to_detector_coordinate_area_measure,
    build_incident_states,
    compile_instrument,
    detector_coordinates_to_angles,
    project_detector_ray,
    project_detector_rays,
)
from rasim_next.materials import AffineCifSiteBasis, material_optics
from rasim_next.measurement import (
    AngleBinGrid,
    ContinuousNormalizedAngleFunction,
    ContinuousRegionQuadrature,
    compile_detector_angle_projector,
    compile_detector_profile_projector,
    project_detector_profiles,
    project_normalized_angle_field,
    to_increasing_phi,
)
from rasim_next.pipeline.bragg_space import (
    AffineCifFiniteStackParameterization,
    Bi2X3FiniteStackParameterization,
    RevisionedStructureStrengthModel,
)
from rasim_next.pipeline.configured_simulation import (
    SimulationConfiguration,
    build_configured_geometry_inputs,
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    build_source_averaged_detector,
    build_source_averaged_structure_detector,
    configured_rod_catalog_revision,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
    rebind_configured_simulation_instrument,
    rebind_source_averaged_detector_incidence_scan,
    sample_configured_nominal_geometry_source,
)
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure, SampleQIntensityEnvelope
from rasim_next.pipeline.source_averaged_detector import (
    SourceAveragedDetectorCoordinateIntensity,
    SourceAveragedDetectorEwaldMeasure,
)
from rasim_next.pipeline.source_averaged_structure import (
    compile_source_averaged_detector_structure_response,
)
from rasim_next.reflectivity import CompiledParrattStitch, ParrattStitchStack

DETECTOR_VIEWER_SCRIPT = Path(__file__).resolve().parents[1] / "interactive" / "detector_viewer.py"


def _single_mean_test_source(source: object) -> object:
    return replace(
        source,
        wavelength_model_id="gaussian.v1",
        wavelength_sigma_A=0.0,
        sample_count=1,
        line_wavelength_A=(),
        line_probability=(),
        common_line_sigma_A=0.0,
    )


def _configured_inputs(
    *,
    sample_count: int,
    sample_angle_deg: float = 5.0,
    detector_path_linear_attenuation_m_inv: float = 0.0,
    reference_wavelength_A: float | None = None,
) -> object:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    rotations = (
        replace(config.instrument.axis_rotations[0], angle_deg=sample_angle_deg),
        *config.instrument.axis_rotations[1:],
    )
    source = (
        _single_mean_test_source(config.source)
        if sample_count == 1
        else replace(config.source, sample_count=sample_count)
    )
    if reference_wavelength_A is not None:
        source = replace(
            source,
            wavelength_model_id="gaussian.v1",
            mean_wavelength_A=reference_wavelength_A,
            wavelength_sigma_A=0.0,
            line_wavelength_A=(),
            line_probability=(),
            common_line_sigma_A=0.0,
            position_divergence_correlation=(0.0, 0.0),
        )
    return build_configured_simulation_inputs(
        replace(
            config,
            source=source,
            instrument=replace(
                config.instrument,
                axis_rotations=rotations,
                detector_path_medium_id=(
                    "standard_dry_air_sensitivity.v1"
                    if detector_path_linear_attenuation_m_inv
                    else "vacuum_or_helium_unity.v1"
                ),
                detector_path_linear_attenuation_m_inv=(detector_path_linear_attenuation_m_inv),
            ),
        )
    )


def _physical_scalar_detector(inputs: object) -> DetectorEwaldMeasure:
    if inputs.incident.states.incident_state_id.size != 1:
        raise ValueError("scalar detector proof requires one physical source state")
    return DetectorEwaldMeasure(
        coating=ContinuousEwaldCoating(
            inputs.bragg_space,
            ki_sample_Ainv=inputs.incident.states.k_film_phase_sample_Ainv[0],
        ),
        incident=inputs.incident,
        material=inputs.material,
        instrument=inputs.instrument,
        rod_catalog_revision=configured_rod_catalog_revision(inputs),
        phase_population_weight=inputs.config.weights.phase_population,
        polarization_weight=inputs.config.weights.polarization,
    )


def test_grazing_detector_exit_checks_dispersion_not_ill_conditioned_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasim_next.pipeline import continuous_detector as module

    inputs = _configured_inputs(sample_count=1, reference_wavelength_A=0.7107)
    detector = _physical_scalar_detector(inputs)
    # A canonical projected ray at 0.1748 degree sample-frame exit elevation.
    column = np.array([1329.3503816762632])
    row = np.array([1528.4682896715926])
    rods = (Rod(1, 0),)
    result = detector.evaluate_detector_coordinates(column, row, rods=rods)
    assert bool(result.geometry.valid[0])
    assert np.isfinite(result.density_A2_per_px2[0])

    solve = module._solve_exit_mode_arrays

    def inconsistent_normal(*args: object, **kwargs: object) -> object:
        modes = solve(*args, **kwargs)
        wavevector = modes.k_air_phase_sample_Ainv.copy()
        wavevector[:, 2] += 1.0e-5
        return replace(modes, k_air_phase_sample_Ainv=wavevector)

    monkeypatch.setattr(module, "_solve_exit_mode_arrays", inconsistent_normal)
    with pytest.raises(FloatingPointError, match="canonical exit round trip"):
        detector.evaluate_detector_coordinates(column, row, rods=rods)


def _generic_pbi2_config(*, include_detector_visible_m0: bool) -> SimulationConfiguration:
    root = Path(__file__).resolve().parents[1]
    base = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    return replace(
        base,
        material=replace(
            base.material,
            cif_path=root / "examples" / "pbi2" / "structures" / "PbI2_2H.cif",
            phase_id="PbI2",
        ),
        source=replace(
            _single_mean_test_source(base.source),
            spatial_sigma_m=(0.0, 0.0),
            divergence_sigma_rad=(0.0, 0.0),
        ),
        structure_factor=replace(
            base.structure_factor,
            model_id="cif_conventional_cell_finite_repeat.v1",
            layers=None,
            repeats=5,
            normalization="FINITE_TOTAL",
            shared_disorder_epsilon=0.0,
            unknown_u_iso_A2=0.0,
        ),
        bragg=replace(
            base.bragg,
            include_detector_visible_m0=include_detector_visible_m0,
        ),
    )


def test_commanded_angle_rebind_retains_every_sample_wavelength_rod() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    broad_source = replace(
        config.source,
        sample_count=40,
        seed=1,
        wavelength_model_id="gaussian.v1",
        wavelength_sigma_A=0.25,
        line_wavelength_A=(),
        line_probability=(),
        common_line_sigma_A=0.0,
    )

    def at_angle(angle_deg: float) -> object:
        rotations = (
            replace(config.instrument.axis_rotations[0], angle_deg=angle_deg),
            *config.instrument.axis_rotations[1:],
        )
        return replace(
            config,
            source=broad_source,
            instrument=replace(config.instrument, axis_rotations=rotations),
            bragg=replace(config.bragg, include_detector_visible_m0=False),
        )

    base = build_configured_simulation_inputs(at_angle(0.0))
    rebound = rebind_configured_simulation_instrument(base, at_angle(5.0))
    fresh = build_configured_simulation_inputs(at_angle(5.0))

    assert tuple((rod.h, rod.k) for rod in rebound.rods) == tuple(
        (rod.h, rod.k) for rod in fresh.rods
    )

    base_5deg = build_configured_simulation_inputs(at_angle(5.0))
    rebound_20deg = rebind_configured_simulation_instrument(base_5deg, at_angle(20.0))
    fresh_20deg = build_configured_simulation_inputs(at_angle(20.0))
    rebound_detector = build_source_averaged_detector(base_5deg).rebind_geometry(
        incident=rebound_20deg.incident,
        instrument=rebound_20deg.instrument,
    )
    fresh_detector = build_source_averaged_detector(fresh_20deg)
    markers = evaluate_nominal_integer_l_markers(build_nominal_ewald_context(fresh_20deg))
    marker_index = int(np.flatnonzero((markers.family_m == 1) & (markers.root_sign != 0))[0])
    rebound_density = rebound_detector.evaluate_detector_coordinates_all_roots(
        [markers.column_px[marker_index]],
        [markers.row_px[marker_index]],
    )
    fresh_density = fresh_detector.evaluate_detector_coordinates_all_roots(
        [markers.column_px[marker_index]],
        [markers.row_px[marker_index]],
    )
    assert np.any(fresh_density.per_rod_density_A2_per_px2 > 0.0)
    np.testing.assert_allclose(
        rebound_density.per_rod_density_A2_per_px2,
        fresh_density.per_rod_density_A2_per_px2,
        rtol=3.0e-12,
        atol=1.0e-24,
    )


def test_nominal_geometry_reference_cannot_enter_source_averaged_intensity() -> None:
    from rasim_next.pipeline.continuous_detector import DetectorQuadrature

    physical = _configured_inputs(sample_count=2)
    nominal_samples = sample_configured_nominal_geometry_source(physical.config.source)
    nominal_material = material_optics(physical.crystal, nominal_samples.wavelength_A)
    nominal_incident = build_incident_states(
        nominal_samples,
        nominal_material,
        physical.instrument,
    )
    nominal = replace(
        physical,
        samples=nominal_samples,
        material=nominal_material,
        incident=nominal_incident,
    )

    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        build_source_averaged_detector(nominal)
    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        build_source_averaged_structure_detector(nominal)
    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        compile_source_averaged_detector_structure_response(
            np.asarray([1000.0]),
            np.asarray([1000.0]),
            reciprocal_basis_Ainv=nominal.reciprocal.basis_Ainv,
            crystal_to_sample=nominal.instrument.sample_from_crystal.rotation,
            rods=nominal.rods,
            rod_catalog_revision=configured_rod_catalog_revision(nominal),
            mosaic=nominal.mosaic,
            reference_strength_model=nominal.strength,
            incident=nominal.incident,
            material=nominal.material,
            instrument=nominal.instrument,
        )
    nominal_context = build_nominal_ewald_context(physical)
    rod = next(item for item in nominal_context.rods if item.family_m != 0)
    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        nominal_context.geometry.map_latent(
            rod=rod,
            branch=2,
            alpha_rad=0.01,
            beta_rad=0.02,
        )
    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        nominal_context.geometry.evaluate_detector_coordinates(
            [1000.0],
            [1000.0],
            rods=(rod,),
        )
    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        nominal_context.geometry.evaluate_detector_structure_response(
            [1000.0],
            [1000.0],
            rods=(rod,),
        )
    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        nominal_context.geometry.integrate_native_pixels(
            rods=(rod,),
            branch=2,
            quadrature=DetectorQuadrature(),
        )
    assert build_source_averaged_detector(_configured_inputs(sample_count=1)) is not None
    assert _physical_scalar_detector(_configured_inputs(sample_count=1)) is not None


def test_incidence_angle_quadrature_is_a_normalized_uniform_measure() -> None:
    from rasim_next.pipeline.incidence_acquisition import IncidenceAngleQuadrature

    lower = math.radians(3.25)
    upper = math.radians(19.75)
    quadrature = IncidenceAngleQuadrature.uniform_legendre(
        lower,
        upper,
        gauss_order=6,
        subdivision_count=3,
        incidence_angle_calibration_revision="test-effective-incidence-axis.v1",
    )

    assert np.all(np.diff(quadrature.incidence_angle_rad) > 0.0)
    assert np.all(quadrature.incidence_angle_rad > lower)
    assert np.all(quadrature.incidence_angle_rad < upper)
    assert math.fsum(quadrature.exposure_probability_mass) == pytest.approx(
        1.0,
        rel=0.0,
        abs=4.0e-16,
    )
    assert math.fsum(
        quadrature.incidence_angle_rad * quadrature.exposure_probability_mass
    ) == pytest.approx(0.5 * (lower + upper), rel=0.0, abs=3.0e-16)
    assert not quadrature.incidence_angle_rad.flags.writeable
    assert not quadrature.exposure_probability_mass.flags.writeable
    assert quadrature.quadrature_revision != (
        IncidenceAngleQuadrature.uniform_legendre(
            lower,
            upper,
            gauss_order=8,
            subdivision_count=3,
            incidence_angle_calibration_revision="test-effective-incidence-axis.v1",
        ).quadrature_revision
    )
    with pytest.raises(ValueError, match="already sum to one"):
        IncidenceAngleQuadrature(
            incidence_angle_rad=np.asarray((lower, upper)),
            exposure_probability_mass=np.asarray((0.2, 0.7)),
            exposure_density_id="unnormalized-test-exposure.v1",
            incidence_angle_calibration_revision="test-effective-incidence-axis.v1",
        )


def test_incidence_angle_averaged_detector_equals_explicit_weighted_sum() -> None:
    from types import SimpleNamespace

    from rasim_next.pipeline.incidence_acquisition import IncidenceAngleQuadrature
    from rasim_next.pipeline.incidence_angle_average import (
        build_incidence_angle_averaged_detector,
    )

    rods = (Rod(-1, 0), Rod(1, 0))
    quadrature = IncidenceAngleQuadrature(
        incidence_angle_rad=np.asarray((0.1, 0.4, 0.9)),
        exposure_probability_mass=np.asarray((0.2, 0.3, 0.5)),
        exposure_density_id="explicit-test-exposure.v1",
        incidence_angle_calibration_revision="analytic-test-angle-axis.v1",
    )
    analytic_geometry_revisions = tuple(
        f"analytic-pose-{float(angle):.17g}" for angle in quadrature.incidence_angle_rad
    )
    analytic_calibration_binding_revision = incidence_scan_calibration_binding_revision(
        scan_calibration_revision=quadrature.incidence_angle_calibration_revision,
        component_sample_geometry_revision=analytic_geometry_revisions,
        component_incidence_axis_angle_rad=quadrature.incidence_angle_rad,
        effective_incidence_angle_rad=quadrature.incidence_angle_rad,
        detector_panel_revision="analytic-fixed-panel.v1",
        source_revision="analytic-shared-source.v1",
        source_state_count=4,
    )

    class AnalyticAllRootDetector:
        measure_id = "raw_detector_coordinate_density_A2_per_px2.v1"
        root_policy = "all_retained_roots.v1"
        source_revision = "analytic-shared-source.v1"
        source_state_count = 4
        rod_catalog_revision = "analytic-angle-scan-rods.v1"
        detector_panel_revision = "analytic-fixed-panel.v1"
        detector_shape_rc = (10, 12)
        incidence_angle_static_physics_revision = "analytic-fixed-physics.v1"
        detector_visible_m0_q_gap_Ainv = None

        def __init__(self, incidence_angle_rad: float) -> None:
            self.incidence_angle_rad = incidence_angle_rad
            self.incidence_axis_angle_rad = incidence_angle_rad
            self.rods = rods
            self.sample_geometry_revision = f"analytic-pose-{incidence_angle_rad:.17g}"
            self.scan_calibration_binding_revision = analytic_calibration_binding_revision

        def evaluate_detector_coordinates_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
            cuda_coordinate_chunk_size: int | None = None,
        ) -> object:
            assert execution_backend == "cpu"
            assert cuda_coordinate_chunk_size is None
            column, row = np.broadcast_arrays(
                np.asarray(column_px, dtype=np.float64),
                np.asarray(row_px, dtype=np.float64),
            )
            first = 1.0 + self.incidence_angle_rad * column**2
            second = 2.0 + self.incidence_angle_rad**2 * (row + 1.0)
            per_rod = np.stack((first, second), axis=-1)
            valid_count = np.where(
                column + row > 1.0,
                4,
                1 + int(self.incidence_angle_rad > 0.3) + int(self.incidence_angle_rad > 0.8),
            )
            caustic = np.zeros(per_rod.shape, dtype=np.bool_)
            return SimpleNamespace(
                column_px=column,
                row_px=row,
                rods=rods,
                rod_catalog_revision=self.rod_catalog_revision,
                branch=None,
                root_policy=self.root_policy,
                per_rod_density_A2_per_px2=per_rod,
                density_A2_per_px2=np.sum(per_rod, axis=-1),
                caustic=caustic,
                valid_source_count=valid_count,
                source_state_count=self.source_state_count,
                source_revision=self.source_revision,
                measure_id=self.measure_id,
                execution_backend="numba_cpu_source_averaged.v1",
                execution_device=None,
            )

    detector = build_incidence_angle_averaged_detector(
        quadrature,
        AnalyticAllRootDetector,
    )
    column_px = np.asarray((0.5, 2.0))[:, None]
    row_px = np.asarray((0.0, 1.0, 3.0))[None, :]
    expected_components = tuple(
        AnalyticAllRootDetector(float(angle)).evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
        )
        for angle in quadrature.incidence_angle_rad
    )
    expected_per_rod = sum(
        mass * component.per_rod_density_A2_per_px2
        for mass, component in zip(
            quadrature.exposure_probability_mass,
            expected_components,
            strict=True,
        )
    )
    expected_valid_fraction = sum(
        mass * component.valid_source_count / component.source_state_count
        for mass, component in zip(
            quadrature.exposure_probability_mass,
            expected_components,
            strict=True,
        )
    )

    evaluated = detector.evaluate_detector_coordinates_all_roots(column_px, row_px)
    total_only = detector.evaluate_detector_density_all_roots(column_px, row_px)
    np.testing.assert_allclose(
        evaluated.per_rod_density_A2_per_px2,
        expected_per_rod,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluated.density_A2_per_px2,
        np.sum(expected_per_rod, axis=-1),
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluated.valid_incident_state_fraction,
        expected_valid_fraction,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(evaluated.caustic, False)
    np.testing.assert_array_equal(total_only.caustic, False)
    np.testing.assert_allclose(
        total_only.density_A2_per_px2,
        evaluated.density_A2_per_px2,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        total_only.valid_incident_state_fraction,
        expected_valid_fraction,
        rtol=2.0e-15,
        atol=0.0,
    )
    expected_column, expected_row = np.broadcast_arrays(column_px, row_px)
    np.testing.assert_array_equal(evaluated.column_px, expected_column)
    np.testing.assert_array_equal(evaluated.row_px, expected_row)
    assert evaluated.per_rod_density_A2_per_px2.shape == (2, 3, 2)
    assert evaluated.reduction_id == (
        "fixed_quadrature_estimate_of_incoherent_incidence_angle_probability_average.v1"
    )
    assert evaluated.quadrature_revision == quadrature.quadrature_revision
    assert evaluated.source_revision == "analytic-shared-source.v1"
    assert evaluated.scan_revision == detector.scan_revision
    with pytest.raises(TypeError, match="built by their detector evaluator"):
        replace(evaluated, source_state_count=0)

    class ReducedAnalyticAllRootDetector(AnalyticAllRootDetector):
        detailed_call_count = 0
        total_call_count = 0

        def evaluate_detector_coordinates_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
            cuda_coordinate_chunk_size: int | None = None,
        ) -> object:
            type(self).detailed_call_count += 1
            return super().evaluate_detector_coordinates_all_roots(
                column_px,
                row_px,
                execution_backend=execution_backend,
                cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
            )

        def evaluate_detector_density_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
            cuda_coordinate_chunk_size: int | None = None,
        ) -> object:
            type(self).total_call_count += 1
            detailed = super().evaluate_detector_coordinates_all_roots(
                column_px,
                row_px,
                execution_backend=execution_backend,
                cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
            )
            return SimpleNamespace(
                column_px=detailed.column_px,
                row_px=detailed.row_px,
                rods=detailed.rods,
                rod_catalog_revision=detailed.rod_catalog_revision,
                branch=None,
                density_A2_per_px2=detailed.density_A2_per_px2,
                caustic=np.any(detailed.caustic, axis=-1),
                valid_source_count=detailed.valid_source_count,
                source_state_count=detailed.source_state_count,
                source_revision=detailed.source_revision,
                root_policy=detailed.root_policy,
                measure_id=detailed.measure_id,
                execution_backend=detailed.execution_backend,
                execution_device=detailed.execution_device,
            )

    reduced = build_incidence_angle_averaged_detector(
        quadrature,
        ReducedAnalyticAllRootDetector,
    ).evaluate_detector_density_all_roots(column_px, row_px)
    np.testing.assert_allclose(
        reduced.density_A2_per_px2,
        evaluated.density_A2_per_px2,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced.valid_incident_state_fraction,
        expected_valid_fraction,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert ReducedAnalyticAllRootDetector.total_call_count == 3
    assert ReducedAnalyticAllRootDetector.detailed_call_count == 0

    class ExactCausticDetector(AnalyticAllRootDetector):
        def evaluate_detector_coordinates_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
            cuda_coordinate_chunk_size: int | None = None,
        ) -> object:
            result = super().evaluate_detector_coordinates_all_roots(
                column_px,
                row_px,
                execution_backend=execution_backend,
                cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
            )
            result.caustic = np.ones_like(result.caustic)
            return result

    exact_caustic = build_incidence_angle_averaged_detector(
        quadrature,
        ExactCausticDetector,
    )
    with pytest.raises(FloatingPointError, match="exact caustic"):
        exact_caustic.evaluate_detector_coordinates_all_roots(column_px, row_px)
    with pytest.raises(FloatingPointError, match="exact caustic"):
        exact_caustic.evaluate_detector_density_all_roots(column_px, row_px)

    class InvalidDensityDetector(AnalyticAllRootDetector):
        def evaluate_detector_coordinates_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
            cuda_coordinate_chunk_size: int | None = None,
        ) -> object:
            result = super().evaluate_detector_coordinates_all_roots(
                column_px,
                row_px,
                execution_backend=execution_backend,
                cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
            )
            result.per_rod_density_A2_per_px2 = -result.per_rod_density_A2_per_px2
            result.density_A2_per_px2 = -result.density_A2_per_px2
            return result

    invalid_density = build_incidence_angle_averaged_detector(
        quadrature,
        InvalidDensityDetector,
    )
    with pytest.raises(RuntimeError, match="invalid density"):
        invalid_density.evaluate_detector_coordinates_all_roots(column_px, row_px)
    with pytest.raises(RuntimeError, match="invalid density"):
        invalid_density.evaluate_detector_density_all_roots(column_px, row_px)

    class ComplexDensityDetector(AnalyticAllRootDetector):
        def evaluate_detector_coordinates_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
            cuda_coordinate_chunk_size: int | None = None,
        ) -> object:
            result = super().evaluate_detector_coordinates_all_roots(
                column_px,
                row_px,
                execution_backend=execution_backend,
                cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
            )
            result.per_rod_density_A2_per_px2 = result.per_rod_density_A2_per_px2 + 1.0j
            result.density_A2_per_px2 = result.density_A2_per_px2 + 1.0j
            return result

    complex_density = build_incidence_angle_averaged_detector(
        quadrature,
        ComplexDensityDetector,
    )
    with pytest.raises(RuntimeError, match="complex detector density"):
        complex_density.evaluate_detector_coordinates_all_roots(column_px, row_px)

    class SourceInvalidDensityDetector(AnalyticAllRootDetector):
        def evaluate_detector_coordinates_all_roots(
            self, *args: object, **kwargs: object
        ) -> object:
            result = super().evaluate_detector_coordinates_all_roots(*args, **kwargs)
            result.valid_source_count = np.zeros_like(result.valid_source_count)
            return result

    source_invalid_density = build_incidence_angle_averaged_detector(
        quadrature,
        SourceInvalidDensityDetector,
    )
    with pytest.raises(RuntimeError, match="source-invalid"):
        source_invalid_density.evaluate_detector_density_all_roots(column_px, row_px)

    class MismatchedSourceDetector(AnalyticAllRootDetector):
        def __init__(self, incidence_angle_rad: float) -> None:
            super().__init__(incidence_angle_rad)
            self.source_revision = f"angle-dependent-source-{incidence_angle_rad:.17g}"

    with pytest.raises(ValueError, match="calibrated scan binding"):
        build_incidence_angle_averaged_detector(
            quadrature,
            MismatchedSourceDetector,
        )

    class IgnoredAngleDetector(AnalyticAllRootDetector):
        def __init__(self, incidence_angle_rad: float) -> None:
            super().__init__(incidence_angle_rad)
            self.incidence_axis_angle_rad = 0.1

    with pytest.raises(ValueError, match="do not match their quadrature angles"):
        build_incidence_angle_averaged_detector(
            quadrature,
            IgnoredAngleDetector,
        )


def test_incidence_angle_average_wraps_real_shared_source_engines() -> None:
    from rasim_next.fitting.fixed_experiment import (
        FixedPositionState,
        build_fixed_incidence_scan_series,
    )
    from rasim_next.fitting.fixed_lattice import FixedLatticeState
    from rasim_next.fitting.indexed_series import SharedGeometryCorrections
    from rasim_next.materials import read_crystal
    from rasim_next.pipeline.incidence_acquisition import IncidenceAngleQuadrature
    from rasim_next.pipeline.incidence_angle_average import IncidenceAngleAveragedDetector

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_r3_simulation.yaml")
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)
    delta = math.radians(0.4)
    position = FixedPositionState(
        artifact_revision=f"sha256-{'a' * 64}",
        corrections=SharedGeometryCorrections.zero(),
        incidence_angle_delta_rad=delta,
        commanded_incidence_angles_rad=(math.radians(5.0),),
        beam_center_column_row_px=tuple(config.instrument.detector_reference_coordinate_px),
    )
    commanded = tuple(map(math.radians, (5.0, 10.0)))
    scan = build_fixed_incidence_scan_series(
        config,
        position=position,
        fixed_lattice=lattice,
        commanded_incidence_angles_rad=commanded,
        source_sample_count=2,
        gaussian_sigma_rad=math.radians(1.0),
        lorentzian_half_width_rad=math.radians(0.5),
        lorentzian_probability=0.0,
    )
    series = scan.inputs
    components = tuple(build_source_averaged_structure_detector(item) for item in series)
    quadrature = IncidenceAngleQuadrature(
        incidence_angle_rad=np.asarray(commanded) + delta,
        exposure_probability_mass=np.asarray((0.25, 0.75)),
        exposure_density_id="real-engine-test-exposure.v1",
        incidence_angle_calibration_revision=scan.scan_calibration_revision,
    )
    detector = IncidenceAngleAveragedDetector(quadrature, components)
    changed_position = replace(position, artifact_revision=f"sha256-{'b' * 64}")
    changed_scan = build_fixed_incidence_scan_series(
        config,
        position=changed_position,
        fixed_lattice=lattice,
        commanded_incidence_angles_rad=commanded,
        source_sample_count=2,
        gaussian_sigma_rad=math.radians(1.0),
        lorentzian_half_width_rad=math.radians(0.5),
        lorentzian_probability=0.0,
    )
    changed_components = tuple(
        build_source_averaged_structure_detector(item) for item in changed_scan.inputs
    )
    with pytest.raises(ValueError, match="calibrated scan binding"):
        IncidenceAngleAveragedDetector(quadrature, changed_components)
    shifted_components = tuple(
        replace(
            component,
            incidence_axis_angle_rad=float(component.incidence_axis_angle_rad) + 0.01,
        )
        for component in components
    )
    shifted_quadrature = replace(
        quadrature,
        incidence_angle_rad=quadrature.incidence_angle_rad + 0.01,
    )
    with pytest.raises(ValueError, match="calibrated scan binding"):
        IncidenceAngleAveragedDetector(shifted_quadrature, shifted_components)
    from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope

    with pytest.raises(ValueError, match="only incidence geometry may vary"):
        IncidenceAngleAveragedDetector(
            quadrature,
            (
                components[0],
                replace(
                    components[1],
                    intensity_envelope=SampleQIntensityEnvelope(u_radial_A2=0.01),
                ),
            ),
        )
    common_rod = Rod(-1, 1)
    detector = detector.restrict_rods((common_rod,))
    components = detector.detectors
    mapped = build_nominal_ewald_context(series[0]).map_latent_geometry(
        rod=common_rod,
        branch=2,
        alpha_rad=math.radians(2.0),
        beta_rad=math.radians(178.0),
    )
    column_px = np.asarray((float(mapped.column_px),))
    row_px = np.asarray((float(mapped.row_px),))
    explicit = tuple(
        component.evaluate_detector_coordinates_all_roots(column_px, row_px)
        for component in components
    )

    evaluated = detector.evaluate_detector_coordinates_all_roots(column_px, row_px)

    np.testing.assert_allclose(
        evaluated.per_rod_density_A2_per_px2,
        sum(
            mass * value.per_rod_density_A2_per_px2
            for mass, value in zip(
                quadrature.exposure_probability_mass,
                explicit,
                strict=True,
            )
        ),
        rtol=4.0e-11,
        atol=0.0,
    )
    assert series[0].samples is series[1].samples
    assert evaluated.source_revision == series[0].samples.source_revision
    assert evaluated.angle_evaluation_count == 2
    assert np.any(evaluated.density_A2_per_px2 > 0.0)

    template = build_source_averaged_detector(series[0]).restrict_rods((common_rod,))
    fresh_second = build_source_averaged_detector(series[1]).restrict_rods((common_rod,))
    with pytest.raises(ValueError, match="calibrated incidence-scan metadata"):
        rebind_source_averaged_detector_incidence_scan(
            template,
            (build_configured_simulation_inputs(config),),
        )
    rebound_components = rebind_source_averaged_detector_incidence_scan(
        template,
        series,
    )
    second_mapped = build_nominal_ewald_context(series[1]).map_latent_geometry(
        rod=common_rod,
        branch=2,
        alpha_rad=math.radians(2.0),
        beta_rad=math.radians(178.0),
    )
    second_column_px = np.asarray((float(second_mapped.column_px),))
    second_row_px = np.asarray((float(second_mapped.row_px),))
    rebound_value = rebound_components[1].evaluate_detector_density_all_roots(
        second_column_px,
        second_row_px,
    )
    fresh_value = fresh_second.evaluate_detector_density_all_roots(
        second_column_px,
        second_row_px,
    )
    np.testing.assert_allclose(
        rebound_value.density_A2_per_px2,
        fresh_value.density_A2_per_px2,
        rtol=4.0e-11,
        atol=0.0,
    )
    np.testing.assert_array_equal(rebound_value.caustic, fresh_value.caustic)
    np.testing.assert_array_equal(
        rebound_value.valid_source_count,
        fresh_value.valid_source_count,
    )
    assert np.any(fresh_value.density_A2_per_px2 > 0.0)
    assert tuple(component.incidence_axis_angle_rad for component in rebound_components) == tuple(
        map(float, quadrature.incidence_angle_rad)
    )
    assert all(
        component.scan_calibration_binding_revision == series[0].scan_calibration_binding_revision
        for component in rebound_components
    )
    uncalibrated_rebind = template.rebind_geometry(
        incident=series[1].incident,
        instrument=series[1].instrument,
    )
    assert uncalibrated_rebind.incidence_axis_angle_rad is None
    assert uncalibrated_rebind.scan_calibration_binding_revision is None

    optimized = IncidenceAngleAveragedDetector(quadrature, rebound_components)
    optimized_total = optimized.evaluate_detector_density_all_roots(column_px, row_px)
    optimized_detailed = optimized.evaluate_detector_coordinates_all_roots(column_px, row_px)
    explicit_total = tuple(
        component.evaluate_detector_density_all_roots(column_px, row_px)
        for component in rebound_components
    )
    np.testing.assert_allclose(
        optimized_total.density_A2_per_px2,
        sum(
            mass * value.density_A2_per_px2
            for mass, value in zip(
                quadrature.exposure_probability_mass,
                explicit_total,
                strict=True,
            )
        ),
        rtol=4.0e-11,
        atol=0.0,
    )
    np.testing.assert_allclose(
        optimized_total.density_A2_per_px2,
        optimized_detailed.density_A2_per_px2,
        rtol=4.0e-11,
        atol=0.0,
    )
    np.testing.assert_allclose(
        optimized_total.density_A2_per_px2,
        evaluated.density_A2_per_px2,
        rtol=4.0e-11,
        atol=0.0,
    )
    np.testing.assert_allclose(
        optimized_total.valid_incident_state_fraction,
        sum(
            mass * value.valid_source_count / value.source_state_count
            for mass, value in zip(
                quadrature.exposure_probability_mass,
                explicit_total,
                strict=True,
            )
        ),
        rtol=2.0e-15,
        atol=0.0,
    )


def _forward_monte_carlo_fixture(
    *,
    source_count: int = 1,
    worker_count: int = 1,
    sample_angle_deg: float = 5.0,
    detector_path_linear_attenuation_m_inv: float = 0.0,
    detector_path_wavelength_A: tuple[float, ...] = (),
    detector_path_linear_attenuation_m_inv_by_wavelength: tuple[float, ...] = (),
) -> tuple[object, tuple[Rod, ...], object]:
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    rotations = (
        replace(config.instrument.axis_rotations[0], angle_deg=sample_angle_deg),
        *config.instrument.axis_rotations[1:],
    )
    inputs = build_configured_simulation_inputs(
        replace(
            config,
            source=(
                _single_mean_test_source(config.source)
                if source_count == 1
                else replace(config.source, sample_count=source_count)
            ),
            instrument=replace(
                config.instrument,
                axis_rotations=rotations,
                detector_shape_rc=(64, 64),
                detector_row_pitch_m=2.0e-3,
                detector_column_pitch_m=2.0e-3,
                detector_reference_coordinate_px=(31.5, 31.5),
                detector_path_medium_id=(
                    "standard_dry_air_sensitivity.v1"
                    if detector_path_linear_attenuation_m_inv or detector_path_wavelength_A
                    else "vacuum_or_helium_unity.v1"
                ),
                detector_path_linear_attenuation_m_inv=(detector_path_linear_attenuation_m_inv),
                detector_path_wavelength_A=detector_path_wavelength_A,
                detector_path_linear_attenuation_m_inv_by_wavelength=(
                    detector_path_linear_attenuation_m_inv_by_wavelength
                ),
            ),
            bragg=replace(config.bragg, rod_population=0.37),
            weights=replace(config.weights, phase_population=0.41, polarization=0.73),
            numerics=replace(config.numerics, worker_count=worker_count),
        )
    )
    rods = (
        next(rod for rod in inputs.rods if (rod.h, rod.k) == (0, 0)),
        next(rod for rod in inputs.rods if (rod.h, rod.k) == (-1, 1)),
    )
    detector = build_source_averaged_detector(inputs).restrict_rods(rods)
    return inputs, rods, detector


def test_monte_carlo_pixel_mass_matches_the_forward_latent_oracle() -> None:
    from painted_ewald import wrapped_mosaic_line_density_rad_inv
    from rasim_next.core.scattering import scattering_polarization_weight
    from rasim_next.optics.attenuation import (
        external_path_attenuation,
        incident_illuminated_path_weight,
        mode_decay_constant,
        scalar_optical_weight,
        uniform_depth_attenuation,
    )
    from rasim_next.optics.refraction import solve_exit_mode

    detector_path_mu_m_inv = 1.2
    inputs, rods, detector = _forward_monte_carlo_fixture(
        detector_path_linear_attenuation_m_inv=detector_path_mu_m_inv
    )
    seed = 3276
    draw_count = 3
    sampled = detector.sample_native_pixel_mass(
        draws_per_source_state=draw_count,
        seed=seed,
    )

    philox_key = np.random.SeedSequence(seed).generate_state(2, dtype=np.uint64)
    latent_uniform = np.vstack(
        [
            np.random.Generator(
                np.random.Philox(
                    key=philox_key,
                    counter=draw_index << 64,
                )
            ).random((1, 8))[0, :5]
            for draw_index in range(draw_count)
        ]
    )
    gaussian_radius = np.sqrt(-2.0 * np.log(np.maximum(latent_uniform[:, 1], np.finfo(float).tiny)))
    signed_tilt = (
        inputs.mosaic.gaussian_sigma_rad
        * gaussian_radius
        * np.cos(2.0 * np.pi * latent_uniform[:, 2])
    )
    alpha = np.abs((signed_tilt + np.pi) % (2.0 * np.pi) - np.pi)
    beta = 2.0 * np.pi * latent_uniform[:, 4]

    oracle = _physical_scalar_detector(inputs)
    source_phase_weight = float(
        inputs.incident.states.source_weight[0]
        * inputs.incident.states.footprint_acceptance[0]
        * incident_illuminated_path_weight(inputs.incident.states.direction_sample[0])
        * inputs.config.weights.phase_population
        * inputs.config.weights.polarization
    )
    expected_mass: dict[int, float] = {}
    expected_roots: set[tuple[int, int, int]] = set()
    expected_hit_count = 0
    expected_replicate_total = np.zeros(draw_count, dtype=np.float64)
    _, columns = inputs.instrument.detector_shape_rc
    for draw in range(draw_count):
        proposal_density = float(
            2.0 * wrapped_mosaic_line_density_rad_inv(alpha[draw], inputs.mosaic) / (2.0 * np.pi)
        )
        for rod in rods:
            for branch in (0,) if rod.family_m == 0 else (1, 2):
                if rod.family_m == 0:
                    coating = oracle.map_detector_visible_coating(
                        rod=rod,
                        branch=branch,
                        alpha_rad=alpha[draw],
                        beta_rad=beta[draw],
                    )
                    geometry = coating.geometry
                    if not bool(geometry.valid):
                        continue
                    exit_mode = solve_exit_mode(
                        geometry.ewald_geometry.kf_sample_Ainv,
                        inputs.samples.wavelength_A[0],
                        inputs.material,
                    )
                    attenuation = uniform_depth_attenuation(
                        mode_decay_constant(inputs.incident.states.kz_film_Ainv[0], -1),
                        mode_decay_constant(exit_mode.kz_film_Ainv, 1),
                        inputs.instrument.film_thickness_A,
                    )
                    optical = scalar_optical_weight(
                        inputs.incident.states.entrance_amplitude[0],
                        exit_mode.exit_amplitude,
                        attenuation,
                    )
                    polarization = scattering_polarization_weight(
                        inputs.incident.states.direction_sample[0],
                        exit_mode.k_air_phase_sample_Ainv
                        / (2.0 * np.pi / inputs.samples.wavelength_A[0]),
                        model_id=inputs.incident.states.polarization_state_id[0],
                    )
                    weight = (
                        float(coating.coating_intensity_density_A2_rad2_inv)
                        * optical
                        * external_path_attenuation(
                            detector_path_mu_m_inv,
                            geometry.ray_distance_m,
                        )
                        * polarization
                        * source_phase_weight
                        / proposal_density
                    )
                else:
                    mapped = oracle.map_latent(
                        rod=rod,
                        branch=branch,
                        alpha_rad=alpha[draw],
                        beta_rad=beta[draw],
                    )
                    geometry = mapped.geometry
                    if not bool(geometry.valid):
                        continue
                    weight = float(mapped.postoptical_density_A2_rad2_inv) / proposal_density
                if weight == 0.0:
                    continue
                expected_roots.add((rod.h, rod.k, branch))
                column = math.floor(float(geometry.column_px) + 0.5)
                row = math.floor(float(geometry.row_px) + 0.5)
                flat_index = row * columns + column
                expected_mass[flat_index] = expected_mass.get(flat_index, 0.0) + weight / draw_count
                expected_hit_count += 1
                expected_replicate_total[draw] += weight

    assert sampled.measure_id == "raw_detector_pixel_mass_monte_carlo_estimate_A2.v1"
    assert sampled.rng_model_id == "numpy.philox.fixed_width_source_draw.v1"
    assert sampled.execution_backend == "numba_cpu_forward_monte_carlo.v2"
    assert sampled.execution_device is None
    assert sampled.detector_visible_m0_q_gap_Ainv == detector.detector_visible_m0_q_gap_Ainv
    assert {(0, 0, 0), (-1, 1, 1), (-1, 1, 2)} <= expected_roots
    assert sampled.attempted_root_count == 3 * draw_count
    expected_index = np.asarray(sorted(expected_mass), dtype=np.int64)
    np.testing.assert_array_equal(np.flatnonzero(sampled.image_A2), expected_index)
    np.testing.assert_allclose(
        sampled.image_A2.ravel()[expected_index],
        [expected_mass[index] for index in expected_index],
        rtol=1.0e-8,
        atol=5.0e-19,
    )
    np.testing.assert_allclose(
        sampled.replicate_total_mass_A2,
        expected_replicate_total,
        rtol=1.0e-8,
        atol=5.0e-19,
    )
    assert sampled.total_detector_mass_A2 == pytest.approx(
        math.fsum(expected_mass.values()),
        rel=1.0e-8,
        abs=5.0e-19,
    )
    assert sampled.visible_hit_count == expected_hit_count


def test_forward_monte_carlo_is_prefix_stable_and_worker_order_invariant() -> None:
    from rasim_next.pipeline.source_averaged_detector import _sample_mosaic_orientation_matrix

    _, _, serial_detector = _forward_monte_carlo_fixture(source_count=5, worker_count=1)
    _, _, parallel_detector = _forward_monte_carlo_fixture(source_count=5, worker_count=12)
    seed = 9182

    small_alpha, small_beta = _sample_mosaic_orientation_matrix(
        serial_detector.mosaic,
        5,
        seed=seed,
        source_state_count=3,
    )
    large_alpha, large_beta = _sample_mosaic_orientation_matrix(
        serial_detector.mosaic,
        5,
        seed=seed,
        source_state_count=5,
    )
    np.testing.assert_array_equal(small_alpha, large_alpha[:3])
    np.testing.assert_array_equal(small_beta, large_beta[:3])
    suffix_alpha, suffix_beta = _sample_mosaic_orientation_matrix(
        serial_detector.mosaic,
        5,
        seed=seed,
        source_state_count=5,
        draw_start=3,
    )
    np.testing.assert_array_equal(suffix_alpha, large_alpha[:, 3:])
    np.testing.assert_array_equal(suffix_beta, large_beta[:, 3:])

    first_three = serial_detector.sample_native_pixel_mass(
        draws_per_source_state=3,
        seed=seed,
    )
    first_five = serial_detector.sample_native_pixel_mass(
        draws_per_source_state=5,
        seed=seed,
    )
    np.testing.assert_array_equal(
        first_five.replicate_total_mass_A2[:3],
        first_three.replicate_total_mass_A2,
    )
    added_raw_mass = 5.0 * first_five.image_A2 - 3.0 * first_three.image_A2
    assert float(np.min(added_raw_mass)) >= -2.0e-15 * max(
        float(np.max(5.0 * first_five.image_A2)),
        1.0,
    )

    parallel = parallel_detector.sample_native_pixel_mass(
        draws_per_source_state=5,
        seed=seed,
    )
    np.testing.assert_array_equal(parallel.image_A2, first_five.image_A2)
    np.testing.assert_array_equal(
        parallel.replicate_total_mass_A2,
        first_five.replicate_total_mass_A2,
    )
    assert parallel.total_detector_mass_A2 == first_five.total_detector_mass_A2
    assert parallel.attempted_root_count == first_five.attempted_root_count
    assert parallel.visible_hit_count == first_five.visible_hit_count
    assert parallel.maximum_root_deposit_A2 == first_five.maximum_root_deposit_A2
    assert first_five.execution_worker_count == 1
    assert 1 < parallel.execution_worker_count <= 4

    progressive = parallel_detector.compile_monte_carlo_sampler(
        execution_backend="cpu",
        seed=seed,
    )
    preview_one = progressive.advance_preview_to(1)
    stage_one = progressive.advance_to(1)
    assert preview_one.image_A2.dtype == np.float32
    assert preview_one.image_A2.shape == stage_one.image_A2.shape
    np.testing.assert_array_equal(preview_one.image_A2, stage_one.image_A2.astype(np.float32))
    stage_four = progressive.advance_to(4)
    settled = progressive.advance_to(5)
    np.testing.assert_array_equal(
        stage_four.replicate_total_mass_A2[:1],
        stage_one.replicate_total_mass_A2,
    )
    np.testing.assert_allclose(
        settled.image_A2,
        parallel.image_A2,
        rtol=32.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        settled.replicate_total_mass_A2,
        parallel.replicate_total_mass_A2,
    )
    assert settled.image_A2.base is None
    assert not settled.image_A2.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        settled.image_A2[0, 0] = 0.0

    changed_topology = parallel_detector.restrict_rods((serial_detector.rods[0],))
    with pytest.raises(ValueError, match="unchanged source, rods, physics"):
        progressive.rebind_geometry(changed_topology)

    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    rebound_instrument = viewer["apply_geometry_deltas"](
        parallel_detector.instrument,
        viewer["GeometryDeltas"](
            detector_pitch_offset_deg=0.2,
            detector_row_translation_mm=0.3,
        ),
    )
    rebound = parallel_detector.rebind_geometry(
        incident=parallel_detector.incident,
        instrument=rebound_instrument,
    )
    progressive.rebind_detector_pose(rebound_instrument)
    fast_pose = progressive.advance_to(5)
    fresh_pose = rebound.sample_native_pixel_mass(
        draws_per_source_state=5,
        seed=seed,
        execution_backend="cpu",
    )
    np.testing.assert_array_equal(fast_pose.image_A2, fresh_pose.image_A2)
    np.testing.assert_array_equal(
        fast_pose.replicate_total_mass_A2,
        fresh_pose.replicate_total_mass_A2,
    )
    changed_sample_instrument = viewer["apply_geometry_deltas"](
        parallel_detector.instrument,
        viewer["GeometryDeltas"](effective_incidence_angle_offset_deg=0.1),
    )
    with pytest.raises(ValueError, match="unchanged detector calibration and sample pose"):
        progressive.rebind_detector_pose(changed_sample_instrument)


def test_forward_monte_carlo_honors_a_cancelled_request() -> None:
    from rasim_next.pipeline.source_averaged_detector import MonteCarloSamplingCancelled

    _, _, detector = _forward_monte_carlo_fixture()
    with pytest.raises(MonteCarloSamplingCancelled, match="cancelled"):
        detector.sample_native_pixel_mass(
            draws_per_source_state=3,
            seed=7,
            cancel_requested=lambda: True,
        )


def test_bi2te3_compiled_detector_uses_te_factors() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2te3_simulation.yaml")
    inputs = build_configured_simulation_inputs(
        replace(config, source=_single_mean_test_source(config.source))
    )
    context = build_nominal_ewald_context(inputs)
    scalar = _physical_scalar_detector(inputs)
    markers = evaluate_nominal_integer_l_markers(context)
    selected = int(
        np.flatnonzero((markers.family_m == 1) & (markers.branch == 2) & (markers.root_sign != 0))[
            0
        ]
    )
    rods = tuple(rod for rod in inputs.rods if rod.family_m == 1)
    column = np.asarray([markers.column_px[selected]])
    row = np.asarray([markers.row_px[selected]])

    direct = scalar.evaluate_detector_coordinates(
        column,
        row,
        rods=rods,
    )
    compiled, count, caustic, _ = scalar._compiled_evaluator(rods).evaluate(
        column,
        row,
        branch=2,
    )

    np.testing.assert_allclose(
        compiled,
        direct.per_rod_density_A2_per_px2,
        rtol=3.0e-12,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(count, direct.per_rod_inverse_branch_count)
    np.testing.assert_array_equal(caustic, direct.caustic)


def _intrinsic_detector_tilt(
    base_rotation: np.ndarray,
    column_deg: float,
    row_deg: float,
) -> np.ndarray:
    column = math.radians(column_deg)
    row = math.radians(row_deg)
    about_column = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(column), -math.sin(column)),
            (0.0, math.sin(column), math.cos(column)),
        )
    )
    about_current_row = np.asarray(
        (
            (math.cos(row), 0.0, math.sin(row)),
            (0.0, 1.0, 0.0),
            (-math.sin(row), 0.0, math.cos(row)),
        )
    )
    return np.asarray(base_rotation) @ about_column @ about_current_row


def test_continuous_upper_m1_maps_through_canonical_exit_before_pixel_binning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numba import cuda

    from painted_ewald import (
        BraggSpaceConfig,
        ContinuousEwaldCoating,
        MosaicBraggSpace,
        MosaicParameters,
        Rod,
    )
    from rasim_next.geometry import project_detector_ray
    from rasim_next.optics.attenuation import (
        external_path_attenuation,
        incident_illuminated_path_weight,
        mode_decay_constant,
        scalar_optical_weight,
        uniform_depth_attenuation,
    )
    from rasim_next.optics.refraction import solve_exit_mode
    from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
    from rasim_next.pipeline.continuous_detector import (
        DetectorEwaldMeasure,
        DetectorQuadrature,
        IntensityStatus,
        PixelIntegrationMethod,
    )
    from rasim_next.pipeline.source_averaged_detector import (
        SourceAveragedDetectorEwaldMeasure,
    )

    detector_path_mu_m_inv = 1.2
    inputs = _configured_inputs(
        sample_count=1,
        detector_path_linear_attenuation_m_inv=detector_path_mu_m_inv,
        reference_wavelength_A=1.540592925,
    )
    samples = inputs.samples
    instrument = inputs.instrument
    crystal = inputs.crystal
    material = inputs.material
    incident = inputs.incident
    reciprocal = inputs.reciprocal
    rods = tuple(
        Rod(h, k)
        for h, k in (
            (0, 0),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
        )
    )
    air_k0_Ainv = 2.0 * np.pi / samples.wavelength_A[0]
    bragg = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=reciprocal.basis_Ainv,
            crystal_to_sample=instrument.sample_from_crystal.rotation,
            rods=rods,
            mosaic=MosaicParameters(
                gaussian_sigma_rad=math.radians(5.0),
                lorentzian_half_width_rad=math.radians(2.0),
                lorentzian_probability=0.1,
                alpha_panel_count=12,
                alpha_gauss_order=6,
                azimuth_count=32,
            ),
            k_norm_Ainv=air_k0_Ainv,
        ),
        Bi2X3FiniteStackStrength(
            crystal=crystal,
            layers=7,
            normalization=EventIntensityNormalization.FINITE_TOTAL,
        ),
    )
    coating = ContinuousEwaldCoating(
        bragg,
        ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
    )
    detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=instrument,
    )
    rod = Rod(-1, 1)
    alpha_rad = math.radians(15.0)
    beta_rad = 2.21704764194

    mapped = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
    )

    assert mapped.intensity_status is IntensityStatus.INCLUDED
    assert bool(mapped.geometry.valid)
    assert mapped.geometry.column_px == pytest.approx(1267.56918, abs=2.0e-5)
    assert mapped.geometry.row_px == pytest.approx(577.77153, abs=2.0e-5)
    latent = coating.evaluate_latent(
        rod=rod,
        branch=2,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
    )
    exit_mode = solve_exit_mode(
        latent.geometry.kf_sample_Ainv,
        samples.wavelength_A[0],
        material,
    )
    incident_kappa = mode_decay_constant(
        incident.states.kz_film_Ainv[0],
        -1,
    )
    exit_kappa = mode_decay_constant(exit_mode.kz_film_Ainv, 1)
    attenuation = uniform_depth_attenuation(
        incident_kappa,
        exit_kappa,
        instrument.film_thickness_A,
    )
    optical = scalar_optical_weight(
        incident.states.entrance_amplitude[0],
        exit_mode.exit_amplitude,
        attenuation,
    )
    expected_kf_air_lab = instrument.lab_from_sample.apply_vector(exit_mode.k_air_phase_sample_Ainv)
    projection = project_detector_ray(
        incident.states.sample_intersection_lab_m[0],
        expected_kf_air_lab / air_k0_Ainv,
        instrument,
    )
    np.testing.assert_allclose(
        mapped.geometry.kf_air_lab_Ainv,
        expected_kf_air_lab,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert mapped.geometry.column_px == pytest.approx(projection.column_px, abs=2.0e-12)
    assert mapped.geometry.row_px == pytest.approx(projection.row_px, abs=2.0e-12)
    assert mapped.attenuation_weight == pytest.approx(attenuation, abs=2.0e-15)
    path_weight = external_path_attenuation(
        detector_path_mu_m_inv,
        mapped.geometry.ray_distance_m,
    )
    assert mapped.optical_weight == pytest.approx(optical * path_weight, abs=2.0e-15)
    expected_source_phase_weight = (
        incident.states.source_weight[0]
        * incident.states.footprint_acceptance[0]
        * incident_illuminated_path_weight(incident.states.direction_sample[0])
    )
    assert mapped.source_phase_weight == expected_source_phase_weight
    assert mapped.postoptical_density_A2_rad2_inv == pytest.approx(
        latent.coating_intensity_density_A2_rad2_inv
        * optical
        * path_weight
        * mapped.scattering_polarization_weight
        * expected_source_phase_weight,
        rel=0.0,
        abs=2.0e-20,
    )
    assert mapped.postoptical_density_A2_rad2_inv != pytest.approx(
        latent.coating_intensity_density_A2_rad2_inv
        * optical
        * mapped.geometry.pixel_solid_angle_sr,
        rel=1.0e-6,
        abs=0.0,
    )

    caustic_density = detector.evaluate_detector_coordinates(
        mapped.geometry.column_px,
        mapped.geometry.row_px,
        rods=(rod,),
    )
    assert bool(caustic_density.caustic)
    assert np.isinf(caustic_density.density_A2_per_px2)
    internal_k = float(np.linalg.norm(coating.ki_sample_Ainv))
    caustic_direction = latent.geometry.kf_sample_Ainv / internal_k
    intrinsic_caustic = detector.evaluate_intrinsic_ewald_directions(
        caustic_direction,
        rods=(rod,),
        branch=2,
    )
    assert bool(intrinsic_caustic.caustic)
    assert np.isposinf(intrinsic_caustic.density_A2_per_sr)

    m0_rod = Rod(0, 0)
    m0_bragg = MosaicBraggSpace(replace(bragg.config, rods=(m0_rod,)), bragg.strength_model)
    m0_detector = DetectorEwaldMeasure(
        coating=ContinuousEwaldCoating(
            m0_bragg,
            ki_sample_Ainv=coating.ki_sample_Ainv,
        ),
        incident=incident,
        material=material,
        instrument=instrument,
    )
    with pytest.raises(ValueError, match="m=0 intensity is excluded"):
        m0_detector.evaluate_intrinsic_ewald_directions(
            caustic_direction,
            rods=(m0_rod,),
        )

    zero_rod = Rod(rod.h, rod.k, population=0.0)
    zero_bragg = MosaicBraggSpace(replace(bragg.config, rods=(zero_rod,)), bragg.strength_model)
    zero_intrinsic_detector = DetectorEwaldMeasure(
        coating=ContinuousEwaldCoating(
            zero_bragg,
            ki_sample_Ainv=coating.ki_sample_Ainv,
        ),
        incident=incident,
        material=material,
        instrument=instrument,
    )
    zero_intrinsic_caustic = zero_intrinsic_detector.evaluate_intrinsic_ewald_directions(
        caustic_direction,
        rods=(zero_rod,),
        branch=2,
    )
    assert bool(zero_intrinsic_caustic.caustic)
    assert zero_intrinsic_caustic.density_A2_per_sr == 0.0
    zero_weight_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=instrument,
        polarization_weight=0.0,
    )
    zero_caustic_density = zero_weight_detector.evaluate_detector_coordinates(
        mapped.geometry.column_px,
        mapped.geometry.row_px,
        rods=(rod,),
    )
    assert bool(zero_caustic_density.caustic)
    assert zero_caustic_density.density_A2_per_px2 == 0.0
    tiny_weight_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=instrument,
        polarization_weight=1.0e-300,
    )
    tiny_caustic_density = tiny_weight_detector.evaluate_detector_coordinates(
        mapped.geometry.column_px,
        mapped.geometry.row_px,
        rods=(rod,),
    )
    assert bool(tiny_caustic_density.caustic)
    assert np.isinf(tiny_caustic_density.density_A2_per_px2)

    caustic_rods = (Rod(0, 0), rod)
    averaged_caustic_detector = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=bragg.config.reciprocal_basis_Ainv,
        crystal_to_sample=instrument.sample_from_crystal.rotation,
        rods=caustic_rods,
        rod_catalog_revision="integration-test-rods.v1",
        mosaic=bragg.config.mosaic,
        strength_model=bragg.strength_model,
        incident=incident,
        material=material,
        instrument=instrument,
    )
    detailed_caustic = averaged_caustic_detector.evaluate_detector_coordinates_all_roots(
        np.asarray([mapped.geometry.column_px]),
        np.asarray([mapped.geometry.row_px]),
        execution_backend="cpu",
    )
    assert not detailed_caustic.caustic[0, 0]
    assert detailed_caustic.caustic[0, 1]
    averaged_cpu_caustic = averaged_caustic_detector.evaluate_detector_density_all_roots(
        np.asarray([mapped.geometry.column_px]),
        np.asarray([mapped.geometry.row_px]),
        execution_backend="cpu",
    )
    assert averaged_cpu_caustic.caustic.item()
    assert np.isinf(averaged_cpu_caustic.density_A2_per_px2.item())

    averaged_zero_caustic_detector = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=bragg.config.reciprocal_basis_Ainv,
        crystal_to_sample=instrument.sample_from_crystal.rotation,
        rods=caustic_rods,
        rod_catalog_revision="integration-test-rods.v1",
        mosaic=bragg.config.mosaic,
        strength_model=bragg.strength_model,
        incident=incident,
        material=material,
        instrument=instrument,
        phase_population_weight=0.0,
    )
    averaged_zero_cpu_caustic = averaged_zero_caustic_detector.evaluate_detector_density_all_roots(
        np.asarray([mapped.geometry.column_px]),
        np.asarray([mapped.geometry.row_px]),
        execution_backend="cpu",
    )
    assert averaged_zero_cpu_caustic.caustic.item()
    assert averaged_zero_cpu_caustic.density_A2_per_px2.item() == 0.0

    if cuda.is_available():
        averaged_cuda_caustic = averaged_caustic_detector.evaluate_detector_density_all_roots(
            np.asarray([mapped.geometry.column_px]),
            np.asarray([mapped.geometry.row_px]),
            execution_backend="cuda",
        )
        assert averaged_cuda_caustic.caustic.item()
        assert np.isinf(averaged_cuda_caustic.density_A2_per_px2.item())
        np.testing.assert_array_equal(
            averaged_cuda_caustic.valid_source_count,
            averaged_cpu_caustic.valid_source_count,
        )
        averaged_zero_cuda_caustic = (
            averaged_zero_caustic_detector.evaluate_detector_density_all_roots(
                np.asarray([mapped.geometry.column_px]),
                np.asarray([mapped.geometry.row_px]),
                execution_backend="cuda",
            )
        )
        assert averaged_zero_cuda_caustic.caustic.item()
        assert averaged_zero_cuda_caustic.density_A2_per_px2.item() == 0.0
        np.testing.assert_array_equal(
            averaged_zero_cuda_caustic.valid_source_count,
            averaged_zero_cpu_caustic.valid_source_count,
        )

    regular_alpha = math.radians(2.0)
    regular_beta = math.radians(178.0)
    regular_mapped = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=regular_alpha,
        beta_rad=regular_beta,
    )
    assert bool(regular_mapped.geometry.valid)
    step = 1.0e-6
    alpha_pair = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=np.array([regular_alpha - step, regular_alpha + step]),
        beta_rad=regular_beta,
    )
    beta_pair = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=regular_alpha,
        beta_rad=np.array([regular_beta - step, regular_beta + step]),
    )
    d_column_d_alpha = np.diff(alpha_pair.geometry.column_px)[0] / (2.0 * step)
    d_row_d_alpha = np.diff(alpha_pair.geometry.row_px)[0] / (2.0 * step)
    d_column_d_beta = np.diff(beta_pair.geometry.column_px)[0] / (2.0 * step)
    d_row_d_beta = np.diff(beta_pair.geometry.row_px)[0] / (2.0 * step)
    detector_jacobian = abs(d_column_d_alpha * d_row_d_beta - d_column_d_beta * d_row_d_alpha)
    regular_density = detector.evaluate_detector_coordinates(
        regular_mapped.geometry.column_px,
        regular_mapped.geometry.row_px,
        rods=(rod,),
    )
    internal_k = np.linalg.norm(incident.states.k_film_phase_sample_Ainv[0])
    expected_surface_jacobian = (
        internal_k
        * air_k0_Ainv
        * regular_density.geometry.kf_air_sample_Ainv[2]
        / regular_density.geometry.kf_film_sample_Ainv[2]
        * regular_mapped.geometry.pixel_solid_angle_sr
    )
    assert regular_density.geometry.q_surface_jacobian_Ainv2_per_px2 == pytest.approx(
        expected_surface_jacobian,
        rel=2.0e-14,
    )
    assert np.linalg.norm(
        regular_density.geometry.q_sample_Ainv + incident.states.k_film_phase_sample_Ainv[0]
    ) == pytest.approx(internal_k, abs=3.0e-15)
    assert regular_density.per_rod_inverse_branch_count == 1
    assert not bool(regular_density.caustic)
    assert regular_density.density_A2_per_px2 == pytest.approx(
        regular_mapped.postoptical_density_A2_rad2_inv / detector_jacobian,
        rel=2.0e-8,
    )

    regular_latent = coating.evaluate_latent(
        rod=rod,
        branch=2,
        alpha_rad=regular_alpha,
        beta_rad=regular_beta,
    )
    alpha_direction = alpha_pair.geometry.ewald_geometry.kf_sample_Ainv / internal_k
    beta_direction = beta_pair.geometry.ewald_geometry.kf_sample_Ainv / internal_k
    direction_d_alpha = (alpha_direction[1] - alpha_direction[0]) / (2.0 * step)
    direction_d_beta = (beta_direction[1] - beta_direction[0]) / (2.0 * step)
    solid_angle_jacobian = np.linalg.norm(np.cross(direction_d_alpha, direction_d_beta))
    outgoing_direction = regular_latent.geometry.kf_sample_Ainv / internal_k
    intrinsic = detector.evaluate_intrinsic_ewald_directions(
        outgoing_direction,
        rods=(rod,),
        branch=2,
    )
    np.testing.assert_allclose(
        intrinsic.q_sample_Ainv,
        regular_latent.geometry.q_sample_Ainv,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert intrinsic.per_rod_inverse_branch_count == 1
    assert not bool(intrinsic.caustic)
    assert intrinsic.density_A2_per_sr == pytest.approx(
        regular_latent.coating_intensity_density_A2_rad2_inv / solid_angle_jacobian,
        rel=5.0e-8,
    )
    assert intrinsic.density_A2_per_sr == pytest.approx(
        regular_density.density_A2_per_px2
        * internal_k**2
        / regular_density.geometry.q_surface_jacobian_Ainv2_per_px2
        / regular_mapped.optical_weight
        / regular_mapped.scattering_polarization_weight
        / regular_mapped.source_phase_weight,
        rel=3.0e-13,
    )
    assert intrinsic.measure_id == "intrinsic_ewald_direction_density_A2_per_sr.v1"
    assert not intrinsic.density_A2_per_sr.flags.writeable

    detector_pose = instrument.lab_from_detector
    flipped_instrument = replace(
        instrument,
        lab_from_detector=RigidTransform(
            detector_pose.rotation @ np.diag((1.0, -1.0, -1.0)),
            detector_pose.translation_m,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
    )
    flipped_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=flipped_instrument,
    )
    _, reference_row = instrument.detector_reference_coordinate_px
    back_facing_column = np.asarray([regular_mapped.geometry.column_px])
    back_facing_row = np.asarray([2.0 * reference_row - regular_mapped.geometry.row_px])
    back_facing_numpy = flipped_detector.evaluate_detector_coordinates(
        back_facing_column,
        back_facing_row,
        rods=(rod,),
        branch=2,
    )
    assert back_facing_numpy.geometry.status.item() == "BACKWARD"
    assert back_facing_numpy.density_A2_per_px2.item() == 0.0
    compiled_back_facing = flipped_detector._compiled_evaluator((rod,)).evaluate(
        back_facing_column,
        back_facing_row,
        branch=2,
    )
    assert not compiled_back_facing[3].item()
    assert compiled_back_facing[0].item() == 0.0

    two_branch_seed = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=math.radians(10.0),
        beta_rad=2.0,
    )
    two_branch_density = detector.evaluate_detector_coordinates(
        two_branch_seed.geometry.column_px,
        two_branch_seed.geometry.row_px,
        rods=(rod,),
    )
    assert two_branch_density.per_rod_inverse_branch_count == 2
    oracle_contributions = []
    for inverse_alpha, inverse_beta in (
        (0.17453292519943342, 2.000000000000001),
        (0.34730205158064265, 2.3814947269767046),
    ):
        inverse_forward = detector.map_latent(
            rod=rod,
            branch=2,
            alpha_rad=inverse_alpha,
            beta_rad=inverse_beta,
        )
        assert inverse_forward.geometry.column_px == pytest.approx(
            two_branch_seed.geometry.column_px, abs=5.0e-10
        )
        assert inverse_forward.geometry.row_px == pytest.approx(
            two_branch_seed.geometry.row_px, abs=5.0e-10
        )
        alpha_pair = detector.map_latent(
            rod=rod,
            branch=2,
            alpha_rad=np.array([inverse_alpha - step, inverse_alpha + step]),
            beta_rad=inverse_beta,
        )
        beta_pair = detector.map_latent(
            rod=rod,
            branch=2,
            alpha_rad=inverse_alpha,
            beta_rad=np.remainder(
                np.array([inverse_beta - step, inverse_beta + step]),
                2.0 * np.pi,
            ),
        )
        dc_da = np.diff(alpha_pair.geometry.column_px)[0] / (2.0 * step)
        dr_da = np.diff(alpha_pair.geometry.row_px)[0] / (2.0 * step)
        dc_db = np.diff(beta_pair.geometry.column_px)[0] / (2.0 * step)
        dr_db = np.diff(beta_pair.geometry.row_px)[0] / (2.0 * step)
        forward_jacobian = abs(dc_da * dr_db - dc_db * dr_da)
        oracle_contributions.append(
            float(inverse_forward.postoptical_density_A2_rad2_inv / forward_jacobian)
        )
    assert two_branch_density.density_A2_per_px2 == pytest.approx(
        math.fsum(oracle_contributions),
        rel=5.0e-8,
        abs=0.0,
    )
    sphere_direction = two_branch_seed.geometry.ewald_geometry.kf_sample_Ainv / internal_k
    sphere_all = detector.evaluate_intrinsic_ewald_directions(
        sphere_direction,
        rods=(rod,),
    )
    sphere_lower = detector.evaluate_intrinsic_ewald_directions(
        sphere_direction,
        rods=(rod,),
        branch=1,
    )
    sphere_upper = detector.evaluate_intrinsic_ewald_directions(
        sphere_direction,
        rods=(rod,),
        branch=2,
    )
    assert sphere_all.branch is None
    assert sphere_all.per_rod_inverse_branch_count == 4
    np.testing.assert_array_equal(
        sphere_all.per_rod_inverse_branch_count,
        sphere_lower.per_rod_inverse_branch_count + sphere_upper.per_rod_inverse_branch_count,
    )
    np.testing.assert_allclose(
        sphere_all.density_A2_per_sr,
        sphere_lower.density_A2_per_sr + sphere_upper.density_A2_per_sr,
        rtol=0.0,
        atol=0.0,
    )
    sphere_preimages = {
        1: (
            (2.7942906020091502, 2.0),
            (2.96705972839036, 2.3814947269767046),
        ),
        2: (
            (0.17453292519943298, 2.0),
            (0.34730205158064287, 2.3814947269767046),
        ),
    }
    for branch, preimages in sphere_preimages.items():
        branch_oracle = []
        for inverse_alpha, inverse_beta in preimages:
            inverse_forward = coating.evaluate_latent(
                rod=rod,
                branch=branch,
                alpha_rad=inverse_alpha,
                beta_rad=inverse_beta,
            )
            np.testing.assert_allclose(
                inverse_forward.geometry.q_sample_Ainv,
                sphere_all.q_sample_Ainv,
                rtol=0.0,
                atol=8.0e-15,
            )
            alpha_pair = coating.evaluate_latent(
                rod=rod,
                branch=branch,
                alpha_rad=np.asarray((inverse_alpha - step, inverse_alpha + step)),
                beta_rad=inverse_beta,
            )
            beta_pair = coating.evaluate_latent(
                rod=rod,
                branch=branch,
                alpha_rad=inverse_alpha,
                beta_rad=np.remainder(
                    np.asarray((inverse_beta - step, inverse_beta + step)),
                    2.0 * np.pi,
                ),
            )
            direction_d_alpha = np.diff(
                alpha_pair.geometry.kf_sample_Ainv / internal_k,
                axis=0,
            )[0] / (2.0 * step)
            direction_d_beta = np.diff(
                beta_pair.geometry.kf_sample_Ainv / internal_k,
                axis=0,
            )[0] / (2.0 * step)
            solid_angle_jacobian = np.linalg.norm(np.cross(direction_d_alpha, direction_d_beta))
            branch_oracle.append(
                float(inverse_forward.coating_intensity_density_A2_rad2_inv / solid_angle_jacobian)
            )
        branch_result = sphere_lower if branch == 1 else sphere_upper
        assert branch_result.per_rod_inverse_branch_count == 2
        assert branch_result.density_A2_per_sr == pytest.approx(
            math.fsum(branch_oracle),
            rel=5.0e-8,
            abs=0.0,
        )
    near_unit = detector.evaluate_intrinsic_ewald_directions(
        sphere_direction * (1.0 + 5.0e-13),
        rods=(rod,),
    )
    np.testing.assert_allclose(
        np.linalg.norm(near_unit.outgoing_direction_sample),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.linalg.norm(near_unit.q_sample_Ainv + coating.ki_sample_Ainv),
        internal_k,
        rtol=0.0,
        atol=2.0e-15,
    )
    mixed_density = detector.evaluate_detector_coordinates(
        np.array([regular_mapped.geometry.column_px, 0.0]),
        np.array([regular_mapped.geometry.row_px, 0.0]),
        rods=(rod,),
    )
    np.testing.assert_array_equal(
        mixed_density.per_rod_inverse_branch_count[0],
        regular_density.per_rod_inverse_branch_count,
    )
    np.testing.assert_allclose(
        mixed_density.density_A2_per_px2[0],
        regular_density.density_A2_per_px2,
        rtol=0.0,
        atol=0.0,
    )

    seam_rod = Rod(0, -1)
    seam_seed = detector.map_latent(
        rod=seam_rod,
        branch=2,
        alpha_rad=math.radians(0.2),
        beta_rad=0.0,
    )
    assert bool(seam_seed.geometry.valid)
    proof_column = np.array(
        [
            regular_mapped.geometry.column_px,
            two_branch_seed.geometry.column_px,
            mapped.geometry.column_px,
            -1.0,
            seam_seed.geometry.column_px,
        ]
    )
    proof_row = np.array(
        [
            regular_mapped.geometry.row_px,
            two_branch_seed.geometry.row_px,
            mapped.geometry.row_px,
            -1.0,
            seam_seed.geometry.row_px,
        ]
    )
    m1_rods = tuple(candidate for candidate in rods if candidate.family_m == 1)
    numpy_proof = detector.evaluate_detector_coordinates(
        proof_column,
        proof_row,
        rods=m1_rods,
    )
    compiled_density, compiled_count, compiled_caustic, _ = detector._compiled_evaluator(
        m1_rods
    ).evaluate(
        proof_column,
        proof_row,
        branch=2,
    )
    np.testing.assert_allclose(
        compiled_density,
        numpy_proof.per_rod_density_A2_per_px2,
        rtol=3.0e-12,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(compiled_count, numpy_proof.per_rod_inverse_branch_count)
    np.testing.assert_array_equal(compiled_caustic, numpy_proof.caustic)
    seam_point_index = proof_column.size - 1
    seam_rod_index = m1_rods.index(seam_rod)
    seam_density = numpy_proof.per_rod_density_A2_per_px2[seam_point_index, seam_rod_index]
    assert numpy_proof.per_rod_inverse_branch_count[seam_point_index, seam_rod_index] == 2
    assert not numpy_proof.caustic[seam_point_index, seam_rod_index]
    assert math.isfinite(float(seam_density))
    assert seam_density > 0.0

    # The production pixel kernel must integrate the same arbitrary continuous
    # detector-coordinate density as the independent NumPy point evaluator.
    detector_columns = detector.instrument.detector_shape_rc[1]
    fused_pixel_row = np.asarray(
        [
            round(float(regular_mapped.geometry.row_px)),
            round(float(two_branch_seed.geometry.row_px)),
        ],
        dtype=np.int64,
    )
    fused_pixel_column = np.asarray(
        [
            round(float(regular_mapped.geometry.column_px)),
            round(float(two_branch_seed.geometry.column_px)),
        ],
        dtype=np.int64,
    )
    fused_flat_index = fused_pixel_row * detector_columns + fused_pixel_column
    fused_offset = np.asarray((-0.25, 0.25), dtype=np.float64)
    fused_weight = np.asarray((0.5, 0.5), dtype=np.float64)
    fused = detector._compiled_evaluator(m1_rods).integrate_pixel_boxes(
        fused_flat_index,
        offset_px=fused_offset,
        one_dimensional_weight=fused_weight,
        branch=2,
        include_center_diagnostics=False,
    )
    oracle_column, oracle_row = np.broadcast_arrays(
        fused_pixel_column[:, None, None] + fused_offset[None, None, :],
        fused_pixel_row[:, None, None] + fused_offset[None, :, None],
    )
    oracle_density = detector.evaluate_detector_coordinates(
        oracle_column,
        oracle_row,
        rods=m1_rods,
        branch=2,
    ).per_rod_density_A2_per_px2
    oracle_mass = np.sum(
        oracle_density * (fused_weight[:, None] * fused_weight[None, :])[None, :, :, None],
        axis=(1, 2),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        fused.per_rod_mass_A2,
        oracle_mass,
        rtol=3.0e-12,
        atol=2.0e-24,
    )
    assert fused.center_per_rod_density_A2_per_px2.size == 0
    assert fused.per_rod_inverse_count_min.size == 0
    assert fused.per_rod_inverse_count_max.size == 0
    assert fused.per_rod_caustic.size == 0
    assert fused.valid_any.size == 0
    assert fused.valid_all.size == 0
    assert fused.center_valid.size == 0

    specular = detector.map_specular_geometry(
        rod=Rod(0, 0),
        alpha_rad=0.0,
        beta_rad=0.0,
    )
    assert specular.intensity_status is IntensityStatus.SPECULAR_INTENSITY_EXCLUDED
    assert bool(specular.geometry.valid)
    assert not hasattr(specular, "postoptical_density_A2_rad2_inv")

    tiny_instrument = replace(
        instrument,
        detector_shape_rc=(3, 3),
        detector_reference_coordinate_px=(
            instrument.detector_reference_coordinate_px[0]
            + 1.0
            - regular_mapped.geometry.column_px,
            instrument.detector_reference_coordinate_px[1] + 1.0 - regular_mapped.geometry.row_px,
        ),
    )
    tiny_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=tiny_instrument,
    )
    quadrature = DetectorQuadrature(pixel_gauss_order=2, row_chunk_size=2)
    pixels = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=quadrature,
    )
    assert pixels.image_A2.shape == (3, 3)
    assert pixels.image_A2.flags.writeable is False
    assert pixels.rods == m1_rods
    assert pixels.branch == 2
    assert pixels.total_detector_mass_A2 == pytest.approx(
        np.sum(pixels.per_rod_detector_mass_A2),
        rel=0.0,
        abs=2.0e-20,
    )
    assert np.sum(pixels.image_A2, dtype=np.float64) == pytest.approx(
        pixels.total_detector_mass_A2,
        rel=0.0,
        abs=2.0e-20,
    )
    assert pixels.total_detector_mass_A2 > 0.0
    assert pixels.sampled_valid_pixel_center is None
    assert not hasattr(pixels, "alpha_rad")
    assert not hasattr(pixels, "beta_rad")

    accelerated_quadrature = DetectorQuadrature(
        method=PixelIntegrationMethod.ADAPTIVE_COMPILED,
        pixel_gauss_order=2,
        relative_tolerance=1.0e-6,
        max_depth=3,
        row_chunk_size=2,
        worker_count=1,
    )
    accelerated = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=accelerated_quadrature,
    )
    parallel = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=replace(accelerated_quadrature, worker_count=2),
    )
    np.testing.assert_allclose(
        accelerated.image_A2,
        pixels.image_A2,
        rtol=2.0e-5,
        atol=2.0e-20,
    )
    np.testing.assert_array_equal(parallel.image_A2, accelerated.image_A2)
    np.testing.assert_array_equal(
        parallel.per_rod_detector_mass_A2,
        accelerated.per_rod_detector_mass_A2,
    )
    center_column, center_row = np.meshgrid(
        np.arange(3, dtype=np.float64),
        np.arange(3, dtype=np.float64),
    )
    center_geometry = tiny_detector.evaluate_detector_geometry(
        center_column,
        center_row,
        include_surface_jacobian=False,
    )
    np.testing.assert_array_equal(center_geometry.q_surface_jacobian_Ainv2_per_px2, 0.0)
    expected_center_valid = center_geometry.valid
    assert accelerated.sampled_valid_pixel_center is not None
    assert accelerated.sampled_valid_pixel_center.flags.writeable is False
    np.testing.assert_array_equal(
        accelerated.sampled_valid_pixel_center,
        expected_center_valid,
    )
    np.testing.assert_array_equal(
        parallel.sampled_valid_pixel_center,
        accelerated.sampled_valid_pixel_center,
    )
    assert parallel.adaptive_refined_pixel_count == accelerated.adaptive_refined_pixel_count
    assert parallel.adaptive_unresolved_pixel_count == accelerated.adaptive_unresolved_pixel_count
    assert parallel.sampled_invalid_pixel_count == accelerated.sampled_invalid_pixel_count
    assert parallel.coordinate_evaluation_count == accelerated.coordinate_evaluation_count
    assert parallel.estimated_l1_error_A2 == accelerated.estimated_l1_error_A2
    assert accelerated.coordinate_evaluation_count > 0
    assert accelerated.execution_backend == "numba_nogil_thread_tiles.v1"

    from rasim_next.pipeline._continuous_detector_kernel import CompiledDetectorEvaluator

    def reject_node_field_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("adaptive pixel integration materialized a node-scale field")

    monkeypatch.setattr(CompiledDetectorEvaluator, "evaluate", reject_node_field_materialization)
    fused_only = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=accelerated_quadrature,
    )
    np.testing.assert_array_equal(fused_only.image_A2, accelerated.image_A2)
    np.testing.assert_array_equal(
        fused_only.per_rod_detector_mass_A2,
        accelerated.per_rod_detector_mass_A2,
    )
    np.testing.assert_array_equal(
        fused_only.sampled_valid_pixel_center,
        accelerated.sampled_valid_pixel_center,
    )

    caustic_instrument = replace(
        instrument,
        detector_shape_rc=(3, 3),
        detector_reference_coordinate_px=(
            instrument.detector_reference_coordinate_px[0] + 1.0 - mapped.geometry.column_px,
            instrument.detector_reference_coordinate_px[1] + 1.0 - mapped.geometry.row_px,
        ),
    )
    caustic_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=caustic_instrument,
    )
    caustic_pixels = caustic_detector.integrate_native_pixels(
        rods=(rod,),
        branch=2,
        quadrature=DetectorQuadrature(
            pixel_gauss_order=2,
            fold_gauss_order=4,
            fold_subdivision_count=8,
            row_chunk_size=2,
        ),
    )
    assert caustic_pixels.fold_refined_pixel_count > 0
    assert np.all(np.isfinite(caustic_pixels.image_A2))
    assert caustic_pixels.fold_refinement_l1_A2 / caustic_pixels.total_detector_mass_A2 < 2.0e-2
    assert caustic_pixels.fold_refinement_centroid_shift_px < 5.0e-2
    compiled_caustic_pixels = caustic_detector.integrate_native_pixels(
        rods=(rod,),
        branch=2,
        quadrature=replace(
            accelerated_quadrature,
            relative_tolerance=1.0e-3,
            worker_count=2,
        ),
    )
    assert np.all(np.isfinite(compiled_caustic_pixels.image_A2))
    assert compiled_caustic_pixels.adaptive_refined_pixel_count > 0
    assert compiled_caustic_pixels.adaptive_unresolved_pixel_count > 0
    assert not compiled_caustic_pixels.adaptive_tolerance_satisfied
    zero_caustic_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=caustic_instrument,
        polarization_weight=0.0,
    )
    zero_pixels = zero_caustic_detector.integrate_native_pixels(
        rods=(rod,),
        branch=2,
        quadrature=DetectorQuadrature(
            pixel_gauss_order=2,
            fold_gauss_order=4,
            fold_subdivision_count=8,
            row_chunk_size=2,
        ),
    )
    assert zero_pixels.total_detector_mass_A2 == 0.0
    np.testing.assert_array_equal(zero_pixels.per_rod_detector_mass_A2, 0.0)
    assert not np.any(zero_pixels.image_A2)
    assert zero_pixels.fold_refinement_l1_A2 == 0.0
    assert zero_pixels.fold_refinement_centroid_shift_px == 0.0


def test_detector_event_envelope_uses_sample_q_after_mosaic_rotation() -> None:
    from painted_ewald import (
        BraggSpaceConfig,
        ContinuousEwaldCoating,
        MosaicBraggSpace,
        MosaicParameters,
        Rod,
    )
    from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
    from rasim_next.pipeline.continuous_detector import (
        DetectorEwaldMeasure,
        SampleQIntensityEnvelope,
    )

    inputs = _configured_inputs(sample_count=1)
    rod = Rod(-1, 1)
    rods = (rod,)
    strength = Bi2X3FiniteStackStrength(
        crystal=inputs.crystal,
        layers=7,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    bragg = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
            crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
            rods=rods,
            mosaic=MosaicParameters(
                gaussian_sigma_rad=math.radians(5.0),
                lorentzian_half_width_rad=math.radians(2.0),
                lorentzian_probability=0.1,
            ),
            k_norm_Ainv=2.0 * np.pi / inputs.samples.wavelength_A[0],
        ),
        strength,
    )
    coating = ContinuousEwaldCoating(
        bragg,
        ki_sample_Ainv=inputs.incident.states.k_film_phase_sample_Ainv[0],
    )
    baseline = DetectorEwaldMeasure(
        coating=coating,
        incident=inputs.incident,
        material=inputs.material,
        instrument=inputs.instrument,
    )
    envelope = SampleQIntensityEnvelope(
        u_radial_A2=0.017,
        u_normal_A2=0.031,
    )
    damped = DetectorEwaldMeasure(
        coating=coating,
        incident=inputs.incident,
        material=inputs.material,
        instrument=inputs.instrument,
        intensity_envelope=envelope,
    )
    mapped = baseline.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=math.radians(10.0),
        beta_rad=2.0,
    )
    damped_mapped = damped.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=math.radians(10.0),
        beta_rad=2.0,
    )
    latent_envelope = envelope.evaluate(mapped.geometry.ewald_geometry.q_sample_Ainv)
    np.testing.assert_allclose(
        damped_mapped.event_intensity_envelope,
        latent_envelope,
    )
    np.testing.assert_allclose(
        damped_mapped.postoptical_density_A2_rad2_inv,
        mapped.postoptical_density_A2_rad2_inv * latent_envelope,
    )
    column_px = np.asarray([float(mapped.geometry.column_px)])
    row_px = np.asarray([float(mapped.geometry.row_px)])
    baseline_result = baseline.evaluate_detector_coordinates(
        column_px,
        row_px,
        rods=rods,
        branch=2,
    )
    damped_result = damped.evaluate_detector_coordinates(
        column_px,
        row_px,
        rods=rods,
        branch=2,
    )
    assert not bool(baseline_result.caustic[0, 0])
    assert baseline_result.per_rod_density_A2_per_px2[0, 0] > 0.0
    q_sample = baseline_result.geometry.q_sample_Ainv[0]
    expected = math.exp(
        -envelope.u_radial_A2 * float(q_sample[0] ** 2 + q_sample[1] ** 2)
        - envelope.u_normal_A2 * float(q_sample[2] ** 2)
    )
    actual = (
        damped_result.per_rod_density_A2_per_px2[0, 0]
        / baseline_result.per_rod_density_A2_per_px2[0, 0]
    )
    assert actual == pytest.approx(expected, rel=3.0e-13, abs=0.0)
    baseline_compiled, _, _, _ = baseline._compiled_evaluator(rods).evaluate(
        column_px,
        row_px,
        branch=2,
    )
    damped_compiled, _, _, _ = damped._compiled_evaluator(rods).evaluate(
        column_px,
        row_px,
        branch=2,
    )
    np.testing.assert_allclose(baseline_compiled, baseline_result.per_rod_density_A2_per_px2)
    np.testing.assert_allclose(damped_compiled, damped_result.per_rod_density_A2_per_px2)


def _two_state_source_averaged_detector_fixture(
    *,
    detector_shape_rc: tuple[int, int] | None = None,
    detector_path_linear_attenuation_m_inv_by_wavelength: tuple[float, float] | None = None,
) -> tuple[object, tuple[object, ...]]:
    from painted_ewald import (
        BraggSpaceConfig,
        ContinuousEwaldCoating,
        MosaicBraggSpace,
        MosaicParameters,
        Rod,
    )
    from rasim_next.core.contracts import IncidentSampleBatch
    from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
    from rasim_next.pipeline.continuous_detector import (
        DetectorEwaldMeasure,
    )
    from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure

    inputs = _configured_inputs(sample_count=2)
    samples = inputs.samples
    instrument = inputs.instrument
    crystal = inputs.crystal
    material = inputs.material
    incident = inputs.incident
    reciprocal = inputs.reciprocal
    rods = tuple(Rod(h, k) for h, k in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)))
    mosaic = MosaicParameters(
        gaussian_sigma_rad=math.radians(5.0),
        lorentzian_half_width_rad=math.radians(2.0),
        lorentzian_probability=0.1,
    )
    strength = Bi2X3FiniteStackStrength(
        crystal=crystal,
        layers=7,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    if detector_path_linear_attenuation_m_inv_by_wavelength is not None:
        instrument = replace(
            instrument,
            detector_path_medium_id="standard_dry_air_wavelength_table_sensitivity.v1",
            detector_path_wavelength_A=tuple(float(value) for value in samples.wavelength_A),
            detector_path_linear_attenuation_m_inv_by_wavelength=(
                detector_path_linear_attenuation_m_inv_by_wavelength
            ),
        )
    singleton_incidents = []
    coatings = []
    for state_index in range(2):
        singleton = IncidentSampleBatch(
            incident_sample_id=np.asarray([state_index], dtype=np.int64),
            origin_lab_m=samples.origin_lab_m[state_index : state_index + 1],
            direction_lab=samples.direction_lab[state_index : state_index + 1],
            wavelength_A=samples.wavelength_A[state_index : state_index + 1],
            source_weight=np.asarray([1.0]),
            polarization_state_id=(samples.polarization_state_id[state_index],),
            source_sampling_model_id="explicit_external_source.v1",
            source_rng_model_id="no_rng.v1",
            source_seed=state_index,
            source_parameter_provenance=f"two-state scalar oracle row {state_index}",
        )
        singleton_incident = build_incident_states(singleton, material, instrument)
        singleton_incidents.append(singleton_incident)
        bragg = MosaicBraggSpace(
            BraggSpaceConfig(
                reciprocal_basis_Ainv=reciprocal.basis_Ainv,
                crystal_to_sample=instrument.sample_from_crystal.rotation,
                rods=rods,
                mosaic=mosaic,
                k_norm_Ainv=2.0 * np.pi / singleton.wavelength_A[0],
            ),
            strength,
        )
        coatings.append(
            ContinuousEwaldCoating(
                bragg,
                ki_sample_Ainv=singleton_incident.states.k_film_phase_sample_Ainv[0],
            )
        )
    if detector_shape_rc is not None:
        provisional = DetectorEwaldMeasure(
            coating=coatings[0],
            incident=singleton_incidents[0],
            material=material,
            instrument=instrument,
        )
        seed = provisional.map_latent(
            rod=rods[1],
            branch=2,
            alpha_rad=math.radians(2.0),
            beta_rad=math.radians(178.0),
        )
        rows, columns = detector_shape_rc
        instrument = replace(
            instrument,
            detector_shape_rc=detector_shape_rc,
            detector_reference_coordinate_px=(
                instrument.detector_reference_coordinate_px[0]
                + 0.5 * (columns - 1)
                - float(seed.geometry.column_px),
                instrument.detector_reference_coordinate_px[1]
                + 0.5 * (rows - 1)
                - float(seed.geometry.row_px),
            ),
        )
    averaged = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=reciprocal.basis_Ainv,
        crystal_to_sample=instrument.sample_from_crystal.rotation,
        rods=rods,
        rod_catalog_revision="integration-test-rods.v1",
        mosaic=mosaic,
        strength_model=strength,
        incident=incident,
        material=material,
        instrument=instrument,
        worker_count=2,
    )
    scalar_detectors = tuple(
        DetectorEwaldMeasure(
            coating=coating,
            incident=singleton_incident,
            material=material,
            instrument=instrument,
        )
        for coating, singleton_incident in zip(coatings, singleton_incidents, strict=True)
    )
    return averaged, scalar_detectors


def test_source_averaged_sparse_structure_response_matches_compiled_bi2x3() -> None:
    averaged, scalar_detectors = _two_state_source_averaged_detector_fixture(
        detector_path_linear_attenuation_m_inv_by_wavelength=(
            1.194448861306956,
            1.2033257750023947,
        )
    )
    rods = averaged.rods
    mapped = tuple(
        detector.map_latent(
            rod=rods[1],
            branch=2,
            alpha_rad=math.radians(2.0),
            beta_rad=math.radians(178.0),
        )
        for detector in scalar_detectors
    )
    column_px = np.asarray([item.geometry.column_px for item in mapped])
    row_px = np.asarray([item.geometry.row_px for item in mapped])

    response = compile_source_averaged_detector_structure_response(
        column_px,
        row_px,
        reciprocal_basis_Ainv=averaged.strength_model.reciprocal_basis_Ainv,
        crystal_to_sample=averaged.instrument.sample_from_crystal.rotation,
        rods=rods,
        rod_catalog_revision=averaged.rod_catalog_revision,
        mosaic=averaged.mosaic,
        reference_strength_model=averaged.strength_model,
        intensity_envelope=averaged.intensity_envelope,
        incident=averaged.incident,
        material=averaged.material,
        instrument=averaged.instrument,
    )
    air_instrument = replace(
        averaged.instrument,
        detector_path_linear_attenuation_m_inv_by_wavelength=(1.3, 1.4),
    )
    air_response = compile_source_averaged_detector_structure_response(
        column_px,
        row_px,
        reciprocal_basis_Ainv=averaged.strength_model.reciprocal_basis_Ainv,
        crystal_to_sample=air_instrument.sample_from_crystal.rotation,
        rods=rods,
        rod_catalog_revision=averaged.rod_catalog_revision,
        mosaic=averaged.mosaic,
        reference_strength_model=averaged.strength_model,
        intensity_envelope=averaged.intensity_envelope,
        incident=averaged.incident,
        material=averaged.material,
        instrument=air_instrument,
    )
    assert air_response.fixed_physics_revision != response.fixed_physics_revision
    assert not np.array_equal(
        air_response.term_fixed_density_per_strength_px2_inv,
        response.term_fixed_density_per_strength_px2_inv,
    )
    actual = response.apply_strength(averaged.strength_model)
    expected = averaged.evaluate_detector_coordinates_all_roots(column_px, row_px)

    class _ComplexStrength:
        reciprocal_basis_Ainv = averaged.strength_model.reciprocal_basis_Ainv
        structure_model_revision = "f" * 64

        @staticmethod
        def evaluate_hkl(**kwargs: object) -> np.ndarray:
            return np.full(np.asarray(kwargs["L"]).shape, 1.0 + 1.0j)

    with pytest.raises(ValueError, match="evaluate_hkl result must be real"):
        response.apply_strength(_ComplexStrength())

    assert response.source_state_count == 2
    assert np.unique(response.term_k_norm_Ainv).size == 2
    assert len(response.response_revision) == 64
    assert actual.execution_backend == "numpy_cpu_sparse_source_averaged.v1"
    np.testing.assert_allclose(
        actual.per_rod_density_A2_per_px2,
        expected.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        actual.density_A2_per_px2,
        expected.density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(actual.caustic, expected.caustic)

    from painted_ewald.normal_density import SphericalMosaicDensity

    law = SphericalMosaicDensity(averaged.mosaic)
    term_hk = np.asarray([(rod.h, rod.k) for rod in response.rods])[response.term_rod_index]
    term_strength = averaged.strength_model.evaluate_hkl(
        h=term_hk[:, 0],
        k=term_hk[:, 1],
        L=response.term_L,
        k_norm_Ainv=response.term_k_norm_Ainv,
    )
    term_density = (
        response.term_fixed_density_per_mosaic_density_per_strength_px2_inv
        * term_strength
        * law.latent_density_rad_inv2(response.term_alpha_rad)
    )
    rod_count = len(response.rods)
    spherical_sparse = np.bincount(
        response.term_coordinate_index * rod_count + response.term_rod_index,
        weights=term_density,
        minlength=column_px.size * rod_count,
    ).reshape(column_px.size, rod_count)
    spherical_compiled = np.zeros_like(spherical_sparse)
    for block in averaged._evaluator_blocks:
        for indexed in block:
            per_source = indexed.evaluator.evaluate_all_roots(
                column_px,
                row_px,
                mosaic_density=law,
            )[0]
            spherical_compiled[:, indexed.master_rod_index] += per_source
    np.testing.assert_allclose(spherical_sparse, spherical_compiled, rtol=4e-11, atol=3e-24)

    class _CountingStrength:
        reciprocal_basis_Ainv = averaged.strength_model.reciprocal_basis_Ainv
        structure_model_revision = averaged.strength_model.structure_model_revision

        def __init__(self) -> None:
            self.call_count = 0

        def evaluate_hkl(self, **kwargs: object) -> np.ndarray:
            self.call_count += 1
            return averaged.strength_model.evaluate_hkl(
                h=kwargs["h"],
                k=kwargs["k"],
                L=kwargs["L"],
                k_norm_Ainv=kwargs["k_norm_Ainv"],
            )

    wider_mosaic = replace(
        averaged.mosaic,
        gaussian_sigma_rad=1.3 * averaged.mosaic.gaussian_sigma_rad,
        lorentzian_half_width_rad=0.7 * averaged.mosaic.lorentzian_half_width_rad,
    )
    counting_strength = _CountingStrength()
    rebound = response.apply_strength_for_mosaics(
        counting_strength,
        (averaged.mosaic, wider_mosaic),
    )
    assert counting_strength.call_count == 1
    rebound_expected = averaged.rebind_physics(
        mosaic=wider_mosaic
    ).evaluate_detector_coordinates_all_roots(column_px, row_px)
    np.testing.assert_allclose(
        rebound[0].per_rod_density_A2_per_px2,
        actual.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        rebound[1].per_rod_density_A2_per_px2,
        rebound_expected.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )

    envelope = SampleQIntensityEnvelope(u_radial_A2=0.017, u_normal_A2=0.006)
    separable_response = compile_source_averaged_detector_structure_response(
        column_px,
        row_px,
        reciprocal_basis_Ainv=averaged.strength_model.reciprocal_basis_Ainv,
        crystal_to_sample=averaged.instrument.sample_from_crystal.rotation,
        rods=rods,
        rod_catalog_revision=averaged.rod_catalog_revision,
        mosaic=averaged.mosaic,
        reference_strength_model=averaged.strength_model,
        intensity_envelope=SampleQIntensityEnvelope(),
        incident=averaged.incident,
        material=averaged.material,
        instrument=averaged.instrument,
    )
    replayed_envelope = separable_response.apply_strength(
        averaged.strength_model,
        intensity_envelope=envelope,
    )
    exact_envelope = averaged.rebind_physics(
        intensity_envelope=envelope
    ).evaluate_detector_coordinates_all_roots(column_px, row_px)
    np.testing.assert_allclose(
        replayed_envelope.per_rod_density_A2_per_px2,
        exact_envelope.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(actual.valid_source_count, expected.valid_source_count)
    assert actual.source_revision == expected.source_revision


@pytest.mark.parametrize(
    "config_name",
    ("bi2se3_simulation.yaml", "bi2te3_simulation.yaml"),
)
def test_configured_bi2x3_provider_uses_shared_sparse_fitting_detector(
    config_name: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_simulation_config(root / "configs" / config_name)
    config = replace(
        base,
        source=replace(
            _single_mean_test_source(base.source),
            spatial_sigma_m=(0.0, 0.0),
            divergence_sigma_rad=(0.0, 0.0),
        ),
    )
    inputs = build_configured_simulation_inputs(config)
    shared = build_source_averaged_structure_detector(inputs)
    optimized = build_source_averaged_detector(inputs)
    nominal = build_nominal_ewald_context(inputs)

    class _UnrevisionedStrength:
        reciprocal_basis_Ainv = inputs.strength.reciprocal_basis_Ainv

    with pytest.raises(ValueError, match="structure-model revision"):
        build_source_averaged_structure_detector(
            inputs,
            strength_model=cast(RevisionedStructureStrengthModel, _UnrevisionedStrength()),
        )

    mapped = None
    for rod in nominal.rods:
        if rod.h == 0 and rod.k == 0:
            continue
        for branch in (1, 2):
            candidate = nominal.map_latent_geometry(
                rod=rod,
                branch=branch,
                alpha_rad=math.radians(2.0),
                beta_rad=math.radians(178.0),
            )
            if bool(candidate.valid):
                mapped = candidate
                break
        if mapped is not None:
            break
    assert mapped is not None
    column_px = np.asarray((mapped.column_px,))
    row_px = np.asarray((mapped.row_px,))

    actual = shared.evaluate_detector_coordinates_all_roots(column_px, row_px)
    expected = optimized.evaluate_detector_coordinates_all_roots(column_px, row_px)

    assert shared.strength_model is inputs.strength
    np.testing.assert_allclose(
        actual.per_rod_density_A2_per_px2,
        expected.per_rod_density_A2_per_px2,
        rtol=5.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        actual.density_A2_per_px2,
        expected.density_A2_per_px2,
        rtol=5.0e-11,
        atol=3.0e-24,
    )


@pytest.mark.parametrize(
    ("relative_cif", "phase_id"),
    (
        ("examples/bi2se3/structures/Bi2Se3_vesta.cif", "Bi2Se3"),
        ("examples/bi2te3/structures/Bi2Te3_cod_9011962.cif", "Bi2Te3"),
        ("examples/pbi2/structures/PbI2_2H.cif", "PbI2"),
    ),
)
def test_configured_generic_cif_detector_matches_direct_all_root_sum(
    relative_cif: str,
    phase_id: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    config = replace(
        base,
        material=replace(
            base.material,
            cif_path=root / relative_cif,
            phase_id=phase_id,
        ),
        source=_single_mean_test_source(base.source),
        structure_factor=replace(
            base.structure_factor,
            model_id="cif_conventional_cell_finite_repeat.v1",
            layers=None,
            repeats=5,
            normalization="FINITE_TOTAL",
            shared_disorder_epsilon=0.0,
            unknown_u_iso_A2=0.0,
        ),
        bragg=replace(base.bragg, include_detector_visible_m0=False),
    )
    inputs = build_configured_simulation_inputs(config)
    detector = build_source_averaged_structure_detector(inputs)
    nominal = build_nominal_ewald_context(inputs)
    scalar = _physical_scalar_detector(inputs)

    mapped = None
    for rod in nominal.rods:
        if (rod.h, rod.k) == (0, 0):
            continue
        for branch in (1, 2):
            candidate = nominal.map_latent_geometry(
                rod=rod,
                branch=branch,
                alpha_rad=math.radians(2.0),
                beta_rad=math.radians(178.0),
            )
            if bool(candidate.valid):
                mapped = candidate
                break
        if mapped is not None:
            break
    assert mapped is not None
    column_px = np.asarray((float(mapped.column_px),), dtype=np.float64)
    row_px = np.asarray((float(mapped.row_px),), dtype=np.float64)

    actual = detector.evaluate_detector_coordinates_all_roots(column_px, row_px)
    lower = scalar.evaluate_detector_coordinates(
        column_px,
        row_px,
        rods=detector.rods,
        branch=1,
    )
    upper = scalar.evaluate_detector_coordinates(
        column_px,
        row_px,
        rods=detector.rods,
        branch=2,
    )
    expected = lower.per_rod_density_A2_per_px2 + upper.per_rod_density_A2_per_px2

    assert inputs.strength.normalization == "FINITE_TOTAL"
    np.testing.assert_allclose(
        actual.per_rod_density_A2_per_px2,
        expected,
        rtol=3.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(actual.caustic, lower.caustic | upper.caustic)
    np.testing.assert_array_equal(actual.valid_source_count, np.ones(1, dtype=np.int64))


def test_pbi2_parent_mixture_uses_shared_sparse_detector_including_regular_00l() -> None:
    source_cif_sha256 = "7cf2a5e1957ea63d277c704cff390724175f96e6d26f982287490eedc24afbf9"
    base_config = _generic_pbi2_config(include_detector_visible_m0=True)
    config = replace(
        base_config,
        structure_factor=replace(
            base_config.structure_factor,
            normalization="FINITE_PER_LAYER",
        ),
    )
    inputs = build_configured_simulation_inputs(config)
    reference = Pbi2ParentMixtureStrength(
        crystal=inputs.crystal,
        source_cif_sha256=source_cif_sha256,
        layers=5,
        domain_fraction=(0.36, 0.19, 0.13, 0.17, 0.15),
    )
    detector = build_source_averaged_structure_detector(
        inputs,
        strength_model=reference,
    )
    nominal = build_nominal_ewald_context(inputs)

    m0_landmark = ExactTagGeometryModel(
        build_configured_geometry_inputs(config)
    ).predict_m0_minimum_tilt_exact_l_landmarks((1,))
    assert m0_landmark.detector_status.tolist() == ["VALID"]
    m0_response = detector.compile_structure_response(
        m0_landmark.coordinates_px[:, 0],
        m0_landmark.coordinates_px[:, 1],
    )
    m0_pure = tuple(
        m0_response.apply_strength(
            reference.rebind_domain_fraction(
                tuple(float(index == component_index) for index in range(5))
            )
        )
        for component_index in range(5)
    )
    m0_rod_index = m0_response.rods.index(Rod(0, 0))
    m0_parent_density = np.asarray(
        [item.per_rod_density_A2_per_px2[0, m0_rod_index] for item in m0_pure]
    )
    assert np.all(m0_parent_density > 0.0)
    np.testing.assert_allclose(
        np.ptp(m0_parent_density),
        0.0,
        rtol=0.0,
        atol=2.0e-13 * np.max(m0_parent_density),
    )

    mapped = []
    for rod in nominal.rods:
        if (rod.h, rod.k) == (0, 0):
            continue
        for branch in (1, 2):
            candidate = nominal.map_latent_geometry(
                rod=rod,
                branch=branch,
                alpha_rad=math.radians(2.0),
                beta_rad=math.radians(165.0 + 3.0 * len(mapped)),
            )
            if bool(candidate.valid):
                mapped.append(candidate)
                break
        if len(mapped) == 4:
            break
    assert len(mapped) == 4
    column_px = np.asarray([value.column_px for value in mapped])
    row_px = np.asarray([value.row_px for value in mapped])
    response = detector.compile_structure_response(column_px, row_px)
    actual = response.apply_strength(reference)
    pure = tuple(
        response.apply_strength(
            reference.rebind_domain_fraction(
                tuple(float(index == component_index) for index in range(5))
            )
        )
        for component_index in range(5)
    )
    expected_per_rod = np.sum(
        np.stack(tuple(item.per_rod_density_A2_per_px2 for item in pure))
        * reference.domain_fraction[:, None, None],
        axis=0,
    )

    np.testing.assert_allclose(
        actual.per_rod_density_A2_per_px2,
        expected_per_rod,
        rtol=3.0e-13,
        atol=3.0e-24,
    )
    fit_reference = reference.rebind_domain_fraction((0.5, 0.5, 0.0, 0.0, 0.0))
    fit_response = build_source_averaged_structure_detector(
        inputs,
        strength_model=fit_reference,
    ).compile_structure_response(column_px, row_px)
    parameterization = Pbi2ParentLogRatioParameterization(
        reference_strength=fit_reference,
        active_component_ids=("2H", "4H+"),
    )
    observation_count = column_px.size + 2
    model = ParameterizedStructureRegionModel(
        parameterization=parameterization,
        blocks=(
            StructureRegionResponseBlock(
                dataset_id="pbi2-parent-mixture",
                response=fit_response,
                quadrature=ContinuousRegionQuadrature(
                    column_px=column_px,
                    row_px=row_px,
                    detector_area_weight_px2=np.ones(column_px.size),
                    observation_row=np.arange(column_px.size),
                    background_coordinate=np.zeros(column_px.size),
                    observation_count=observation_count,
                    chart_revision="pbi2-parent-mixture-points.v1",
                ),
            ),
        ),
    )
    truth = np.asarray((math.log(0.7 / 0.3),))
    truth_mass = model.predict_mass_A2(truth)
    reference_peak = float(np.max(model.predict_mass_A2(model.reference_parameters)))
    dataset_scale = 2.0 / reference_peak
    fixed_background_mass = np.full(observation_count, 12.0)
    observations = MatchedRegionObservations(
        dataset_ids=("pbi2-parent-mixture",),
        dataset_index=np.zeros(observation_count, dtype=np.int64),
        block_index=np.zeros(observation_count, dtype=np.int64),
        signal_family=np.asarray((*range(column_px.size), -1, -1)),
        is_background=np.asarray((*([False] * column_px.size), True, True)),
        count_mass=fixed_background_mass + dataset_scale * truth_mass,
        support_px2=np.ones(observation_count),
        background_coordinate=np.asarray((*([0.0] * column_px.size), -1.0, 1.0)),
        required_signal_families=tuple(range(column_px.size)),
        count_covariance_count2=np.eye(observation_count),
    )
    fit = fit_parameterized_matched_regions(
        observations,
        model,
        fixed_background=FixedMatchedRegionBackground(
            count_mass=fixed_background_mass,
            covariance_count2=np.zeros((observation_count, observation_count)),
            revision="pbi2-parent-mixture-background.v1",
        ),
        initial_parameters=(np.asarray((-1.0,)), np.asarray((1.0,))),
        lower_bounds=np.asarray((-4.0,)),
        upper_bounds=np.asarray((4.0,)),
        parameter_scales=np.asarray((1.0,)),
    )
    np.testing.assert_allclose(fit.fit.parameters, truth, rtol=0.0, atol=2.0e-8)
    assert fit.fit.sensitivity_rank == 1
    assert fit.fit.sensitivity_condition == pytest.approx(1.0)


def test_generic_pbi2_detector_uses_unchanged_mosaic_fitter() -> None:
    from rasim_next.fitting.mosaic import (
        MosaicProfileDefinition,
        MosaicProfileIdentity,
        MosaicReflectionGroupKey,
        compile_continuous_mosaic_component_profile_bank,
        evaluate_continuous_mosaic_profiles,
        fit_refined_mosaic_component_profiles,
    )
    from rasim_next.selection import build_osc_angle_frame

    config = _generic_pbi2_config(include_detector_visible_m0=True)
    inputs = build_configured_simulation_inputs(config)
    detector = build_source_averaged_structure_detector(inputs)
    nominal = build_nominal_ewald_context(inputs)

    mapped = None
    mapped_rod = None
    mapped_branch = None
    for rod in nominal.rods:
        if (rod.h, rod.k) == (0, 0):
            continue
        for branch in (1, 2):
            candidate = nominal.map_latent_geometry(
                rod=rod,
                branch=branch,
                alpha_rad=math.radians(2.0),
                beta_rad=math.radians(178.0),
            )
            if bool(candidate.valid):
                mapped = candidate
                mapped_rod = rod
                mapped_branch = branch
                break
        if mapped is not None:
            break
    assert mapped is not None and mapped_rod is not None and mapped_branch is not None

    angle_frame = build_osc_angle_frame(
        mean_direction_lab=config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=nominal.incident.states.sample_intersection_lab_m[0],
        revision="generic-pbi2-mosaic-proof.v1",
    )
    angles = detector_coordinates_to_angles(
        np.asarray((mapped.column_px,)),
        np.asarray((mapped.row_px,)),
        instrument=inputs.instrument,
        angle_frame=angle_frame,
    )
    m0_landmark = ExactTagGeometryModel(
        build_configured_geometry_inputs(config)
    ).predict_m0_minimum_tilt_exact_l_landmarks((1,))
    assert m0_landmark.detector_status.tolist() == ["VALID"]
    m0_angles = detector_coordinates_to_angles(
        m0_landmark.coordinates_px[:, 0],
        m0_landmark.coordinates_px[:, 1],
        instrument=inputs.instrument,
        angle_frame=angle_frame,
    )
    assert np.all(m0_angles.valid & m0_angles.azimuth_valid)
    definitions = (
        MosaicProfileDefinition(
            identity=MosaicProfileIdentity(
                dataset_id="generic-pbi2-mosaic-proof",
                incidence_angle_rad=math.radians(config.instrument.axis_rotations[0].angle_deg),
                group_key=MosaicReflectionGroupKey(
                    group_id="generic-pbi2-mosaic-proof:one-rod",
                    rod_catalog_revision=detector.rod_catalog_revision,
                    member_rod_hk=((mapped_rod.h, mapped_rod.k),),
                    branch_mode="EXPLICIT_NONZERO",
                ),
                branch_id=mapped_branch,
                analytic_branch_id=mapped_branch,
            ),
            center_two_theta_rad=float(angles.two_theta_rad[0]),
            center_phi_rad=float(angles.phi_rad[0]),
            two_theta_half_width_rad=math.radians(0.05),
            phi_half_width_rad=math.radians(4.0),
            phi_bin_count=11,
            two_theta_gauss_order=2,
            phi_gauss_order=2,
        ),
        MosaicProfileDefinition(
            identity=MosaicProfileIdentity(
                dataset_id="generic-pbi2-mosaic-proof",
                incidence_angle_rad=math.radians(config.instrument.axis_rotations[0].angle_deg),
                group_key=MosaicReflectionGroupKey(
                    group_id="generic-pbi2-mosaic-proof:001",
                    rod_catalog_revision=detector.rod_catalog_revision,
                    member_rod_hk=((0, 0),),
                    branch_mode="COLLAPSED_00L",
                    layered_family_m=0,
                    layered_integer_L=1,
                ),
                branch_id=None,
                analytic_branch_id=0,
            ),
            center_two_theta_rad=float(m0_angles.two_theta_rad[0]),
            center_phi_rad=float(m0_angles.phi_rad[0]),
            two_theta_half_width_rad=math.radians(0.05),
            phi_half_width_rad=math.radians(4.0),
            phi_bin_count=11,
            two_theta_gauss_order=2,
            phi_gauss_order=2,
        ),
    )

    def evaluate_component(
        gaussian_sigma_rad: float,
        lorentzian_half_width_rad: float,
        lorentzian_probability: float,
    ) -> object:
        return evaluate_continuous_mosaic_profiles(
            replace(
                detector,
                mosaic=replace(
                    detector.mosaic,
                    gaussian_sigma_rad=gaussian_sigma_rad,
                    lorentzian_half_width_rad=lorentzian_half_width_rad,
                    lorentzian_probability=lorentzian_probability,
                ),
            ),
            angle_frame=angle_frame,
            definitions=definitions,
            profile_revision="generic-pbi2-mosaic-proof.v1",
        )

    truth_gaussian = math.radians(1.0)
    truth_lorentzian = math.radians(0.5)
    truth_eta = 0.25
    gaussian = evaluate_component(truth_gaussian, truth_lorentzian, 0.0)
    lorentzian = evaluate_component(truth_gaussian, truth_lorentzian, 1.0)
    assert np.all(np.sum(gaussian.signal, axis=1) > 0.0)
    observations = replace(
        gaussian,
        signal=(1.0 - truth_eta) * gaussian.signal + truth_eta * lorentzian.signal,
        source_revision=None,
        execution_backend=None,
        execution_device=None,
        observation_revision="generic-pbi2-synthetic-observation.v1",
    )
    gaussian_widths = np.radians((0.8, 1.0))
    lorentzian_widths = np.radians((0.4, 0.5))
    mixed_dataset_definition = replace(
        definitions[1],
        identity=replace(
            definitions[1].identity,
            dataset_id="generic-pbi2-mosaic-proof-other",
        ),
    )
    with pytest.raises(ValueError, match="only one dataset"):
        compile_continuous_mosaic_component_profile_bank(
            detector,
            angle_frame=angle_frame,
            definitions=(definitions[0], mixed_dataset_definition),
            observations=observations,
            gaussian_sigma_rad=gaussian_widths,
            lorentzian_half_width_rad=lorentzian_widths,
            profile_revision="generic-pbi2-mosaic-proof.v1",
        )
    stale_two_theta_bounds = np.array(observations.two_theta_bounds_rad, copy=True)
    stale_two_theta_bounds[0, 0] -= 1.0e-8
    with pytest.raises(ValueError, match="different layouts"):
        compile_continuous_mosaic_component_profile_bank(
            detector,
            angle_frame=angle_frame,
            definitions=definitions,
            observations=replace(
                observations,
                two_theta_bounds_rad=stale_two_theta_bounds,
            ),
            gaussian_sigma_rad=gaussian_widths,
            lorentzian_half_width_rad=lorentzian_widths,
            profile_revision="generic-pbi2-mosaic-proof.v1",
        )
    compiled_bank = compile_continuous_mosaic_component_profile_bank(
        detector,
        angle_frame=angle_frame,
        definitions=definitions,
        observations=observations,
        gaussian_sigma_rad=gaussian_widths,
        lorentzian_half_width_rad=lorentzian_widths,
        profile_revision="generic-pbi2-mosaic-proof.v1",
    )
    exact_gaussian = tuple(
        evaluate_component(float(width), truth_lorentzian, 0.0) for width in gaussian_widths
    )
    exact_lorentzian = tuple(
        evaluate_component(truth_gaussian, float(width), 1.0) for width in lorentzian_widths
    )
    for compiled, exact in zip(
        compiled_bank.gaussian_profiles,
        exact_gaussian,
        strict=True,
    ):
        np.testing.assert_allclose(compiled.profile.signal, exact.signal, rtol=4.0e-11)
        np.testing.assert_array_equal(compiled.profile.valid, exact.valid)
        np.testing.assert_array_equal(compiled.profile.normalization, exact.normalization)
    for compiled, exact in zip(
        compiled_bank.lorentzian_profiles,
        exact_lorentzian,
        strict=True,
    ):
        np.testing.assert_allclose(compiled.profile.signal, exact.signal, rtol=4.0e-11)
        np.testing.assert_array_equal(compiled.profile.valid, exact.valid)
        np.testing.assert_array_equal(compiled.profile.normalization, exact.normalization)
    result = fit_refined_mosaic_component_profiles(
        observations,
        evaluate_gaussian_profile=lambda width: evaluate_component(
            width,
            truth_lorentzian,
            0.0,
        ),
        evaluate_lorentzian_profile=lambda width: evaluate_component(
            truth_gaussian,
            width,
            1.0,
        ),
        gaussian_sigma_bounds_rad=np.asarray((math.radians(0.5), math.radians(2.0))),
        lorentzian_half_width_bounds_rad=np.asarray((math.radians(0.25), math.radians(1.0))),
        coarse_width_count=3,
        refinement_width_count=3,
        refinement_levels=0,
        near_optimal_objective_delta=0.0,
    )

    assert inputs.strength.normalization == "FINITE_TOTAL"
    assert gaussian.execution_backend == "numpy_cpu_sparse_source_averaged.v1"
    assert result.fit.gaussian_sigma_rad == pytest.approx(truth_gaussian, abs=1.0e-15)
    assert result.fit.lorentzian_half_width_rad == pytest.approx(
        truth_lorentzian,
        abs=1.0e-15,
    )
    assert result.fit.lorentzian_probability == pytest.approx(truth_eta, abs=1.0e-12)
    assert result.fit.objective < 1.0e-24
    assert result.fit.sensitivity_rank == 3
    assert result.fit.sensitivity_condition < 100.0


def test_generic_cif_structure_uses_unchanged_matched_region_fitter() -> None:
    config = _generic_pbi2_config(include_detector_visible_m0=True)
    inputs = build_configured_simulation_inputs(config)
    detector = build_source_averaged_structure_detector(inputs)
    nominal = build_nominal_ewald_context(inputs)

    site_count = len(inputs.crystal.sites)
    fractional = np.zeros((site_count, 3, 2), dtype=np.float64)
    occupancy = np.zeros((site_count, 2), dtype=np.float64)
    u_iso = np.zeros((site_count, 2), dtype=np.float64)
    fractional[1, 2, 0] = 1.0
    fractional[2, 2, 0] = -1.0
    occupancy[1:, 1] = 1.0
    basis = AffineCifSiteBasis(
        reference_crystal=inputs.crystal,
        parameter_names=("iodine_z_shift", "iodine_occupancy"),
        parameter_units=("fractional", "1"),
        reference_parameters=np.asarray((0.0, 1.0)),
        fractional_coefficients=fractional,
        occupancy_coefficients=occupancy,
        u_iso_A2_coefficients=u_iso,
        unknown_u_iso_A2=0.0,
    )

    requested = (
        ((-1, 1), 170.0),
        ((0, 1), 170.0),
        ((1, 0), 170.0),
        ((-1, 1), 175.0),
        ((0, 1), 175.0),
        ((-1, 1), 190.0),
    )
    mapped = []
    for rod_hk, beta_deg in requested:
        rod = next(rod for rod in nominal.rods if (rod.h, rod.k) == rod_hk)
        point = nominal.map_latent_geometry(
            rod=rod,
            branch=2,
            alpha_rad=math.radians(1.0),
            beta_rad=math.radians(beta_deg),
        )
        assert bool(point.valid)
        mapped.append(point)
    column_px = np.asarray([point.column_px for point in mapped])
    row_px = np.asarray([point.row_px for point in mapped])
    signal_count = len(mapped)
    rows_per_dataset = signal_count + 2
    observation_count = 2 * rows_per_dataset
    quadratures = tuple(
        ContinuousRegionQuadrature(
            column_px=column_px,
            row_px=row_px,
            detector_area_weight_px2=np.ones(signal_count),
            observation_row=dataset_index * rows_per_dataset + np.arange(signal_count),
            background_coordinate=np.zeros(signal_count),
            observation_count=observation_count,
            chart_revision=f"generic-cif-ordered-sf-points.dataset-{dataset_index}.v1",
        )
        for dataset_index in range(2)
    )
    response = detector.compile_structure_response(column_px, row_px)
    parameterization = AffineCifFiniteStackParameterization(inputs.strength, basis)
    m0_landmark = ExactTagGeometryModel(
        build_configured_geometry_inputs(config)
    ).predict_m0_minimum_tilt_exact_l_landmarks((1,))
    assert m0_landmark.detector_status.tolist() == ["VALID"]
    m0_response = detector.compile_structure_response(
        m0_landmark.coordinates_px[:, 0],
        m0_landmark.coordinates_px[:, 1],
    )
    m0_rod_index = m0_response.rods.index(Rod(0, 0))
    m0_density = np.asarray(
        [
            m0_response.apply_strength(
                parameterization.bind_strength((iodine_z_shift, 1.0))
            ).per_rod_density_A2_per_px2[0, m0_rod_index]
            for iodine_z_shift in (-0.01, 0.0, 0.01)
        ]
    )
    assert np.all(m0_density > 0.0)
    assert np.all(np.diff(m0_density) < 0.0)
    model = ParameterizedStructureRegionModel(
        parameterization=parameterization,
        blocks=tuple(
            StructureRegionResponseBlock(
                dataset_id=dataset_id,
                response=response,
                quadrature=quadrature,
            )
            for dataset_id, quadrature in zip(
                ("first", "second"),
                quadratures,
                strict=True,
            )
        ),
    )

    truth = np.asarray((0.008, 0.86))
    reference_peak = float(np.max(model.predict_mass_A2(parameterization.reference_parameters)))
    dataset_scales = np.asarray((2.0, 3.0)) / reference_peak
    truth_model = model.predict_mass_A2(truth)
    dataset_index = np.repeat(np.arange(2), rows_per_dataset)
    is_background = np.tile(
        np.asarray((*([False] * signal_count), True, True)),
        2,
    )
    fixed_background_mass = np.full(2 * rows_per_dataset, 20.0)
    count_mass = fixed_background_mass + dataset_scales[dataset_index] * truth_model
    observations = MatchedRegionObservations(
        dataset_ids=("first", "second"),
        dataset_index=dataset_index,
        block_index=np.repeat(np.arange(2), rows_per_dataset),
        signal_family=np.tile(np.asarray((*range(signal_count), -1, -1)), 2),
        is_background=is_background,
        count_mass=count_mass,
        support_px2=np.ones(2 * rows_per_dataset),
        background_coordinate=np.tile(
            np.asarray((*([0.0] * signal_count), -1.0, 1.0)),
            2,
        ),
        required_signal_families=tuple(range(signal_count)),
        count_covariance_count2=np.eye(2 * rows_per_dataset),
    )
    fixed_background = FixedMatchedRegionBackground(
        count_mass=fixed_background_mass,
        covariance_count2=np.zeros((2 * rows_per_dataset, 2 * rows_per_dataset)),
        revision="generic-cif-ordered-sf-fixed-background.v1",
    )
    result = fit_parameterized_matched_regions(
        observations,
        model,
        fixed_background=fixed_background,
        initial_parameters=(np.asarray((0.0, 0.9)), np.asarray((0.015, 0.8))),
        lower_bounds=np.asarray((-0.02, 0.7)),
        upper_bounds=np.asarray((0.02, 1.0)),
        parameter_scales=np.asarray((0.01, 0.1)),
    )

    np.testing.assert_allclose(result.fit.parameters, truth, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(result.fit.dataset_scales, dataset_scales, rtol=2.0e-8, atol=0.0)
    assert result.fit.sensitivity_rank == 2
    assert result.fit.sensitivity_condition < 100.0
    assert result.fit.success
    assert result.model_revision == model.model_revision

    scale_basis = AffineCifSiteBasis(
        reference_crystal=inputs.crystal,
        parameter_names=("iodine_z_shift", "common_site_occupancy"),
        parameter_units=("fractional", "1"),
        reference_parameters=np.asarray((0.0, 1.0)),
        fractional_coefficients=fractional,
        occupancy_coefficients=np.column_stack((np.zeros(site_count), np.ones(site_count))),
        u_iso_A2_coefficients=np.zeros((site_count, 2)),
        unknown_u_iso_A2=0.0,
    )

    scale_parameterization = AffineCifFiniteStackParameterization(inputs.strength, scale_basis)
    scale_gauged_model = ParameterizedStructureRegionModel(
        parameterization=scale_parameterization,
        blocks=model.blocks,
    )

    scale_gauged_truth = np.asarray((0.008, 0.86))
    scale_gauged_truth_model = scale_gauged_model.predict_mass_A2(scale_gauged_truth)
    scale_gauged_observations = replace(
        observations,
        count_mass=(
            fixed_background_mass + dataset_scales[dataset_index] * scale_gauged_truth_model
        ),
    )
    with pytest.raises(StructureRegionIdentifiabilityError, match="rank deficient") as captured:
        fit_parameterized_matched_regions(
            scale_gauged_observations,
            scale_gauged_model,
            fixed_background=fixed_background,
            initial_parameters=(np.asarray((0.0, 0.9)),),
            lower_bounds=np.asarray((-0.02, 0.7)),
            upper_bounds=np.asarray((0.02, 1.0)),
            parameter_scales=np.asarray((0.01, 0.1)),
        )
    assert captured.value.result.sensitivity_rank == 1

    stale_parameterization = AffineCifFiniteStackParameterization(
        replace(inputs.strength, repeats=inputs.strength.repeats + 1),
        basis,
    )
    with pytest.raises(ValueError, match="different reference structure model"):
        ParameterizedStructureRegionModel(
            parameterization=stale_parameterization,
            blocks=model.blocks,
        )


@pytest.mark.parametrize(
    "config_name",
    ("bi2se3_simulation.yaml", "bi2te3_simulation.yaml"),
)
def test_bi2x3_structure_uses_same_parameterized_region_fitter(config_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_simulation_config(root / "configs" / config_name)
    config = replace(
        base,
        source=replace(
            _single_mean_test_source(base.source),
            spatial_sigma_m=(0.0, 0.0),
            divergence_sigma_rad=(0.0, 0.0),
        ),
        bragg=replace(base.bragg, include_detector_visible_m0=False),
    )
    inputs = build_configured_simulation_inputs(config)
    detector = build_source_averaged_structure_detector(inputs)
    nominal = build_nominal_ewald_context(inputs)
    parameterization = Bi2X3FiniteStackParameterization(
        inputs.strength,
        ("outer_bi_antisite_fraction",),
    )

    mapped = []
    for rod in nominal.rods:
        if rod.h == 0 and rod.k == 0:
            continue
        for branch in (1, 2):
            candidate = nominal.map_latent_geometry(
                rod=rod,
                branch=branch,
                alpha_rad=math.radians(1.5),
                beta_rad=math.radians(165.0 + 3.0 * len(mapped)),
            )
            if bool(candidate.valid):
                mapped.append(candidate)
                break
        if len(mapped) == 4:
            break
    assert len(mapped) == 4
    column_px = np.asarray([point.column_px for point in mapped])
    row_px = np.asarray([point.row_px for point in mapped])
    signal_count = len(mapped)
    rows_per_dataset = signal_count + 2
    observation_count = 2 * rows_per_dataset
    response = detector.compile_structure_response(column_px, row_px)
    model = ParameterizedStructureRegionModel(
        parameterization=parameterization,
        blocks=tuple(
            StructureRegionResponseBlock(
                dataset_id=dataset_id,
                response=response,
                quadrature=ContinuousRegionQuadrature(
                    column_px=column_px,
                    row_px=row_px,
                    detector_area_weight_px2=np.ones(signal_count),
                    observation_row=dataset_index * rows_per_dataset + np.arange(signal_count),
                    background_coordinate=np.zeros(signal_count),
                    observation_count=observation_count,
                    chart_revision=(f"shared-bi2x3-sf-{config_name}.dataset-{dataset_index}.v1"),
                ),
            )
            for dataset_index, dataset_id in enumerate(("first", "second"))
        ),
    )
    truth = np.asarray((0.08,))
    truth_mass = model.predict_mass_A2(truth)
    reference_peak = float(np.max(model.predict_mass_A2(model.reference_parameters)))
    dataset_scales = np.asarray((2.0, 3.0)) / reference_peak
    dataset_index = np.repeat(np.arange(2), rows_per_dataset)
    fixed_background_mass = np.full(observation_count, 15.0)
    observations = MatchedRegionObservations(
        dataset_ids=("first", "second"),
        dataset_index=dataset_index,
        block_index=np.repeat(np.arange(2), rows_per_dataset),
        signal_family=np.tile(np.asarray((*range(signal_count), -1, -1)), 2),
        is_background=np.tile(
            np.asarray((*([False] * signal_count), True, True)),
            2,
        ),
        count_mass=fixed_background_mass + dataset_scales[dataset_index] * truth_mass,
        support_px2=np.ones(observation_count),
        background_coordinate=np.tile(
            np.asarray((*([0.0] * signal_count), -1.0, 1.0)),
            2,
        ),
        required_signal_families=tuple(range(signal_count)),
        count_covariance_count2=np.eye(observation_count),
    )
    result = fit_parameterized_matched_regions(
        observations,
        model,
        fixed_background=FixedMatchedRegionBackground(
            count_mass=fixed_background_mass,
            covariance_count2=np.zeros((observation_count, observation_count)),
            revision=f"shared-bi2x3-sf-{config_name}-background.v1",
        ),
        initial_parameters=(np.asarray((0.02,)), np.asarray((0.14,))),
        lower_bounds=np.asarray((0.0,)),
        upper_bounds=np.asarray((0.2,)),
        parameter_scales=np.asarray((0.1,)),
    )

    np.testing.assert_allclose(result.fit.parameters, truth, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(result.fit.dataset_scales, dataset_scales, rtol=3.0e-8)
    assert result.fit.sensitivity_rank == 1
    assert result.fit.sensitivity_condition == pytest.approx(1.0)


def test_source_averaged_detector_compiles_all_blocks_before_parallel_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from rasim_next.pipeline._continuous_detector_kernel import CompiledDetectorEvaluator

    averaged, scalar_detectors = _two_state_source_averaged_detector_fixture()
    mapped = scalar_detectors[0].map_latent(
        rod=averaged.rods[1],
        branch=2,
        alpha_rad=math.radians(2.0),
        beta_rad=math.radians(178.0),
    )
    column_px = np.asarray((mapped.geometry.column_px,), dtype=np.float64)
    row_px = np.asarray((mapped.geometry.row_px,), dtype=np.float64)
    column_px.flags.writeable = False
    row_px.flags.writeable = False

    main_thread = threading.get_ident()
    warm_call_count = 0
    evaluated_evaluators: set[int] = set()
    phase = "parallel"
    original_evaluate_all_roots = CompiledDetectorEvaluator.evaluate_all_roots

    def evaluate_all_roots_spy(
        self: CompiledDetectorEvaluator,
        column_px: np.ndarray,
        row_px: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonlocal warm_call_count
        if not column_px.size:
            assert phase == "parallel"
            assert threading.get_ident() == main_thread
            assert not column_px.flags.writeable
            assert not row_px.flags.writeable
            warm_call_count += 1
        elif phase == "parallel":
            assert threading.get_ident() != main_thread
            assert warm_call_count == 1
            assert not column_px.flags.writeable
            assert not row_px.flags.writeable
            evaluated_evaluators.add(id(self))
        return original_evaluate_all_roots(self, column_px, row_px)

    monkeypatch.setattr(
        CompiledDetectorEvaluator,
        "evaluate_all_roots",
        evaluate_all_roots_spy,
    )
    parallel = averaged.evaluate_detector_coordinates_all_roots(column_px, row_px)

    assert warm_call_count == 1
    assert len(evaluated_evaluators) == averaged.valid_source_state_count
    phase = "serial"
    monkeypatch.setattr(type(averaged), "_thread_pool", lambda self: None)
    serial = averaged.evaluate_detector_coordinates_all_roots(column_px, row_px)

    np.testing.assert_array_equal(
        parallel.per_rod_density_A2_per_px2,
        serial.per_rod_density_A2_per_px2,
    )
    np.testing.assert_array_equal(parallel.density_A2_per_px2, serial.density_A2_per_px2)
    np.testing.assert_array_equal(parallel.caustic, serial.caustic)
    np.testing.assert_array_equal(parallel.valid_source_count, serial.valid_source_count)
    assert np.any(parallel.density_A2_per_px2 > 0.0)


def test_hybrid_cuda_preserves_regular_blocks_and_serializes_local_m0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasim_next.pipeline import _continuous_detector_cuda as cuda_module
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector

    inputs = _configured_inputs(sample_count=3)
    rods = tuple(rod for rod in inputs.rods if rod.family_m in (0, 1))
    detector = (
        build_source_averaged_detector(inputs)
        .restrict_rods(rods)
        .with_maximum_state_block_count(2)
        .with_specular_stitch(
            ParrattStitchStack(
                substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
            )
        )
    )
    column_px = np.asarray((1448.2, 1448.2, 1109.5), dtype=np.float64)
    row_px = np.asarray((1182.6, 1400.0, 1349.5), dtype=np.float64)
    oracle = detector.with_maximum_state_block_count(1).evaluate_detector_coordinates_all_roots(
        column_px, row_px
    )
    m0_index = next(index for index, rod in enumerate(detector.rods) if rod.family_m == 0)
    assert np.any(oracle.per_rod_density_A2_per_px2[:, m0_index] > 0.0)
    regular_block_counts: list[int] = []

    def fake_cuda(
        evaluator_blocks: tuple[tuple[object, ...], ...],
        column_px: np.ndarray,
        row_px: np.ndarray,
        *,
        detector_shape_rc: tuple[int, int],
        master_rod_count: int,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        del row_px, detector_shape_rc
        regular_block_counts.append(len(evaluator_blocks))
        return (
            np.asarray(oracle.per_rod_density_A2_per_px2, dtype=np.float64).copy(),
            np.asarray(oracle.caustic, dtype=np.bool_).copy(),
            np.asarray(oracle.valid_source_count, dtype=np.int64).copy(),
            "test-cuda",
        )

    monkeypatch.setattr(cuda_module, "evaluate_source_averaged_all_roots_cuda", fake_cuda)
    local_m0_block_counts: list[int] = []

    def local_m0_thread_pool_spy(self: SourceAveragedDetectorEwaldMeasure) -> None:
        active_rods = self.rods
        assert active_rods and all(rod.family_m == 0 for rod in active_rods)
        block_count = len(self._evaluator_blocks)
        local_m0_block_counts.append(block_count)
        assert block_count == 1
        return None

    monkeypatch.setattr(type(detector), "_thread_pool", local_m0_thread_pool_spy)
    result = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )

    assert regular_block_counts == [2]
    assert local_m0_block_counts == [1]
    assert result.execution_backend == "hybrid_cuda_cpu_local_m0.v1"
    np.testing.assert_allclose(
        result.per_rod_density_A2_per_px2,
        oracle.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(result.caustic, oracle.caustic)
    np.testing.assert_array_equal(result.valid_source_count, oracle.valid_source_count)


def test_stitched_local_m0_applies_external_path_attenuation_once() -> None:
    from rasim_next.optics.attenuation import external_path_attenuation
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector
    from rasim_next.pipeline.continuous_detector import evaluate_detector_coordinates_geometry

    mu_m_inv = 1.2
    baseline_inputs = _configured_inputs(
        sample_count=1,
        reference_wavelength_A=1.540592925,
    )
    attenuated_inputs = _configured_inputs(
        sample_count=1,
        reference_wavelength_A=1.540592925,
        detector_path_linear_attenuation_m_inv=mu_m_inv,
    )
    m0_rods = tuple(rod for rod in baseline_inputs.rods if rod.family_m == 0)
    stitch = ParrattStitchStack(
        substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
    )
    baseline = (
        build_source_averaged_detector(baseline_inputs)
        .restrict_rods(m0_rods)
        .with_specular_stitch(stitch)
    )
    attenuated = (
        build_source_averaged_detector(attenuated_inputs)
        .restrict_rods(m0_rods)
        .with_specular_stitch(stitch)
    )
    column_px = np.asarray((1448.2, 1448.2), dtype=np.float64)
    row_px = np.asarray((1182.6, 1400.0), dtype=np.float64)
    baseline_density = baseline.evaluate_detector_coordinates_all_roots(column_px, row_px)
    attenuated_density = attenuated.evaluate_detector_coordinates_all_roots(column_px, row_px)
    geometry = evaluate_detector_coordinates_geometry(
        column_px,
        row_px,
        incident=baseline_inputs.incident,
        instrument=baseline_inputs.instrument,
        ki_sample_Ainv=baseline_inputs.incident.states.k_film_phase_sample_Ainv[0],
        incident_state_index=0,
    )
    positive = baseline_density.per_rod_density_A2_per_px2[:, 0] > 0.0
    assert np.any(positive)
    np.testing.assert_allclose(
        attenuated_density.per_rod_density_A2_per_px2[positive, 0]
        / baseline_density.per_rod_density_A2_per_px2[positive, 0],
        external_path_attenuation(mu_m_inv, geometry.ray_distance_m[positive]),
        rtol=4.0e-13,
        atol=0.0,
    )


def test_source_averaged_detector_density_equals_independent_state_sum() -> None:
    from rasim_next.pipeline.continuous_detector import (
        DetectorEwaldMeasure,
        SampleQIntensityEnvelope,
    )

    averaged, scalar_detectors = _two_state_source_averaged_detector_fixture()
    rods = averaged.rods
    seed_rod = rods[1]
    mapped = tuple(
        detector.map_latent(
            rod=seed_rod,
            branch=2,
            alpha_rad=math.radians(2.0),
            beta_rad=math.radians(178.0),
        )
        for detector in scalar_detectors
    )
    column_px = np.asarray([item.geometry.column_px for item in mapped])
    row_px = np.asarray([item.geometry.row_px for item in mapped])

    result = averaged.evaluate_detector_coordinates(column_px, row_px, branch=2)
    scalar = tuple(
        detector.evaluate_detector_coordinates(column_px, row_px, rods=rods, branch=2)
        for detector in scalar_detectors
    )
    source_weight = averaged.incident.states.source_weight
    expected_per_rod = (
        source_weight[0] * scalar[0].per_rod_density_A2_per_px2
        + source_weight[1] * scalar[1].per_rod_density_A2_per_px2
    )

    np.testing.assert_allclose(
        result.per_rod_density_A2_per_px2,
        expected_per_rod,
        rtol=3.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_allclose(
        result.density_A2_per_px2,
        np.sum(expected_per_rod, axis=-1),
        rtol=3.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(result.caustic, scalar[0].caustic | scalar[1].caustic)
    np.testing.assert_array_equal(result.valid_source_count, np.asarray([2, 2]))
    assert result.source_revision == averaged.incident.states.source_revision
    assert result.measure_id == "raw_detector_coordinate_density_A2_per_px2.v1"
    for detector, scalar_result in zip(scalar_detectors, scalar, strict=True):
        internal_ki = detector.coating.ki_sample_Ainv
        valid = scalar_result.geometry.valid
        np.testing.assert_allclose(
            np.linalg.norm(scalar_result.geometry.q_sample_Ainv[valid] + internal_ki, axis=-1),
            np.linalg.norm(internal_ki),
            rtol=0.0,
            atol=4.0e-15,
        )

    envelope = SampleQIntensityEnvelope(u_radial_A2=0.017, u_normal_A2=0.031)
    enveloped = averaged.rebind_physics(intensity_envelope=envelope)
    enveloped_scalar_detectors = tuple(
        DetectorEwaldMeasure(
            coating=detector.coating,
            incident=detector.incident,
            material=averaged.material,
            instrument=detector.instrument,
            intensity_envelope=envelope,
        )
        for detector in scalar_detectors
    )
    enveloped_result = enveloped.evaluate_detector_coordinates(column_px, row_px, branch=2)
    enveloped_scalar = tuple(
        detector.evaluate_detector_coordinates(column_px, row_px, rods=rods, branch=2)
        for detector in enveloped_scalar_detectors
    )
    expected_enveloped = (
        source_weight[0] * enveloped_scalar[0].per_rod_density_A2_per_px2
        + source_weight[1] * enveloped_scalar[1].per_rod_density_A2_per_px2
    )
    np.testing.assert_allclose(
        enveloped_result.per_rod_density_A2_per_px2,
        expected_enveloped,
        rtol=3.0e-11,
        atol=2.0e-24,
    )


def test_parratt_stitch_is_m0_only_and_uses_one_local_lamella_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rasim_next.pipeline.source_averaged_detector as source_detector_module
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector

    inputs = _configured_inputs(sample_count=1)
    rods = tuple(rod for rod in inputs.rods if rod.family_m in (0, 1))
    plain = build_source_averaged_detector(inputs).restrict_rods(rods)
    stitched = plain.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
        )
    )
    column_px = np.asarray((1448.2, 1448.2, 1109.5), dtype=np.float64)
    row_px = np.asarray((1182.6, 1400.0, 1349.5), dtype=np.float64)
    baseline = plain.evaluate_detector_coordinates_all_roots(column_px, row_px)
    candidate = stitched.evaluate_detector_coordinates_all_roots(column_px, row_px)
    nonzero_rod = np.asarray([rod.family_m != 0 for rod in rods])

    np.testing.assert_array_equal(
        candidate.per_rod_density_A2_per_px2[:, nonzero_rod],
        baseline.per_rod_density_A2_per_px2[:, nonzero_rod],
    )
    assert np.any(
        candidate.per_rod_density_A2_per_px2[:, ~nonzero_rod]
        != baseline.per_rod_density_A2_per_px2[:, ~nonzero_rod]
    )
    assert stitched.specular_stitch_stack is not None

    def active_low_branch_stitch(
        stack: ParrattStitchStack,
        kinematic_at_l: object,
        *,
        wavelength_A: float,
        film_refractive_index: complex,
        film_thickness_A: float,
        c_A: float,
        grid_size: int = 513,
    ) -> CompiledParrattStitch:
        del wavelength_A, c_A, grid_size
        zero = float(np.asarray(kinematic_at_l(np.zeros(1, dtype=np.float64)), dtype=np.float64)[0])
        return CompiledParrattStitch(
            film_refractive_index=film_refractive_index,
            substrate_refractive_index=stack.substrate_refractive_index,
            film_thickness_A=film_thickness_A,
            top_roughness_A=stack.top_roughness_A,
            bottom_roughness_A=stack.bottom_roughness_A,
            qc_Ainv=1.0,
            zero_strength_A2=zero,
            dimensionless_scale_factor=1.0,
            blend_bounds_q_over_qc=(3.0, 6.0),
            blend_selection="fallback",
        )

    monkeypatch.setattr(
        source_detector_module,
        "compile_parratt_stitch",
        active_low_branch_stitch,
    )
    active = plain.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
        )
    )
    active_values = active.evaluate_detector_coordinates_all_roots(column_px, row_px)
    np.testing.assert_array_equal(
        active_values.per_rod_density_A2_per_px2[:, nonzero_rod],
        baseline.per_rod_density_A2_per_px2[:, nonzero_rod],
    )
    assert np.any(
        active_values.per_rod_density_A2_per_px2[:, ~nonzero_rod]
        != baseline.per_rod_density_A2_per_px2[:, ~nonzero_rod]
    )
    rebound = active.rebind_physics()
    rebound_values = rebound.evaluate_detector_coordinates_all_roots(column_px, row_px)
    np.testing.assert_array_equal(
        rebound_values.per_rod_density_A2_per_px2,
        active_values.per_rod_density_A2_per_px2,
    )
    assert active.detector_visible_m0_q_gap_Ainv == 0.0
    with pytest.raises(
        ValueError,
        match="forward pixel sampling does not implement local-lamella stitched m=0",
    ):
        active.sample_native_pixel_mass(
            draws_per_source_state=8,
            seed=4381,
            execution_backend="cpu",
        )
    from numba import cuda

    if cuda.is_available():
        gpu = active.evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
            execution_backend="cuda",
        )
        np.testing.assert_allclose(
            gpu.per_rod_density_A2_per_px2,
            active_values.per_rod_density_A2_per_px2,
            rtol=2.0e-11,
            atol=2.0e-24,
        )
        np.testing.assert_array_equal(
            gpu.valid_source_count,
            active_values.valid_source_count,
        )
        assert gpu.execution_backend == "hybrid_cuda_cpu_local_m0.v1"
        m0_group = np.asarray(
            [[rod.family_m == 0 for rod in active.rods]],
            dtype=np.bool_,
        )
        with pytest.raises(
            ValueError,
            match="CUDA selected source/rod-group evaluation does not implement",
        ):
            active.evaluate_selected_source_rod_groups_all_roots(
                column_px[:1],
                row_px[:1],
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                m0_group,
                execution_backend="cuda",
            )
        with pytest.raises(
            ValueError,
            match="forward pixel sampling does not implement local-lamella stitched m=0",
        ):
            active.sample_native_pixel_mass(
                draws_per_source_state=8,
                seed=4381,
                execution_backend="cuda",
            )


def test_fixed_external_qz_parratt_stitch_is_m0_only_and_preserves_cuda_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rasim_next.pipeline.source_averaged_detector as source_detector_module
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector

    inputs = _configured_inputs(sample_count=1)
    rods = tuple(rod for rod in inputs.rods if rod.family_m in (0, 1))
    plain = build_source_averaged_detector(inputs).restrict_rods(rods)

    def active_low_branch_stitch(
        stack: ParrattStitchStack,
        kinematic_at_l: object,
        *,
        wavelength_A: float,
        film_refractive_index: complex,
        film_thickness_A: float,
        c_A: float,
        grid_size: int = 513,
    ) -> CompiledParrattStitch:
        del wavelength_A, c_A, grid_size
        zero = float(np.asarray(kinematic_at_l(np.zeros(1, dtype=np.float64)), dtype=np.float64)[0])
        return CompiledParrattStitch(
            film_refractive_index=film_refractive_index,
            substrate_refractive_index=stack.substrate_refractive_index,
            film_thickness_A=film_thickness_A,
            top_roughness_A=stack.top_roughness_A,
            bottom_roughness_A=stack.bottom_roughness_A,
            qc_Ainv=1.0,
            zero_strength_A2=zero,
            dimensionless_scale_factor=1.0,
            blend_bounds_q_over_qc=(3.0, 6.0),
            blend_selection="fallback",
            interface_assumption=stack.interface_assumption,
        )

    monkeypatch.setattr(
        source_detector_module,
        "compile_parratt_stitch",
        active_low_branch_stitch,
    )
    stitched = plain.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
            interface_assumption="fixed_external_qz_m0_strength.v1",
        )
    )
    column_px = np.asarray((1448.2, 1448.2, 1109.5), dtype=np.float64)
    row_px = np.asarray((1182.6, 1400.0, 1349.5), dtype=np.float64)
    baseline = plain.evaluate_detector_coordinates_all_roots(column_px, row_px)
    candidate = stitched.evaluate_detector_coordinates_all_roots(column_px, row_px)
    nonzero_rod = np.asarray([rod.family_m != 0 for rod in rods])

    np.testing.assert_array_equal(
        candidate.per_rod_density_A2_per_px2[:, nonzero_rod],
        baseline.per_rod_density_A2_per_px2[:, nonzero_rod],
    )
    assert np.any(
        candidate.per_rod_density_A2_per_px2[:, ~nonzero_rod]
        != baseline.per_rod_density_A2_per_px2[:, ~nonzero_rod]
    )
    assert stitched.detector_visible_m0_q_gap_Ainv is not None
    assert stitched.detector_visible_m0_q_gap_Ainv > 0.0
    rebound = stitched.rebind_physics()
    np.testing.assert_array_equal(
        rebound.evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
        ).per_rod_density_A2_per_px2,
        candidate.per_rod_density_A2_per_px2,
    )

    sampler = stitched.compile_monte_carlo_sampler(execution_backend="cpu", seed=4381)
    local_stitch = plain.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
        )
    )
    changed_fixed_stitch = plain.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
            bottom_roughness_A=11.0,
            interface_assumption="fixed_external_qz_m0_strength.v1",
        )
    )
    for changed in (local_stitch, changed_fixed_stitch):
        with pytest.raises(
            ValueError,
            match="unchanged source, rods, physics, and detector shape",
        ):
            sampler.rebind_geometry(changed)

    from numba import cuda

    if cuda.is_available():
        gpu = stitched.evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
            execution_backend="cuda",
        )
        np.testing.assert_allclose(
            gpu.per_rod_density_A2_per_px2,
            candidate.per_rod_density_A2_per_px2,
            rtol=2.0e-11,
            atol=2.0e-24,
        )
        assert gpu.execution_backend == "numba_cuda_source_averaged.v1"


def test_parratt_undefined_intensity_reaches_the_public_failure_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numba import cuda

    from rasim_next.pipeline import source_averaged_detector as module

    inputs = _configured_inputs(sample_count=1)
    plain = build_source_averaged_detector(inputs)
    compile_stitch = module.compile_parratt_stitch

    def overflowing_stitch(*args: object, **kwargs: object) -> CompiledParrattStitch:
        return replace(
            compile_stitch(*args, **kwargs),
            qc_Ainv=1.0,
            blend_bounds_q_over_qc=(3.0, 6.0),
            zero_strength_A2=1e200,
            dimensionless_scale_factor=1e-200,
        )

    monkeypatch.setattr(module, "compile_parratt_stitch", overflowing_stitch)
    overflowed = plain.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
            interface_assumption="fixed_external_qz_m0_strength.v1",
        )
    )
    for backend in ("cpu", "cuda") if cuda.is_available() else ("cpu",):
        for evaluate in (
            overflowed.evaluate_detector_density_all_roots,
            overflowed.evaluate_detector_coordinates_all_roots,
        ):
            with pytest.raises(
                FloatingPointError, match=r"nonfinite Parratt|undefined physical intensity"
            ):
                evaluate(np.array([1448.2]), np.array([1400.0]), execution_backend=backend)


def test_compiled_parratt_preserves_equal_zero_limit_and_rejects_undefined_recursion() -> None:
    """A singular representation must never substitute unrelated kinematic strength."""
    from rasim_next.pipeline._continuous_detector_kernel import _empirical_parratt_strength_A2
    from rasim_next.reflectivity import parratt_reflectivity

    film = complex(0.999999)
    qz = 2.0 * math.sqrt(1.0 - film.real**2)
    arguments = (7.0, qz, 1.0, film, film, 20.0, 0.0, 0.0, qz, 2.0, 1.0, 3.0, 6.0)
    oracle = parratt_reflectivity(
        qz,
        2.0 * math.pi,
        refractive_index=(1.0, film, film),
        thickness_A=(None, 20.0, None),
        roughness_A=(0.0, 0.0),
    )
    expected = qz**2 * float(oracle.reflectivity) * 2.0
    assert _empirical_parratt_strength_A2(*arguments) == pytest.approx(expected, rel=1e-14)
    singular = (*arguments[:4], complex(1.0), *arguments[5:])
    with pytest.raises(ZeroDivisionError, match="Parratt recursion denominator is zero"):
        _empirical_parratt_strength_A2(*singular)

    from numba import cuda

    if cuda.is_available():
        from rasim_next.pipeline._continuous_detector_cuda import (
            _empirical_parratt_strength_A2 as device_strength,
        )

        @cuda.jit
        def evaluate_limits(output):
            output[0] = device_strength(
                7.0, qz, 1.0, film, film, 20.0, 0.0, 0.0, qz, 2.0, 1.0, 3.0, 6.0
            )
            output[1] = device_strength(
                7.0, qz, 1.0, film, complex(1.0), 20.0, 0.0, 0.0, qz, 2.0, 1.0, 3.0, 6.0
            )

        output = cuda.device_array(2, dtype=np.float64)
        evaluate_limits[1, 1](output)
        result = output.copy_to_host()
        assert result[0] == pytest.approx(expected, rel=1e-14)
        assert np.isnan(result[1])


def test_local_m0_preserves_signed_atomic_strength_and_positive_caustics() -> None:
    from painted_ewald import MosaicParameters
    from painted_ewald.normal_density import SphericalMosaicDensity
    from rasim_next.core.scattering import CLASSICAL_ELECTRON_RADIUS_A
    from rasim_next.pipeline._continuous_detector_kernel import CompiledDetectorEvaluator

    inputs = _configured_inputs(sample_count=1)
    detector = build_source_averaged_detector(inputs).restrict_rods((Rod(0, 0),))
    stitched = detector.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=0.99999 + 1e-8j,
        )
    )
    template = next(item.evaluator.state for block in stitched._evaluator_blocks for item in block)
    alpha = 0.3
    normal = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
    ki = np.array([math.sqrt(0.99), 0.0, -0.1])
    outgoing = ki - 2 * np.dot(ki, normal) * normal
    q = np.linalg.norm(outgoing - ki)
    f0 = np.zeros((2, 11))
    f0[:, 0] = 1.0
    state = replace(
        template,
        detector_zero_lab_m=outgoing,
        detector_column_step_lab_m=np.array([0.001, 0.0, 0.0]),
        detector_row_step_lab_m=np.array([0.0, 0.001, 0.0]),
        detector_pixel_area_vector_lab_m2=np.array([0.0, 0.0, 1e-6]),
        ray_origin_lab_m=np.zeros(3),
        sample_from_lab=np.eye(3),
        sample_from_local=np.eye(3),
        ki_film_sample_Ainv=ki,
        internal_k_Ainv=1.0,
        air_k0_Ainv=1.0,
        refractive_index=1 + 0j,
        entrance_amplitude=1 + 0j,
        incident_decay_Ainv=0.0,
        source_phase_weight=1.0,
        polarization_model_code=0,
        detector_path_linear_attenuation_m_inv=0.0,
        specular_qc_Ainv=1e-6,
        specular_substrate_refractive_index=1 + 0j,
        rod_u_bounds_Ainv=np.array([[-2.0, 2.0]]),
        b3_norm_Ainv=q,
        atom_fractional_offset=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]]),
        atom_occupancy_element=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        rod_atom_inplane_factor=np.ones((1, 2), dtype=np.complex128),
        f0_parameters=f0,
        anomalous_factor_e=np.array([0j, 1j]),
        layers=1,
        shared_disorder_epsilon=0.0,
        normalization_divisor=1.0,
        u_radial_A2=0.0,
        u_normal_A2=0.0,
        intensity_envelope_u_radial_A2=0.0,
        intensity_envelope_u_normal_A2=0.0,
        gaussian_sigma_rad=0.3,
        lorentzian_hwhm_rad=0.6,
        lorentzian_probability=0.4,
    )
    evaluator = CompiledDetectorEvaluator(state, (1, 1))
    law = SphericalMosaicDensity(MosaicParameters(0.3, 0.6, 0.4))
    coordinate = np.zeros(1)
    actual = evaluator.evaluate_all_roots(coordinate, coordinate, mosaic_density=law)[0][0, 0]
    # ell=1: |1+(1+i)exp(i*pi/2)|²=1; ell=-1 gives 5.
    expected = (
        CLASSICAL_ELECTRON_RADIUS_A**2
        * (1e-6 * outgoing[2])
        / q**2
        * (law.directed_density_sr_inv(alpha) + 5 * law.directed_density_sr_inv(math.pi - alpha))
    )
    assert actual == pytest.approx(float(expected), rel=2e-12, abs=0.0)

    small_f0 = np.zeros((2, 11))
    small_f0[0, 0] = math.sqrt(np.nextafter(0.0, 1.0) / CLASSICAL_ELECTRON_RADIUS_A**2)
    properties = state.atom_occupancy_element.copy()
    properties[1, 0] = 0.0
    small = replace(
        state,
        detector_zero_lab_m=np.array([0.6, 0.0, 0.8]),
        ki_film_sample_Ainv=np.array([0.6, 0.0, -0.8]),
        b3_norm_Ainv=1.6,
        gaussian_sigma_rad=2.0,
        lorentzian_hwhm_rad=1.0,
        lorentzian_probability=0.0,
        f0_parameters=small_f0,
        anomalous_factor_e=np.zeros(2, dtype=np.complex128),
        atom_occupancy_element=properties,
    )
    density, _, caustic, valid = CompiledDetectorEvaluator(small, (1, 1)).evaluate_all_roots(
        coordinate,
        coordinate,
    )
    assert np.isposinf(density[0, 0]) and caustic[0, 0] and valid[0]


def test_parratt_stitch_is_one_continuous_low_and_high_q_m0_field() -> None:
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector

    inputs = _configured_inputs(sample_count=1)
    m0_rods = tuple(rod for rod in inputs.rods if rod.family_m == 0)
    plain = build_source_averaged_detector(inputs).restrict_rods(m0_rods)
    stack = ParrattStitchStack(
        substrate_refractive_index=0.9999929532364343 + 9.672907455164902e-8j,
        bottom_roughness_A=10.0,
    )
    stitched = plain.with_specular_stitch(stack)
    full_stitched = build_source_averaged_detector(inputs).with_specular_stitch(stack)

    from painted_ewald.normal_density import SphericalMosaicDensity

    law = SphericalMosaicDensity(plain.mosaic)
    m0_evaluator = next(item.evaluator for block in stitched._evaluator_blocks for item in block)
    column = np.array([1448.2, 1420.0, 1460.0])
    row = np.array([1400.0, 1430.0, 1450.0])
    transfer = m0_evaluator.compile_local_m0_transfer(column, row, mosaic_density=law)
    direct = m0_evaluator.evaluate_all_roots(column, row, mosaic_density=law)[0][:, 0]
    np.testing.assert_allclose(
        transfer[:, 1] * law.directed_density_sr_inv(transfer[:, 0])
        + transfer[:, 2] * law.directed_density_sr_inv(np.pi - transfer[:, 0]),
        direct,
        rtol=2e-12,
        atol=2e-24,
    )
    assert np.all(np.isfinite(direct)) and np.any(direct > 0)
    non_m0_rod = next(rod for rod in full_stitched.rods if rod.family_m != 0)
    non_m0_only = full_stitched.restrict_rods((non_m0_rod,))
    assert non_m0_only.specular_stitch_stack is None
    non_m0_only.compile_monte_carlo_sampler(execution_backend="cpu", seed=4381)

    beta = math.radians(70.0)
    incident_sample = np.asarray(plain.incident.states.direction_sample[0])

    def local_reflection_coordinate(alpha_deg: float) -> tuple[float, float]:
        alpha = math.radians(alpha_deg)
        local_normal_sample = np.asarray(
            (
                math.sin(alpha) * math.cos(beta),
                math.sin(alpha) * math.sin(beta),
                math.cos(alpha),
            )
        )
        outgoing_sample = (
            incident_sample
            - 2.0 * float(np.dot(incident_sample, local_normal_sample)) * local_normal_sample
        )
        projection = project_detector_ray(
            np.asarray(plain.incident.states.sample_intersection_lab_m[0]),
            plain.instrument.lab_from_sample.apply_vector(outgoing_sample),
            plain.instrument,
        )
        assert projection.status.value == "VALID"
        return projection.column_px, projection.row_px

    low_parratt = local_reflection_coordinate(5.0)
    evanescent_mean_exit = local_reflection_coordinate(2.65414)
    direct_beam = project_detector_ray(
        np.asarray(plain.incident.states.sample_intersection_lab_m[0]),
        plain.instrument.lab_from_sample.apply_vector(incident_sample),
        plain.instrument,
    )
    assert direct_beam.status.value == "VALID"
    column_px = np.asarray(
        (low_parratt[0], evanescent_mean_exit[0], 1448.2, direct_beam.column_px),
        dtype=np.float64,
    )
    row_px = np.asarray(
        (low_parratt[1], evanescent_mean_exit[1], 1400.0, direct_beam.row_px),
        dtype=np.float64,
    )

    baseline = plain.evaluate_detector_density_all_roots(column_px, row_px)
    candidate = stitched.evaluate_detector_density_all_roots(column_px, row_px)

    np.testing.assert_array_equal(baseline.density_A2_per_px2[:2], np.zeros(2))
    assert np.all(np.isfinite(candidate.density_A2_per_px2))
    assert np.all(candidate.density_A2_per_px2[:3] > 0.0)
    assert candidate.density_A2_per_px2[3] == 0.0
    np.testing.assert_array_equal(candidate.valid_source_count, np.asarray((1, 1, 1, 0)))
    smooth = plain.with_specular_stitch(
        ParrattStitchStack(
            substrate_refractive_index=stack.substrate_refractive_index,
            bottom_roughness_A=0.0,
        )
    ).evaluate_detector_density_all_roots(column_px[:3], row_px[:3])
    assert not np.isclose(
        candidate.density_A2_per_px2[0],
        smooth.density_A2_per_px2[0],
        rtol=1.0e-2,
        atol=0.0,
    )
    np.testing.assert_allclose(
        candidate.density_A2_per_px2[1:3],
        smooth.density_A2_per_px2[1:3],
        rtol=2.0e-11,
        atol=2.0e-24,
    )
    structure = full_stitched.strength_model.structure_parameters
    assert structure is not None
    changed_strength = replace(
        full_stitched.strength_model,
        structure_parameters=replace(
            structure,
            bi_occupancy=0.71,
            se1_occupancy=0.83,
            se2_occupancy=0.77,
        ),
    )
    rebound = full_stitched.rebind_physics(strength_model=changed_strength)
    assert (
        rebound.incidence_angle_static_physics_revision
        != full_stitched.incidence_angle_static_physics_revision
    )
    m0_index = next(index for index, rod in enumerate(rebound.rods) if rod.family_m == 0)
    rebound_full = rebound.evaluate_detector_coordinates_all_roots(column_px, row_px)
    rebind_then_restrict = rebound.restrict_rods(
        (rebound.rods[m0_index],)
    ).evaluate_detector_coordinates_all_roots(column_px, row_px)
    restrict_then_rebind = (
        full_stitched.restrict_rods((full_stitched.rods[m0_index],))
        .rebind_physics(strength_model=changed_strength)
        .evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
        )
    )
    np.testing.assert_allclose(
        rebind_then_restrict.per_rod_density_A2_per_px2[:, 0],
        rebound_full.per_rod_density_A2_per_px2[:, m0_index],
        rtol=4.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_allclose(
        rebind_then_restrict.per_rod_density_A2_per_px2,
        restrict_then_rebind.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(rebind_then_restrict.caustic, restrict_then_rebind.caustic)
    np.testing.assert_array_equal(
        rebind_then_restrict.valid_source_count,
        restrict_then_rebind.valid_source_count,
    )
    from numba import cuda

    if cuda.is_available():
        cpu = rebound.evaluate_detector_density_all_roots(column_px, row_px)
        gpu = rebound.evaluate_detector_density_all_roots(
            column_px,
            row_px,
            execution_backend="cuda",
        )
        np.testing.assert_allclose(
            gpu.density_A2_per_px2,
            cpu.density_A2_per_px2,
            rtol=2.0e-11,
            atol=2.0e-24,
        )
        np.testing.assert_array_equal(gpu.valid_source_count, cpu.valid_source_count)
        assert gpu.execution_backend == "hybrid_cuda_cpu_local_m0.v1"


def test_continuous_fold_plan_binds_full_detector_geometry_and_active_rods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasim_next.pipeline.continuous_fold import (
        ContinuousFoldCorrectionPlan,
        apply_continuous_fold_correction_plan,
    )
    from rasim_next.pipeline.source_averaged_detector import (
        SourceAveragedDetectorEwaldMeasure,
        source_averaged_detector_geometry_revision,
    )

    averaged, _ = _two_state_source_averaged_detector_fixture()
    first = averaged.restrict_rods(averaged.rods[:3])
    second = averaged.restrict_rods(averaged.rods[3:])
    assert first.rod_catalog_revision == second.rod_catalog_revision
    first_revision = source_averaged_detector_geometry_revision(first)
    assert source_averaged_detector_geometry_revision(second) != first_revision

    detector_pose = averaged.instrument.lab_from_detector
    changed_pose = replace(
        detector_pose,
        translation_m=detector_pose.translation_m + np.asarray((1.0e-6, 0.0, 0.0)),
    )
    pose_rebound = first.rebind_geometry(
        incident=first.incident,
        instrument=replace(first.instrument, lab_from_detector=changed_pose),
    )
    assert source_averaged_detector_geometry_revision(pose_rebound) != first_revision

    physics_rebound = first.rebind_physics(
        mosaic=replace(
            first.mosaic,
            gaussian_sigma_rad=0.9 * first.mosaic.gaussian_sigma_rad,
        )
    )
    assert source_averaged_detector_geometry_revision(physics_rebound) == first_revision

    plan = ContinuousFoldCorrectionPlan(
        column_px=np.asarray((0.0, 0.0)),
        row_px=np.asarray((0.0, 0.0)),
        signed_detector_area_weight_px2=np.asarray((-1.0, 1.0)),
        observation_row=np.asarray((0, 0)),
        evaluator_index_by_coordinate=np.asarray((0, 0)),
        rod_group_index_by_coordinate=np.asarray((0, 0)),
        group_master_rod_mask=np.asarray(((True, False, False),)),
        transformed_node=np.asarray((False, True)),
        observation_count=1,
        detector_geometry_revision=first_revision,
        plan_sha256="0" * 64,
        crossing_entry_count=1,
    )
    evaluation_count = 0

    def evaluate_stub(self: object, *args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal evaluation_count
        evaluation_count += 1
        return (
            np.ones(2, dtype=np.float64),
            np.zeros(2, dtype=np.bool_),
            np.ones(2, dtype=np.bool_),
            "cpu",
            None,
        )

    monkeypatch.setattr(
        SourceAveragedDetectorEwaldMeasure,
        "evaluate_selected_source_rod_groups_all_roots",
        evaluate_stub,
    )
    for incompatible in (second, pose_rebound):
        with pytest.raises(ValueError, match="does not match"):
            apply_continuous_fold_correction_plan(incompatible, plan)
    assert evaluation_count == 0
    correction = apply_continuous_fold_correction_plan(physics_rebound, plan)
    assert evaluation_count == 1
    np.testing.assert_array_equal(correction.correction_A2, 0.0)


def test_source_averaged_detector_rebinds_mosaic_and_structure_with_function_parity() -> None:
    from rasim_next.ordered import (
        Bi2X3QuintupleLayerParameters,
        SiteDisplacementProfile,
        TransverseIsotropicSiteDisplacement,
    )
    from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope
    from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure

    averaged, scalar_detectors = _two_state_source_averaged_detector_fixture()
    reference = scalar_detectors[0]
    mapped = reference.map_latent(
        rod=averaged.rods[1],
        branch=2,
        alpha_rad=math.radians(2.0),
        beta_rad=math.radians(178.0),
    )
    column_px = np.asarray([float(mapped.geometry.column_px)])
    row_px = np.asarray([float(mapped.geometry.row_px)])

    changed_mosaic = replace(
        reference.coating.bragg_space.config.mosaic,
        gaussian_sigma_rad=math.radians(2.5),
        lorentzian_half_width_rad=math.radians(0.4),
        lorentzian_probability=0.27,
    )
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(
        reference.coating.bragg_space.strength_model.crystal
    )
    changed_strength = replace(
        reference.coating.bragg_space.strength_model,
        shared_disorder_epsilon=0.03,
        structure_parameters=replace(
            baseline,
            bi_fractional_z=baseline.bi_fractional_z + 0.001,
            se2_fractional_z=baseline.se2_fractional_z - 0.002,
            bi_occupancy=0.91,
            se1_occupancy=0.79,
            se2_occupancy=0.84,
            u_radial_A2=0.0,
            u_normal_A2=0.0,
            outer_bi_antisite_fraction=0.012,
        ),
        site_displacement_profile=SiteDisplacementProfile(
            sites=(
                TransverseIsotropicSiteDisplacement("Bi", 0.008, 0.031),
                TransverseIsotropicSiteDisplacement("Se1", 0.014, 0.019),
                TransverseIsotropicSiteDisplacement("Se2", 0.027, 0.011),
            ),
            scale=1.3,
            provenance="test site profile",
        ),
    )
    changed_envelope = SampleQIntensityEnvelope(u_radial_A2=0.006, u_normal_A2=0.013)
    geometry_rebound = averaged.rebind_geometry(
        incident=averaged.incident,
        instrument=averaged.instrument,
    )
    original_states = averaged.incident.states
    changed_states = object.__new__(type(original_states))
    for name in original_states.__slots__:
        object.__setattr__(changed_states, name, getattr(original_states, name))
    object.__setattr__(changed_states, "incident_model_id", "different-incident-transport.v1")
    changed_transport = replace(averaged.incident, states=changed_states)
    with pytest.raises(ValueError, match="transport identity"):
        averaged.rebind_geometry(
            incident=changed_transport,
            instrument=averaged.instrument,
        )
    rebound = geometry_rebound.rebind_physics(
        mosaic=changed_mosaic,
        strength_model=changed_strength,
        intensity_envelope=changed_envelope,
    )
    fresh = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=reference.coating.bragg_space.config.reciprocal_basis_Ainv,
        crystal_to_sample=averaged.instrument.sample_from_crystal.rotation,
        rods=averaged.rods,
        rod_catalog_revision=averaged.rod_catalog_revision,
        mosaic=changed_mosaic,
        strength_model=changed_strength,
        intensity_envelope=changed_envelope,
        incident=averaged.incident,
        material=averaged.material,
        instrument=averaged.instrument,
        worker_count=2,
    )

    rebound_value = rebound.evaluate_detector_coordinates_all_roots(column_px, row_px)
    fresh_value = fresh.evaluate_detector_coordinates_all_roots(column_px, row_px)
    np.testing.assert_allclose(
        rebound_value.per_rod_density_A2_per_px2,
        fresh_value.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(rebound_value.caustic, fresh_value.caustic)
    restricted = rebound.restrict_rods((rebound.rods[1],))
    restricted_value = restricted.evaluate_detector_coordinates_all_roots(column_px, row_px)
    np.testing.assert_allclose(
        restricted_value.per_rod_density_A2_per_px2[:, 0],
        rebound_value.per_rod_density_A2_per_px2[:, 1],
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(restricted_value.caustic[:, 0], rebound_value.caustic[:, 1])
    reordered = rebound.restrict_rods((rebound.rods[2], rebound.rods[0]))
    reordered_value = reordered.evaluate_detector_coordinates_all_roots(column_px, row_px)
    np.testing.assert_allclose(
        reordered_value.per_rod_density_A2_per_px2,
        rebound_value.per_rod_density_A2_per_px2[:, (2, 0)],
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(
        reordered_value.caustic,
        rebound_value.caustic[:, (2, 0)],
    )
    assert restricted.incident is rebound.incident
    assert restricted.instrument is rebound.instrument
    assert rebound.incident is averaged.incident
    assert rebound.instrument is averaged.instrument
    assert rebound.material is averaged.material


def test_pixel_center_sampling_conservatively_prunes_impossible_top_exit_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rasim_next.pipeline.configured_simulation as configured_simulation_module
    from rasim_next.pipeline.configured_simulation import sample_detector_pixel_center_density
    from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure

    averaged, _ = _two_state_source_averaged_detector_fixture(detector_shape_rc=(3, 4))
    tiny = averaged.restrict_rods((averaged.rods[0],))
    monkeypatch.setattr(configured_simulation_module, "_MAXIMUM_MACROBIN_COORDINATES_PER_CALL", 4)

    def with_instrument(instrument: CompiledInstrument) -> SourceAveragedDetectorEwaldMeasure:
        return SourceAveragedDetectorEwaldMeasure(
            reciprocal_basis_Ainv=tiny.strength_model.reciprocal_basis_Ainv,
            crystal_to_sample=instrument.sample_from_crystal.rotation,
            rods=tiny.rods,
            rod_catalog_revision=tiny.rod_catalog_revision,
            mosaic=tiny.mosaic,
            strength_model=tiny.strength_model,
            incident=tiny.incident,
            material=tiny.material,
            instrument=instrument,
            worker_count=2,
        )

    base = replace(tiny.instrument, detector_row_pitch_m=1.0e-3)
    normal_lab = base.sample_from_lab.rotation[2]
    detector_rotation = base.lab_from_detector.rotation
    column_step_lab = detector_rotation[:, 0] * base.detector_column_pitch_m
    row_step_lab = detector_rotation[:, 1] * base.detector_row_pitch_m
    reference_column, reference_row = base.detector_reference_coordinate_px
    detector_zero_lab = (
        base.lab_from_detector.translation_m
        - reference_column * column_step_lab
        - reference_row * row_step_lab
    )
    minimum_origin_normal = float(
        np.min(
            tiny.incident.states.sample_intersection_lab_m[tiny.incident.states.valid] @ normal_lab
        )
    )
    row_step_normal = float(row_step_lab @ normal_lab)
    reference_shift = (
        float((detector_zero_lab + row_step_lab) @ normal_lab) - minimum_origin_normal
    ) / row_step_normal
    partial_instrument = replace(
        base,
        detector_reference_coordinate_px=(reference_column, reference_row + reference_shift),
    )
    partial = with_instrument(partial_instrument)
    column_grid, row_grid = np.meshgrid(np.arange(4.0), np.arange(3.0))
    direct = partial.evaluate_detector_density_all_roots(column_grid, row_grid)
    sampled = sample_detector_pixel_center_density(partial, execution_backend="cpu")

    assert sampled.coordinate_evaluation_count == 8
    np.testing.assert_array_equal(sampled.image_A2_per_px2, direct.density_A2_per_px2)
    np.testing.assert_array_equal(sampled.valid_source_count, direct.valid_source_count)
    np.testing.assert_array_equal(sampled.valid_source_count[0], 2)
    assert np.all(sampled.image_A2_per_px2[0] > 0.0)

    partial_reference_column, partial_reference_row = (
        partial_instrument.detector_reference_coordinate_px
    )
    partial_zero_lab = (
        partial_instrument.lab_from_detector.translation_m
        - partial_reference_column * column_step_lab
        - partial_reference_row * row_step_lab
    )
    point_normal = (
        partial_zero_lab @ normal_lab
        + column_grid * (column_step_lab @ normal_lab)
        + row_grid * row_step_normal
    )
    target_maximum = minimum_origin_normal - 1.0e-6
    all_culled_shift = (float(np.max(point_normal)) - target_maximum) / row_step_normal
    all_culled_instrument = replace(
        partial_instrument,
        detector_reference_coordinate_px=(
            partial_reference_column,
            partial_reference_row + all_culled_shift,
        ),
    )
    all_culled = with_instrument(all_culled_instrument)
    direct_zero = all_culled.evaluate_detector_density_all_roots(column_grid, row_grid)
    sampled_zero = sample_detector_pixel_center_density(all_culled, execution_backend="cpu")

    assert sampled_zero.coordinate_evaluation_count == 0
    assert sampled_zero.execution_backend == "numba_cpu_source_averaged.v1"
    assert sampled_zero.execution_device is None
    np.testing.assert_array_equal(sampled_zero.image_A2_per_px2, direct_zero.density_A2_per_px2)
    np.testing.assert_array_equal(sampled_zero.valid_source_count, direct_zero.valid_source_count)
    np.testing.assert_array_equal(sampled_zero.image_A2_per_px2, 0.0)


def test_total_detector_density_preserves_partial_valid_source_count() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_nominal_ewald_context,
        build_source_averaged_detector,
    )

    base = _configured_inputs(
        sample_count=2,
        reference_wavelength_A=1.5419012588932806,
    )
    points_sample_m = base.instrument.sample_from_lab.apply_point(
        base.incident.states.sample_intersection_lab_m
    )
    abs_x_m = np.abs(points_sample_m[:, 0])
    assert abs_x_m[0] != abs_x_m[1]

    finite_config = replace(
        base.config,
        instrument=replace(
            base.config.instrument,
            sample_support_model_id="finite_rectangle.v1",
            sample_width_m=float(np.sum(abs_x_m)),
            sample_length_m=4.0 * float(np.max(np.abs(points_sample_m[:, 1]))),
        ),
    )
    inputs = build_configured_simulation_inputs(finite_config)
    assert np.count_nonzero(inputs.incident.states.valid) == 1

    nominal = build_nominal_ewald_context(inputs)
    mapped = nominal.map_latent_geometry(
        rod=Rod(-1, 1),
        branch=2,
        alpha_rad=math.radians(2.0),
        beta_rad=math.radians(178.0),
    )
    assert bool(mapped.valid)

    result = build_source_averaged_detector(inputs).evaluate_detector_density_all_roots(
        np.asarray([mapped.column_px]),
        np.asarray([mapped.row_px]),
    )
    assert result.source_state_count == 2
    np.testing.assert_array_equal(result.valid_source_count, np.asarray([1]))


def test_source_averaged_pixel_integral_is_one_outer_integral_of_state_sum() -> None:
    from rasim_next.pipeline.continuous_detector import DetectorQuadrature

    detector_shape = (8, 8)
    averaged, scalar_detectors = _two_state_source_averaged_detector_fixture(
        detector_shape_rc=detector_shape
    )
    quadrature = DetectorQuadrature(
        pixel_gauss_order=2,
        fold_gauss_order=2,
        fold_subdivision_count=1,
        row_chunk_size=2,
    )

    with pytest.raises(ValueError, match="per-rod pixel evidence is disabled by default"):
        averaged.integrate_native_pixels(branch=2, quadrature=quadrature)
    result = averaged.integrate_native_pixels(
        branch=2,
        quadrature=quadrature,
        include_per_rod_evidence=True,
    )
    scalar = tuple(
        detector.integrate_native_pixels(
            rods=averaged.rods,
            branch=2,
            quadrature=quadrature,
        )
        for detector in scalar_detectors
    )
    source_weight = averaged.incident.states.source_weight
    expected_image = source_weight[0] * scalar[0].image_A2 + source_weight[1] * scalar[1].image_A2
    expected_per_rod = (
        source_weight[0] * scalar[0].per_rod_detector_mass_A2
        + source_weight[1] * scalar[1].per_rod_detector_mass_A2
    )

    np.testing.assert_allclose(result.image_A2, expected_image, rtol=3.0e-11, atol=2.0e-24)
    np.testing.assert_allclose(
        result.per_rod_detector_mass_A2,
        expected_per_rod,
        rtol=3.0e-11,
        atol=2.0e-24,
    )
    assert result.coordinate_evaluation_count == detector_shape[0] * detector_shape[1] * 2**2
    assert result.execution_backend == "numba_source_averaged.v1"
    assert not result.adaptive_tolerance_satisfied


def test_detector_macrobin_total_matches_explicit_per_rod_quadrature_oracle() -> None:
    from rasim_next.pipeline.configured_simulation import integrate_detector_macrobins

    detector_shape = (8, 8)
    averaged, _ = _two_state_source_averaged_detector_fixture(detector_shape_rc=detector_shape)
    bin_size_px = 2
    gauss_order = 2
    column_center = -0.5 + (np.arange(detector_shape[1] // bin_size_px) + 0.5) * bin_size_px
    row_center = -0.5 + (np.arange(detector_shape[0] // bin_size_px) + 0.5) * bin_size_px
    nodes, weights = np.polynomial.legendre.leggauss(gauss_order)
    half_width = 0.5 * bin_size_px
    offset = half_width * nodes
    mapped_weight = half_width * weights
    column_grid, row_grid, row_offset_grid, column_offset_grid = np.broadcast_arrays(
        column_center[None, :, None, None],
        row_center[:, None, None, None],
        offset[None, None, :, None],
        offset[None, None, None, :],
    )
    detailed = averaged.evaluate_detector_coordinates_all_roots(
        column_grid + column_offset_grid,
        row_grid + row_offset_grid,
    )
    assert not np.any(detailed.caustic)
    node_weight = mapped_weight[:, None] * mapped_weight[None, :]
    old_order_per_rod_A2 = np.sum(
        detailed.per_rod_density_A2_per_px2 * node_weight[None, None, :, :, None],
        axis=(2, 3),
        dtype=np.float64,
    )
    old_order_image_A2 = np.sum(old_order_per_rod_A2, axis=-1, dtype=np.float64)
    assert np.ptp(old_order_image_A2) > 0.0
    production = integrate_detector_macrobins(
        averaged,
        bin_size_px=bin_size_px,
        gauss_order=gauss_order,
    )
    np.testing.assert_allclose(
        production.image_A2,
        old_order_image_A2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )


def test_source_average_all_roots_includes_detector_regularized_m0() -> None:
    from painted_ewald import (
        BraggSpaceConfig,
        ContinuousEwaldCoating,
        MosaicBraggSpace,
        Rod,
    )
    from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure, DetectorQuadrature
    from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure

    nonzero, scalar_detectors = _two_state_source_averaged_detector_fixture()
    rods = (Rod(0, 0), *nonzero.rods)
    strength = scalar_detectors[0].coating.bragg_space.strength_model
    material = material_optics(strength.crystal, nonzero.incident.states.wavelength_A)
    detector = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=scalar_detectors[0].coating.bragg_space.config.reciprocal_basis_Ainv,
        crystal_to_sample=nonzero.instrument.sample_from_crystal.rotation,
        rods=rods,
        rod_catalog_revision="integration-test-rods.v1",
        mosaic=scalar_detectors[0].coating.bragg_space.config.mosaic,
        strength_model=strength,
        incident=nonzero.incident,
        material=material,
        instrument=nonzero.instrument,
        worker_count=2,
    )
    with pytest.raises(ValueError, match="pixel integration cannot include m=0"):
        detector.integrate_native_pixels(
            branch=2,
            include_per_rod_evidence=True,
            quadrature=DetectorQuadrature(
                pixel_gauss_order=2,
                fold_gauss_order=2,
                fold_subdivision_count=1,
                row_chunk_size=2,
            ),
        )
    seed = scalar_detectors[0].map_latent(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=math.radians(2.0),
        beta_rad=math.radians(178.0),
    )
    column_px = np.asarray([seed.geometry.column_px])
    row_px = np.asarray([seed.geometry.row_px])

    intrinsic_nonzero = scalar_detectors[0].map_detector_visible_coating(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=np.asarray([math.radians(2.0)]),
        beta_rad=np.asarray([math.radians(178.0)]),
    )
    direct_nonzero = scalar_detectors[0].coating.evaluate_latent(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=np.asarray([math.radians(2.0)]),
        beta_rad=np.asarray([math.radians(178.0)]),
    )
    assert intrinsic_nonzero.geometry.valid[0]
    np.testing.assert_allclose(
        intrinsic_nonzero.coating_intensity_density_A2_rad2_inv,
        direct_nonzero.coating_intensity_density_A2_rad2_inv,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        intrinsic_nonzero.measure_id
        == "detector_visible_intrinsic_ewald_latent_density_A2_rad2_inv.v1"
    )
    zero_weight_detector = DetectorEwaldMeasure(
        coating=scalar_detectors[0].coating,
        incident=scalar_detectors[0].incident,
        material=material,
        instrument=nonzero.instrument,
        phase_population_weight=0.0,
        polarization_weight=0.0,
    )
    zero_weight_intrinsic = zero_weight_detector.map_detector_visible_coating(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=np.asarray([math.radians(2.0)]),
        beta_rad=np.asarray([math.radians(178.0)]),
    )
    np.testing.assert_allclose(
        zero_weight_intrinsic.coating_intensity_density_A2_rad2_inv,
        intrinsic_nonzero.coating_intensity_density_A2_rad2_inv,
        rtol=0.0,
        atol=0.0,
    )

    all_roots = detector.evaluate_detector_coordinates_all_roots(column_px, row_px)
    total_density = detector.evaluate_detector_density_all_roots(column_px, row_px)
    explicit_cpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    np.testing.assert_array_equal(
        explicit_cpu.per_rod_density_A2_per_px2,
        all_roots.per_rod_density_A2_per_px2,
    )
    np.testing.assert_allclose(
        total_density.density_A2_per_px2,
        np.sum(all_roots.per_rod_density_A2_per_px2, axis=-1, dtype=np.float64),
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(total_density.caustic, np.any(all_roots.caustic, axis=-1))
    np.testing.assert_array_equal(
        total_density.valid_source_count,
        all_roots.valid_source_count,
    )
    assert total_density.source_state_count == all_roots.source_state_count
    assert total_density.source_revision == all_roots.source_revision
    assert total_density.root_policy == all_roots.root_policy
    assert total_density.detector_visible_m0_q_gap_Ainv == (
        all_roots.detector_visible_m0_q_gap_Ainv
    )
    assert total_density.measure_id == all_roots.measure_id
    assert total_density.execution_backend == all_roots.execution_backend
    with pytest.raises(ValueError, match="infinite detector density requires a caustic"):
        replace(
            total_density,
            density_A2_per_px2=np.full(column_px.shape, np.inf),
            caustic=np.zeros(column_px.shape, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="identify every CUDA-backed result"):
        replace(all_roots, execution_device="unexpected device")
    with pytest.raises(ValueError, match="identify every CUDA-backed result"):
        replace(all_roots, execution_backend="numba_cuda_source_averaged.v1")
    with pytest.raises(ValueError, match="execution_backend"):
        detector.evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
            execution_backend="automatic",
        )
    lower = nonzero.evaluate_detector_coordinates(column_px, row_px, branch=1)
    upper = nonzero.evaluate_detector_coordinates(column_px, row_px, branch=2)

    assert all_roots.root_policy == "all_retained_roots.v1"
    assert all_roots.branch is None
    assert all_roots.detector_visible_m0_q_gap_Ainv > 0.3
    assert all_roots.rods[0] == Rod(0, 0)
    assert np.isfinite(all_roots.per_rod_density_A2_per_px2[..., 0]).all()
    assert np.all(all_roots.per_rod_density_A2_per_px2[..., 0] > 0.0)
    expected_m0 = np.zeros(column_px.shape, dtype=np.float64)
    first_m0_oracle = None
    for scalar_detector, source_weight in zip(
        scalar_detectors,
        detector.incident.states.source_weight,
        strict=True,
    ):
        scalar_bragg = MosaicBraggSpace(
            BraggSpaceConfig(
                reciprocal_basis_Ainv=(
                    scalar_detector.coating.bragg_space.config.reciprocal_basis_Ainv
                ),
                crystal_to_sample=nonzero.instrument.sample_from_crystal.rotation,
                rods=rods,
                mosaic=scalar_detector.coating.bragg_space.config.mosaic,
                k_norm_Ainv=scalar_detector.coating.bragg_space.config.k_norm_Ainv,
            ),
            strength,
        )
        scalar_oracle = DetectorEwaldMeasure(
            coating=ContinuousEwaldCoating(
                scalar_bragg,
                ki_sample_Ainv=scalar_detector.coating.ki_sample_Ainv,
            ),
            incident=scalar_detector.incident,
            material=material,
            instrument=nonzero.instrument,
        )
        if first_m0_oracle is None:
            first_m0_oracle = scalar_oracle
            intrinsic_m0 = scalar_oracle.map_detector_visible_coating(
                rod=rods[0],
                branch=0,
                alpha_rad=np.asarray([0.0]),
                beta_rad=np.asarray([0.0]),
            )
            assert intrinsic_m0.geometry.valid[0]
            assert intrinsic_m0.detector_visible_m0_q_gap_Ainv > 0.3
            m0_geometry = intrinsic_m0.geometry.ewald_geometry
            q_norm = float(np.linalg.norm(m0_geometry.q_sample_Ainv[0]))
            assert q_norm > intrinsic_m0.detector_visible_m0_q_gap_Ainv
            assert abs(float(m0_geometry.u_Ainv[0])) > 0.0
            assert m0_geometry.ewald_residual_Ainv[0] < 4.0e-15
            latent_m0 = scalar_bragg.evaluate_latent(
                rod=rods[0],
                alpha_rad=m0_geometry.alpha_rad,
                beta_rad=m0_geometry.beta_rad,
                u_Ainv=m0_geometry.u_Ainv,
            )
            incident_norm = float(np.linalg.norm(scalar_oracle.coating.ki_sample_Ainv))
            np.testing.assert_allclose(
                intrinsic_m0.coating_intensity_density_A2_rad2_inv,
                latent_m0.intensity_density_A2_rad2_inv * (2.0 * incident_norm / q_norm),
                rtol=3.0e-14,
                atol=0.0,
            )
            with pytest.raises(ValueError, match="branch 0"):
                scalar_oracle.map_detector_visible_coating(
                    rod=rods[0],
                    branch=1,
                    alpha_rad=0.0,
                    beta_rad=0.0,
                )
        geometry, optical = scalar_oracle._detector_coordinate_state(column_px, row_px)
        scattering_polarization = scalar_oracle._event_scattering_polarization(
            geometry.kf_air_sample_Ainv,
            geometry.valid,
        )
        for branch in (1, 2):
            density, _, _ = scalar_oracle._inverse_rod_density(
                q_sample_Ainv=geometry.q_sample_Ainv,
                kf_sample_Ainv=geometry.kf_film_sample_Ainv,
                surface_jacobian_Ainv2_per_output=(geometry.q_surface_jacobian_Ainv2_per_px2),
                coordinate_valid=geometry.valid,
                optical_weight=optical,
                source_phase_weight=scalar_oracle._source_phase_weight,
                rod=rods[0],
                branch=branch,
                scattering_polarization=scattering_polarization,
            )
            expected_m0 += source_weight * density
    np.testing.assert_allclose(
        all_roots.per_rod_density_A2_per_px2[..., 0],
        expected_m0,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    sparse_response = compile_source_averaged_detector_structure_response(
        column_px,
        row_px,
        reciprocal_basis_Ainv=(
            scalar_detectors[0].coating.bragg_space.config.reciprocal_basis_Ainv
        ),
        crystal_to_sample=nonzero.instrument.sample_from_crystal.rotation,
        rods=rods,
        rod_catalog_revision=detector.rod_catalog_revision,
        mosaic=scalar_detectors[0].coating.bragg_space.config.mosaic,
        reference_strength_model=strength,
        intensity_envelope=detector.intensity_envelope,
        incident=detector.incident,
        material=detector.material,
        instrument=detector.instrument,
    )
    sparse = sparse_response.apply_strength(strength)
    np.testing.assert_allclose(
        sparse.per_rod_density_A2_per_px2,
        all_roots.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    m0_terms = sparse_response.term_rod_index == 0
    assert np.count_nonzero(m0_terms) == 4
    assert set(sparse_response.term_source_state_index[m0_terms]) == {0, 1}
    assert set(sparse_response.term_root_sign[m0_terms]) == {-1, 1}

    assert first_m0_oracle is not None
    positive_k = first_m0_oracle.coating.ki_sample_Ainv.copy()
    positive_k[2] = abs(positive_k[2])
    positive_kz = first_m0_oracle.incident.states.kz_film_Ainv.copy()
    positive_kz.real[:] = np.abs(positive_kz.real)
    positive_states = replace(
        first_m0_oracle.incident.states,
        k_film_phase_sample_Ainv=positive_k[None, :],
        kz_film_Ainv=positive_kz,
    )
    positive_incident = replace(first_m0_oracle.incident, states=positive_states)
    with pytest.raises(ValueError, match="negative sample-normal half-space"):
        SourceAveragedDetectorEwaldMeasure(
            reciprocal_basis_Ainv=first_m0_oracle.coating.bragg_space.config.reciprocal_basis_Ainv,
            crystal_to_sample=nonzero.instrument.sample_from_crystal.rotation,
            rods=rods,
            rod_catalog_revision="integration-test-rods.v1",
            mosaic=first_m0_oracle.coating.bragg_space.config.mosaic,
            strength_model=strength,
            incident=positive_incident,
            material=material,
            instrument=nonzero.instrument,
        )
    positive_detector = DetectorEwaldMeasure(
        coating=ContinuousEwaldCoating(
            first_m0_oracle.coating.bragg_space,
            ki_sample_Ainv=positive_k,
        ),
        incident=positive_incident,
        material=material,
        instrument=nonzero.instrument,
    )
    with pytest.raises(ValueError, match="negative incident sample-normal"):
        positive_detector.map_detector_visible_coating(
            rod=rods[0],
            branch=0,
            alpha_rad=0.0,
            beta_rad=0.0,
        )
    np.testing.assert_allclose(
        all_roots.per_rod_density_A2_per_px2[..., 1:],
        lower.per_rod_density_A2_per_px2 + upper.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )


def test_detector_visible_ewald_direction_density_includes_regular_m0() -> None:
    context = build_nominal_ewald_context(_configured_inputs(sample_count=1, sample_angle_deg=10.0))
    m0_rod = next(rod for rod in context.rods if rod.family_m == 0)
    alpha_rad = math.radians(2.0)
    beta_rad = math.radians(90.0)
    derivative_step_rad = 1.0e-6
    k_magnitude_Ainv = float(np.linalg.norm(context.ki_sample_Ainv))

    seed = context.geometry.map_detector_visible_coating(
        rod=m0_rod,
        branch=0,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
    )
    assert bool(seed.geometry.valid)

    evaluated = context.geometry.evaluate_detector_visible_ewald_directions(
        seed.geometry.column_px,
        seed.geometry.row_px,
        rods=(m0_rod,),
    )

    expected_density_A2_per_sr = 0.0
    for inverse_alpha_rad, inverse_beta_rad in (
        (alpha_rad, beta_rad),
        (np.pi - alpha_rad, np.remainder(beta_rad + np.pi, 2.0 * np.pi)),
    ):
        center = context.geometry.map_detector_visible_coating(
            rod=m0_rod,
            branch=0,
            alpha_rad=inverse_alpha_rad,
            beta_rad=inverse_beta_rad,
        )
        alpha_pair = context.geometry.map_detector_visible_coating(
            rod=m0_rod,
            branch=0,
            alpha_rad=np.asarray(
                (inverse_alpha_rad - derivative_step_rad, inverse_alpha_rad + derivative_step_rad)
            ),
            beta_rad=inverse_beta_rad,
        )
        beta_pair = context.geometry.map_detector_visible_coating(
            rod=m0_rod,
            branch=0,
            alpha_rad=inverse_alpha_rad,
            beta_rad=np.asarray(
                (inverse_beta_rad - derivative_step_rad, inverse_beta_rad + derivative_step_rad)
            ),
        )
        dn_dalpha = np.diff(
            alpha_pair.geometry.ewald_geometry.kf_sample_Ainv / k_magnitude_Ainv,
            axis=0,
        )[0] / (2.0 * derivative_step_rad)
        dn_dbeta = np.diff(
            beta_pair.geometry.ewald_geometry.kf_sample_Ainv / k_magnitude_Ainv,
            axis=0,
        )[0] / (2.0 * derivative_step_rad)
        expected_density_A2_per_sr += float(
            center.coating_intensity_density_A2_rad2_inv
            / np.linalg.norm(np.cross(dn_dalpha, dn_dbeta))
        )

    assert bool(evaluated.detector_visible)
    assert evaluated.detector_status.item() == "VALID"
    assert evaluated.detector_visible_m0_q_gap_Ainv == pytest.approx(0.7071558945421385)
    assert evaluated.per_rod_inverse_branch_count.item() == 2
    assert np.isfinite(evaluated.density_A2_per_sr)
    assert np.linalg.norm(evaluated.geometry.q_sample_Ainv) > (
        evaluated.detector_visible_m0_q_gap_Ainv
    )
    np.testing.assert_allclose(
        evaluated.density_A2_per_sr,
        expected_density_A2_per_sr,
        rtol=5.0e-8,
        atol=0.0,
    )
    assert evaluated.measure_id == "detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1"

    outside = context.geometry.evaluate_detector_visible_ewald_directions(
        -1.0,
        -1.0,
        rods=(m0_rod,),
    )
    assert not bool(outside.detector_visible)
    assert outside.detector_status.item() == "OUTSIDE_SUPPORT"
    assert outside.density_A2_per_sr == 0.0
    assert not evaluated.density_A2_per_sr.flags.writeable
    assert not evaluated.detector_visible.flags.writeable

    nonzero_rod = next(rod for rod in context.rods if (rod.h, rod.k) == (-1, 0))
    nonzero_seed = context.geometry.map_latent_geometry(
        rod=nonzero_rod,
        branch=2,
        alpha_rad=math.radians(0.1),
        beta_rad=0.0,
    )
    assert bool(nonzero_seed.valid)
    nonzero_visible = context.geometry.evaluate_detector_visible_ewald_directions(
        nonzero_seed.column_px,
        nonzero_seed.row_px,
        rods=(nonzero_rod,),
    )
    nonzero_intrinsic = context.geometry.evaluate_intrinsic_ewald_directions(
        nonzero_seed.ewald_geometry.kf_sample_Ainv / k_magnitude_Ainv,
        rods=(nonzero_rod,),
    )
    np.testing.assert_allclose(
        nonzero_visible.density_A2_per_sr,
        nonzero_intrinsic.density_A2_per_sr,
        rtol=4.0e-13,
        atol=0.0,
    )
    assert nonzero_visible.detector_visible_m0_q_gap_Ainv is None


def test_cuda_detector_backend_fails_closed_without_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasim_next.pipeline import _continuous_detector_cuda as cuda_backend
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    config = replace(config, source=_single_mean_test_source(config.source))
    detector = build_source_averaged_detector(build_configured_simulation_inputs(config))
    monkeypatch.setattr(cuda_backend.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        detector.evaluate_detector_density_all_roots(
            np.asarray([0.0]),
            np.asarray([0.0]),
            execution_backend="cuda",
        )
    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        detector.sample_native_pixel_mass(
            draws_per_source_state=1,
            seed=7,
            execution_backend="cuda",
        )


def test_cuda_forward_monte_carlo_matches_cpu_and_progressive_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numba import cuda

    from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope

    if not cuda.is_available():
        pytest.skip("requires a CUDA device")

    _, _, detector = _forward_monte_carlo_fixture(
        source_count=3,
        worker_count=4,
        detector_path_wavelength_A=(1.540592925, 1.5419012588932806, 1.544427),
        detector_path_linear_attenuation_m_inv_by_wavelength=(0.4, 1.2, 2.4),
    )
    detector = detector.rebind_physics(
        intensity_envelope=SampleQIntensityEnvelope(u_radial_A2=0.006, u_normal_A2=0.013)
    )
    seed = 3565
    draws = 5
    cpu = detector.sample_native_pixel_mass(
        draws_per_source_state=draws,
        seed=seed,
        execution_backend="cpu",
    )
    gpu = detector.sample_native_pixel_mass(
        draws_per_source_state=draws,
        seed=seed,
        execution_backend="cuda",
    )

    assert gpu.execution_backend == "numba_cuda_forward_monte_carlo.v1"
    assert gpu.execution_device
    assert gpu.execution_worker_count is None
    assert gpu.rng_model_id == cpu.rng_model_id
    assert gpu.source_revision == cpu.source_revision
    assert gpu.rod_catalog_revision == cpu.rod_catalog_revision
    assert gpu.rods == cpu.rods
    assert gpu.attempted_root_count == cpu.attempted_root_count
    assert gpu.visible_hit_count == cpu.visible_hit_count
    np.testing.assert_array_equal(np.flatnonzero(gpu.image_A2), np.flatnonzero(cpu.image_A2))
    np.testing.assert_allclose(gpu.image_A2, cpu.image_A2, rtol=8.0e-11, atol=3.0e-24)
    np.testing.assert_allclose(
        gpu.replicate_total_mass_A2,
        cpu.replicate_total_mass_A2,
        rtol=8.0e-11,
        atol=3.0e-24,
    )
    assert gpu.total_detector_mass_A2 == pytest.approx(
        cpu.total_detector_mass_A2,
        rel=8.0e-11,
        abs=3.0e-24,
    )
    assert gpu.maximum_root_deposit_A2 == pytest.approx(
        cpu.maximum_root_deposit_A2,
        rel=8.0e-11,
        abs=3.0e-24,
    )

    progressive = detector.compile_monte_carlo_sampler(
        execution_backend="cuda",
        seed=seed,
    )
    preview = progressive.advance_preview_to(1)
    first = progressive.advance_to(1)
    assert preview.image_A2.dtype == np.float32
    np.testing.assert_allclose(
        preview.image_A2,
        first.image_A2.astype(np.float32),
        rtol=2.0 * np.finfo(np.float32).eps,
        atol=0.0,
    )
    settled = progressive.advance_to(draws)
    np.testing.assert_array_equal(
        settled.replicate_total_mass_A2[:1],
        first.replicate_total_mass_A2,
    )
    np.testing.assert_allclose(
        settled.image_A2,
        gpu.image_A2,
        rtol=8.0e-11,
        atol=3.0e-24,
    )

    from rasim_next.pipeline.source_averaged_detector import MonteCarloSamplingCancelled

    progressive.reset()
    progressive.advance_to(1)
    cancellation_polls = 0

    def cancel_after_launch() -> bool:
        nonlocal cancellation_polls
        cancellation_polls += 1
        return cancellation_polls >= 4

    with pytest.raises(MonteCarloSamplingCancelled, match="cancelled"):
        progressive.advance_preview_to(draws, cancel_requested=cancel_after_launch)
    assert progressive.draws_completed == 0
    recovered = progressive.advance_to(draws)
    np.testing.assert_allclose(recovered.image_A2, gpu.image_A2, rtol=8.0e-11, atol=3.0e-24)

    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    rebound_instrument = viewer["apply_geometry_deltas"](
        detector.instrument,
        viewer["GeometryDeltas"](
            detector_pitch_offset_deg=0.15,
            detector_column_translation_mm=0.25,
        ),
    )
    rebound = detector.rebind_geometry(
        incident=detector.incident,
        instrument=rebound_instrument,
    )
    fast_pose_sampler = detector.compile_monte_carlo_sampler(
        execution_backend="cuda",
        seed=seed,
    )
    fast_workspace = fast_pose_sampler._cuda_workspace
    projection_names = (
        "_device_detector_covectors",
        "_device_detector_normal",
        "_device_ray_origin_column_row",
        "_device_ray_origin_normal",
    )
    transport_names = (
        "_device_sample_from_local",
        "_device_ki_film",
        "_device_state_real",
        "_device_state_complex",
    )
    original_projection = tuple(id(getattr(fast_workspace, name)) for name in projection_names)
    original_transport = tuple(id(getattr(fast_workspace, name)) for name in transport_names)
    fast_pose_sampler.rebind_detector_pose(rebound_instrument)
    assert tuple(id(getattr(fast_workspace, name)) for name in projection_names) != (
        original_projection
    )
    assert (
        tuple(id(getattr(fast_workspace, name)) for name in transport_names) == original_transport
    )
    fast_pose = fast_pose_sampler.advance_to(draws)
    progressive.rebind_geometry(rebound)
    reused = progressive.advance_to(draws)
    fresh = rebound.sample_native_pixel_mass(
        draws_per_source_state=draws,
        seed=seed,
        execution_backend="cuda",
    )
    np.testing.assert_array_equal(
        np.flatnonzero(reused.image_A2),
        np.flatnonzero(fresh.image_A2),
    )
    np.testing.assert_allclose(reused.image_A2, fresh.image_A2, rtol=8.0e-11, atol=3.0e-24)
    np.testing.assert_allclose(
        reused.replicate_total_mass_A2,
        fresh.replicate_total_mass_A2,
        rtol=8.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(fast_pose.image_A2),
        np.flatnonzero(fresh.image_A2),
    )
    np.testing.assert_allclose(
        fast_pose.image_A2,
        fresh.image_A2,
        rtol=8.0e-11,
        atol=3.0e-24,
    )
    progressive.rebind_detector_pose(detector.instrument)
    mixed_rebind = progressive.advance_to(draws)
    np.testing.assert_allclose(
        mixed_rebind.image_A2,
        gpu.image_A2,
        rtol=8.0e-11,
        atol=3.0e-24,
    )

    poisoned = detector.compile_monte_carlo_sampler(
        execution_backend="cuda",
        seed=seed,
    )
    poisoned.advance_to(1)
    reset_poisoned = detector.compile_monte_carlo_sampler(
        execution_backend="cuda",
        seed=seed,
    )
    reset_poisoned.advance_to(1)
    projection_poisoned = detector.compile_monte_carlo_sampler(
        execution_backend="cuda",
        seed=seed,
    )
    projection_poisoned.advance_to(1)
    projection_workspace = projection_poisoned._cuda_workspace
    active_projection = tuple(id(getattr(projection_workspace, name)) for name in projection_names)
    workspace = poisoned._cuda_workspace
    active_geometry = tuple(
        id(getattr(workspace, name))
        for name in (
            "_device_detector_covectors",
            "_device_detector_normal",
            "_device_ray_origin_column_row",
            "_device_ray_origin_normal",
            "_device_sample_from_local",
            "_device_ki_film",
            "_device_state_real",
            "_device_state_complex",
        )
    )
    import rasim_next.pipeline._forward_detector_cuda as forward_cuda

    def fail_staged_transfer() -> None:
        raise RuntimeError("injected geometry transfer failure")

    monkeypatch.setattr(forward_cuda.cuda, "synchronize", fail_staged_transfer)
    with pytest.raises(RuntimeError, match="injected geometry transfer failure"):
        poisoned.rebind_geometry(rebound)
    assert (
        tuple(
            id(getattr(workspace, name))
            for name in (
                "_device_detector_covectors",
                "_device_detector_normal",
                "_device_ray_origin_column_row",
                "_device_ray_origin_normal",
                "_device_sample_from_local",
                "_device_ki_film",
                "_device_state_real",
                "_device_state_complex",
            )
        )
        == active_geometry
    )
    with pytest.raises(RuntimeError, match="must be discarded"):
        poisoned.advance_to(draws)
    with pytest.raises(RuntimeError, match="injected geometry transfer failure"):
        reset_poisoned.reset()
    with pytest.raises(RuntimeError, match="must be discarded"):
        reset_poisoned.advance_to(draws)
    with pytest.raises(RuntimeError, match="injected geometry transfer failure"):
        projection_poisoned.rebind_detector_pose(rebound_instrument)
    assert (
        tuple(id(getattr(projection_workspace, name)) for name in projection_names)
        == active_projection
    )
    with pytest.raises(RuntimeError, match="must be discarded"):
        projection_poisoned.advance_to(draws)


def test_cuda_default_source_blocks_match_cpu_with_shared_disorder() -> None:
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("requires a CUDA device")

    from painted_ewald import MosaicBraggSpace
    from rasim_next.ordered import Bi2X3QuintupleLayerParameters
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )
    from rasim_next.proof.tolerances import load_stage_tolerances

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    config = replace(
        config,
        instrument=replace(
            config.instrument,
            detector_path_medium_id="standard_dry_air_wavelength_table_sensitivity.v1",
            detector_path_wavelength_A=(1.540592925, 1.5419012588932806, 1.544427),
            detector_path_linear_attenuation_m_inv_by_wavelength=(
                1.194448861306956,
                1.19747301703896,
                1.2033257750023947,
            ),
        ),
    )
    inputs = build_configured_simulation_inputs(config)
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(inputs.crystal)
    candidate_strength = replace(
        inputs.strength,
        shared_disorder_epsilon=0.001,
        structure_parameters=replace(
            baseline,
            bi_fractional_z=baseline.bi_fractional_z + 0.001,
            se2_fractional_z=baseline.se2_fractional_z - 0.002,
            bi_occupancy=0.91,
            se1_occupancy=0.83,
            se2_occupancy=0.74,
            u_radial_A2=0.006,
            u_normal_A2=0.032,
        ),
    )
    inputs = replace(
        inputs,
        strength=candidate_strength,
        bragg_space=MosaicBraggSpace(inputs.bragg_space.config, candidate_strength),
    )
    detector = build_source_averaged_detector(inputs)
    assert inputs.strength.shared_disorder_epsilon == pytest.approx(0.001)
    unique_count, frequency = np.unique(
        detector.reachable_rod_count_per_source_state,
        return_counts=True,
    )
    assert dict(zip(unique_count.tolist(), frequency.tolist(), strict=True)) == {85: 1000}

    panel_rows, panel_columns = detector.instrument.detector_shape_rc
    column_px = np.asarray(
        [1109.5, 1469.5, 2206.820508075689, -1.0, float(panel_columns)],
        dtype=np.float64,
    )
    row_px = np.asarray(
        [1349.5, 1469.5, 1272.1794919243112, 0.0, float(panel_rows)],
        dtype=np.float64,
    )
    cpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    cpu_total = detector.evaluate_detector_density_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    gpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )
    gpu_total = detector.evaluate_detector_density_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )
    repeated = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )
    reblocked_detector = detector.with_maximum_state_block_count(16)
    reblocked = reblocked_detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )
    chunked = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
        cuda_coordinate_chunk_size=2,
    )

    assert reblocked_detector is not detector
    assert reblocked_detector.rods == detector.rods
    assert reblocked_detector.rod_catalog_revision == detector.rod_catalog_revision
    assert reblocked_detector.source_state_count == detector.source_state_count
    with pytest.raises(ValueError, match="positive integer"):
        detector.with_maximum_state_block_count(0)
    with pytest.raises(ValueError, match="requires the CUDA"):
        detector.evaluate_detector_density_all_roots(
            column_px,
            row_px,
            execution_backend="cpu",
            cuda_coordinate_chunk_size=2,
        )
    with pytest.raises(ValueError, match="positive integer"):
        detector.evaluate_detector_density_all_roots(
            column_px[:0],
            row_px[:0],
            execution_backend="cuda",
            cuda_coordinate_chunk_size=0,
        )
    chunked_total = detector.evaluate_detector_density_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
        cuda_coordinate_chunk_size=2,
    )

    assert np.any(cpu.per_rod_density_A2_per_px2[0, 1:] > 0.0)
    cuda_compound_relative_tolerance = 6.0e-11
    assert (
        cuda_compound_relative_tolerance
        <= load_stage_tolerances()["stacking.finite_intensity"].rtol
    )
    np.testing.assert_allclose(
        gpu.per_rod_density_A2_per_px2,
        cpu.per_rod_density_A2_per_px2,
        rtol=cuda_compound_relative_tolerance,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        gpu.density_A2_per_px2,
        cpu.density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        cpu_total.density_A2_per_px2,
        cpu.density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        gpu_total.density_A2_per_px2,
        gpu.density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(cpu_total.caustic, np.any(cpu.caustic, axis=-1))
    np.testing.assert_array_equal(gpu_total.caustic, np.any(gpu.caustic, axis=-1))
    np.testing.assert_array_equal(cpu_total.valid_source_count, cpu.valid_source_count)
    np.testing.assert_array_equal(gpu_total.valid_source_count, gpu.valid_source_count)
    np.testing.assert_array_equal(gpu.caustic, cpu.caustic)
    np.testing.assert_array_equal(gpu.valid_source_count, cpu.valid_source_count)
    np.testing.assert_array_equal(
        repeated.per_rod_density_A2_per_px2, gpu.per_rod_density_A2_per_px2
    )
    np.testing.assert_array_equal(repeated.caustic, gpu.caustic)
    np.testing.assert_array_equal(repeated.valid_source_count, gpu.valid_source_count)
    np.testing.assert_allclose(
        reblocked.per_rod_density_A2_per_px2,
        cpu.per_rod_density_A2_per_px2,
        rtol=cuda_compound_relative_tolerance,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(reblocked.caustic, gpu.caustic)
    np.testing.assert_array_equal(reblocked.valid_source_count, gpu.valid_source_count)
    np.testing.assert_array_equal(
        chunked.per_rod_density_A2_per_px2, gpu.per_rod_density_A2_per_px2
    )
    np.testing.assert_array_equal(chunked.caustic, gpu.caustic)
    np.testing.assert_array_equal(chunked.valid_source_count, gpu.valid_source_count)
    np.testing.assert_array_equal(
        chunked_total.density_A2_per_px2,
        gpu_total.density_A2_per_px2,
    )
    np.testing.assert_array_equal(chunked_total.caustic, gpu_total.caustic)
    np.testing.assert_array_equal(
        chunked_total.valid_source_count,
        gpu_total.valid_source_count,
    )
    assert gpu.execution_backend == "numba_cuda_source_averaged.v1"
    assert gpu.execution_device
    assert gpu_total.execution_backend == "numba_cuda_source_averaged.v1"
    assert gpu_total.execution_device == gpu.execution_device
    assert chunked_total.execution_backend == gpu_total.execution_backend
    assert chunked_total.execution_device == gpu_total.execution_device
    assert np.all(gpu.per_rod_density_A2_per_px2[3:] == 0.0)


def test_three_r_disorder_detector_matches_cpu_proof_and_polarization() -> None:
    from painted_ewald import MosaicBraggSpace
    from rasim_next.core.scattering import (
        THOMSON_UNPOLARIZED_UNANALYSED,
        UNITY_APPROXIMATION,
        scattering_polarization_weight,
    )
    from rasim_next.geometry import build_incident_states, detector_coordinate_to_ray
    from rasim_next.ordered import (
        Bi2X3QuintupleLayerParameters,
        SiteDisplacementProfile,
        TransverseIsotropicSiteDisplacement,
    )
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )
    from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope
    from rasim_next.stacking import Parent

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_r3_simulation.yaml",
        repository_root=root,
    )
    inputs = build_configured_simulation_inputs(config)
    assert config.structure_factor.model_id == "r3m_quintuple_finite_3r.v1"
    assert inputs.strength.parent is Parent.THREE_R
    assert inputs.strength.shared_disorder_epsilon == 0.0
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(inputs.crystal)
    candidate_strength = replace(
        inputs.strength,
        shared_disorder_epsilon=0.03,
        structure_parameters=replace(
            baseline,
            bi_occupancy=0.82,
            se1_occupancy=0.97,
            se2_occupancy=0.76,
            u_radial_A2=0.0,
            u_normal_A2=0.0,
        ),
        site_displacement_profile=SiteDisplacementProfile(
            sites=(
                TransverseIsotropicSiteDisplacement("Bi", 0.031, 0.067),
                TransverseIsotropicSiteDisplacement("Se1", 0.012, 0.024),
                TransverseIsotropicSiteDisplacement("Se2", 0.043, 0.018),
            ),
            scale=0.8,
            provenance="test site profile",
        ),
    )
    candidate_inputs = replace(
        inputs,
        strength=candidate_strength,
        bragg_space=MosaicBraggSpace(inputs.bragg_space.config, candidate_strength),
    )
    assert candidate_strength.shared_disorder_epsilon == 0.03
    column_px = np.asarray((1109.5, 1469.5, 2206.820508075689))
    row_px = np.asarray((1349.5, 1469.5, 1272.1794919243112))
    envelope = SampleQIntensityEnvelope(u_radial_A2=0.006, u_normal_A2=0.013)
    for epsilon in (0.0, 0.03):
        strength = replace(candidate_strength, shared_disorder_epsilon=epsilon)
        active_inputs = replace(
            candidate_inputs,
            strength=strength,
            bragg_space=MosaicBraggSpace(candidate_inputs.bragg_space.config, strength),
        )
        optimized = build_source_averaged_detector(active_inputs).rebind_physics(
            strength_model=strength,
            intensity_envelope=envelope,
        )
        proof = build_source_averaged_structure_detector(
            active_inputs,
            strength_model=strength,
        )
        proof = replace(proof, intensity_envelope=envelope)
        optimized_result = optimized.evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
            execution_backend="cpu",
        )
        proof_result = proof.evaluate_detector_coordinates_all_roots(column_px, row_px)
        assert np.any(proof_result.per_rod_density_A2_per_px2 > 0.0)
        np.testing.assert_allclose(
            optimized_result.per_rod_density_A2_per_px2,
            proof_result.per_rod_density_A2_per_px2,
            rtol=6.0e-11,
            atol=3.0e-24,
        )
        np.testing.assert_array_equal(optimized_result.caustic, proof_result.caustic)
        np.testing.assert_array_equal(
            optimized_result.valid_source_count,
            proof_result.valid_source_count,
        )
        if epsilon == 0.03:
            candidate_inputs = active_inputs
    inputs = candidate_inputs
    state_index = int(np.flatnonzero(inputs.incident.states.valid)[0])

    def single_state_inputs(polarization_model_id: str):
        samples = replace(
            inputs.samples,
            incident_sample_id=inputs.samples.incident_sample_id[state_index : state_index + 1],
            origin_lab_m=inputs.samples.origin_lab_m[state_index : state_index + 1],
            direction_lab=inputs.samples.direction_lab[state_index : state_index + 1],
            wavelength_A=inputs.samples.wavelength_A[state_index : state_index + 1],
            source_weight=np.ones(1, dtype=np.float64),
            polarization_state_id=(polarization_model_id,),
        )
        incident = build_incident_states(samples, inputs.material, inputs.instrument)
        return replace(inputs, samples=samples, incident=incident)

    thomson_inputs = single_state_inputs(THOMSON_UNPOLARIZED_UNANALYSED)
    unity_inputs = single_state_inputs(UNITY_APPROXIMATION)
    thomson = build_source_averaged_detector(thomson_inputs).rebind_physics(
        strength_model=candidate_strength,
        intensity_envelope=SampleQIntensityEnvelope(u_radial_A2=0.006, u_normal_A2=0.013),
    )
    unity = build_source_averaged_detector(unity_inputs).rebind_physics(
        strength_model=candidate_strength,
        intensity_envelope=SampleQIntensityEnvelope(u_radial_A2=0.006, u_normal_A2=0.013),
    )
    thomson_density = thomson.evaluate_detector_density_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    ).density_A2_per_px2
    unity_density = unity.evaluate_detector_density_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    ).density_A2_per_px2
    expected = np.asarray(
        [
            scattering_polarization_weight(
                thomson_inputs.samples.direction_lab[0],
                detector_coordinate_to_ray(
                    float(column),
                    float(row),
                    origin_lab_m=thomson_inputs.incident.states.sample_intersection_lab_m[0],
                    instrument=thomson_inputs.instrument,
                ).direction_lab,
                model_id=THOMSON_UNPOLARIZED_UNANALYSED,
            )
            for column, row in zip(column_px, row_px, strict=True)
        ]
    )
    nonzero = unity_density > np.finfo(np.float64).tiny
    assert np.any(nonzero)
    np.testing.assert_allclose(
        thomson_density[nonzero] / unity_density[nonzero],
        expected[nonzero],
        rtol=8.0e-13,
        atol=8.0e-15,
    )


def test_three_r_disorder_detector_cuda_matches_cpu() -> None:
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("requires a CUDA device")

    from painted_ewald import MosaicBraggSpace
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )
    from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope

    root = Path(__file__).resolve().parents[1]
    inputs = build_configured_simulation_inputs(
        load_simulation_config(
            root / "configs" / "bi2se3_r3_simulation.yaml",
            repository_root=root,
        )
    )
    strength = replace(inputs.strength, shared_disorder_epsilon=0.03)
    active_inputs = replace(
        inputs,
        strength=strength,
        bragg_space=MosaicBraggSpace(inputs.bragg_space.config, strength),
    )
    detector = build_source_averaged_detector(active_inputs).rebind_physics(
        strength_model=strength,
        intensity_envelope=SampleQIntensityEnvelope(
            u_radial_A2=0.006,
            u_normal_A2=0.013,
        ),
    )
    column_px = np.asarray((1109.5, 1469.5, 2206.820508075689))
    row_px = np.asarray((1349.5, 1469.5, 1272.1794919243112))
    cpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    gpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )

    assert np.any(cpu.per_rod_density_A2_per_px2 > 0.0)
    np.testing.assert_allclose(
        gpu.per_rod_density_A2_per_px2,
        cpu.per_rod_density_A2_per_px2,
        rtol=6.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        gpu.density_A2_per_px2,
        cpu.density_A2_per_px2,
        rtol=6.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(gpu.caustic, cpu.caustic)
    np.testing.assert_array_equal(gpu.valid_source_count, cpu.valid_source_count)


def test_cuda_compound_detector_tilt_matches_cpu() -> None:
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("requires a CUDA device")

    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    base_detector = config.instrument.lab_from_detector
    tilted_rotation = _intrinsic_detector_tilt(
        np.asarray(base_detector.rotation),
        0.7,
        -1.1,
    )
    tilted_detector = replace(
        base_detector,
        rotation=tuple(tuple(float(entry) for entry in row) for row in tilted_rotation),
    )
    config = replace(
        config,
        source=_single_mean_test_source(config.source),
        instrument=replace(config.instrument, lab_from_detector=tilted_detector),
    )
    detector = build_source_averaged_detector(build_configured_simulation_inputs(config))
    column_px = np.asarray([1109.5, 1469.5, -1.0])
    row_px = np.asarray([1349.5, 1469.5, 0.0])
    cpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    gpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )

    assert np.any(cpu.per_rod_density_A2_per_px2[:2] > 0.0)
    np.testing.assert_allclose(
        gpu.per_rod_density_A2_per_px2,
        cpu.per_rod_density_A2_per_px2,
        rtol=6.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        gpu.density_A2_per_px2,
        cpu.density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(gpu.caustic, cpu.caustic)
    np.testing.assert_array_equal(gpu.valid_source_count, cpu.valid_source_count)
    assert np.all(gpu.per_rod_density_A2_per_px2[2] == 0.0)

    flipped_rotation = tilted_rotation @ np.diag((1.0, -1.0, -1.0))
    back_facing_config = replace(
        config,
        instrument=replace(
            config.instrument,
            lab_from_detector=replace(
                tilted_detector,
                rotation=tuple(tuple(float(entry) for entry in row) for row in flipped_rotation),
            ),
        ),
    )
    back_facing_detector = build_source_averaged_detector(
        build_configured_simulation_inputs(back_facing_config)
    )
    reference_row = config.instrument.detector_reference_coordinate_px[1]
    back_facing_column = column_px[:1]
    back_facing_row = 2.0 * reference_row - row_px[:1]
    back_facing_cpu = back_facing_detector.evaluate_detector_coordinates_all_roots(
        back_facing_column,
        back_facing_row,
        execution_backend="cpu",
    )
    back_facing_gpu = back_facing_detector.evaluate_detector_coordinates_all_roots(
        back_facing_column,
        back_facing_row,
        execution_backend="cuda",
    )
    assert back_facing_cpu.valid_source_count.item() == 0
    assert back_facing_gpu.valid_source_count.item() == 0
    assert back_facing_cpu.density_A2_per_px2.item() == 0.0
    assert back_facing_gpu.density_A2_per_px2.item() == 0.0


def test_yaml_simulation_config_is_strict_and_plans_all_elastic_rods(
    tmp_path: Path,
) -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    default_path = root / "configs" / "bi2se3_simulation.yaml"
    config = load_simulation_config(default_path, repository_root=root)

    assert config.schema_version == "rasim-simulation-v2"
    assert config.enabled_artifact_names == (
        "reciprocal_space",
        "ewald_surface",
        "detector",
    )
    assert config.bragg.selection_model == "all_elastic_reachable.v1"
    assert config.bragg.include_detector_visible_m0
    assert config.source.sample_count == 1_000
    assert config.source.wavelength_model_id == "discrete_gaussian_lines.v1"
    assert config.source.line_probability == pytest.approx(
        (0.6587615283267457, 0.3412384716732543),
        rel=0.0,
        abs=2.0e-16,
    )
    assert config.source.position_divergence_correlation == pytest.approx(
        (-0.1477984535, 0.3873100736)
    )
    assert config.instrument.detector_path_medium_id == "vacuum_or_helium_unity.v1"
    assert config.instrument.detector_path_linear_attenuation_m_inv == 0.0
    assert config.instrument.detector_path_wavelength_A == ()
    assert config.numerics.detector_execution_backend == "cpu"
    assert config.mosaic.gaussian_sigma_deg == pytest.approx(1.0)
    assert config.mosaic.lorentzian_probability == 0.0
    assert config.structure_factor.layers == 52
    assert config.structure_factor.shared_disorder_epsilon == 0.0
    assert not config.output_directory.is_relative_to(root)
    assert (
        config.physics_revision
        == "783487a356e22c414e7ee5683c867ffb806186754662aec2dbefa61642338889"
    )

    with pytest.raises(ValueError, match="active Gaussian mosaic width"):
        build_configured_simulation_inputs(
            replace(
                config,
                mosaic=replace(
                    config.mosaic,
                    gaussian_sigma_deg=0.0,
                    lorentzian_probability=0.0,
                ),
            )
        )
    with pytest.raises(ValueError, match="active Lorentzian mosaic width"):
        build_configured_simulation_inputs(
            replace(
                config,
                mosaic=replace(
                    config.mosaic,
                    lorentzian_hwhm_deg=0.0,
                    lorentzian_probability=0.5,
                ),
            )
        )

    inputs = build_configured_simulation_inputs(config)
    assert 2.0 * np.pi / inputs.bragg_space.config.k_norm_Ainv == pytest.approx(
        config.source.mean_wavelength_A,
        rel=0.0,
        abs=2.0e-15,
    )
    expected_multiplicity = {
        0: 1,
        1: 6,
        3: 6,
        4: 6,
        7: 12,
        9: 6,
        12: 6,
        13: 12,
        16: 6,
        19: 12,
        21: 12,
    }
    assert {
        family: sum(rod.family_m == family for rod in inputs.rods)
        for family in sorted({rod.family_m for rod in inputs.rods})
    } == expected_multiplicity
    detector = build_source_averaged_detector(inputs)
    unique_count, frequency = np.unique(
        detector.reachable_rod_count_per_source_state,
        return_counts=True,
    )
    assert dict(zip(unique_count.tolist(), frequency.tolist(), strict=True)) == {85: 1000}
    assert not detector.reachable_rod_count_per_source_state.flags.writeable

    portable_default = default_path.read_text(encoding="utf-8").replace(
        "../examples/bi2se3/structures/Bi2Se3_vesta.cif",
        (root / "examples/bi2se3/structures/Bi2Se3_vesta.cif").as_posix(),
    )

    generic_yaml = tmp_path / "generic-cif.yaml"
    generic_yaml.write_text(
        portable_default.replace(
            "model_id: r3m_quintuple_finite_2h.v1",
            "model_id: cif_conventional_cell_finite_repeat.v1",
        )
        .replace("  shared_disorder_epsilon: 0.0\n", "")
        .replace("  layers: 52", "  repeats: 5"),
        encoding="utf-8",
    )
    generic_config = load_simulation_config(generic_yaml, repository_root=root)
    assert generic_config.structure_factor.shared_disorder_epsilon == 0.0
    assert generic_config.structure_factor.layers is None
    assert generic_config.structure_factor.repeats == 5

    legacy_gaussian_yaml = tmp_path / "legacy-gaussian.yaml"
    legacy_gaussian_yaml.write_text(
        portable_default.replace("  wavelength_model_id: discrete_gaussian_lines.v1\n", "")
        .replace("  wavelength_sigma_A: 0.0\n", "  wavelength_sigma_A: 0.001\n")
        .replace("  line_wavelength_A: [1.540592925, 1.544427]\n", "")
        .replace(
            "  line_probability: [0.6587615283267457, 0.3412384716732543]\n",
            "",
        )
        .replace("  common_line_sigma_A: 0.0\n", ""),
        encoding="utf-8",
    )
    legacy_gaussian = load_simulation_config(legacy_gaussian_yaml, repository_root=root)
    assert legacy_gaussian.source.wavelength_model_id == "gaussian.v1"
    legacy_samples = build_configured_simulation_inputs(legacy_gaussian).samples
    np.testing.assert_array_equal(
        legacy_samples.source_weight,
        np.full(legacy_gaussian.source.sample_count, 1.0 / legacy_gaussian.source.sample_count),
    )

    missing_nominal_air_yaml = tmp_path / "missing-nominal-air-wavelength.yaml"
    missing_nominal_air_text = portable_default.replace(
        "  detector_path_medium_id: vacuum_or_helium_unity.v1\n",
        "  detector_path_medium_id: standard_dry_air_wavelength_table_sensitivity.v1\n",
    ).replace(
        "  detector_path_linear_attenuation_m_inv: 0.0\n",
        "  detector_path_linear_attenuation_m_inv: 0.0\n"
        "  detector_path_wavelength_A: [1.540592925, 1.544427]\n"
        "  detector_path_linear_attenuation_m_inv_by_wavelength: "
        "[1.194448861306956, 1.2033257750023947]\n",
    )
    missing_nominal_air_yaml.write_text(missing_nominal_air_text, encoding="utf-8")
    with pytest.raises(ValueError, match="every source line and the nominal mean wavelength"):
        load_simulation_config(missing_nominal_air_yaml, repository_root=root)

    dry_air_yaml = tmp_path / "dry-air-wavelength-table.yaml"
    dry_air_text = missing_nominal_air_text.replace(
        "detector_path_wavelength_A: [1.540592925, 1.544427]",
        "detector_path_wavelength_A: [1.540592925, 1.5419012588932806, 1.544427]",
    ).replace(
        "[1.194448861306956, 1.2033257750023947]",
        "[1.194448861306956, 1.19747301703896, 1.2033257750023947]",
    )
    dry_air_yaml.write_text(dry_air_text, encoding="utf-8")
    dry_air = load_simulation_config(dry_air_yaml, repository_root=root)
    assert dry_air.instrument.detector_path_wavelength_A == (
        1.540592925,
        1.5419012588932806,
        1.544427,
    )
    assert dry_air.instrument.detector_path_linear_attenuation_m_inv_by_wavelength == (
        1.194448861306956,
        1.19747301703896,
        1.2033257750023947,
    )
    dry_air_inputs = build_configured_simulation_inputs(dry_air)
    build_nominal_ewald_context(dry_air_inputs)
    with pytest.raises(ValueError, match="every source line and the nominal mean wavelength"):
        replace(
            dry_air,
            instrument=replace(
                dry_air.instrument,
                detector_path_wavelength_A=(1.5419012588932806, 1.544427),
                detector_path_linear_attenuation_m_inv_by_wavelength=(
                    1.19747301703896,
                    1.2033257750023947,
                ),
            ),
        )
    with pytest.raises(ValueError, match="zero-width source"):
        replace(
            dry_air,
            source=replace(
                dry_air.source,
                wavelength_model_id="gaussian.v1",
                wavelength_sigma_A=0.001,
                line_wavelength_A=(),
                line_probability=(),
            ),
        )
    with pytest.raises(ValueError, match="zero-width source lines"):
        replace(
            dry_air,
            source=replace(dry_air.source, common_line_sigma_A=0.001),
        )
    duplicate_air_yaml = tmp_path / "duplicate-air-wavelength.yaml"
    duplicate_air_yaml.write_text(
        dry_air_text.replace(
            "detector_path_wavelength_A: [1.540592925, 1.5419012588932806, 1.544427]",
            "detector_path_wavelength_A: [1.540592925, 1.540592925, 1.544427]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_simulation_config(duplicate_air_yaml, repository_root=root)

    with pytest.raises(ValueError, match="wavelength_model_id"):
        replace(config.source, wavelength_model_id="unknown.v1")
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        replace(config.source, seed=2**64)
    with pytest.raises(ValueError, match="wavelength_sigma_A must be zero"):
        replace(config.source, wavelength_sigma_A=0.001)
    with pytest.raises(ValueError, match="discrete line centroid"):
        replace(config.source, mean_wavelength_A=config.source.mean_wavelength_A + 1.0e-4)
    with pytest.raises(ValueError, match="Gaussian source"):
        replace(config.source, wavelength_model_id="gaussian.v1")
    with pytest.raises(ValueError, match="real number"):
        replace(config.source, line_wavelength_A=(True, 1.544427))
    with pytest.raises(ValueError, match="real number"):
        replace(config.source, line_probability=(True, False))
    for change in (
        {"mean_wavelength_A": np.bool_(True)},
        {"wavelength_sigma_A": np.complex128(0.0 + 1.0j)},
        {"common_line_sigma_A": np.bool_(False)},
        {"position_divergence_correlation": (np.bool_(False), 0.0)},
        {"line_wavelength_A": (np.bool_(True), 1.544427)},
        {"line_probability": (np.complex128(0.5 + 0.1j), 0.5)},
    ):
        with pytest.raises(ValueError, match="real"):
            replace(config.source, **change)
    gaussian_with_mutable_empty_lines = replace(
        config.source,
        wavelength_model_id="gaussian.v1",
        wavelength_sigma_A=0.0,
        line_wavelength_A=[],
        line_probability=[],
    )
    assert gaussian_with_mutable_empty_lines.line_wavelength_A == ()
    assert gaussian_with_mutable_empty_lines.line_probability == ()
    with pytest.raises(ValueError, match="too small"):
        replace(
            config.source,
            line_probability=(1.0, np.nextafter(0.0, 1.0)),
        )

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: rasim-simulation-v2\nschema_version: duplicate\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate key.*schema_version"):
        load_simulation_config(duplicate, repository_root=root)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        default_path.read_text(encoding="utf-8") + "\nunknown_parameter: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unknown key.*unknown_parameter"):
        load_simulation_config(unknown, repository_root=root)

    duplicate_filename = tmp_path / "duplicate-filename.yaml"
    duplicate_filename.write_text(
        portable_default.replace(
            "filename: ewald-surface.png",
            "filename: reciprocal-space.png",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output filenames must be unique"):
        load_simulation_config(duplicate_filename, repository_root=root)

    unsupported_backend = tmp_path / "unsupported-backend.yaml"
    unsupported_backend.write_text(
        portable_default.replace(
            "detector_execution_backend: cpu",
            "detector_execution_backend: automatic",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"detector_execution_backend.*cpu or cuda"):
        load_simulation_config(unsupported_backend, repository_root=root)

    missing_backend = tmp_path / "missing-backend.yaml"
    missing_backend.write_text(
        portable_default.replace("  detector_execution_backend: cpu\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"numerics.*missing.*detector_execution_backend"):
        load_simulation_config(missing_backend, repository_root=root)

    negative_source_sigma = tmp_path / "negative-source-sigma.yaml"
    negative_source_sigma.write_text(
        portable_default.replace(
            "spatial_sigma_m: [1.4312946262159279e-4, 1.2325011424826383e-4]",
            "spatial_sigma_m: [-1.0, 1.2325011424826383e-4]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"source\.spatial_sigma_m.*nonnegative"):
        load_simulation_config(negative_source_sigma, repository_root=root)

    excessive_reciprocal_tilt = tmp_path / "excessive-reciprocal-tilt.yaml"
    excessive_reciprocal_tilt.write_text(
        portable_default.replace(
            "reciprocal_alpha_max_deg: 5.0",
            "reciprocal_alpha_max_deg: 181.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"reciprocal_alpha_max_deg.*180"):
        load_simulation_config(excessive_reciprocal_tilt, repository_root=root)

    all_disabled = tmp_path / "all-disabled.yaml"
    all_disabled.write_text(
        portable_default.replace("enabled: true", "enabled: false"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one output"):
        load_simulation_config(all_disabled, repository_root=root)


def test_nominal_ewald_gap_is_attached_only_to_visible_m0_support() -> None:
    from painted_ewald import MosaicBraggSpace
    from rasim_next.pipeline.configured_simulation import (
        build_nominal_ewald_context,
        evaluate_nominal_ewald_surface,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    config = replace(
        config,
        source=_single_mean_test_source(config.source),
        instrument=replace(
            config.instrument,
            detector_reference_coordinate_px=(-1000.0, 1500.0),
        ),
    )
    inputs = build_configured_simulation_inputs(config)
    stale_bragg = MosaicBraggSpace(
        replace(
            inputs.bragg_space.config,
            k_norm_Ainv=inputs.bragg_space.config.k_norm_Ainv * 1.001,
        ),
        inputs.strength,
    )
    with pytest.raises(ValueError, match="wavelength does not match"):
        build_nominal_ewald_context(replace(inputs, bragg_space=stale_bragg))
    display = evaluate_nominal_ewald_surface(
        build_nominal_ewald_context(inputs),
        alpha_count=16,
        beta_count=72,
        alpha_max_deg=5.0,
    )

    assert np.any(display.family_m != 0)
    assert not np.any(display.family_m == 0)
    assert display.detector_visible_m0_q_gap_Ainv is None


@pytest.mark.parametrize(
    ("sample_angle_deg", "integer_L", "expected_column_px", "expected_row_px"),
    (
        (
            5.0,
            2,
            np.asarray((1103.45904245, 1802.78095755)),
            1512.7160794,
        ),
        (
            2.0,
            1,
            np.asarray((1105.66640479, 1800.57359521)),
            1553.2784267,
        ),
    ),
)
def test_nominal_integer_l_markers_are_exact_visible_roundtrips(
    sample_angle_deg: float,
    integer_L: int,
    expected_column_px: np.ndarray,
    expected_row_px: float,
) -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_nominal_ewald_context,
        evaluate_nominal_integer_l_markers,
    )

    inputs = _configured_inputs(sample_count=1, sample_angle_deg=sample_angle_deg)
    context = build_nominal_ewald_context(inputs)
    markers = evaluate_nominal_integer_l_markers(context)
    assert markers.definition_id == "peak_mosaic_alpha0_integer_L_center.v3"
    assert markers.source_state_policy == "mean_source_state.v1"
    assert markers.reference_wavelength_A == pytest.approx(1.5419012588932806, abs=2.0e-15)
    if sample_angle_deg == 5.0:
        assert not np.any((markers.family_m == 1) & (markers.integer_L == 1))
        np.testing.assert_array_equal(np.unique(markers.family_m), np.asarray((1, 3, 4)))
        np.testing.assert_array_equal(np.unique(markers.branch), np.asarray((2,)))
        np.testing.assert_array_equal(np.unique(markers.root_sign), np.asarray((-1, 1)))
        assert markers.column_px.size == 84
        assert (
            len(
                set(
                    zip(
                        markers.family_m,
                        markers.integer_L,
                        markers.branch,
                        markers.root_sign,
                        strict=True,
                    )
                )
            )
            == 84
        )
        assert np.all((markers.column_px >= -0.5) & (markers.column_px < 2999.5))
        assert np.all((markers.row_px >= -0.5) & (markers.row_px < 2999.5))
        assert float(np.max(markers.ewald_residual_Ainv)) < 2.0e-13
    selected = np.flatnonzero(
        (markers.family_m == 1) & (markers.integer_L == integer_L) & (markers.branch == 2)
    )

    assert selected.size == 2
    order = selected[np.argsort(markers.column_px[selected])]
    np.testing.assert_allclose(
        markers.column_px[order],
        expected_column_px,
        rtol=0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        markers.row_px[order],
        expected_row_px,
        rtol=0.0,
        atol=2.0e-6,
    )
    assert [markers.labels[index] for index in order] == [
        f"m=1, L={integer_L}, b=2, s={int(markers.root_sign[order[0]]):+d}",
        f"m=1, L={integer_L}, b=2, s={int(markers.root_sign[order[1]]):+d}",
    ]

    space = context.geometry.coating.bragg_space
    b3_norm_Ainv = float(np.linalg.norm(space.config.reciprocal_basis_Ainv[:, 2]))
    mean_axis_crystal = space.config.reciprocal_basis_Ainv[:, 2] / b3_norm_Ainv
    mean_axis_sample = space.config.crystal_to_sample @ mean_axis_crystal
    ki_sample_Ainv = context.geometry.coating.ki_sample_Ainv
    k_norm_Ainv = float(np.linalg.norm(ki_sample_Ainv))
    rods = {(rod.h, rod.k): rod for rod in space.config.rods}
    if sample_angle_deg == 5.0:
        l1_rod = next(rod for rod in space.config.rods if rod.family_m == 1)
        q_parallel = (
            l1_rod.h * space.config.reciprocal_basis_Ainv[:, 0]
            + l1_rod.k * space.config.reciprocal_basis_Ainv[:, 1]
        )
        q_axis = float(q_parallel @ mean_axis_crystal) * mean_axis_crystal
        q_perpendicular = q_parallel - q_axis
        q_quadrature = np.cross(mean_axis_crystal, q_perpendicular)
        cosine_coefficient = 2.0 * float(
            ki_sample_Ainv @ (space.config.crystal_to_sample @ q_perpendicular)
        )
        sine_coefficient = 2.0 * float(
            ki_sample_Ainv @ (space.config.crystal_to_sample @ q_quadrature)
        )
        amplitude = math.hypot(cosine_coefficient, sine_coefficient)
        q_base = q_axis + b3_norm_Ainv * mean_axis_crystal
        q_unrotated = q_parallel + b3_norm_Ainv * mean_axis_crystal
        constant = float(
            q_unrotated @ q_unrotated
            + 2.0 * ki_sample_Ainv @ (space.config.crystal_to_sample @ q_base)
        )
        phase = math.atan2(sine_coefficient, cosine_coefficient)
        delta = math.acos(-constant / amplitude)
        l1_beta = np.mod(np.asarray((phase - delta, phase + delta)), 2.0 * np.pi)
        backward_l1 = context.map_latent_geometry(
            rod=l1_rod,
            branch=1,
            alpha_rad=np.zeros(2),
            beta_rad=l1_beta,
        )
        np.testing.assert_allclose(
            backward_l1.ewald_geometry.L,
            1.0,
            rtol=0.0,
            atol=2.0e-13,
        )
        np.testing.assert_array_equal(
            backward_l1.exit_status,
            np.asarray(("BACKWARD", "BACKWARD")),
        )
        np.testing.assert_allclose(
            backward_l1.ewald_geometry.kf_sample_Ainv @ mean_axis_sample,
            -0.134836295949742,
            rtol=0.0,
            atol=5.0e-11,
        )
        all_pulled_back = context.geometry.evaluate_detector_geometry(
            markers.column_px,
            markers.row_px,
            include_surface_jacobian=False,
        )
        assert np.all(all_pulled_back.valid)
        np.testing.assert_allclose(
            all_pulled_back.q_sample_Ainv,
            markers.q_sample_Ainv,
            rtol=0.0,
            atol=3.0e-12,
        )
        assert np.all((ki_sample_Ainv + markers.q_sample_Ainv) @ mean_axis_sample > 0.0)
        for family, hk_group in zip(
            markers.family_m,
            markers.contributing_rod_hk,
            strict=True,
        ):
            assert len(hk_group) == 6
            assert all(rods[hk].family_m == family for hk in hk_group)
    expected_hk = {(rod.h, rod.k) for rod in space.config.rods if rod.family_m == 1}
    for index in order:
        assert set(markers.contributing_rod_hk[index]) == expected_hk
        assert len(markers.contributing_beta_rad[index]) == len(expected_hk) == 6
        assert markers.family_strength_weight_A2[index] == pytest.approx(
            sum(markers.per_rod_strength_weight_A2[index]),
            rel=2.0e-15,
            abs=0.0,
        )
        for hk, beta_rad, strength_weight_A2 in zip(
            markers.contributing_rod_hk[index],
            markers.contributing_beta_rad[index],
            markers.per_rod_strength_weight_A2[index],
            strict=True,
        ):
            rod = rods[hk]
            assert rod.family_m == 1
            q_parallel = (
                rod.h * space.config.reciprocal_basis_Ainv[:, 0]
                + rod.k * space.config.reciprocal_basis_Ainv[:, 1]
            )
            q_perpendicular = q_parallel - float(q_parallel @ mean_axis_crystal) * mean_axis_crystal
            q_quadrature = np.cross(mean_axis_crystal, q_perpendicular)
            phase = math.atan2(
                float(ki_sample_Ainv @ (space.config.crystal_to_sample @ q_quadrature)),
                float(ki_sample_Ainv @ (space.config.crystal_to_sample @ q_perpendicular)),
            )
            assert int(np.sign(np.sin(beta_rad - phase))) == int(markers.root_sign[index])
            q_sample_Ainv = space.map_latent(
                rod=rod,
                alpha_rad=0.0,
                beta_rad=beta_rad,
                u_Ainv=integer_L * b3_norm_Ainv,
            )
            np.testing.assert_allclose(
                q_sample_Ainv,
                markers.q_sample_Ainv[index],
                rtol=0.0,
                atol=3.0e-12,
            )
            if index == order[0]:
                np.testing.assert_array_max_ulp(
                    strength_weight_A2,
                    rod.population
                    * space.strength_model.evaluate(
                        rod=rod,
                        L=float(integer_L),
                        k_norm_Ainv=space.config.k_norm_Ainv,
                    ),
                    maxulp=32,
                )
        kf_film_sample_Ainv = ki_sample_Ainv + markers.q_sample_Ainv[index]
        assert float(kf_film_sample_Ainv @ mean_axis_sample) > 0.0
        assert abs(float(np.linalg.norm(kf_film_sample_Ainv) - k_norm_Ainv)) < 2.0e-13
        pulled_back = context.geometry.evaluate_detector_geometry(
            markers.column_px[index],
            markers.row_px[index],
            include_surface_jacobian=False,
        )
        assert bool(pulled_back.valid)
        np.testing.assert_allclose(
            pulled_back.q_sample_Ainv,
            markers.q_sample_Ainv[index],
            rtol=0.0,
            atol=3.0e-12,
        )


def test_integer_l_marker_sites_do_not_depend_on_render_sampling_or_strength() -> None:
    from painted_ewald import MosaicBraggSpace
    from rasim_next.pipeline.configured_simulation import (
        build_nominal_ewald_context,
        evaluate_nominal_integer_l_markers,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    one_state = replace(config, source=_single_mean_test_source(config.source))
    changed_numerics = replace(
        config.numerics,
        detector_macrobin_size_px=100,
        detector_gauss_order=1,
        reciprocal_alpha_count=1,
        reciprocal_beta_count=3,
        reciprocal_u_count=7,
        ewald_alpha_count=3,
        ewald_beta_count=5,
    )
    base_inputs = build_configured_simulation_inputs(one_state)
    base = evaluate_nominal_integer_l_markers(build_nominal_ewald_context(base_inputs))
    changed = evaluate_nominal_integer_l_markers(
        build_nominal_ewald_context(
            build_configured_simulation_inputs(replace(one_state, numerics=changed_numerics))
        )
    )
    zero_parameters = replace(
        base_inputs.strength.structure_parameters,
        bi_occupancy=0.0,
        se1_occupancy=0.0,
        se2_occupancy=0.0,
    )
    zero_strength = replace(base_inputs.strength, structure_parameters=zero_parameters)
    zero_strength_inputs = replace(
        base_inputs,
        strength=zero_strength,
        bragg_space=MosaicBraggSpace(base_inputs.bragg_space.config, zero_strength),
    )
    zero_strength_markers = evaluate_nominal_integer_l_markers(
        build_nominal_ewald_context(zero_strength_inputs)
    )

    for candidate in (changed, zero_strength_markers):
        for name in (
            "family_m",
            "integer_L",
            "branch",
            "root_sign",
            "column_px",
            "row_px",
            "q_sample_Ainv",
        ):
            np.testing.assert_array_equal(getattr(candidate, name), getattr(base, name))
        assert candidate.contributing_rod_hk == base.contributing_rod_hk
        assert candidate.contributing_beta_rad == base.contributing_beta_rad
    assert np.all(base.family_strength_weight_A2 > 0.0)
    assert np.all(zero_strength_markers.family_strength_weight_A2 == 0.0)
    assert all(
        strength == 0.0
        for group in zero_strength_markers.per_rod_strength_weight_A2
        for strength in group
    )


def test_yaml_detector_two_axis_tilt_folds_into_canonical_pose(tmp_path: Path) -> None:
    from rasim_next.core.validity import ValidityCode
    from rasim_next.geometry import detector_coordinate_to_ray, project_detector_ray
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    default_path = root / "configs" / "bi2se3_simulation.yaml"
    portable_default = default_path.read_text(encoding="utf-8").replace(
        "../examples/bi2se3/structures/Bi2Se3_vesta.cif",
        (root / "examples/bi2se3/structures/Bi2Se3_vesta.cif").as_posix(),
    )
    zero_tilt = "  detector_tilt:\n    about_column_axis_deg: 0.0\n    about_row_axis_deg: 0.0\n"
    base_path = tmp_path / "base.yaml"
    zero_path = tmp_path / "zero.yaml"
    tilted_path = tmp_path / "tilted.yaml"
    base_path.write_text(portable_default.replace(zero_tilt, "", 1), encoding="utf-8")
    zero_path.write_text(portable_default, encoding="utf-8")
    tilted_path.write_text(
        portable_default.replace(
            zero_tilt,
            "  detector_tilt:\n    about_column_axis_deg: 7.0\n    about_row_axis_deg: -11.0\n",
            1,
        ),
        encoding="utf-8",
    )
    base = load_simulation_config(base_path, repository_root=root)
    zero = load_simulation_config(zero_path, repository_root=root)
    tilted = load_simulation_config(tilted_path, repository_root=root)

    assert zero.instrument.lab_from_detector == base.instrument.lab_from_detector
    assert zero.physics_revision == base.physics_revision
    base_rotation = np.asarray(base.instrument.lab_from_detector.rotation)
    expected_rotation = _intrinsic_detector_tilt(base_rotation, 7.0, -11.0)
    np.testing.assert_allclose(
        tilted.instrument.lab_from_detector.rotation,
        expected_rotation,
        rtol=0.0,
        atol=3.0e-16,
    )
    assert (
        tilted.instrument.lab_from_detector.translation_m
        == base.instrument.lab_from_detector.translation_m
    )
    assert tilted.physics_revision != base.physics_revision

    one_sample = replace(tilted, source=_single_mean_test_source(tilted.source))
    instrument = build_configured_simulation_inputs(one_sample).instrument
    np.testing.assert_allclose(
        instrument.lab_from_detector.rotation,
        expected_rotation,
        rtol=0.0,
        atol=3.0e-16,
    )
    column_px, row_px = 1200.25, 1500.75
    origin_lab_m = np.zeros(3)
    ray = detector_coordinate_to_ray(
        column_px,
        row_px,
        origin_lab_m=origin_lab_m,
        instrument=instrument,
    )
    assert ray.status is ValidityCode.VALID
    projection = project_detector_ray(origin_lab_m, ray.direction_lab, instrument)
    assert projection.status is ValidityCode.VALID
    assert projection.column_px == pytest.approx(column_px, rel=0.0, abs=2.0e-10)
    assert projection.row_px == pytest.approx(row_px, rel=0.0, abs=2.0e-10)
    pixel_area_m2 = instrument.detector_column_pitch_m * instrument.detector_row_pitch_m
    expected_solid_angle_sr = (
        pixel_area_m2
        * abs(float(expected_rotation[:, 2] @ ray.direction_lab))
        / ray.ray_distance_m**2
    )
    assert projection.pixel_solid_angle_sr == pytest.approx(
        expected_solid_angle_sr,
        rel=2.0e-15,
        abs=0.0,
    )


def test_detector_macrobin_preview_integrates_total_continuous_density_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import rasim_next.pipeline.configured_simulation as configured_simulation_module
    from rasim_next.pipeline.configured_simulation import (
        CONFIGURED_RESULT_SCHEMA_VERSION,
        integrate_detector_macrobins,
    )

    evaluated_shapes: list[tuple[int, ...]] = []
    monkeypatch.setattr(configured_simulation_module, "_MAXIMUM_MACROBIN_COORDINATES_PER_CALL", 4)

    class ConstantDetector:
        instrument = SimpleNamespace(detector_shape_rc=(4, 6))

        @staticmethod
        def evaluate_detector_density_all_roots(
            column_px: np.ndarray,
            row_px: np.ndarray,
        ) -> object:
            shape = np.broadcast_shapes(column_px.shape, row_px.shape)
            evaluated_shapes.append(shape)
            return SimpleNamespace(
                density_A2_per_px2=np.full(shape, 5.0),
                caustic=np.zeros(shape, dtype=np.bool_),
                valid_source_count=np.full(shape, 7, dtype=np.int64),
                execution_backend="numba_cpu_source_averaged.v1",
                execution_device=None,
            )

    result = integrate_detector_macrobins(
        ConstantDetector(),
        bin_size_px=2,
        gauss_order=2,
    )
    np.testing.assert_allclose(result.image_A2, 20.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(result.valid_source_count_min, 7)
    assert not hasattr(result, "per_rod_image_A2")
    assert result.coordinate_evaluation_count == 24
    assert evaluated_shapes == [(1, 3)] * 8
    assert result.measure_id == "raw_detector_macrobin_fixed_quadrature_estimate_A2.v1"
    assert CONFIGURED_RESULT_SCHEMA_VERSION == "rasim-configured-result-v2"
    with pytest.raises(ValueError, match="identify exactly the CUDA backend"):
        replace(result, execution_device="unexpected device")
    with pytest.raises(ValueError, match="identify exactly the CUDA backend"):
        replace(result, execution_backend="numba_cuda_source_averaged.v1")


def test_continuous_detector_cli_exposes_mosaic_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    namespace = runpy.run_path(str(root / "scripts" / "generate_bi2se3_continuous_detector.py"))

    with pytest.raises(SystemExit) as exit_info:
        namespace["main"](["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--gaussian-sigma-deg" in help_text
    assert "--lorentzian-hwhm-deg" in help_text
    assert "--eta" in help_text
    assert "--layers" in help_text
    assert "--stacking-epsilon" in help_text

    with pytest.raises(SystemExit) as invalid_exit:
        namespace["main"](
            [
                "--numeric-only",
                "--gaussian-sigma-deg",
                "0",
                "--lorentzian-hwhm-deg",
                "0",
                "--eta",
                "0",
            ]
        )

    assert invalid_exit.value.code == 2
    assert "Gaussian sigma must be positive" in capsys.readouterr().err


def test_continuous_detector_reference_requires_complete_identity_and_arrays(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts" / "generate_bi2se3_continuous_detector.py"))
    identity = {
        "reference_comparison_model_id": "one_ki_selected_rods_detector_mass.v2",
        "configured_physics_revision": "configured-physics-revision",
        "result_measure_id": "raw_detector_pixel_mass_A2.v1",
        "wavelength_A": 1.5419012588932806,
        "cif_path": "examples/bi2se3/structures/Bi2Se3_vesta.cif",
        "strength_parent": "3R",
        "strength_structure_model_revision": "structure-revision",
        "strength_layer_count": 50,
        "strength_normalization": "finite_per_layer",
        "strength_shared_disorder_epsilon": 0.0,
        "detector_shape_rc": [2, 3],
        "m1_rod_keys": [[-1, 0], [0, 1]],
        "branch": 2,
    }
    reference_image = np.ones((2, 3))
    reference_per_rod = np.asarray((2.5, 3.5))
    identity.update(
        {
            "image_sha256": hashlib.sha256(
                memoryview(np.ascontiguousarray(reference_image)).cast("B")
            ).hexdigest(),
            "nonzero_pixel_count": 6,
            "per_rod_detector_mass_A2": reference_per_rod.tolist(),
            "total_detector_mass_A2": 6.0,
        }
    )
    manifest_json = np.frombuffer(
        json.dumps(identity, sort_keys=True).encode("utf-8"),
        dtype=np.uint8,
    )
    reference = tmp_path / "reference.npz"
    np.savez(
        reference,
        manifest_json=manifest_json,
        image_A2=reference_image,
        per_rod_detector_mass_A2=reference_per_rod,
    )

    namespace["_validate_reference_strength"](reference, identity)
    image, per_rod, image_sha256, diagnostic_sha256 = namespace["_load_reference_arrays"](
        reference,
        image_shape=(2, 3),
        per_rod_shape=(2,),
        expected_identity=identity,
    )
    np.testing.assert_array_equal(image, 1.0)
    np.testing.assert_array_equal(per_rod, reference_per_rod)
    assert image_sha256 == identity["image_sha256"]
    assert diagnostic_sha256 == hashlib.sha256(reference.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="wavelength_A"):
        namespace["_validate_reference_strength"](
            reference,
            {**identity, "wavelength_A": 1.540592925},
        )
    with pytest.raises(ValueError, match="configured_physics_revision"):
        namespace["_validate_reference_strength"](
            reference,
            {**identity, "configured_physics_revision": "different-physics"},
        )

    malformed = tmp_path / "malformed.npz"
    np.savez(
        malformed,
        manifest_json=manifest_json,
        image_A2=np.ones((2, 3)),
        per_rod_detector_mass_A2=np.asarray(1.0),
    )
    with pytest.raises(ValueError, match="per-rod detector mass shape"):
        namespace["_load_reference_arrays"](
            malformed,
            image_shape=(2, 3),
            per_rod_shape=(2,),
            expected_identity=identity,
        )
    tampered_image = tmp_path / "tampered-image.npz"
    np.savez(
        tampered_image,
        manifest_json=manifest_json,
        image_A2=np.full((2, 3), 2.0),
        per_rod_detector_mass_A2=reference_per_rod,
    )
    with pytest.raises(ValueError, match="declared SHA-256"):
        namespace["_load_reference_arrays"](
            tampered_image,
            image_shape=(2, 3),
            per_rod_shape=(2,),
            expected_identity=identity,
        )
    tampered_per_rod = tmp_path / "tampered-per-rod.npz"
    np.savez(
        tampered_per_rod,
        manifest_json=manifest_json,
        image_A2=reference_image,
        per_rod_detector_mass_A2=np.asarray((3.0, 3.0)),
    )
    with pytest.raises(ValueError, match="do not match their manifest"):
        namespace["_load_reference_arrays"](
            tampered_per_rod,
            image_shape=(2, 3),
            per_rod_shape=(2,),
            expected_identity=identity,
        )
    wrong_dtype = tmp_path / "wrong-dtype.npz"
    np.savez(
        wrong_dtype,
        manifest_json=manifest_json,
        image_A2=reference_image.astype(np.int64),
        per_rod_detector_mass_A2=reference_per_rod,
    )
    with pytest.raises(ValueError, match="float64 dtype"):
        namespace["_load_reference_arrays"](
            wrong_dtype,
            image_shape=(2, 3),
            per_rod_shape=(2,),
            expected_identity=identity,
        )


def _instrument(
    *,
    shape_rc: tuple[int, int] = (3, 4),
    reference_cr: tuple[float, float] = (1.5, 1.0),
    detector_rotation: np.ndarray | None = None,
) -> object:
    identity = np.eye(3)
    zero = np.zeros(3)
    rotation = identity if detector_rotation is None else detector_rotation
    configuration = InstrumentConfiguration(
        axis_rotations=(),
        lab_from_goniometer_zero=RigidTransform(identity, zero, FrameId.GONIOMETER, FrameId.LAB),
        goniometer_from_sample=RigidTransform(identity, zero, FrameId.SAMPLE, FrameId.GONIOMETER),
        sample_from_crystal=RigidTransform(identity, zero, FrameId.CRYSTAL, FrameId.SAMPLE),
        lab_from_detector=RigidTransform(
            rotation,
            [1.1e-3, -0.7e-3, 0.82],
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=shape_rc,
        detector_row_pitch_m=3.1e-4,
        detector_column_pitch_m=1.7e-4,
        detector_reference_coordinate_px=reference_cr,
        sample_support_model_id="finite_rectangle.v1",
        sample_width_m=4.0e-4,
        sample_length_m=6.0e-4,
        film_thickness_A=500.0,
    )
    return compile_instrument(configuration)


def test_configured_initial_beam_maps_exactly_to_the_untilted_detector() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    gaussian_source = replace(
        config.source,
        wavelength_model_id="gaussian.v1",
        line_wavelength_A=(),
        line_probability=(),
        common_line_sigma_A=0.0,
        wavelength_sigma_A=0.0,
        position_divergence_correlation=(0.0, 0.0),
        sample_count=6,
    )
    inputs = build_configured_simulation_inputs(
        replace(
            config,
            source=gaussian_source,
            instrument=replace(
                config.instrument,
                detector_path_medium_id="vacuum_or_helium_unity.v1",
                detector_path_linear_attenuation_m_inv=0.0,
            ),
        )
    )
    samples = inputs.samples
    instrument = inputs.instrument
    sample_angle_rad = math.radians(5.0)
    cosine = math.cos(sample_angle_rad)
    sine = math.sin(sample_angle_rad)
    expected_lab_from_sample_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    np.testing.assert_allclose(
        instrument.lab_from_sample.rotation,
        expected_lab_from_sample_rotation,
        rtol=0.0,
        atol=2.0e-16,
    )
    incident = inputs.incident.states
    sample_normal_lab = expected_lab_from_sample_rotation[:, 2]
    source_to_sample_m = -(
        (samples.origin_lab_m - instrument.lab_from_sample.translation_m) @ sample_normal_lab
    ) / (samples.direction_lab @ sample_normal_lab)
    expected_sample_intersection_lab_m = (
        samples.origin_lab_m + source_to_sample_m[:, None] * samples.direction_lab
    )
    np.testing.assert_allclose(
        incident.sample_intersection_lab_m,
        expected_sample_intersection_lab_m,
        rtol=0.0,
        atol=2.0e-16,
    )
    k_air_norm_Ainv = 2.0 * np.pi / incident.wavelength_A
    direction_lab = instrument.lab_from_sample.apply_vector(incident.k_air_sample_Ainv)
    direction_lab /= k_air_norm_Ainv[:, None]
    np.testing.assert_allclose(direction_lab, samples.direction_lab, rtol=0.0, atol=5.0e-16)
    projection = project_detector_rays(
        incident.sample_intersection_lab_m,
        direction_lab,
        instrument,
    )

    assert samples.incident_sample_id.size == 6
    np.testing.assert_array_equal(samples.source_weight, np.full(6, 1.0 / 6.0))
    assert np.all(incident.valid)
    assert np.all(projection.valid)
    detector_normal_lab = instrument.lab_from_detector.apply_vector([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(detector_normal_lab, np.array([0.0, 1.0, 0.0]))

    weights = incident.source_weight
    center_column, center_row = instrument.detector_reference_coordinate_px
    centroid = np.asarray(
        (
            np.sum(weights * projection.column_px),
            np.sum(weights * projection.row_px),
        )
    )
    np.testing.assert_allclose(
        centroid,
        np.asarray((center_column, center_row)),
        rtol=0.0,
        atol=3.0e-12,
    )
    detector_plane_y_m = 0.075
    plane_distance_m = (detector_plane_y_m - samples.origin_lab_m[:, 1]) / samples.direction_lab[
        :, 1
    ]
    expected_point_lab_m = samples.origin_lab_m + plane_distance_m[:, None] * samples.direction_lab
    np.testing.assert_allclose(
        projection.column_px,
        center_column + expected_point_lab_m[:, 0] / 1.0e-4,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        projection.row_px,
        center_row - expected_point_lab_m[:, 2] / 1.0e-4,
        rtol=0.0,
        atol=2.0e-12,
    )
    wavelength_mean_A = float(np.sum(weights * incident.wavelength_A))
    assert wavelength_mean_A == pytest.approx(config.source.mean_wavelength_A, abs=2.0e-15)
    assert np.sum(weights[projection.valid]) == pytest.approx(1.0, abs=2.0e-15)


def _frame(origin_lab_m: np.ndarray | list[float] | None = None) -> AngleFrame:
    return AngleFrame(
        origin_lab_m=np.zeros(3) if origin_lab_m is None else origin_lab_m,
        row_down_lab=np.array([0.0, 1.0, 0.0]),
        column_right_lab=np.array([1.0, 0.0, 0.0]),
        direct_beam_lab=np.array([0.0, 0.0, 1.0]),
        revision="integration-angle-frame.v1",
    )


def test_continuous_angle_function_preserves_detector_density_through_s_over_n() -> None:
    inputs = _configured_inputs(sample_count=1)
    detector_function = ContinuousDetectorGeometryModel(inputs).bind(GeometryCorrections.zero())
    frame = AngleFrame(
        origin_lab_m=np.zeros(3),
        row_down_lab=np.array([0.0, 0.0, -1.0]),
        column_right_lab=np.array([1.0, 0.0, 0.0]),
        direct_beam_lab=np.array([0.0, 1.0, 0.0]),
        revision="configured-nominal-angle-frame.v1",
    )
    detector_column = np.array([811.25, 1510.0, 2237.75])
    detector_row = np.array([412.5, 1596.422, 2461.125])
    angles = detector_coordinates_to_angles(
        detector_column,
        detector_row,
        instrument=detector_function.instrument,
        angle_frame=frame,
    )
    assert np.all(angles.valid)
    direct = detector_function(detector_column, detector_row)

    angle_function = ContinuousNormalizedAngleFunction(detector_function, frame)
    evaluated = angle_function(
        np.concatenate((angles.two_theta_rad, [0.0, np.pi / 2.0])),
        np.concatenate((angles.phi_rad, [0.37, 0.0])),
    )
    expected_measure = angles_to_detector_coordinate_area_measure(
        angles.two_theta_rad,
        angles.phi_rad,
        instrument=detector_function.instrument,
        angle_frame=frame,
    )
    expected_normalization = expected_measure.detector_area_jacobian_px2_per_rad2

    np.testing.assert_allclose(
        evaluated.normalization_density_px2_per_rad2[:3],
        expected_normalization,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluated.signal_density_A2_per_rad2[:3],
        direct.density_A2_per_px2 * expected_normalization,
        rtol=3.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluated.intensity_A2_per_px2[:3],
        direct.density_A2_per_px2,
        rtol=3.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluated.signal_density_A2_per_rad2[:3] / evaluated.normalization_density_px2_per_rad2[:3],
        evaluated.intensity_A2_per_px2[:3],
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(evaluated.valid, [True, True, True, False, False])
    np.testing.assert_array_equal(
        evaluated.normalization_density_px2_per_rad2[3:],
        np.zeros(2),
    )
    np.testing.assert_array_equal(evaluated.signal_density_A2_per_rad2[3:], np.zeros(2))
    np.testing.assert_array_equal(evaluated.intensity_A2_per_px2[3:], np.zeros(2))
    assert evaluated.signal_measure_id == "raw_detector_angle_signal_density_A2_per_rad2.v1"
    assert evaluated.normalization_measure_id == "detector_area_density_px2_per_rad2.v1"
    assert evaluated.intensity_measure_id == "raw_detector_area_normalized_intensity_A2_per_px2.v1"
    assert not evaluated.intensity_A2_per_px2.flags.writeable


class _AnalyticDetectorDensity:
    measure_id = "raw_detector_coordinate_density_A2_per_px2.v1"

    def __init__(self, instrument: object) -> None:
        self.instrument = instrument

    def evaluate_detector_coordinates(
        self,
        column_px: object,
        row_px: object,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        column, row = np.broadcast_arrays(
            np.asarray(column_px, dtype=np.float64),
            np.asarray(row_px, dtype=np.float64),
        )
        center_column = self.instrument.detector_reference_coordinate_px[0]
        density = 4.0 + 0.2 * (column - center_column) ** 2
        return SourceAveragedDetectorCoordinateIntensity(
            column_px=column,
            row_px=row,
            rods=(Rod(1, 0),),
            rod_catalog_revision="analytic-angle-bin-rods.v1",
            branch=2,
            per_rod_density_A2_per_px2=density[..., None],
            density_A2_per_px2=density,
            caustic=np.zeros((*density.shape, 1), dtype=np.bool_),
            valid_source_count=np.ones(density.shape, dtype=np.int64),
            source_state_count=1,
            source_revision="analytic-angle-bin.v1",
        )


def _integrate_square_panel_in_angle_coordinates(
    angle_function: ContinuousNormalizedAngleFunction,
    *,
    order: int,
) -> tuple[float, float, float]:
    node, weight = np.polynomial.legendre.leggauss(order)
    signal = 0.0
    normalization = 0.0
    ratio_integral = 0.0
    angular_area = 0.0
    pitch_over_distance = 0.05
    panel_half_width_px = 2.0
    for sector in range(8):
        phi_lower = -np.pi + sector * np.pi / 4.0
        phi_upper = phi_lower + np.pi / 4.0
        phi = 0.5 * (phi_upper - phi_lower) * node + 0.5 * (phi_upper + phi_lower)
        phi_weight = 0.5 * (phi_upper - phi_lower) * weight
        radial_limit_px = panel_half_width_px / np.maximum(np.abs(np.sin(phi)), np.abs(np.cos(phi)))
        theta_upper = np.arctan(pitch_over_distance * radial_limit_px)
        two_theta = 0.5 * (node[None, :] + 1.0) * theta_upper[:, None]
        two_theta_weight = 0.5 * theta_upper[:, None] * weight[None, :]
        phi_grid = np.broadcast_to(phi[:, None], two_theta.shape)
        area_weight = phi_weight[:, None] * two_theta_weight
        evaluated = angle_function(two_theta, phi_grid)
        assert np.all(evaluated.valid)
        signal += float(np.sum(evaluated.signal_density_A2_per_rad2 * area_weight))
        normalization += float(np.sum(evaluated.normalization_density_px2_per_rad2 * area_weight))
        ratio_integral += float(np.sum(evaluated.intensity_A2_per_px2 * area_weight))
        angular_area += float(np.sum(area_weight))
    return signal, normalization, ratio_integral / angular_area


def test_continuous_angle_signal_and_normalization_conserve_a_finite_detector_bin() -> None:
    base = _instrument(shape_rc=(4, 4), reference_cr=(1.5, 1.5))
    instrument = replace(
        base,
        lab_from_detector=RigidTransform(
            np.eye(3),
            [0.0, 0.0, 1.0],
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_row_pitch_m=0.05,
        detector_column_pitch_m=0.05,
    )
    angle_function = ContinuousNormalizedAngleFunction(
        _AnalyticDetectorDensity(instrument),
        _frame(),
    )
    coarse = _integrate_square_panel_in_angle_coordinates(angle_function, order=4)
    medium = _integrate_square_panel_in_angle_coordinates(angle_function, order=8)
    fine = _integrate_square_panel_in_angle_coordinates(angle_function, order=16)
    expected_normalization = 16.0
    expected_signal = 64.0 + 64.0 / 15.0
    expected_intensity = expected_signal / expected_normalization
    fine_error = abs(fine[0] - expected_signal) + abs(fine[1] - expected_normalization)

    assert fine_error < abs(medium[0] - expected_signal) + abs(medium[1] - expected_normalization)
    assert fine_error < abs(coarse[0] - expected_signal) + abs(coarse[1] - expected_normalization)
    assert fine[0] == pytest.approx(expected_signal, rel=2.0e-14)
    assert fine[1] == pytest.approx(expected_normalization, rel=2.0e-14)
    assert fine[0] / fine[1] == pytest.approx(expected_intensity, rel=2.0e-14)
    assert abs(fine[2] - expected_intensity) > 1.0e-3


def _full_grid(instrument: object, frame: AngleFrame, *, radial_bins: int = 5) -> AngleBinGrid:
    rows, columns = instrument.detector_shape_rc
    corner_column, corner_row = np.meshgrid(
        np.arange(columns + 1, dtype=np.float64) - 0.5,
        np.arange(rows + 1, dtype=np.float64) - 0.5,
    )
    angles = detector_coordinates_to_angles(
        corner_column,
        corner_row,
        instrument=instrument,
        angle_frame=frame,
    )
    assert np.all(angles.valid)
    theta_max = float(np.max(angles.two_theta_rad))
    return AngleBinGrid(
        two_theta_edges_rad=np.linspace(0.0, np.nextafter(theta_max, np.inf), radial_bins + 1),
        chi_raw_edges_rad=np.linspace(-np.pi, np.pi, 17),
        revision="integration-grid.v1",
    )


def _unwrap(raw_chi: np.ndarray) -> np.ndarray:
    result = np.array(raw_chi, dtype=np.float64, copy=True)
    for index in range(1, result.size):
        delta = (raw_chi[index] - raw_chi[index - 1] + np.pi) % (2.0 * np.pi) - np.pi
        result[index] = result[index - 1] + delta
    return result


def _cross(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _inside_convex(point: np.ndarray, polygon: np.ndarray) -> bool:
    edge = np.roll(polygon, -1, axis=0) - polygon
    offset = point - polygon
    crosses = edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0]
    return bool(np.all(crosses >= -2e-14) or np.all(crosses <= 2e-14))


def _segment_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    denominator = _cross(first_delta, second_delta)
    if abs(denominator) <= 1e-18:
        return None
    offset = second_start - first_start
    first_fraction = _cross(offset, second_delta) / denominator
    second_fraction = _cross(offset, first_delta) / denominator
    if -2e-14 <= first_fraction <= 1.0 + 2e-14 and -2e-14 <= second_fraction <= 1.0 + 2e-14:
        return first_start + first_fraction * first_delta
    return None


def _independent_intersection_area(polygon: np.ndarray, rectangle: np.ndarray) -> float:
    lower = rectangle[0]
    upper = rectangle[2]
    points: list[np.ndarray] = []
    for point in polygon:
        if np.all(point >= lower - 2e-14) and np.all(point <= upper + 2e-14):
            points.append(point)
    for point in rectangle:
        if _inside_convex(point, polygon):
            points.append(point)
    for subject_index in range(polygon.shape[0]):
        subject_start = polygon[subject_index]
        subject_end = polygon[(subject_index + 1) % polygon.shape[0]]
        for clip_index in range(4):
            intersection = _segment_intersection(
                subject_start,
                subject_end,
                rectangle[clip_index],
                rectangle[(clip_index + 1) % 4],
            )
            if intersection is not None:
                points.append(intersection)
    unique = [
        point
        for index, point in enumerate(points)
        if not any(np.linalg.norm(point - earlier) <= 2e-13 for earlier in points[:index])
    ]
    if len(unique) < 3:
        return 0.0
    try:
        return float(ConvexHull(np.asarray(unique)).volume)
    except QhullError:
        return 0.0


def _independent_pixel_weights(
    column: int,
    row: int,
    *,
    instrument: object,
    frame: AngleFrame,
    grid: AngleBinGrid,
    pole_cr: tuple[float, float] | None = None,
) -> np.ndarray:
    corner_column = np.array([column - 0.5, column + 0.5, column + 0.5, column - 0.5])
    corner_row = np.array([row - 0.5, row - 0.5, row + 0.5, row + 0.5])
    angles = detector_coordinates_to_angles(
        corner_column,
        corner_row,
        instrument=instrument,
        angle_frame=frame,
    )
    physical_corners = np.column_stack((corner_column, corner_row))
    contains_pole = pole_cr is not None and (
        column - 0.5 <= pole_cr[0] <= column + 0.5 and row - 0.5 <= pole_cr[1] <= row + 0.5
    )
    pieces = []
    if contains_pole:
        pole = np.asarray(pole_cr)
        for first in range(4):
            second = (first + 1) % 4
            if (
                abs(
                    _cross(
                        physical_corners[first] - pole,
                        physical_corners[second] - pole,
                    )
                )
                <= 1e-15
            ):
                continue
            assert angles.azimuth_valid[first] and angles.azimuth_valid[second]
            chi = _unwrap(angles.chi_raw_rad[[first, second]])
            pieces.append(
                np.array(
                    [
                        [0.0, chi[0]],
                        [angles.two_theta_rad[first], chi[0]],
                        [angles.two_theta_rad[second], chi[1]],
                        [0.0, chi[1]],
                    ]
                )
            )
    else:
        assert np.all(angles.azimuth_valid)
        for indices in ((0, 1, 2), (0, 2, 3)):
            index = np.asarray(indices)
            pieces.append(
                np.column_stack((angles.two_theta_rad[index], _unwrap(angles.chi_raw_rad[index])))
            )
    piece_area = math.fsum(float(ConvexHull(piece).volume) for piece in pieces)
    expected = np.zeros(grid.shape, dtype=np.float64)
    period = 2.0 * np.pi
    for chi_bin in range(grid.shape[0]):
        for theta_bin in range(grid.shape[1]):
            theta_lower = grid.two_theta_edges_rad[theta_bin]
            theta_upper = grid.two_theta_edges_rad[theta_bin + 1]
            for shift in range(-2, 3):
                chi_lower = grid.chi_raw_edges_rad[chi_bin] + shift * period
                chi_upper = grid.chi_raw_edges_rad[chi_bin + 1] + shift * period
                rectangle = np.array(
                    [
                        [theta_lower, chi_lower],
                        [theta_upper, chi_lower],
                        [theta_upper, chi_upper],
                        [theta_lower, chi_upper],
                    ]
                )
                expected[chi_bin, theta_bin] += math.fsum(
                    _independent_intersection_area(piece, rectangle) for piece in pieces
                )
    return expected / piece_area


def _dense_projector(projector: object) -> np.ndarray:
    bins = int(np.prod(projector.grid.shape))
    pixels = int(np.prod(projector.instrument.detector_shape_rc))
    dense = np.zeros((bins, pixels), dtype=np.float64)
    np.add.at(
        dense, (projector.coverage_bin_index, projector.coverage_pixel_index), projector.weight
    )
    return dense


def test_angle_projector_fingerprint_tracks_only_detector_geometry() -> None:
    instrument = _instrument(shape_rc=(2, 3))
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    grid = _full_grid(instrument, frame, radial_bins=4)
    baseline = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
    )
    assert baseline.instrument_fingerprint.endswith(".v2")

    detector = instrument.lab_from_detector
    angle_rad = np.deg2rad(3.0)
    detector_rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    detector_causal_instruments = (
        replace(
            instrument,
            lab_from_detector=RigidTransform(
                detector_rotation,
                detector.translation_m,
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
        ),
        replace(
            instrument,
            lab_from_detector=RigidTransform(
                detector.rotation,
                detector.translation_m + np.array([1.0e-4, 0.0, 0.0]),
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
        ),
        replace(instrument, detector_shape_rc=(3, 3)),
        replace(instrument, detector_row_pitch_m=instrument.detector_row_pitch_m * 1.01),
        replace(
            instrument,
            detector_column_pitch_m=instrument.detector_column_pitch_m * 1.01,
        ),
        replace(
            instrument,
            detector_reference_coordinate_px=(
                instrument.detector_reference_coordinate_px[0] + 0.25,
                instrument.detector_reference_coordinate_px[1],
            ),
        ),
    )
    causal_projectors = tuple(
        compile_detector_angle_projector(
            instrument=item,
            angle_frame=frame,
            grid=grid,
        )
        for item in detector_causal_instruments
    )
    causal_fingerprints = {item.instrument_fingerprint for item in causal_projectors}
    assert baseline.instrument_fingerprint not in causal_fingerprints
    assert len(causal_fingerprints) == len(causal_projectors)
    assert all(item.cache_key != baseline.cache_key for item in causal_projectors)

    sample_rotation = np.array(
        [
            [np.cos(angle_rad), 0.0, np.sin(angle_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_rad), 0.0, np.cos(angle_rad)],
        ]
    )
    excluded_instruments = (
        replace(
            instrument,
            lab_from_sample=RigidTransform(
                sample_rotation,
                np.array([1.0e-4, -2.0e-4, 3.0e-4]),
                FrameId.SAMPLE,
                FrameId.LAB,
            ),
        ),
        replace(
            instrument,
            sample_from_crystal=RigidTransform(
                sample_rotation,
                np.array([2.0e-4, 0.0, 0.0]),
                FrameId.CRYSTAL,
                FrameId.SAMPLE,
            ),
        ),
        replace(instrument, sample_width_m=instrument.sample_width_m * 1.1),
        replace(
            instrument,
            sample_support_model_id="unbounded_plane.v1",
            sample_width_m=None,
            sample_length_m=None,
        ),
        replace(instrument, film_thickness_A=instrument.film_thickness_A * 1.2),
    )
    for excluded_instrument in excluded_instruments:
        candidate = compile_detector_angle_projector(
            instrument=excluded_instrument,
            angle_frame=frame,
            grid=grid,
        )
        assert candidate.instrument_fingerprint == baseline.instrument_fingerprint
        assert candidate.cache_key == baseline.cache_key
        for name in (
            "detector_valid_mask",
            "angle_bin_valid_mask",
            "coverage_pixel_index",
            "coverage_bin_index",
            "weight",
            "lost_support",
        ):
            np.testing.assert_array_equal(getattr(candidate, name), getattr(baseline, name))

    changed_frame = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="separate-angle-frame-owner.v2"),
        grid=grid,
    )
    assert changed_frame.instrument_fingerprint == baseline.instrument_fingerprint
    assert changed_frame.cache_key != baseline.cache_key


def test_sparse_projector_matches_independent_polygon_oracle_across_seam() -> None:
    angle = np.deg2rad(7.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    instrument = _instrument(shape_rc=(2, 3), detector_rotation=rotation)
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    grid = _full_grid(instrument, frame, radial_bins=4)
    near_canonical_theta = grid.two_theta_edges_rad.copy()
    near_canonical_chi = grid.chi_raw_edges_rad.copy()
    near_canonical_theta[0] = np.nextafter(0.0, np.inf)
    near_canonical_chi[[0, -1]] = np.nextafter([-np.pi, np.pi], 0.0)
    canonicalized = AngleBinGrid(near_canonical_theta, near_canonical_chi, "canonicalized.v1")
    assert canonicalized.two_theta_edges_rad[0] == 0.0
    np.testing.assert_array_equal(canonicalized.chi_raw_edges_rad[[0, -1]], [-np.pi, np.pi])
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
    )
    dense = _dense_projector(projector)

    seam_pixel: int | None = None

    for row in range(2):
        for column in range(3):
            pixel = row * 3 + column
            pixel_angles = detector_coordinates_to_angles(
                [column - 0.5, column + 0.5, column + 0.5, column - 0.5],
                [row - 0.5, row - 0.5, row + 0.5, row + 0.5],
                instrument=instrument,
                angle_frame=frame,
            )
            if np.ptp(pixel_angles.chi_raw_rad) > np.pi:
                seam_pixel = pixel
            expected = _independent_pixel_weights(
                column,
                row,
                instrument=instrument,
                frame=frame,
                grid=grid,
            )
            np.testing.assert_allclose(
                dense[:, pixel].reshape(grid.shape),
                expected,
                rtol=3e-11,
                atol=3e-13,
            )
    assert seam_pixel is not None
    seam_column = dense[:, seam_pixel].reshape(grid.shape)
    assert np.any(seam_column[0] > 0.0)
    assert np.any(seam_column[-1] > 0.0)
    np.testing.assert_allclose(np.sum(dense, axis=0), 1.0, rtol=0.0, atol=3e-12)
    np.testing.assert_allclose(projector.lost_support, 0.0, rtol=0.0, atol=3e-12)
    pair_key = projector.coverage_pixel_index * math.prod(grid.shape) + projector.coverage_bin_index
    assert np.all(np.diff(pair_key) > 0)
    assert np.all(projector.weight > 0.0)
    assert not projector.weight.flags.writeable
    assert not projector.coverage_bin_index.flags.writeable
    assert not projector.coverage_pixel_index.flags.writeable

    with pytest.raises(ValueError, match="cache_key"):
        replace(projector, cache_key="stale.v1")
    split_revision_a = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="ab"),
        grid=replace(grid, revision="c"),
    )
    split_revision_b = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="a"),
        grid=replace(grid, revision="bc"),
    )
    changed_mask = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
        detector_valid_mask=np.array([[False, True, True], [True, True, True]]),
    )
    assert (
        len(
            {
                projector.cache_key,
                split_revision_a.cache_key,
                split_revision_b.cache_key,
                changed_mask.cache_key,
            }
        )
        == 4
    )


def test_cropped_profile_projector_matches_the_full_exact_polygon_reduction() -> None:
    instrument = _instrument(shape_rc=(25, 25), reference_cr=(12.0, 12.0))
    frame = _frame([1.1e-3, -0.7e-3, 0.0])
    grid = _full_grid(instrument, frame, radial_bins=16)
    full_projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
    )
    signal = np.arange(1.0, 626.0).reshape(25, 25)
    normalization = 1.0 + (np.arange(625.0).reshape(25, 25) % 4.0)
    full = to_increasing_phi(
        project_normalized_angle_field(
            full_projector,
            signal,
            normalization,
        )
    )
    phi_edges = grid.phi_edges_rad[6:10][None, :]
    theta_bounds = np.asarray(((grid.two_theta_edges_rad[3], grid.two_theta_edges_rad[5]),))
    profile_mask = np.asarray(((True, True, True),))
    local_projector = compile_detector_profile_projector(
        instrument=instrument,
        angle_frame=frame,
        two_theta_bounds_rad=theta_bounds,
        phi_bin_edges_rad=phi_edges,
        profile_bin_valid_mask=profile_mask,
    )
    local = project_detector_profiles(local_projector, signal, normalization)
    expected_signal = np.sum(full.S[6:9, 3:5], axis=1)[None, :]
    expected_normalization = np.sum(full.N[6:9, 3:5], axis=1)[None, :]
    expected_signal[~profile_mask] = 0.0
    expected_normalization[~profile_mask] = 0.0
    np.testing.assert_allclose(local.S, expected_signal, rtol=3e-11, atol=3e-13)
    np.testing.assert_allclose(local.N, expected_normalization, rtol=3e-11, atol=3e-13)
    np.testing.assert_array_equal(local.valid, profile_mask)
    np.testing.assert_allclose(
        local.I[profile_mask],
        (expected_signal / np.where(expected_normalization > 0.0, expected_normalization, 1.0))[
            profile_mask
        ],
        rtol=3e-11,
        atol=3e-13,
    )
    assert np.all(local_projector.profile_pixel_bounds_cr[:, (0, 2)] >= 0)
    assert local.projector_cache_key == local_projector.cache_key

    seam_phi_edges = np.asarray(
        (
            (
                grid.phi_edges_rad[14],
                grid.phi_edges_rad[15],
                grid.phi_edges_rad[16],
                grid.phi_edges_rad[1] + 2.0 * np.pi,
                grid.phi_edges_rad[2] + 2.0 * np.pi,
            ),
        )
    )
    seam_theta_bounds = np.asarray(((grid.two_theta_edges_rad[1], grid.two_theta_edges_rad[2]),))
    seam_mask = np.asarray(((True, False, True, True),))
    seam_projector = compile_detector_profile_projector(
        instrument=instrument,
        angle_frame=frame,
        two_theta_bounds_rad=seam_theta_bounds,
        phi_bin_edges_rad=seam_phi_edges,
        profile_bin_valid_mask=seam_mask,
    )
    seam = project_detector_profiles(seam_projector, signal, normalization)
    full_phi_indices = np.asarray((14, 15, 0, 1))
    seam_expected_signal = full.S[full_phi_indices, 1][None, :]
    seam_expected_normalization = full.N[full_phi_indices, 1][None, :]
    seam_expected_signal[~seam_mask] = 0.0
    seam_expected_normalization[~seam_mask] = 0.0
    np.testing.assert_allclose(seam.S, seam_expected_signal, rtol=3e-11, atol=3e-13)
    np.testing.assert_allclose(
        seam.N,
        seam_expected_normalization,
        rtol=3e-11,
        atol=3e-13,
    )
    np.testing.assert_array_equal(seam.valid, seam_mask)

    broad_grid = AngleBinGrid(
        two_theta_edges_rad=grid.two_theta_edges_rad,
        chi_raw_edges_rad=np.linspace(-np.pi, np.pi, 65),
        revision="integration-broad-phi-grid.v1",
    )
    broad_full = to_increasing_phi(
        project_normalized_angle_field(
            compile_detector_angle_projector(
                instrument=instrument,
                angle_frame=frame,
                grid=broad_grid,
            ),
            signal,
            normalization,
        )
    )
    broad_phi_edges = broad_grid.phi_edges_rad[1:][None, :]
    broad_mask = np.ones((1, broad_phi_edges.shape[1] - 1), dtype=np.bool_)
    broad_projector = compile_detector_profile_projector(
        instrument=instrument,
        angle_frame=frame,
        two_theta_bounds_rad=theta_bounds,
        phi_bin_edges_rad=broad_phi_edges,
        profile_bin_valid_mask=broad_mask,
    )
    broad = project_detector_profiles(broad_projector, signal, normalization)
    broad_expected_signal = np.sum(broad_full.S[1:, 3:5], axis=1)[None, :]
    broad_expected_normalization = np.sum(broad_full.N[1:, 3:5], axis=1)[None, :]
    np.testing.assert_allclose(broad.S, broad_expected_signal, rtol=3e-11, atol=3e-13)
    np.testing.assert_allclose(
        broad.N,
        broad_expected_normalization,
        rtol=3e-11,
        atol=3e-13,
    )

    invalid_detector_mask = np.array(seam_projector.detector_valid_mask, copy=True)
    invalid_detector_mask.ravel()[seam_projector.coverage_pixel_index[0]] = False
    with pytest.raises(ValueError, match="valid mask entries"):
        replace(seam_projector, detector_valid_mask=invalid_detector_mask)

    with pytest.raises(ValueError, match="less than one azimuth period"):
        compile_detector_profile_projector(
            instrument=instrument,
            angle_frame=frame,
            two_theta_bounds_rad=seam_theta_bounds,
            phi_bin_edges_rad=np.linspace(-np.pi, np.pi, 4)[None, :],
        )

    near_edge = detector_coordinates_to_angles(
        np.asarray([23.0, 24.0]),
        np.asarray([12.0, 12.0]),
        instrument=instrument,
        angle_frame=frame,
    )
    edge_phi = float(near_edge.phi_rad[0]) + np.linspace(-0.02, 0.02, 4)
    with pytest.raises(ValueError, match="physical detector edge"):
        compile_detector_profile_projector(
            instrument=instrument,
            angle_frame=frame,
            two_theta_bounds_rad=np.asarray(
                ((near_edge.two_theta_rad[0], near_edge.two_theta_rad[1]),)
            ),
            phi_bin_edges_rad=edge_phi[None, :],
        )


def test_measured_mosaic_policy_allows_no_secondary_lobes_and_branchless_exclusions(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    runner = runpy.run_path(
        Path(__file__).resolve().parents[1] / "scripts" / "recover_bi2se3_mosaic.py"
    )
    parse_policy = runner["_measured_profile_policy"]
    case_hash = "a" * 64
    branchless = SimpleNamespace(
        identity=SimpleNamespace(
            dataset_id="material-5deg",
            analytic_branch_id=0,
            branch_id=None,
            group_key=SimpleNamespace(layered_family_m=0, layered_integer_L=6),
        )
    )
    common = (
        'schema_version = "rasim-measured-mosaic-profile-policy-v1"\n'
        f'base_case_sha256 = "{case_hash}"\n'
        "minimum_excess_energy_over_side_scatter = 5.0\n"
        "sideband_two_theta_offsets_deg = [-0.5, 0.5]\n"
    )
    empty_path = tmp_path / "empty-policy.toml"
    empty_path.write_text(common, encoding="utf-8")
    empty = parse_policy(empty_path, case_sha256=case_hash, definitions=((branchless,),))
    assert empty.excluded_profile_keys == frozenset()

    branchless_path = tmp_path / "branchless-policy.toml"
    branchless_path.write_text(
        common
        + "\n[[excluded_profiles]]\n"
        + 'dataset_id = "material-5deg"\n'
        + "family_m = 0\n"
        + "integer_L = 6\n"
        + "analytic_branch_id = 0\n"
        + 'reason = "USER_AUTHORIZED_SECONDARY_LOBE"\n',
        encoding="utf-8",
    )
    parsed = parse_policy(
        branchless_path,
        case_sha256=case_hash,
        definitions=((branchless,),),
    )
    assert parsed.excluded_profile_keys == frozenset({("material-5deg", 0, 6, 0, None)})


def test_mosaic_runner_separates_nominal_geometry_from_one_shared_source_ensemble() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_mosaic.py")
    case_path = root / "examples" / "bi2se3" / "experiment" / "mosaic_fit_truth.toml"
    case, _, _ = runner["_case"](case_path)

    incidence_angle_delta_rad = math.radians(0.2)
    position = runner["_FixedPositionState"](
        artifact_revision="sha256-" + "0" * 64,
        corrections=runner["SharedGeometryCorrections"].from_array(
            case["shared_geometry_corrections"]
        ),
        incidence_angle_delta_rad=incidence_angle_delta_rad,
    )
    base, series, nominal_series = runner["_fixed_geometry_inputs"](
        case_path,
        case,
        source_sample_count=4,
        position=position,
    )

    assert base.samples.incident_sample_id.size == 4
    assert base.config.source.sample_count == 4
    assert all(item.samples is base.samples for item in series)
    assert all(
        item.incident.states.source_revision == base.samples.source_revision for item in series
    )
    assert all(np.all(item.incident.states.valid) for item in series)
    assert np.unique(base.samples.origin_lab_m, axis=0).shape[0] == 2
    assert np.unique(base.samples.direction_lab, axis=0).shape[0] == 2
    np.testing.assert_array_equal(
        np.unique(base.samples.wavelength_A),
        np.asarray(base.config.source.line_wavelength_A),
    )
    for wavelength_A, probability in zip(
        base.config.source.line_wavelength_A,
        base.config.source.line_probability,
        strict=True,
    ):
        assert np.sum(
            base.samples.source_weight[base.samples.wavelength_A == wavelength_A]
        ) == pytest.approx(probability, rel=0.0, abs=5.0e-16)
    assert all(item.samples.incident_sample_id.size == 1 for item in nominal_series)
    from rasim_next.pipeline.configured_simulation import build_geometry_only_ewald_context

    nominal_contexts = tuple(build_geometry_only_ewald_context(item) for item in nominal_series)
    assert all(item.incident.states.valid.tolist() == [True] for item in nominal_contexts)
    assert all(
        item.samples.source_sampling_model_id == "nominal_mean_geometry_reference.v1"
        for item in nominal_series
    )
    assert all(
        float(item.samples.wavelength_A[0]) == base.config.source.mean_wavelength_A
        for item in nominal_series
    )
    expected_effective_angles_deg = (5.2, 10.2, 15.2)
    assert tuple(
        item.config.instrument.axis_rotations[0].angle_deg for item in series
    ) == pytest.approx(expected_effective_angles_deg, abs=1.0e-14)
    assert tuple(
        item.config.instrument.axis_rotations[0].angle_deg for item in nominal_series
    ) == pytest.approx(expected_effective_angles_deg, abs=1.0e-14)

    ordered_runner = runpy.run_path(root / "scripts" / "recover_bi2se3_ordered_intensity.py")
    corrections = runner["SharedGeometryCorrections"].from_array(
        case["shared_geometry_corrections"]
    )
    fixed_geometry = json.loads(
        json.dumps(
            runner["_fixed_geometry_record"](
                case,
                base,
                runner["_FixedPositionState"](
                    artifact_revision="sha256-" + "1" * 64,
                    corrections=corrections,
                    incidence_angle_delta_rad=incidence_angle_delta_rad,
                ),
            )
        )
    )
    fixed_position = ordered_runner["_fixed_position_state"](fixed_geometry, case)
    assert ordered_runner["_fixed_position_record"](fixed_position) == fixed_geometry
    unknown_position_field = {**fixed_geometry, "independent_angle_delta_rad": 0.0}
    with pytest.raises(ValueError, match="invalid fixed position record"):
        ordered_runner["_fixed_position_state"](unknown_position_field, case)
    ordered_series = ordered_runner["_fixed_inputs"](
        case_path,
        case,
        source_sample_count=2,
        mosaic_parameters={
            "gaussian_sigma_deg": float(case["truth"]["gaussian_sigma_deg"]),
            "lorentzian_hwhm_deg": float(case["truth"]["lorentzian_hwhm_deg"]),
            "lorentzian_probability": float(case["truth"]["lorentzian_probability"]),
        },
        fixed_position=fixed_position,
    )
    assert tuple(
        item.config.instrument.axis_rotations[0].angle_deg for item in ordered_series
    ) == pytest.approx(expected_effective_angles_deg, abs=1.0e-14)
    for mosaic_inputs, ordered_inputs in zip(series, ordered_series, strict=True):
        np.testing.assert_array_equal(
            ordered_inputs.instrument.sample_from_crystal.rotation,
            mosaic_inputs.instrument.sample_from_crystal.rotation,
        )
        np.testing.assert_array_equal(
            ordered_inputs.instrument.sample_from_crystal.translation_m,
            mosaic_inputs.instrument.sample_from_crystal.translation_m,
        )
        np.testing.assert_array_equal(
            ordered_inputs.instrument.lab_from_sample.rotation,
            mosaic_inputs.instrument.lab_from_sample.rotation,
        )
        np.testing.assert_array_equal(
            ordered_inputs.instrument.lab_from_sample.translation_m,
            mosaic_inputs.instrument.lab_from_sample.translation_m,
        )
        np.testing.assert_array_equal(
            ordered_inputs.instrument.lab_from_detector.rotation,
            mosaic_inputs.instrument.lab_from_detector.rotation,
        )
        np.testing.assert_array_equal(
            ordered_inputs.instrument.lab_from_detector.translation_m,
            mosaic_inputs.instrument.lab_from_detector.translation_m,
        )


def test_measured_mosaic_cli_requires_one_atomic_position_artifact_state(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_mosaic.py")
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="requires --position-artifact"):
        runner["main"](
            [
                "--output-directory",
                str(output),
                "--observation-mode",
                "osc",
            ]
        )
    assert not output.exists()

    invalid_artifact = tmp_path / "geometry.json"
    invalid_artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported staged-fit artifact schema"):
        runner["main"](
            [
                "--output-directory",
                str(output),
                "--observation-mode",
                "osc",
                "--position-artifact",
                str(invalid_artifact),
            ]
        )
    assert not output.exists()


def test_ordered_recovery_requires_mosaic_or_explicit_synthetic_proof(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_ordered_intensity.py")

    with pytest.raises(ValueError, match="requires --mosaic-result or --synthetic-truth-proof"):
        runner["main"]([])
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner["main"](
            [
                "--mosaic-result",
                str(tmp_path / "unused.json"),
                "--synthetic-truth-proof",
            ]
        )

    case_path = runner["DEFAULT_CASE"]
    _, mosaic_case_path, mosaic_case = runner["_load_case"](case_path)
    position = runner["_case_fixed_position_state"](mosaic_case_path, mosaic_case)
    with pytest.raises(ValueError, match="complete upstream mosaic artifact state"):
        runner["run_recovery"](
            case_path,
            source_sample_count=1,
            fixed_position=position,
        )


def test_mosaic_runner_profile_is_weighted_source_state_sum_not_nominal_only() -> None:
    from rasim_next.core.contracts import IncidentSampleBatch
    from rasim_next.fitting.mosaic import (
        MosaicProfileDefinition,
        MosaicProfileIdentity,
        MosaicReflectionGroupKey,
    )
    from rasim_next.selection import build_osc_angle_frame

    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_mosaic.py")
    case_path = root / "examples" / "bi2se3" / "experiment" / "mosaic_fit_truth.toml"
    case, _, _ = runner["_case"](case_path)
    base, series, nominal_series = runner["_fixed_geometry_inputs"](
        case_path,
        case,
        source_sample_count=2,
        position=runner["_case_fixed_position_state"](case),
    )
    physics, geometry = runner["_profile_forward_contexts"](base, (series[0],))
    nominal_context = build_nominal_ewald_context(series[0])
    markers = evaluate_nominal_integer_l_markers(nominal_context)
    candidates = np.flatnonzero((markers.family_m == 1) & (markers.root_sign != 0))
    marker_index = int(candidates[np.argmax(markers.family_strength_weight_A2[candidates])])
    frame = build_osc_angle_frame(
        mean_direction_lab=nominal_series[0].config.source.mean_direction_lab,
        instrument=nominal_series[0].instrument,
        sample_intersection_lab_m=(nominal_context.incident.states.sample_intersection_lab_m[0]),
        revision="two-state-profile-source-aggregation-proof.v1",
    )
    angles = detector_coordinates_to_angles(
        np.asarray([markers.column_px[marker_index]]),
        np.asarray([markers.row_px[marker_index]]),
        instrument=nominal_series[0].instrument,
        angle_frame=frame,
    )
    root_sign = int(markers.root_sign[marker_index])
    definition = MosaicProfileDefinition(
        identity=MosaicProfileIdentity(
            dataset_id="two-state-source-proof",
            incidence_angle_rad=math.radians(float(case["incidence_angles_deg"][0])),
            group_key=MosaicReflectionGroupKey(
                group_id="two-state-source-proof:m=1",
                rod_catalog_revision=runner["configured_rod_catalog_revision"](series[0]),
                member_rod_hk=markers.contributing_rod_hk[marker_index],
                branch_mode="EXPLICIT_NONZERO",
                layered_family_m=1,
                layered_integer_L=int(markers.integer_L[marker_index]),
            ),
            branch_id=1 if root_sign < 0 else 2,
            analytic_branch_id=int(markers.branch[marker_index]),
        ),
        center_two_theta_rad=float(angles.two_theta_rad[0]),
        center_phi_rad=float(angles.phi_rad[0]),
        two_theta_half_width_rad=math.radians(0.02),
        phi_half_width_rad=math.radians(0.15),
        phi_bin_count=5,
        two_theta_gauss_order=2,
        phi_gauss_order=2,
    )
    mosaic = runner["_mosaic_parameters"](
        gaussian_sigma_rad=math.radians(1.0),
        lorentzian_half_width_rad=math.radians(0.5),
        lorentzian_probability=0.1,
        context=physics,
    )
    profile_revision = "two-state-profile-source-aggregation-proof.v1"
    combined, _ = runner["_evaluate_profile_series"](
        physics,
        geometry,
        (frame,),
        ((definition,),),
        mosaic,
        profile_revision=profile_revision,
        execution_backend="cpu",
    )

    explicit_signal = np.zeros_like(combined.signal)
    for state_index, source_weight in enumerate(base.samples.source_weight):
        singleton_samples = IncidentSampleBatch(
            incident_sample_id=base.samples.incident_sample_id[state_index : state_index + 1],
            origin_lab_m=base.samples.origin_lab_m[state_index : state_index + 1],
            direction_lab=base.samples.direction_lab[state_index : state_index + 1],
            wavelength_A=base.samples.wavelength_A[state_index : state_index + 1],
            source_weight=np.asarray([1.0]),
            polarization_state_id=(base.samples.polarization_state_id[state_index],),
            source_sampling_model_id="explicit_external_source.v1",
            source_rng_model_id="no_rng.v1",
            source_seed=state_index,
            source_parameter_provenance=f"profile source oracle row {state_index}",
        )
        singleton_incident = build_incident_states(
            singleton_samples,
            physics.material,
            geometry[0].instrument,
        )
        singleton_geometry = (
            runner["_ProfileGeometryContext"](
                incident=singleton_incident,
                instrument=geometry[0].instrument,
            ),
        )
        singleton, _ = runner["_evaluate_profile_series"](
            physics,
            singleton_geometry,
            (frame,),
            ((definition,),),
            mosaic,
            profile_revision=profile_revision,
            execution_backend="cpu",
        )
        explicit_signal += float(source_weight) * singleton.signal
        np.testing.assert_allclose(singleton.normalization, combined.normalization, rtol=0, atol=0)
        np.testing.assert_array_equal(singleton.valid, combined.valid)

    np.testing.assert_allclose(combined.signal, explicit_signal, rtol=3.0e-11, atol=1.0e-22)
    from rasim_next.pipeline.configured_simulation import build_geometry_only_ewald_context

    geometry_only_context = build_geometry_only_ewald_context(nominal_series[0])
    nominal_physics = replace(physics, material=nominal_series[0].material)
    nominal_geometry = (
        runner["_ProfileGeometryContext"](
            incident=geometry_only_context.incident,
            instrument=nominal_series[0].instrument,
        ),
    )
    assert np.linalg.norm(combined.signal) > 0.0
    with pytest.raises(ValueError, match="cannot contribute to intensity"):
        runner["_evaluate_profile_series"](
            nominal_physics,
            nominal_geometry,
            (frame,),
            ((definition,),),
            mosaic,
            profile_revision=profile_revision,
            execution_backend="cpu",
        )


def test_mosaic_runner_passes_nominally_unsupported_m0_to_combined_source_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_mosaic.py")
    case_path = root / "examples" / "bi2se3" / "experiment" / "mosaic_fit_truth.toml"
    case, _, _ = runner["_case"](case_path)
    base, series, nominal_series = runner["_fixed_geometry_inputs"](
        case_path,
        case,
        source_sample_count=2,
        position=runner["_case_fixed_position_state"](case),
    )
    shared = {
        "source_inputs": series[1],
        "case_path": case_path,
        "incidence_deg": float(case["incidence_angles_deg"][1]),
        "osc_observation": case["m0_observations"][1],
        "nonzero_centroid_provenance": case["nonzero_centroid_provenance"],
        "centroid_provenance": case["m0_centroid_provenance"],
        "profile_config": case["profiles"],
        "rod_catalog_revision": runner["configured_rod_catalog_revision"](base),
    }
    _, nominal_only, _, _ = runner["_profile_definitions"](
        nominal_series[1],
        include_nominally_unsupported_m0=False,
        **shared,
    )
    _, combined_candidates, _, audit = runner["_profile_definitions"](
        nominal_series[1],
        include_nominally_unsupported_m0=True,
        **shared,
    )

    def m0_orders(definitions) -> set[int]:
        return {
            definition.identity.group_key.layered_integer_L
            for definition in definitions
            if definition.identity.group_key.layered_family_m == 0
        }

    assert m0_orders(nominal_only) == {6, 9}
    assert m0_orders(combined_candidates) == {3, 6, 9}
    assert audit["combined_source_gate_candidate_integer_L"] == [6, 9, 3]
    has_support = runner["source_averaged_profile_has_support"]
    supported = has_support(family_m=0, profile_signal_A2=1.0)
    assert type(supported) is bool
    assert supported
    assert not has_support(family_m=0, profile_signal_A2=0.0)
    assert has_support(family_m=1, profile_signal_A2=0.0)

    ordered_runner = runpy.run_path(root / "scripts" / "recover_bi2se3_ordered_intensity.py")
    ordered_case, _, _ = ordered_runner["_load_case"](
        root / "examples" / "bi2se3" / "experiment" / "ordered_intensity_fit_truth.toml"
    )
    provisional_key = frozenset({("Bi2Se3-10deg", 0, 3, 0, None)})
    _, ordered_definitions = ordered_runner["_profile_definitions"](
        series[1],
        incidence_deg=float(case["incidence_angles_deg"][1]),
        m0_observation=case["m0_observations"][1],
        profile_config=ordered_case["profiles"],
        two_theta_gauss_order=2,
        phi_gauss_order=2,
        eligible_profile_keys=provisional_key,
    )
    assert {
        ordered_runner["_definition_identity_key"](definition) for definition in ordered_definitions
    } == set(provisional_key)
    with pytest.raises(ValueError, match="unknown dataset"):
        ordered_runner["_validate_eligible_profile_dataset_ids"](
            frozenset({("TYPO", 0, 3, 0, None)}),
            frozenset({"Bi2Se3-5deg", "Bi2Se3-10deg", "Bi2Se3-15deg"}),
        )


def test_ordered_recovery_validates_gate_v2_measured_profile_catalog() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_ordered_intensity.py")
    case_path = root / "examples" / "bi2se3" / "experiment" / "ordered_intensity_fit_truth.toml"
    case, mosaic_case_path, mosaic_case = runner["_load_case"](case_path)
    mosaic_parameters = {
        "gaussian_sigma_deg": 1.322875655532295,
        "lorentzian_hwhm_deg": 0.4898979485566357,
        "lorentzian_probability": 0.4480961629924146,
    }
    series = runner["_fixed_inputs"](
        mosaic_case_path,
        mosaic_case,
        source_sample_count=2,
        mosaic_parameters=mosaic_parameters,
        fixed_position=runner["_case_fixed_position_state"](
            mosaic_case_path,
            mosaic_case,
        ),
    )
    eligible = frozenset(
        {
            ("Bi2Se3-5deg", 0, 3, 0, None),
            ("Bi2Se3-5deg", 0, 6, 0, None),
            ("Bi2Se3-10deg", 0, 6, 0, None),
            ("Bi2Se3-10deg", 1, 5, 2, 1),
            ("Bi2Se3-10deg", 1, 5, 2, 2),
            ("Bi2Se3-10deg", 1, 10, 2, 1),
            ("Bi2Se3-10deg", 1, 10, 2, 2),
            ("Bi2Se3-15deg", 0, 6, 0, None),
            ("Bi2Se3-15deg", 0, 9, 0, None),
            ("Bi2Se3-15deg", 1, 5, 2, 1),
            ("Bi2Se3-15deg", 1, 5, 2, 2),
            ("Bi2Se3-15deg", 1, 10, 2, 1),
            ("Bi2Se3-15deg", 1, 10, 2, 2),
            ("Bi2Se3-15deg", 1, 11, 2, 1),
            ("Bi2Se3-15deg", 1, 11, 2, 2),
        }
    )
    catalogs, records = runner["_validated_profile_catalogs"](
        series,
        incidence_angles_deg=mosaic_case["incidence_angles_deg"],
        m0_observations=mosaic_case["m0_observations"],
        profile_config=case["profiles"],
        eligible_profile_keys=eligible,
    )

    assert [(record["total"], record["m0"]) for record in records] == [
        (2, 2),
        (5, 1),
        (8, 2),
    ]
    assert [record["profile_catalog_revision"] for record in records] == [
        "5b7b3fceee0621ce95cfcbac637a0a3e5ca78ea038f4d19bb4611ecd35863e04",
        "31f8b35c28eb34cfc31ed6f3870a4611c342a122355b976532cd379c02caf6b6",
        "e78a32d1bc1cb9e107ed9b352a7dbbbe99b8cf3c01c96a2d32aa268bf42cdb26",
    ]
    assert {
        runner["_definition_identity_key"](definition)
        for _, definitions in catalogs
        for definition in definitions
    } == set(eligible)
    actual = [dict(record, invalid_or_caustic_anchors=0) for record in records]
    runner["_validate_anchor_catalog_records"](actual, records)
    actual[0]["total"] = 88
    with pytest.raises(ValueError, match="frozen anchor catalogs"):
        runner["_validate_anchor_catalog_records"](actual, records)


def test_recovery_runners_reject_pre_combined_source_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_ordered_intensity.py")

    with pytest.raises(ValueError, match="accepted real-OSC mosaic result"):
        runner["_validated_mosaic_result"](
            {
                "schema_version": "rasim-bi2se3-real-mosaic-fit-v1",
                "status": "MODEL_LIMITED_EFFECTIVE_RADIAL_MOSAIC_ESTIMATE",
            },
            mosaic_case_path=root / "obsolete.toml",
            mosaic_case={},
            source_sample_count=250,
        )

    validate_ordered = runner["_validate_ordered_result_header"]
    with pytest.raises(ValueError, match="accepted source-averaged"):
        validate_ordered(
            {
                "schema_version": "rasim-bi2se3-ordered-intensity-recovery-v2",
                "accepted": True,
                "positions_frozen": True,
            }
        )
    current = {
        "schema_version": "rasim-bi2se3-ordered-intensity-recovery-v4",
        "accepted": True,
        "positions_frozen": True,
        "response_contract": runner["_response_contract_record"](),
    }
    validate_ordered(current)
    current["response_contract"] = {"revision": "stale"}
    with pytest.raises(ValueError, match="stale response compiler contract"):
        validate_ordered(current)


def test_ordered_intensity_runner_rejects_malformed_combined_source_support() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(root / "scripts" / "recover_bi2se3_ordered_intensity.py")
    validate_support = runner["_validate_combined_source_profile_support"]

    def observations(
        *,
        family_m: object = 0,
        modeled_signal: object = 1.0,
        modeled_support: object = True,
        fit_eligible: bool = True,
        gate_revision: str = "positive-combined-detector-m0-profile-signal.v2",
    ) -> dict[str, object]:
        return {
            "candidate_profile_count": 1,
            "fitted_profile_count": int(fit_eligible),
            "measured_profile_policy": {
                "selection_sampler_revision": (
                    "detector-native-bilinear-profile-centerline-sidebands.v1"
                ),
                "source_averaged_modeled_support_gate_revision": gate_revision,
                "profile_selection": [
                    {
                        "dataset_id": "test-dataset",
                        "family_m": family_m,
                        "integer_L": 3,
                        "analytic_branch_id": 0,
                        "root_side_branch_id": None,
                        "source_averaged_modeled_signal_A2": modeled_signal,
                        "source_averaged_modeled_support": modeled_support,
                        "fit_eligible": fit_eligible,
                    }
                ],
            },
        }

    eligible_key = ("test-dataset", 0, 3, 0, None)
    assert validate_support(observations()) == frozenset({eligible_key})
    assert (
        validate_support(
            observations(modeled_signal=0.0, modeled_support=False, fit_eligible=False)
        )
        == frozenset()
    )
    assert validate_support(observations(family_m=1, modeled_signal=0.0)) == frozenset(
        {("test-dataset", 1, 3, 0, None)}
    )
    with pytest.raises(ValueError, match="current combined-source profile-support gate"):
        validate_support(
            observations(gate_revision="positive-combined-detector-m0-profile-signal.v1")
        )
    for malformed in (
        observations(modeled_signal=-1.0),
        observations(modeled_signal=0.0),
    ):
        with pytest.raises(ValueError, match="invalid combined-source support audit"):
            validate_support(malformed)
    with pytest.raises(ValueError, match="invalid profile identity"):
        validate_support(observations(family_m="0"))

    group_type = runner["MosaicReflectionGroupKey"]
    identity_type = runner["MosaicProfileIdentity"]
    definition_type = runner["MosaicProfileDefinition"]

    def definition(integer_l: int):
        return definition_type(
            identity=identity_type(
                dataset_id="test-dataset",
                incidence_angle_rad=math.radians(5.0),
                group_key=group_type(
                    group_id=f"test:m=0:L={integer_l}",
                    rod_catalog_revision="test-rods.v1",
                    member_rod_hk=((0, 0),),
                    branch_mode="COLLAPSED_00L",
                    layered_family_m=0,
                    layered_integer_L=integer_l,
                ),
                branch_id=None,
                analytic_branch_id=0,
            ),
            center_two_theta_rad=0.4,
            center_phi_rad=0.1,
            two_theta_half_width_rad=0.01,
            phi_half_width_rad=0.02,
            phi_bin_count=5,
            two_theta_gauss_order=2,
            phi_gauss_order=2,
        )

    candidates = (definition(3), definition(6))
    selected = runner["_filter_profile_definitions"](candidates, frozenset({eligible_key}))
    assert selected == (candidates[0],)


@pytest.mark.parametrize("reference_cr", [(1.0, 1.0), (1.5, 1.0), (1.5, 1.5)])
def test_direct_beam_pole_ties_cover_full_detector_support(
    reference_cr: tuple[float, float],
) -> None:
    base = _instrument(shape_rc=(3, 3), reference_cr=reference_cr)
    instrument = replace(
        base,
        lab_from_detector=RigidTransform(
            np.eye(3), [0.0, 0.0, 0.82], FrameId.DETECTOR, FrameId.LAB
        ),
    )
    frame = _frame()
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=_full_grid(instrument, frame),
    )
    dense = _dense_projector(projector)
    np.testing.assert_allclose(projector.lost_support, 0.0, rtol=0.0, atol=4e-12)
    np.testing.assert_allclose(
        np.sum(dense, axis=0) + projector.lost_support.ravel(),
        1.0,
        rtol=0.0,
        atol=4e-12,
    )
    touching = [
        row * 3 + column
        for row in range(3)
        for column in range(3)
        if column - 0.5 <= reference_cr[0] <= column + 0.5
        and row - 0.5 <= reference_cr[1] <= row + 0.5
    ]
    assert len(touching) in (1, 2, 4)
    supports = []
    for pixel in touching:
        row, column = divmod(pixel, 3)
        expected = _independent_pixel_weights(
            column,
            row,
            instrument=instrument,
            frame=frame,
            grid=projector.grid,
            pole_cr=reference_cr,
        )
        actual = dense[:, pixel].reshape(projector.grid.shape)
        np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=4e-13)
        supports.append(np.flatnonzero(np.any(actual > 0.0, axis=1)))
    assert all(support.size == projector.grid.shape[0] // len(touching) for support in supports)
    np.testing.assert_array_equal(
        np.unique(np.concatenate(supports)),
        np.arange(projector.grid.shape[0]),
    )

    if reference_cr == (1.5, 1.0):
        for shifted_column in (np.nextafter(1.5, -np.inf), np.nextafter(1.5, np.inf)):
            shifted = replace(
                instrument,
                detector_reference_coordinate_px=(shifted_column, 1.0),
            )
            shifted_projector = compile_detector_angle_projector(
                instrument=shifted,
                angle_frame=frame,
                grid=projector.grid,
            )
            np.testing.assert_allclose(
                _dense_projector(shifted_projector),
                dense,
                rtol=0.0,
                atol=4e-12,
            )


def test_normalized_field_freezes_masks_losses_divide_order_and_phi_permutation() -> None:
    instrument = _instrument(shape_rc=(2, 3))
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    full_grid = _full_grid(instrument, frame, radial_bins=2)
    detector_mask = np.array([[True, True, False], [True, True, True]])
    angle_mask = np.ones(full_grid.shape, dtype=np.bool_)
    angle_mask[0, 0] = False
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=full_grid,
        detector_valid_mask=detector_mask,
        angle_bin_valid_mask=angle_mask,
    )
    signal = np.array([[2.0, 9.0, 7.0], [5.0, 4.0, 11.0]])
    normalization = np.array([[1.0, 3.0, 2.0], [5.0, 2.0, 4.0]])
    field = project_normalized_angle_field(projector, signal, normalization)
    dense = _dense_projector(projector)
    expected_signal = (dense @ np.where(detector_mask, signal, 0.0).ravel()).reshape(
        full_grid.shape
    )
    expected_normalization = (dense @ np.where(detector_mask, normalization, 0.0).ravel()).reshape(
        full_grid.shape
    )
    angle_excluded_signal = float(expected_signal[~angle_mask].sum())
    angle_excluded_normalization = float(expected_normalization[~angle_mask].sum())
    expected_signal[~angle_mask] = 0.0
    expected_normalization[~angle_mask] = 0.0
    expected_valid = angle_mask & (expected_normalization > 0.0)
    expected_intensity = np.zeros(full_grid.shape)
    np.divide(
        expected_signal,
        expected_normalization,
        out=expected_intensity,
        where=expected_valid,
    )

    np.testing.assert_allclose(field.S, expected_signal, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(field.N, expected_normalization, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(field.I, expected_intensity, rtol=0.0, atol=3e-14)
    np.testing.assert_array_equal(field.valid, expected_valid)
    assert field.detector_mask_excluded_signal == pytest.approx(7.0)
    assert field.detector_mask_excluded_normalization == pytest.approx(2.0)
    assert field.angle_mask_excluded_signal == pytest.approx(angle_excluded_signal)
    assert field.angle_mask_excluded_normalization == pytest.approx(angle_excluded_normalization)
    assert (
        field.S.sum() + field.angle_mask_excluded_signal + field.angular_lost_signal
        == pytest.approx(signal[detector_mask].sum(), abs=2e-12)
    )
    assert (
        field.N.sum() + field.angle_mask_excluded_normalization + field.angular_lost_normalization
        == pytest.approx(normalization[detector_mask].sum(), abs=2e-12)
    )

    ratio_first = np.zeros_like(signal)
    np.divide(signal, normalization, out=ratio_first, where=normalization > 0.0)
    wrong = (dense @ ratio_first.ravel()).reshape(full_grid.shape)
    assert np.max(np.abs(wrong[expected_valid] - field.I[expected_valid])) > 1e-3

    phi = to_increasing_phi(field)
    assert np.any(full_grid.chi_raw_edges_rad == np.pi / 2.0)
    mapped_phi = (-np.pi / 2.0 - full_grid.chi_raw_centers_rad + np.pi) % (2.0 * np.pi) - np.pi
    permutation = np.argsort(mapped_phi, kind="stable")
    assert np.all(np.diff(phi.grid.phi_centers_rad) > 0.0)
    np.testing.assert_allclose(
        phi.grid.phi_centers_rad,
        mapped_phi[permutation],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(phi.S, field.S[permutation])
    np.testing.assert_array_equal(phi.N, field.N[permutation])
    np.testing.assert_array_equal(phi.I, field.I[permutation])
    np.testing.assert_array_equal(phi.valid, field.valid[permutation])
    np.testing.assert_array_equal(
        phi.angle_bin_valid_mask,
        field.angle_bin_valid_mask[permutation],
    )
    assert phi.projector_cache_key == field.projector_cache_key == projector.cache_key
    assert phi.angular_lost_signal == field.angular_lost_signal
    for array_value in (field.S, phi.valid, projector.weight):
        assert not array_value.flags.writeable

    zero = project_normalized_angle_field(projector, signal, np.zeros_like(normalization))
    assert not np.any(zero.valid)
    assert not np.any(zero.I)
    np.testing.assert_array_equal(zero.S, field.S)
    np.testing.assert_array_equal(zero.N, np.zeros_like(field.N))
    with pytest.raises(ValueError, match="nonnegative"):
        project_normalized_angle_field(projector, -signal, normalization)

    clipped_grid = AngleBinGrid(
        two_theta_edges_rad=np.linspace(
            0.0,
            0.55 * full_grid.two_theta_edges_rad[-1],
            3,
        ),
        chi_raw_edges_rad=full_grid.chi_raw_edges_rad,
        revision="clipped-grid.v1",
    )
    clipped = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=clipped_grid,
    )
    clipped_field = project_normalized_angle_field(clipped, signal, normalization)
    assert np.any(clipped.lost_support > 0.0)
    clipped_dense = _dense_projector(clipped)
    oracle_column = _independent_pixel_weights(
        0,
        0,
        instrument=instrument,
        frame=frame,
        grid=clipped_grid,
    )
    np.testing.assert_allclose(
        clipped_dense[:, 0].reshape(clipped_grid.shape),
        oracle_column,
        rtol=3e-11,
        atol=3e-13,
    )
    assert clipped.lost_support[0, 0] == pytest.approx(
        1.0 - float(oracle_column.sum()),
        abs=3e-12,
    )
    assert float(clipped_dense[:, 0].sum()) < 1.0
    assert clipped_field.S.sum() + clipped_field.angular_lost_signal == pytest.approx(
        signal.sum(), abs=2e-12
    )
    assert clipped_field.N.sum() + clipped_field.angular_lost_normalization == pytest.approx(
        normalization.sum(), abs=2e-12
    )


def test_interactive_detector_texture_preserves_native_pixel_ownership() -> None:
    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    source = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float64)
    source_bytes = source.tobytes()
    texture = viewer["_prepare_full_native_texture"](source)
    assert texture.image_A2.dtype == np.float32
    assert texture.image_A2.flags.c_contiguous
    assert texture.image_A2.shape == source.shape
    assert texture.image_A2.size == source.size
    assert texture.image_A2[0, 0] == 1.0
    assert texture.image_A2[0, -1] == 3.0
    assert texture.image_A2[-1, 0] == 4.0
    assert texture.image_A2[-1, -1] == 6.0
    assert texture.high_A2 == 6.0
    assert texture.low_A2 == pytest.approx(6.0e-8)
    assert source.tobytes() == source_bytes
    quad = viewer["_FULL_SCREEN_TEXTURE_XY_UV"]
    assert quad[0, 1] == -1.0 and quad[0, 3] == 1.0
    assert quad[2, 1] == 3.0 and quad[2, 3] == -1.0
    shader = viewer["_full_screen_vertex_shader"]()
    assert "vec2(-1.0, -1.0)" in shader
    assert "vec2(0.0, -1.0)" in shader


def test_interactive_incidence_control_is_absolute_zero_to_twenty_degrees() -> None:
    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    incidence_field = "effective_incidence_angle_offset_deg"
    incidence_spec = next(
        spec for spec in viewer["_CONTROL_SPECS"] if spec.field_name == incidence_field
    )

    assert incidence_spec.label == r"effective incidence $\theta_i$ (deg)"
    assert (incidence_spec.minimum, incidence_spec.maximum) == (0.0, 20.0)
    to_deltas = viewer["_control_values_to_geometry_deltas"]
    for incidence_angle_deg, expected_offset_deg in (
        (0.0, -5.0),
        (5.0, 0.0),
        (20.0, 15.0),
    ):
        deltas = to_deltas(
            {incidence_field: incidence_angle_deg},
            configured_incidence_angle_deg=5.0,
        )
        assert deltas.effective_incidence_angle_offset_deg == expected_offset_deg


def test_interactive_render_scheduler_is_latest_only_and_progressive() -> None:
    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    RenderRequest = viewer["_RenderRequest"]
    RenderScheduler = viewer["_ProgressiveRenderScheduler"]
    GeometryDeltas = viewer["GeometryDeltas"]

    def request(revision: int) -> object:
        return RenderRequest(
            revision=revision,
            source_sample_count=7,
            draws_per_source_state=49,
            deltas=GeometryDeltas(detector_pitch_offset_deg=float(revision)),
        )

    scheduler = RenderScheduler()
    request_a, request_b, request_c = request(1), request(2), request(3)
    scheduler.submit(request_a, settled=True)
    stage_a = scheduler.start_next()
    assert stage_a is not None
    assert stage_a.draws_per_source_state == 1

    scheduler.submit(request_b, settled=True)
    scheduler.submit(request_c, settled=True)
    assert stage_a.cancellation.cancelled
    assert not scheduler.complete(stage_a)

    observed_draws = []
    while (stage := scheduler.start_next()) is not None:
        assert stage.request is request_c
        observed_draws.append(stage.draws_per_source_state)
        assert scheduler.complete(stage)
    assert observed_draws == [1, 4, 8, 49]

    scheduler.submit(request_c, settled=True)
    assert scheduler.start_next() is None

    request_d = request(4)
    scheduler.submit(request_d, settled=False)
    first_d = scheduler.start_next()
    assert first_d is not None
    assert first_d.request is request_d
    assert first_d.draws_per_source_state == 1
    scheduler.submit(request_d, settled=True)
    assert not first_d.cancellation.cancelled
    assert scheduler.complete(first_d)
    refinement = []
    while (stage := scheduler.start_next()) is not None:
        refinement.append((stage.draws_per_source_state, stage.materialize_result))
        assert scheduler.complete(stage)
    assert refinement == [(4, False), (8, False), (49, True)]

    request_e = request(5)
    scheduler.submit(request_e, settled=True)
    failed = scheduler.start_next()
    assert failed is not None
    assert scheduler.fail(failed)
    scheduler.submit(request_e, settled=True)
    retry = scheduler.start_next()
    assert retry is not None
    assert retry.draws_per_source_state == 1

    request_f = request(6)
    scheduler = RenderScheduler()
    scheduler.submit(request_f, settled=True)
    for expected_draw_count in (1, 4, 8):
        stage = scheduler.start_next()
        assert stage is not None
        assert stage.draws_per_source_state == expected_draw_count
        assert scheduler.complete(stage)
    cancelled_final = scheduler.start_next()
    assert cancelled_final is not None
    assert cancelled_final.draws_per_source_state == 49
    scheduler.cancel()
    scheduler.reset_latest()
    scheduler.submit(request_f, settled=True)
    assert cancelled_final.cancellation.cancelled
    assert not scheduler.complete(cancelled_final)
    restarted = scheduler.start_next()
    assert restarted is not None
    assert restarted.draws_per_source_state == 1


def test_interactive_detector_only_change_reuses_incident_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    bundle = viewer["_build_bundle"](
        config,
        config.source.minimum_physical_sample_count,
    )
    detector_for_deltas = viewer["_detector_for_deltas"]

    def fail_incident_rebuild(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("detector-only controls must reuse incident transport")

    original_incident_builder = detector_for_deltas.__globals__["build_incident_states"]
    detector_for_deltas.__globals__["build_incident_states"] = fail_incident_rebuild
    detector = detector_for_deltas(
        bundle,
        viewer["GeometryDeltas"](
            detector_pitch_offset_deg=0.25,
            detector_column_translation_mm=0.5,
        ),
    )

    assert detector.incident is bundle.inputs.incident
    assert not np.array_equal(
        detector.instrument.lab_from_detector.rotation,
        bundle.inputs.instrument.lab_from_detector.rotation,
    )
    scopes_by_field = {
        spec.field_name: spec.invalidation_scope for spec in viewer["_CONTROL_SPECS"]
    }
    expected_detector_only = {
        "detector_pitch_offset_deg",
        "detector_yaw_offset_deg",
        "detector_in_plane_rotation_offset_deg",
        "detector_column_translation_mm",
        "detector_row_translation_mm",
        "detector_distance_offset_mm",
    }
    assert {
        name for name, scope in scopes_by_field.items() if scope == "detector"
    } == expected_detector_only
    assert {name for name, scope in scopes_by_field.items() if scope == "incident"} == set(
        scopes_by_field
    ) - expected_detector_only
    previous = viewer["GeometryDeltas"](
        effective_incidence_angle_offset_deg=0.2,
        sample_in_plane_x_translation_mm=0.4,
    )
    detector_revision = replace(previous, detector_pitch_offset_deg=0.3)
    assert viewer["_changed_delta_fields"](previous, detector_revision) == {
        "detector_pitch_offset_deg"
    }
    assert (
        viewer["_changed_delta_fields"](
            previous,
            detector_revision,
        )
        <= viewer["_DETECTOR_ONLY_DELTA_FIELDS"]
    )

    detector_for_deltas.__globals__["build_incident_states"] = original_incident_builder
    session = viewer["_DetectorRenderSession"](
        config,
        detector_seed=7,
        execution_backend="cpu",
        prepare_texture=False,
    )
    session._bundle = bundle
    physical_source_count = config.source.minimum_physical_sample_count

    def stage(revision: int, deltas: object) -> object:
        request = viewer["_RenderRequest"](
            revision=revision,
            source_sample_count=physical_source_count,
            draws_per_source_state=1,
            deltas=deltas,
        )
        return viewer["_ScheduledRender"](
            request=request,
            draws_per_source_state=1,
            materialize_result=True,
            cancellation=viewer["_RenderCancellation"](),
        )

    sample_corrected = viewer["GeometryDeltas"](
        sample_in_plane_x_translation_mm=0.1,
    )
    session.render(stage(1, sample_corrected), stop_requested=viewer["threading"].Event())
    detector_for_deltas.__globals__["build_incident_states"] = fail_incident_rebuild
    mixed_revision = replace(sample_corrected, detector_pitch_offset_deg=0.2)
    cancelled_revision = stage(2, mixed_revision)
    sampler_type = type(session._sampler)
    original_pose_rebind = sampler_type.rebind_detector_pose

    def rebind_then_cancel(sampler: object, instrument: object) -> None:
        original_pose_rebind(sampler, instrument)
        cancelled_revision.cancellation.cancel()

    monkeypatch.setattr(sampler_type, "rebind_detector_pose", rebind_then_cancel)
    with pytest.raises(viewer["MonteCarloSamplingCancelled"], match="cancelled"):
        session.render(
            cancelled_revision,
            stop_requested=viewer["threading"].Event(),
        )
    assert session._bound_deltas == mixed_revision
    session.reset_after_cancellation()

    pose_rebind_count = 0

    def count_pose_rebind(sampler: object, instrument: object) -> None:
        nonlocal pose_rebind_count
        pose_rebind_count += 1
        original_pose_rebind(sampler, instrument)

    monkeypatch.setattr(sampler_type, "rebind_detector_pose", count_pose_rebind)
    session.render(stage(3, sample_corrected), stop_requested=viewer["threading"].Event())
    assert pose_rebind_count == 1


def test_interactive_detector_viewer_requires_all_m_catalogue() -> None:
    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    without_m0 = replace(
        config,
        bragg=replace(config.bragg, include_detector_visible_m0=False),
    )

    with pytest.raises(ValueError, match="all-m display"):
        viewer["_build_bundle"](without_m0, 1)


def test_interactive_detector_viewer_reenumerates_rods_after_validity_change() -> None:
    from painted_ewald import enumerate_rods_within_ewald_sphere

    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    config = replace(
        config,
        source=replace(
            config.source,
            wavelength_model_id="gaussian.v1",
            line_wavelength_A=(),
            line_probability=(),
            common_line_sigma_A=0.0,
            wavelength_sigma_A=0.4,
        ),
        instrument=replace(
            config.instrument,
            sample_support_model_id="finite_rectangle.v1",
            sample_width_m=0.0002,
            sample_length_m=0.0002,
        ),
    )
    bundle = viewer["_build_bundle"](config, 25)
    assert bundle.detector.source_state_count == 25
    assert bundle.inputs.config.source.sample_count == 25
    states = bundle.inputs.incident.states
    base_minimum_wavelength_A = float(np.min(states.wavelength_A[states.valid]))
    base_rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=bundle.inputs.reciprocal.basis_Ainv,
        k_norm_Ainv=2.0 * np.pi / base_minimum_wavelength_A,
        population=config.bragg.rod_population,
    )
    assert bundle.detector.rods == base_rods

    deltas = viewer["GeometryDeltas"](
        goniometer_axis_pitch_offset_deg=-2.0,
        sample_in_plane_y_translation_mm=1.0,
    )
    changed_instrument = viewer["apply_geometry_deltas"](
        bundle.inputs.instrument,
        deltas,
        configured_axis_rotations=bundle.inputs.config.instrument.axis_rotations,
    )
    changed_incident = build_incident_states(
        bundle.inputs.samples,
        bundle.inputs.material,
        changed_instrument,
    )
    assert not np.array_equal(changed_incident.states.valid, states.valid)
    changed_minimum_wavelength_A = float(
        np.min(changed_incident.states.wavelength_A[changed_incident.states.valid])
    )
    changed_rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=bundle.inputs.reciprocal.basis_Ainv,
        k_norm_Ainv=2.0 * np.pi / changed_minimum_wavelength_A,
        population=config.bragg.rod_population,
    )
    assert changed_minimum_wavelength_A < base_minimum_wavelength_A
    assert len(changed_rods) > len(base_rods)
    assert {(rod.h, rod.k) for rod in changed_rods} <= {
        (rod.h, rod.k) for rod in bundle.inputs.rods
    }

    detector = viewer["_detector_for_deltas"](bundle, deltas)
    assert len(detector.rods) == len(changed_rods)


def test_interactive_geometry_deltas_use_domain_names_and_apply_base_local_pose() -> None:
    from dataclasses import fields

    from rasim_next.geometry.instrument import CompiledInstrument

    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    GeometryDeltas = viewer["GeometryDeltas"]
    apply_geometry_deltas = viewer["apply_geometry_deltas"]

    expected_names = (
        "detector_pitch_offset_deg",
        "detector_yaw_offset_deg",
        "detector_in_plane_rotation_offset_deg",
        "detector_column_translation_mm",
        "detector_row_translation_mm",
        "detector_distance_offset_mm",
        "goniometer_axis_pitch_offset_deg",
        "goniometer_axis_yaw_offset_deg",
        "effective_incidence_angle_offset_deg",
        "effective_sample_tilt_offset_deg",
        "sample_in_plane_rotation_offset_deg",
        "sample_in_plane_x_translation_mm",
        "sample_in_plane_y_translation_mm",
        "sample_normal_translation_mm",
    )
    assert tuple(item.name for item in fields(GeometryDeltas)) == expected_names
    assert tuple(spec.field_name for spec in viewer["_CONTROL_SPECS"]) == expected_names
    labels_by_name = {spec.field_name: spec.label for spec in viewer["_CONTROL_SPECS"]}
    required_label_terms = {
        "detector_pitch_offset_deg": r"-\Delta\gamma_{\rm RA}",
        "detector_yaw_offset_deg": r"\Delta\Gamma_{\rm RA}",
        "detector_in_plane_rotation_offset_deg": r"\Delta\chi_D",
        "detector_column_translation_mm": r"\Delta x_D",
        "detector_row_translation_mm": r"\Delta y_D",
        "detector_distance_offset_mm": r"\Delta D_n",
        "goniometer_axis_pitch_offset_deg": r"\Delta\alpha$ [RA-SIM cor_angle]",
        "goniometer_axis_yaw_offset_deg": r"\Delta\psi_g$ [RA-SIM psi_z]",
        "effective_incidence_angle_offset_deg": r"\theta_i",
        "effective_sample_tilt_offset_deg": r"\Delta\delta",
        "sample_in_plane_rotation_offset_deg": r"RA-SIM $-\Delta\psi$",
        "sample_in_plane_x_translation_mm": r"\Delta x_S",
        "sample_in_plane_y_translation_mm": r"\Delta y_S",
        "sample_normal_translation_mm": r"\Delta n_S=-\Delta z_S",
    }
    assert labels_by_name.keys() == required_label_terms.keys()
    for name, required_term in required_label_terms.items():
        assert required_term in labels_by_name[name]

    identity = np.eye(3)
    detector_rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    instrument = CompiledInstrument(
        lab_from_sample=RigidTransform(
            identity,
            np.array((0.1, 0.2, 0.3)),
            FrameId.SAMPLE,
            FrameId.LAB,
        ),
        sample_from_crystal=RigidTransform(
            identity,
            np.zeros(3),
            FrameId.CRYSTAL,
            FrameId.SAMPLE,
        ),
        lab_from_detector=RigidTransform(
            detector_rotation,
            np.array((0.4, 0.5, 0.6)),
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=(6, 10),
        detector_row_pitch_m=2.0e-4,
        detector_column_pitch_m=1.0e-4,
        detector_reference_coordinate_px=(4.0, 2.0),
        sample_support_model_id="unbounded_plane.v1",
        sample_width_m=None,
        sample_length_m=None,
        film_thickness_A=500.0,
    )

    zero = apply_geometry_deltas(instrument, GeometryDeltas.zero())
    np.testing.assert_array_equal(
        zero.lab_from_sample.rotation,
        instrument.lab_from_sample.rotation,
    )
    np.testing.assert_array_equal(
        zero.lab_from_detector.translation_m,
        instrument.lab_from_detector.translation_m,
    )

    detector_angles_deg = (7.0, -11.0, 13.0)
    detector_translation_mm = (2.0, -3.0, 4.0)
    sample_angles_deg = (-5.0, 6.0, -9.0)
    sample_translation_mm = (-1.5, 3.0, 0.75)
    changed = apply_geometry_deltas(
        instrument,
        GeometryDeltas(
            detector_pitch_offset_deg=detector_angles_deg[0],
            detector_yaw_offset_deg=detector_angles_deg[1],
            detector_in_plane_rotation_offset_deg=detector_angles_deg[2],
            detector_column_translation_mm=detector_translation_mm[0],
            detector_row_translation_mm=detector_translation_mm[1],
            detector_distance_offset_mm=detector_translation_mm[2],
            effective_incidence_angle_offset_deg=sample_angles_deg[0],
            effective_sample_tilt_offset_deg=sample_angles_deg[1],
            sample_in_plane_rotation_offset_deg=sample_angles_deg[2],
            sample_in_plane_x_translation_mm=sample_translation_mm[0],
            sample_in_plane_y_translation_mm=sample_translation_mm[1],
            sample_normal_translation_mm=sample_translation_mm[2],
        ),
    )

    def intrinsic_xyz_deg(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
        x_rad, y_rad, z_rad = np.radians((x_deg, y_deg, z_deg))
        cx, sx = np.cos(x_rad), np.sin(x_rad)
        cy, sy = np.cos(y_rad), np.sin(y_rad)
        cz, sz = np.cos(z_rad), np.sin(z_rad)
        rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
        rotation_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
        rotation_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
        return rotation_x @ rotation_y @ rotation_z

    np.testing.assert_allclose(
        changed.lab_from_detector.rotation,
        detector_rotation @ intrinsic_xyz_deg(*detector_angles_deg),
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        changed.lab_from_detector.translation_m,
        instrument.lab_from_detector.translation_m
        + detector_rotation @ (1.0e-3 * np.asarray(detector_translation_mm)),
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        changed.lab_from_sample.rotation,
        intrinsic_xyz_deg(*sample_angles_deg),
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        changed.lab_from_sample.translation_m,
        instrument.lab_from_sample.translation_m + 1.0e-3 * np.asarray(sample_translation_mm),
        rtol=0.0,
        atol=2.0e-16,
    )
    assert changed.detector_shape_rc == instrument.detector_shape_rc
    assert changed.detector_reference_coordinate_px == instrument.detector_reference_coordinate_px


def test_interactive_goniometer_axis_deltas_rebuild_the_pivoted_commanded_motion() -> None:
    from rasim_next.geometry import AxisRotation
    from rasim_next.pipeline.configured_simulation import AxisRotationConfiguration

    viewer = runpy.run_path(DETECTOR_VIEWER_SCRIPT)
    GeometryDeltas = viewer["GeometryDeltas"]
    apply_geometry_deltas = viewer["apply_geometry_deltas"]

    def rotation_y(angle_rad: float) -> np.ndarray:
        cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
        return np.array(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))

    def rotation_z(angle_rad: float) -> np.ndarray:
        cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
        return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))

    def rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        x_axis, y_axis, z_axis = axis
        cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
        complement = 1.0 - cosine
        return np.array(
            (
                (
                    cosine + x_axis * x_axis * complement,
                    x_axis * y_axis * complement - z_axis * sine,
                    x_axis * z_axis * complement + y_axis * sine,
                ),
                (
                    y_axis * x_axis * complement + z_axis * sine,
                    cosine + y_axis * y_axis * complement,
                    y_axis * z_axis * complement - x_axis * sine,
                ),
                (
                    z_axis * x_axis * complement - y_axis * sine,
                    z_axis * y_axis * complement + x_axis * sine,
                    cosine + z_axis * z_axis * complement,
                ),
            )
        )

    commanded_angle_deg = 12.0
    pivot_lab_m = np.array((0.04, -0.02, 0.01))
    zero_rotation = rotation_z(math.radians(17.0))
    zero_translation_m = np.array((0.02, -0.01, 0.03))
    mount_rotation = rotation_y(math.radians(-8.0))
    mount_translation_m = np.array((0.004, 0.003, -0.002))
    base_pitch_deg = -0.8
    base_yaw_deg = 1.1
    base_pitch_rad = math.radians(base_pitch_deg)
    base_yaw_rad = math.radians(base_yaw_deg)
    base_axis = np.array(
        (
            math.cos(base_pitch_rad) * math.cos(base_yaw_rad),
            -math.cos(base_pitch_rad) * math.sin(base_yaw_rad),
            math.sin(base_pitch_rad),
        )
    )
    base_axis_rotation = AxisRotation(
        axis_lab=base_axis,
        angle_rad=math.radians(commanded_angle_deg),
        pivot_lab_m=pivot_lab_m,
    )
    base_configuration = InstrumentConfiguration(
        axis_rotations=(base_axis_rotation,),
        lab_from_goniometer_zero=RigidTransform(
            zero_rotation,
            zero_translation_m,
            FrameId.GONIOMETER,
            FrameId.LAB,
        ),
        goniometer_from_sample=RigidTransform(
            mount_rotation,
            mount_translation_m,
            FrameId.SAMPLE,
            FrameId.GONIOMETER,
        ),
        sample_from_crystal=RigidTransform(
            np.eye(3),
            np.zeros(3),
            FrameId.CRYSTAL,
            FrameId.SAMPLE,
        ),
        lab_from_detector=RigidTransform(
            np.eye(3),
            np.array((0.0, 0.1, 0.0)),
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=(6, 10),
        detector_row_pitch_m=2.0e-4,
        detector_column_pitch_m=1.0e-4,
        detector_reference_coordinate_px=(4.0, 2.0),
        sample_support_model_id="unbounded_plane.v1",
        sample_width_m=None,
        sample_length_m=None,
        film_thickness_A=500.0,
    )
    base_instrument = compile_instrument(base_configuration)
    configured_axes = (
        AxisRotationConfiguration(
            axis_lab=tuple(base_axis),
            angle_deg=commanded_angle_deg,
            pivot_lab_m=tuple(pivot_lab_m),
        ),
    )

    pitch_deg = 1.5
    yaw_deg = 0.7
    changed = apply_geometry_deltas(
        base_instrument,
        GeometryDeltas(
            goniometer_axis_pitch_offset_deg=pitch_deg,
            goniometer_axis_yaw_offset_deg=yaw_deg,
        ),
        configured_axis_rotations=configured_axes,
    )
    pitch_rad = math.radians(base_pitch_deg + pitch_deg)
    yaw_rad = math.radians(base_yaw_deg + yaw_deg)
    corrected_axis = np.array(
        (
            math.cos(pitch_rad) * math.cos(yaw_rad),
            -math.cos(pitch_rad) * math.sin(yaw_rad),
            math.sin(pitch_rad),
        )
    )
    commanded_rotation = rodrigues(corrected_axis, math.radians(commanded_angle_deg))
    mounted_rotation = zero_rotation @ mount_rotation
    mounted_translation_m = zero_translation_m + zero_rotation @ mount_translation_m
    expected_rotation = commanded_rotation @ mounted_rotation
    expected_translation_m = (
        pivot_lab_m - commanded_rotation @ pivot_lab_m + commanded_rotation @ mounted_translation_m
    )
    np.testing.assert_allclose(
        changed.lab_from_sample.rotation,
        expected_rotation,
        rtol=0.0,
        atol=4.0e-16,
    )
    np.testing.assert_allclose(
        changed.lab_from_sample.translation_m,
        expected_translation_m,
        rtol=0.0,
        atol=4.0e-16,
    )
    np.testing.assert_array_equal(
        changed.lab_from_detector.rotation,
        base_instrument.lab_from_detector.rotation,
    )

    zero_command_configuration = replace(
        base_configuration,
        axis_rotations=(replace(base_axis_rotation, angle_rad=0.0),),
    )
    zero_command_instrument = compile_instrument(zero_command_configuration)
    zero_command_changed = apply_geometry_deltas(
        zero_command_instrument,
        GeometryDeltas(
            goniometer_axis_pitch_offset_deg=pitch_deg,
            goniometer_axis_yaw_offset_deg=yaw_deg,
        ),
        configured_axis_rotations=(replace(configured_axes[0], angle_deg=0.0),),
    )
    np.testing.assert_array_equal(
        zero_command_changed.lab_from_sample.rotation,
        zero_command_instrument.lab_from_sample.rotation,
    )
    np.testing.assert_array_equal(
        zero_command_changed.lab_from_sample.translation_m,
        zero_command_instrument.lab_from_sample.translation_m,
    )

    with pytest.raises(ValueError, match="exactly one configured goniometer axis"):
        apply_geometry_deltas(
            base_instrument,
            GeometryDeltas(goniometer_axis_pitch_offset_deg=1.0),
            configured_axis_rotations=configured_axes * 2,
        )


@pytest.mark.parametrize("quadrature_kind", ("sobol", "composite_gauss"))
def test_joint_fiber_quadrature_integrates_nonseparable_continuous_field(quadrature_kind) -> None:
    from scipy.integrate import quad

    from painted_ewald import MosaicParameters
    from rasim_next.pipeline.fiber_detector import sample_conditional_fiber_coordinates

    arguments = dict(
        axial_bounds_Ainv=(0.2, 1.0),
        axial_peak_centers_Ainv=[0.4, 0.7],
        axial_peak_half_width_Ainv=0.2,
        radial_Ainv=0.3,
        ki_sample_Ainv=[0, 0, -2],
        normal_sample=[0, 0, 1],
        source_region_bounds=[[0, 4, 0, 2 * np.pi]],
        reference_mosaic=MosaicParameters(0.4, 0.7, 0.3),
        axial_power=12 if quadrature_kind == "sobol" else 8,
        angular_power=5 if quadrature_kind == "sobol" else 7,
        axial_seed=66546,
        angular_shift_seed=1104,
        quadrature_kind=quadrature_kind,
    )
    nodes = sample_conditional_fiber_coordinates(**arguments)
    axial = nodes.positive_axial_Ainv[nodes.axial_index]
    actual = nodes.weight_Ainv_rad @ np.exp(axial * nodes.ewald_azimuth_rad / (2 * np.pi))
    expected = 2 * np.pi * quad(lambda t: np.expm1(t) / t, 0.2, 1.0, epsabs=1e-13)[0]
    assert actual == pytest.approx(expected, rel=2e-5)
    assert nodes.weight_Ainv_rad.sum() == pytest.approx(0.8 * 2 * np.pi, rel=1e-7)
    assert not nodes.positive_axial_Ainv.flags.writeable
    repeated = sample_conditional_fiber_coordinates(**arguments)
    np.testing.assert_array_equal(repeated.weight_Ainv_rad, nodes.weight_Ainv_rad)
    empty = sample_conditional_fiber_coordinates(
        **(arguments | {"source_region_bounds": [[3.0, 4.0, 0, 2 * np.pi]]})
    )
    assert not len(empty.weight_Ainv_rad)
    np.testing.assert_array_equal(empty.positive_axial_Ainv, nodes.positive_axial_Ainv)
    if quadrature_kind == "composite_gauss":
        # A narrow native-support feature lies between coarse proposal peaks.
        # Physical panel refinement must recover its integral without a new PDF.
        refined = sample_conditional_fiber_coordinates(
            **(arguments | {"axial_power": 3, "maximum_axial_panel_width_Ainv": 0.02})
        )
        t = refined.positive_axial_Ainv[refined.axial_index]
        narrow = np.exp(-0.5 * ((t - 0.55) / 0.008) ** 2)
        assert refined.weight_Ainv_rad @ narrow == pytest.approx(
            2 * np.pi * np.sqrt(2 * np.pi) * 0.008, rel=1e-7
        )
        assert np.max(np.diff(refined.positive_axial_Ainv)) < 0.02
        split = sample_conditional_fiber_coordinates(
            **(
                arguments
                | {
                    "source_region_bounds": [[0, 4, 0, 0.4], [0, 4, 5, 0.2], [0, 4, 6.1, 0.3]],
                }
            )
        )
        arcs = [(0, 0.4), (5, 5.2), (6.1, 2 * np.pi)]
        expected = sum(
            quad(
                lambda t, left=left, right=right: (
                    2
                    * np.pi
                    / t
                    * (np.exp(t * right / (2 * np.pi)) - np.exp(t * left / (2 * np.pi)))
                ),
                0.2,
                1.0,
                epsabs=1e-13,
            )[0]
            for left, right in arcs
        )
        axial = split.positive_axial_Ainv[split.axial_index]
        actual = split.weight_Ainv_rad @ np.exp(axial * split.ewald_azimuth_rad / (2 * np.pi))
        assert actual == pytest.approx(expected, rel=1e-10)
        assert split.weight_Ainv_rad.sum() == pytest.approx(
            0.8 * sum(b - a for a, b in arcs), rel=1e-10
        )
        changed_shift = sample_conditional_fiber_coordinates(
            **(arguments | {"angular_shift_seed": 90210})
        )
        np.testing.assert_array_equal(changed_shift.ewald_azimuth_rad, nodes.ewald_azimuth_rad)


def test_fixed_angular_support_preserves_continuity_across_observation_q_bounds() -> None:
    from painted_ewald import MosaicParameters
    from rasim_next.pipeline.fiber_detector import sample_conditional_fiber_coordinates

    center = np.sqrt(0.6**2 - 0.3**2)
    arguments = dict(
        axial_bounds_Ainv=(center - 1e-9, center + 1e-9),
        axial_peak_centers_Ainv=[center],
        axial_peak_half_width_Ainv=0.02,
        radial_Ainv=0.3,
        ki_sample_Ainv=[0, 0, -2],
        normal_sample=[0, 0, 1],
        source_region_bounds=[[0, 2, 0, 1.3], [0.6, 2, 0.9, 1.4]],
        reference_mosaic=MosaicParameters(0.4, 0.7, 0.3),
        axial_power=3,
        angular_power=4,
        axial_seed=0,
        angular_shift_seed=0,
        quadrature_kind="composite_gauss",
        angular_support="fixed_union",
    )
    nodes = sample_conditional_fiber_coordinates(**arguments)
    acceptance = np.exp(-0.5 * ((nodes.ewald_azimuth_rad - 0.6) / 0.03) ** 2)
    mass = np.bincount(nodes.axial_index, weights=nodes.weight_Ainv_rad)
    moment = np.bincount(nodes.axial_index, weights=nodes.weight_Ainv_rad * acceptance) / mass
    assert np.ptp(moment) < 1e-8
    # Continuity alone does not certify accuracy; refine against an analytic integral.
    refined = sample_conditional_fiber_coordinates(**(arguments | {"angular_power": 8}))
    lower, upper = arguments["axial_bounds_Ainv"]
    assert refined.weight_Ainv_rad.sum() == pytest.approx(2.3 * (upper - lower), rel=1e-8)
    acceptance = np.exp(-0.5 * ((refined.ewald_azimuth_rad - 0.6) / 0.03) ** 2)
    average = (refined.weight_Ainv_rad @ acceptance) / refined.weight_Ainv_rad.sum()
    assert average == pytest.approx(np.sqrt(2 * np.pi) * 0.03 / 2.3, rel=1e-8)
