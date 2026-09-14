"""Spatial probability, exact ray transport and raw detector-mass contracts."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry.detector import _intersect_detector_plane
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.sample import _intersect_sample_rays
from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
from rasim_next.pipeline.source_spatial import (
    DetectorSpatialKernels,
    NativeSpatialRegionProjection,
    compile_conditional_spatial_kernels,
)
from rasim_next.sampling.source import (
    quadrature_conditional_gaussian_source,
    sample_conditional_gaussian_source,
)


def configuration():
    angle = 0.23
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    instrument = CompiledInstrument(
        lab_from_sample=RigidTransform(np.eye(3), np.zeros(3), FrameId.SAMPLE, FrameId.LAB),
        sample_from_crystal=RigidTransform(np.eye(3), np.zeros(3), FrameId.CRYSTAL, FrameId.SAMPLE),
        lab_from_detector=RigidTransform(
            rotation, np.array([0.0, 0.0, 0.20]), FrameId.DETECTOR, FrameId.LAB
        ),
        detector_shape_rc=(80, 90),
        detector_row_pitch_m=0.0008,
        detector_column_pitch_m=0.001,
        detector_reference_coordinate_px=(44.0, 39.0),
        sample_support_model_id="unbounded_plane.v1",
        sample_width_m=None,
        sample_length_m=None,
        film_thickness_A=500.0,
    )
    outgoing = np.array([[0.02, 0.01, 1.0], [0.4, -0.2, 1.0]])
    outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
    return dict(
        instrument=instrument,
        source=sample_conditional_gaussian_source(
            mean_origin_lab_m=[0.0, 0.0, 0.05],
            mean_direction_lab=[0.0, 0.0, -1.0],
            transverse_axes_lab=np.eye(3)[:2],
            spatial_sigma_m=[0.001, 0.002],
            divergence_sigma_rad=[0.08, 0.04],
            position_divergence_correlation=[-0.3, 0.4],
            line_wavelength_A=[1.54, 1.544],
            line_probability=[0.66, 0.34],
            common_wavelength_sigma_A=0.0,
            sample_count=12,
            seed=723,
            polarization_state_id="unpolarized.v1",
        ),
        source_state_index=0,
        outgoing_direction_lab=outgoing,
        maximum_backward_probability=1e-12,
    )


def test_native_physics_input_preserves_expanded_sites_and_stacking_averages(tmp_path):
    import json
    from dataclasses import fields
    from pathlib import Path

    from painted_ewald import MosaicParameters
    from rasim_next.fitting.native_input import load_native_fit_physics
    from rasim_next.materials import read_crystal
    from rasim_next.pipeline.fiber_detector import FiberIntegrationRule
    from rasim_next.stacking import Parent

    def values(item):
        return {field.name: getattr(item, field.name) for field in fields(item) if field.init}

    crystal = read_crystal(
        Path(__file__).resolve().parents[1] / "examples/pbi2/structures/PbI2_2H.cif",
        phase_id="PbI2",
    )
    crystal = replace(
        crystal,
        source_path=tmp_path / "nonexistent_provenance.cif",
        sites=tuple(replace(site, u_iso_A2=0.027) for site in crystal.sites),
    )
    crystal_record = values(crystal)
    crystal_record["sites"] = [values(site) for site in crystal.sites]
    instrument = values(configuration()["instrument"])
    for name in ("lab_from_sample", "sample_from_crystal", "lab_from_detector"):
        instrument[name] = values(instrument[name])
    record = dict(
        schema="rasim-native-fit-physics-v1",
        sample_id="boundary-proof",
        instrument=instrument,
        material=dict(
            material_id="Pb-proof",
            wavelength_A=[1.54, 1.544],
            n_real=[0.99994, 0.99993],
            n_imag=[1e-6, 2e-6],
            provenance="boundary-proof",
        ),
        source=dict(
            mean_origin_lab_m=[0.0, 0.0, 0.05],
            mean_direction_lab=[0.0, 0.0, -1.0],
            transverse_axes_lab=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            spatial_sigma_m=[0.001, 0.002],
            divergence_sigma_rad=[0.002, 0.003],
            position_divergence_correlation=[-0.3, 0.4],
            line_wavelength_A=[1.54, 1.544],
            line_probability=[0.66, 0.34],
            common_wavelength_sigma_A=0.0,
            polarization_state_id="UNITY_APPROXIMATION",
        ),
        source_rule=dict(kind="gauss_hermite", divergence_order=2),
        rods=[dict(h=0, k=0, population=0.7), dict(h=1, k=0, population=0.9)],
        rod_catalog_revision="boundary-proof",
        reciprocal_basis_Ainv=2 * np.pi * np.linalg.inv(crystal.direct_basis_A).T,
        crystal_to_sample=np.eye(3),
        structure=dict(
            crystals=[crystal_record],
            normalization="FINITE_TOTAL",
            unknown_u_iso_A2=None,
            initial_population=dict(plus=0.3, minus=0.7),
            stacking_phases=[
                dict(
                    fault_parameter="ordered",
                    parents=[Parent.FOUR_H_PLUS, Parent.SIX_H_MINUS],
                    weights=[0.25, 0.75],
                    average_transition_laws=False,
                ),
                dict(
                    fault_parameter="mixed",
                    parents=[Parent.TWO_H, Parent.FOUR_H_PLUS, Parent.SIX_H_PLUS],
                    weights=[0.2, 0.3, 0.5],
                    average_transition_laws=True,
                ),
            ],
        ),
        integration_rule=values(FiberIntegrationRule(axial_power=4, angular_power=1)),
        spatial_quadrature_order=8,
        phase_population_weight=0.4,
        polarization_weight=0.5,
        specular_stitch_stack=None,
    )
    path = tmp_path / "physics.json"

    def write():
        path.write_text(
            json.dumps(
                record, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)
            ),
            encoding="utf-8",
        )

    write()
    physics = load_native_fit_physics(path)
    assert len(physics.source.mean_rays.wavelength_A) == 8
    assert physics.structure.crystals[0].source_path == crystal.source_path
    assert all(site.u_iso_A2 == 0.027 for site in physics.structure.crystals[0].sites)
    detector = physics.detector(
        mosaic=MosaicParameters(0.1, 0.2, 0.3),
        coherent_repeats=8,
        film_thickness_A=500.0,
        surface_fractions=(0.2, 0.3, 0.5),
        phase_fractions=(0.6, 0.4),
        fault_parameters=dict(ordered=0.08, mixed=0.16),
    )
    strength = detector.strength_model
    np.testing.assert_allclose(strength.population_fractions, [0.15, 0.45, 0.4], rtol=0, atol=1e-16)
    np.testing.assert_allclose(
        [p.model.as_array() for p in strength.populations],
        [
            [0.02, 0.02, 0.02, 0.92, 0.02],
            [0.02, 0.02, 0.92, 0.02, 0.02],
            [0.20, 0.44, 0.04, 0.28, 0.04],
        ],
        rtol=0,
        atol=1e-15,
    )
    assert all(p.initial.plus == 0.3 and p.initial.minus == 0.7 for p in strength.populations)
    np.testing.assert_array_equal(detector.material.n_complex, [0.99994 + 1e-6j, 0.99993 + 2e-6j])
    assert detector.phase_population_weight == 0.4 and detector.polarization_weight == 0.5
    assert tuple(r.population for r in detector.rods) == (0.7, 0.9)
    # The complete Pb candidate and acquisition adapter reuse these exact input
    # owners; this also checks that source definitions survive numeric loading.
    from test_native_search import observations as make_observations

    from rasim_next.fitting.native_instrument import NativeInstrumentModel
    from rasim_next.fitting.native_joint import NativeJointEvaluator
    from rasim_next.fitting.pb_native import (
        PbCellSiteParameters,
        PbJointModel,
        PbNativeStructureModel,
    )
    from rasim_next.materials.optics import material_optics
    from rasim_next.measurement.continuous_regions import NativePixelRegionProjection

    physics = replace(
        physics, material=material_optics(crystal, physics.source.mean_rays.wavelength_A)
    )
    atomic = PbNativeStructureModel(physics)
    model = PbJointModel(atomic)
    seed = dict(
        coherent_repeats=8,
        film_thickness_nm=50.0,
        gaussian_sigma_deg=6.0,
        lorentzian_hwhm_deg=12.0,
        eta=0.3,
        surface_fractions=(0.2, 0.3, 0.5),
        phase_fractions=(0.6, 0.4),
        fault_parameters=dict(ordered=0.08, mixed=0.16),
    )
    initial = model.initial_values(seed)
    bound, arguments, _, _ = model.bind(initial, 8)
    np.testing.assert_allclose(
        bound.structure.strength(
            **{k: v for k, v in arguments.items() if k != "film_thickness_A"}
        ).evaluate_hkl(h=[0, 1, -1], k=[0, 0, 1], L=[0.37, 1.27, -2.38], k_norm_Ainv=4.08),
        strength.evaluate_hkl(h=[0, 1, -1], k=[0, 0, 1], L=[0.37, 1.27, -2.38], k_norm_Ainv=4.08),
        rtol=3e-13,
    )
    for i, delta in enumerate((0.001, 0.002, 0.0001, -0.01, -0.01, 0.003, 0.003, 0.003, 0.003)):
        trial = atomic.reference_parameters.as_array()
        trial[i] += delta
        rebound = atomic.bind(PbCellSiteParameters.from_array(trial))
        assert (not np.array_equal(rebound.material.n_complex, bound.material.n_complex)) == (
            i in (0, 1, 3, 4)
        )
        for site, base in zip(rebound.structure.crystals[0].sites, crystal.sites, strict=True):
            if i != 2:
                assert site.fractional == base.fractional
            elif base.element == "I":
                assert site.fractional[2] - base.fractional[2] == pytest.approx(
                    delta if base.fractional[2] < 0.5 else -delta, abs=1e-15
                )
    with pytest.raises(ValueError, match="smaller"):
        model.initial_values(dict(seed, film_thickness_nm=1.0))
    shape = physics.instrument.detector_shape_rc
    flat = np.arange(np.prod(shape))
    projection = NativePixelRegionProjection(
        shape,
        flat,
        (((flat % shape[1] - 44) ** 2 + (flat // shape[1] - 39) ** 2) >= 20**2).astype(int),
        flat,
        np.ones(len(flat)),
        2,
        "joint-pb-proof",
    )
    obs = replace(make_observations([1, 1]), projection=projection)
    instrument_model = NativeInstrumentModel(physics, "test-acquisition")
    evaluator = NativeJointEvaluator(
        model, obs, MosaicParameters(0.1, 0.2, 0.3), instrument_model, 2
    )
    coordinates = np.r_[initial, instrument_model.initial_values]
    predicted = evaluator.predict(coordinates, 8)
    assert np.any(predicted > 0)
    inactive_probe = coordinates.copy()
    inactive_probe[11], inactive_probe[12], inactive_probe[16] = 0, 1, 1
    inactive = evaluator.inactive_parameters(inactive_probe, 8)
    assert set(inactive) == {
        "lorentzian_half_width_rad",
        "surface_1_share_of_remainder",
        "phase_1_epsilon",
        "phase_1_parent_0_share_of_remainder",
        "phase_1_parent_1_share_of_remainder",
    }
    with pytest.raises(ValueError, match="aligned"):
        evaluator.predict(coordinates.reshape(1, -1), 8)
    with pytest.raises(ValueError, match="aligned"):
        evaluator.predict(coordinates.astype(complex) + 1j, 8)
    coordinates[14] += 10
    thickness = evaluator.predict(coordinates, 8)
    assert evaluator.compile_count == 1
    fresh = NativeJointEvaluator(model, obs, evaluator.proposal_mosaic, instrument_model, 2)
    np.testing.assert_allclose(thickness, fresh.predict(coordinates, 8), rtol=3e-13)
    # Representative changes for each distinct geometry/source dependency.
    for i, delta in (
        (0, 0.2),
        (2, 0.0001),
        (5, 0.0002),
        (7, 0.000005),
        (8, 0.00003),
        (10, 0.0001),
        (12, 0.03),
        (14, 0.01),
        (15, 0.00002),
        (16, 0.00001),
    ):
        coordinates[len(initial) + i] += delta
        actual = evaluator.predict(coordinates, 8)
        independent = NativeJointEvaluator(
            model, obs, evaluator.proposal_mosaic, instrument_model, 2
        )
        np.testing.assert_allclose(
            actual, independent.predict(coordinates, 8), rtol=3e-13, atol=1e-18
        )
    assert evaluator.compile_count == 10  # Spectral population reuses its exact response.
    from rasim_next.fitting.native_search import FitParameter, fit_native_parameters

    # Real-forward recovery with all other coordinates fixed distinguishes this
    # integration proof from the independent analytic optimizer tests.
    truth = np.r_[initial, instrument_model.initial_values]
    raw = evaluator.predict(truth, 8)
    scale = 500 / np.max(raw)
    target = scale * raw
    synthetic = replace(obs, net_count=target, fit_target=target, allow_guard_constraints=False)
    specification = tuple(
        FitParameter(
            name,
            unit,
            "synthetic",
            float(value - abs(value) * 0.1 - 0.005),
            float(value + abs(value) * 0.1 + 0.005),
            0.001,
        )
        for name, unit, value in zip(
            evaluator.parameter_names, evaluator.parameter_units, truth, strict=True
        )
    )
    perturbed = truth.copy()
    perturbed[9] += 0.005
    recovery = fit_native_parameters(
        lambda v: evaluator.predict(v, 8),
        synthetic,
        specification,
        [perturbed],
        fixed_values={p.name: float(truth[i]) for i, p in enumerate(specification) if i != 9},
        maximum_iterations=60,
        finite_difference_step=1e-6,
    )
    assert recovery.best_converged is not None
    assert recovery.best_converged.data_chi_square < 1e-6
    assert recovery.best_converged.parameter_values[9] == pytest.approx(truth[9], abs=2e-5)
    record["material"]["n_imag"] = [1e-6]
    write()
    with pytest.raises(ValueError, match="refractive indices"):
        load_native_fit_physics(path)
    record["material"]["n_imag"] = [1e-6, 2e-6]
    record["unrecognized_physics"] = True
    write()
    with pytest.raises(ValueError, match="unrecognized"):
        load_native_fit_physics(path)


def test_streamed_fiber_matches_native_pixels_and_individual_rods():
    from painted_ewald import MosaicParameters, Rod
    from painted_ewald.normal_density import SphericalMosaicDensity
    from rasim_next.core.contracts import MaterialOptics
    from rasim_next.geometry.transport import build_incident_states
    from rasim_next.pipeline.fiber_detector import (
        FiberIntegrationRule,
        compile_conditional_fiber_transfer,
        conditional_ewald_region_bounds,
        iter_conditional_fiber_transfers,
    )

    shape = (12, 14)
    instrument = replace(
        configuration()["instrument"],
        detector_shape_rc=shape,
        detector_row_pitch_m=0.008,
        detector_column_pitch_m=0.006,
        detector_reference_coordinate_px=(6.5, 5.5),
    )
    source = sample_conditional_gaussian_source(
        mean_origin_lab_m=[0, 0, 0.05],
        mean_direction_lab=[0, 0, -1],
        transverse_axes_lab=np.eye(3)[:2],
        spatial_sigma_m=[0.001, 0.002],
        divergence_sigma_rad=[0.002, 0.003],
        position_divergence_correlation=[-0.3, 0.4],
        line_wavelength_A=[1.54, 1.544],
        line_probability=[0.66, 0.34],
        common_wavelength_sigma_A=0,
        sample_count=4,
        seed=723,
        polarization_state_id="UNITY_APPROXIMATION",
    )
    material = MaterialOptics(
        "vacuum", np.array([1.54, 1.544]), np.ones(2, complex), "stream-boundary-proof"
    )
    incident = build_incident_states(source.mean_rays, material, instrument)
    basis = np.array([[0.3, 0, 0], [0, 0.3, 0], [0.07, -0.04, 0.2]])
    b3 = np.linalg.norm(basis[:, 2])
    normal = basis[:, 2] / b3
    rods = (Rod(0, 0, 0.4), Rod(1, 0, 0.3), Rod(-1, 0, 0.7))
    mosaic = MosaicParameters(0.4, 0.7, 0.3)
    law = SphericalMosaicDensity(mosaic)
    flat = np.arange(np.prod(shape))
    owner = (flat % shape[1] >= 7).astype(int)
    projection = NativePixelRegionProjection(
        shape, flat, owner, flat, np.ones(len(flat)), 2, "stream-panel-partition"
    )
    spatial = NativeSpatialRegionProjection(projection)
    options = dict(quadrature_order=8, gaussian_tail_radius=8.0)
    context = dict(source=source, incident=incident, material=material, instrument=instrument)
    arguments = dict(
        **context,
        rods=rods,
        reciprocal_basis_Ainv=basis,
        crystal_to_sample=np.eye(3),
        native_bounds_px=[[-0.5, 13.5, -0.5, 11.5]],
        reference_mosaic=mosaic,
        local_stitched_m0=False,
    )

    def strength(rod, ell):
        return (1 + 0.07 * rod.h - 0.03 * rod.k + 0.01 * ell) ** 2 + 0.1

    small_rule = FiberIntegrationRule(axial_power=4, angular_power=1, seed=1, batch_size=100000)
    full = tuple(iter_conditional_fiber_transfers(**arguments, rule=small_rule))
    reference = {(batch.source_state_index, batch.radial_Ainv): batch for batch in full}
    assert len(reference) == len(full) > 0

    refined_source = sample_conditional_gaussian_source(
        mean_origin_lab_m=[0, 0, 0.05],
        mean_direction_lab=[0, 0, -1],
        transverse_axes_lab=np.eye(3)[:2],
        spatial_sigma_m=[0.001, 0.002],
        divergence_sigma_rad=[0.002, 0.003],
        position_divergence_correlation=[-0.3, 0.4],
        line_wavelength_A=[1.54, 1.544],
        line_probability=[0.66, 0.34],
        common_wavelength_sigma_A=0,
        sample_count=8,
        seed=723,
        polarization_state_id="UNITY_APPROXIMATION",
    )
    fixed_rule = replace(small_rule, regular_q_bounds_Ainv=(0.0, 9.0))
    axial_grids = []
    for source_rule in (source, refined_source):
        stream = tuple(
            iter_conditional_fiber_transfers(
                **(
                    arguments
                    | dict(
                        source=source_rule,
                        incident=build_incident_states(source_rule.mean_rays, material, instrument),
                    )
                ),
                rule=fixed_rule,
            )
        )
        axial_grids.append({b.radial_Ainv: b.positive_axial_Ainv for b in stream})
        for batch in stream:
            np.testing.assert_array_equal(
                batch.positive_axial_Ainv, axial_grids[-1][batch.radial_Ainv]
            )
    assert axial_grids[0].keys() == axial_grids[1].keys()
    for radius in axial_grids[0]:
        np.testing.assert_array_equal(axial_grids[0][radius], axial_grids[1][radius])
    with pytest.raises(ValueError, match="does not enclose"):
        tuple(
            iter_conditional_fiber_transfers(
                **arguments,
                rule=replace(small_rule, regular_q_bounds_Ainv=(0.0, 0.01)),
            )
        )

    def assert_same_stream(actual):
        actual = tuple(actual)
        assert len(actual) == len(reference)
        found = {(batch.source_state_index, batch.radial_Ainv): batch for batch in actual}
        assert found.keys() == reference.keys()
        for key, batch in found.items():
            expected = reference[key]
            for name in ("positive_axial_Ainv", "axial_index", "integrated_coefficient"):
                np.testing.assert_array_equal(getattr(batch, name), getattr(expected, name))
            for name in ("polar_angle_rad", "cone_angle_rad"):
                np.testing.assert_array_equal(
                    getattr(batch.transfer, name), getattr(expected.transfer, name)
                )
            for name in ("mean_px", "factor_px"):
                np.testing.assert_array_equal(
                    getattr(batch.transfer.spatial, name),
                    getattr(expected.transfer.spatial, name),
                )

    assert_same_stream(
        batch
        for indices in ((0, 1), (2, 3))
        for batch in iter_conditional_fiber_transfers(
            **arguments, rule=small_rule, source_state_indices=indices
        )
    )
    # The tilted detector-plane extension lies below the top-exit hemisphere.
    empty_bounds = [[1000.0, 1001.0, 5.0, 6.0]]
    for si in range(4):
        bounds = conditional_ewald_region_bounds(
            **context,
            native_bounds_px=empty_bounds,
            source_state_index=si,
            source_latent_radius=8.0,
            local_m0=False,
        )
        assert np.all(bounds[:, 0] > bounds[:, 1])
    assert not tuple(
        iter_conditional_fiber_transfers(
            **(arguments | {"native_bounds_px": empty_bounds}), rule=small_rule
        )
    )
    assert_same_stream(
        iter_conditional_fiber_transfers(
            **(arguments | {"native_bounds_px": [*arguments["native_bounds_px"], *empty_bounds]}),
            rule=small_rule,
        )
    )

    results = []
    for batch_size in (100000, 13):
        image, roi = np.zeros(shape), np.zeros(2)
        seen = set()
        for batch in iter_conditional_fiber_transfers(
            **arguments,
            rule=FiberIntegrationRule(
                axial_power=6, angular_power=2, seed=1, batch_size=batch_size
            ),
        ):
            seen.update((rod.h, rod.k) for rod in batch.rods)
            t = batch.positive_axial_Ainv[batch.axial_index]
            f = batch.transfer
            mass = np.zeros(len(t))
            for rod in batch.rods:
                offset = (rod.h * basis[:, 0] + rod.k * basis[:, 1]) @ normal
                for sign, cone in ((1, f.cone_angle_rad), (-1, np.pi - f.cone_angle_rad)):
                    ell = (sign * t - offset) / b3
                    mass += (
                        batch.integrated_coefficient
                        * rod.population
                        * strength(rod, ell)
                        * law.cone_average_sr_inv(f.polar_angle_rad, cone)
                    )
            if batch_size == 100000:
                # Recover Ewald azimuth from independent mean detector intersections.
                local = np.column_stack(
                    (
                        (f.spatial.mean_px[:, 0] - 6.5) * 0.006,
                        (f.spatial.mean_px[:, 1] - 5.5) * 0.008,
                        np.zeros(len(t)),
                    )
                )
                points = instrument.lab_from_detector.apply_point(local)
                si = batch.source_state_index
                outgoing = points - incident.states.sample_intersection_lab_m[si]
                outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
                ki = incident.states.k_film_phase_sample_Ainv[si]
                q = (2 * np.pi / source.mean_rays.wavelength_A[si]) * outgoing - ki
                axis = ki / np.linalg.norm(ki)
                first = np.array([1.0, 0, 0]) - axis[0] * axis
                first /= np.linalg.norm(first)
                azimuth = np.mod(np.arctan2(q @ np.cross(axis, first), q @ first), 2 * np.pi)
                quadrature_weight = batch.integrated_coefficient * b3 / f.coefficient_per_L_rad
                individual = np.zeros(len(t))
                for rod in batch.rods:
                    offset = (rod.h * basis[:, 0] + rod.k * basis[:, 1]) @ normal
                    for sign in (1, -1):
                        ell = (sign * t - offset) / b3
                        direct = compile_conditional_fiber_transfer(
                            **context,
                            rod=rod,
                            reciprocal_basis_Ainv=basis,
                            crystal_to_sample=np.eye(3),
                            L=ell,
                            ewald_azimuth_rad=azimuth,
                            source_state_index=si,
                            maximum_backward_probability=1e-12,
                        )
                        assert direct is not None
                        np.testing.assert_array_equal(direct.quadrature_index, np.arange(len(t)))
                        individual += (
                            quadrature_weight
                            * direct.coefficient_per_L_rad
                            / b3
                            * strength(rod, ell)
                            * law.cone_average_sr_inv(direct.polar_angle_rad, direct.cone_angle_rad)
                        )
                np.testing.assert_allclose(mass, individual, rtol=2e-11, atol=2e-16)
            image += f.spatial.integrate_native_pixels(shape, integrated_mass=mass, **options)
            roi += mass @ spatial.probabilities(f.spatial, **options)
            assert not batch.positive_axial_Ainv.flags.writeable
            assert not batch.integrated_coefficient.flags.writeable
        assert seen == {(0, 0), (1, 0), (-1, 0)}
        np.testing.assert_allclose(
            roi, np.bincount(owner, weights=image.ravel(), minlength=2), rtol=2e-12, atol=2e-15
        )
        assert image.sum() > 0
        results.append((image, roi))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=2e-12, atol=2e-15)
    # The production SF owner must reproduce that individual-rod construction.
    from dataclasses import dataclass

    from rasim_next.pipeline.conditional_detector import ConditionalStructureDetector

    @dataclass(frozen=True)
    class SignedStrength:
        reciprocal_basis_Ainv: np.ndarray
        structure_model_revision: str = "a" * 64

        def evaluate_hkl(self, *, h, k, L, k_norm_Ainv):
            return (1 + 0.07 * h - 0.03 * k + 0.01 * L) ** 2 + 0.1

    detector = ConditionalStructureDetector(
        **context,
        rods=rods,
        reciprocal_basis_Ainv=basis,
        crystal_to_sample=np.eye(3),
        rod_catalog_revision="signed-offset-stream-proof",
        strength_model=SignedStrength(basis),
        mosaic=mosaic,
        integration_rule=FiberIntegrationRule(
            axial_power=6, angular_power=2, seed=1, batch_size=100000
        ),
        spatial_quadrature_order=8,
    )
    np.testing.assert_allclose(
        detector.integrate_native_pixels(), results[0][0], rtol=2e-11, atol=2e-15
    )
    total_projection = NativePixelRegionProjection(
        shape,
        flat,
        np.zeros(len(flat), dtype=int),
        flat,
        np.ones(len(flat)),
        1,
        "native-total-with-identical-continuous-domain",
    )
    response = detector.compile_native_response(total_projection)
    np.testing.assert_allclose(response.evaluate(), [results[0][0].sum()], rtol=2e-11, atol=2e-15)
    parallel = detector.compile_native_response(total_projection, worker_count=3)
    np.testing.assert_array_equal(parallel.evaluate(), response.evaluate())
    for left, right in zip(parallel.region_probability, response.region_probability, strict=True):
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    changed_thickness = replace(detector, instrument=replace(instrument, film_thickness_A=750))
    wider = MosaicParameters(0.3, 0.5, 0.4)
    np.testing.assert_allclose(
        response.evaluate(thickness_A=750, mosaic=wider),
        changed_thickness.compile_native_response(total_projection).evaluate(mosaic=wider),
        rtol=2e-12,
        atol=2e-15,
    )
    # Progressive images average complete intensity estimates with exact random prefixes.
    from concurrent.futures import CancelledError

    first = detector.sample_native_pixel_mass(draws_per_batch=3, seed=9)
    extra = detector.sample_native_pixel_mass(draws_per_batch=4, draw_offset=3, seed=9)
    whole = detector.sample_native_pixel_mass(draws_per_batch=7, seed=9)
    np.testing.assert_allclose((3 * first + 4 * extra) / 7, whole, rtol=2e-12, atol=2e-15)
    with pytest.raises(CancelledError):
        detector.sample_native_pixel_mass(draws_per_batch=1, seed=9, cancel_requested=lambda: True)
    # Default SF lineage and integer proposal seeds cannot alias response identities.
    changed = replace(detector, strength_model=SignedStrength(basis, "b" * 64))
    assert changed.fixed_physics_revision == detector.fixed_physics_revision
    assert (
        changed.compile_native_response(total_projection).response_revision
        != response.response_revision
    )
    seed_a = replace(detector, integration_rule=replace(detector.integration_rule, seed=2**53))
    seed_b = replace(detector, integration_rule=replace(detector.integration_rule, seed=2**53 + 1))
    assert seed_a.fixed_physics_revision != seed_b.fixed_physics_revision
    # Unsupported measures fail even before an unreachable rod can yield no events.
    bad = arguments | {
        "rods": (Rod(10000, 0),),
        "instrument": replace(
            instrument,
            detector_path_medium_id="declared_absorbing_medium.v1",
            detector_path_linear_attenuation_m_inv=1.0,
        ),
    }
    with pytest.raises(ValueError, match="external-path absorption"):
        tuple(iter_conditional_fiber_transfers(**bad, rule=FiberIntegrationRule()))


def test_conditional_gaussian_matches_direct_rays_and_moments():
    args = configuration()
    instrument = args["instrument"]
    kernels = compile_conditional_spatial_kernels(**args)
    nodes, weights = hermgauss(4)
    z = np.stack(np.meshgrid(nodes, nodes), axis=-1).reshape(-1, 2) * np.sqrt(2.0)
    probability = np.outer(weights, weights).ravel() / np.pi
    source = args["source"]
    origins = source.mean_rays.origin_lab_m[0] + z @ source.conditional_origin_factor_lab_m.T
    sample = _intersect_sample_rays(
        origins,
        np.broadcast_to(source.mean_rays.direction_lab[0], origins.shape),
        lab_from_sample=instrument.lab_from_sample,
        sample_from_lab=instrument.sample_from_lab,
        sample_support_model_id=instrument.sample_support_model_id,
        sample_width_m=None,
        sample_length_m=None,
    )
    for i, outgoing in enumerate(args["outgoing_direction_lab"]):
        direct = _intersect_detector_plane(
            sample.point_lab_m, np.broadcast_to(outgoing, origins.shape), instrument
        )
        hits = np.column_stack((direct.column_px, direct.row_px))
        np.testing.assert_allclose(
            kernels.mean_px[i] + z @ kernels.factor_px[i].T, hits, rtol=0, atol=8e-14
        )
        mean = probability @ hits
        delta = hits - mean
        covariance = delta.T @ (probability[:, None] * delta)
        np.testing.assert_allclose(mean, kernels.mean_px[i], rtol=0, atol=3e-14)
        np.testing.assert_allclose(
            covariance, kernels.factor_px[i] @ kernels.factor_px[i].T, rtol=2e-14, atol=1e-14
        )
    assert kernels.mean_px[1, 0] > instrument.detector_shape_rc[1]  # off-panel center retained
    assert not kernels.mean_px.flags.writeable


def test_local_native_mixture_stitches_components_and_preserves_signed_sf():
    from dataclasses import dataclass
    from pathlib import Path

    from painted_ewald import MosaicParameters, Rod
    from painted_ewald.normal_density import SphericalMosaicDensity
    from rasim_next.core.contracts import MaterialOptics
    from rasim_next.core.scattering import CLASSICAL_ELECTRON_RADIUS_A
    from rasim_next.geometry.transport import build_incident_states
    from rasim_next.materials import read_crystal
    from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength, IncoherentStructureMixture
    from rasim_next.pipeline.conditional_detector import ConditionalStructureDetector
    from rasim_next.pipeline.fiber_detector import FiberIntegrationRule
    from rasim_next.reflectivity.parratt import parratt_reflectivity
    from rasim_next.reflectivity.specular import ParrattStitchStack, compile_parratt_stitch

    angle, shape = 0.025, (6, 20)
    instrument = replace(
        configuration()["instrument"],
        lab_from_detector=RigidTransform(
            np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([0.2, 0.0, 0.016]),
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=shape,
        detector_row_pitch_m=0.001,
        detector_column_pitch_m=0.001,
        detector_reference_coordinate_px=(9.5, 2.5),
    )
    source = sample_conditional_gaussian_source(
        mean_origin_lab_m=[-0.1, 0.0, 0.1 * np.tan(angle)],
        mean_direction_lab=[np.cos(angle), 0.0, -np.sin(angle)],
        transverse_axes_lab=[[np.sin(angle), 0.0, np.cos(angle)], [0.0, 1.0, 0.0]],
        spatial_sigma_m=[0.00005, 0.00005],
        divergence_sigma_rad=[0.0001, 0.0001],
        position_divergence_correlation=[-0.3, 0.4],
        line_wavelength_A=[1.54, 1.544],
        line_probability=[0.66, 0.34],
        common_wavelength_sigma_A=0.0,
        sample_count=4,
        seed=723,
        polarization_state_id="UNITY_APPROXIMATION",
    )
    material = MaterialOptics(
        "Bi2Te3-proof", np.array([1.54, 1.544]), np.full(2, 0.99994 + 1e-6j), "local-mixture-proof"
    )
    incident = build_incident_states(source.mean_rays, material, instrument)
    crystal = read_crystal(
        Path(__file__).resolve().parents[1] / "examples/bi2te3/structures/Bi2Te3_cod_9011962.cif",
        phase_id="Bi2Te3",
    )
    models = tuple(Bi2X3FiniteStackStrength(crystal, n) for n in (4, 11))
    mixture = IncoherentStructureMixture(models, (0.37, 0.63))
    stack = ParrattStitchStack(0.99998 + 1e-7j, top_roughness_A=1.0, bottom_roughness_A=2.0)
    detector = ConditionalStructureDetector(
        reciprocal_basis_Ainv=models[0].reciprocal_basis_Ainv,
        crystal_to_sample=np.eye(3),
        rods=(Rod(0, 0, 0.7),),
        rod_catalog_revision="local-mixture-one-rod",
        source=source,
        incident=incident,
        material=material,
        instrument=instrument,
        strength_model=mixture,
        mosaic=MosaicParameters(0.15, 0.3, 0.4),
        integration_rule=FiberIntegrationRule(axial_power=6, angular_power=2, seed=1),
        specular_stitch_stack=stack,
        phase_population_weight=0.8,
        polarization_weight=0.9,
        spatial_quadrature_order=8,
    )
    flat = np.arange(np.prod(shape))
    projection = NativePixelRegionProjection(
        shape,
        flat,
        (flat % shape[1] // 5).astype(int),
        flat,
        np.ones(len(flat)),
        4,
        "local-mixture-four-strips",
    )
    response = detector.compile_native_response(projection)
    actual = response.evaluate()
    expected = np.array(mixture.probabilities) @ np.array([response.evaluate(m) for m in models])
    assert np.all(actual > 0.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)
    changed_stack = replace(stack, top_roughness_A=3.0, bottom_roughness_A=7.0)
    changed_mixture = replace(mixture, probabilities=(0.6, 0.4))
    changed_detector = replace(
        detector,
        strength_model=changed_mixture,
        instrument=replace(instrument, film_thickness_A=750),
        specular_stitch_stack=changed_stack,
    )
    target_mosaic = MosaicParameters(0.2, 0.4, 0.6)
    reused = response.evaluate(
        changed_mixture, mosaic=target_mosaic, thickness_A=750, specular_stitch_stack=changed_stack
    )
    rebuilt = changed_detector.compile_native_response(projection, worker_count=2)
    np.testing.assert_allclose(reused, rebuilt.evaluate(mosaic=target_mosaic), rtol=3e-13, atol=0)
    assert not np.allclose(reused, actual, rtol=1e-4, atol=0)

    @dataclass(frozen=True)
    class Premixed:
        wrapped: object
        structure_model_revision: str = "deliberately-premixed-stitch"

        @property
        def reciprocal_basis_Ainv(self):
            return self.wrapped.reciprocal_basis_Ainv

        def evaluate_hkl(self, **kwargs):
            return self.wrapped.evaluate_hkl(**kwargs)

    # Hiding the populations reproduces the incorrect stitch-after-mixing order.
    wrong = response.evaluate(Premixed(mixture))
    assert np.max(abs(wrong - actual) / actual) > 0.1

    @dataclass(frozen=True)
    class TwoAtom:
        reciprocal_basis_Ainv: np.ndarray
        structure_model_revision: str
        offset: float
        second: complex

        def evaluate_hkl(self, *, h, k, L, k_norm_Ainv):
            return (
                CLASSICAL_ELECTRON_RADIUS_A**2
                * abs(1 + self.second * np.exp(2j * np.pi * self.offset * np.asarray(L))) ** 2
            )

    signed_models = (
        TwoAtom(models[0].reciprocal_basis_Ainv, "two-atom-1", 0.25, 1 + 1j),
        TwoAtom(models[0].reciprocal_basis_Ainv, "two-atom-2", 0.37, 0.8 - 0.4j),
    )
    signed_mix = IncoherentStructureMixture(signed_models, (0.37, 0.63))
    wide = MosaicParameters(0.8, 1.2, 0.8)
    signed_actual = response.evaluate(signed_mix, mosaic=wide)
    # Independent Parratt and blend equations, without the detector's SF helper.
    tables = []
    b3 = np.linalg.norm(signed_mix.reciprocal_basis_Ainv[:, 2])
    for grid in response.grids:
        q, wavelength = grid.external_q_Ainv, grid.wavelength_A
        film = material.n_complex[np.flatnonzero(material.wavelength_A == wavelength)[0]]
        pure = parratt_reflectivity(
            q,
            wavelength,
            refractive_index=(1 + 0j, film, stack.substrate_refractive_index),
            thickness_A=(None, 500.0, None),
            roughness_A=(1.0, 2.0),
        )
        ell = 2 * np.maximum(pure.kz_Ainv[:, 1].real, 0.0) / b3
        table = np.zeros((2, len(q)))
        for model, population in zip(signed_models, signed_mix.probabilities, strict=True):
            stitch = compile_parratt_stitch(
                stack,
                lambda L, model=model, wavelength=wavelength: model.evaluate_hkl(
                    h=0, k=0, L=L, k_norm_Ainv=2 * np.pi / wavelength
                ),
                wavelength_A=wavelength,
                film_refractive_index=film,
                film_thickness_A=500.0,
                c_A=2 * np.pi / b3,
            )
            raw = model.evaluate_hkl(
                h=0, k=0, L=np.array([1.0, -1.0])[:, None] * ell, k_norm_Ainv=2 * np.pi / wavelength
            )
            low = (
                q
                * q
                * pure.reflectivity
                * stitch.zero_strength_A2
                / stitch.dimensionless_scale_factor
            )
            lower, upper = stitch.blend_bounds_q_over_qc
            x = np.clip((q / stitch.qc_Ainv - lower) / (upper - lower), 0.0, 1.0)
            weight = x**3 * (10 - 15 * x + 6 * x * x)
            floor = np.finfo(float).tiny
            stitched = np.exp(
                (1 - weight) * np.log(np.maximum(low, floor))
                + weight * np.log(np.maximum(raw, floor))
            )
            above = q / stitch.qc_Ainv >= upper
            stitched[:, above] = raw[:, above]
            table += population * 0.7 * stitched
        tables.append(table)
    law = SphericalMosaicDensity(wide)
    manual, absolute_l_wrong = np.zeros(4), np.zeros(4)
    for node, probability in zip(response.nodes, response.region_probability, strict=True):
        table = tables[node.grid_index][:, node.axial_index]
        plus = law.cone_average_sr_inv(node.polar_angle_rad, node.cone_angle_rad)
        minus = law.cone_average_sr_inv(node.polar_angle_rad, np.pi - node.cone_angle_rad)
        coefficient = (
            node.integrated_coefficient
            * 0.8
            * 0.9
            * detector.source.mean_rays.source_weight[node.source_state_index]
        )
        manual += (coefficient * (table[0] * plus + table[1] * minus)) @ probability
        absolute_l_wrong += (coefficient * table[0] * (plus + minus)) @ probability
    np.testing.assert_allclose(signed_actual, manual, rtol=5e-12, atol=0.0)
    assert np.max(abs(absolute_l_wrong - signed_actual) / signed_actual) > 0.01


def test_source_integrates_position_with_total_covariance_and_exact_line_masses():
    sigma, rho, divergence = (
        np.array([0.001, 0.002]),
        np.array([-0.3, 0.4]),
        np.array([0.003, 0.007]),
    )
    options = dict(
        mean_origin_lab_m=[0.0, 0.0, 0.05],
        mean_direction_lab=[0.0, 0.0, -1.0],
        transverse_axes_lab=np.eye(3)[:2],
        spatial_sigma_m=sigma,
        divergence_sigma_rad=divergence,
        position_divergence_correlation=rho,
        line_wavelength_A=[1.54, 1.544],
        line_probability=[0.66, 0.34],
        common_wavelength_sigma_A=0.0,
        sample_count=35,
        seed=72,
        polarization_state_id="unpolarized.v1",
    )
    source = sample_conditional_gaussian_source(**options)
    rays, z = source.mean_rays, source.standardized_divergence
    assert len(rays.wavelength_A) == 35
    for line, probability in zip(
        options["line_wavelength_A"], options["line_probability"], strict=True
    ):
        np.testing.assert_allclose(
            rays.source_weight[rays.wavelength_A == line].sum(), probability, rtol=0, atol=4e-16
        )
    # Independently recover small-angle exponential-map coordinates from directions.
    angle = np.arccos(-rays.direction_lab[:, 2])
    recovered = (
        rays.direction_lab[:, :2]
        * np.divide(angle, np.sin(angle), out=np.ones_like(angle), where=angle != 0)[:, None]
    )
    np.testing.assert_allclose(recovered / divergence, z, atol=4e-14)
    position = rays.origin_lab_m[:, :2] / sigma
    joint_mean = np.column_stack((position, z))
    covariance = joint_mean.T @ (rays.source_weight[:, None] * joint_mean)
    conditional = source.conditional_origin_factor_lab_m[:2] / sigma[:, None]
    covariance[:2, :2] += conditional @ conditional.T
    latent_covariance = z.T @ (rays.source_weight[:, None] * z)
    correlation = np.diag(rho)
    target = np.block(
        [
            [
                correlation @ latent_covariance @ correlation + np.diag(1 - rho**2),
                correlation @ latent_covariance,
            ],
            [latent_covariance @ correlation, latent_covariance],
        ]
    )
    np.testing.assert_allclose(covariance, target, rtol=0, atol=2e-14)
    np.testing.assert_allclose(rays.source_weight @ joint_mean, 0.0, atol=1e-15)
    assert not source.standardized_divergence.flags.writeable
    assert not source.conditional_origin_factor_lab_m.flags.writeable
    assert sample_conditional_gaussian_source(**options).revision == source.revision
    assert (
        sample_conditional_gaussian_source(**(options | {"seed": 73})).revision != source.revision
    )
    equal = sample_conditional_gaussian_source(
        **(options | {"sample_count": 512, "common_wavelength_sigma_A": 0.0001})
    )
    np.testing.assert_array_equal(
        equal.standardized_divergence[:256], equal.standardized_divergence[256:]
    )
    np.testing.assert_allclose(
        equal.mean_rays.wavelength_A[:256] - 1.54,
        equal.mean_rays.wavelength_A[256:] - 1.544,
        atol=3e-16,
    )
    physical = {key: value for key, value in options.items() if key not in ("sample_count", "seed")}
    physical.update(
        line_wavelength_A=[2.0, 4.0], line_probability=[0.2, 0.8], common_wavelength_sigma_A=0.25
    )
    gh = quadrature_conditional_gaussian_source(**physical, divergence_order=3, wavelength_order=3)
    rays, z = gh.mean_rays, gh.standardized_divergence
    assert len(rays.wavelength_A) == 54
    assert rays.source_rng_model_id == "no_rng.v1"
    for line, probability in zip(
        physical["line_wavelength_A"], physical["line_probability"], strict=True
    ):
        keep = abs(rays.wavelength_A - line) < 1.0
        latent = np.column_stack((z[keep], (rays.wavelength_A[keep] - line) / 0.25))
        weight = rays.source_weight[keep] / probability
        np.testing.assert_allclose(rays.source_weight[keep].sum(), probability, rtol=0, atol=4e-16)
        np.testing.assert_allclose(weight @ latent, 0, rtol=0, atol=2e-14)
        np.testing.assert_allclose(
            latent.T @ (weight[:, None] * latent), np.eye(3), rtol=0, atol=2e-14
        )
        np.testing.assert_allclose(weight @ latent**4, 3, rtol=0, atol=2e-14)
        np.testing.assert_allclose(weight @ np.prod(latent**2, axis=1), 1, rtol=0, atol=2e-14)
    position = (
        (rays.origin_lab_m - physical["mean_origin_lab_m"])
        @ np.asarray(physical["transverse_axes_lab"]).T
    ) / sigma
    joint = np.column_stack((position, z))
    covariance = joint.T @ (rays.source_weight[:, None] * joint)
    conditional = gh.conditional_origin_factor_lab_m[:2] / sigma[:, None]
    covariance[:2, :2] += conditional @ conditional.T
    target = np.eye(4)
    target[0, 2] = target[2, 0] = rho[0]
    target[1, 3] = target[3, 1] = rho[1]
    np.testing.assert_allclose(covariance, target, rtol=0, atol=2e-14)
    zero_width = quadrature_conditional_gaussian_source(
        **(physical | {"common_wavelength_sigma_A": 0.0}),
        divergence_order=3,
        wavelength_order=3,
    )
    assert len(zero_width.mean_rays.wavelength_A) == 18
    np.testing.assert_array_equal(
        np.unique(zero_width.mean_rays.wavelength_A), physical["line_wavelength_A"]
    )


def test_continuous_mixture_preserves_mass_and_off_panel_tails():
    kernels = compile_conditional_spatial_kernels(**configuration())
    mass = np.array([2.0, 7.0])
    points = kernels.mean_px[1] + np.array([[0.0, 0.0], [2.0, -3.0], [-1.0, 4.0]])
    expected = sum(
        weight * multivariate_normal.pdf(points, mean, factor @ factor.T)
        for mean, factor, weight in zip(kernels.mean_px, kernels.factor_px, mass, strict=True)
    )
    np.testing.assert_allclose(
        kernels.density_at(points[:, 0], points[:, 1], integrated_mass=mass), expected, rtol=2e-14
    )
    # Integrate each entire kernel in independent standardized coordinates.
    nodes, weights = np.polynomial.legendre.leggauss(90)
    z = np.stack(np.meshgrid(9 * nodes, 9 * nodes), axis=-1).reshape(-1, 2)
    area = np.outer(9 * weights, 9 * weights).ravel()
    for i in range(2):
        points = kernels.mean_px[i] + z @ kernels.factor_px[i].T
        single = np.zeros(2)
        single[i] = mass[i]
        integral = np.sum(
            area * kernels.density_at(points[:, 0], points[:, 1], integrated_mass=single)
        ) * abs(np.linalg.det(kernels.factor_px[i]))
        np.testing.assert_allclose(integral, mass[i], rtol=2e-14)
    samples = kernels.sample(150_000, integrated_mass=mass, rng=np.random.default_rng(921))
    expected_mean = mass @ kernels.mean_px / mass.sum()
    np.testing.assert_allclose(samples.mean(axis=0), expected_mean, rtol=0, atol=0.25)
    assert np.any(samples[:, 0] > 90.0)


def test_unsupported_probability_measures_fail_explicitly():
    args = configuration()
    with pytest.raises(ValueError, match="unbounded"):
        compile_conditional_spatial_kernels(
            **dict(
                args,
                instrument=replace(
                    args["instrument"],
                    sample_support_model_id="finite_rectangle.v1",
                    sample_width_m=0.01,
                    sample_length_m=0.01,
                ),
            )
        )
    with pytest.raises(ValueError, match="absorption"):
        compile_conditional_spatial_kernels(
            **dict(
                args,
                instrument=replace(
                    args["instrument"],
                    detector_path_medium_id="constant_mu_m_inv.v1",
                    detector_path_linear_attenuation_m_inv=1.0,
                ),
            )
        )
    with pytest.raises(ValueError, match="backward Gaussian"):
        compile_conditional_spatial_kernels(
            **dict(
                args,
                source=replace(args["source"], conditional_origin_factor_lab_m=np.eye(3)[:, :2]),
            )
        )
    with pytest.raises(ValueError, match="backward Gaussian"):
        compile_conditional_spatial_kernels(**dict(args, maximum_backward_probability=0.0))
    with pytest.raises(ValueError, match="singular"):
        DetectorSpatialKernels(np.zeros((1, 2)), np.zeros((1, 2, 2)), np.zeros(1))
    with pytest.raises(ValueError, match="singular"):
        compile_conditional_spatial_kernels(
            **dict(
                args,
                source=replace(args["source"], conditional_origin_factor_lab_m=np.zeros((3, 2))),
            )
        )


def test_oblique_narrow_beams_preserve_native_pixel_mass():
    """Near-singular projected beams retain their finite rectangle probabilities."""
    shape = (4, 4)
    mean = np.array([1.12, 1.6])
    sigma = np.array([0.8, 1.2])
    for rho in (-1 + 1e-12, -0.99, -0.4, 0.4, 0.99, 1 - 1e-12):
        conditional_sigma = np.sqrt(1.0 - rho * rho)
        factor = np.array([[sigma[0], 0.0], [rho * sigma[1], conditional_sigma * sigma[1]]])
        kernels = DetectorSpatialKernels(mean[None], factor[None], np.zeros(1))
        actual = kernels.integrate_native_pixels(
            shape, integrated_mass=np.ones(1), quadrature_order=16, gaussian_tail_radius=8.0
        )
        expected = np.empty(shape)
        for row, column in np.ndindex(shape):
            x_lower, x_upper = (np.array([column - 0.5, column + 0.5]) - mean[0]) / sigma[0]
            y_lower, y_upper = (np.array([row - 0.5, row + 0.5]) - mean[1]) / sigma[1]

            def integrand(x, y_lower, y_upper, rho, conditional_sigma):
                probability = ndtr((y_upper - rho * x) / conditional_sigma) - ndtr(
                    (y_lower - rho * x) / conditional_sigma
                )
                return np.exp(-0.5 * x * x) * probability / np.sqrt(2.0 * np.pi)

            # Resolve both sharp conditional-CDF transitions independently.
            transitions = np.array([y_lower, y_upper])[:, None] / rho + (
                conditional_sigma / abs(rho) * np.array([-10.0, 0.0, 10.0])
            )
            points = np.unique(transitions[(transitions > x_lower) & (transitions < x_upper)])
            expected[row, column] = quad(
                integrand,
                x_lower,
                x_upper,
                args=(y_lower, y_upper, rho, conditional_sigma),
                points=points,
                epsabs=1e-14,
                epsrel=1e-12,
            )[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=5e-13)
        # A negatively correlated beam reaches the narrow row only after its
        # first column. The overlapping whole panel must not suppress that hit.
        flat = np.arange(np.prod(shape))
        middle_row = flat[flat // shape[1] == 1]
        projection = NativePixelRegionProjection(
            shape,
            flat,
            np.r_[np.zeros(len(flat), dtype=int), np.ones(len(middle_row), dtype=int)],
            np.r_[flat, middle_row],
            np.ones(len(flat) + len(middle_row)),
            2,
            "oblique-panel-and-row.v1",
        )
        regions = NativeSpatialRegionProjection(projection).probabilities(
            kernels, quadrature_order=16, gaussian_tail_radius=8.0
        )
        np.testing.assert_allclose(
            regions.toarray()[0], [expected.sum(), expected[1].sum()], rtol=2e-11, atol=5e-13
        )


def test_pixel_roi_and_monte_carlo_share_integrated_gaussian_mass():
    shape = (8, 9)
    means = np.array([[-0.7, 2.4], [3.2, 4.1], [8.3, 5.8]])
    factors = np.array(
        [[[0.8, 0.1], [0.5, 1.2]], [[0.05, 0.0], [0.0, 0.09]], [[0.7, 0.0], [0.4, 0.6]]]
    )
    kernels = DetectorSpatialKernels(means, factors, np.zeros(3))
    mass = np.array([2.0, 3.0, 7.0])
    options = dict(quadrature_order=8, gaussian_tail_radius=8.0)
    image = kernels.integrate_native_pixels(shape, integrated_mass=mass, **options)

    # Independent two-dimensional tensor integral of the public continuous field.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    y, x = np.indices(shape)
    c = x[..., None, None] + 0.5 * nodes[None, None, :, None] + np.zeros((1, 1, 1, len(nodes)))
    r = y[..., None, None] + 0.5 * nodes[None, None, None, :] + np.zeros((1, 1, len(nodes), 1))
    density = kernels.density_at(c, r, integrated_mass=mass)
    oracle = np.einsum("rcij,i,j->rc", density, weights, weights) * 0.25
    np.testing.assert_allclose(image, oracle, atol=3e-12, rtol=2e-11)
    assert 3 < image.sum() < mass.sum()  # Narrow core conserved; off-panel mass lost.
    assert image[:, 0].sum() > 0.1  # Off-panel center still contributes.

    flat = np.arange(np.prod(shape))
    # Overlap with different weights must preserve every observation's mass.
    owner = np.r_[(x.ravel() >= 4).astype(int), np.full(len(flat), 2), np.full(len(flat), 3)]
    pixel_column = np.tile(flat, 3)
    alternating_weight = np.where(y.ravel() % 2 == 0, 1.0, 0.3)
    region_weight = np.r_[alternating_weight, np.ones(len(flat)), 0.7 * alternating_weight]
    keep = (owner == 2) | (x.ravel()[pixel_column] != 2)
    owner, pixel_column, region_weight = (a[keep] for a in (owner, pixel_column, region_weight))
    p = NativePixelRegionProjection(
        shape, flat, owner, pixel_column, region_weight, 4, "overlapping-native-regions.v1"
    )
    response = NativeSpatialRegionProjection(p).probabilities(kernels, **options)
    projected = mass @ response
    expected = np.bincount(owner, weights=region_weight * image.ravel()[pixel_column])
    np.testing.assert_allclose(projected, expected, atol=3e-12, rtol=2e-11)
    np.testing.assert_allclose(projected[3], 0.7 * (projected[0] + projected[1]), atol=3e-12)

    count, seed = 20000, 853
    chosen = np.random.default_rng(seed).choice(3, size=count, p=mass / mass.sum())
    sampled_weights = np.bincount(chosen, minlength=3) * mass.sum() / count
    sampled = kernels.sample_native_pixel_mass(
        count, shape, integrated_mass=mass, rng=np.random.default_rng(seed), **options
    )
    exact_sampled = kernels.integrate_native_pixels(
        shape, integrated_mass=sampled_weights, **options
    )
    np.testing.assert_allclose(sampled, exact_sampled, atol=2e-15)
    assert np.linalg.norm(sampled - image) / np.linalg.norm(image) < 0.02
    empty = NativePixelRegionProjection(shape, [0], [], [], [], 1, "empty-support.v1")
    empty_response = NativeSpatialRegionProjection(empty).probabilities(kernels, **options)
    assert empty_response.shape == (3, 1)
    assert empty_response.nnz == 0
