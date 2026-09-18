"""Physical site tensors, finite surface windows and complete native fit candidates."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rasim_next.core.scattering import CLASSICAL_ELECTRON_RADIUS_A
from rasim_next.materials import read_crystal
from rasim_next.materials.optics import atomic_scattering_factor_e
from rasim_next.pipeline.bragg_space import CifFiniteStackStrength


def bi_physics():
    from test_source_spatial import configuration

    from painted_ewald import Rod
    from rasim_next.core.contracts import EventIntensityNormalization
    from rasim_next.fitting.native_input import FiniteStructureRecipe, NativeFitPhysics
    from rasim_next.materials.optics import material_optics
    from rasim_next.pipeline.fiber_detector import FiberIntegrationRule
    from rasim_next.reflectivity.specular import ParrattStitchStack

    crystal = read_crystal(
        Path(__file__).resolve().parents[1] / "examples/bi2te3/structures/Bi2Te3_cod_9011962.cif",
        phase_id="Te",
    )
    labels = sorted({site.source_label for site in crystal.sites})
    crystal = replace(
        crystal,
        sites=tuple(
            replace(site, u_iso_A2=0.005 * labels.index(site.source_label))
            for site in crystal.sites
        ),
    )
    surfaces = tuple(
        replace(
            crystal,
            sites=tuple(
                replace(
                    site,
                    fractional=(
                        *site.fractional[:2],
                        site.fractional[2] + (site.fractional[2] < cut),
                    ),
                )
                for site in crystal.sites
            ),
        )
        for cut in (0, 0.1, 0.24)
    )
    config = configuration()
    source = config["source"]
    source = replace(
        source,
        mean_rays=replace(
            source.mean_rays,
            polarization_state_id=("UNITY_APPROXIMATION",) * len(source.mean_rays.wavelength_A),
        ),
    )
    return NativeFitPhysics(
        "Te",
        config["instrument"],
        material_optics(crystal, source.mean_rays.wavelength_A),
        source,
        (Rod(0, 0), Rod(1, 0)),
        "two-physical-rods",
        CifFiniteStackStrength(crystal, 1).reciprocal_basis_Ainv,
        np.eye(3),
        FiniteStructureRecipe(surfaces, EventIntensityNormalization.FINITE_TOTAL, None),
        FiberIntegrationRule(axial_power=3, angular_power=1),
        8,
        1.0,
        1.0,
        ParrattStitchStack(0.99998 + 1e-7j),
        "native-bi-proof",
    )


def test_native_axial_panel_vectors_preserve_signed_detector_mass():
    from painted_ewald import MosaicParameters
    from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
    from rasim_next.pipeline.fiber_detector import AxialPanelMesh

    physics = bi_physics()
    meshes = (
        AxialPanelMesh(((0, 0),), "external_local_m0_q", (0.0, 7.5, 7.8, 8.0, 8.2, 9.0)),
        AxialPanelMesh(((1, 0),), "positive_phase_axial", (0.0, 7.5, 7.8, 8.0, 8.2, 9.0)),
    )
    physics = replace(
        physics,
        integration_rule=replace(
            physics.integration_rule,
            quadrature_kind="composite_gauss",
            axial_meshes=meshes,
            angular_support="fixed_union",
            angular_power=3,
            angular_panel_edges_rad=(0.0, np.pi / 2, np.pi, 2 * np.pi),
        ),
    )
    detector = physics.detector(
        mosaic=MosaicParameters(0.5, 0.6, 0.4),
        coherent_repeats=3,
        film_thickness_A=200.0,
        surface_fractions=(0.2, 0.3, 0.5),
        phase_fractions=(1.0,),
        fault_parameters={},
    )
    changed = replace(
        detector,
        integration_rule=replace(
            detector.integration_rule, angular_panel_edges_rad=(0.0, np.pi / 3, np.pi, 2 * np.pi)
        ),
    )
    assert changed.fixed_physics_revision != detector.fixed_physics_revision
    from rasim_next.fitting.native_workflow import native_physics_with

    roundtrip = native_physics_with(
        physics,
        {},
        {
            "integration": {
                "angular_panel_edges_rad": list(detector.integration_rule.angular_panel_edges_rad)
            }
        },
    )
    assert roundtrip.integration_rule == physics.integration_rule
    rows, columns = detector.detector_shape_rc
    flat = np.arange(rows * columns)
    projection = NativePixelRegionProjection(
        (rows, columns),
        flat,
        (flat // columns >= rows // 2).astype(int),
        flat,
        np.ones(len(flat)),
        2,
        "axial-panel-proof",
    )
    response = detector.compile_native_response(projection)
    panels = response.evaluate(resolve_axial_panels=True)
    assert panels.shape == (10, 2)
    assert np.any(panels > 0)
    np.testing.assert_allclose(panels.sum(axis=0), response.evaluate(), rtol=2e-13)


def test_fixed_proposal_reuses_scattering_and_separates_spectral_mass_from_pixels():
    from painted_ewald import MosaicParameters
    from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
    from rasim_next.pipeline.fiber_detector import FiberScatteringCache

    physics = bi_physics()
    physics = replace(
        physics,
        integration_rule=replace(
            physics.integration_rule,
            axial_power=5,
            angular_power=2,
            frozen_ewald_bounds_Ainv_rad=(0.0, 9.0, 0.0, 2 * np.pi),
        ),
    )
    proposal = MosaicParameters(0.5, 0.6, 0.4)
    physical = MosaicParameters(0.7, 0.9, 0.3)
    arguments = dict(
        coherent_repeats=3,
        film_thickness_A=200.0,
        surface_fractions=(0.2, 0.3, 0.5),
        phase_fractions=(1.0,),
        fault_parameters={},
    )
    detector = physics.detector(mosaic=physical, **arguments)
    detector = replace(
        detector,
        proposal_mosaic=proposal,
        specular_stitch_stack=replace(
            detector.specular_stitch_stack, top_roughness_A=2.0, bottom_roughness_A=3.0
        ),
    )
    rows, columns = detector.detector_shape_rc
    flat = np.arange(rows * columns)
    projection = NativePixelRegionProjection(
        (rows, columns),
        flat,
        (flat // columns >= rows // 2).astype(int),
        flat,
        np.ones(len(flat)),
        2,
        "reuse-and-native-render-proof",
    )
    cache = FiberScatteringCache()
    response = detector.compile_native_response(projection, scattering_cache=cache)
    expected = response.evaluate()
    assert np.any(expected > 0)
    image = detector.integrate_native_pixels()
    np.testing.assert_allclose(
        projection.integrate_field(image, np.ones_like(image))[0], expected, rtol=2e-10, atol=1e-18
    )
    parts = [
        detector.integrate_native_pixels(row_bounds=(0, rows // 2)),
        detector.integrate_native_pixels(row_bounds=(rows // 2, rows)),
    ]
    np.testing.assert_array_equal(np.vstack(parts), image)
    contributions = list(detector.iter_native_pixel_batches())
    np.testing.assert_array_equal(sum(contributions), image)
    np.testing.assert_array_equal(
        sum(detector.iter_native_pixel_batches(batch_offset=1), contributions[0].copy()), image
    )
    built = cache.build_count
    moved = replace(
        detector,
        instrument=replace(detector.instrument, detector_reference_coordinate_px=(48.0, 42.0)),
    )
    reused = moved.compile_native_response(projection, scattering_cache=cache).evaluate()
    np.testing.assert_array_equal(reused, moved.compile_native_response(projection).evaluate())
    assert cache.build_count == built and cache.reuse_count > 0
    assert not np.array_equal(reused, expected)
    weights = detector.source.mean_rays.source_weight
    first_line = detector.source.mean_rays.wavelength_A == detector.source.mean_rays.wavelength_A[0]
    line0 = np.where(first_line, weights / weights[first_line].sum(), 0.0)
    line1 = np.where(~first_line, weights / weights[~first_line].sum(), 0.0)
    endpoints = [response.evaluate(source_weights=w) for w in (line0, line1)]
    np.testing.assert_allclose(
        response.evaluate(source_weights=0.2 * line0 + 0.8 * line1),
        0.2 * endpoints[0] + 0.8 * endpoints[1],
        rtol=3e-14,
        atol=0,
    )
    new_source = replace(
        detector.source, mean_rays=replace(detector.source.mean_rays, source_weight=line0)
    )
    rebound = replace(physics, source=new_source).detector(mosaic=physical, **arguments)
    rebound = replace(
        rebound, proposal_mosaic=proposal, specular_stitch_stack=detector.specular_stitch_stack
    )
    np.testing.assert_allclose(
        rebound.compile_native_response(projection).evaluate(), endpoints[0], rtol=3e-13, atol=0
    )
    changed = replace(
        detector,
        integration_rule=replace(
            detector.integration_rule, frozen_ewald_bounds_Ainv_rad=(0, 10, 0, 2 * np.pi)
        ),
    )
    assert changed.fixed_physics_revision != detector.fixed_physics_revision
    with pytest.raises(ValueError, match="does not enclose"):
        replace(
            detector,
            integration_rule=replace(
                detector.integration_rule, frozen_ewald_bounds_Ainv_rad=(0, 0.01, 0, 2 * np.pi)
            ),
        ).compile_native_response(projection)


def test_bi_thirteen_coordinates_bind_optics_lattice_orbits_and_surface_windows():
    from painted_ewald import MosaicParameters
    from rasim_next.fitting.bi_native import BiCellSiteParameters, BiNativeStructureModel

    reference = bi_physics()
    model = BiNativeStructureModel(reference)
    parameters = model.reference_parameters
    initial = model.bind(parameters)
    args = dict(h=[0, 1, -1, 2], k=[0, 0, 1, -1], L=[0.37, 1.23, -2.38, 3.14], k_norm_Ainv=4.08)
    strength_args = dict(
        coherent_repeats=3,
        surface_fractions=(0.2, 0.3, 0.5),
        phase_fractions=(1.0,),
        fault_parameters={},
    )
    baseline = initial.structure.strength(**strength_args).evaluate_hkl(**args)
    np.testing.assert_allclose(
        baseline,
        reference.structure.strength(**strength_args).evaluate_hkl(**args),
        rtol=3e-13,
        atol=1e-18,
    )
    np.testing.assert_allclose(initial.material.n_complex, reference.material.n_complex, atol=1e-20)
    steps = np.array([0.002, 0.004, 0.0002, 0.0002, -0.01, -0.01, -0.01, *([0.003] * 6)])
    for index, step in enumerate(steps):
        values = parameters.as_array()
        values[index] += step
        candidate = model.bind(BiCellSiteParameters.from_array(values))
        direct = candidate.structure.crystals[0].direct_basis_A
        np.testing.assert_allclose(
            direct.T @ candidate.reciprocal_basis_Ainv, 2 * np.pi * np.eye(3), atol=3e-15
        )
        assert (
            not np.array_equal(candidate.reciprocal_basis_Ainv, initial.reciprocal_basis_Ainv)
        ) == (index < 2)
        assert (not np.array_equal(candidate.material.n_complex, initial.material.n_complex)) == (
            index in (0, 1, 4, 5, 6)
        )
        changed = candidate.structure.strength(**strength_args)
        assert not np.allclose(changed.evaluate_hkl(**args), baseline, rtol=1e-7, atol=1e-18)
        for surface, original in zip(
            candidate.structure.crystals, reference.structure.crystals, strict=True
        ):
            for site, base, first, ref_first in zip(
                surface.sites,
                original.sites,
                candidate.structure.crystals[0].sites,
                reference.structure.crystals[0].sites,
                strict=True,
            ):
                np.testing.assert_allclose(
                    np.subtract(site.fractional, first.fractional),
                    np.subtract(base.fractional, ref_first.fractional),
                    atol=2e-16,
                )
    with pytest.raises(ValueError, match="coherent stack extent"):
        initial.detector(
            mosaic=MosaicParameters(0.01, 0.02, 0.1), film_thickness_A=20, **strength_args
        )


def test_native_gls_scale_and_all_parameter_optimizer_recover_known_counts():
    from rasim_next.fitting.bi_joint import BI_JOINT_PARAMETER_NAMES
    from rasim_next.fitting.bi_native import BiNativeStructureModel
    from rasim_next.fitting.native_observations import NativeFitObservations
    from rasim_next.fitting.native_search import FitParameter, fit_native_parameters
    from rasim_next.measurement.continuous_regions import NativePixelRegionProjection

    values = np.r_[
        BiNativeStructureModel(bi_physics()).reference_parameters.as_array(),
        0.01,
        0.02,
        0.3,
        0.2,
        0.6,
        50,
        2,
        4,
    ]
    lower = np.r_[
        values[:2] * 0.99, values[2:4] - 0.005, [0.8] * 3, [0.0] * 6, 0.005, 0.005, 0, 0, 0, 0, 0, 0
    ]
    upper = np.r_[
        values[:2] * 1.01,
        values[2:4] + 0.005,
        [1.0] * 3,
        [0.08] * 6,
        0.03,
        0.04,
        1,
        1,
        1,
        100,
        10,
        10,
    ]
    truth = np.linspace(0.25, 0.75, 21)
    target = 7 * np.r_[1, 2 + truth]
    projection = NativePixelRegionProjection(
        (1, 22), np.arange(22), np.arange(22), np.arange(22), np.ones(22), 22, "GLS-proof"
    )
    covariance = np.diag(np.linspace(1, 3, 22)) + 0.2 * np.ones((22, 22))
    observations = NativeFitObservations(
        projection,
        target,
        np.ones(22, dtype=bool),
        covariance,
        np.eye(22),
        target,
        np.eye(22),
        np.array([0, 22]),
        np.array([100.0]),
        "known-count-proof",
    )
    raw = np.r_[1, 2 + truth]
    scale, residual = observations.profile_scale(raw)
    assert scale == pytest.approx(7, abs=2e-14)
    assert residual @ residual < 1e-25
    offset = np.linspace(-0.5, 0.7, 22)
    scores = observations.scores(target + offset)
    assert scores["gls_chi_square"] == pytest.approx(offset @ np.linalg.solve(covariance, offset))
    assert scores["historical_loss"] == pytest.approx(offset @ offset)
    one_guard = np.zeros((1, 22))
    one_guard[0, 0] = 1
    constrained = replace(
        observations,
        guard_operator=one_guard,
        guard_pointer=np.array([0, 1]),
        guard_limit=np.array([0.1]),
    )
    shape = np.r_[1.0, np.full(21, 10.0)]
    assert constrained.profile_scale(shape)[0] < 6.9
    assert constrained.guard_scale_interval(shape) == pytest.approx((6.9, 7.1), abs=1e-12)
    guarded_scale, _ = constrained.profile_scale(shape, enforce_guards=True)
    assert guarded_scale == pytest.approx(6.9, abs=1e-12)
    assert constrained.scores(guarded_scale * shape)["guards_pass"]
    with pytest.raises(ValueError, match="must align"):
        replace(observations, guard_limit=observations.guard_limit[:, None])

    class KnownNativeObservable:
        def __init__(self):
            self.observations = observations
            self.seen = []

        def predict(self, candidate):
            self.seen.append(candidate)
            return np.r_[1, 2 + (candidate - lower) / (upper - lower)]

    evaluator = KnownNativeObservable()
    result = fit_native_parameters(
        evaluator.predict,
        observations,
        tuple(
            FitParameter(name, "1", "specimen", lo, hi, (hi - lo) / 10)
            for name, lo, hi in zip(BI_JOINT_PARAMETER_NAMES, lower, upper, strict=True)
        ),
        [values],
        method="trf",
        maximum_function_evaluations=100,
        finite_difference_step=1e-5,
    )
    assert result.best_converged is not None
    assert np.all(np.ptp(np.array(evaluator.seen), axis=0) > 0)
    np.testing.assert_allclose(
        (result.best_converged.parameter_values - lower) / (upper - lower), truth, atol=2e-3
    )
    assert result.best_converged.scores["guards_pass"]
    assert result.numerical_status == "not_qualified"


def test_native_bi_reuse_invalidates_changed_density_and_cell():
    from painted_ewald import MosaicParameters
    from rasim_next.fitting.bi_joint import BiJointModel
    from rasim_next.fitting.bi_native import BiNativeStructureModel
    from rasim_next.fitting.native_joint import NativeJointEvaluator
    from rasim_next.fitting.native_observations import NativeFitObservations
    from rasim_next.measurement.continuous_regions import NativePixelRegionProjection

    physics = bi_physics()
    model = BiNativeStructureModel(physics)
    shape = physics.instrument.detector_shape_rc
    flat = np.arange(np.prod(shape))
    projection = NativePixelRegionProjection(
        shape, flat, flat // (len(flat) // 2), flat, np.ones(len(flat)), 2, "Bi-native-half-panels"
    )
    observations = NativeFitObservations(
        projection,
        np.ones(2),
        np.ones(2, dtype=bool),
        np.eye(2),
        np.eye(2),
        np.ones(2),
        np.eye(2),
        np.array([0, 2]),
        np.array([10.0]),
        "Bi-reuse",
    )
    from rasim_next.pipeline.conditional_detector import NativeMosaicCache
    from rasim_next.pipeline.source_spatial import NativeSpatialRegionProjection

    detector = physics.detector(
        mosaic=MosaicParameters(0.5, 0.6, 0.4),
        coherent_repeats=3,
        film_thickness_A=200.0,
        surface_fractions=(0.2, 0.3, 0.5),
        phase_fractions=(1.0,),
        fault_parameters={},
    )
    response = detector.compile_native_response(projection)
    # One full-rod evaluation preserves the explicitly split local-m0 rule.
    split_rule = replace(
        detector.integration_rule,
        quadrature_kind="composite_gauss",
        maximum_axial_panel_width_Ainv=None,
        local_m0_maximum_axial_panel_width_Ainv=0.2,
        local_m0_angular_power=2,
    )
    split_detector = replace(detector, integration_rule=split_rule)
    uncapped = replace(
        split_detector,
        integration_rule=replace(split_rule, local_m0_maximum_axial_panel_width_Ainv=None),
    )
    assert split_detector.fixed_physics_revision != uncapped.fixed_physics_revision
    global_angular = replace(
        split_detector, integration_rule=replace(split_rule, local_m0_angular_power=None)
    )
    assert split_detector.fixed_physics_revision != global_angular.fixed_physics_revision
    joint = split_detector.compile_native_response(projection)
    pieces = []
    for local, cap in ((True, 0.2), (False, None)):
        part = replace(
            detector,
            rods=tuple(r for r in detector.rods if (r.h == r.k == 0) == local),
            integration_rule=replace(
                split_rule,
                maximum_axial_panel_width_Ainv=cap,
                local_m0_maximum_axial_panel_width_Ainv=None,
                angular_power=2 if local else split_rule.angular_power,
                local_m0_angular_power=None,
            ),
        )
        pieces.append(part.compile_native_response(projection).evaluate())
    assert all(np.any(piece > 0) for piece in pieces)
    np.testing.assert_allclose(joint.evaluate(), sum(pieces), rtol=3e-13, atol=0)
    assert joint.response_revision != response.response_revision
    with pytest.raises(ValueError, match="axial panel width"):
        replace(split_rule, local_m0_maximum_axial_panel_width_Ainv=-1)
    with pytest.raises(ValueError, match="local_m0_angular_power"):
        replace(split_rule, local_m0_angular_power=True)
    projector = NativeSpatialRegionProjection(projection)
    projected = detector.compile_native_response(projection, spatial_projection=projector)
    np.testing.assert_array_equal(projected.evaluate(), response.evaluate())
    with pytest.raises(ValueError, match="another native observation"):
        detector.compile_native_response(
            replace(projection, quadrature_revision="other"), spatial_projection=projector
        )
    cache = NativeMosaicCache(response)
    for sigma, gamma, eta, thickness, order in (
        (0.5, 0.6, 0.4, 200.0, 16),
        (0.55, 0.6, 0.4, 200.0, 16),
        (0.55, 0.7, 0.4, 200.0, 16),
        (0.55, 0.7, 0.0, 210.0, 16),
        (0.55, 0.7, 1.0, 210.0, 16),
        (0.55, 0.7, 0.4, 210.0, 24),
    ):
        options = dict(
            mosaic=MosaicParameters(sigma, gamma, eta),
            thickness_A=thickness,
            cone_quadrature_order=order,
        )
        np.testing.assert_allclose(
            response.evaluate(mosaic_cache=cache, **options),
            response.evaluate(**options),
            rtol=3e-13,
            atol=0,
        )
    with pytest.raises(ValueError, match="another native response"):
        replace(response, projection_revision="other").evaluate(mosaic_cache=cache)
    values = np.r_[model.reference_parameters.as_array(), 0.5, 0.6, 0.4, 0.2, 0.6, 100, 2, 4]
    evaluator = NativeJointEvaluator(
        BiJointModel(model), observations, MosaicParameters(0.5, 0.6, 0.4), worker_count=2
    )
    initial = evaluator.predict(values, 3)
    assert np.any(initial > 0)
    assert not initial.flags.writeable
    np.testing.assert_array_equal(evaluator.predict(values, 3), initial)
    assert evaluator.evaluation_count == 1
    values[2] += 0.001
    values[18:] += 1
    reused = evaluator.predict(values, 3)
    assert evaluator.compile_count == 1
    independent = NativeJointEvaluator(BiJointModel(model), observations, evaluator.proposal_mosaic)
    np.testing.assert_allclose(reused, independent.predict(values, 3), rtol=3e-13, atol=0)
    values[4] -= 0.03
    density = evaluator.predict(values, 3)
    assert evaluator.compile_count == 2
    np.testing.assert_allclose(density, independent.predict(values, 3), rtol=3e-13, atol=0)
    values[0] += 0.01
    evaluator.predict(values, 3)
    assert evaluator.compile_count == 3
    with pytest.raises(ValueError, match="misses potentially elastic rods"):
        model.validate_rod_coverage(a_bounds_A=(4.3, 4.5), c_bounds_A=(30.0, 31.0))


def test_generic_finite_site_tensors_match_direct_signed_atom_and_repeat_sum():
    root = Path(__file__).resolve().parents[1]
    crystal = read_crystal(
        root / "examples/bi2te3/structures/Bi2Te3_cod_9011962.cif", phase_id="Te"
    )
    crystal = replace(
        crystal,
        sites=tuple(
            replace(
                s, u_iso_A2=0.012, fractional=(*s.fractional[:2], s.fractional[2] + (i % 3 == 0))
            )
            for i, s in enumerate(crystal.sites)
        ),
    )
    tensors = np.array(
        [np.diag([0.004 + i * 0.001, 0.008, 0.03]) for i in range(len(crystal.sites))]
    )
    tensors[:, 0, 2] = tensors[:, 2, 0] = 0.001
    model = CifFiniteStackStrength(crystal, 3, site_displacement_tensors_A2=tensors)
    hkl = np.array([[0, 0, 0.37], [1, 0, 1.27], [-1, 1, -2.38], [2, -1, 3.14]])
    hkl = np.vstack((hkl, -hkl))
    wave = np.tile([1.54, 1.544, 1.54, 1.544], 2)
    q = hkl @ model.reciprocal_basis_Ainv.T
    amplitude = np.zeros(len(hkl), dtype=complex)
    for i, site in enumerate(crystal.sites):
        factor, _ = atomic_scattering_factor_e(
            species=site.species,
            element=site.element,
            charge=site.charge,
            q_magnitude_Ainv=np.linalg.norm(q, axis=1),
            wavelength_A=wave,
        )
        damping = np.exp(-0.5 * np.einsum("ni,ij,nj->n", q, tensors[i], q))
        for repeat in range(3):
            position = crystal.direct_basis_A @ (
                np.array(site.fractional) + np.array([0, 0, repeat])
            )
            amplitude += site.occupancy * factor * damping * np.exp(1j * (q @ position))
    actual = model.evaluate_hkl(h=hkl[:, 0], k=hkl[:, 1], L=hkl[:, 2], k_norm_Ainv=2 * np.pi / wave)
    paired = np.tile(hkl.reshape(2, 4, 3), (1, 8, 1))
    np.testing.assert_allclose(
        model.evaluate_hkl(
            h=paired[..., 0],
            k=paired[..., 1],
            L=paired[..., 2],
            k_norm_Ainv=2 * np.pi / np.tile(wave.reshape(2, 4), (1, 8)),
        ),
        np.tile(actual.reshape(2, 4), (1, 8)),
        rtol=3e-13,
        atol=2e-18,
    )
    np.testing.assert_allclose(
        actual, abs(amplitude) ** 2 * CLASSICAL_ELECTRON_RADIUS_A**2, rtol=3e-13, atol=2e-18
    )
    # Geometric phases conjugate under inversion; anomalous species factors do
    # not. Both signed intensities must survive the shared arithmetic.
    assert not np.allclose(actual[:4], actual[4:], rtol=1e-4, atol=1e-18)
    isotropic = replace(model, site_displacement_tensors_A2=np.tile(0.012 * np.eye(3), (15, 1, 1)))
    arguments = dict(h=hkl[:, 0], k=hkl[:, 1], L=hkl[:, 2], k_norm_Ainv=2 * np.pi / wave)
    np.testing.assert_allclose(
        isotropic.evaluate_hkl(**arguments),
        CifFiniteStackStrength(crystal, 3).evaluate_hkl(**arguments),
        rtol=3e-13,
    )
    changed = tensors.copy()
    changed[:, 0, 0] += 0.5
    changed[:, 1, 1] += 0.5
    np.testing.assert_allclose(
        replace(model, site_displacement_tensors_A2=changed).evaluate_hkl(**arguments)[0],
        actual[0],
        rtol=3e-13,
    )
    wrapped = replace(
        crystal,
        sites=tuple(replace(s, fractional=tuple(np.mod(s.fractional, 1))) for s in crystal.sites),
    )
    assert not np.allclose(
        replace(model, crystal=wrapped).evaluate_hkl(**arguments), actual, rtol=1e-4, atol=1e-18
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(model, site_displacement_tensors_A2=-tensors)


def test_pb_site_displacements_and_finite_windows_match_explicit_atomic_paths():
    from itertools import pairwise, product

    from rasim_next.core.contracts import EventIntensityNormalization
    from rasim_next.ordered.motifs import (
        SiteDisplacementProfile,
        TransverseIsotropicSiteDisplacement,
    )
    from rasim_next.pipeline.bragg_space import Pbi2FiniteSurfaceStrength
    from rasim_next.stacking import (
        InitialPopulation,
        StackingPopulation,
        TransitionLaw,
        full_transition_matrix,
    )

    crystal = read_crystal(
        Path(__file__).resolve().parents[1] / "examples/pbi2/structures/PbI2_2H.cif", phase_id="Pb"
    )
    crystal = replace(
        crystal,
        sites=tuple(
            replace(
                s,
                fractional=(*np.round(np.array(s.fractional[:2]) * 3) / 3, s.fractional[2]),
                u_iso_A2=0.0,
            )
            for s in crystal.sites
        ),
    )
    profile = SiteDisplacementProfile(
        (
            TransverseIsotropicSiteDisplacement("Pb1", 0.008, 0.024),
            TransverseIsotropicSiteDisplacement("I1", 0.017, 0.039),
        )
    )
    initial = InitialPopulation(0.65, 0.35)
    law = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18)
    model = Pbi2FiniteSurfaceStrength(
        crystal,
        2,
        (StackingPopulation("mixture", law, initial),),
        (1.0,),
        (0.2, 0.3, 0.5),
        EventIntensityNormalization.FINITE_TOTAL,
        site_displacement_profile=profile,
    )
    hkl = np.array([[0, 0, 0.37], [1, 0, 1.27], [-1, 1, -2.38]])
    waves = np.array([1.54, 1.544, 1.54])
    q = hkl @ model.reciprocal_basis_Ainv.T
    z = min(s.fractional[2] for s in crystal.sites if s.element == "I")
    # Explicit manuscript plus orientation, independent of motif extraction.
    xyz = np.array([[0.0, 0.0, 0.0], [1 / 3, 2 / 3, -z], [2 / 3, 1 / 3, z]])
    atomic = []
    for element, label in (("Pb", "Pb1"), ("I", "I1"), ("I", "I1")):
        factor, _ = atomic_scattering_factor_e(
            species=element,
            element=element,
            charge=0,
            q_magnitude_Ainv=np.linalg.norm(q, axis=1),
            wavelength_A=waves,
        )
        radial, normal = profile.components_A2(label)
        atomic.append(
            factor * np.exp(-0.5 * (radial * (q[:, 0] ** 2 + q[:, 1] ** 2) + normal * q[:, 2] ** 2))
        )
    atomic = np.array(atomic).T
    transition = full_transition_matrix(law)
    expected = np.zeros((3, 3))
    for first, tail in product((0, 3), product(range(6), repeat=2)):
        path = (first, *tail)
        probability = (initial.plus, initial.minus)[first // 3] * np.prod(
            [transition[a, b] for a, b in pairwise(path)]
        )
        for window in range(3):
            amplitude = np.zeros(3, dtype=complex)
            for layer, state in enumerate(path):
                positions = xyz.copy()
                positions[:, 2] *= 1 if state < 3 else -1
                if layer == 0:
                    keep = (
                        np.ones(3, dtype=bool)
                        if window == 0
                        else positions[:, 2] >= 0
                        if window == 1
                        else positions[:, 2] > 0
                    )
                elif layer == 2:
                    keep = (
                        np.zeros(3, dtype=bool)
                        if window == 0
                        else positions[:, 2] < 0
                        if window == 1
                        else positions[:, 2] <= 0
                    )
                else:
                    keep = np.ones(3, dtype=bool)
                positions += np.array([state % 3 / 3, 2 * (state % 3) / 3, layer])
                amplitude += np.sum(
                    atomic[:, keep] * np.exp(2j * np.pi * hkl @ positions[keep].T), axis=1
                )
            expected[window] += probability * abs(amplitude) ** 2 * CLASSICAL_ELECTRON_RADIUS_A**2
    args = dict(h=hkl[:, 0], k=hkl[:, 1], L=hkl[:, 2], k_norm_Ainv=2 * np.pi / waves)
    np.testing.assert_allclose(
        model.evaluate_components_hkl(**args)[:, 0], expected, rtol=3e-13, atol=1e-18
    )
    per_layer = replace(model, normalization=EventIntensityNormalization.FINITE_PER_LAYER)
    np.testing.assert_allclose(
        per_layer.evaluate_components_hkl(**args)[:, 0], expected / 2, rtol=3e-13
    )
    changed = replace(
        profile, sites=tuple(replace(s, u_radial_A2=s.u_radial_A2 + 0.1) for s in profile.sites)
    )
    np.testing.assert_allclose(
        replace(model, site_displacement_profile=changed).evaluate_hkl(**args)[0],
        model.evaluate_hkl(**args)[0],
        rtol=3e-13,
    )


def test_axial_preparation_seeds_stencil_elastic_cutoffs_before_adaptation():
    from dataclasses import asdict

    from painted_ewald import MosaicParameters
    from rasim_next.fitting.bi_joint import BiJointModel
    from rasim_next.fitting.bi_native import BiNativeStructureModel
    from rasim_next.fitting.native_accuracy import compare_native_predictions
    from rasim_next.fitting.native_input import NativeSourceDefinition
    from rasim_next.fitting.native_joint import NativeJointEvaluator
    from rasim_next.fitting.native_observations import NativeFitObservations
    from rasim_next.fitting.native_search import FitParameter
    from rasim_next.fitting.native_workflow import (
        make_native_evaluator,
        prepare_native_axial_meshes,
    )
    from rasim_next.materials.optics import material_optics
    from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
    from rasim_next.pipeline.fiber_detector import AxialPanelMesh, elastic_axial_cutoff_Ainv

    physics = bi_physics()
    source_definition = NativeSourceDefinition(
        mean_origin_lab_m=np.array([0.0, 0.0, 0.05]),
        mean_direction_lab=np.array([0.0, 0.0, -1.0]),
        transverse_axes_lab=np.eye(3)[:2],
        spatial_sigma_m=np.array([0.001, 0.002]),
        divergence_sigma_rad=np.array([0.08, 0.04]),
        line_wavelength_A=np.array([1.54, 1.544]),
        line_probability=np.array([0.66, 0.34]),
        common_wavelength_sigma_A=0.0,
        polarization_state_id="UNITY_APPROXIMATION",
        kind="gauss_hermite",
        position_divergence_correlation=(-0.3, 0.4),
        divergence_order=1,
        wavelength_order=1,
        local_m0_divergence_order=2,
    )
    source = source_definition.sample()
    physics = replace(
        physics,
        source=source,
        source_definition=source_definition,
        material=material_optics(physics.structure.crystals[0], source.mean_rays.wavelength_A),
    )
    atomic = BiNativeStructureModel(physics)
    center = np.r_[atomic.reference_parameters.as_array(), 0.5, 0.55, 0.4, 0.2, 0.6, 50, 2, 4]
    rows, columns = physics.instrument.detector_shape_rc
    flat = np.arange(rows * columns)
    projection = NativePixelRegionProjection(
        (rows, columns),
        flat,
        (flat // columns >= rows // 2).astype(int) * 2 + (flat % columns >= columns // 2),
        flat,
        np.ones(len(flat)),
        4,
        "elastic-cutoff-quadrants",
    )
    observations = NativeFitObservations(
        projection,
        np.ones(4),
        np.ones(4, dtype=bool),
        np.eye(4),
        np.eye(4),
        np.ones(4),
        np.eye(4),
        np.array([0, 4]),
        np.array([1e6]),
        "elastic-cutoff-regression",
        allow_guard_constraints=False,
    )
    evaluator = NativeJointEvaluator(
        BiJointModel(atomic), observations, MosaicParameters(0.5, 0.6, 0.4)
    )
    candidates = np.tile(center, (3, 1))
    candidates[:, 0] += [0.0, 1e-7, -1e-7]
    candidates[:, 14] = [0.55, 0.5501, 0.6]
    parameters = tuple(
        FitParameter(name, unit, "specimen:Te", float(value - 0.01), float(value + 0.01), 0.001)
        for name, unit, value in zip(
            evaluator.parameter_names, evaluator.parameter_units, center, strict=True
        )
    )
    parameters = tuple(
        replace(parameter, lower=0.5, upper=0.7, sensitivity_scale=0.1) if i == 14 else parameter
        for i, parameter in enumerate(parameters)
    )
    tolerances = dict(
        maximum_whitened_rms=0.1,
        maximum_contrast_rms=0.05,
        maximum_objective_contrast_error=0.5,
    )
    plan = dict(
        parameters=[asdict(parameter) for parameter in parameters],
        proposal_mosaic=[0.5, 0.6, 0.4],
        fit_instrument=False,
        acquisition_id="elastic-cutoff-regression",
        workers=1,
        repeat_choices=[3],
        qualification_candidates=candidates.tolist(),
        numerical_tolerances=tolerances,
    )
    seeds = tuple(
        AxialPanelMesh(rods, coordinate, (0.0, 7.5, 7.8, 8.0, 8.2, 9.0))
        for rods, coordinate in (
            (((0, 0),), "external_local_m0_q"),
            (((1, 0),), "positive_phase_axial"),
        )
    )
    physics = replace(
        physics,
        integration_rule=replace(
            physics.integration_rule,
            axial_meshes=seeds,
            quadrature_kind="composite_gauss",
            angular_support="fixed_union",
        ),
    )

    expected = [set() for _ in seeds]
    for candidate in candidates:
        bound, arguments, _, _ = evaluator.bind(candidate, 3)
        for part in bound.integration_parts():
            detector = part.detector(mosaic=evaluator.proposal_mosaic, **arguments)
            basis = detector.reciprocal_basis_Ainv
            normal = basis[:, 2] / np.linalg.norm(basis[:, 2])
            for i, mesh in enumerate(seeds):
                if not set(mesh.rods_hk).issubset({(rod.h, rod.k) for rod in part.rods}):
                    continue
                h, k = mesh.rods_hk[0]
                anchor = h * basis[:, 0] + k * basis[:, 1]
                radial = np.linalg.norm(anchor - (anchor @ normal) * normal)
                incident = detector.incident.states
                ki = (
                    2 * np.pi / incident.wavelength_A[:, None] * incident.direction_sample
                    if mesh.coordinate == "external_local_m0_q"
                    else incident.k_film_phase_sample_Ainv
                )
                reachable = 4 * np.sum(ki**2, axis=1) - radial**2
                expected[i].update(np.sqrt(reachable[reachable > 0]))

    fixed_scale = 54935822361.00364

    def predict(meshes):
        rule = replace(physics.integration_rule, axial_meshes=tuple(meshes))
        current = make_native_evaluator(replace(physics, integration_rule=rule), observations, plan)
        result = np.array([current.predict(value, 3) for value in candidates])
        current.clear_responses()
        return result

    reference_meshes = tuple(
        replace(
            mesh,
            edges_Ainv=tuple(
                np.unique(np.r_[mesh.edges_Ainv, sorted(edges), np.arange(7.7, 8.251, 0.002)])
            ),
        )
        for mesh, edges in zip(seeds, expected, strict=True)
    )
    reference = predict(reference_meshes)
    counts = fixed_scale * reference[-1]
    observations = replace(
        observations,
        net_count=counts,
        fit_target=counts,
        covariance_count2=np.diag(np.maximum(counts, 1)),
    )
    missed = compare_native_predictions(
        observations, predict(seeds), reference, fixed_scale=fixed_scale, **tolerances
    )
    assert not missed["empirical_agreement"]

    accepted, report = prepare_native_axial_meshes(
        physics,
        observations,
        plan,
        seeds,
        fixed_scale=fixed_scale,
        maximum_panels=512,
        maximum_passes=24,
    )
    assert report["comparison"]["empirical_agreement"]
    assert [len(row["inserted_edges_Ainv"]) for row in report["elastic_endpoint_seeding"]] == [
        2,
        6,
    ]
    assert elastic_axial_cutoff_Ainv(ki_sample_Ainv=[1.0, 0.0, 0.0], radial_Ainv=2.0) == 0
    assert elastic_axial_cutoff_Ainv(ki_sample_Ainv=[1.0, 0.0, 0.0], radial_Ainv=2.1) is None
    actual = predict(accepted.axial_meshes)
    restored = compare_native_predictions(
        observations, actual, reference, fixed_scale=fixed_scale, **tolerances
    )
    assert restored["empirical_agreement"]
    for seed, mesh, endpoints in zip(seeds, accepted.axial_meshes, expected, strict=True):
        assert set(seed.edges_Ainv).issubset(mesh.edges_Ainv)
        for endpoint in endpoints:
            assert np.any(np.abs(np.asarray(mesh.edges_Ainv) - endpoint) <= 1e-10)

    off_stencil = candidates[0].copy()
    off_stencil[0] += 0.004
    bound, arguments, _, _ = evaluator.bind(off_stencil, 3)
    regular = next(part for part in bound.integration_parts() if any(rod.h for rod in part.rods))
    detector = regular.detector(mosaic=evaluator.proposal_mosaic, **arguments)
    basis = detector.reciprocal_basis_Ainv
    normal = basis[:, 2] / np.linalg.norm(basis[:, 2])
    anchor = basis[:, 0]
    radial = np.linalg.norm(anchor - (anchor @ normal) * normal)
    ki = detector.incident.states.k_film_phase_sample_Ainv[0]
    moved_endpoint = np.sqrt(4 * (ki @ ki) - radial**2)
    assert np.all(np.abs(np.asarray(accepted.axial_meshes[1].edges_Ainv) - moved_endpoint) > 1e-10)
