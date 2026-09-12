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
    from rasim_next.fitting.bi_joint import BiJointCandidate, fit_bi_joint
    from rasim_next.fitting.bi_native import BiNativeStructureModel
    from rasim_next.fitting.native_observations import NativeFitObservations
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
            self.seen.append(candidate.values)
            return np.r_[1, 2 + (candidate.values - lower) / (upper - lower)]

    evaluator = KnownNativeObservable()
    result = fit_bi_joint(
        evaluator,
        BiJointCandidate(values, 3),
        lower=lower,
        upper=upper,
        maximum_iterations=100,
        finite_difference_step=1e-5,
    )
    assert result.success
    assert len(result.active_parameters) == 21
    assert np.all(np.ptp(np.array(evaluator.seen), axis=0) > 0)
    np.testing.assert_allclose(
        (result.candidate.values - lower) / (upper - lower), truth, atol=2e-3
    )
    assert result.scores["guards_pass"]
    assert result.numerical_status == "not_qualified"


def test_native_bi_reuse_invalidates_changed_density_and_cell():
    from painted_ewald import MosaicParameters
    from rasim_next.fitting.bi_joint import BiJointCandidate, BiNativeFitEvaluator
    from rasim_next.fitting.bi_native import BiNativeStructureModel
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
    evaluator = BiNativeFitEvaluator(model, observations, MosaicParameters(0.5, 0.6, 0.4), 2)
    initial = evaluator.predict(BiJointCandidate(values, 3))
    assert np.any(initial > 0)
    assert not initial.flags.writeable
    np.testing.assert_array_equal(evaluator.predict(BiJointCandidate(values, 3)), initial)
    assert evaluator.evaluation_count == 1
    values[2] += 0.001
    values[18:] += 1
    reused = evaluator.predict(BiJointCandidate(values, 3))
    assert evaluator.compile_count == 1
    independent = BiNativeFitEvaluator(model, observations, evaluator.proposal_mosaic)
    np.testing.assert_allclose(
        reused, independent.predict(BiJointCandidate(values, 3)), rtol=3e-13, atol=0
    )
    values[4] -= 0.03
    density = evaluator.predict(BiJointCandidate(values, 3))
    assert evaluator.compile_count == 2
    np.testing.assert_allclose(
        density, independent.predict(BiJointCandidate(values, 3)), rtol=3e-13, atol=0
    )
    values[0] += 0.01
    evaluator.predict(BiJointCandidate(values, 3))
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
    model = CifFiniteStackStrength(crystal, 3, site_displacement_tensors_A2=tensors)
    hkl = np.array([[0, 0, 0.37], [1, 0, 1.27], [-1, 1, -2.38], [2, -1, 3.14]])
    wave = np.array([1.54, 1.544, 1.54, 1.544])
    q = hkl @ model.reciprocal_basis_Ainv.T
    amplitude = np.zeros(4, dtype=complex)
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
    np.testing.assert_allclose(
        actual, abs(amplitude) ** 2 * CLASSICAL_ELECTRON_RADIUS_A**2, rtol=3e-13, atol=2e-18
    )
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
