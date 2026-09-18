"""Frozen-input adoption and recoverable rendering cross the public file boundary."""

import hashlib
import importlib.util
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_joint import BiJointModel
from rasim_next.fitting.bi_native import BiNativeStructureModel
from rasim_next.fitting.native_input import load_native_fit_physics
from rasim_next.fitting.native_joint import NativeJointEvaluator
from rasim_next.fitting.native_observations import load_native_fit_observations
from rasim_next.fitting.native_workflow import make_native_evaluator, native_physics_with
from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
from rasim_next.pipeline.conditional_detector import ConditionalStructureDetector
from rasim_next.proof.diagnostics import write_diagnostic


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiment(directory):
    from test_native_fitting import bi_physics

    directory.mkdir()
    physics = bi_physics()

    def plain(value):
        if hasattr(value, "__dataclass_fields__"):
            return {f.name: plain(getattr(value, f.name)) for f in fields(value) if f.init}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    record = plain(physics)
    for name in ("input_revision", "source_definition"):
        record.pop(name)
    record["schema"] = "rasim-native-fit-physics-v1"
    material = record["material"]
    material.pop("n_complex")
    material["n_real"] = physics.material.n_complex.real.tolist()
    material["n_imag"] = physics.material.n_complex.imag.tolist()
    record["source"] = dict(
        mean_origin_lab_m=[0, 0, 0.05],
        mean_direction_lab=[0, 0, -1],
        transverse_axes_lab=[[1, 0, 0], [0, 1, 0]],
        spatial_sigma_m=[0.001, 0.002],
        divergence_sigma_rad=[0.08, 0.04],
        position_divergence_correlation=[-0.3, 0.4],
        line_wavelength_A=[1.54, 1.544],
        line_probability=[0.66, 0.34],
        common_wavelength_sigma_A=0,
        polarization_state_id="UNITY_APPROXIMATION",
    )
    record["source_rule"] = dict(kind="latin_hypercube", sample_count=12, seed=723)
    record["specular_stitch_stack"]["substrate_refractive_index"] = [0.99998, 1e-7]
    physics_path = directory / "physics.json"
    physics_path.write_text(json.dumps(record), encoding="utf-8")
    shape = physics.instrument.detector_shape_rc
    raw = 10 + np.arange(np.prod(shape)).reshape(shape) % 17
    raw_path = directory / "raw.npz"
    np.savez(raw_path, detector_native_counts=raw)
    signal = NativePixelRegionProjection(
        shape,
        np.array([3400, 3401, 3500]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 2]),
        np.array([1.0, 0.5, 0.25, 1.0]),
        2,
        "frozen-signal",
    )
    control = NativePixelRegionProjection(
        shape, np.array([0]), np.array([0]), np.array([0]), np.array([1.0]), 1, "frozen-control"
    )
    measured, covariance = signal.integrate_field(raw, np.maximum(raw, 1))
    arrays = dict(
        measured=measured,
        count_covariance=covariance,
        background=np.array([2.0, 3.0]),
        frozen_net=measured - [2, 3],
        background_modes=np.array([[0.3, 0.7]]),
        valid=np.ones(2, dtype=bool),
        frozen_fit_operator=np.eye(2),
        frozen_target=measured - [2, 3],
        frozen_guard_operator=np.eye(2),
        frozen_guard_pointer=np.array([0, 2]),
        frozen_guard_limit=np.array([1e6]),
        control_variance=np.array([raw[0, 0]]),
        control_split=np.array([0]),
    )

    def projection_record(projection, prefix):
        names = (
            "flat_pixel_index",
            "observation_row",
            "pixel_column_index",
            "detector_area_weight_px2",
        )
        arrays.update({prefix + name: getattr(projection, name) for name in names})
        return dict(
            detector_shape_rc=shape,
            observation_count=projection.observation_count,
            quadrature_revision=projection.quadrature_revision,
            projection_revision=projection.projection_revision,
            arrays={name: prefix + name for name in names},
        )

    signal_record = projection_record(signal, "signal_")
    control_record = projection_record(control, "control_")
    array_path = directory / "observations.npz"
    np.savez(array_path, **arrays)

    def reference(path):
        return dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    record = dict(
        schema="rasim-native-fit-observations-v1",
        sample_id=physics.sample_id,
        arrays=reference(array_path),
        physical_input=reference(physics_path),
        raw_acquisition=dict(kind="decoded_detector_native_npz", **reference(raw_path)),
        signal_projection=signal_record,
        background=dict(
            dark=dict(kind="none"),
            controls=dict(
                projection=control_record,
                revision="frozen-control",
                arrays=dict(measurement_variance_count2="control_variance", split="control_split"),
            ),
            guard_controls=[],
        ),
    )
    observation_path = directory / "observations.json"
    observation_path.write_text(json.dumps(record), encoding="utf-8")
    return physics_path, observation_path


def test_preparation_preserves_shared_pixel_covariance_and_rejects_changed_raw(tmp_path):
    physics, source = experiment(tmp_path / "source")
    prepare = script("prepare_native").prepare
    output = prepare(source, tmp_path / "portable")
    old, adopted = map(load_native_fit_observations, (source, output))
    np.testing.assert_array_equal(adopted.net_count, old.net_count)
    np.testing.assert_array_equal(adopted.covariance_count2, old.covariance_count2)
    assert adopted.covariance_count2[0, 1] > 0.21
    assert prepare(source, output.parent) == output
    record = json.loads(output.read_bytes())
    assert (
        load_native_fit_physics(output.parent / record["physical_input"]["path"]).input_revision
        == hashlib.sha256(physics.read_bytes()).hexdigest()
    )
    assert all(
        not Path(record[k]["path"]).is_absolute()
        for k in ("arrays", "physical_input", "raw_acquisition")
    )
    (source.parent / "raw.npz").write_bytes(b"changed acquisition")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_fixed_sf_control_keeps_declared_values_and_full_release_remains_required(
    tmp_path, monkeypatch
):
    physics_path, observation_path = experiment(tmp_path / "inputs")
    physics = load_native_fit_physics(physics_path)
    observations = load_native_fit_observations(observation_path)
    atomic = BiNativeStructureModel(physics)
    values = np.r_[atomic.reference_parameters.as_array(), 0.1, 0.2, 0.3, 0.2, 0.6, 50, 2, 4]
    evaluator = NativeJointEvaluator(
        BiJointModel(atomic), observations, MosaicParameters(0.15, 0.25, 0.4)
    )
    names = evaluator.parameter_names
    acquisition = json.loads(observation_path.read_bytes())["raw_acquisition"]["sha256"]
    plan = dict(
        schema="rasim-native-refinement-plan-v1",
        fit_instrument=False,
        workers=1,
        acquisition_id=acquisition,
        proposal_mosaic=[0.15, 0.25, 0.4],
        parameters=[
            dict(
                name=name,
                unit=unit,
                owner="specimen:" + physics.sample_id,
                lower=float(value - 0.01),
                upper=float(value + 0.01),
                sensitivity_scale=0.001,
            )
            for name, unit, value in zip(names, evaluator.parameter_units, values, strict=True)
        ],
        starts=[values.tolist()],
        repeat_choices=[2],
        finite_difference_step=1e-4,
        fixed_parameters=dict(zip(names[:13], values[:13], strict=True)),
        stages=[
            dict(
                name="control",
                active_parameters=list(names[13:]),
                maximum_iterations=1,
                maximum_function_evaluations=1,
                enforce_historical_guards=False,
            )
        ],
        numerical_checks=[],
        require_initial_qualification=False,
        profiles=[],
        sensitivity=False,
        controls=False,
        synthetic=dict(truth=values.tolist(), coherent_repeats=2, scale=7, add_noise=False),
        numerical_tolerances=dict(
            maximum_whitened_rms=0.1,
            maximum_contrast_rms=0.05,
            maximum_objective_contrast_error=0.5,
        ),
    )
    plan["stages"][0]["reference_correction"] = dict(
        numerical_override={},
        trust_radii=[0.005] * len(values),
        maximum_updates=1,
        maximum_function_evaluations=1,
    )
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "control.ra_diag.npz"
    runner = script("refine_native")
    # This compact fixture owns two rods; full catalogue coverage has its own proof.
    monkeypatch.setattr(runner, "validate_native_rod_coverage", lambda *a, **k: {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "refine_native",
            "--physics",
            str(physics_path),
            "--observations",
            str(observation_path),
            "--plan",
            str(plan_path),
            "--output",
            str(output),
        ],
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runner.main()
    with np.load(output, allow_pickle=False) as saved:
        manifest = json.loads(saved["manifest_json"].tobytes())
        assert manifest["fit_scope"] == "fixed_parameter_control"
        assert manifest["fixed_parameters"] == plan["fixed_parameters"]
        assert manifest["selected"] is None
        acceleration = manifest["reference_acceleration"][0]
        assert acceleration["acceptance"] == "exact_checked_warm_start_only"
        assert acceleration["records"][0]["comparison"]["empirical_agreement"]
        candidates = saved["prediction_values"]
        assert len(candidates) > 1
        np.testing.assert_array_equal(
            candidates[:, :13], np.tile(values[:13], (len(candidates), 1))
        )
    fixed = plan.pop("fixed_parameters")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="final fit stage must release every"):
        runner.main()
    plan["fixed_parameters"] = fixed
    changed = values.copy()
    changed[0] += 0.001
    plan["qualification_candidates"] = [values.tolist(), changed.tolist()]
    plan["numerical_checks"] = [dict(name="angular", integration=dict(angular_power=2))]
    plan["qualification_scale"] = 7
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output.unlink()
    with pytest.raises(ValueError, match="qualification cannot change"):
        runner.main()

    changed = values.copy()
    changed[14] += 0.001
    plan.update(
        stages=[],
        source_override=dict(kind="gauss_hermite", divergence_order=2, wavelength_order=1),
        qualification_candidates=[values.tolist(), changed.tolist()],
        numerical_checks=[dict(name="local_source", source=dict(local_m0_divergence_order=3))],
        numerical_tolerances=dict(
            maximum_whitened_rms=0.1,
            maximum_contrast_rms=0.05,
            maximum_objective_contrast_error=0.5,
        ),
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output.unlink(missing_ok=True)
    runner.main()
    with np.load(output, allow_pickle=False) as saved:
        manifest = json.loads(saved["manifest_json"].tobytes())
        check = manifest["numerical_checks"][0]
        assert check["name"] == "local_source"
        assert check["source_partitions_candidate_index"] == 1
        assert [p["divergence_order"] for p in check["source_partitions"]] == [2, 3]
        assert [p["row_count"] for p in check["source_partitions"]] == [8, 18]

    from rasim_next.pipeline.fiber_detector import AxialPanelMesh

    meshes = [
        dict(rods_hk=[[0, 0]], coordinate="external_local_m0_q", edges_Ainv=[0, 7.5, 8, 8.5, 9]),
        dict(rods_hk=[[1, 0]], coordinate="positive_phase_axial", edges_Ainv=[0, 7.5, 8, 8.5, 9]),
    ]
    plan.update(
        source_override=dict(kind="gauss_hermite", divergence_order=1, wavelength_order=1),
        integration_override=dict(
            angular_power=0,
            quadrature_kind="composite_gauss",
            maximum_axial_panel_width_Ainv=1.0,
            local_m0_maximum_axial_panel_width_Ainv=0.5,
        ),
        axial_adaptation=dict(initial_meshes=meshes, fixed_scale=1),
        numerical_tolerances=dict(
            maximum_whitened_rms=1e6,
            maximum_contrast_rms=1e6,
            maximum_objective_contrast_error=1e6,
        ),
    )
    # CLI preparation owns finite evidence/provenance, not fit qualification.
    plan.pop("synthetic")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output.unlink()
    monkeypatch.setattr("sys.argv", [*sys.argv, "--prepare-axial-mesh"])
    runner.main()
    with np.load(output, allow_pickle=False) as saved:
        manifest = json.loads(saved["manifest_json"].tobytes())
        assert manifest["acceptance"] == "empirical_mesh_agreement_only"
        assert manifest["report"]["comparison"]["empirical_agreement"]
        assert manifest["implementation"]["source_sha256"]
        mesh_plan = dict(integration_override=manifest["integration_override"])
        bound = native_physics_with(physics, mesh_plan, {})
        assert all(isinstance(m, AxialPanelMesh) for m in bound.integration_rule.axial_meshes)
        for mesh, original in zip(bound.integration_rule.axial_meshes, meshes, strict=True):
            cap = 0.5 if mesh.coordinate == "external_local_m0_q" else 1.0
            assert np.max(np.diff(mesh.edges_Ainv)) <= cap * (1 + 1e-12)
            assert set(original["edges_Ainv"]).issubset(mesh.edges_Ainv)
        with pytest.raises(ValueError, match="ignored by the effective mesh"):
            native_physics_with(physics, mesh_plan, dict(integration=dict(axial_peak_spacing_L=2)))
        with pytest.raises(ValueError, match="refine mesh edges"):
            native_physics_with(physics, mesh_plan, dict(integration=dict(axial_power=4)))
    plan["axial_adaptation"]["maximum_panels"] = 4
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output.unlink()
    with pytest.raises(ValueError, match="seed panel widths exceed"):
        runner.main()
    plan["axial_adaptation"].pop("maximum_panels")
    plan["repeat_choices"] = [2, 3]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output.unlink(missing_ok=True)
    with pytest.raises(ValueError, match="prepare each N separately"):
        runner.main()


def test_local_source_rule_preserves_raw_partition_sum_and_physical_source_rebinding(tmp_path):
    from rasim_next.fitting.native_instrument import NativeInstrumentModel

    physics_path, observation_path = experiment(tmp_path / "source-parts")
    original = load_native_fit_physics(physics_path)
    observations = load_native_fit_observations(observation_path)
    physics = native_physics_with(
        original,
        {},
        {
            "source": {
                "kind": "gauss_hermite",
                "divergence_order": 2,
                "wavelength_order": 1,
                "local_m0_divergence_order": 3,
            }
        },
    )
    atomic = BiNativeStructureModel(physics)
    instrument = NativeInstrumentModel(physics, "raw")
    values = np.r_[
        atomic.reference_parameters.as_array(),
        0.1,
        0.2,
        0.3,
        0.2,
        0.6,
        50,
        2,
        4,
        instrument.initial_values,
    ]
    proposal = MosaicParameters(0.15, 0.25, 0.4)
    evaluator = NativeJointEvaluator(BiJointModel(atomic), observations, proposal, instrument)
    for source_change in (False, True):
        if source_change:
            values[31] *= 1.1
            values[33] += 0.05
        bound, arguments, mosaic, stack = evaluator.bind(values, 2)
        expected = np.zeros(len(observations.net_count))
        for local, order in ((False, 2), (True, 3)):
            part = native_physics_with(
                bound,
                {},
                {"source": {"local_m0_divergence_order": None, "divergence_order": order}},
            )
            part = replace(part, rods=tuple(r for r in part.rods if (r.h == r.k == 0) == local))
            response = part.detector(mosaic=proposal, **arguments).compile_native_response(
                observations.projection
            )
            expected += response.evaluate(mosaic=mosaic, specular_stitch_stack=stack)
        np.testing.assert_allclose(evaluator.predict(values, 2), expected, rtol=3e-13, atol=0)
        assert evaluator.compile_count == (4 if source_change else 2)
        with pytest.raises(ValueError, match="integration_parts"):
            bound.detector(mosaic=proposal, **arguments)
    values[35] -= 0.03
    evaluator.predict(values, 2)
    assert evaluator.compile_count == 4
    same = replace(physics.source_definition, local_m0_divergence_order=2)
    equal = replace(physics, source_definition=same)
    assert equal.integration_parts()[0] is equal
    default = replace(equal, source_definition=replace(same, local_m0_divergence_order=None))
    for candidate in (equal, default):
        ev = NativeJointEvaluator(
            BiJointModel(BiNativeStructureModel(candidate)),
            observations,
            proposal,
            NativeInstrumentModel(candidate, "raw"),
        )
        raw = ev.predict(values, 2)
        if candidate is equal:
            equal_raw = raw
        else:
            np.testing.assert_array_equal(raw, equal_raw)
    with pytest.raises(ValueError, match="local_m0_divergence_order"):
        replace(same, local_m0_divergence_order=True)
    with pytest.raises(ValueError, match="Gauss-Hermite"):
        replace(same, kind="latin_hypercube")


def test_render_recovery_preserves_proposal_roughness_scale_and_candidate_status(
    tmp_path, monkeypatch
):
    physics_path, observation_path = experiment(tmp_path / "inputs")
    physics = load_native_fit_physics(physics_path)
    source_override = dict(
        kind="gauss_hermite", divergence_order=2, wavelength_order=1, local_m0_divergence_order=3
    )
    physics = native_physics_with(physics, {"source_override": source_override}, {})
    observations = load_native_fit_observations(observation_path)
    atomic = BiNativeStructureModel(physics)
    values = np.r_[atomic.reference_parameters.as_array(), 0.1, 0.2, 0.3, 0.2, 0.6, 50, 2, 4]
    proposal = MosaicParameters(0.15, 0.25, 0.4)
    evaluator = NativeJointEvaluator(BiJointModel(atomic), observations, proposal)
    plan = dict(
        source_override=source_override,
        fit_instrument=False,
        workers=1,
        acquisition_id="raw",
        proposal_mosaic=[0.15, 0.25, 0.4],
        parameters=[
            dict(
                name=name,
                unit=unit,
                owner="specimen:" + physics.sample_id,
                lower=float(v - 0.01),
                upper=float(v + 0.01),
                sensitivity_scale=0.001,
            )
            for name, unit, v in zip(
                evaluator.parameter_names, evaluator.parameter_units, values, strict=True
            )
        ],
    )
    evaluator = make_native_evaluator(physics, observations, plan)
    expected = 7 * evaluator.predict(values, 2)
    point = dict(parameters=values.tolist(), N=2, scale=7)
    result = tmp_path / "fit.ra_diag.npz"
    write_diagnostic(
        result,
        arrays=dict(optimizer_candidate_prediction_count=expected),
        manifest=dict(
            schema="rasim-native-refinement-result-v2",
            physics_input_revision=physics.input_revision,
            observation_input_revision=observations.input_revision,
            optimizer_candidate=point,
            selected=None,
            plan=plan,
            numerical_status="not_qualified",
        ),
        repository_root=Path(__file__).resolve().parents[1],
    )
    render = script("render_native").render
    output = tmp_path / "resumed.ra_diag.npz"
    options = dict(candidate=True, full_image=True, checkpoint_seconds=1e-12)
    original = ConditionalStructureDetector.iter_native_pixel_batches

    def interrupted(self, **kwargs):
        for index, pixels in enumerate(original(self, **kwargs)):
            if index == 1 and all(r.h == r.k == 0 for r in self.rods):
                raise RuntimeError("simulated interrupted render")
            yield pixels

    with monkeypatch.context() as patch:
        patch.setattr(ConditionalStructureDetector, "iter_native_pixel_batches", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            render(physics_path, observation_path, result, output, **options)
    with np.load(output, allow_pickle=False) as data:
        saved = json.loads(data["manifest_json"].tobytes())
        assert saved["partition_finished"] == [True, False] and not saved["complete"]
        assert saved["partition_completed_batches"][1] == 1
        assert saved["completed_batches"] == sum(saved["partition_completed_batches"])
    render(physics_path, observation_path, result, output, resume=True, **options)
    direct = tmp_path / "direct.ra_diag.npz"
    render(physics_path, observation_path, result, direct, **options)
    with (
        np.load(output, allow_pickle=False) as resumed,
        np.load(direct, allow_pickle=False) as fresh,
    ):
        np.testing.assert_array_equal(
            resumed["simulated_detector_native_count"], fresh["simulated_detector_native_count"]
        )
        manifest = json.loads(resumed["manifest_json"].tobytes())
        assert manifest["complete"] and manifest["status"] == "unqualified_optimizer_candidate"
        assert manifest["image_numerical_status"] == "not_qualified"
    with pytest.raises(ValueError, match="no requested selected"):
        render(physics_path, observation_path, result, tmp_path / "selected.ra_diag.npz")
