"""Frozen-input adoption and recoverable rendering cross the public file boundary."""

import hashlib
import importlib.util
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_joint import BiJointModel
from rasim_next.fitting.bi_native import BiNativeStructureModel
from rasim_next.fitting.native_input import load_native_fit_physics
from rasim_next.fitting.native_joint import NativeJointEvaluator
from rasim_next.fitting.native_observations import load_native_fit_observations
from rasim_next.fitting.native_workflow import make_native_evaluator
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


def test_render_recovery_preserves_proposal_roughness_scale_and_candidate_status(
    tmp_path, monkeypatch
):
    physics_path, observation_path = experiment(tmp_path / "inputs")
    physics = load_native_fit_physics(physics_path)
    observations = load_native_fit_observations(observation_path)
    atomic = BiNativeStructureModel(physics)
    values = np.r_[atomic.reference_parameters.as_array(), 0.1, 0.2, 0.3, 0.2, 0.6, 50, 2, 4]
    proposal = MosaicParameters(0.15, 0.25, 0.4)
    evaluator = NativeJointEvaluator(BiJointModel(atomic), observations, proposal)
    plan = dict(
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
            if index == 1:
                raise RuntimeError("simulated interrupted render")
            yield pixels

    with monkeypatch.context() as patch:
        patch.setattr(ConditionalStructureDetector, "iter_native_pixel_batches", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            render(physics_path, observation_path, result, output, **options)
    with np.load(output, allow_pickle=False) as data:
        saved = json.loads(data["manifest_json"].tobytes())
        assert saved["completed_batches"] == 1 and not saved["complete"]
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
