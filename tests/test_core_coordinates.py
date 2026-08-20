from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rasim_next.core import contracts
from rasim_next.core.contracts import (
    IncidentSampleBatch,
    canonical_revision_sha256,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.interfaces import scalar_interface_amplitude
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
from rasim_next.core.traces import Measure, QuantityKind, TraceRecord
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.core.wave_modes import normal_wavevector, select_normal_wavevector
from rasim_next.io.orientation import (
    DetectorIndex,
    OscRawIndex,
    detector_native_to_raw,
    detector_to_raw_index,
    raw_to_detector_index,
    raw_to_detector_native,
)
from rasim_next.proof import __main__ as proof_cli
from rasim_next.proof import tolerances as stage_tolerances
from rasim_next.proof.diagnostics import write_diagnostic
from rasim_next.proof.traces import Tolerance, compare_traces


def test_shared_coordinate_and_optical_primitives() -> None:
    raw = np.arange(35, dtype=np.uint32).reshape(5, 7)
    native = raw_to_detector_native(raw)
    raw_index = OscRawIndex(2, 5)
    detector_index = raw_to_detector_index(raw_index, raw.shape)
    assert detector_index == DetectorIndex(5, 2)
    assert detector_to_raw_index(detector_index, raw.shape) == raw_index
    np.testing.assert_array_equal(native, np.rot90(raw, -1))
    np.testing.assert_array_equal(detector_native_to_raw(native), raw)

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = RigidTransform(rotation, np.array([1.0, 2.0, 3.0]), FrameId.CRYSTAL, FrameId.SAMPLE)
    point = np.array([2.0, 0.0, 1.0])
    np.testing.assert_allclose(transform.apply_point(point), [1.0, 4.0, 4.0])
    np.testing.assert_allclose(transform.apply_vector(point), [0.0, 2.0, 1.0])
    np.testing.assert_allclose(transform.inverse().apply_point(transform.apply_point(point)), point)

    kz = normal_wavevector(
        k0_Ainv=2.0,
        refractive_index=1.0,
        k_parallel_Ainv=np.array([1.0, 0.0]),
        propagation_direction=1,
    )
    assert kz**2 + 1.0 == pytest.approx(4.0 + 0.0j)
    assert select_normal_wavevector(-1e-32, -1) == pytest.approx(-1e-16j)
    assert scalar_interface_amplitude(2.0, 2.0) == pytest.approx(1.0)


def test_source_and_incident_contracts_preserve_identity_and_failure_payloads() -> None:
    sample_ids = np.array([10, 11], dtype=np.int64)
    origins = np.zeros((2, 3))
    directions = np.tile([1.0, 0.0, 0.0], (2, 1))
    wavelengths = np.ones(2)
    weights = np.full(2, 0.5)
    polarization_ids = ("linear", "linear")
    provenance = "minimal core contract fixture.v1"
    samples = IncidentSampleBatch(
        incident_sample_id=sample_ids,
        origin_lab_m=origins,
        direction_lab=directions,
        wavelength_A=wavelengths,
        source_weight=weights,
        polarization_state_id=polarization_ids,
        source_sampling_model_id="explicit_test_source.v1",
        source_rng_model_id="no_rng.v1",
        source_seed=0,
        source_parameter_provenance=provenance,
    )
    assert samples.incident_sample_id.size == 2
    assert not samples.origin_lab_m.flags.writeable
    states = contracts.IncidentStateBatch(
        incident_state_id=np.array([20, 21]),
        incident_sample_id=samples.incident_sample_id,
        source_origin_lab_m=samples.origin_lab_m,
        source_direction_lab=samples.direction_lab,
        sample_intersection_lab_m=np.zeros((2, 3)),
        direction_sample=samples.direction_lab,
        k_air_sample_Ainv=2.0 * np.pi * samples.direction_lab,
        k_film_phase_sample_Ainv=2.0 * np.pi * samples.direction_lab,
        kz_film_Ainv=np.zeros(2, dtype=np.complex128),
        entrance_amplitude=np.ones(2, dtype=np.complex128),
        footprint_acceptance=np.ones(2),
        source_weight=samples.source_weight,
        wavelength_A=samples.wavelength_A,
        polarization_state_id=samples.polarization_state_id,
        status=(ValidityCode.VALID, ValidityCode.VALID),
        valid=np.ones(2, dtype=np.bool_),
        source_sampling_model_id=samples.source_sampling_model_id,
        source_rng_model_id=samples.source_rng_model_id,
        source_seed=samples.source_seed,
        source_parameter_provenance=samples.source_parameter_provenance,
        source_parameter_revision=samples.source_parameter_revision,
        source_weight_revision=samples.source_weight_revision,
        source_revision=samples.source_revision,
        sample_geometry_revision=canonical_revision_sha256(
            ("sample_geometry", "minimal core contract fixture.v1")
        ),
        material_revision=canonical_revision_sha256(
            ("material", "minimal core contract fixture.v1")
        ),
        incident_model_id="one_transmitted_channel.v1",
    )
    weighted_samples = replace(samples, source_weight=np.array([0.25, 0.75]))
    np.testing.assert_array_equal(weighted_samples.source_weight, [0.25, 0.75])
    assert weighted_samples.source_revision != samples.source_revision
    assert weighted_samples.source_weight_revision != samples.source_weight_revision
    with pytest.raises(ValueError, match="source_weight_revision"):
        replace(states, source_weight=np.array([0.25, 0.75]))
    changed_weight_revision = canonical_revision_sha256(
        ("incident_sample_id", states.incident_sample_id),
        ("source_weight", np.array([0.25, 0.75])),
    )
    with pytest.raises(ValueError, match="source_revision"):
        replace(
            states,
            source_weight=np.array([0.25, 0.75]),
            source_weight_revision=changed_weight_revision,
        )
    for batch in (samples, states):
        with pytest.raises(ValueError, match="summing to one"):
            replace(batch, source_weight=np.array([0.25, 0.5]))
        for invalid_weight in (
            (True, 0.0),
            np.asarray((True, False)),
            np.asarray((0.25 + 0.1j, 0.75)),
            np.asarray(("0.25", "0.75")),
        ):
            with pytest.raises(ValueError, match="real numbers"):
                replace(batch, source_weight=invalid_weight)

    assert states.incident_state_id.tolist() == [20, 21]
    assert states.incident_sample_id.tolist() == [10, 11]
    assert states.polarization_state_id == polarization_ids
    assert all(
        len(getattr(states, name)) == 64
        for name in (
            "source_parameter_revision",
            "source_weight_revision",
            "source_revision",
            "sample_geometry_revision",
            "material_revision",
        )
    )
    assert not states.wavelength_A.flags.writeable

    optical_k_film = states.k_film_phase_sample_Ainv.copy()
    optical_k_film[0] = 0.0
    optical_kz = states.kz_film_Ainv.copy()
    optical_kz[0] = 0.0
    optical_amplitude = states.entrance_amplitude.copy()
    optical_amplitude[0] = 0.0
    optical_failure = replace(
        states,
        k_film_phase_sample_Ainv=optical_k_film,
        kz_film_Ainv=optical_kz,
        entrance_amplitude=optical_amplitude,
        status=(ValidityCode.NUMERIC_FAILURE, ValidityCode.VALID),
        valid=np.array([False, True]),
    )
    np.testing.assert_array_equal(optical_failure.k_air_sample_Ainv[0], states.k_air_sample_Ainv[0])
    assert optical_failure.footprint_acceptance[0] == states.footprint_acceptance[0]

    geometry_direction = states.direction_sample.copy()
    geometry_direction[0] = 0.0
    geometry_k_air = states.k_air_sample_Ainv.copy()
    geometry_k_air[0] = 0.0
    geometry_k_film = states.k_film_phase_sample_Ainv.copy()
    geometry_k_film[0] = 0.0
    geometry_kz = states.kz_film_Ainv.copy()
    geometry_kz[0] = 0.0
    geometry_amplitude = states.entrance_amplitude.copy()
    geometry_amplitude[0] = 0.0
    geometry_footprint = states.footprint_acceptance.copy()
    geometry_footprint[0] = 0.0
    geometry_failure = replace(
        states,
        direction_sample=geometry_direction,
        k_air_sample_Ainv=geometry_k_air,
        k_film_phase_sample_Ainv=geometry_k_film,
        kz_film_Ainv=geometry_kz,
        entrance_amplitude=geometry_amplitude,
        footprint_acceptance=geometry_footprint,
        status=(ValidityCode.BACKWARD, ValidityCode.VALID),
        valid=np.array([False, True]),
    )
    assert geometry_failure.wavelength_A[0] == states.wavelength_A[0]
    assert geometry_failure.polarization_state_id[0] == states.polarization_state_id[0]
    assert geometry_failure.source_revision == states.source_revision

    direction_mutation = states.direction_sample.copy()
    direction_mutation[0, 0] = 0.5
    k_air_mutation = states.k_air_sample_Ainv.copy()
    k_air_mutation[0, 0] += 0.1
    tangential_mutation = states.k_film_phase_sample_Ainv.copy()
    tangential_mutation[0, 0] += 0.1
    normal_mutation = states.kz_film_Ainv.copy()
    normal_mutation[0] += 0.1
    for changes, message in (
        ({"valid": np.array([False, True])}, "valid must agree"),
        ({"wavelength_A": np.array([2.0, 1.0])}, "k_air_sample_Ainv"),
        ({"direction_sample": direction_mutation}, "unit direction_sample"),
        ({"k_air_sample_Ainv": k_air_mutation}, "k_air_sample_Ainv"),
        ({"k_film_phase_sample_Ainv": tangential_mutation}, "tangential"),
        ({"kz_film_Ainv": normal_mutation}, "film phase normal"),
        ({"footprint_acceptance": np.array([1.1, 1.0])}, "footprint_acceptance"),
        ({"incident_sample_id": np.array([10, 10])}, "unique source sample"),
        ({"source_parameter_revision": "0" * 64}, "canonical provenance"),
        ({"direction_sample": states.direction_sample}, "geometry-failure payload"),
        ({"k_film_phase_sample_Ainv": states.k_film_phase_sample_Ainv}, "optical-failure"),
    ):
        base = (
            geometry_failure
            if "geometry-failure" in message
            else optical_failure
            if "optical-failure" in message
            else states
        )
        with pytest.raises(ValueError, match=message):
            replace(base, **changes)


def test_layer_phase_and_intensity_conversion_contracts_are_explicit() -> None:
    amplitudes = contracts.LayerAmplitudeResult(
        event_id=np.array([100]),
        rod_id=np.array([30]),
        phase_id=("phase-a",),
        f_plus_e=np.array([1.0 + 2.0j]),
        f_minus_e=None,
        normalization=contracts.LayerAmplitudeNormalization.ONE_REGISTRY_FREE_LAYER,
        phase_sign=contracts.LayerPhaseSign.POSITIVE_Q_DOT_R,
        gauge_id="pbi2.pb_centered.v1",
        layer_normal_crystal=np.array([0.0, 0.0, 1.0]),
        layer_repeat_A=3.4,
    )
    for changes, message in (
        ({"layer_normal_crystal": np.array([0.0, 0.0, 2.0])}, "unit vector"),
        ({"layer_repeat_A": 0.0}, "positive"),
        ({"gauge_id": "pbi2.pb_centered"}, "versioned"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(amplitudes, **changes)

    contracts.LayerNormalQBatch(
        event_id=amplitudes.event_id,
        rod_id=amplitudes.rod_id,
        phase_id=amplitudes.phase_id,
        layer_normal_q_Ainv=np.array([0.63]),
        gauge_id=amplitudes.gauge_id,
    )
    converted = electron_squared_to_scattering_strength_A2(np.array([1.0]))
    np.testing.assert_allclose(converted, [7.940787682024163e-10], rtol=0.0, atol=0.0)


def test_first_stage_comparator_and_single_external_diagnostic(tmp_path: Path) -> None:
    def trace(stage: str, value: np.ndarray) -> TraceRecord:
        return TraceRecord(
            "core",
            stage,
            value,
            "px",
            "detector",
            Measure.NONE,
            QuantityKind.POINT,
            "bootstrap-v1",
            "analytic fixture",
        )

    reference = (trace("osc.beam_center_native", np.array([2.0, 3.0])),)
    candidate = (trace("osc.beam_center_native", np.array([2.5, 3.5])),)
    comparison = compare_traces(reference, candidate)
    assert comparison.first_failing_stage == "osc.beam_center_native"
    assert comparison.maximum_error == pytest.approx(0.5)

    repository_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside the repository"):
        write_diagnostic(
            repository_root / "forbidden.ra_diag.npz",
            arrays={"value": np.array([1.0])},
            manifest={"case_id": "core"},
            repository_root=repository_root,
        )
    output = tmp_path / "core.ra_diag.npz"
    write_diagnostic(
        output,
        arrays={"value": np.array([1.0])},
        manifest={"case_id": "core"},
        repository_root=repository_root,
    )
    assert list(tmp_path.iterdir()) == [output]
    with np.load(output, allow_pickle=False) as data:
        assert json.loads(bytes(data["manifest_json"]).decode()) == {"case_id": "core"}


def test_stage_tolerance_artifact_is_versioned_hashed_and_bindable() -> None:
    stages = stage_tolerances.load_stage_tolerances()
    bound = stages["geometry.detector_pixel_solid_angle"].bind(1e-6)
    assert isinstance(bound, Tolerance) and bound.scale == 1e-6
    with pytest.raises(ValueError, match="nonnegative"):
        stages["geometry.detector_pixel_solid_angle"].bind(-1.0)


def test_core_proof_cli_emits_one_passing_json_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for module_name, function_name in proof_cli._COMMANDS.values():
        assert callable(getattr(importlib.import_module(module_name), function_name))

    completed = subprocess.run(
        [sys.executable, "-m", "rasim_next.proof", "core", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 1
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert all(check["status"] == "PASS" for check in result["checks"])

    expected_unknown = (
        '{"command":"not-a-proof","error":"unknown proof command or option",'
        '"status":"FAIL","unknown_options":[]}\n'
    )
    for _ in range(2):
        assert proof_cli.main(["not-a-proof", "--json"]) == 2
        captured = capsys.readouterr()
        assert captured.out == expected_unknown
        assert captured.err == ""
