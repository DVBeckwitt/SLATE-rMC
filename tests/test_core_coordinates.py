from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rasim_next.core import contracts
from rasim_next.core.contracts import IncidentSampleBatch
from rasim_next.core.frames import FrameId
from rasim_next.core.interfaces import scalar_interface_amplitude
from rasim_next.core.scattering import electron_squared_to_intensity_per_sr
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
from rasim_next.proof import tolerances as stage_tolerances
from rasim_next.proof.core import run_synthetic_plumbing
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
    transform = RigidTransform(
        rotation, np.array([1.0, 2.0, 3.0]), FrameId.CRYSTAL, FrameId.SAMPLE
    )
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


def test_minimal_contract_flow_preserves_event_identity_and_mass() -> None:
    samples = IncidentSampleBatch(
        np.array([10], dtype=np.int64),
        np.zeros((1, 3)),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]),
        np.array([1.0]),
        ("linear",),
        "explicit_joint",
    )
    assert samples.incident_sample_id.size == 1
    assert not samples.origin_lab_m.flags.writeable

    result = run_synthetic_plumbing()
    np.testing.assert_array_equal(result.event_id, [100])
    np.testing.assert_allclose(result.event_mass, [0.06], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result.detector_image, [[0.015, 0.045]], rtol=0.0, atol=1e-15)
    assert result.detector_image.sum() == pytest.approx(result.event_mass.sum())
    assert len(result.factor_names) == 8


def test_scattering_event_contract_is_explicit() -> None:
    q_internal = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
    events = contracts.ScatteringEventBatch(
        event_id=np.array([100, 200]),
        incident_state_id=np.array([10, 20]),
        orientation_id=np.array([40, 40]),
        rod_id=np.array([30, 40]),
        wavelength_A=np.ones(2),
        q_internal_sample_Ainv=q_internal,
        q_sample_normal_Ainv=q_internal[:, 2],
        l_coordinate=np.zeros(2),
        kf_film_phase_sample_Ainv=q_internal,
        reciprocal_weight=np.array([0.5, 0.0]),
        ewald_residual_Ainv=np.array([0.0, 1e-6]),
        status=(ValidityCode.VALID, ValidityCode.RESIDUAL_EXCEEDED),
        valid=np.array([True, False]),
    )
    assert events.status == (ValidityCode.VALID, ValidityCode.RESIDUAL_EXCEEDED)
    with pytest.raises(ValueError, match="status"):
        replace(events, status=(ValidityCode.VALID,))
    with pytest.raises(ValueError, match="status"):
        replace(events, valid=np.array([True, True]))
    with pytest.raises(ValueError, match="sample-normal"):
        replace(events, q_sample_normal_Ainv=np.array([0.1, 0.3]))


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
    converted = electron_squared_to_intensity_per_sr(np.array([1.0]))
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
    bound = stages["measurement.pixel_solid_angle"].bind(1e-6)
    assert isinstance(bound, Tolerance) and bound.scale == 1e-6
    with pytest.raises(ValueError, match="nonnegative"):
        stages["measurement.pixel_solid_angle"].bind(-1.0)


def test_core_proof_cli_emits_one_passing_json_object() -> None:
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
