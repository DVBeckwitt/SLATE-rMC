from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rasim_next.core.contracts import IncidentSampleBatch
from rasim_next.core.frames import FrameId
from rasim_next.core.interfaces import scalar_interface_amplitude
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.wave_modes import normal_wavevector, select_normal_wavevector
from rasim_next.io.orientation import (
    DetectorIndex,
    OscRawIndex,
    detector_native_to_raw,
    detector_to_raw_index,
    raw_to_detector_index,
    raw_to_detector_native,
)
from rasim_next.proof.core import run_synthetic_plumbing
from rasim_next.proof.diagnostics import write_diagnostic
from rasim_next.proof.traces import Measure, QuantityKind, TraceRecord, compare_traces


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
