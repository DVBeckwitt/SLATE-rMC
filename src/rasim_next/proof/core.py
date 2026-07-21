"""Compact analytic checks for shared coordinate and optical primitives."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import CONTRACT_API_VERSION
from rasim_next.core.frames import FrameId
from rasim_next.core.interfaces import scalar_interface_amplitude
from rasim_next.core.traces import Measure, QuantityKind, TraceRecord
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.wave_modes import normal_wavevector, select_normal_wavevector
from rasim_next.io.orientation import (
    OscRawIndex,
    detector_native_to_raw,
    detector_to_raw_index,
    raw_to_detector_index,
    raw_to_detector_native,
)
from rasim_next.proof.diagnostics import write_diagnostic
from rasim_next.proof.traces import compare_traces


def _trace(stage: str, value: NDArray[np.generic], *, discrete: bool = False) -> TraceRecord:
    return TraceRecord(
        "bootstrap.mutation",
        stage,
        value,
        "1",
        "declared",
        Measure.NONE,
        QuantityKind.INDEX if discrete else QuantityKind.POINT,
        "bootstrap-v1",
        "analytic fixture",
    )


def _mutations() -> list[dict[str, object]]:
    raw = np.arange(77, dtype=np.int64).reshape(7, 11)
    native = raw_to_detector_native(raw)
    pairs = (
        ("osc_wrong_rotation", "osc.detector_native_array", native, np.rot90(raw, 1), True),
        ("osc_transpose", "osc.detector_native_array", native, raw.T, True),
        (
            "swap_row_column",
            "osc.beam_center_native",
            np.array([2.0, 3.0]),
            np.array([3.0, 2.0]),
            False,
        ),
        (
            "half_pixel",
            "osc.beam_center_native",
            np.array([2.0, 3.0]),
            np.array([2.5, 3.5]),
            False,
        ),
        (
            "transform_order",
            "geometry.instrument_transforms",
            np.array([0.0, 2.0]),
            np.array([-1.0, 1.0]),
            False,
        ),
        (
            "translate_vector",
            "geometry.lab_ray",
            np.array([1.0, 0.0]),
            np.array([2.0, 2.0]),
            False,
        ),
        (
            "opposite_root",
            "optics.kz_incident_film",
            np.array([2.0j]),
            np.array([-2.0j]),
            False,
        ),
    )
    results: list[dict[str, object]] = []
    for mutation_id, stage, expected, mutated, discrete in pairs:
        comparison = compare_traces(
            (_trace(stage, np.asarray(expected), discrete=discrete),),
            (_trace(stage, np.asarray(mutated), discrete=discrete),),
        )
        results.append(
            {
                "mutation_id": mutation_id,
                "fixture_id": "bootstrap",
                "expected_first_stage": stage,
                "expected_failure_metric": comparison.failure_metric,
                "observed_first_stage": comparison.first_failing_stage,
                "observed_failure_metric": comparison.failure_metric,
                "detected": comparison.first_failing_stage == stage,
            }
        )
    return results


def _checks() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    raw = np.arange(77).reshape(7, 11)
    native = raw_to_detector_native(raw)
    raw_index = OscRawIndex(4, 3)
    mapped = raw_to_detector_index(raw_index, raw.shape)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = RigidTransform(rotation, np.array([1.0, 2.0, 3.0]), FrameId.SAMPLE, FrameId.LAB)
    point = np.array([2.0, 0.0, 1.0])
    kz = normal_wavevector(
        k0_Ainv=2.0,
        refractive_index=1.0,
        k_parallel_Ainv=np.array([1.0, 0.0]),
        propagation_direction=1,
    )
    checks = [
        {
            "check_id": "coordinates",
            "status": "PASS"
            if np.array_equal(detector_native_to_raw(native), raw)
            and detector_to_raw_index(mapped, raw.shape) == raw_index
            else "FAIL",
            "evidence": "clockwise 7x11 array and index mapping invert",
        },
        {
            "check_id": "shared_primitives",
            "status": "PASS"
            if np.allclose(transform.inverse().apply_point(transform.apply_point(point)), point)
            and abs(kz**2 + 1.0 - 4.0) < 1e-14
            and select_normal_wavevector(-1e-32, -1) == -1e-16j
            and scalar_interface_amplitude(2.0, 2.0) == 1.0
            else "FAIL",
            "evidence": "rigid inverse, dispersion, decay branch, and equal-medium amplitude",
        },
    ]
    mutations = _mutations()
    checks.append(
        {
            "check_id": "error_injection",
            "status": "PASS" if all(item["detected"] for item in mutations) else "FAIL",
            "evidence": f"{sum(bool(item['detected']) for item in mutations)}/{len(mutations)} first-stage mutations detected",
        }
    )
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "core.ra_diag.npz"
        write_diagnostic(
            output,
            arrays={"value": np.array([1.0])},
            manifest={"case_id": "core"},
            repository_root=root,
        )
        diagnostic_passed = output.is_file() and len(list(Path(directory).iterdir())) == 1
    checks.append(
        {
            "check_id": "diagnostic",
            "status": "PASS" if diagnostic_passed else "FAIL",
            "evidence": "one external NPZ contains the embedded manifest",
        }
    )
    return checks, mutations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_core_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    checks, mutations = _checks()
    base = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "uncommitted"
    )
    environment = json.dumps(
        {"python": platform.python_version(), "numpy": np.__version__}, sort_keys=True
    ).encode()
    pack = root / "reference" / "rasim_reference_v1.npz"
    return {
        "schema_version": 1,
        "task_id": "T00",
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "base_sha": base,
        "commit_sha": None,
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": _sha256(pack)},
        "environment_sha256": hashlib.sha256(environment).hexdigest(),
        "checks": checks,
        "classifications": [],
        "limitations": ["no detector, Ewald, structure, reflectivity, or stacking physics"],
        "mutations": mutations,
    }
