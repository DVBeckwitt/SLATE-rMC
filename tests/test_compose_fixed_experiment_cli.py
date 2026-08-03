from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import yaml

from rasim_next.fitting import (
    FIXED_EXPERIMENT_STATE_SCHEMA_VERSION,
    POSITION_FIT_RESULT_SCHEMA_VERSION,
    SHARED_GEOMETRY_PARAMETER_NAMES,
    FixedMosaicState,
    FixedPositionState,
    SharedGeometryCorrections,
    zero_sum_helmert_basis,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compose_fixed_experiment.py"
SPEC = importlib.util.spec_from_file_location("compose_fixed_experiment_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def _position() -> FixedPositionState:
    image_ids = ("five", "ten", "fifteen")
    trims = tuple(map(math.radians, (-0.05, 0.02, 0.03)))
    trim_by_id = dict(zip(image_ids, trims, strict=True))
    canonical = tuple(trim_by_id[image_id] for image_id in sorted(image_ids))
    contrasts = tuple(zero_sum_helmert_basis(3).T @ np.asarray(canonical))
    return FixedPositionState(
        artifact_revision=f"sha256-{'a' * 64}",
        corrections=SharedGeometryCorrections.zero(),
        incidence_angle_delta_rad=math.radians(0.4),
        commanded_incidence_angles_rad=tuple(map(math.radians, (5.0, 10.0, 15.0))),
        beam_center_column_row_px=(1453.12, 1596.422),
        incidence_angle_image_ids=image_ids,
        incidence_angle_trim_rad=trims,
        incidence_angle_trim_contrast_rad=contrasts,
        incidence_angle_trim_prior_sigma_rad=math.radians(0.25),
        incidence_angle_trim_contrast_half_span_rad=math.radians(0.5),
    )


def test_compose_fixed_experiment_writes_resumable_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = ROOT.parent / f"compose-fixed-experiment-{uuid4().hex}"
    workspace.mkdir()
    try:
        position_path = workspace / "position.json"
        osc_paths = tuple(
            workspace / f"{image_id}.osc.gz" for image_id in _position().incidence_angle_image_ids
        )
        for path in osc_paths:
            path.write_bytes(path.name.encode("ascii"))
        manifest_path = workspace / "geometry.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "rasim-osc-geometry-fit-v1",
                    "simulation_config": str(
                        (ROOT / "configs/bi2se3_r3_simulation.yaml").resolve()
                    ),
                    "incidence_axis_index": 0,
                    "images": [
                        {
                            "image_id": image_id,
                            "osc_path": str(osc_path),
                            "axis_rotation_angles_deg": [angle],
                        }
                        for image_id, osc_path, angle in zip(
                            _position().incidence_angle_image_ids,
                            osc_paths,
                            (5.0, 10.0, 15.0),
                            strict=True,
                        )
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        position_path.write_text(
            json.dumps(
                {
                    "schema": POSITION_FIT_RESULT_SCHEMA_VERSION,
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "indexed_manifest_hash": "frozen-selection",
                    "run_completed": True,
                    "root_audit": {"classification": "SAME"},
                    "outer_audit": {"classification": "SAME"},
                    "qualification": {"requested": False, "accepted": False},
                    "fit": {
                        "success": True,
                        "jacobian_parameter_names": ["shared", "delta", "trim_0", "trim_1"],
                        "jacobian_rank": 4,
                        "incidence_angle_delta_fitted": True,
                        "incidence_angle_delta_rad": _position().incidence_angle_delta_rad,
                        "corrections": {
                            name: float(value)
                            for name, value in zip(
                                SHARED_GEOMETRY_PARAMETER_NAMES,
                                _position().corrections.as_array(),
                                strict=True,
                            )
                        },
                        "incidence_angle_trim_by_image_id_rad": dict(
                            zip(
                                _position().incidence_angle_image_ids,
                                _position().incidence_angle_trim_rad,
                                strict=True,
                            )
                        ),
                    },
                    "fixed_position": _position().to_record(),
                }
            ),
            encoding="utf-8",
        )
        mosaic_path = workspace / "mosaic.json"
        mosaic_path.write_text(
            json.dumps(
                FixedMosaicState(
                    gaussian_sigma_deg=1.2,
                    lorentzian_hwhm_deg=0.3,
                    lorentzian_probability=0.25,
                    provenance="provided test prior",
                ).to_record()
            )
            + "\n",
            encoding="utf-8",
        )

        class DetectorCounts:
            shape = (3000, 3000)
            dtype = np.dtype("int32")

            @staticmethod
            def tobytes(*, order: str) -> bytes:
                assert order == "C"
                return b"detector-native-test-counts"

        monkeypatch.setattr(
            ADAPTER,
            "read_osc",
            lambda _path: SimpleNamespace(detector_native_counts=DetectorCounts()),
        )
        destination = workspace / "fixed_experiment.json"

        ADAPTER.compose_fixed_experiment(
            position_path=position_path,
            geometry_manifest_path=manifest_path,
            recipe_path=ROOT / "examples/bi2se3/experiment/figure7_matched_regions.toml",
            mosaic_state_path=mosaic_path,
            lattice_path=None,
            source_state_count=1,
            destination=destination,
        )

        document = json.loads(destination.read_text(encoding="utf-8"))
        assert document["schema_version"] == FIXED_EXPERIMENT_STATE_SCHEMA_VERSION
        assert document["position_status"] == "POSITION_MODEL_LIMITED"
        assert (
            document["fixed_mosaic"]
            == FixedMosaicState(
                gaussian_sigma_deg=1.2,
                lorentzian_hwhm_deg=0.3,
                lorentzian_probability=0.25,
                provenance="provided test prior",
            ).to_record()
        )
        assert document["model_pixelized"] is False
        assert document["smoothing_applied"] is False
        assert [item["position_image_id"] for item in document["datasets"]] == [
            "five",
            "ten",
            "fifteen",
        ]
        assert [item["dataset_id"] for item in document["datasets"]] == [
            "Bi2Se3-5deg",
            "Bi2Se3-10deg",
            "Bi2Se3-15deg",
        ]

        modular_position_path = workspace / "modular-position.json"
        modular_position_path.write_text(
            json.dumps(
                {
                    "schema_version": "rasim-layered-position-fit-v2",
                    "accepted": True,
                    "intensity_evaluated": False,
                    "model_pixelized": False,
                    "status": "POSITION_MODEL_LIMITED",
                    "scientific_revision": _position().artifact_revision,
                    "fixed_position": _position().to_record(),
                    "provenance": {
                        "manifest": {
                            "path": str(manifest_path),
                            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        modular_position, modular_status = ADAPTER._position_state(
            modular_position_path,
            geometry_manifest_path=manifest_path,
        )
        assert modular_position.to_record() == _position().to_record()
        assert modular_status == "POSITION_MODEL_LIMITED"

        position_document = json.loads(position_path.read_text(encoding="utf-8"))
        position_document["qualification"] = {"requested": True, "accepted": False}
        position_path.write_text(json.dumps(position_document), encoding="utf-8")
        with pytest.raises(ValueError, match="qualification was not accepted"):
            ADAPTER._position_state(
                position_path,
                geometry_manifest_path=manifest_path,
            )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.parametrize("value", (True, 0, -1))
def test_source_state_count_rejects_nonpositive_or_boolean(value: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ADAPTER._source_state_count(value)
