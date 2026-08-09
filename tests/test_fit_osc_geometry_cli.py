from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _load_fit_cli() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "fit_osc_geometry.py"
    specification = importlib.util.spec_from_file_location("fit_osc_geometry_cli_test", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_render_cli() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_osc_geometry_fit.py"
    specification = importlib.util.spec_from_file_location("render_osc_geometry_cli_test", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_fit_cli_forwards_active_complement_and_rejects_all_frozen(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_fit_cli()
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_fit(manifest_path: object, **options: object) -> dict[str, object]:
        calls.append((manifest_path, options))
        return {
            "qualification": {"requested": False, "accepted": False},
            "run_completed": True,
        }

    monkeypatch.setattr(module, "fit_osc_geometry_series", fake_fit)
    frozen = (
        "detector_row_tilt_rad",
        "sample_normal_x_tilt_rad",
        "goniometer_axis_yaw_rad",
        "sample_plane_normal_offset_m",
    )
    arguments = ["series.yaml", "--fit-incidence-angle-delta", "--json"]
    for name in frozen:
        arguments.extend(("--freeze-parameter", name))

    assert module.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["run_completed"] is True
    assert len(calls) == 1
    assert calls[0][0] == Path("series.yaml")
    assert calls[0][1]["fitted_parameter_names"] == tuple(
        name for name in module.SHARED_GEOMETRY_PARAMETER_NAMES if name not in frozen
    )
    assert calls[0][1]["fit_incidence_angle_delta"] is True
    assert calls[0][1]["incidence_angle_delta_half_span_deg"] == 0.5

    all_frozen = ["series.yaml", "--json"]
    for name in module.SHARED_GEOMETRY_PARAMETER_NAMES:
        all_frozen.extend(("--freeze-parameter", name))
    assert module.main(all_frozen) == 1
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["error"]["type"] == "ValueError"
    assert "at least one shared geometry parameter" in rejection["error"]["message"]
    assert len(calls) == 1


def test_fit_cli_persists_the_raw_position_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_fit_cli()
    payload = {
        "schema": "rasim-osc-geometry-fit-result-v6",
        "qualification": {"requested": False, "accepted": False},
        "run_completed": True,
    }
    monkeypatch.setattr(module, "fit_osc_geometry_series", lambda *_args, **_options: payload)
    destination = tmp_path / "position.json"

    assert (
        module.main(
            [
                "series.yaml",
                "--fit-incidence-angle-delta",
                "--json",
                "--destination",
                str(destination),
            ]
        )
        == 0
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == payload


def test_bi2se3_qualification_requires_zero_gauge_common_delta_and_trim_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_fit_cli()
    monkeypatch.setattr(
        module,
        "load_osc_geometry_series",
        lambda _path: SimpleNamespace(qualification_profile="bi2se3-osc-5-10-15.v1"),
    )
    fitted = tuple(
        name
        for name in module.SHARED_GEOMETRY_PARAMETER_NAMES
        if name != "sample_normal_x_tilt_rad"
    )
    nonzero_gauge = module.SharedGeometryCorrections.from_array(
        (0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    with pytest.raises(ValueError, match="fixed at zero"):
        module.fit_osc_geometry_series(
            "series.yaml",
            fitted_parameter_names=fitted,
            fit_incidence_angle_delta=True,
            fit_incidence_angle_trim=True,
            initial=nonzero_gauge,
        )
    with pytest.raises(ValueError, match="qualified bounds/prior"):
        module.fit_osc_geometry_series(
            "series.yaml",
            fitted_parameter_names=fitted,
            fit_incidence_angle_delta=True,
            fit_incidence_angle_trim=True,
            incidence_angle_delta_half_span_deg=0.4,
        )
    with pytest.raises(ValueError, match="qualified bounds/prior"):
        module.fit_osc_geometry_series(
            "series.yaml",
            fitted_parameter_names=fitted,
            fit_incidence_angle_delta=True,
            fit_incidence_angle_trim=True,
            incidence_angle_trim_prior_sigma_deg=0.2,
        )


def test_primary_prediction_payload_retains_signed_and_whitened_detector_residuals() -> None:
    module = _load_fit_cli()
    key = SimpleNamespace(
        family_m=3,
        integer_L=11,
        branch=2,
        root_sign=-1,
        representative_rod_hk=(-1, 0),
    )
    observations = SimpleNamespace(
        keys=(key,),
        coordinates_px=np.asarray(((100.0, 200.0),)),
        covariance_px2=np.asarray((((4.0, 0.0), (0.0, 9.0)),)),
        whitening_matrix_px_inv=np.asarray((((0.5, 0.0), (0.0, 1.0 / 3.0)),)),
    )
    prediction = SimpleNamespace(
        coordinates_px=np.asarray(((102.0, 197.0),)),
        detector_status=np.asarray(("VALID",)),
    )
    image = SimpleNamespace(
        image_id="sample_10deg",
        observations=observations,
        predict_integer_l_tags=lambda *_args, **_kwargs: prediction,
    )

    payload = module._prediction_payload((image,), module.SharedGeometryCorrections.zero(), 0.0)

    site = payload[0]["sites"][0]
    assert site["error_column_px"] == 2.0
    assert site["error_row_px"] == -3.0
    assert site["error_norm_px"] == pytest.approx(np.sqrt(13.0))
    assert site["whitened_error_norm"] == pytest.approx(np.sqrt(2.0))
    assert site["covariance_px2"] == [[4.0, 0.0], [0.0, 9.0]]
    assert site["key"] == {
        "family_m": 3,
        "integer_L": 11,
        "analytic_ewald_branch": 2,
        "root_sign": -1,
        "representative_rod_hk": (-1, 0),
    }


def test_geometry_renderer_binds_position_to_manifest_and_native_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_render_cli()
    manifest = tmp_path / "series.yaml"
    manifest.write_text("series\n", encoding="utf-8")
    image_record = SimpleNamespace(
        image_id="sample_5deg",
        osc_path=tmp_path / "sample.osc",
        axis_rotation_angles_deg=(5.0,),
    )
    monkeypatch.setattr(
        module,
        "load_osc_geometry_series",
        lambda _path: SimpleNamespace(incidence_axis_index=0, images=(image_record,)),
    )
    monkeypatch.setattr(
        module,
        "read_osc",
        lambda _path: SimpleNamespace(detector_native_counts=np.arange(80).reshape(8, 10)),
    )
    site = {
        "key": {"family_m": 1, "integer_L": 8},
        "observed_coordinate_px": [4.0, 3.0],
        "predicted_coordinate_px": [4.5, 2.75],
        "error_column_px": 0.5,
        "error_row_px": -0.25,
        "error_norm_px": float(np.hypot(0.5, 0.25)),
    }
    position = tmp_path / "position.json"
    position.write_text(
        json.dumps(
            {
                "schema": "rasim-osc-geometry-fit-result-v6",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "predictions": [{"image_id": "sample_5deg", "sites": [site]}],
                "cross_validation": None,
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "geometry.png"

    result = module.render_geometry_fit(manifest, position, destination)

    assert destination.is_file()
    assert result["figure_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert result["images"][0]["primary_rms_px"] == pytest.approx(np.hypot(0.5, 0.25))
