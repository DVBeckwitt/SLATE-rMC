from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_fit_cli() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "fit_osc_geometry.py"
    specification = importlib.util.spec_from_file_location("fit_osc_geometry_cli_test", script)
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


def test_bi2se3_qualification_requires_zero_gauge_and_legacy_delta_bounds(
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
            initial=nonzero_gauge,
        )
    with pytest.raises(ValueError, match=r"exactly \+/-0\.5 degrees"):
        module.fit_osc_geometry_series(
            "series.yaml",
            fitted_parameter_names=fitted,
            fit_incidence_angle_delta=True,
            incidence_angle_delta_half_span_deg=0.4,
        )
