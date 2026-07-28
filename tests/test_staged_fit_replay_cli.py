from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[1]


def _load_replay_cli() -> ModuleType:
    script = ROOT / "scripts" / "replay_staged_fit.py"
    specification = importlib.util.spec_from_file_location("staged_fit_replay_cli_test", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _stage_envelope(
    module: ModuleType,
    case,
    stage: str,
    upstream: dict | None,
    *,
    summary: dict,
    state: dict,
) -> dict:
    geometry_source_revisions = {
        "Bi2Se3": "86a5a9f191688065c8265df9bf47175a1b070115a4ce20129a321cd79d2dc416",
        "Bi2Te3": "3a68902a791bd9e28d50205b4596738b19a6ea36ce169fe4ff39441798a24659",
    }
    source_state_count = 1 if stage == "geometry" else case.source_state_count
    source_revision = (
        geometry_source_revisions[case.material_id]
        if stage == "geometry"
        else case.expected_scientific_summary["source_revision"]
    )
    payload = {
        "schema_version": "rasim-staged-fit-replay-stage-v1",
        "stage": stage,
        "case_id": case.case_id,
        "material_id": case.material_id,
        "case_sha256": hashlib.sha256(case.path.read_bytes()).hexdigest(),
        "execution_backend": "cuda",
        "runtime": case.runtime_identity,
        "source_state_count": source_state_count,
        "source_seed": case.source_seed,
        "source_revision": source_revision,
        "upstream_scientific_revision": (
            None if upstream is None else upstream["scientific_revision"]
        ),
        "scientific_summary": {
            "case_id": case.case_id,
            "material_id": case.material_id,
            "source_state_count": case.source_state_count,
            "source_revision": case.expected_scientific_summary["source_revision"],
            stage: summary,
        },
        "state": state,
    }
    payload["scientific_revision"] = module.scientific_revision(stage, payload)
    return payload


def test_tracked_replay_cases_are_relative_and_hash_complete(tmp_path: Path) -> None:
    module = _load_replay_cli()
    cases = tuple(
        module.load_replay_case(path)
        for path in (
            ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml",
            ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml",
        )
    )

    assert tuple(case.material_id for case in cases) == ("Bi2Se3", "Bi2Te3")
    assert all(case.source_state_count == 250 for case in cases)
    assert all(case.incidence_angles_deg == (5.0, 10.0, 15.0) for case in cases)
    assert all(path.is_relative_to(ROOT) for case in cases for path in case.input_paths.values())
    bi2te3 = cases[1]
    locked_selection = bi2te3.expected_scientific_summary["geometry"]["selection_revision"]
    historical_selection = bi2te3.stage_config["geometry"]["historical_selection_revision"]
    catalog_manifest = bi2te3.stage_config["geometry"]["catalog_manifest_revision"]
    assert locked_selection == (
        "sha256-97ba51fce133a27404d4b571e40fc21413fb13e8ccc70d725486b5f12fefff51"
    )
    assert historical_selection == (
        "sha256-79f5028d1ee7d2b3bb67bcdd15d92822d2d4b6bd43ee7fe1cb0377e43941146a"
    )
    assert catalog_manifest == (
        "sha256-4f3755acdc0c3416e05cb58778d3f07ddf12d56cb49ef3b7abbdeb057d706abc"
    )

    original = (ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml").read_text(
        encoding="utf-8"
    )
    absolute = original.replace(
        'path = "../../../configs/bi2te3_simulation.yaml"',
        f'path = "{(ROOT / "configs" / "bi2te3_simulation.yaml").as_posix()}"',
        1,
    )
    bad_case = tmp_path / "absolute.toml"
    bad_case.write_text(absolute, encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        module.load_replay_case(bad_case, repository_root=ROOT)

    unknown = tmp_path / "unknown.toml"
    unknown.write_text("unexpected = true\n" + original, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        module.load_replay_case(unknown, repository_root=ROOT)

    nested_unknown = tmp_path / "nested_unknown.toml"
    nested_unknown.write_text(
        original.replace(
            "[expected.geometry]\n",
            "[expected.geometry]\nunverified_metric = 1\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"expected\.geometry contains unknown key"):
        module.load_replay_case(nested_unknown, repository_root=ROOT)

    nonfinite = tmp_path / "nonfinite.toml"
    nonfinite.write_text(
        original.replace(
            "site_rms_px = 7.175090155549409",
            "site_rms_px = nan",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="site_rms_px must be finite"):
        module.load_replay_case(nonfinite, repository_root=ROOT)

    invalid_render = tmp_path / "invalid_render.toml"
    invalid_render.write_text(
        original.replace("cuda_coordinate_chunk = 200000", "cuda_coordinate_chunk = 0", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cuda_coordinate_chunk must be a positive integer"):
        module.load_replay_case(invalid_render, repository_root=ROOT)


def test_replay_rejects_nested_input_path_decoys(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_cli()
    safe_load = module.yaml.safe_load

    def decoy_geometry_series(payload: str):
        document = safe_load(payload)
        if isinstance(document, dict) and document.get("schema_version") == (
            "rasim-osc-geometry-fit-v1"
        ):
            document = copy.deepcopy(document)
            document["simulation_config"] = "bi2se3_simulation.yaml"
        return document

    monkeypatch.setattr(module.yaml, "safe_load", decoy_geometry_series)
    with pytest.raises(ValueError, match="does not resolve to its declared replay role"):
        module.load_replay_case(
            ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
        )


def test_replay_requires_the_locked_numerical_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_replay_cli()
    lock = ROOT / "uv.lock"
    case_path = ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"

    identity = module.load_replay_case(case_path).runtime_identity

    assert identity["environment_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    installed_version = module.importlib_metadata.version
    assert identity["packages"]["numpy"] == installed_version("numpy")
    expected_packages = {
        "gemmi",
        "llvmlite",
        "numba",
        "numpy",
        "packaging",
        "platformdirs",
        "pyyaml",
        "scipy",
        "sqlalchemy",
        "typing-extensions",
        "xraydb",
    }
    lock_document = tomllib.loads(lock.read_text(encoding="utf-8"))
    sqlalchemy = next(
        record for record in lock_document["package"] if record["name"] == "sqlalchemy"
    )
    greenlet = next(
        dependency for dependency in sqlalchemy["dependencies"] if dependency["name"] == "greenlet"
    )
    if Marker(greenlet["marker"]).evaluate():
        expected_packages.add("greenlet")
    assert set(identity["packages"]) == expected_packages

    def mismatched_version(name: str) -> str:
        return "0.0.0" if name == "numpy" else installed_version(name)

    monkeypatch.setattr(module.importlib_metadata, "version", mismatched_version)
    with pytest.raises(RuntimeError, match=r"numpy.*uv run --frozen"):
        module.load_replay_case(case_path)


def test_runtime_identity_rechecks_the_case_bound_lock_bytes(tmp_path: Path) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    changed_lock = tmp_path / "uv.lock"
    changed_lock.write_bytes((ROOT / "uv.lock").read_bytes() + b"\n")
    environment_record = next(
        record for record in case.file_records if record["role"] == "environment_lock"
    )

    with pytest.raises(ValueError, match="environment lock content hash changed"):
        module._validated_runtime_identity(
            changed_lock,
            expected_sha256=environment_record["sha256"],
        )


def test_replay_reloads_and_rejects_mutated_case_mappings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    case.input_paths["simulation_config"] = case.input_paths["cif"]
    calls: list[str] = []

    def forbidden(**_kwargs):
        calls.append("runner")
        raise AssertionError("a stage runner must not execute")

    monkeypatch.setattr(module, "_run_geometry_stage", forbidden)
    output = tmp_path / "mutated_case_output"
    with pytest.raises(ValueError, match="loaded input mappings changed"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
        )

    assert calls == []
    assert not output.exists()


def test_scientific_revision_excludes_artifact_location_and_container_hash() -> None:
    module = _load_replay_cli()
    first = {
        "runtime": {"python_version": "3.13.13"},
        "state": {
            "artifact": "C:/first/result.json",
            "artifact_sha256": "0" * 64,
            "parameters": [1.0, 2.0],
        },
    }
    relocated = copy.deepcopy(first)
    relocated["state"]["artifact"] = "/other/machine/result.json"
    relocated["state"]["artifact_sha256"] = "f" * 64
    relocated["runtime"]["python_version"] = "0.0.0"

    assert module.scientific_revision("mosaic", first) == module.scientific_revision(
        "mosaic", relocated
    )


def test_replay_runs_stages_in_order_and_chains_scientific_revisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    calls: list[tuple[str, str | None, str]] = []

    def stage(name: str):
        def run(*, case, upstream, backend, output_directory):
            upstream_revision = None if upstream is None else upstream["scientific_revision"]
            calls.append((name, upstream_revision, backend))
            state = {}
            if name == "render":
                pixels = bytes((1, 2, 3, 4))
                artifact = tmp_path / "ordered_stage_render.png"
                Image.frombytes("L", (2, 2), pixels).save(artifact)
                state = {
                    "artifact": [str(artifact)],
                    "artifact_identity": [
                        {
                            "decoded_mode": "L",
                            "decoded_size": [2, 2],
                            "decoded_pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                        }
                    ],
                }
            return _stage_envelope(
                module,
                case,
                name,
                upstream,
                summary={},
                state=state,
            )

        return run

    monkeypatch.setattr(module, "_run_geometry_stage", stage("geometry"))
    monkeypatch.setattr(module, "_run_mosaic_stage", stage("mosaic"))
    monkeypatch.setattr(module, "_run_ordered_intensity_stage", stage("ordered_intensity"))
    monkeypatch.setattr(module, "_run_render_stage", stage("render"))

    result = module.run_replay(
        case,
        output_directory=tmp_path / "result",
        backend="cuda",
        through="render",
        verify=False,
    )

    assert tuple(name for name, *_ in calls) == (
        "geometry",
        "mosaic",
        "ordered_intensity",
        "render",
    )
    assert all(backend == "cuda" for *_, backend in calls)
    assert calls[0][1] is None
    assert calls[1][1] == result["stages"]["geometry"]["scientific_revision"]
    assert calls[2][1] == result["stages"]["mosaic"]["scientific_revision"]
    assert calls[3][1] == result["stages"]["ordered_intensity"]["scientific_revision"]
    geometry = result["stages"]["geometry"]
    assert geometry["source_state_count"] == 1
    assert geometry["source_seed"] == 1729
    assert geometry["source_revision"] == (
        "3a68902a791bd9e28d50205b4596738b19a6ea36ce169fe4ff39441798a24659"
    )
    assert geometry["source_revision"] != result["stages"]["mosaic"]["source_revision"]
    assert all(
        stage["source_state_count"] == 250
        for name, stage in result["stages"].items()
        if name != "geometry"
    )
    assert all(
        stage["scientific_summary"]["source_state_count"] == 250
        for stage in result["stages"].values()
    )
    assert result["verification_runtime"]["packages"]["pillow"] == (
        module.importlib_metadata.version("pillow")
    )
    assert "pillow" not in result["stages"]["geometry"]["runtime"]["packages"]
    assert result["stages"]["render"]["runtime"]["packages"]["pillow"] == (
        module.importlib_metadata.version("pillow")
    )


def test_fresh_stage_recomputes_revision_before_persisting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )

    def changed_stage(*, case, upstream, backend, output_directory):
        del backend, output_directory
        result = _stage_envelope(
            module,
            case,
            "geometry",
            upstream,
            summary={},
            state={},
        )
        result["state"]["changed_after_revision"] = True
        return result

    monkeypatch.setattr(module, "_run_geometry_stage", changed_stage)
    output = tmp_path / "changed_fresh_stage"
    with pytest.raises(ValueError, match="changed its scientific revision"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
            verify=False,
        )

    assert not (output / "geometry.json").exists()


def test_stage_summary_is_verified_before_persisting_or_running_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    geometry_summary = copy.deepcopy(case.expected_scientific_summary["geometry"])
    geometry_summary["corrections"][2] += 10.0 * case.tolerances["geometry_correction_absolute"]
    calls: list[str] = []

    def geometry(*, case, upstream, backend, output_directory):
        del backend, output_directory
        calls.append("geometry")
        return _stage_envelope(
            module,
            case,
            "geometry",
            upstream,
            summary=geometry_summary,
            state={"corrections": list(geometry_summary["corrections"])},
        )

    def forbidden(**_kwargs):
        calls.append("mosaic")
        raise AssertionError("mosaic must not execute after rejected geometry")

    monkeypatch.setattr(module, "_run_geometry_stage", geometry)
    monkeypatch.setattr(module, "_run_mosaic_stage", forbidden)
    output = tmp_path / "rejected_geometry"
    with pytest.raises(module.ReplayMismatchError, match=r"geometry\.corrections"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
        )

    assert calls == ["geometry"]
    assert not (output / "geometry.json").exists()


def test_bi2te3_cpu_replay_rejects_before_output_or_stage_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    calls: list[str] = []

    def forbidden(**_kwargs):
        calls.append("runner")
        raise AssertionError("a stage runner must not execute")

    for name in (
        "_run_geometry_stage",
        "_run_mosaic_stage",
        "_run_ordered_intensity_stage",
        "_run_render_stage",
    ):
        monkeypatch.setattr(module, name, forbidden)
    output = tmp_path / "cpu_output"
    with pytest.raises(ValueError, match="CUDA-qualified only"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cpu",
            through="mosaic",
        )

    assert calls == []
    assert not output.exists()


def test_render_preflight_requires_importable_pillow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    imported = module.importlib.import_module

    def fail_pillow(name: str):
        if name == "PIL.Image":
            raise ImportError("synthetic missing image module")
        return imported(name)

    monkeypatch.setattr(module.importlib, "import_module", fail_pillow)
    output = tmp_path / "render_output"
    with pytest.raises(RuntimeError, match=r"PIL\.Image.*--extra visualization"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="render",
            verify=False,
        )

    assert not output.exists()


def test_scientific_verifier_uses_tolerances_but_rejects_identity_changes() -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    actual = copy.deepcopy(case.expected_scientific_summary)

    actual["geometry"]["corrections"][2] += 0.25 * case.tolerances["geometry_correction_absolute"]
    actual["mosaic"]["parameters"][0] += 0.25 * case.tolerances["mosaic_parameter_absolute"]
    module.verify_scientific_summary(case, actual)

    wrong_source = copy.deepcopy(actual)
    wrong_source["source_revision"] = "sha256-" + "0" * 64
    with pytest.raises(module.ReplayMismatchError, match="source_revision"):
        module.verify_scientific_summary(case, wrong_source)

    wrong_profiles = copy.deepcopy(actual)
    wrong_profiles["mosaic"]["profile_identities"].pop()
    with pytest.raises(module.ReplayMismatchError, match="profile_identities"):
        module.verify_scientific_summary(case, wrong_profiles)

    wrong_m0 = copy.deepcopy(actual)
    wrong_m0["mosaic"]["m0_profile_identities"] = []
    with pytest.raises(module.ReplayMismatchError, match="m0_profile_identities"):
        module.verify_scientific_summary(case, wrong_m0)

    wrong_parameter = copy.deepcopy(actual)
    wrong_parameter["ordered_intensity"]["parameters"][1] += (
        10.0 * case.tolerances["ordered_parameter_absolute"]
    )
    with pytest.raises(module.ReplayMismatchError, match=r"ordered_intensity\.parameters"):
        module.verify_scientific_summary(case, wrong_parameter)


def test_partial_verification_and_resume_reject_tampered_scientific_state(
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    geometry_summary = copy.deepcopy(case.expected_scientific_summary["geometry"])
    partial = {
        "case_id": case.case_id,
        "material_id": case.material_id,
        "source_state_count": case.source_state_count,
        "source_revision": case.expected_scientific_summary["source_revision"],
        "geometry": geometry_summary,
    }
    module.verify_scientific_summary(case, partial, through="geometry")

    output = tmp_path / "resume"
    output.mkdir()
    stage = _stage_envelope(
        module,
        case,
        "geometry",
        None,
        summary=geometry_summary,
        state={"corrections": list(geometry_summary["corrections"])},
    )
    (output / "geometry.json").write_text(
        json.dumps(stage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="geometry",
        resume=True,
    )
    assert result["verified"]

    changed_runtime = copy.deepcopy(stage)
    changed_runtime["runtime"]["python_version"] = "0.0.0"
    (output / "geometry.json").write_text(
        json.dumps(changed_runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed runtime"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
            resume=True,
        )

    (output / "geometry.json").write_text(
        json.dumps(stage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tampered = json.loads((output / "geometry.json").read_text(encoding="utf-8"))
    tampered["state"]["corrections"][2] += 1.0e-3
    (output / "geometry.json").write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scientific revision"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
            resume=True,
        )


def test_resume_requires_the_external_json_artifact_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    output = tmp_path / "resume_json"
    artifact = tmp_path / "mosaic.json"
    artifact.write_text("{}\n", encoding="utf-8")

    def runner(stage: str):
        def run(*, case, upstream, backend, output_directory):
            del backend, output_directory
            state = {}
            if stage == "mosaic":
                state = {
                    "artifact": str(artifact),
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            return _stage_envelope(
                module,
                case,
                stage,
                upstream,
                summary={},
                state=state,
            )

        return run

    monkeypatch.setattr(module, "_run_geometry_stage", runner("geometry"))
    monkeypatch.setattr(module, "_run_mosaic_stage", runner("mosaic"))
    module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="mosaic",
        verify=False,
    )

    mosaic_path = output / "mosaic.json"
    original_mosaic = json.loads(mosaic_path.read_text(encoding="utf-8"))
    artifact.write_text('{"substituted": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed its external result"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
            verify=False,
            resume=True,
        )

    artifact.write_text("{}\n", encoding="utf-8")
    mosaic = copy.deepcopy(original_mosaic)
    mosaic["state"].pop("artifact")
    mosaic["state"].pop("artifact_sha256")
    mosaic_path.write_text(
        json.dumps(mosaic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lacks its external JSON result"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
            verify=False,
            resume=True,
        )


def test_render_resume_rejects_changed_decoded_pixels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    output = tmp_path / "resume_render"
    image_path = output / "render.png"
    pixels = bytes((1, 2, 3, 4))

    def runner(stage: str):
        def run(*, case, upstream, backend, output_directory):
            state = {}
            if stage == "render":
                Image.frombytes("L", (2, 2), pixels).save(image_path)
                state = {
                    "artifact": [str(image_path)],
                    "artifact_identity": [
                        {
                            "decoded_mode": "L",
                            "decoded_size": [2, 2],
                            "decoded_pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                        }
                    ],
                }
            return _stage_envelope(
                module,
                case,
                stage,
                upstream,
                summary={},
                state=state,
            )

        return run

    monkeypatch.setattr(module, "_run_geometry_stage", runner("geometry"))
    monkeypatch.setattr(module, "_run_mosaic_stage", runner("mosaic"))
    monkeypatch.setattr(module, "_run_ordered_intensity_stage", runner("ordered_intensity"))
    monkeypatch.setattr(module, "_run_render_stage", runner("render"))
    module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="render",
        verify=False,
    )

    Image.new("L", (2, 2), color=9).save(image_path)
    with pytest.raises(ValueError, match="decoded pixels"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="render",
            verify=False,
            resume=True,
        )

    render = json.loads((output / "render.json").read_text(encoding="utf-8"))
    render["state"].pop("artifact")
    (output / "render.json").write_text(
        json.dumps(render, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identities are incomplete"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="render",
            verify=False,
            resume=True,
        )
