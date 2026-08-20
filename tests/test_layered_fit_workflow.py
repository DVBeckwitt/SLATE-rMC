from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from rasim_next.fitting import workflow as workflow_module
from rasim_next.fitting.workflow import (
    FIT_WORKFLOW_STAGE_NAMES,
    load_fit_workflow,
    planned_fit_workflow,
    run_fit_workflow,
)

ROOT = Path(__file__).resolve().parents[1]


def _quoted_toml(value: str | Path) -> str:
    return json.dumps(str(value))


def test_workflow_runs_geometry_mosaic_sf_once_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_module,
        "_repository_scientific_revision",
        lambda _root: "sha256-" + "1" * 64,
    )
    case_path = tmp_path / "case.toml"
    output_directory = tmp_path / "run"
    call_log = tmp_path / "calls.txt"
    stage_input = tmp_path / "stage-input.txt"
    stage_input.write_text("frozen input", encoding="utf-8")
    stage_tables: list[str] = []
    for stage in FIT_WORKFLOW_STAGE_NAMES:
        code = (
            "from pathlib import Path; "
            f"log=Path(r'{call_log}'); "
            f"log.write_text((log.read_text() if log.exists() else '')+'{stage}\\n'); "
            "out=Path(r'{stage_dir}/result.txt'); "
            "out.parent.mkdir(parents=True,exist_ok=True); "
            f"out.write_text('{stage}')"
        )
        stage_tables.append(
            "\n".join(
                (
                    f"[stages.{stage}]",
                    'completion_artifacts = ["result.txt"]',
                    "commands = [["
                    + ", ".join(
                        (
                            _quoted_toml(sys.executable),
                            '"-c"',
                            _quoted_toml(code),
                            _quoted_toml(stage_input),
                        )
                    )
                    + "]]",
                )
            )
        )
    case_path.write_text(
        "\n".join(
            (
                'schema_version = "rasim-layered-fit-workflow-v1"',
                'material_id = "test-material"',
                'model_family = "bi2x3_quintuple.v1"',
                'backend = "cpu"',
                *stage_tables,
            )
        ),
        encoding="utf-8",
    )

    workflow = load_fit_workflow(case_path, output_directory=output_directory)
    first = run_fit_workflow(workflow)
    second = run_fit_workflow(workflow)

    assert tuple(result.stage for result in first) == FIT_WORKFLOW_STAGE_NAMES
    assert all(not result.reused for result in first)
    assert all(result.reused for result in second)
    assert call_log.read_text(encoding="utf-8").splitlines() == list(FIT_WORKFLOW_STAGE_NAMES)
    for stage in FIT_WORKFLOW_STAGE_NAMES:
        assert (output_directory / stage / "result.txt").read_text(encoding="utf-8") == stage
        assert (output_directory / stage / "stage.json").is_file()
    geometry_manifest_path = output_directory / "geometry" / "stage.json"
    geometry_manifest = json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
    for field in ("material_id", "model_family"):
        geometry_manifest_path.write_text(
            json.dumps({**geometry_manifest, field: "tampered"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="does not match this workflow plan"):
            run_fit_workflow(workflow)
    geometry_manifest_path.write_text(json.dumps(geometry_manifest), encoding="utf-8")
    geometry_manifest_path.write_text(
        json.dumps(
            {
                **geometry_manifest,
                "stage_revision": "sha256-" + "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage revision changed"):
        run_fit_workflow(workflow)
    geometry_manifest_path.write_text(json.dumps(geometry_manifest), encoding="utf-8")
    stage_input.write_text("changed input", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match this workflow plan"):
        run_fit_workflow(workflow)
    stage_input.write_text("frozen input", encoding="utf-8")
    (output_directory / "geometry" / "result.txt").write_text("changed artifact")
    with pytest.raises(ValueError, match="artifact content changed"):
        run_fit_workflow(workflow)

    stage_directory = output_directory / "geometry"
    stage_document = json.loads((stage_directory / "stage.json").read_text(encoding="utf-8"))
    (stage_directory / "stage.json").unlink()
    (stage_directory / "stage.progress.json").write_text(
        json.dumps(
            {
                "schema_version": workflow_module.FIT_WORKFLOW_PROGRESS_SCHEMA,
                "stage": "geometry",
                "status": "started",
                "plan_revision": stage_document["plan_revision"],
                "repository_scientific_revision": stage_document["repository_scientific_revision"],
                "upstream_revision": None,
                "completed_command_count": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete outer workflow stage"):
        run_fit_workflow(workflow)


def test_workflow_scientific_revision_binds_every_live_repository_file(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    (repository / "docs").mkdir()
    (repository / "src" / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
    (repository / "src" / "run.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repository / "docs" / "note.md").write_text("note one\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(("git", "config", "user.name", "test"), cwd=repository, check=True)
    subprocess.run(("git", "config", "core.filemode", "false"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(
        ("git", "update-index", "--chmod=+x", "src/run.sh"),
        cwd=repository,
        check=True,
    )
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=repository, check=True)

    original = workflow_module._repository_scientific_revision(repository)
    (repository / "docs" / "note.md").write_text("note two\n", encoding="utf-8")
    assert workflow_module._repository_scientific_revision(repository) != original
    (repository / "docs" / "note.md").write_text("note one\n", encoding="utf-8")
    assert workflow_module._repository_scientific_revision(repository) == original
    (repository / "src" / "model.py").write_text("MODEL = 2\n", encoding="utf-8")
    dirty_revision = workflow_module._repository_scientific_revision(repository)
    assert dirty_revision != original
    subprocess.run(("git", "add", "src/model.py"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "update model"), cwd=repository, check=True)
    assert workflow_module._repository_scientific_revision(repository) == dirty_revision
    (repository / "src" / "run.sh").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    dirty_executable_revision = workflow_module._repository_scientific_revision(repository)
    subprocess.run(("git", "add", "src/run.sh"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "update executable"), cwd=repository, check=True)
    assert workflow_module._repository_scientific_revision(repository) == dirty_executable_revision
    subprocess.run(
        ("git", "update-index", "--assume-unchanged", "src/model.py"),
        cwd=repository,
        check=True,
    )
    try:
        with pytest.raises(ValueError, match="assume-unchanged or skip-worktree"):
            workflow_module._repository_scientific_revision(repository)
    finally:
        subprocess.run(
            ("git", "update-index", "--no-assume-unchanged", "src/model.py"),
            cwd=repository,
            check=True,
        )


def test_workflow_hashes_the_command_executable(tmp_path: Path) -> None:
    helper = tmp_path / "helper.bin"
    helper.write_bytes(b"first")
    workflow = workflow_module.FitWorkflow(
        case_path=tmp_path / "case.toml",
        repository_root=tmp_path,
        output_directory=tmp_path / "run",
        material_id="test-material",
        model_family="bi2x3_quintuple.v1",
        backend="cpu",
        fit_parameter_seed=None,
        stages={},
    )
    workflow.case_path.write_text("case", encoding="utf-8")

    first = workflow_module._command_input_hashes(workflow, ((str(helper),),))
    helper.write_bytes(b"second")
    second = workflow_module._command_input_hashes(workflow, ((str(helper),),))

    assert dict(first)[str(helper.resolve())] != dict(second)[str(helper.resolve())]

    repository_helper = tmp_path / "tools" / "helper.bin"
    repository_helper.parent.mkdir()
    repository_helper.write_bytes(b"repository helper")
    relative = workflow_module._command_input_hashes(
        workflow,
        ((str(Path("tools") / "helper.bin"),),),
    )
    assert dict(relative)[str(repository_helper.resolve())] == workflow_module._file_sha256(
        repository_helper
    )


@pytest.mark.parametrize(
    "artifact_declaration",
    (
        '["stage.progress.json"]',
        '["result.txt", "result.txt"]',
    ),
)
def test_workflow_rejects_completion_artifacts_reserved_or_duplicated(
    tmp_path: Path,
    artifact_declaration: str,
) -> None:
    case_path = tmp_path / "case.toml"
    case_path.write_text(
        "\n".join(
            (
                'schema_version = "rasim-layered-fit-workflow-v1"',
                'material_id = "test-material"',
                'model_family = "bi2x3_quintuple.v1"',
                'backend = "cpu"',
                "[stages.geometry]",
                f"completion_artifacts = {artifact_declaration}",
                "commands = []",
                "[stages.mosaic]",
                'completion_artifacts = ["result.txt"]',
                "commands = []",
                "[stages.sf]",
                'completion_artifacts = ["result.txt"]',
                "commands = []",
            )
        ),
        encoding="utf-8",
    )
    workflow = load_fit_workflow(case_path, output_directory=tmp_path / "run")

    with pytest.raises(ValueError, match="collide with stage transaction files"):
        run_fit_workflow(workflow, through="geometry")


@pytest.mark.parametrize("invalid_value", (True, "0.25", 10**400))
def test_fit_parameter_seed_rejects_non_numeric_or_overflowing_values(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    source = ROOT / "configs" / "fit_workflows" / "bi2se3_current_fit.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["structure_function"]["parameter_values"][0] = invalid_value
    path = tmp_path / "invalid-seed.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="finite JSON number"):
        workflow_module._load_fit_parameter_seed(
            path,
            material_id="Bi2Se3",
            model_family="bi2x3_quintuple.v1",
        )


def test_workflow_failed_first_command_cannot_resume_partial_output(tmp_path: Path) -> None:
    case_path = tmp_path / "case.toml"
    output_directory = tmp_path / "run"
    call_log = tmp_path / "calls.txt"
    failure = (
        f"from pathlib import Path; Path(r'{call_log}').write_text('called'); raise SystemExit(2)"
    )
    case_path.write_text(
        "\n".join(
            (
                'schema_version = "rasim-layered-fit-workflow-v1"',
                'material_id = "test-material"',
                'model_family = "bi2x3_quintuple.v1"',
                'backend = "cpu"',
                "[stages.geometry]",
                'completion_artifacts = ["result.txt"]',
                f'commands = [[{_quoted_toml(sys.executable)}, "-c", {_quoted_toml(failure)}]]',
                "[stages.mosaic]",
                'completion_artifacts = ["result.txt"]',
                "commands = []",
                "[stages.sf]",
                'completion_artifacts = ["result.txt"]',
                "commands = []",
            )
        ),
        encoding="utf-8",
    )
    workflow = load_fit_workflow(case_path, output_directory=output_directory)

    with pytest.raises(subprocess.CalledProcessError):
        run_fit_workflow(workflow, through="geometry")
    progress_path = output_directory / "geometry" / "stage.progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "started"
    assert progress["completed_command_count"] == 0
    assert not (output_directory / "geometry" / "stage.json").exists()
    with pytest.raises(ValueError, match="incomplete outer workflow stage"):
        run_fit_workflow(workflow, through="geometry")
    assert call_log.read_text(encoding="utf-8") == "called"


def test_workflow_rejects_input_mutation_during_a_command(tmp_path: Path) -> None:
    case_path = tmp_path / "case.toml"
    output_directory = tmp_path / "run"
    stage_input = tmp_path / "input.txt"
    stage_input.write_text("before", encoding="utf-8")
    mutation = (
        "from pathlib import Path; "
        f"Path(r'{stage_input}').write_text('after'); "
        "out=Path(r'{stage_dir}/result.txt'); "
        "out.parent.mkdir(parents=True,exist_ok=True); out.write_text('result')"
    )
    case_path.write_text(
        "\n".join(
            (
                'schema_version = "rasim-layered-fit-workflow-v1"',
                'material_id = "test-material"',
                'model_family = "bi2x3_quintuple.v1"',
                'backend = "cpu"',
                "[stages.geometry]",
                'completion_artifacts = ["result.txt"]',
                "commands = [["
                + ", ".join(
                    (
                        _quoted_toml(sys.executable),
                        '"-c"',
                        _quoted_toml(mutation),
                        _quoted_toml(stage_input),
                    )
                )
                + "]]",
                "[stages.mosaic]",
                'completion_artifacts = ["result.txt"]',
                "commands = []",
                "[stages.sf]",
                'completion_artifacts = ["result.txt"]',
                "commands = []",
            )
        ),
        encoding="utf-8",
    )
    workflow = load_fit_workflow(case_path, output_directory=output_directory)

    with pytest.raises(RuntimeError, match="workflow inputs changed"):
        run_fit_workflow(workflow, through="geometry")
    progress = json.loads(
        (output_directory / "geometry" / "stage.progress.json").read_text(encoding="utf-8")
    )
    assert progress["completed_command_count"] == 0
    assert not (output_directory / "geometry" / "stage.json").exists()


@pytest.mark.parametrize(
    ("relative_path", "material_id", "profile_replay"),
    (
        ("configs/fit_workflows/bi2se3.toml", "Bi2Se3", False),
        ("configs/fit_workflows/bi2te3.toml", "Bi2Te3", True),
    ),
)
def test_nominal_bi2x3_cases_share_one_stage_contract(
    tmp_path: Path,
    relative_path: str,
    material_id: str,
    profile_replay: bool,
) -> None:
    output_directory = tmp_path / f"{material_id} nominal plan"
    workflow = load_fit_workflow(ROOT / relative_path, output_directory=output_directory)
    plan = planned_fit_workflow(workflow)

    assert workflow.material_id == material_id
    assert workflow.model_family == "bi2x3_quintuple.v1"
    assert workflow.fit_parameter_seed is not None
    assert workflow.fit_parameter_seed.material_id == material_id
    assert workflow.fit_parameter_seed.model_family == workflow.model_family
    fit_command = plan["stages"][-1]["commands"][3]
    fit_plan_path = Path(fit_command[fit_command.index("--fit-plan") + 1])
    fit_plan = tomllib.loads(fit_plan_path.read_text(encoding="utf-8"))
    assert workflow.fit_parameter_seed.parameter_names == tuple(fit_plan["parameter_names"])
    assert len(workflow.fit_parameter_seed.parameter_values) == 5
    expected_vacancy = 0.011 if material_id == "Bi2Se3" else 0.0
    parameter_values = dict(
        zip(
            workflow.fit_parameter_seed.parameter_names,
            workflow.fit_parameter_seed.parameter_values,
            strict=True,
        )
    )
    assert parameter_values["outer_chalcogen_vacancy_fraction"] == pytest.approx(expected_vacancy)
    assert "vacancy" in workflow.fit_parameter_seed.structure_seed_semantics
    assert "not a relabelled antisite estimate" in (
        workflow.fit_parameter_seed.structure_seed_semantics
    )
    assert set(dict(workflow.fit_parameter_seed.dataset_scales)) == {
        f"{material_id}-5deg",
        f"{material_id}-10deg",
        f"{material_id}-15deg",
    }
    assert (
        workflow.fit_parameter_seed.profile_specular_interface_assumption
        == "local_lamella_follows_mosaic.v1"
    )
    assert plan["fit_parameter_seed"] == str(workflow.fit_parameter_seed.path)
    assert tuple(workflow.stages) == FIT_WORKFLOW_STAGE_NAMES
    assert workflow.stages["geometry"].commands
    assert workflow.stages["sf"].commands
    assert workflow.stages["mosaic"].completion_artifacts
    assert tuple(stage["stage"] for stage in plan["stages"]) == FIT_WORKFLOW_STAGE_NAMES
    sf_commands = plan["stages"][-1]["commands"]
    assert [Path(command[1]).name for command in sf_commands] == [
        "compose_fixed_experiment.py",
        "fit_layered_quintuple_regions.py",
        "fit_layered_quintuple_regions.py",
        "fit_layered_quintuple_regions.py",
        "fit_layered_quintuple_regions.py",
        "fit_layered_quintuple_regions.py",
    ]
    assert [command[2] for command in sf_commands[1:]] == [
        "prepare",
        "background",
        "fit",
        "profiles",
        "render",
    ]
    fit_command = sf_commands[3]
    initial_index = fit_command.index("--initial-parameters") + 1
    assert tuple(float(value) for value in fit_command[initial_index:]) == (
        workflow.fit_parameter_seed.parameter_values
    )
    profile_command = sf_commands[4]
    if profile_replay:
        assumption_index = profile_command.index("--specular-interface-assumption") + 1
        assert profile_command[assumption_index] == (
            workflow.fit_parameter_seed.profile_specular_interface_assumption
        )
    else:
        assert "--specular-interface-assumption" not in profile_command
    assert sf_commands[5][-2] == "--output-directory"
    assert Path(sf_commands[5][-1]).resolve() == (output_directory / "sf" / "rendered").resolve()
    assert {
        "rendered/" + material_id.lower() + "_figure7_matched_model.png",
        "rendered/" + material_id.lower() + "_figure7_matched_model.pdf",
        "rendered/" + material_id.lower() + "_figure7_matched_model.json",
        "rendered/" + material_id.lower() + "_peak_alignment.json",
    }.issubset(workflow.stages["sf"].completion_artifacts)
    assert not output_directory.exists()


def test_bi2se3_current_workflow_uses_a_separate_model_limited_geometry_manifest(
    tmp_path: Path,
) -> None:
    workflow = load_fit_workflow(
        ROOT / "configs" / "fit_workflows" / "bi2se3.toml",
        output_directory=tmp_path / "bi2se3-current",
    )
    plan = planned_fit_workflow(workflow)
    geometry_manifest = ROOT / "configs" / "bi2se3_osc_geometry_fit_model_limited.yaml"

    geometry_command = plan["stages"][0]["commands"][0]
    compose_command = plan["stages"][2]["commands"][0]
    assert Path(geometry_command[2]).resolve() == geometry_manifest.resolve()
    assert "--fit-incidence-angle-trim" in geometry_command
    assert Path(compose_command[compose_command.index("--geometry-manifest") + 1]).resolve() == (
        geometry_manifest.resolve()
    )
