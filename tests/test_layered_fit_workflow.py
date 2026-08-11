from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rasim_next.fitting.workflow import (
    FIT_WORKFLOW_STAGE_NAMES,
    load_fit_workflow,
    planned_fit_workflow,
    run_fit_workflow,
)

ROOT = Path(__file__).resolve().parents[1]


def _quoted_toml(value: str | Path) -> str:
    return json.dumps(str(value))


def test_workflow_runs_geometry_mosaic_sf_once_and_resumes(tmp_path: Path) -> None:
    case_path = tmp_path / "case.toml"
    output_directory = tmp_path / "run"
    call_log = tmp_path / "calls.txt"
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
                    + ", ".join((_quoted_toml(sys.executable), '"-c"', _quoted_toml(code)))
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
    assert len(workflow.fit_parameter_seed.parameter_names) == 5
    assert len(workflow.fit_parameter_seed.parameter_values) == 5
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
