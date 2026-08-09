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
    ("relative_path", "material_id"),
    (
        ("configs/fit_workflows/bi2se3.toml", "Bi2Se3"),
        ("configs/fit_workflows/bi2te3.toml", "Bi2Te3"),
    ),
)
def test_nominal_bi2x3_cases_share_one_stage_contract(
    tmp_path: Path,
    relative_path: str,
    material_id: str,
) -> None:
    output_directory = tmp_path / f"{material_id} nominal plan"
    workflow = load_fit_workflow(ROOT / relative_path, output_directory=output_directory)
    plan = planned_fit_workflow(workflow)

    assert workflow.material_id == material_id
    assert workflow.model_family == "bi2x3_quintuple.v1"
    assert tuple(workflow.stages) == FIT_WORKFLOW_STAGE_NAMES
    assert workflow.stages["geometry"].commands
    assert workflow.stages["sf"].commands
    assert workflow.stages["mosaic"].completion_artifacts
    assert tuple(stage["stage"] for stage in plan["stages"]) == FIT_WORKFLOW_STAGE_NAMES
    assert not output_directory.exists()
