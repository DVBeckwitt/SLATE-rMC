"""Small autonomous geometry -> mosaic -> structure-factor workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIT_WORKFLOW_SCHEMA = "rasim-layered-fit-workflow-v1"
FIT_WORKFLOW_STAGE_SCHEMA = "rasim-layered-fit-workflow-stage-v1"
FIT_WORKFLOW_PROGRESS_SCHEMA = "rasim-layered-fit-workflow-progress-v1"
FIT_WORKFLOW_STAGE_NAMES = ("geometry", "mosaic", "sf")


@dataclass(frozen=True, slots=True)
class FitWorkflowStage:
    """One independently executable stage command plan."""

    name: str
    commands: tuple[tuple[str, ...], ...]
    completion_artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FitWorkflow:
    """One material-neutral staged-fit case."""

    case_path: Path
    repository_root: Path
    output_directory: Path
    material_id: str
    model_family: str
    backend: str
    stages: Mapping[str, FitWorkflowStage]


@dataclass(frozen=True, slots=True)
class FitWorkflowStageResult:
    """Completed or reused workflow-stage result."""

    stage: str
    reused: bool
    manifest_path: Path
    revision: str
    completion_artifacts: tuple[Path, ...]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _stage_from_record(name: str, record: object) -> FitWorkflowStage:
    if not isinstance(record, dict):
        raise ValueError(f"stages.{name} must be a table")
    raw_commands = record.get("commands", ())
    raw_artifacts = record.get("completion_artifacts")
    if not isinstance(raw_commands, list):
        raise ValueError(f"stages.{name}.commands must be an array")
    commands: list[tuple[str, ...]] = []
    for command_index, command in enumerate(raw_commands):
        if not isinstance(command, list) or not command:
            raise ValueError(f"stages.{name}.commands[{command_index}] must be a nonempty array")
        commands.append(
            tuple(
                _nonempty_string(argument, f"stages.{name}.commands[{command_index}]")
                for argument in command
            )
        )
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError(f"stages.{name}.completion_artifacts must be a nonempty array")
    artifacts = tuple(
        _nonempty_string(value, f"stages.{name}.completion_artifacts")
        for value in raw_artifacts
    )
    return FitWorkflowStage(name=name, commands=tuple(commands), completion_artifacts=artifacts)


def load_fit_workflow(
    path: str | Path,
    *,
    output_directory: str | Path,
) -> FitWorkflow:
    """Load one three-stage workflow without constructing scientific state."""

    case_path = Path(path).resolve()
    try:
        document = tomllib.loads(case_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"failed to load fit workflow {case_path}: {error}") from error
    if document.get("schema_version") != FIT_WORKFLOW_SCHEMA:
        raise ValueError("unsupported layered-fit workflow schema")
    material_id = _nonempty_string(document.get("material_id"), "material_id")
    model_family = _nonempty_string(document.get("model_family"), "model_family")
    backend = _nonempty_string(document.get("backend", "cuda"), "backend")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    stage_records = document.get("stages")
    if not isinstance(stage_records, dict) or tuple(stage_records) != FIT_WORKFLOW_STAGE_NAMES:
        raise ValueError("workflow stages must be declared once in geometry, mosaic, sf order")
    stages = {
        name: _stage_from_record(name, stage_records[name]) for name in FIT_WORKFLOW_STAGE_NAMES
    }
    repository_root = _repository_root()
    resolved_output = Path(output_directory).resolve()
    if resolved_output == repository_root or resolved_output.is_relative_to(repository_root):
        raise ValueError("fit workflow output must be outside the repository")
    return FitWorkflow(
        case_path=case_path,
        repository_root=repository_root,
        output_directory=resolved_output,
        material_id=material_id,
        model_family=model_family,
        backend=backend,
        stages=stages,
    )


def _format_context(workflow: FitWorkflow, stage: str) -> dict[str, str]:
    output = workflow.output_directory
    return {
        "python": sys.executable,
        "repo": str(workflow.repository_root),
        "case_dir": str(workflow.case_path.parent),
        "output_dir": str(output),
        "geometry_dir": str(output / "geometry"),
        "mosaic_dir": str(output / "mosaic"),
        "sf_dir": str(output / "sf"),
        "stage_dir": str(output / stage),
        "backend": workflow.backend,
        "material_id": workflow.material_id,
        "model_family": workflow.model_family,
    }


def _expand(value: str, context: Mapping[str, str]) -> str:
    try:
        return value.format_map(context)
    except KeyError as error:
        raise ValueError(f"unknown workflow placeholder {error.args[0]!r}") from error


def _expanded_stage(
    workflow: FitWorkflow,
    stage: FitWorkflowStage,
) -> tuple[tuple[tuple[str, ...], ...], tuple[Path, ...]]:
    context = _format_context(workflow, stage.name)
    commands = tuple(
        tuple(_expand(argument, context) for argument in command) for command in stage.commands
    )
    stage_directory = workflow.output_directory / stage.name
    artifacts: list[Path] = []
    for declared in stage.completion_artifacts:
        expanded = Path(_expand(declared, context))
        artifacts.append(expanded if expanded.is_absolute() else stage_directory / expanded)
    return commands, tuple(path.resolve() for path in artifacts)


def _revision(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256-" + hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read workflow state {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"workflow state {path} must contain a JSON object")
    return value


def _stage_plan(
    workflow: FitWorkflow,
    stage: FitWorkflowStage,
    upstream_revision: str | None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[Path, ...], str]:
    commands, artifacts = _expanded_stage(workflow, stage)
    plan_revision = _revision(
        {
            "schema": FIT_WORKFLOW_SCHEMA,
            "stage": stage.name,
            "material_id": workflow.material_id,
            "model_family": workflow.model_family,
            "backend": workflow.backend,
            "commands": commands,
            "completion_artifacts": [str(path) for path in artifacts],
            "upstream_revision": upstream_revision,
        }
    )
    return commands, artifacts, plan_revision


def planned_fit_workflow(
    workflow: FitWorkflow,
    *,
    through: str = "sf",
    only: str | None = None,
) -> dict[str, object]:
    """Return the expanded command plan without creating output."""

    selected = _selected_stages(through=through, only=only)
    plans: list[dict[str, object]] = []
    upstream_revision: str | None = None
    for name in FIT_WORKFLOW_STAGE_NAMES:
        stage = workflow.stages[name]
        commands, artifacts, plan_revision = _stage_plan(workflow, stage, upstream_revision)
        if name in selected:
            plans.append(
                {
                    "stage": name,
                    "commands": [list(command) for command in commands],
                    "completion_artifacts": [str(path) for path in artifacts],
                }
            )
        upstream_revision = _revision({"stage": name, "plan_revision": plan_revision})
    return {
        "schema_version": FIT_WORKFLOW_SCHEMA,
        "material_id": workflow.material_id,
        "model_family": workflow.model_family,
        "backend": workflow.backend,
        "stages": plans,
    }


def _selected_stages(*, through: str, only: str | None) -> tuple[str, ...]:
    if through not in FIT_WORKFLOW_STAGE_NAMES:
        raise ValueError(f"unknown terminal stage {through!r}")
    if only is not None:
        if only not in FIT_WORKFLOW_STAGE_NAMES:
            raise ValueError(f"unknown independent stage {only!r}")
        return (only,)
    terminal = FIT_WORKFLOW_STAGE_NAMES.index(through)
    return FIT_WORKFLOW_STAGE_NAMES[: terminal + 1]


def _reused_result(
    manifest_path: Path,
    *,
    stage: str,
    plan_revision: str,
    upstream_revision: str | None,
    artifacts: tuple[Path, ...],
) -> FitWorkflowStageResult | None:
    if not manifest_path.exists():
        return None
    document = _json_mapping(manifest_path)
    expected_paths = [str(path) for path in artifacts]
    if (
        document.get("schema_version") != FIT_WORKFLOW_STAGE_SCHEMA
        or document.get("stage") != stage
        or document.get("plan_revision") != plan_revision
        or document.get("upstream_revision") != upstream_revision
        or document.get("completion_artifacts") != expected_paths
    ):
        raise ValueError(f"completed {stage} stage does not match this workflow plan")
    if any(not path.exists() for path in artifacts):
        raise ValueError(f"completed {stage} stage is missing a declared artifact")
    revision = _nonempty_string(document.get("stage_revision"), "stage_revision")
    return FitWorkflowStageResult(
        stage=stage,
        reused=True,
        manifest_path=manifest_path,
        revision=revision,
        completion_artifacts=artifacts,
    )


def _progress_count(
    progress_path: Path,
    *,
    stage: str,
    plan_revision: str,
    upstream_revision: str | None,
    command_count: int,
) -> int:
    if not progress_path.exists():
        return 0
    document = _json_mapping(progress_path)
    completed = document.get("completed_command_count")
    if (
        document.get("schema_version") != FIT_WORKFLOW_PROGRESS_SCHEMA
        or document.get("stage") != stage
        or document.get("plan_revision") != plan_revision
        or document.get("upstream_revision") != upstream_revision
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or completed > command_count
    ):
        raise ValueError(f"saved {stage} progress does not match this workflow plan")
    return completed


def _run_stage(
    workflow: FitWorkflow,
    stage: FitWorkflowStage,
    upstream_revision: str | None,
    *,
    execute: bool,
) -> FitWorkflowStageResult:
    stage_directory = workflow.output_directory / stage.name
    manifest_path = stage_directory / "stage.json"
    progress_path = stage_directory / "stage.progress.json"
    commands, artifacts, plan_revision = _stage_plan(workflow, stage, upstream_revision)
    reused = _reused_result(
        manifest_path,
        stage=stage.name,
        plan_revision=plan_revision,
        upstream_revision=upstream_revision,
        artifacts=artifacts,
    )
    if reused is not None:
        return reused
    if not execute:
        raise ValueError(f"independent {stage.name} stage requires completed predecessors")
    stage_directory.mkdir(parents=True, exist_ok=True)
    completed = _progress_count(
        progress_path,
        stage=stage.name,
        plan_revision=plan_revision,
        upstream_revision=upstream_revision,
        command_count=len(commands),
    )
    for command_index, command in enumerate(commands[completed:], start=completed):
        subprocess.run(command, cwd=workflow.repository_root, check=True)
        _atomic_json(
            progress_path,
            {
                "schema_version": FIT_WORKFLOW_PROGRESS_SCHEMA,
                "stage": stage.name,
                "plan_revision": plan_revision,
                "upstream_revision": upstream_revision,
                "completed_command_count": command_index + 1,
            },
        )
    missing = [path for path in artifacts if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{stage.name} stage completed its commands without artifact {missing[0]}"
        )
    stage_revision = _revision(
        {
            "schema": FIT_WORKFLOW_STAGE_SCHEMA,
            "stage": stage.name,
            "plan_revision": plan_revision,
            "upstream_revision": upstream_revision,
        }
    )
    _atomic_json(
        manifest_path,
        {
            "schema_version": FIT_WORKFLOW_STAGE_SCHEMA,
            "stage": stage.name,
            "material_id": workflow.material_id,
            "model_family": workflow.model_family,
            "plan_revision": plan_revision,
            "upstream_revision": upstream_revision,
            "stage_revision": stage_revision,
            "completion_artifacts": [str(path) for path in artifacts],
        },
    )
    progress_path.unlink(missing_ok=True)
    return FitWorkflowStageResult(
        stage=stage.name,
        reused=False,
        manifest_path=manifest_path,
        revision=stage_revision,
        completion_artifacts=artifacts,
    )


def run_fit_workflow(
    workflow: FitWorkflow,
    *,
    through: str = "sf",
    only: str | None = None,
) -> tuple[FitWorkflowStageResult, ...]:
    """Run or resume the selected stages in independent child processes."""

    selected = _selected_stages(through=through, only=only)
    results: list[FitWorkflowStageResult] = []
    upstream_revision: str | None = None
    for name in FIT_WORKFLOW_STAGE_NAMES:
        if only is None and name not in selected:
            break
        result = _run_stage(
            workflow,
            workflow.stages[name],
            upstream_revision,
            execute=name in selected,
        )
        upstream_revision = result.revision
        if name in selected:
            results.append(result)
        if only is not None and name == only:
            break
    return tuple(results)


__all__ = [
    "FIT_WORKFLOW_SCHEMA",
    "FIT_WORKFLOW_STAGE_NAMES",
    "FitWorkflow",
    "FitWorkflowStage",
    "FitWorkflowStageResult",
    "load_fit_workflow",
    "planned_fit_workflow",
    "run_fit_workflow",
]
