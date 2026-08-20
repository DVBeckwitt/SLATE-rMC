"""Small autonomous geometry -> mosaic -> structure-factor workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import distributions
from numbers import Real
from pathlib import Path
from typing import Any

from rasim_next.fitting.fixed_experiment import FixedMosaicState, FixedPositionState

FIT_WORKFLOW_SCHEMA = "rasim-layered-fit-workflow-v1"
FIT_WORKFLOW_STAGE_SCHEMA = "rasim-layered-fit-workflow-stage-v3"
FIT_WORKFLOW_PROGRESS_SCHEMA = "rasim-layered-fit-workflow-progress-v2"
FIT_PARAMETER_SEED_SCHEMA = "rasim-layered-material-fit-seed-v2"
FIT_WORKFLOW_STAGE_NAMES = ("geometry", "mosaic", "sf")


@dataclass(frozen=True, slots=True)
class FitWorkflowStage:
    """One independently executable stage command plan."""

    name: str
    commands: tuple[tuple[str, ...], ...]
    completion_artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FitParameterSeed:
    """Current material parameters used to initialize a repeatable fit."""

    path: Path
    material_id: str
    model_family: str
    fixed_position: FixedPositionState
    mosaic: FixedMosaicState
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    dataset_scales: tuple[tuple[str, float], ...]
    structure_seed_semantics: str
    profile_specular_interface_assumption: str


@dataclass(frozen=True, slots=True)
class FitWorkflow:
    """One material-neutral staged-fit case."""

    case_path: Path
    repository_root: Path
    output_directory: Path
    material_id: str
    model_family: str
    backend: str
    fit_parameter_seed: FitParameterSeed | None
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


def _finite_json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite JSON number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite JSON number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite JSON number")
    return result


def _load_fit_parameter_seed(
    path: Path,
    *,
    material_id: str,
    model_family: str,
) -> FitParameterSeed:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load fit parameter seed {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != (
        FIT_PARAMETER_SEED_SCHEMA
    ):
        raise ValueError("unsupported fit parameter seed schema")
    if document.get("material_id") != material_id or document.get("model_family") != model_family:
        raise ValueError("fit parameter seed does not match the workflow material")
    structure = document.get("structure_function")
    profile = document.get("profile")
    provenance = document.get("provenance")
    if (
        not isinstance(structure, dict)
        or not isinstance(profile, dict)
        or not isinstance(provenance, dict)
    ):
        raise ValueError("fit parameter seed is missing structure or profile state")
    raw_names = structure.get("parameter_names")
    raw_values = structure.get("parameter_values")
    raw_scales = structure.get("dataset_scales")
    if (
        not isinstance(raw_names, list)
        or not isinstance(raw_values, list)
        or not isinstance(raw_scales, dict)
    ):
        raise ValueError("fit parameter seed structure state is invalid")
    names = tuple(_nonempty_string(value, "fit parameter name") for value in raw_names)
    values = tuple(_finite_json_number(value, "fit parameter seed value") for value in raw_values)
    scales = tuple(
        (
            _nonempty_string(key, "fit dataset ID"),
            _finite_json_number(value, "fit dataset scale"),
        )
        for key, value in raw_scales.items()
    )
    if (
        not names
        or len(names) != len(values)
        or len(set(names)) != len(names)
        or any(not math.isfinite(value) for value in values)
        or not scales
        or any(not math.isfinite(value) or value <= 0.0 for _, value in scales)
    ):
        raise ValueError("fit parameter seed values are invalid")
    return FitParameterSeed(
        path=path,
        material_id=material_id,
        model_family=model_family,
        fixed_position=FixedPositionState.from_record(document.get("fixed_position")),
        mosaic=FixedMosaicState.from_record(document.get("mosaic")),
        parameter_names=names,
        parameter_values=values,
        dataset_scales=scales,
        structure_seed_semantics=_nonempty_string(
            provenance.get("structure_seed_semantics"),
            "fit parameter seed structure semantics",
        ),
        profile_specular_interface_assumption=_nonempty_string(
            profile.get("specular_interface_assumption"),
            "profile specular interface assumption",
        ),
    )


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
        _nonempty_string(value, f"stages.{name}.completion_artifacts") for value in raw_artifacts
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
    seed_reference = document.get("fit_parameter_seed")
    fit_parameter_seed: FitParameterSeed | None = None
    if seed_reference is not None:
        seed_path = Path(_nonempty_string(seed_reference, "fit_parameter_seed"))
        if not seed_path.is_absolute():
            seed_path = case_path.parent / seed_path
        fit_parameter_seed = _load_fit_parameter_seed(
            seed_path.resolve(),
            material_id=material_id,
            model_family=model_family,
        )
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
        fit_parameter_seed=fit_parameter_seed,
        stages=stages,
    )


def _format_context(workflow: FitWorkflow, stage: str) -> dict[str, str]:
    output = workflow.output_directory
    context = {
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
    if workflow.fit_parameter_seed is not None:
        context["fit_profile_interface_assumption"] = (
            workflow.fit_parameter_seed.profile_specular_interface_assumption
        )
    return context


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
    commands: list[tuple[str, ...]] = []
    for command in stage.commands:
        expanded: list[str] = []
        for argument in command:
            if argument == "{fit_parameter_values}":
                if workflow.fit_parameter_seed is None:
                    raise ValueError("workflow command requires a fit parameter seed")
                expanded.extend(
                    repr(value) for value in workflow.fit_parameter_seed.parameter_values
                )
            else:
                expanded.append(_expand(argument, context))
        commands.append(tuple(expanded))
    stage_directory = workflow.output_directory / stage.name
    artifacts: list[Path] = []
    for declared in stage.completion_artifacts:
        expanded = Path(_expand(declared, context))
        artifacts.append(expanded if expanded.is_absolute() else stage_directory / expanded)
    return tuple(commands), tuple(path.resolve() for path in artifacts)


def _revision(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256-" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _repository_scientific_revision(repository_root: Path) -> str:
    """Bind the canonical current repository content and scientific runtime."""

    try:
        staged_payload = subprocess.check_output(
            ("git", "ls-files", "--stage", "-z"),
            cwd=repository_root,
        )
        tracked_flag_payload = subprocess.check_output(
            ("git", "ls-files", "-v", "-z"),
            cwd=repository_root,
        )
        dirty_payload = subprocess.check_output(
            ("git", "diff-files", "--name-only", "-z"),
            cwd=repository_root,
        )
        untracked_payload = subprocess.check_output(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=repository_root,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "fit workflow requires a readable Git scientific source closure"
        ) from error

    inventory: dict[str, tuple[str, str]] = {}
    hidden_paths = []
    for raw in tracked_flag_payload.split(b"\0"):
        if not raw:
            continue
        marker, encoded_path = raw.split(b" ", 1)
        if marker in {b"h", b"S", b"s"}:
            hidden_paths.append(encoded_path.decode("utf-8"))
    if hidden_paths:
        raise ValueError(
            f"fit workflow rejects assume-unchanged or skip-worktree inputs: {hidden_paths[0]}"
        )
    for raw in staged_payload.split(b"\0"):
        if not raw:
            continue
        metadata, encoded_path = raw.split(b"\t", 1)
        mode, object_id, stage = metadata.split()
        if stage != b"0":
            raise ValueError("fit workflow does not accept unresolved Git index stages")
        inventory[encoded_path.decode("utf-8")] = (
            mode.decode("ascii"),
            object_id.decode("ascii"),
        )

    dirty_paths = {raw.decode("utf-8") for raw in dirty_payload.split(b"\0") if raw}
    untracked_paths = {raw.decode("utf-8") for raw in untracked_payload.split(b"\0") if raw}
    filemode_result = subprocess.run(
        ("git", "config", "--bool", "core.filemode"),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if filemode_result.returncode not in {0, 1}:
        raise ValueError("fit workflow could not read the Git file-mode policy")
    core_filemode = filemode_result.stdout.strip() == "true"
    for relative in sorted(dirty_paths | untracked_paths):
        path = repository_root / relative
        if not path.is_file() and not path.is_symlink():
            inventory.pop(relative, None)
            continue
        is_symlink = path.is_symlink()
        indexed = inventory.get(relative)
        if is_symlink:
            mode = "120000"
        elif indexed is not None and not core_filemode:
            mode = indexed[0]
        else:
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        try:
            if is_symlink:
                object_id = (
                    subprocess.check_output(
                        ("git", "hash-object", "--stdin"),
                        cwd=repository_root,
                        input=os.readlink(path).encode("utf-8"),
                    )
                    .decode("ascii")
                    .strip()
                )
            else:
                object_id = subprocess.check_output(
                    ("git", "hash-object", f"--path={relative}", relative),
                    cwd=repository_root,
                    text=True,
                ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"failed to hash live repository input {relative}") from error
        inventory[relative] = (mode, object_id)

    runtime_packages: dict[str, str] = {}
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = raw_name.lower().replace("_", "-")
        installed = distribution.version
        previous = runtime_packages.setdefault(name, installed)
        if previous != installed:
            raise ValueError(f"fit workflow runtime has conflicting versions of {name}")
    lock_path = repository_root / "uv.lock"
    environment_lock_sha256 = _file_sha256(lock_path) if lock_path.is_file() else None
    return _revision(
        {
            "model_id": "git_current_content_and_runtime_closure.v4",
            "repository_content": [
                (path, mode, object_id) for path, (mode, object_id) in sorted(inventory.items())
            ],
            "environment_lock_sha256": environment_lock_sha256,
            "python_version": sys.version,
            "python_executable_sha256": _file_sha256(Path(sys.executable)),
            "runtime_package_version": sorted(runtime_packages.items()),
        }
    )


def _command_input_hashes(
    workflow: FitWorkflow,
    commands: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, str], ...]:
    candidates = {workflow.case_path.resolve()}
    if workflow.fit_parameter_seed is not None:
        candidates.add(workflow.fit_parameter_seed.path.resolve())
    for command in commands:
        for index, argument in enumerate(command):
            candidate = Path(argument)
            if not candidate.is_absolute():
                executable = (
                    shutil.which(argument) if index == 0 and candidate.name == argument else None
                )
                candidate = (
                    Path(executable)
                    if executable is not None
                    else workflow.repository_root / candidate
                )
            try:
                resolved = candidate.resolve()
                is_file = resolved.is_file()
            except OSError:
                continue
            if is_file and not resolved.is_relative_to(workflow.output_directory):
                candidates.add(resolved)
    return tuple((str(path), _file_sha256(path)) for path in sorted(candidates))


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _exclusive_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as error:
        raise ValueError(f"workflow state already exists at {path}") from error


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
    repository_scientific_revision: str,
) -> tuple[tuple[tuple[str, ...], ...], tuple[Path, ...], str]:
    commands, artifacts = _expanded_stage(workflow, stage)
    command_input_hashes = _command_input_hashes(workflow, commands)
    plan_revision = _revision(
        {
            "schema": FIT_WORKFLOW_SCHEMA,
            "stage": stage.name,
            "material_id": workflow.material_id,
            "model_family": workflow.model_family,
            "backend": workflow.backend,
            "commands": commands,
            "command_input_sha256": command_input_hashes,
            "repository_scientific_revision": repository_scientific_revision,
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
    repository_scientific_revision = _repository_scientific_revision(workflow.repository_root)
    plans: list[dict[str, object]] = []
    upstream_revision: str | None = None
    for name in FIT_WORKFLOW_STAGE_NAMES:
        stage = workflow.stages[name]
        commands, artifacts, plan_revision = _stage_plan(
            workflow,
            stage,
            upstream_revision,
            repository_scientific_revision,
        )
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
        "repository_scientific_revision": repository_scientific_revision,
        "fit_parameter_seed": (
            None if workflow.fit_parameter_seed is None else str(workflow.fit_parameter_seed.path)
        ),
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


def _stage_revision(
    *,
    stage: str,
    plan_revision: str,
    repository_scientific_revision: str,
    upstream_revision: str | None,
    artifact_hashes: Mapping[str, str],
) -> str:
    return _revision(
        {
            "schema": FIT_WORKFLOW_STAGE_SCHEMA,
            "stage": stage,
            "plan_revision": plan_revision,
            "repository_scientific_revision": repository_scientific_revision,
            "upstream_revision": upstream_revision,
            "completion_artifact_sha256": artifact_hashes,
        }
    )


def _reused_result(
    manifest_path: Path,
    *,
    stage: str,
    plan_revision: str,
    repository_scientific_revision: str,
    upstream_revision: str | None,
    artifacts: tuple[Path, ...],
    material_id: str,
    model_family: str,
) -> FitWorkflowStageResult | None:
    if not manifest_path.exists():
        return None
    document = _json_mapping(manifest_path)
    expected_paths = [str(path) for path in artifacts]
    if (
        document.get("schema_version") != FIT_WORKFLOW_STAGE_SCHEMA
        or document.get("stage") != stage
        or document.get("material_id") != material_id
        or document.get("model_family") != model_family
        or document.get("plan_revision") != plan_revision
        or document.get("repository_scientific_revision") != repository_scientific_revision
        or document.get("upstream_revision") != upstream_revision
        or document.get("completion_artifacts") != expected_paths
    ):
        raise ValueError(f"completed {stage} stage does not match this workflow plan")
    if any(not path.exists() for path in artifacts):
        raise ValueError(f"completed {stage} stage is missing a declared artifact")
    expected_hashes = {str(path): _file_sha256(path) for path in artifacts}
    if document.get("completion_artifact_sha256") != expected_hashes:
        raise ValueError(f"completed {stage} stage artifact content changed")
    revision = _stage_revision(
        stage=stage,
        plan_revision=plan_revision,
        repository_scientific_revision=repository_scientific_revision,
        upstream_revision=upstream_revision,
        artifact_hashes=expected_hashes,
    )
    if document.get("stage_revision") != revision:
        raise ValueError(f"completed {stage} stage revision changed")
    return FitWorkflowStageResult(
        stage=stage,
        reused=True,
        manifest_path=manifest_path,
        revision=revision,
        completion_artifacts=artifacts,
    )


def _reject_existing_progress(
    progress_path: Path,
    *,
    stage: str,
    plan_revision: str,
    repository_scientific_revision: str,
    upstream_revision: str | None,
    command_count: int,
) -> None:
    if not progress_path.exists():
        return
    document = _json_mapping(progress_path)
    completed = document.get("completed_command_count")
    if (
        document.get("schema_version") != FIT_WORKFLOW_PROGRESS_SCHEMA
        or document.get("stage") != stage
        or document.get("status") != "started"
        or document.get("plan_revision") != plan_revision
        or document.get("repository_scientific_revision") != repository_scientific_revision
        or document.get("upstream_revision") != upstream_revision
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or completed > command_count
    ):
        raise ValueError(f"saved {stage} progress does not match this workflow plan")
    raise ValueError(
        f"saved {stage} progress is an incomplete outer workflow stage; "
        "choose a new output directory instead of reusing intermediate commands"
    )


def _run_stage(
    workflow: FitWorkflow,
    stage: FitWorkflowStage,
    upstream_revision: str | None,
    repository_scientific_revision: str,
    *,
    execute: bool,
) -> FitWorkflowStageResult:
    stage_directory = workflow.output_directory / stage.name
    manifest_path = stage_directory / "stage.json"
    progress_path = stage_directory / "stage.progress.json"
    commands, artifacts, plan_revision = _stage_plan(
        workflow,
        stage,
        upstream_revision,
        repository_scientific_revision,
    )
    reserved_paths = {
        manifest_path,
        progress_path,
        manifest_path.with_name(manifest_path.name + ".partial"),
        progress_path.with_name(progress_path.name + ".partial"),
    }
    if len(set(artifacts)) != len(artifacts) or any(path in reserved_paths for path in artifacts):
        raise ValueError("workflow completion artifacts collide with stage transaction files")

    def verify_active_plan() -> None:
        current_repository_revision = _repository_scientific_revision(workflow.repository_root)
        if current_repository_revision != repository_scientific_revision:
            raise RuntimeError("repository or scientific runtime changed during workflow stage")
        _, _, current_plan_revision = _stage_plan(
            workflow,
            stage,
            upstream_revision,
            current_repository_revision,
        )
        if current_plan_revision != plan_revision:
            raise RuntimeError("workflow inputs changed during workflow stage")

    _reject_existing_progress(
        progress_path,
        stage=stage.name,
        plan_revision=plan_revision,
        repository_scientific_revision=repository_scientific_revision,
        upstream_revision=upstream_revision,
        command_count=len(commands),
    )
    reused = _reused_result(
        manifest_path,
        stage=stage.name,
        plan_revision=plan_revision,
        repository_scientific_revision=repository_scientific_revision,
        upstream_revision=upstream_revision,
        artifacts=artifacts,
        material_id=workflow.material_id,
        model_family=workflow.model_family,
    )
    if reused is not None:
        verify_active_plan()
        confirmed = _reused_result(
            manifest_path,
            stage=stage.name,
            plan_revision=plan_revision,
            repository_scientific_revision=repository_scientific_revision,
            upstream_revision=upstream_revision,
            artifacts=artifacts,
            material_id=workflow.material_id,
            model_family=workflow.model_family,
        )
        if confirmed is None:
            raise RuntimeError(f"completed {stage.name} stage disappeared during reuse")
        verify_active_plan()
        return confirmed
    if not execute:
        raise ValueError(f"independent {stage.name} stage requires completed predecessors")
    stage_directory.mkdir(parents=True, exist_ok=True)
    if any(stage_directory.iterdir()):
        raise ValueError(
            f"unmanifested content exists in the {stage.name} stage directory; "
            "choose a new output directory"
        )
    _exclusive_json(
        progress_path,
        {
            "schema_version": FIT_WORKFLOW_PROGRESS_SCHEMA,
            "stage": stage.name,
            "status": "started",
            "plan_revision": plan_revision,
            "repository_scientific_revision": repository_scientific_revision,
            "upstream_revision": upstream_revision,
            "completed_command_count": 0,
        },
    )

    for command_index, command in enumerate(commands):
        verify_active_plan()
        subprocess.run(command, cwd=workflow.repository_root, check=True)
        verify_active_plan()
        _atomic_json(
            progress_path,
            {
                "schema_version": FIT_WORKFLOW_PROGRESS_SCHEMA,
                "stage": stage.name,
                "status": "started",
                "plan_revision": plan_revision,
                "repository_scientific_revision": repository_scientific_revision,
                "upstream_revision": upstream_revision,
                "completed_command_count": command_index + 1,
            },
        )
    verify_active_plan()
    missing = [path for path in artifacts if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{stage.name} stage completed its commands without artifact {missing[0]}"
        )
    artifact_hashes = {str(path): _file_sha256(path) for path in artifacts}
    verify_active_plan()
    stage_revision = _stage_revision(
        stage=stage.name,
        plan_revision=plan_revision,
        repository_scientific_revision=repository_scientific_revision,
        upstream_revision=upstream_revision,
        artifact_hashes=artifact_hashes,
    )
    _atomic_json(
        manifest_path,
        {
            "schema_version": FIT_WORKFLOW_STAGE_SCHEMA,
            "stage": stage.name,
            "material_id": workflow.material_id,
            "model_family": workflow.model_family,
            "plan_revision": plan_revision,
            "repository_scientific_revision": repository_scientific_revision,
            "upstream_revision": upstream_revision,
            "stage_revision": stage_revision,
            "completion_artifacts": [str(path) for path in artifacts],
            "completion_artifact_sha256": artifact_hashes,
        },
    )
    verify_active_plan()
    final_artifact_hashes = {str(path): _file_sha256(path) for path in artifacts}
    if final_artifact_hashes != artifact_hashes:
        raise RuntimeError(f"{stage.name} stage artifact changed during manifest publication")
    verify_active_plan()
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
    repository_scientific_revision = _repository_scientific_revision(workflow.repository_root)
    results: list[FitWorkflowStageResult] = []
    upstream_revision: str | None = None
    for name in FIT_WORKFLOW_STAGE_NAMES:
        if only is None and name not in selected:
            break
        result = _run_stage(
            workflow,
            workflow.stages[name],
            upstream_revision,
            repository_scientific_revision,
            execute=name in selected,
        )
        upstream_revision = result.revision
        if name in selected:
            results.append(result)
        if only is not None and name == only:
            break
    return tuple(results)


__all__ = [
    "FIT_PARAMETER_SEED_SCHEMA",
    "FIT_WORKFLOW_SCHEMA",
    "FIT_WORKFLOW_STAGE_NAMES",
    "FitParameterSeed",
    "FitWorkflow",
    "FitWorkflowStage",
    "FitWorkflowStageResult",
    "load_fit_workflow",
    "planned_fit_workflow",
    "run_fit_workflow",
]
