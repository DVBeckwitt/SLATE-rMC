"""Bounded prepared-input descriptions; editing never admits an engine launch."""

import json
from dataclasses import asdict, dataclass
from uuid import UUID

from sample_state import encoded, payload_hash


def document(text, limit=128 * 1024):
    if type(text) is not str or not 0 < len(text.encode()) <= limit:
        raise ValueError("Prepared description exceeds its bounded object admission")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Prepared description must be an object")
    encoded(value)
    return value


def validate_description(value):
    if (
        set(value) != {"schema", "inputs", "plan", "definition", "sha256"}
        or value["schema"] != "slate.prepared-draft.v1"
    ):
        raise ValueError("Unsupported prepared draft description")
    if value["sha256"] != payload_hash({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Prepared draft description identity differs")
    inputs, definition, plan = value["inputs"], value["definition"], value["plan"]
    if (
        inputs["schema"] != "slate.native-fit-inputs.v1"
        or plan["schema"] != "rasim-native-refinement-plan-v1"
    ):
        raise ValueError("Unsupported prepared input/plan schema")
    files = inputs["files"]
    if not 4 <= len(files) <= 32 or len({v["path"] for v in files}) != len(files):
        raise ValueError("Prepared predecessor references must be complete and unique")
    from pathlib import Path

    for row in files:
        if (
            set(row) != {"kind", "path", "sha256", "size"}
            or not Path(row["path"]).is_absolute()
            or len(row["path"]) > 4096
            or type(row["size"]) is not int
            or row["size"] < 0
        ):
            raise ValueError("Malformed prepared predecessor")
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("Prepared predecessor needs SHA256")
    if definition["sha256"] != payload_hash({k: v for k, v in definition.items() if k != "sha256"}):
        raise ValueError("Parameter definition identity differs")
    if definition["declared_parameters"] != plan["parameters"]:
        raise ValueError("Saved parameter definitions differ from the plan")
    if [(p["owner"], p["name"], p["unit"]) for p in definition["capabilities"]] != [
        (p["owner"], p["name"], p["unit"]) for p in plan["parameters"]
    ]:
        raise ValueError("Saved capabilities differ from the parameter identities")
    for key in (
        "physics_sha256",
        "observation_sha256",
        "projection_revision",
        "raw_acquisition_sha256",
    ):
        digest = inputs[key]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Prepared inputs require exact SHA256 identities")
    by_kind = {v["kind"]: v for v in files}
    if (
        not {"physics", "observations", "plan", "arrays", "raw_acquisition"} <= set(by_kind)
        or by_kind["physics"]["sha256"] != inputs["physics_sha256"]
        or by_kind["observations"]["sha256"] != inputs["observation_sha256"]
        or by_kind["raw_acquisition"]["sha256"] != inputs["raw_acquisition_sha256"]
    ):
        raise ValueError("Prepared input identities differ from predecessor references")
    keys = [(p["owner"], p["name"]) for p in plan["parameters"]]
    if not 0 < len(keys) <= 128 or len(set(keys)) != len(keys):
        raise ValueError("Prepared parameters require unique owner/name identities")
    if not 0 < len(plan["starts"]) <= 32 or any(len(v) != len(keys) for v in plan["starts"]):
        raise ValueError("Prepared starts must align with the frozen parameter order")
    if definition["ordered_identities"] != [list(k) for k in keys]:
        raise ValueError("Definition identities differ from the draft parameter order")
    return value


def description(inputs, plan, definition):
    value = {
        "schema": "slate.prepared-draft.v1",
        "inputs": inputs,
        "plan": plan,
        "definition": definition,
    }
    value["sha256"] = payload_hash(value)
    validate_description(value)
    return encoded(value)


def reuse_plan(old, current):
    """Explicit review: align unchanged definitions by owner/name; preserve every other field."""
    old, current = validate_description(old), validate_description(current)
    if (
        old["definition"]["engine_revision"] != current["definition"]["engine_revision"]
        or old["definition"]["model_id"] != current["definition"]["model_id"]
    ):
        raise ValueError("Engine/model definition changed; review the newly loaded plan")
    before = {(p["owner"], p["name"]): (i, p) for i, p in enumerate(old["plan"]["parameters"])}
    after = {(p["owner"], p["name"]): (i, p) for i, p in enumerate(current["plan"]["parameters"])}
    if set(before) != set(after):
        raise ValueError(
            "Added/removed/scope-changed parameters require explicit review of authoritative loaded starts; old fields retained in history"
        )
    for key in before:
        if before[key][1] != after[key][1]:
            raise ValueError(
                "Unit/domain/bound/scale definition changed for " + key[1] + "; reuse unavailable"
            )
    plan = json.loads(encoded(current["plan"]))
    plan["starts"] = [
        [row[before[(p["owner"], p["name"])][0]] for p in plan["parameters"]]
        for row in old["plan"]["starts"]
    ]
    if old["plan"].get("fixed_parameters", {}) != plan.get("fixed_parameters", {}):
        raise ValueError("Fixed-state definition changed; review the newly loaded plan")
    return plan


@dataclass(frozen=True, slots=True)
class NativeFitSession:
    session_id: UUID
    current_json: str
    history_json: tuple[str, ...] = ()
    draft_json: str | None = None
    revision: int = 0
    exports: tuple[tuple[str, str], ...] = ()
    selected_stage: str | None = None

    def __post_init__(self):
        if (
            not isinstance(self.session_id, UUID)
            or type(self.revision) is not int
            or not 0 <= self.revision < 2**63
        ):
            raise ValueError("Invalid prepared session identity/revision")
        validate_description(document(self.current_json, 192 * 1024))
        if type(self.history_json) is not tuple or len(self.history_json) > 8:
            raise ValueError("Prepared history is full; explicitly remove a historical description")
        for text in self.history_json:
            validate_description(document(text, 192 * 1024))
        stages = json.loads(self.current_json)["plan"].get("stages", ())
        if self.selected_stage is not None and (
            type(self.selected_stage) is not str
            or self.selected_stage not in {s["name"] for s in stages}
        ):
            raise ValueError("Selected stage must name a current declared stage")
        if self.draft_json is not None:
            pending = document(self.draft_json, 96 * 1024)
            current = json.loads(self.current_json)["plan"]
            identities = {encoded((p["owner"], p["name"])) for p in current["parameters"]}
            stages = {s["name"] for s in current.get("stages", ())}
            if (
                set(pending) != {"parameters", "stages", "step", "start_index"}
                or type(pending["start_index"]) is not int
                or not 0 <= pending["start_index"] < len(current["starts"])
                or set(pending["parameters"]) - identities
                or set(pending["stages"]) - stages
            ):
                raise ValueError("Pending edit identities differ from the current definitions")
            if any(set(v) - {"3", "4", "6"} for v in pending["parameters"].values()) or any(
                set(v)
                - {
                    "method",
                    "maximum_iterations",
                    "maximum_function_evaluations",
                    "active_parameters",
                    "enforce_historical_guards",
                }
                for v in pending["stages"].values()
            ):
                raise ValueError("Unsupported pending edit fields")
        if len(self.exports) > 16 or any(
            not p or len(p) > 4096 or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for p, h in self.exports
        ):
            raise ValueError("Invalid bounded prepared export references")

    @property
    def nbytes(self):
        return len(encoded(native_fit_session_document(self)).encode()) + 8192


def native_fit_session_document(session):
    return None if session is None else {**asdict(session), "session_id": str(session.session_id)}


def native_fit_session_from_document(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) not in (
        set(NativeFitSession.__dataclass_fields__),
        set(NativeFitSession.__dataclass_fields__) - {"selected_stage"},
    ):
        raise ValueError("Invalid prepared session fields")
    data = dict(value)
    data.setdefault("selected_stage", None)
    data["session_id"] = UUID(data["session_id"])
    data["history_json"] = tuple(data["history_json"])
    data["exports"] = tuple(tuple(v) for v in data["exports"])
    return NativeFitSession(**data)


def stage_review(value, name, pending=None):
    """Present declared stage ownership; never infer capability from a stage label."""
    plan = value["plan"]
    stages = plan.get("stages", ())
    stage = next((s for s in stages if s["name"] == name), None)
    if stage is None:
        return {"readiness": "No declared stage selected; no template or default is invented"}
    index = next(i for i, s in enumerate(stages) if s["name"] == name)
    pending = {} if pending is None else pending
    declarations = {**stage, **pending.get("stages", {}).get(name, {})}
    try:
        active = declarations["active_parameters"]
        active = json.loads(active) if isinstance(active, str) else active
        if type(active) is not list or any(type(v) is not str for v in active):
            raise ValueError("Active coordinates require a JSON list of canonical names")
        pending_error = None
    except (ValueError, TypeError) as exc:
        active, pending_error = stage["active_parameters"], str(exc)
    fixed = plan.get("fixed_parameters", {})
    parameters = plan["parameters"]
    capabilities = {(p["owner"], p["name"]): p for p in value["definition"]["capabilities"]}
    canonical = {
        tuple(key): i for i, key in enumerate(value["definition"].get("canonical_order", ()))
    }
    rows = []
    for i, p in enumerate(parameters):
        rows.append(
            dict(
                owner=p["owner"],
                name=p["name"],
                unit=p["unit"],
                declared_index=i,
                canonical_index=canonical.get((p["owner"], p["name"], p["unit"])),
                role="declared fixed"
                if p["name"] in fixed
                else "stage active"
                if p["name"] in active
                else "held at upstream stage start",
                declared_fixed_value=fixed.get(p["name"]),
                support=capabilities[(p["owner"], p["name"])]["reason"]
                or "Current owner coordinate; coupled/gauge launch admission unavailable",
            )
        )
    return dict(
        stage_name=name,
        declared_order=index,
        upstream="Declared initial starts"
        if index == 0
        else dict(
            previous_stage=stages[index - 1]["name"],
            policy="Script execution warms from its chosen candidate; no candidate or warm start generated here",
        ),
        description_sha256=value["sha256"],
        definition_sha256=value["definition"]["sha256"],
        engine_revision=value["definition"]["engine_revision"],
        model_id=value["definition"]["model_id"],
        frozen_inputs=value["inputs"],
        upstream_declarations={
            k: plan[k]
            for k in (
                "experiment_binding",
                "source_override",
                "integration_override",
                "repeat_choices",
                "proposal_mosaic",
                "controls",
            )
            if k in plan
        },
        derived_state="Thickness/phase-parent fractions and candidate-dependent inactive directions belong to the material owner. No result-derived values or inactive statuses are inferred from a stage name.",
        parameter_roles=rows,
        declared_active_order=active,
        pending_error=pending_error,
        numerical_declarations=declarations,
        inherited_plan_numerical_declarations={
            k: plan[k]
            for k in (
                "method",
                "maximum_iterations",
                "maximum_function_evaluations",
                "finite_difference_step",
                "numerical_tolerances",
                "numerical_checks",
            )
            if k in plan
        },
        nuisance=dict(
            scale="Existing native_search profiles one nonnegative acquisition scale; historical guarded SLSQP owns its literal scale coordinate. No new nuisance controls.",
            background="Frozen observation background and full covariance; no fitted background parameter or new transfer recipe.",
            calibration=plan.get("calibration", []),
        ),
        fixed_and_final_policy="Global fixed/gauge controls remain read-only. Final active order must equal every declared free coordinate. Earlier stage active lists are structurally checked draft proposals.",
        unsupported_disorder="Native Bi disorder is unavailable"
        if value["definition"]["model_id"].endswith("BiJointModel")
        else "Only coordinates in this actual material definition are available; no additional disorder template",
        diagnostic_status="Native stage-result import unavailable: no independent typed read-only result owner outside refine_native/render_native script execution. Rank, covariance, weak directions, initial/candidate/selected and qualification cannot be fabricated or recomputed. Existing geometric diagnostics remain with their original owners.",
        readiness="Draft declaration only; no full engine admission, execution or current-run outcome",
    )
