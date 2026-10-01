"""Bounded display names and inspection identities alongside immutable route owners."""

import json
from dataclasses import dataclass, replace
from uuid import UUID

from native_simulation_state import (
    NativeSimulationReference,
    native_reference_document,
    native_reference_from_document,
)
from sample_state import encoded, payload_hash
from simulation_state import (
    SimulationReference,
    simulation_reference_document,
    simulation_reference_from_document,
)

ROUTES = ("configured", "native", "prepared", "hbn", "sample", "joint")
MAX_ATTEMPT_BYTES = 128 * 1024
MAX_SAVED_RESULTS = 8


def _identity(value):
    if type(value) is not str:
        raise ValueError("Attempt identity must be UUID or SHA256 text")
    if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return
    if str(UUID(value)) != value:
        raise ValueError("Attempt UUID must be canonical")


def _key(value):
    if type(value) not in (list, tuple) or len(value) != 3 or value[0] not in ROUTES:
        raise ValueError("Attempt key requires route, owner and immutable identity")
    _identity(value[1])
    _identity(value[2])
    return tuple(value)


@dataclass(frozen=True, slots=True)
class AttemptHistory:
    names: tuple[tuple[str, str, str, str], ...] = ()
    inspection: tuple[str, str, str] | None = None
    configured: tuple[SimulationReference, ...] = ()
    native: tuple[NativeSimulationReference, ...] = ()
    inherited_from: UUID | None = None


def history_text(history):
    value = {
        "schema": "slate.attempts.v1",
        "names": [list(v) for v in history.names],
        "inspection": history.inspection,
        "configured": [simulation_reference_document(v) for v in history.configured],
        "native": [native_reference_document(v) for v in history.native],
        "inherited_from": None if history.inherited_from is None else str(history.inherited_from),
    }
    text = encoded(value)
    if len(text.encode()) > MAX_ATTEMPT_BYTES:
        raise ValueError("Attempt metadata exceeds 128 KiB")
    return text


def read_history(text):
    if type(text) is not str or len(text.encode()) > MAX_ATTEMPT_BYTES:
        raise ValueError("Attempt metadata exceeds 128 KiB")
    value = json.loads(text)
    if value == {}:
        return AttemptHistory()
    if (
        type(value) is not dict
        or set(value) != {"schema", "names", "inspection", "configured", "native", "inherited_from"}
        or value["schema"] != "slate.attempts.v1"
    ):
        raise ValueError("Unsupported attempt metadata")
    names = value["names"]
    if type(names) is not list or len(names) > 128:
        raise ValueError("At most 128 attempt display names are retained")
    for row in names:
        if type(row) is not list or len(row) != 4:
            raise ValueError("Malformed attempt display name")
        _key(row[:3])
        if type(row[3]) is not str or not row[3].strip() or len(row[3]) > 256:
            raise ValueError("Attempt display name needs 1-256 characters")
    if len({tuple(row[:3]) for row in names}) != len(names):
        raise ValueError("Duplicate attempt display names")
    collections = []
    for route, parser in (
        ("configured", simulation_reference_from_document),
        ("native", native_reference_from_document),
    ):
        rows = value[route]
        if type(rows) is not list or len(rows) > MAX_SAVED_RESULTS:
            raise ValueError("At most eight retained outputs per simulation route")
        refs = tuple(parser(row) for row in rows)
        if any(v is None for v in refs) or len({v.sha256 for v in refs}) != len(refs):
            raise ValueError("Retained outputs require unique exact references")
        collections.append(refs)
    inherited = value["inherited_from"]
    if inherited is not None and str(UUID(inherited)) != inherited:
        raise ValueError("Inherited project UUID must be canonical")
    return AttemptHistory(
        tuple(tuple(row) for row in names),
        None if value["inspection"] is None else _key(value["inspection"]),
        *collections,
        None if inherited is None else UUID(inherited),
    )


def retain_results(text, configured, native):
    history = read_history(text)
    for route, current in (("configured", configured), ("native", native)):
        if current is None:
            continue
        refs = getattr(history, route)
        previous = next((v for v in refs if v.sha256 == current.sha256), None)
        if previous is not None and replace(previous, path=current.path) != current:
            raise ValueError("Retained result identity conflicts with its scientific reference")
        refs = tuple(current if v.sha256 == current.sha256 else v for v in refs)
        if previous is None:
            if len(refs) >= MAX_SAVED_RESULTS:
                raise ValueError(
                    "Retained output limit reached; remove an old reference explicitly"
                )
            refs += (current,)
        history = replace(history, **{route: refs})
    return history_text(history)


@dataclass(frozen=True, slots=True)
class AttemptRow:
    key: tuple[str, str, str]
    name: str
    state: str
    detail: str


def draft_attempt_identity(document):
    """Keep top-level storage relocation outside the scientific attempt identity."""
    return payload_hash(
        {
            key: value
            for key, value in document.items()
            if key not in {"configuration_path", "cif_path", "physics_path"}
        }
    )


def attempt_rows(view):
    """Inventory committed records without opening files, evaluating physics or changing owners."""
    from native_simulation_state import native_draft_document
    from simulation_state import simulation_draft_document

    history = read_history(
        retain_results(view.attempts_json, view.simulation_result, view.native_simulation_result)
    )
    aliases = {v[:3]: v[3] for v in history.names}
    rows = []

    def add(route, owner, identity, name, state, value):
        key = (route, str(owner), identity)
        inherited = " · inherited inspection" if history.inherited_from else ""
        rows.append(
            AttemptRow(
                key,
                aliases.get(key, name),
                state + inherited,
                json.dumps(value, indent=2, ensure_ascii=False),
            )
        )

    for route, draft, serializer in (
        ("configured", view.simulation_draft, simulation_draft_document),
        ("native", view.native_simulation_draft, native_draft_document),
    ):
        if draft is not None:
            document = serializer(draft)
            add(
                route,
                draft.draft_id,
                draft_attempt_identity(document),
                "Committed simulation draft",
                "committed; nominal forward model",
                document,
            )
        current = view.simulation_result if route == "configured" else view.native_simulation_result
        for ref in getattr(history, route):
            document = (
                simulation_reference_document
                if route == "configured"
                else native_reference_document
            )(ref)
            add(
                route,
                ref.draft_id,
                ref.sha256,
                "Retained simulation output",
                "selected output"
                if current and current.sha256 == ref.sha256
                else "retained output",
                document,
            )
    session = view.native_fit_session
    if session is not None:
        for text in (session.current_json, *session.history_json):
            document = json.loads(text)
            current = text == session.current_json
            add(
                "prepared",
                document["definition"]["sha256"],
                document["sha256"],
                "Prepared draft",
                "committed / initial starts; Run unavailable"
                if current
                else "historical definitions / initial starts; Run unavailable",
                {
                    "description": document,
                    "pending_text": session.draft_json if current else None,
                    "session_id": str(session.session_id),
                },
            )
    for route, sessions in (
        ("hbn", view.hbn_sessions),
        ("sample", (view.sample_session,)),
        ("joint", (view.joint_session,)),
    ):
        for session in sessions:
            if session is None:
                continue
            # Current session documents preserve pending decisions, initial values and exact inputs.
            if route == "hbn":
                from hbn_state import hbn_session_document

                current = hbn_session_document(session)
            elif route == "sample":
                from sample_state import sample_session_document

                current = sample_session_document(session)
            else:
                from joint_state import joint_session_document

                current = joint_session_document(session)
            add(
                route,
                session.session_id,
                payload_hash(current),
                "Current owner draft",
                "committed / pending / initial state; inspect exact fields",
                current,
            )
            for text in session.results_json:
                record = json.loads(text)
                add(
                    route,
                    session.session_id,
                    record["result_id"],
                    record.get("name", "Result"),
                    "selected result"
                    if session.selected_result_id == record["result_id"]
                    else "candidate result",
                    record,
                )
    return tuple(rows)


def selected_history(text, routes):
    history = read_history(text)
    return history_text(
        replace(
            history,
            names=tuple(v for v in history.names if v[0] in routes),
            inspection=history.inspection
            if history.inspection and history.inspection[0] in routes
            else None,
            configured=history.configured if "configured" in routes else (),
            native=history.native if "native" in routes else (),
        )
    )
