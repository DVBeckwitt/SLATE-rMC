"""Bounded immutable hBN drafts, reviewed packs and identity-bound result records."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from uuid import UUID, uuid4

HBN_NAMES = (
    "detector_column_tilt_rad",
    "detector_row_tilt_rad",
    "beam_center_column_px",
    "beam_center_row_px",
    "calibrant_distance_m",
)
HBN_UNITS = ("radian", "radian", "native column px", "native row px", "metre; calibrant-private")


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def payload_hash(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def finite_tuple(values, size):
    return (
        type(values) is tuple
        and len(values) == size
        and all(type(v) in (int, float) and math.isfinite(v) for v in values)
    )


@dataclass(frozen=True, slots=True)
class HbnSession:
    session_id: UUID
    acquisition_id: UUID
    revision: int
    inputs_json: str
    initial: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    f_scale: float = 1.0
    max_nfev: int = 1000
    candidates: tuple[tuple[float, float, int, int], ...] = ()
    exclusions: tuple[tuple[int, str], ...] = ()
    frozen_json: str = ""
    center_proposal_json: str = ""
    results_json: tuple[str, ...] = ()
    selected_result_id: str | None = None
    exports: tuple[tuple[str, str], ...] = ()
    initial_provenance_json: str = ""

    def __post_init__(self):
        if (
            not isinstance(self.session_id, UUID)
            or not isinstance(self.acquisition_id, UUID)
            or type(self.revision) is not int
            or not 0 <= self.revision < 2**63
        ):
            raise ValueError("invalid hBN draft identity/revision")
        if type(self.inputs_json) is not str or not 0 < len(self.inputs_json.encode()) <= 96 * 1024:
            raise ValueError("hBN input snapshot exceeds 96 KiB")
        inputs = json.loads(self.inputs_json)
        if (
            inputs.get("acquisition_id") != str(self.acquisition_id)
            or inputs.get("schema") != "slate.hbn-inputs.v1"
        ):
            raise ValueError("hBN inputs do not match acquisition")
        for key in (
            "source_file_sha256",
            "source_sha256",
            "dark_file_sha256",
            "dark_sha256",
            "configuration_sha256",
            "cif_sha256",
        ):
            value = inputs[key]
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("invalid hBN input hash")
        shape = inputs["shape_rc"]
        if (
            type(shape) is not list
            or len(shape) != 2
            or any(type(v) is not int or not 1 <= v <= 16384 for v in shape)
            or math.prod(shape) > 12_000_000
        ):
            raise ValueError("hBN native shape exceeds desktop limits")
        for key in ("source_path", "dark_path", "configuration_path", "cif_path"):
            if type(inputs[key]) is not str or not 0 < len(inputs[key]) <= 4096:
                raise ValueError("invalid hBN source path")
        if not all(finite_tuple(v, 5) for v in (self.initial, self.lower, self.upper)):
            raise ValueError("hBN seeds/bounds require five finite coordinates")
        admitted_lower = (-0.15, -0.15, 0, 0, 0.04)
        admitted_upper = (0.15, 0.15, shape[1] - 1, shape[0] - 1, 0.12)
        if any(
            lo < a or hi > b or not lo < hi or not lo <= v <= hi
            for v, lo, hi, a, b in zip(
                self.initial, self.lower, self.upper, admitted_lower, admitted_upper, strict=True
            )
        ):
            raise ValueError("hBN seeds/bounds exceed the admitted coordinate domains")
        if (
            type(self.f_scale) not in (int, float)
            or not math.isfinite(self.f_scale)
            or self.f_scale <= 0
            or type(self.max_nfev) is not int
            or not 1 <= self.max_nfev <= 1000
        ):
            raise ValueError("hBN soft_l1 scale/max_nfev are invalid")
        if type(self.candidates) is not tuple or len(self.candidates) > 180:
            raise ValueError("hBN review supports at most 180 canonical sector points")
        for row in self.candidates:
            if (
                type(row) is not tuple
                or len(row) != 4
                or not all(type(v) in (int, float) and math.isfinite(v) for v in row[:2])
                or type(row[2]) is not int
                or not 0 <= row[0] <= shape[1] - 1
                or not 0 <= row[1] <= shape[0] - 1
                or not 0 <= row[2] < 5
                or type(row[3]) is not int
                or not 0 <= row[3] < 36
            ):
                raise ValueError("invalid canonical hBN candidate")
        if len({(p[2], p[3]) for p in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate hBN ring/sector candidate")
        if (
            type(self.exclusions) is not tuple
            or len(self.exclusions) > len(self.candidates)
            or len({i for i, _ in self.exclusions}) != len(self.exclusions)
            or any(
                type(i) is not int
                or not 0 <= i < len(self.candidates)
                or type(reason) is not str
                or not 0 < len(reason) <= 256
                for i, reason in self.exclusions
            )
        ):
            raise ValueError("hBN exclusions need unique candidate IDs and reasons")
        for text, limit in (
            (self.frozen_json, 64 * 1024),
            (self.center_proposal_json, 16 * 1024),
            (self.initial_provenance_json, 16 * 1024),
        ):
            if type(text) is not str or len(text.encode()) > limit:
                raise ValueError("hBN review/proposal exceeds its byte cap")
            if text:
                json.loads(text)
        if self.frozen_json:
            pack = json.loads(self.frozen_json)
            digest = pack.pop("sha256")
            UUID(pack["pack_id"])
            if digest != payload_hash(pack) or pack["inputs_sha256"] != self.inputs_sha256:
                raise ValueError("hBN frozen pack identity mismatch")
            included = [p for i, p in enumerate(self.candidates) if i not in dict(self.exclusions)]
            if (
                pack["schema"] != "slate.hbn-observations.v1"
                or pack["inputs"] != inputs
                or pack["review_revision"] != self.revision
                or pack["coordinates_px"] != [list(p[:2]) for p in included]
                or pack["ring_index"] != [p[2] for p in included]
                or pack["angular_sector"] != [p[3] for p in included]
                or pack["excluded_candidates"] != [list(p) for p in self.exclusions]
                or len(included) < 20
                or len({p[2] for p in included}) < 2
            ):
                raise ValueError("hBN frozen pack differs from admitted reviewed candidates")

        if (
            type(self.results_json) is not tuple
            or len(self.results_json) > 4
            or any(type(v) is not str or len(v.encode()) > 192 * 1024 for v in self.results_json)
        ):
            raise ValueError("hBN results exceed the four-record cap")
        ids = []
        for text in self.results_json:
            record = json.loads(text)
            UUID(record["result_id"])
            if record["schema"] != "slate.hbn-result.v1" or record["acquisition_id"] != str(
                self.acquisition_id
            ):
                raise ValueError("hBN result belongs to another acquisition")
            row = record.copy()
            digest = row.pop("sha256")
            qualification = (
                "qualified by existing hBN owner"
                if record["calibration"]["success"]
                else "unqualified candidate; existing hBN checks did not qualify"
            )
            if (
                digest != payload_hash(row)
                or record["qualification"] != qualification
                or record["parameter_units"] != list(HBN_UNITS)
            ):
                raise ValueError("hBN result content/qualification/units mismatch")
            launch = record["launch"]
            pack = launch["pack"]
            pack_row = pack.copy()
            pack_digest = pack_row.pop("sha256")
            if (
                pack_digest != payload_hash(pack_row)
                or pack["inputs"] != launch["inputs"]
                or pack["inputs_sha256"] != payload_hash(launch["inputs"])
            ):
                raise ValueError("hBN result observation/input identity mismatch")
            ids.append(record["result_id"])
        if len(set(ids)) != len(ids) or (
            self.selected_result_id is not None and self.selected_result_id not in ids
        ):
            raise ValueError("hBN selection does not identify a retained result")
        if (
            type(self.exports) is not tuple
            or len(self.exports) > 16
            or any(
                type(path) is not str
                or not 0 < len(path) <= 4096
                or type(sha) is not str
                or len(sha) != 64
                for path, sha in self.exports
            )
        ):
            raise ValueError("hBN export references exceed their cap")

    @property
    def inputs_sha256(self):
        return hashlib.sha256(self.inputs_json.encode()).hexdigest()

    @property
    def launch_sha256(self):
        return payload_hash(
            {
                "inputs": self.inputs_sha256,
                "initial": self.initial,
                "lower": self.lower,
                "upper": self.upper,
                "f_scale": self.f_scale,
                "max_nfev": self.max_nfev,
                "frozen": self.frozen_json,
            }
        )

    @property
    def nbytes(self):
        return len(encoded(hbn_session_document(self)).encode()) + 8192


def hbn_session_document(session):
    if session is None:
        return None
    return {
        **asdict(session),
        "session_id": str(session.session_id),
        "acquisition_id": str(session.acquisition_id),
    }


def hbn_session_from_document(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) != set(HbnSession.__dataclass_fields__):
        raise ValueError("invalid hBN session document")
    row = value.copy()
    row["session_id"] = UUID(row["session_id"])
    row["acquisition_id"] = UUID(row["acquisition_id"])
    for key in ("initial", "lower", "upper", "results_json"):
        row[key] = tuple(row[key])
    for key in ("candidates", "exclusions", "exports"):
        row[key] = tuple(tuple(v) for v in row[key])
    return HbnSession(**row)


def freeze_hbn_observations(session):
    import numpy as np

    from rasim_next.fitting.hbn import HbnRingObservations, hbn_two_theta_rad

    excluded = dict(session.exclusions)
    points = [p for i, p in enumerate(session.candidates) if i not in excluded]
    if len(points) < 20:
        raise ValueError("hBN observations require at least 20 reviewed points in two rings")
    values = np.asarray(points, dtype=np.float64)
    inputs = json.loads(session.inputs_json)
    angles = hbn_two_theta_rad(
        lattice_a_A=inputs["lattice_a_A"],
        lattice_c_A=inputs["lattice_c_A"],
        wavelength_A=inputs["wavelength_A"],
    )
    observations = HbnRingObservations(
        values[:, :2], values[:, 2].astype(np.int64), angles, values[:, 3].astype(np.int64)
    )
    pack = {
        "schema": "slate.hbn-observations.v1",
        "pack_id": str(uuid4()),
        "inputs_sha256": session.inputs_sha256,
        "review_revision": session.revision,
        "coordinates_px": observations.coordinates_px.tolist(),
        "ring_index": observations.ring_index.tolist(),
        "two_theta_rad": angles.tolist(),
        "angular_sector": observations.angular_sector.tolist(),
        "excluded_candidates": [list(v) for v in session.exclusions],
        "units": "native (column_px,row_px); radian angles",
        "inputs": inputs,
    }
    return encoded({**pack, "sha256": payload_hash(pack)})
