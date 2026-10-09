"""Bounded immutable captures, controls and historical joint geometry results."""

import json
from dataclasses import asdict, dataclass
from uuid import UUID

from hbn_state import hbn_session_from_document
from sample_state import (
    _pack_text,
    _unpack_text,
    encoded,
    payload_hash,
    sample_session_from_document,
)


def joint_controls(initial, lower, upper):
    from rasim_next.fitting.joint_geometry import (
        JOINT_GEOMETRY_PARAMETER_NAMES,
        LOCAL_PARAMETER_NAMES,
        JointGeometryBounds,
        JointGeometryState,
        validate_joint_geometry_start,
    )

    unobserved = tuple(n for n in LOCAL_PARAMETER_NAMES if n.startswith("pbi2_"))
    state = JointGeometryState.from_array(initial)
    bounds = JointGeometryBounds(
        JointGeometryState.from_array(lower), JointGeometryState.from_array(upper)
    )
    validate_joint_geometry_start(state, bounds, unobserved_parameters=unobserved)
    return {
        "names": list(JOINT_GEOMETRY_PARAMETER_NAMES),
        "initial": list(initial),
        "lower": list(lower),
        "upper": list(upper),
    }


def validate_captures(captures):
    if not set(captures) <= {"hbn", "bi2se3", "bi2te3"}:
        raise ValueError("only reviewed hBN, Bi2Se3 and Bi2Te3 captures are supported")
    for group, capture in captures.items():
        if capture["sha256"] != payload_hash({k: v for k, v in capture.items() if k != "sha256"}):
            raise ValueError("joint capture identity mismatch")
        session = (hbn_session_from_document if group == "hbn" else sample_session_from_document)(
            capture["session"]
        )
        if not session.frozen_json:
            raise ValueError("joint captures require reviewed frozen observations")
        if group != "hbn":
            if json.loads(session.inputs_json)["phase_id"].lower() != group:
                raise ValueError("joint specimen capture does not match its declared material")
        else:
            seed = capture["calibration_record"]
            if seed["sha256"] != payload_hash(
                {k: v for k, v in seed.items() if k != "sha256"}
            ) or seed["acquisition_id"] != str(session.acquisition_id):
                raise ValueError("joint hBN seed record identity mismatch")
            from rasim_next.fitting.hbn import HbnDetectorCalibration, hbn_calibration_is_qualified

            calibration = HbnDetectorCalibration(**seed["calibration"])
            if (
                calibration.success
                != hbn_calibration_is_qualified(
                    **{
                        n: (
                            float("inf")
                            if n == "scaled_jacobian_condition" and getattr(calibration, n) is None
                            else getattr(calibration, n)
                        )
                        for n in (
                            "solver_success",
                            "jacobian_rank",
                            "scaled_jacobian_condition",
                            "active_bounds",
                            "ring_point_count",
                            "ring_angular_coverage_fraction",
                            "ring_rms_px",
                        )
                    }
                )
                or calibration.values.tolist() != seed["fitted_values"]
            ):
                raise ValueError(
                    "joint hBN seed contradicts canonical recorded qualification/values"
                )
    return captures


def validate_joint_record(record):
    from rasim_next.fitting.joint_geometry_report import validate_joint_geometry_report

    UUID(record["result_id"])
    if record["schema"] != "slate.joint-result.v1" or record["sha256"] != payload_hash(
        {k: v for k, v in record.items() if k != "sha256"}
    ):
        raise ValueError("joint result identity/hash mismatch")
    if (
        not isinstance(record["name"], str)
        or not record["name"].strip()
        or len(record["name"]) > 256
    ):
        raise ValueError("joint result needs a bounded name")
    validate_joint_geometry_report(record["report"])
    launch = record["launch"]
    if launch is not None:
        validate_captures(launch["captures"])
        if set(launch["captures"]) != {"hbn", "bi2se3", "bi2te3"}:
            raise ValueError("joint launch requires three complete captures")
        controls = joint_controls(
            **{k: launch["controls"][k] for k in ("initial", "lower", "upper")}
        )
        if controls != launch["controls"] or record["launch_sha256"] != payload_hash(launch):
            raise ValueError("joint result controls/launch changed")
        roster = []
        for group in ("bi2se3", "bi2te3"):
            ss = sample_session_from_document(launch["captures"][group]["session"])
            for obs in json.loads(ss.frozen_json)["observations"]:
                roster.append((group, obs["image_id"], len(obs["keys"])))
        if roster != [
            (v["specimen_id"], v["image_id"], v["site_count"])
            for v in record["report"]["crystalline_metrics"]["per_image"]
        ]:
            raise ValueError("joint result image roster differs from frozen launch")
        params = {
            **record["report"]["global"],
            **record["report"]["local"],
            **record["report"]["nuisance"],
        }
        for n, lo, hi in zip(controls["names"], controls["lower"], controls["upper"], strict=True):
            if not lo <= params[n]["value"] <= hi:
                raise ValueError("joint fitted value violates its saved bounds")
        import math

        for n, lo, hi in zip(controls["names"], controls["lower"], controls["upper"], strict=True):
            row = params[n]
            if row["role"] == "fitted":
                span = (hi - lo) / 2
                contact = row["value"] - lo <= 1e-6 * span or hi - row["value"] <= 1e-6 * span
                expected_confident = (
                    not contact
                    and row["standard_error"] is not None
                    and math.isfinite(row["standard_error"])
                    and row["standard_error"] < 0.5 * span
                )
                if (
                    row["active_bound"] != contact
                    or row["confidence_qualified"] != expected_confident
                ):
                    raise ValueError("joint recorded uncertainty/bounds contradict launch")
        expected_points = []
        for group in ("bi2se3", "bi2te3"):
            ss = sample_session_from_document(launch["captures"][group]["session"])
            for obs in json.loads(ss.frozen_json)["observations"]:
                for key, xy, identity in zip(
                    obs["keys"], obs["coordinates_px"], obs["observation_ids"], strict=True
                ):
                    expected_points.append((group, obs["image_id"], identity, key, xy))
        actual_points = [
            (p["specimen_id"], p["image_id"], p["observation_id"], p["key"], p["observed_px"])
            for p in record["points"]
        ]
        if expected_points != actual_points:
            raise ValueError("joint result observation identities/coordinates changed")
        import numpy as np

        for p in record["points"]:
            predicted = np.asarray(p["predicted_px"], dtype=np.float64)
            if (
                predicted.shape != (2,)
                or not np.all(np.isfinite(predicted))
                or not np.array_equal(predicted - np.asarray(p["observed_px"]), p["residual_px"])
            ):
                raise ValueError("joint saved site residual contradicts its coordinates")
        for metric in record["report"]["crystalline_metrics"]["per_image"]:
            errors = [
                np.linalg.norm(p["residual_px"])
                for p in record["points"]
                if (p["specimen_id"], p["image_id"]) == (metric["specimen_id"], metric["image_id"])
            ]
            if not math.isclose(
                math.sqrt(sum(v * v for v in errors) / len(errors)),
                metric["site_rms_px"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(max(errors), metric["site_max_px"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("joint saved sites contradict recorded image diagnostics")
        hbn = hbn_session_from_document(launch["captures"]["hbn"]["session"])
        residual = np.asarray(record["hbn_residual_px"], dtype=np.float64)
        if (
            residual.shape != (len(json.loads(hbn.frozen_json)["coordinates_px"]),)
            or not np.all(np.isfinite(residual))
            or not math.isclose(
                float(np.sqrt(np.mean(residual**2))),
                record["report"]["hbn_automatic_trace"]["joint_rms_px"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("joint frozen hBN residual diagnostics disagree")
    elif record["launch_sha256"] is not None or record["points"]:
        raise ValueError(
            "external historical report cannot claim a frozen launch or invented points"
        )
    return record


@dataclass(frozen=True, slots=True)
class JointSession:
    session_id: UUID
    captures_json: str = "{}"
    controls_json: str | None = None
    revision: int = 0
    draft_json: str | None = None
    results_json: tuple[str, ...] = ()
    selected_result_id: str | None = None
    exports: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.session_id, UUID)
            or type(self.revision) is not int
            or not 0 <= self.revision < 2**63
        ):
            raise ValueError("invalid joint session identity/revision")
        if len(self.captures_json.encode()) > 4 * 1024**2:
            raise ValueError("joint captures exceed 4 MiB")
        validate_captures(json.loads(self.captures_json))
        if self.controls_json is not None:
            if len(self.controls_json.encode()) > 16384:
                raise ValueError("joint controls exceed 16 KiB")
            c = json.loads(self.controls_json)
            if c != joint_controls(**{k: c[k] for k in ("initial", "lower", "upper")}):
                raise ValueError("joint controls are not canonical")
        if self.draft_json is not None and (
            len(self.draft_json.encode()) > 65536 or type(json.loads(self.draft_json)) is not list
        ):
            raise ValueError("joint visible draft exceeds its bounded admission")
        if len(self.results_json) > 4:
            raise ValueError("joint history is capped at four records")
        ids = []
        for text in self.results_json:
            if len(text.encode()) > 6 * 1024**2:
                raise ValueError("joint result exceeds 6 MiB")
            ids.append(validate_joint_record(json.loads(text))["result_id"])
        if len(set(ids)) != len(ids) or (
            self.selected_result_id is not None and self.selected_result_id not in ids
        ):
            raise ValueError("invalid joint result selection/history")
        if self.selected_result_id is not None:
            selected = next(
                json.loads(v)
                for v in self.results_json
                if json.loads(v)["result_id"] == self.selected_result_id
            )
            if (
                self.draft_json is not None
                or selected["launch"] is None
                or selected["launch_sha256"] != self.launch_sha256
            ):
                raise ValueError(
                    "joint selection must belong to the current committed frozen launch"
                )
        if len(self.exports) > 16 or any(
            not p or len(p) > 4096 or len(h) != 64 for p, h in self.exports
        ):
            raise ValueError("joint export reference cap/identity invalid")

    @property
    def launch(self):
        return {
            "captures": json.loads(self.captures_json),
            "controls": None if self.controls_json is None else json.loads(self.controls_json),
        }

    @property
    def launch_sha256(self):
        return payload_hash(self.launch)

    @property
    def nbytes(self):
        return (
            len(self.captures_json.encode())
            + len((self.controls_json or "").encode())
            + len((self.draft_json or "").encode())
            + sum(len(v.encode()) for v in self.results_json)
            + len(encoded(joint_session_document(self)).encode())
            + 8192
        )


def joint_session_document(session):
    if session is None:
        return None
    data = asdict(session)
    data["session_id"] = str(session.session_id)
    data["captures_json"] = _pack_text(data["captures_json"])
    data["results_json"] = [_pack_text(v) for v in data["results_json"]]
    return data


def joint_session_from_document(data):
    if data is None:
        return None
    if type(data) is not dict or set(data) != set(JointSession.__dataclass_fields__):
        raise ValueError("invalid joint session document fields")
    values = dict(data)
    values["session_id"] = UUID(values["session_id"])
    values["captures_json"] = _unpack_text(values["captures_json"], limit=4 * 1024**2)
    values["results_json"] = tuple(
        _unpack_text(v, limit=6 * 1024**2) for v in values["results_json"]
    )
    values["exports"] = tuple(tuple(v) for v in values["exports"])
    return JointSession(**values)
