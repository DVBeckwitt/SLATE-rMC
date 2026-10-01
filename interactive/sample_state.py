"""Bounded immutable sample-series drafts, reviewed packs and result history."""

import base64
import hashlib
import json
import zlib
from dataclasses import asdict, dataclass
from uuid import UUID


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_hash(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SampleSession:
    session_id: UUID
    inputs_json: str
    controls_json: str
    revision: int = 0
    prepared_json: str | None = None
    exclusions: tuple[tuple[str, str], ...] = ()
    frozen_json: str | None = None
    results_json: tuple[str, ...] = ()
    selected_result_id: str | None = None
    exports: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.session_id, UUID)
            or type(self.revision) is not int
            or not 0 <= self.revision < 2**63
        ):
            raise ValueError("invalid sample session identity/revision")
        for name, limit in (
            ("inputs_json", 128 * 1024),
            ("controls_json", 16 * 1024),
            ("prepared_json", 1024 * 1024),
            ("frozen_json", 256 * 1024),
        ):
            value = getattr(self, name)
            if value is None and name in ("prepared_json", "frozen_json"):
                continue
            if (
                type(value) is not str
                or len(value.encode()) > limit
                or type(json.loads(value)) is not dict
            ):
                raise ValueError("sample " + name + " exceeds its document admission")
        inputs = json.loads(self.inputs_json)
        if not 2 <= len(inputs["images"]) <= 8 or len(
            {v["image_id"] for v in inputs["images"]}
        ) != len(inputs["images"]):
            raise ValueError("sample-only geometry requires 2-8 unique declared images")
        if (
            len(self.exclusions) > 512
            or len(dict(self.exclusions)) != len(self.exclusions)
            or any(not a or not b.strip() or len(b) > 256 for a, b in self.exclusions)
        ):
            raise ValueError("sample exclusions need unique observation IDs and bounded reasons")
        if self.frozen_json is not None:
            pack = json.loads(self.frozen_json)
            if pack["inputs_sha256"] != payload_hash(inputs) or pack["review"] != [
                list(v) for v in self.exclusions
            ]:
                raise ValueError("sample frozen pack disagrees with current inputs/review")
            if pack["sha256"] != payload_hash({k: v for k, v in pack.items() if k != "sha256"}):
                raise ValueError("sample frozen pack hash mismatch")
        if type(self.results_json) is not tuple or len(self.results_json) > 4:
            raise ValueError("sample result history is capped at four records")
        ids = []
        for text in self.results_json:
            if type(text) is not str or len(text.encode()) > 2 * 1024**2:
                raise ValueError("sample result exceeds 2 MiB")
            record = json.loads(text)
            UUID(record["result_id"])
            if record["sha256"] != payload_hash({k: v for k, v in record.items() if k != "sha256"}):
                raise ValueError("sample result hash mismatch")
            ids.append(record["result_id"])
        if len(set(ids)) != len(ids) or (
            self.selected_result_id is not None and self.selected_result_id not in ids
        ):
            raise ValueError("invalid sample result selection/history")
        if len(self.exports) > 16 or any(
            not p or len(p) > 4096 or len(h) != 64 for p, h in self.exports
        ):
            raise ValueError("sample export reference cap/identity invalid")

    @property
    def launch_sha256(self):
        return payload_hash([self.inputs_json, self.controls_json, self.frozen_json])

    @property
    def nbytes(self):
        return (
            len(encoded(sample_session_document(self)).encode())
            + sum(
                len(v.encode())
                for v in (
                    self.inputs_json,
                    self.controls_json,
                    self.prepared_json or "",
                    self.frozen_json or "",
                    *self.results_json,
                )
            )
            + 8192
        )


def sample_session_document(session):
    if session is None:
        return None
    result = asdict(session)
    result["session_id"] = str(session.session_id)
    for name in ("prepared_json", "frozen_json"):
        result[name] = _pack_text(result[name])
    result["results_json"] = [_pack_text(v) for v in session.results_json]
    return result


def sample_session_from_document(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) != set(SampleSession.__dataclass_fields__):
        raise ValueError("invalid sample session document fields")
    data = dict(value)
    data["session_id"] = UUID(data["session_id"])
    for name in ("prepared_json", "frozen_json"):
        data[name] = _unpack_text(data[name])
    data["results_json"] = [_unpack_text(v) for v in data["results_json"]]
    for name in ("exclusions", "exports"):
        data[name] = tuple(tuple(v) for v in data[name])
    data["results_json"] = tuple(data["results_json"])
    return SampleSession(**data)


def _pack_text(value):
    if value is None:
        return None
    return "zlib-base64:" + base64.b64encode(zlib.compress(value.encode(), level=6)).decode("ascii")


def _unpack_text(value, *, limit=2 * 1024**2):
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("sample packed text must be a string")
    if not value.startswith("zlib-base64:"):
        return value
    if len(value) > 3 * limit // 2:
        raise ValueError("sample compressed text exceeds its bounded admission")
    raw = base64.b64decode(value.removeprefix("zlib-base64:"), validate=True)
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(raw, limit + 1)
    if len(decoded) > limit or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("sample compressed text exceeds its exact bounded expansion")
    return decoded.decode("utf-8")
