import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from rasim_next.proof.traces import Tolerance

STAGE_TOLERANCE_VERSION = "rasim-stage-tolerances-v1"
STAGE_TOLERANCE_SHA256 = "e3b88394a27208e97ac62066354affaf5ca0844a9902961aaf624a008599e0d2"
_TOP_KEYS = {
    "artifact_version",
    "contract_api_version",
    "criterion",
    "entries",
    "trace_schema_version",
}
_ENTRY_KEYS = {"stage_ids", "unit", "atol", "rtol", "scale_rule", "rationale"}


@dataclass(frozen=True, slots=True)
class StageTolerance:
    unit: str
    atol: float
    rtol: float
    scale_rule: str
    rationale: str

    def bind(self, scale: float) -> Tolerance:
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("tolerance scale must be finite and nonnegative")
        return Tolerance(self.atol, self.rtol, scale)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate tolerance key")
    return result


def load_stage_tolerances() -> MappingProxyType[str, StageTolerance]:
    data = json.loads(
        Path(__file__).with_name("stage_tolerances_v1.json").read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(data, dict) or set(data) != _TOP_KEYS:
        raise ValueError("invalid tolerance artifact fields")
    metadata = (
        data["artifact_version"],
        data["contract_api_version"],
        data["trace_schema_version"],
    )
    if (
        metadata != (STAGE_TOLERANCE_VERSION, 5, 4)
        or data["criterion"] != "abs(candidate-reference) <= atol + rtol*scale"
        or not isinstance(data["entries"], list)
    ):
        raise ValueError("invalid tolerance artifact version or criterion")
    stages: dict[str, StageTolerance] = {}
    for entry in data["entries"]:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise ValueError("invalid stage-tolerance fields")
        stage_ids = entry["stage_ids"]
        numbers = (entry["atol"], entry["rtol"])
        valid_stages = (
            isinstance(stage_ids, list)
            and bool(stage_ids)
            and all(isinstance(item, str) and item for item in stage_ids)
        )
        valid_text = all(
            isinstance(entry[key], str) and entry[key]
            for key in ("unit", "scale_rule", "rationale")
        )
        valid_numbers = all(type(item) in (int, float) and math.isfinite(item) for item in numbers)
        if (
            not valid_stages
            or not valid_text
            or not valid_numbers
            or entry["atol"] <= 0.0
            or entry["rtol"] < 0.0
        ):
            raise ValueError("invalid stage-tolerance value")
        tolerance = StageTolerance(entry["unit"], *numbers, entry["scale_rule"], entry["rationale"])
        for stage_id in stage_ids:
            if stage_id in stages:
                raise ValueError(f"duplicate tolerance stage: {stage_id}")
            stages[stage_id] = tolerance
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != STAGE_TOLERANCE_SHA256:
        raise ValueError("stage-tolerance artifact hash mismatch")
    return MappingProxyType(stages)
