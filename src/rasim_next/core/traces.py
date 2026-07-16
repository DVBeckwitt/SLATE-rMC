from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Measure(StrEnum):
    NONE = "NONE"
    PROBABILITY_DENSITY = "PROBABILITY_DENSITY"
    PROBABILITY_MASS = "PROBABILITY_MASS"
    DIFFERENTIAL_PER_SOLID_ANGLE = "DIFFERENTIAL_PER_SOLID_ANGLE"
    INTEGRATED_EVENT_MASS = "INTEGRATED_EVENT_MASS"
    DETECTOR_PIXEL_MASS = "DETECTOR_PIXEL_MASS"


class QuantityKind(StrEnum):
    SCALAR = "SCALAR"
    INDEX = "INDEX"
    POINT = "POINT"
    VECTOR = "VECTOR"
    MATRIX = "MATRIX"
    AMPLITUDE = "AMPLITUDE"
    INTENSITY = "INTENSITY"
    IMAGE = "IMAGE"
    MASK = "MASK"


@dataclass(frozen=True, slots=True)
class TraceRecord:
    case_id: str
    stage_id: str
    value: NDArray[Any]
    unit: str
    frame: str
    measure: Measure
    quantity_kind: QuantityKind
    model_version: str
    provenance: str

    def __post_init__(self) -> None:
        value = np.array(self.value, copy=True, order="C")
        if value.dtype.kind == "O" or any(
            not item
            for item in (
                self.case_id,
                self.stage_id,
                self.unit,
                self.frame,
                self.model_version,
                self.provenance,
            )
        ):
            raise ValueError("trace values and metadata must be explicit")
        value.setflags(write=False)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "measure", Measure(self.measure))
        object.__setattr__(self, "quantity_kind", QuantityKind(self.quantity_kind))
