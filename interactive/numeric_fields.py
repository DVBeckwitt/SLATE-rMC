"""Authoritative units, paths and domains for supported numeric initial values."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParameterDescription:
    field: str
    label: str
    path: tuple[str | int, ...]
    stored_unit: str
    display_unit: str
    display_to_stored: float
    frame: str
    scope: str
    domain: str
    editable: bool = True
    reason: str = ""


PARAMETERS = (
    ParameterDescription(
        "source.origin_x",
        "Beam origin X",
        ("source", "mean_origin_lab_m", 0),
        "m",
        "mm",
        0.001,
        "LAB",
        "shared source",
        "finite",
    ),
    ParameterDescription(
        "source.origin_y",
        "Beam origin Y",
        ("source", "mean_origin_lab_m", 1),
        "m",
        "mm",
        0.001,
        "LAB",
        "shared source",
        "finite",
    ),
    ParameterDescription(
        "source.origin_z",
        "Beam origin Z",
        ("source", "mean_origin_lab_m", 2),
        "m",
        "mm",
        0.001,
        "LAB",
        "shared source",
        "finite",
    ),
    ParameterDescription(
        "source.spatial_sigma_x",
        "Beam width X",
        ("source", "spatial_sigma_m", 0),
        "m",
        "µm",
        1e-6,
        "source transverse",
        "shared source",
        "nonnegative",
    ),
    ParameterDescription(
        "source.spatial_sigma_y",
        "Beam width Y",
        ("source", "spatial_sigma_m", 1),
        "m",
        "µm",
        1e-6,
        "source transverse",
        "shared source",
        "nonnegative",
    ),
    ParameterDescription(
        "source.divergence_x",
        "Divergence X",
        ("source", "divergence_sigma_rad", 0),
        "rad",
        "mrad",
        0.001,
        "source transverse",
        "shared source",
        "nonnegative",
    ),
    ParameterDescription(
        "source.divergence_y",
        "Divergence Y",
        ("source", "divergence_sigma_rad", 1),
        "rad",
        "mrad",
        0.001,
        "source transverse",
        "shared source",
        "nonnegative",
    ),
    ParameterDescription(
        "source.wavelength",
        "Mean wavelength",
        ("source", "mean_wavelength_A"),
        "Å",
        "Å",
        1.0,
        "source",
        "shared source",
        "positive",
    ),
    ParameterDescription(
        "detector.x",
        "Detector X",
        ("instrument", "lab_from_detector", "translation_m", 0),
        "m",
        "mm",
        0.001,
        "LAB",
        "shared instrument",
        "finite",
    ),
    ParameterDescription(
        "detector.y",
        "Detector Y",
        ("instrument", "lab_from_detector", "translation_m", 1),
        "m",
        "mm",
        0.001,
        "LAB",
        "shared instrument",
        "finite",
    ),
    ParameterDescription(
        "detector.z",
        "Detector Z",
        ("instrument", "lab_from_detector", "translation_m", 2),
        "m",
        "mm",
        0.001,
        "LAB",
        "shared instrument",
        "finite",
    ),
    ParameterDescription(
        "detector.reference_column",
        "Detector reference column",
        ("instrument", "detector_reference_coordinate_px", 0),
        "px",
        "px",
        1.0,
        "detector native",
        "shared instrument",
        "finite",
    ),
    ParameterDescription(
        "detector.reference_row",
        "Detector reference row",
        ("instrument", "detector_reference_coordinate_px", 1),
        "px",
        "px",
        1.0,
        "detector native",
        "shared instrument",
        "finite",
    ),
    ParameterDescription(
        "detector.column_tilt",
        "Detector column tilt",
        ("instrument", "detector_tilt", "about_column_axis_deg"),
        "deg",
        "deg",
        1.0,
        "detector intrinsic column axis",
        "shared instrument",
        "finite",
    ),
    ParameterDescription(
        "detector.row_tilt",
        "Detector row tilt",
        ("instrument", "detector_tilt", "about_row_axis_deg"),
        "deg",
        "deg",
        1.0,
        "detector intrinsic row axis",
        "shared instrument",
        "finite",
    ),
    ParameterDescription(
        "goniometer.angle_0",
        "First axis angle",
        ("instrument", "axis_rotations", 0, "angle_deg"),
        "deg",
        "deg",
        1.0,
        "LAB active rotation",
        "acquisition geometry",
        "finite",
    ),
    ParameterDescription(
        "source.direction",
        "Beam direction",
        ("source", "mean_direction_lab"),
        "unit vector",
        "unit vector",
        1.0,
        "LAB",
        "shared source",
        "unit vector and transverse basis must change together",
        False,
        "Edit with the coupled direction/basis form in U12a.",
    ),
    ParameterDescription(
        "detector.rotation",
        "Detector rotation",
        ("instrument", "lab_from_detector", "rotation"),
        "matrix",
        "matrix",
        1.0,
        "LAB from detector",
        "shared instrument",
        "proper active rotation",
        False,
        "Use the constrained geometry editor in U09.",
    ),
    ParameterDescription(
        "detector.shape",
        "Detector shape",
        ("instrument", "detector_shape_rc"),
        "native px",
        "native px",
        1.0,
        "detector native",
        "fixed input",
        "match admitted OSC",
        False,
        "Shape is fixed by the admitted detector and source.",
    ),
)


def description(field: str) -> ParameterDescription:
    for item in PARAMETERS:
        if item.field == field:
            return item
    raise ValueError(f"unsupported numeric field {field}")


def validate_proposal(field: str, value: float, unit: str) -> None:
    item = description(field)
    if not item.editable:
        raise ValueError(item.reason)
    if unit != item.stored_unit:
        raise ValueError(f"{item.label} must use stored unit {item.stored_unit}")
    if not math.isfinite(value):
        raise ValueError(f"{item.label} must be finite")
    if (item.domain == "positive" and value <= 0) or (item.domain == "nonnegative" and value < 0):
        raise ValueError(f"{item.label} must be {item.domain}")
