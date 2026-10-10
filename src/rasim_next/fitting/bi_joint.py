"""Bounded Bi atomic/cell and morphology fitting on a frozen native observable."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_native import (
    BI_CELL_SITE_PARAMETER_NAMES,
    BiCellSiteParameters,
    BiNativeStructureModel,
)

BI_JOINT_PARAMETER_NAMES = (
    *BI_CELL_SITE_PARAMETER_NAMES,
    "gaussian_sigma_rad",
    "lorentzian_half_width_rad",
    "lorentzian_probability",
    "surface_fraction_0",
    "surface_1_share_of_remainder",
    "extra_film_thickness_A",
    "top_roughness_A",
    "bottom_roughness_A",
)


@dataclass(frozen=True, slots=True)
class BiJointCandidate:
    """21 continuous coordinates and an independently selected integer repeat count.

    Film thickness = N*c + extra thickness. Surface fractions are
    (s0, (1-s0)*s1, (1-s0)*(1-s1)); these coordinates preserve physical bounds.
    """

    values: np.ndarray
    coherent_repeats: int

    def __post_init__(self) -> None:
        raw = np.asarray(self.values)
        if raw.shape != (21,) or np.iscomplexobj(raw) or np.any(~np.isfinite(raw)):
            raise ValueError("Bi joint candidates require 21 finite real coordinates")
        value = np.array(raw, dtype=float, copy=True)
        BiCellSiteParameters.from_array(value[:13])
        MosaicParameters(*value[13:16])
        if (
            type(self.coherent_repeats) is not int
            or self.coherent_repeats < 1
            or np.any(value[16:18] < 0)
            or np.any(value[16:18] > 1)
            or np.any(value[18:] < 0)
        ):
            raise ValueError("invalid repeat count, surface fractions, thickness or roughness")
        value.setflags(write=False)
        object.__setattr__(self, "values", value)

    @property
    def film_thickness_A(self) -> float:
        return float(self.coherent_repeats * self.values[1] + self.values[18])

    @property
    def surface_fractions(self) -> tuple[float, float, float]:
        first, second = self.values[16:18]
        return float(first), float((1 - first) * second), float((1 - first) * (1 - second))


@dataclass(frozen=True, slots=True)
class BiJointModel:
    """Material-specific symmetry and physical coordinates; no optimizer or response."""

    atomic: BiNativeStructureModel
    parameter_names = BI_JOINT_PARAMETER_NAMES
    parameter_units = (
        ("angstrom",) * 2
        + ("1",) * 5
        + ("angstrom^2",) * 6
        + ("radian",) * 2
        + ("1",) * 3
        + ("angstrom",) * 3
    )

    def bind(self, values, coherent_repeats, *, optical_factor_cache=None):
        physical = np.asarray(values)
        candidate = BiJointCandidate(physical, coherent_repeats)
        parameters = BiCellSiteParameters.from_array(physical[:13])
        physics = (
            self.atomic.bind(parameters, optical_factor_cache=optical_factor_cache)
            if type(self.atomic) is BiNativeStructureModel
            else self.atomic.bind(parameters)
        )
        stack = physics.specular_stitch_stack
        if stack is None:
            raise ValueError("Bi joint refinement requires its declared local composite")
        stack = replace(
            stack, top_roughness_A=float(physical[19]), bottom_roughness_A=float(physical[20])
        )
        arguments = dict(
            coherent_repeats=coherent_repeats,
            film_thickness_A=candidate.film_thickness_A,
            surface_fractions=candidate.surface_fractions,
            phase_fractions=(1.0,),
            fault_parameters={},
        )
        mosaic = MosaicParameters(*physical[13:16])
        return physics, arguments, mosaic, stack

    def inactive_parameters(self, values):
        return {}
