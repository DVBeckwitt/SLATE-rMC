"""Options for raw structure-factor calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class StructureFactorOptions:
    scattering_table: str = "itc"
    anomalous_mode: str = "xraydb"
    debye_waller_mode: str = "cif"
    occupancy_mode: str = "cif"
    phase_sign: int = 1
    constant_factors: Mapping[str, complex] = field(default_factory=dict)

    @classmethod
    def package_default(cls) -> "StructureFactorOptions":
        return cls(
            scattering_table="itc",
            anomalous_mode="xraydb",
            debye_waller_mode="cif",
            occupancy_mode="cif",
        )

    @classmethod
    def vesta_cu_ka1(cls) -> "StructureFactorOptions":
        return cls(
            scattering_table="waaskirf",
            anomalous_mode="vesta_cu_ka1",
            debye_waller_mode="cif",
            occupancy_mode="cif",
        )
