"""Pure specular reflectivity calculations."""

from rasim_next.reflectivity.parratt import ParrattResult, parratt_reflectivity
from rasim_next.reflectivity.specular import (
    FIXED_EXTERNAL_QZ_INTERFACE,
    LOCAL_LAMELLA_INTERFACE,
    CompiledParrattStitch,
    KinematicScaleSpecularResult,
    ParrattStitchStack,
    SpecularResult,
    compile_parratt_stitch,
    kinematic_scale_specular_stitch,
    manuscript_specular_composite,
    parratt_stitch_interface_assumption,
    parratt_stitch_interface_code,
)

__all__ = [
    "FIXED_EXTERNAL_QZ_INTERFACE",
    "LOCAL_LAMELLA_INTERFACE",
    "CompiledParrattStitch",
    "KinematicScaleSpecularResult",
    "ParrattResult",
    "ParrattStitchStack",
    "SpecularResult",
    "compile_parratt_stitch",
    "kinematic_scale_specular_stitch",
    "manuscript_specular_composite",
    "parratt_reflectivity",
    "parratt_stitch_interface_assumption",
    "parratt_stitch_interface_code",
]
