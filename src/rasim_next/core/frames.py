"""Identifiers for the explicitly declared coordinate frames."""

from enum import StrEnum


class FrameId(StrEnum):
    """Right-handed frames used by the numerical core."""

    LAB = "lab"
    GONIOMETER = "goniometer"
    SAMPLE = "sample"
    CRYSTAL = "crystal"
    DETECTOR = "detector"
    OSC_RAW = "osc_raw"
    NONE = "none"
