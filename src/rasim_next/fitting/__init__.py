"""Detector-native geometric fitting interfaces."""

from rasim_next.fitting.geometry import (
    GeometryCorrectionBounds,
    GeometryCorrections,
    GeometryFitResult,
    GeometryPredictionError,
    GeometryRankError,
    IntegerLGeometryModel,
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
    IntegerLMarkerPrediction,
    IntegerLSelectionAudit,
    audit_integer_l_marker_selection,
    fit_integer_l_marker_geometry,
)

__all__ = [
    "GeometryCorrectionBounds",
    "GeometryCorrections",
    "GeometryFitResult",
    "GeometryPredictionError",
    "GeometryRankError",
    "IntegerLGeometryModel",
    "IntegerLMarkerKey",
    "IntegerLMarkerObservations",
    "IntegerLMarkerPrediction",
    "IntegerLSelectionAudit",
    "audit_integer_l_marker_selection",
    "fit_integer_l_marker_geometry",
]
