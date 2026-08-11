"""Post-discovery admission of exact rational-layer PbI2 landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rasim_next.core.layer_order import CommensurateLayerOrder
from rasim_next.fitting.geometry import ExactTagGeometryModel, LayerLMarkerObservations
from rasim_next.fitting.pbi2_geometry import (
    PBI2_IDEAL_PARENTS,
    Pbi2PolytypeLandmarkCatalogue,
)
from rasim_next.geometry import AngleFrame
from rasim_next.selection.blind import (
    MeasuredPeakDiscovery,
    _canonicalized_peak_angles,
    _discovery_geometry_hash,
)


@dataclass(frozen=True, slots=True)
class Pbi2LayerLPeakAdmission:
    """Auditable join from one blind discovery to exact PbI2 landmarks."""

    discovery: MeasuredPeakDiscovery
    model: ExactTagGeometryModel
    catalogue: Pbi2PolytypeLandmarkCatalogue
    angle_frame: AngleFrame
    allowed_denominators: tuple[int, ...]
    catalogue_indices: tuple[int, ...]
    discovery_peak_indices: tuple[int, ...]
    observations: LayerLMarkerObservations = field(init=False)

    def __post_init__(self) -> None:
        denominators = _validated_denominators(self.allowed_denominators)
        catalogue_indices = tuple(self.catalogue_indices)
        peak_indices = tuple(self.discovery_peak_indices)
        if (
            not catalogue_indices
            or len(catalogue_indices) != len(peak_indices)
            or any(
                isinstance(index, bool) or not isinstance(index, (int, np.integer))
                for index in (*catalogue_indices, *peak_indices)
            )
        ):
            raise ValueError("admission indices must align with every observation")
        catalogue_indices = tuple(int(index) for index in catalogue_indices)
        peak_indices = tuple(int(index) for index in peak_indices)
        if catalogue_indices != tuple(sorted(set(catalogue_indices))) or len(
            set(peak_indices)
        ) != len(peak_indices):
            raise ValueError("admission indices must be unique and catalogue ordered")
        if any(
            index < 0 or index >= len(self.catalogue.definitions) for index in catalogue_indices
        ) or any(index < 0 or index >= len(self.discovery.peaks) for index in peak_indices):
            raise ValueError("admission indices must be in range")
        qualified = _qualified_admission_indices(
            self.discovery,
            model=self.model,
            catalogue=self.catalogue,
            angle_frame=self.angle_frame,
            allowed_denominators=denominators,
        )
        if qualified != (catalogue_indices, peak_indices):
            raise ValueError("admission indices do not satisfy the declared qualification gates")
        expected_definitions = tuple(
            self.catalogue.definitions[index] for index in catalogue_indices
        )
        expected_coordinates = np.asarray(
            [
                (
                    self.discovery.peaks[index].column_px,
                    self.discovery.peaks[index].row_px,
                )
                for index in peak_indices
            ],
            dtype=np.float64,
        )
        expected_covariance = np.asarray(
            [self.discovery.peaks[index].covariance_px2 for index in peak_indices],
            dtype=np.float64,
        )
        object.__setattr__(self, "allowed_denominators", denominators)
        object.__setattr__(self, "catalogue_indices", catalogue_indices)
        object.__setattr__(self, "discovery_peak_indices", peak_indices)
        object.__setattr__(
            self,
            "observations",
            LayerLMarkerObservations(
                definitions=expected_definitions,
                coordinates_px=expected_coordinates,
                covariance_px2=expected_covariance,
                reference_wavelength_A=self.model.reference_wavelength_A,
            ),
        )


def _validated_denominators(values: tuple[int, ...]) -> tuple[int, ...]:
    denominators = tuple(values)
    if (
        not denominators
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in denominators
        )
        or denominators != tuple(sorted(set(int(value) for value in denominators)))
        or any(value not in {2, 3} for value in denominators)
    ):
        raise ValueError("allowed_denominators must be a sorted unique subset of (2, 3)")
    return tuple(int(value) for value in denominators)


def _qualified_admission_indices(
    discovery: MeasuredPeakDiscovery,
    *,
    model: ExactTagGeometryModel,
    catalogue: Pbi2PolytypeLandmarkCatalogue,
    angle_frame: AngleFrame,
    allowed_denominators: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if not isinstance(discovery, MeasuredPeakDiscovery):
        raise TypeError("discovery must be a MeasuredPeakDiscovery")
    if not isinstance(model, ExactTagGeometryModel):
        raise TypeError("model must be an ExactTagGeometryModel")
    if not isinstance(catalogue, Pbi2PolytypeLandmarkCatalogue):
        raise TypeError("catalogue must be a Pbi2PolytypeLandmarkCatalogue")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    denominators = _validated_denominators(allowed_denominators)
    if model.inputs.config.material.phase_id.casefold() != "pbi2":
        raise ValueError("rational-layer admission requires a PbI2 geometry model")
    if set(catalogue.selected_parents) != set(PBI2_IDEAL_PARENTS):
        raise ValueError("blind rational-layer admission requires the complete parent catalogue")
    if catalogue.source_cif_sha256 != model.inputs.config.cif_sha256:
        raise ValueError("catalogue and geometry model CIF revisions disagree")
    if catalogue.reciprocal_basis_revision != model.reciprocal_basis_revision:
        raise ValueError("catalogue and geometry model reciprocal bases disagree")
    if catalogue.geometry_context_revision != model.geometry_context_revision:
        raise ValueError("catalogue and geometry model contexts disagree")
    if discovery.detector_shape_rc != model.instrument.detector_shape_rc:
        raise ValueError("discovery and geometry model detector shapes disagree")
    if discovery.geometry_context_hash != _discovery_geometry_hash(
        model.instrument,
        angle_frame,
    ):
        raise ValueError("discovery was not frozen under the supplied geometry and angle frame")

    peaks = _canonicalized_peak_angles(
        discovery.peaks,
        instrument=model.instrument,
        angle_frame=angle_frame,
        validate_supplied=True,
    )
    candidate_indices = tuple(
        index
        for index, definition in enumerate(catalogue.definitions)
        if bool(catalogue.prediction.active_panel[index])
        and definition.key.layer_order.denominator in denominators
    )
    if not candidate_indices or not peaks:
        return None

    candidate_coordinates = catalogue.prediction.coordinates_px[
        np.asarray(candidate_indices, dtype=np.int64)
    ]
    policy = discovery.policy
    owner_candidates: dict[int, list[tuple[float, float, float, int]]] = {}
    for peak_index, peak in enumerate(peaks):
        if peak.z_score < policy.track_policy.minimum_site_z:
            continue
        observed = np.asarray((peak.column_px, peak.row_px), dtype=np.float64)
        delta = observed - candidate_coordinates
        distance_px = np.linalg.norm(delta, axis=1)
        covariance_px2 = np.asarray(peak.covariance_px2, dtype=np.float64) + (
            policy.track_policy.assignment_model_sigma_px**2 * np.eye(2)
        )
        solved = np.linalg.solve(covariance_px2, delta.T).T
        cost = np.einsum("ij,ij->i", delta, solved)
        eligible = np.flatnonzero(
            (distance_px <= policy.maximum_anchor_distance_px)
            & (np.sqrt(np.maximum(cost, 0.0)) <= policy.uncertainty_sigma)
        )
        if eligible.size == 0:
            continue
        ranked = sorted(
            ((float(cost[index]), candidate_indices[int(index)]) for index in eligible),
            key=lambda item: (item[0], item[1]),
        )
        if (
            len(ranked) > 1
            and ranked[1][0] - ranked[0][0] < policy.track_policy.minimum_assignment_margin
        ):
            continue
        best_cost, catalogue_index = ranked[0]
        owner_candidates.setdefault(catalogue_index, []).append(
            (best_cost, peak.row_px, peak.column_px, peak_index)
        )

    winners: dict[int, int] = {}
    for catalogue_index, owners in owner_candidates.items():
        owners.sort()
        if (
            len(owners) > 1
            and owners[1][0] - owners[0][0] < policy.track_policy.minimum_assignment_margin
        ):
            continue
        winners[catalogue_index] = owners[0][3]

    grouped_indices: dict[tuple[int, CommensurateLayerOrder, int], list[int]] = {}
    for catalogue_index in candidate_indices:
        key = catalogue.definitions[catalogue_index].key
        group_id = (key.family_m, key.layer_order, key.branch)
        grouped_indices.setdefault(group_id, []).append(catalogue_index)
    accepted: list[int] = []
    for expected in grouped_indices.values():
        if all(index in winners for index in expected):
            accepted.extend(expected)
    if not accepted:
        return None
    accepted.sort()
    return tuple(accepted), tuple(winners[index] for index in accepted)


def admit_discovered_pbi2_layer_l_peaks(
    discovery: MeasuredPeakDiscovery,
    *,
    model: ExactTagGeometryModel,
    catalogue: Pbi2PolytypeLandmarkCatalogue,
    angle_frame: AngleFrame,
    allowed_denominators: tuple[int, ...] = (2, 3),
) -> Pbi2LayerLPeakAdmission | None:
    """Admit complete unambiguous half/third-order groups after blind discovery.

    Discovery coordinates and covariances are frozen before this material-specific
    label pass. Missing or ambiguous groups contribute no observation.
    """

    qualified = _qualified_admission_indices(
        discovery,
        model=model,
        catalogue=catalogue,
        angle_frame=angle_frame,
        allowed_denominators=allowed_denominators,
    )
    if qualified is None:
        return None
    accepted, peak_indices = qualified
    return Pbi2LayerLPeakAdmission(
        discovery=discovery,
        model=model,
        catalogue=catalogue,
        angle_frame=angle_frame,
        allowed_denominators=allowed_denominators,
        catalogue_indices=accepted,
        discovery_peak_indices=peak_indices,
    )


__all__ = ["Pbi2LayerLPeakAdmission", "admit_discovered_pbi2_layer_l_peaks"]
