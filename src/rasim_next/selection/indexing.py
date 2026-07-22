"""Conservative measured ridge indexing without simulated-intensity fitting."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.optimize import linear_sum_assignment

from rasim_next.fitting import (
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
    IntegerLMarkerPrediction,
)
from rasim_next.geometry import (
    AngleFrame,
    CompiledInstrument,
    angles_to_detector_coordinates,
    detector_coordinates_to_angles,
)

_MAD_TO_SIGMA = 1.482602218505602
_LARGE_COST = 1.0e12
_TrackKey = tuple[int, int, int, tuple[int, int]]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _wrap_pi(value: ArrayLike) -> NDArray[np.float64]:
    angle = np.asarray(value, dtype=np.float64)
    return np.asarray((angle + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


def _robust_sigma(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    return _MAD_TO_SIGMA * float(np.median(np.abs(array - median)))


def _float_token(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


def _key_payload(key: IntegerLMarkerKey) -> tuple[object, ...]:
    return (
        key.family_m,
        key.integer_L,
        key.branch,
        key.root_sign,
        key.representative_rod_hk,
    )


def _track_key(key: IntegerLMarkerKey) -> _TrackKey:
    return key.family_m, key.branch, key.root_sign, key.representative_rod_hk


class MarkerIndexingStatus(StrEnum):
    """Measured-site state; absence is deliberately not an extinction claim."""

    GEOMETRY_INVISIBLE = "GEOMETRY_INVISIBLE"
    MASKED_OR_CLIPPED = "MASKED_OR_CLIPPED"
    BELOW_DETECTION = "BELOW_DETECTION"
    AMBIGUOUS_BLEND = "AMBIGUOUS_BLEND"
    VISIBLE_CONFIDENT = "VISIBLE_CONFIDENT"


@dataclass(frozen=True, slots=True)
class PeakIndexingPolicy:
    """Frozen thresholds for one measured indexing series."""

    search_radius_px: float = 40.0
    cake_step_px: float = 1.0
    minimum_candidate_z: float = 5.0
    minimum_site_z: float = 10.0
    minimum_assignment_margin: float = 0.5
    assignment_model_sigma_px: float = 6.0
    minimum_track_sites: int = 3
    minimum_common_l_sites: int = 2
    minimum_track_images: int = 2
    maximum_track_rms_px: float = 8.0
    minimum_chart_support: float = 0.90
    candidate_limit_per_marker: int = 4
    candidate_merge_radius_px: float = 2.5
    native_refinement_radius_px: int = 16
    revision: str = "local-angle-chart.native-ridge.branch-track.v2"

    def __post_init__(self) -> None:
        for name in (
            "search_radius_px",
            "cake_step_px",
            "minimum_candidate_z",
            "minimum_site_z",
            "minimum_assignment_margin",
            "assignment_model_sigma_px",
            "maximum_track_rms_px",
            "candidate_merge_radius_px",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.minimum_candidate_z >= self.minimum_site_z:
            raise ValueError("minimum_candidate_z must be smaller than minimum_site_z")
        support = _finite(self.minimum_chart_support, "minimum_chart_support")
        if not 0.0 < support <= 1.0:
            raise ValueError("minimum_chart_support must lie in (0, 1]")
        object.__setattr__(self, "minimum_chart_support", support)
        for name in (
            "minimum_track_sites",
            "minimum_common_l_sites",
            "minimum_track_images",
            "candidate_limit_per_marker",
            "native_refinement_radius_px",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name, minimum in (
            ("minimum_track_sites", 3),
            ("minimum_common_l_sites", 2),
            ("minimum_track_images", 2),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision must be a nonempty string")


@dataclass(frozen=True, slots=True)
class MarkerIndexingDecision:
    """One frozen predicted tag and its measured detector-native decision."""

    key: IntegerLMarkerKey
    predicted_column_px: float
    predicted_row_px: float
    predicted_two_theta_rad: float
    predicted_phi_rad: float
    status: MarkerIndexingStatus
    reason: str
    observed_column_px: float | None = None
    observed_row_px: float | None = None
    observed_two_theta_rad: float | None = None
    observed_phi_rad: float | None = None
    covariance_px2: tuple[tuple[float, float], tuple[float, float]] | None = None
    z_score: float | None = None
    assignment_cost: float | None = None
    assignment_margin: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, IntegerLMarkerKey):
            raise TypeError("key must be an IntegerLMarkerKey")
        for name in (
            "predicted_column_px",
            "predicted_row_px",
            "predicted_two_theta_rad",
            "predicted_phi_rad",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        status = MarkerIndexingStatus(self.status)
        object.__setattr__(self, "status", status)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a nonempty string")
        optional_coordinates = (
            self.observed_column_px,
            self.observed_row_px,
            self.observed_two_theta_rad,
            self.observed_phi_rad,
        )
        has_observation = any(value is not None for value in optional_coordinates)
        if has_observation and any(value is None for value in optional_coordinates):
            raise ValueError("observed detector and angle coordinates must be supplied together")
        if has_observation:
            for name in (
                "observed_column_px",
                "observed_row_px",
                "observed_two_theta_rad",
                "observed_phi_rad",
            ):
                object.__setattr__(self, name, _finite(getattr(self, name), name))
        covariance = self.covariance_px2
        if covariance is not None:
            array = np.asarray(covariance, dtype=np.float64)
            if (
                array.shape != (2, 2)
                or not np.all(np.isfinite(array))
                or not np.array_equal(array, array.T)
                or np.any(np.linalg.eigvalsh(array) <= 0.0)
            ):
                raise ValueError("covariance_px2 must be finite, symmetric, and positive definite")
            covariance = tuple(tuple(float(value) for value in row) for row in array)
            object.__setattr__(self, "covariance_px2", covariance)
        if status == MarkerIndexingStatus.VISIBLE_CONFIDENT and (
            not has_observation or covariance is None
        ):
            raise ValueError("a confident visible marker requires coordinates and covariance")
        for name in (
            "z_score",
            "assignment_cost",
            "assignment_margin",
        ):
            value = getattr(self, name)
            if value is not None:
                value = _finite(value, name)
                if name != "assignment_cost" and value < 0.0:
                    raise ValueError(f"{name} must be nonnegative")
                object.__setattr__(self, name, value)

    @property
    def track_key(self) -> _TrackKey:
        return _track_key(self.key)


@dataclass(frozen=True, slots=True)
class BranchTrackDecision:
    """Confidence decision for one ordered physical root track across L."""

    family_m: int
    branch: int
    root_sign: int
    representative_rod_hk: tuple[int, int]
    image_ids: tuple[str, ...]
    accepted: bool
    confident_site_count: int
    distinct_integer_L: tuple[int, ...]
    rms_px: float | None
    reason: str

    def __post_init__(self) -> None:
        probe = IntegerLMarkerKey(
            self.family_m,
            1,
            self.branch,
            self.root_sign,
            self.representative_rod_hk,
        )
        object.__setattr__(self, "family_m", probe.family_m)
        object.__setattr__(self, "branch", probe.branch)
        object.__setattr__(self, "root_sign", probe.root_sign)
        object.__setattr__(self, "representative_rod_hk", probe.representative_rod_hk)
        image_ids = tuple(sorted(set(self.image_ids)))
        if not image_ids or any(not isinstance(value, str) or not value for value in image_ids):
            raise ValueError("image_ids must contain nonempty unique strings")
        object.__setattr__(self, "image_ids", image_ids)
        count = self.confident_site_count
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("confident_site_count must be a nonnegative integer")
        object.__setattr__(self, "confident_site_count", int(count))
        integer_l = tuple(sorted(set(int(value) for value in self.distinct_integer_L)))
        object.__setattr__(self, "distinct_integer_L", integer_l)
        if self.rms_px is not None:
            rms = _finite(self.rms_px, "rms_px")
            if rms < 0.0:
                raise ValueError("rms_px must be nonnegative")
            object.__setattr__(self, "rms_px", rms)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a nonempty string")

    @property
    def track_key(self) -> _TrackKey:
        return self.family_m, self.branch, self.root_sign, self.representative_rod_hk


def _decision_payload(decision: MarkerIndexingDecision) -> dict[str, object]:
    return {
        "key": _key_payload(decision.key),
        "predicted_column_px": _float_token(decision.predicted_column_px),
        "predicted_row_px": _float_token(decision.predicted_row_px),
        "predicted_two_theta_rad": _float_token(decision.predicted_two_theta_rad),
        "predicted_phi_rad": _float_token(decision.predicted_phi_rad),
        "status": decision.status.value,
        "reason": decision.reason,
        "observed_column_px": _float_token(decision.observed_column_px),
        "observed_row_px": _float_token(decision.observed_row_px),
        "observed_two_theta_rad": _float_token(decision.observed_two_theta_rad),
        "observed_phi_rad": _float_token(decision.observed_phi_rad),
        "covariance_px2": (
            None
            if decision.covariance_px2 is None
            else tuple(
                tuple(_float_token(value) for value in row) for row in decision.covariance_px2
            )
        ),
        "z_score": _float_token(decision.z_score),
        "assignment_cost": _float_token(decision.assignment_cost),
        "assignment_margin": _float_token(decision.assignment_margin),
    }


def _track_payload(track: BranchTrackDecision) -> dict[str, object]:
    return {
        "track_key": track.track_key,
        "image_ids": track.image_ids,
        "accepted": track.accepted,
        "confident_site_count": track.confident_site_count,
        "distinct_integer_L": track.distinct_integer_L,
        "rms_px": _float_token(track.rms_px),
        "reason": track.reason,
    }


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256-{hashlib.sha256(encoded).hexdigest()}"


def _is_sha256_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256-")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


@dataclass(frozen=True, slots=True)
class MeasuredImageIndexingResult:
    """All measured decisions for one image before cross-image track selection."""

    image_id: str
    incidence_angle_rad: float
    reference_wavelength_A: float
    marker_decisions: tuple[MarkerIndexingDecision, ...]
    detector_data_hash: str
    detector_mask_hash: str
    detector_mask_revision: str
    context_hash: str
    policy: PeakIndexingPolicy
    branch_tracks: tuple[BranchTrackDecision, ...] = field(init=False)
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("image_id must be a nonempty string")
        object.__setattr__(
            self,
            "incidence_angle_rad",
            _finite(self.incidence_angle_rad, "incidence_angle_rad"),
        )
        object.__setattr__(
            self,
            "reference_wavelength_A",
            _positive(self.reference_wavelength_A, "reference_wavelength_A"),
        )
        decisions = tuple(sorted(self.marker_decisions, key=lambda item: item.key))
        if len({item.key for item in decisions}) != len(decisions):
            raise ValueError("marker_decisions must contain unique marker keys")
        for name in ("detector_data_hash", "detector_mask_hash", "context_hash"):
            value = getattr(self, name)
            if not _is_sha256_identifier(value):
                raise ValueError(f"{name} must be a SHA-256 identifier")
        if (
            not isinstance(self.detector_mask_revision, str)
            or not self.detector_mask_revision.strip()
        ):
            raise ValueError("detector_mask_revision must be a nonempty string")
        if not isinstance(self.policy, PeakIndexingPolicy):
            raise TypeError("policy must be a PeakIndexingPolicy")
        tracks = _track_decisions(self.image_id, decisions, self.policy)
        object.__setattr__(self, "marker_decisions", decisions)
        object.__setattr__(self, "branch_tracks", tracks)
        payload = {
            "schema": "measured-image-indexing-result.v1",
            "image_id": self.image_id,
            "incidence_angle_rad": _float_token(self.incidence_angle_rad),
            "reference_wavelength_A": _float_token(self.reference_wavelength_A),
            "detector_data_hash": self.detector_data_hash,
            "detector_mask_hash": self.detector_mask_hash,
            "detector_mask_revision": self.detector_mask_revision,
            "context_hash": self.context_hash,
            "policy": asdict(self.policy),
            "marker_decisions": tuple(_decision_payload(item) for item in decisions),
            "branch_tracks": tuple(_track_payload(item) for item in tracks),
        }
        object.__setattr__(self, "result_hash", _sha256_payload(payload))


@dataclass(frozen=True, slots=True)
class MeasuredIndexingResult:
    """Cross-image branch-track manifest and detector-native geometry handoff."""

    image_results: tuple[MeasuredImageIndexingResult, ...]
    policy: PeakIndexingPolicy
    branch_tracks: tuple[BranchTrackDecision, ...] = field(init=False)
    manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        images = tuple(sorted(self.image_results, key=lambda item: item.image_id))
        if not images or len({item.image_id for item in images}) != len(images):
            raise ValueError("image_results must contain unique image IDs")
        if any(item.policy != self.policy for item in images):
            raise ValueError("all image results must use the manifest policy")
        tracks = _cross_image_tracks(images, self.policy)
        object.__setattr__(self, "image_results", images)
        object.__setattr__(self, "branch_tracks", tracks)
        payload = {
            "schema": "measured-indexing-manifest.v2",
            "policy": asdict(self.policy),
            "images": tuple((item.image_id, item.result_hash) for item in images),
            "branch_tracks": tuple(_track_payload(item) for item in tracks),
        }
        object.__setattr__(self, "manifest_hash", _sha256_payload(payload))

    def observations_for(self, image_id: str) -> IntegerLMarkerObservations:
        """Return visible tags whose track and exact L both replicate across images."""

        try:
            image = next(item for item in self.image_results if item.image_id == image_id)
        except StopIteration as error:
            raise KeyError(f"unknown image_id {image_id!r}") from error
        accepted = {
            item.track_key: item
            for item in self.branch_tracks
            if item.accepted and image_id in item.image_ids
        }
        replicated_l: dict[_TrackKey, set[int]] = {}
        for track_key, track in accepted.items():
            image_ids_by_l: dict[int, set[str]] = {}
            for candidate_image in self.image_results:
                if candidate_image.image_id not in track.image_ids:
                    continue
                locally_accepted = {
                    item.track_key for item in candidate_image.branch_tracks if item.accepted
                }
                if track_key not in locally_accepted:
                    continue
                for decision in candidate_image.marker_decisions:
                    if (
                        decision.track_key == track_key
                        and decision.status == MarkerIndexingStatus.VISIBLE_CONFIDENT
                    ):
                        image_ids_by_l.setdefault(decision.key.integer_L, set()).add(
                            candidate_image.image_id
                        )
            replicated_l[track_key] = {
                integer_l
                for integer_l, supporting_images in image_ids_by_l.items()
                if integer_l in track.distinct_integer_L
                and len(supporting_images) >= self.policy.minimum_track_images
            }
        locally_accepted = {item.track_key for item in image.branch_tracks if item.accepted}
        decisions = tuple(
            item
            for item in image.marker_decisions
            if item.status == MarkerIndexingStatus.VISIBLE_CONFIDENT
            and item.track_key in accepted
            and item.track_key in locally_accepted
            and item.key.integer_L in replicated_l[item.track_key]
        )
        if not decisions:
            raise ValueError(f"image {image_id!r} has no accepted visible marker observations")
        coordinates = np.asarray(
            tuple((item.observed_column_px, item.observed_row_px) for item in decisions),
            dtype=np.float64,
        )
        covariance = np.asarray(tuple(item.covariance_px2 for item in decisions), dtype=np.float64)
        return IntegerLMarkerObservations(
            keys=tuple(item.key for item in decisions),
            coordinates_px=coordinates,
            covariance_px2=covariance,
            reference_wavelength_A=image.reference_wavelength_A,
        )


@dataclass(frozen=True, slots=True)
class FrozenMarkerVisibilityImageAudit:
    image_id: str
    expected_count: int
    visible_confident_count: int
    missing_keys: tuple[IntegerLMarkerKey, ...]
    nonconfident_keys: tuple[IntegerLMarkerKey, ...]
    moved_keys: tuple[IntegerLMarkerKey, ...]
    newly_visible_keys: tuple[IntegerLMarkerKey, ...]
    frozen_subset_track_coherent: bool

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("audit image_id must be nonempty")
        for name in ("expected_count", "visible_confident_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("missing_keys", "nonconfident_keys", "moved_keys", "newly_visible_keys"):
            values = tuple(getattr(self, name))
            if any(not isinstance(key, IntegerLMarkerKey) for key in values):
                raise TypeError(f"{name} must contain IntegerLMarkerKey values")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicate keys")
            object.__setattr__(self, name, tuple(sorted(values)))
        if not isinstance(self.frozen_subset_track_coherent, bool):
            raise TypeError("frozen_subset_track_coherent must be bool")


@dataclass(frozen=True, slots=True)
class FrozenMarkerVisibilityAudit:
    classification: str
    images: tuple[FrozenMarkerVisibilityImageAudit, ...]
    same_ridge_tolerance_px: float

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not images or any(
            not isinstance(item, FrozenMarkerVisibilityImageAudit) for item in images
        ):
            raise ValueError("images must contain at least one visibility audit")
        if tuple(item.image_id for item in images) != tuple(
            sorted(item.image_id for item in images)
        ) or len({item.image_id for item in images}) != len(images):
            raise ValueError("visibility audits must have unique canonical image IDs")
        tolerance = _positive(self.same_ridge_tolerance_px, "same_ridge_tolerance_px")
        expected = (
            "SAME"
            if all(
                not item.missing_keys
                and not item.nonconfident_keys
                and not item.moved_keys
                and item.frozen_subset_track_coherent
                for item in images
            )
            else "CHANGED"
        )
        if self.classification != expected:
            raise ValueError("visibility-audit classification disagrees with its images")
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "same_ridge_tolerance_px", tolerance)


def audit_frozen_marker_visibility(
    frozen: MeasuredIndexingResult,
    reindexed: MeasuredIndexingResult,
    *,
    same_ridge_tolerance_px: float | None = None,
) -> FrozenMarkerVisibilityAudit:
    """Audit frozen fitted keys without reapplying mutable all-candidate track gates."""

    if not isinstance(frozen, MeasuredIndexingResult) or not isinstance(
        reindexed, MeasuredIndexingResult
    ):
        raise TypeError("frozen and reindexed must be MeasuredIndexingResult")
    if frozen.policy != reindexed.policy:
        raise ValueError("frozen and reindexed policies differ")
    frozen_by_id = {item.image_id: item for item in frozen.image_results}
    fresh_by_id = {item.image_id: item for item in reindexed.image_results}
    if set(frozen_by_id) != set(fresh_by_id):
        raise ValueError("frozen and reindexed image IDs differ")
    tolerance = (
        frozen.policy.candidate_merge_radius_px
        if same_ridge_tolerance_px is None
        else _positive(same_ridge_tolerance_px, "same_ridge_tolerance_px")
    )
    subset_results: list[MeasuredImageIndexingResult] = []
    partial: list[
        tuple[
            str,
            tuple[IntegerLMarkerKey, ...],
            tuple[IntegerLMarkerKey, ...],
            tuple[IntegerLMarkerKey, ...],
            tuple[IntegerLMarkerKey, ...],
            tuple[IntegerLMarkerKey, ...],
            int,
        ]
    ] = []
    for image_id in sorted(frozen_by_id):
        before = frozen_by_id[image_id]
        after = fresh_by_id[image_id]
        if (
            before.detector_data_hash != after.detector_data_hash
            or before.detector_mask_hash != after.detector_mask_hash
            or before.detector_mask_revision != after.detector_mask_revision
            or not math.isclose(
                before.incidence_angle_rad,
                after.incidence_angle_rad,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
            or not math.isclose(
                before.reference_wavelength_A,
                after.reference_wavelength_A,
                rel_tol=0.0,
                abs_tol=256.0
                * np.finfo(np.float64).eps
                * max(before.reference_wavelength_A, after.reference_wavelength_A, 1.0),
            )
        ):
            raise ValueError(f"image {image_id!r} has incomparable measured provenance")
        observations = frozen.observations_for(image_id)
        expected_keys = observations.keys
        expected_coordinates = {
            key: observations.coordinates_px[index] for index, key in enumerate(expected_keys)
        }
        fresh_decisions = {decision.key: decision for decision in after.marker_decisions}
        missing: list[IntegerLMarkerKey] = []
        nonconfident: list[IntegerLMarkerKey] = []
        moved: list[IntegerLMarkerKey] = []
        matched: list[MarkerIndexingDecision] = []
        for key in expected_keys:
            decision = fresh_decisions.get(key)
            if decision is None:
                missing.append(key)
                continue
            if decision.status != MarkerIndexingStatus.VISIBLE_CONFIDENT:
                nonconfident.append(key)
                continue
            assert decision.observed_column_px is not None
            assert decision.observed_row_px is not None
            distance = math.hypot(
                decision.observed_column_px - float(expected_coordinates[key][0]),
                decision.observed_row_px - float(expected_coordinates[key][1]),
            )
            if distance > tolerance:
                moved.append(key)
            matched.append(decision)
        before_visible = {
            decision.key
            for decision in before.marker_decisions
            if decision.status == MarkerIndexingStatus.VISIBLE_CONFIDENT
        }
        newly_visible = tuple(
            sorted(
                decision.key
                for decision in after.marker_decisions
                if decision.status == MarkerIndexingStatus.VISIBLE_CONFIDENT
                and decision.key not in before_visible
            )
        )
        if not missing and not nonconfident:
            subset_results.append(
                MeasuredImageIndexingResult(
                    image_id=after.image_id,
                    incidence_angle_rad=after.incidence_angle_rad,
                    reference_wavelength_A=after.reference_wavelength_A,
                    marker_decisions=tuple(matched),
                    detector_data_hash=after.detector_data_hash,
                    detector_mask_hash=after.detector_mask_hash,
                    detector_mask_revision=after.detector_mask_revision,
                    context_hash=after.context_hash,
                    policy=after.policy,
                )
            )
        partial.append(
            (
                image_id,
                expected_keys,
                tuple(missing),
                tuple(nonconfident),
                tuple(moved),
                newly_visible,
                len(matched),
            )
        )

    coherent_ids: set[str] = set()
    if len(subset_results) == len(frozen_by_id):
        subset = MeasuredIndexingResult(tuple(subset_results), frozen.policy)
        for image_id, expected_keys, *_ in partial:
            try:
                exported = subset.observations_for(image_id).keys
            except ValueError:
                continue
            if set(exported) == set(expected_keys) and len(exported) == len(expected_keys):
                coherent_ids.add(image_id)
    images = tuple(
        FrozenMarkerVisibilityImageAudit(
            image_id=image_id,
            expected_count=len(expected_keys),
            visible_confident_count=matched_count,
            missing_keys=missing,
            nonconfident_keys=nonconfident,
            moved_keys=moved,
            newly_visible_keys=newly_visible,
            frozen_subset_track_coherent=image_id in coherent_ids,
        )
        for (
            image_id,
            expected_keys,
            missing,
            nonconfident,
            moved,
            newly_visible,
            matched_count,
        ) in partial
    )
    classification = (
        "SAME"
        if all(
            not item.missing_keys
            and not item.nonconfident_keys
            and not item.moved_keys
            and item.frozen_subset_track_coherent
            for item in images
        )
        else "CHANGED"
    )
    return FrozenMarkerVisibilityAudit(classification, images, tolerance)


@dataclass(frozen=True, slots=True)
class _Candidate:
    column_px: float
    row_px: float
    two_theta_rad: float
    phi_rad: float
    covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    localization_covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    z_score: float


def _array_hash(value: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return f"sha256-{digest.hexdigest()}"


def _context_hash(
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    prediction: IntegerLMarkerPrediction,
) -> str:
    payload = {
        "schema": "measured-indexing-context.v1",
        "instrument": {
            "lab_from_detector_rotation": tuple(
                tuple(_float_token(value) for value in row)
                for row in instrument.lab_from_detector.rotation
            ),
            "lab_from_detector_translation_m": tuple(
                _float_token(value) for value in instrument.lab_from_detector.translation_m
            ),
            "detector_shape_rc": instrument.detector_shape_rc,
            "detector_row_pitch_m": _float_token(instrument.detector_row_pitch_m),
            "detector_column_pitch_m": _float_token(instrument.detector_column_pitch_m),
            "detector_reference_coordinate_px": tuple(
                _float_token(value) for value in instrument.detector_reference_coordinate_px
            ),
        },
        "angle_frame": {
            "origin_lab_m": tuple(_float_token(value) for value in angle_frame.origin_lab_m),
            "row_down_lab": tuple(_float_token(value) for value in angle_frame.row_down_lab),
            "column_right_lab": tuple(
                _float_token(value) for value in angle_frame.column_right_lab
            ),
            "direct_beam_lab": tuple(_float_token(value) for value in angle_frame.direct_beam_lab),
            "revision": angle_frame.revision,
        },
        "prediction": tuple(
            (
                _key_payload(key),
                _float_token(coordinate[0]),
                _float_token(coordinate[1]),
                str(status),
                _float_token(residual),
            )
            for key, coordinate, status, residual in sorted(
                zip(
                    prediction.keys,
                    prediction.coordinates_px,
                    prediction.detector_status,
                    prediction.ewald_residual_Ainv,
                    strict=True,
                ),
                key=lambda item: item[0],
            )
        ),
    }
    return _sha256_payload(payload)


def _angle_derivatives_px_per_rad(
    two_theta_rad: float,
    phi_rad: float,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    step = 1.0e-5
    coordinates = angles_to_detector_coordinates(
        np.asarray((two_theta_rad + step, two_theta_rad - step, two_theta_rad, two_theta_rad)),
        np.asarray((phi_rad, phi_rad, phi_rad + step, phi_rad - step)),
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not np.all(coordinates.valid):
        return None
    theta_derivative = np.asarray(
        (
            coordinates.column_px[0] - coordinates.column_px[1],
            coordinates.row_px[0] - coordinates.row_px[1],
        )
    ) / (2.0 * step)
    phi_derivative = np.asarray(
        (
            coordinates.column_px[2] - coordinates.column_px[3],
            coordinates.row_px[2] - coordinates.row_px[3],
        )
    ) / (2.0 * step)
    if min(np.linalg.norm(theta_derivative), np.linalg.norm(phi_derivative)) <= 1.0e-9:
        return None
    return theta_derivative, phi_derivative


def _native_refinement(
    detector_counts: NDArray[np.generic],
    detector_valid_mask: NDArray[np.bool_],
    column_px: float,
    row_px: float,
    chart_z: float,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    policy: PeakIndexingPolicy,
) -> _Candidate | None:
    radius = policy.native_refinement_radius_px
    center_column = round(column_px)
    center_row = round(row_px)
    row_start = center_row - radius
    row_stop = center_row + radius + 1
    column_start = center_column - radius
    column_stop = center_column + radius + 1
    rows, columns = detector_counts.shape
    if row_start < 0 or column_start < 0 or row_stop > rows or column_stop > columns:
        return None
    valid = detector_valid_mask[row_start:row_stop, column_start:column_stop]
    if float(np.mean(valid)) < policy.minimum_chart_support:
        return None
    patch = np.asarray(
        detector_counts[row_start:row_stop, column_start:column_stop],
        dtype=np.float64,
    )
    local_row, local_column = np.indices(patch.shape, dtype=np.float64)
    border = valid & (
        (local_row < 2)
        | (local_row >= patch.shape[0] - 2)
        | (local_column < 2)
        | (local_column >= patch.shape[1] - 2)
    )
    if np.count_nonzero(border) < 12:
        return None
    design = np.column_stack(
        (np.ones(np.count_nonzero(border)), local_column[border], local_row[border])
    )
    coefficients = np.linalg.lstsq(design, patch[border], rcond=None)[0]
    background = coefficients[0] + coefficients[1] * local_column + coefficients[2] * local_row
    residual = np.where(valid, patch - background, 0.0)
    smoothed = gaussian_filter(residual, 1.2, mode="nearest")
    approximate_column = column_px - column_start
    approximate_row = row_px - row_start
    search = (
        valid
        & (np.abs(local_column - approximate_column) <= 4.0)
        & (np.abs(local_row - approximate_row) <= 4.0)
    )
    if not np.any(search):
        return None
    masked_score = np.where(search, smoothed, -np.inf)
    peak_row, peak_column = np.unravel_index(int(np.argmax(masked_score)), patch.shape)
    localization_radius = min(10.0, float(radius - 3))
    localization_core = (local_column - peak_column) ** 2 + (
        local_row - peak_row
    ) ** 2 <= localization_radius**2
    if float(np.mean(valid[localization_core])) < policy.minimum_chart_support:
        return None
    localization = valid & localization_core
    floor = float(np.median(smoothed[border]))
    weight = np.where(localization, np.maximum(smoothed - floor, 0.0), 0.0)
    total_weight = float(np.sum(weight))
    if total_weight <= 0.0:
        return None
    refined_column_local = float(np.sum(weight * local_column) / total_weight)
    refined_row_local = float(np.sum(weight * local_row) / total_weight)
    delta_column = local_column - refined_column_local
    delta_row = local_row - refined_row_local
    shape_covariance = np.asarray(
        (
            (
                float(np.sum(weight * delta_column**2) / total_weight),
                float(np.sum(weight * delta_column * delta_row) / total_weight),
            ),
            (
                float(np.sum(weight * delta_column * delta_row) / total_weight),
                float(np.sum(weight * delta_row**2) / total_weight),
            ),
        )
    )
    native_sigma = _robust_sigma(smoothed[border])
    native_z = (
        math.inf
        if native_sigma <= np.finfo(np.float64).eps
        else max(0.0, (float(smoothed[peak_row, peak_column]) - floor) / native_sigma)
    )
    z_score = min(chart_z, native_z)
    # This is deliberately a conservative ridge-support covariance, not a calibrated
    # centroid standard error.  In particular, high signal must not erase uncertainty
    # along an extended ridge before the detector-geometry fit.
    covariance = shape_covariance + 0.25 * np.eye(2)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvalue = np.maximum(eigenvalue, 0.25)
    covariance = (eigenvector * eigenvalue) @ eigenvector.T
    covariance = 0.5 * (covariance + covariance.T)
    localization_covariance = shape_covariance / max(z_score, 1.0) + 0.25 * np.eye(2)
    localization_eigenvalue, localization_eigenvector = np.linalg.eigh(localization_covariance)
    localization_eigenvalue = np.maximum(localization_eigenvalue, 0.25)
    localization_covariance = (
        localization_eigenvector * localization_eigenvalue
    ) @ localization_eigenvector.T
    localization_covariance = 0.5 * (localization_covariance + localization_covariance.T)
    refined_column = column_start + refined_column_local
    refined_row = row_start + refined_row_local
    angles = detector_coordinates_to_angles(
        refined_column,
        refined_row,
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not bool(angles.valid) or not bool(angles.azimuth_valid):
        return None
    return _Candidate(
        column_px=refined_column,
        row_px=refined_row,
        two_theta_rad=float(angles.two_theta_rad),
        phi_rad=float(angles.phi_rad),
        covariance_px2=tuple(tuple(float(value) for value in row) for row in covariance),
        localization_covariance_px2=tuple(
            tuple(float(value) for value in row) for row in localization_covariance
        ),
        z_score=z_score,
    )


def _local_angle_chart_candidates(
    detector_counts: NDArray[np.generic],
    detector_valid_mask: NDArray[np.bool_],
    detector_valid_mask_u8: NDArray[np.uint8],
    predicted_two_theta_rad: float,
    predicted_phi_rad: float,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    policy: PeakIndexingPolicy,
) -> tuple[tuple[_Candidate, ...], bool]:
    derivatives = _angle_derivatives_px_per_rad(
        predicted_two_theta_rad,
        predicted_phi_rad,
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if derivatives is None:
        return (), False
    theta_derivative, phi_derivative = derivatives
    theta_step = policy.cake_step_px / float(np.linalg.norm(theta_derivative))
    phi_step = policy.cake_step_px / float(np.linalg.norm(phi_derivative))
    half_count = math.ceil(policy.search_radius_px / policy.cake_step_px)
    offset = np.arange(-half_count, half_count + 1, dtype=np.float64)
    phi_grid, theta_grid = np.meshgrid(
        predicted_phi_rad + offset * phi_step,
        predicted_two_theta_rad + offset * theta_step,
        indexing="ij",
    )
    if np.any(theta_grid < 0.0) or np.any(theta_grid > np.pi):
        return (), False
    coordinates = angles_to_detector_coordinates(
        theta_grid,
        phi_grid,
        instrument=instrument,
        angle_frame=angle_frame,
    )
    support = coordinates.valid.copy()
    sampled_mask = map_coordinates(
        detector_valid_mask_u8,
        (coordinates.row_px, coordinates.column_px),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.bool_)
    support &= sampled_mask
    radial_offset_px = np.hypot(
        *np.meshgrid(offset * policy.cake_step_px, offset * policy.cake_step_px)
    )
    search_support = support & (radial_offset_px <= policy.search_radius_px)
    support_fraction = float(np.mean(search_support[radial_offset_px <= policy.search_radius_px]))
    if support_fraction < policy.minimum_chart_support:
        return (), False
    sampled = map_coordinates(
        detector_counts,
        (coordinates.row_px, coordinates.column_px),
        output=np.float64,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    fill = float(np.median(sampled[search_support]))
    sampled = np.where(support, sampled, fill)
    narrow_sigma = max(0.75, 1.3 / policy.cake_step_px)
    broad_sigma = max(narrow_sigma * 2.5, 0.30 * policy.search_radius_px / policy.cake_step_px)
    response = gaussian_filter(sampled, narrow_sigma, mode="nearest") - gaussian_filter(
        sampled,
        broad_sigma,
        mode="nearest",
    )
    sideband = search_support & (radial_offset_px >= 0.75 * policy.search_radius_px)
    response_values = (
        response[sideband] if np.count_nonzero(sideband) >= 20 else response[search_support]
    )
    center = float(np.median(response_values))
    sigma = _robust_sigma(response_values)
    if sigma <= np.finfo(np.float64).eps:
        return (), True
    maximum_size = max(3, 2 * round(2.0 / policy.cake_step_px) + 1)
    local_maximum = response == maximum_filter(response, maximum_size, mode="nearest")
    candidate_mask = (
        search_support
        & (radial_offset_px <= policy.search_radius_px - 2.0 * policy.cake_step_px)
        & local_maximum
        & ((response - center) >= policy.minimum_candidate_z * sigma)
    )
    candidate_index = np.argwhere(candidate_mask)
    if candidate_index.size == 0:
        return (), True
    z_score = (response[candidate_mask] - center) / sigma
    distance = radial_offset_px[candidate_mask]
    ordering = np.lexsort((candidate_index[:, 1], candidate_index[:, 0], distance, -z_score))
    output: list[_Candidate] = []
    for position in ordering[: policy.candidate_limit_per_marker]:
        row_index, column_index = candidate_index[int(position)]
        refined = _native_refinement(
            detector_counts,
            detector_valid_mask,
            float(coordinates.column_px[row_index, column_index]),
            float(coordinates.row_px[row_index, column_index]),
            float(z_score[int(position)]),
            instrument=instrument,
            angle_frame=angle_frame,
            policy=policy,
        )
        if refined is not None:
            output.append(refined)
    return tuple(output), True


def _merge_candidates(
    candidates: tuple[_Candidate, ...],
    merge_radius_px: float,
) -> tuple[_Candidate, ...]:
    merged: list[_Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.z_score, item.row_px, item.column_px),
    ):
        if any(
            math.hypot(candidate.column_px - item.column_px, candidate.row_px - item.row_px)
            <= merge_radius_px
            for item in merged
        ):
            continue
        merged.append(candidate)
    return tuple(sorted(merged, key=lambda item: (item.row_px, item.column_px, -item.z_score)))


def _cake_jacobian_rad_per_px(
    predicted_column_px: float,
    predicted_row_px: float,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> NDArray[np.float64] | None:
    coordinate_step = 0.5
    angles = detector_coordinates_to_angles(
        np.asarray(
            (
                predicted_column_px + coordinate_step,
                predicted_column_px - coordinate_step,
                predicted_column_px,
                predicted_column_px,
            )
        ),
        np.asarray(
            (
                predicted_row_px,
                predicted_row_px,
                predicted_row_px + coordinate_step,
                predicted_row_px - coordinate_step,
            )
        ),
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not np.all(angles.valid) or not np.all(angles.azimuth_valid):
        return None
    return np.asarray(
        (
            (
                (angles.two_theta_rad[0] - angles.two_theta_rad[1]) / (2.0 * coordinate_step),
                (angles.two_theta_rad[2] - angles.two_theta_rad[3]) / (2.0 * coordinate_step),
            ),
            (
                float(_wrap_pi(angles.phi_rad[0] - angles.phi_rad[1])) / (2.0 * coordinate_step),
                float(_wrap_pi(angles.phi_rad[2] - angles.phi_rad[3])) / (2.0 * coordinate_step),
            ),
        )
    )


def _base_decision(
    key: IntegerLMarkerKey,
    coordinate_px: NDArray[np.float64],
    two_theta_rad: float,
    phi_rad: float,
    status: MarkerIndexingStatus,
    reason: str,
) -> MarkerIndexingDecision:
    return MarkerIndexingDecision(
        key=key,
        predicted_column_px=float(coordinate_px[0]),
        predicted_row_px=float(coordinate_px[1]),
        predicted_two_theta_rad=two_theta_rad,
        predicted_phi_rad=phi_rad,
        status=status,
        reason=reason,
    )


def _assigned_decisions(
    keys: tuple[IntegerLMarkerKey, ...],
    coordinates_px: NDArray[np.float64],
    predicted_theta: NDArray[np.float64],
    predicted_phi: NDArray[np.float64],
    candidates: tuple[_Candidate, ...],
    chart_supported: NDArray[np.bool_],
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    policy: PeakIndexingPolicy,
) -> tuple[MarkerIndexingDecision, ...]:
    marker_count = len(keys)
    candidate_count = len(candidates)
    cost = np.full((marker_count, candidate_count), _LARGE_COST, dtype=np.float64)
    for marker_index in range(marker_count):
        if not chart_supported[marker_index]:
            continue
        jacobian = _cake_jacobian_rad_per_px(
            float(coordinates_px[marker_index, 0]),
            float(coordinates_px[marker_index, 1]),
            instrument=instrument,
            angle_frame=angle_frame,
        )
        for candidate_index, candidate in enumerate(candidates):
            native_delta = np.asarray(
                (
                    candidate.column_px - coordinates_px[marker_index, 0],
                    candidate.row_px - coordinates_px[marker_index, 1],
                )
            )
            angle_delta = np.asarray(
                (
                    candidate.two_theta_rad - predicted_theta[marker_index],
                    float(_wrap_pi(candidate.phi_rad - predicted_phi[marker_index])),
                )
            )
            try:
                equivalent_delta = (
                    native_delta if jacobian is None else np.linalg.solve(jacobian, angle_delta)
                )
            except np.linalg.LinAlgError:
                equivalent_delta = native_delta
            value = float(np.linalg.norm(equivalent_delta))
            if value <= policy.search_radius_px:
                covariance = np.asarray(candidate.covariance_px2) + (
                    policy.assignment_model_sigma_px**2 * np.eye(2)
                )
                cost[marker_index, candidate_index] = float(
                    native_delta @ np.linalg.solve(covariance, native_delta)
                )
    dummy_cost = (policy.search_radius_px / policy.assignment_model_sigma_px) ** 2
    assignment = np.full((marker_count, candidate_count + marker_count), _LARGE_COST)
    assignment[:, :candidate_count] = cost
    assignment[np.arange(marker_count), candidate_count + np.arange(marker_count)] = dummy_cost
    row_index, column_index = linear_sum_assignment(assignment)
    selected = dict(zip(row_index.tolist(), column_index.tolist(), strict=True))

    ambiguous_marker: set[int] = set()
    assignment_margin = np.full(marker_count, np.nan, dtype=np.float64)
    base_total = float(np.sum(assignment[row_index, column_index]))
    for marker_index, assigned_column in selected.items():
        if assigned_column >= candidate_count or cost[marker_index, assigned_column] >= _LARGE_COST:
            continue
        alternative = assignment.copy()
        alternative[marker_index, assigned_column] = _LARGE_COST
        alternative_rows, alternative_columns = linear_sum_assignment(alternative)
        margin = float(np.sum(alternative[alternative_rows, alternative_columns]) - base_total)
        assignment_margin[marker_index] = margin
        if margin < policy.minimum_assignment_margin:
            alternative_selected = dict(
                zip(alternative_rows.tolist(), alternative_columns.tolist(), strict=True)
            )
            ambiguous_marker.update(
                row for row in range(marker_count) if alternative_selected[row] != selected[row]
            )

    decisions: list[MarkerIndexingDecision] = []
    for marker_index, key in enumerate(keys):
        if not chart_supported[marker_index]:
            decisions.append(
                _base_decision(
                    key,
                    coordinates_px[marker_index],
                    float(predicted_theta[marker_index]),
                    float(predicted_phi[marker_index]),
                    MarkerIndexingStatus.MASKED_OR_CLIPPED,
                    "local angle-space chart lacks sufficient valid detector support",
                )
            )
            continue
        assigned = selected[marker_index]
        best_eligible = np.flatnonzero(cost[marker_index] < _LARGE_COST)
        if marker_index in ambiguous_marker:
            candidate: _Candidate | None = None
            if assigned < candidate_count and cost[marker_index, assigned] < _LARGE_COST:
                candidate = candidates[assigned]
            decisions.append(
                MarkerIndexingDecision(
                    key=key,
                    predicted_column_px=float(coordinates_px[marker_index, 0]),
                    predicted_row_px=float(coordinates_px[marker_index, 1]),
                    predicted_two_theta_rad=float(predicted_theta[marker_index]),
                    predicted_phi_rad=float(predicted_phi[marker_index]),
                    status=MarkerIndexingStatus.AMBIGUOUS_BLEND,
                    reason="candidate ownership or within-gate assignment is ambiguous",
                    observed_column_px=None if candidate is None else candidate.column_px,
                    observed_row_px=None if candidate is None else candidate.row_px,
                    observed_two_theta_rad=None if candidate is None else candidate.two_theta_rad,
                    observed_phi_rad=None if candidate is None else candidate.phi_rad,
                    covariance_px2=None if candidate is None else candidate.covariance_px2,
                    z_score=None if candidate is None else candidate.z_score,
                    assignment_cost=(
                        None if candidate is None else float(cost[marker_index, assigned])
                    ),
                    assignment_margin=(
                        0.0
                        if candidate is None or not np.isfinite(assignment_margin[marker_index])
                        else float(assignment_margin[marker_index])
                    ),
                )
            )
            continue
        if assigned >= candidate_count or cost[marker_index, assigned] >= _LARGE_COST:
            reason = (
                "no measured ridge passed the frozen local detection gate"
                if best_eligible.size == 0
                else "eligible measured ridge was uniquely owned by another marker"
            )
            decisions.append(
                _base_decision(
                    key,
                    coordinates_px[marker_index],
                    float(predicted_theta[marker_index]),
                    float(predicted_phi[marker_index]),
                    MarkerIndexingStatus.BELOW_DETECTION,
                    reason,
                )
            )
            continue
        candidate = candidates[assigned]
        margin = float(assignment_margin[marker_index])
        status = (
            MarkerIndexingStatus.VISIBLE_CONFIDENT
            if candidate.z_score >= policy.minimum_site_z
            and margin >= policy.minimum_assignment_margin
            else MarkerIndexingStatus.BELOW_DETECTION
        )
        reason = (
            "measured ridge passed cake detection, native confirmation, and unique assignment"
            if status == MarkerIndexingStatus.VISIBLE_CONFIDENT
            else "measured ridge did not pass the frozen site confidence threshold"
        )
        decisions.append(
            MarkerIndexingDecision(
                key=key,
                predicted_column_px=float(coordinates_px[marker_index, 0]),
                predicted_row_px=float(coordinates_px[marker_index, 1]),
                predicted_two_theta_rad=float(predicted_theta[marker_index]),
                predicted_phi_rad=float(predicted_phi[marker_index]),
                status=status,
                reason=reason,
                observed_column_px=candidate.column_px,
                observed_row_px=candidate.row_px,
                observed_two_theta_rad=candidate.two_theta_rad,
                observed_phi_rad=candidate.phi_rad,
                covariance_px2=candidate.covariance_px2,
                z_score=candidate.z_score,
                assignment_cost=float(cost[marker_index, assigned]),
                assignment_margin=margin,
            )
        )
    return tuple(decisions)


def _track_fit_rms_px(confident: tuple[MarkerIndexingDecision, ...]) -> float:
    integer_l = np.asarray([item.key.integer_L for item in confident], dtype=np.float64)
    centered_l = integer_l - float(np.mean(integer_l))
    offset = np.asarray(
        [
            (
                item.observed_column_px - item.predicted_column_px,
                item.observed_row_px - item.predicted_row_px,
            )
            for item in confident
        ]
    )
    design = np.zeros((2 * len(confident), 4), dtype=np.float64)
    target = np.reshape(offset, -1)
    weight = np.zeros((2 * len(confident), 2 * len(confident)), dtype=np.float64)
    for index, (l_value, item) in enumerate(zip(centered_l, confident, strict=True)):
        design[2 * index : 2 * index + 2] = (
            (1.0, l_value, 0.0, 0.0),
            (0.0, 0.0, 1.0, l_value),
        )
        covariance = np.asarray(item.covariance_px2, dtype=np.float64)
        weight[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = np.linalg.inv(covariance)
    normal = design.T @ weight @ design
    coefficient = np.linalg.lstsq(normal, design.T @ weight @ target, rcond=None)[0]
    residual = np.reshape(target - design @ coefficient, (-1, 2))
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def _paired_topology_failures(
    decisions: tuple[MarkerIndexingDecision, ...],
) -> set[_TrackKey]:
    paired: dict[tuple[int, int, int, tuple[int, int]], dict[int, MarkerIndexingDecision]] = {}
    for decision in decisions:
        if decision.status != MarkerIndexingStatus.VISIBLE_CONFIDENT:
            continue
        identity = (
            decision.key.family_m,
            decision.key.integer_L,
            decision.key.branch,
            decision.key.representative_rod_hk,
        )
        paired.setdefault(identity, {})[decision.key.root_sign] = decision
    failed: set[_TrackKey] = set()
    for sides in paired.values():
        if set(sides) != {-1, 1}:
            continue
        negative = sides[-1]
        positive = sides[1]
        predicted_chord = np.asarray(
            (
                positive.predicted_column_px - negative.predicted_column_px,
                positive.predicted_row_px - negative.predicted_row_px,
            )
        )
        observed_chord = np.asarray(
            (
                positive.observed_column_px - negative.observed_column_px,
                positive.observed_row_px - negative.observed_row_px,
            )
        )
        if float(predicted_chord @ observed_chord) <= 0.0:
            failed.update((negative.track_key, positive.track_key))
    return failed


def _track_decisions(
    image_id: str,
    decisions: tuple[MarkerIndexingDecision, ...],
    policy: PeakIndexingPolicy,
) -> tuple[BranchTrackDecision, ...]:
    topology_failures = _paired_topology_failures(decisions)
    groups: dict[_TrackKey, list[MarkerIndexingDecision]] = {}
    for decision in decisions:
        groups.setdefault(decision.track_key, []).append(decision)
    tracks: list[BranchTrackDecision] = []
    for key, group in sorted(groups.items()):
        confident = tuple(
            sorted(
                (item for item in group if item.status == MarkerIndexingStatus.VISIBLE_CONFIDENT),
                key=lambda item: item.key.integer_L,
            )
        )
        integer_l = tuple(item.key.integer_L for item in confident)
        accepted = len(set(integer_l)) >= policy.minimum_track_sites
        rms: float | None = None
        reason = "too few confident distinct-L sites"
        if accepted:
            predicted_theta = np.asarray([item.predicted_two_theta_rad for item in confident])
            observed_theta = np.asarray([item.observed_two_theta_rad for item in confident])
            predicted_order = np.sign(np.diff(predicted_theta))
            observed_order = np.sign(np.diff(observed_theta))
            if np.any(predicted_order == 0.0) or not np.array_equal(
                predicted_order,
                observed_order,
            ):
                accepted = False
                reason = "measured sites violate the predicted increasing-L cake order"
            else:
                rms = _track_fit_rms_px(confident)
                if rms > policy.maximum_track_rms_px:
                    accepted = False
                    reason = "measured track displacement is not geometrically coherent"
                else:
                    reason = "ordered distinct-L sites form a coherent measured branch track"
        if accepted and key in topology_failures:
            accepted = False
            reason = "paired measured roots reverse the predicted branch chord topology"
        tracks.append(
            BranchTrackDecision(
                family_m=key[0],
                branch=key[1],
                root_sign=key[2],
                representative_rod_hk=key[3],
                image_ids=(image_id,),
                accepted=accepted,
                confident_site_count=len(confident),
                distinct_integer_L=tuple(sorted(set(integer_l))),
                rms_px=rms,
                reason=reason,
            )
        )
    return tuple(tracks)


def index_measured_integer_l_branches(
    detector_native_counts: ArrayLike,
    prediction: IntegerLMarkerPrediction,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    image_id: str,
    incidence_angle_rad: float,
    reference_wavelength_A: float,
    detector_valid_mask: ArrayLike | None = None,
    detector_mask_revision: str | None = None,
    policy: PeakIndexingPolicy | None = None,
) -> MeasuredImageIndexingResult:
    """Detect measured ridges in local cake charts and index confident branch tracks."""

    if not isinstance(prediction, IntegerLMarkerPrediction):
        raise TypeError("prediction must be an IntegerLMarkerPrediction")
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    active_policy = PeakIndexingPolicy() if policy is None else policy
    if not isinstance(active_policy, PeakIndexingPolicy):
        raise TypeError("policy must be a PeakIndexingPolicy")
    supplied = np.asarray(detector_native_counts)
    if (
        supplied.shape != instrument.detector_shape_rc
        or np.iscomplexobj(supplied)
        or not np.all(np.isfinite(supplied))
        or np.any(supplied < 0.0)
    ):
        raise ValueError(
            "detector_native_counts must be finite, nonnegative, and match detector_shape_rc"
        )
    if detector_valid_mask is None:
        valid_mask = np.ones(supplied.shape, dtype=np.bool_)
        mask_revision = "all-valid-detector-mask.v1"
    else:
        mask = np.asarray(detector_valid_mask)
        if mask.dtype.kind != "b" or mask.shape != supplied.shape:
            raise ValueError("detector_valid_mask must be boolean with detector_shape_rc")
        valid_mask = np.asarray(mask, dtype=np.bool_)
        mask_revision = detector_mask_revision or "caller-supplied-bool-mask.v1"
    if detector_mask_revision is not None:
        if not isinstance(detector_mask_revision, str) or not detector_mask_revision.strip():
            raise ValueError("detector_mask_revision must be a nonempty string")
        mask_revision = detector_mask_revision
    valid_mask_u8 = np.asarray(valid_mask, dtype=np.uint8)

    ordering = np.argsort(np.asarray(prediction.keys, dtype=object), kind="stable")
    keys = tuple(prediction.keys[int(index)] for index in ordering)
    coordinates_px = np.asarray(prediction.coordinates_px[ordering], dtype=np.float64)
    active = np.asarray(prediction.active_panel[ordering], dtype=np.bool_)
    predicted_angles = detector_coordinates_to_angles(
        coordinates_px[:, 0],
        coordinates_px[:, 1],
        instrument=instrument,
        angle_frame=angle_frame,
    )
    projectable = active & predicted_angles.valid & predicted_angles.azimuth_valid
    all_candidates: list[_Candidate] = []
    chart_supported = np.zeros(len(keys), dtype=np.bool_)
    for marker_index in np.flatnonzero(projectable):
        candidates, supported = _local_angle_chart_candidates(
            supplied,
            valid_mask,
            valid_mask_u8,
            float(predicted_angles.two_theta_rad[marker_index]),
            float(predicted_angles.phi_rad[marker_index]),
            instrument=instrument,
            angle_frame=angle_frame,
            policy=active_policy,
        )
        chart_supported[marker_index] = supported
        all_candidates.extend(candidates)
    candidates = _merge_candidates(
        tuple(all_candidates),
        active_policy.candidate_merge_radius_px,
    )
    measured = _assigned_decisions(
        keys,
        coordinates_px,
        predicted_angles.two_theta_rad,
        predicted_angles.phi_rad,
        candidates,
        chart_supported,
        instrument=instrument,
        angle_frame=angle_frame,
        policy=active_policy,
    )
    decisions = []
    for marker_index, decision in enumerate(measured):
        if projectable[marker_index]:
            decisions.append(decision)
        else:
            decisions.append(
                _base_decision(
                    keys[marker_index],
                    coordinates_px[marker_index],
                    float(predicted_angles.two_theta_rad[marker_index]),
                    float(predicted_angles.phi_rad[marker_index]),
                    MarkerIndexingStatus.GEOMETRY_INVISIBLE,
                    "predicted tag is not an active projectable detector point",
                )
            )
    frozen_decisions = tuple(decisions)
    return MeasuredImageIndexingResult(
        image_id=image_id,
        incidence_angle_rad=incidence_angle_rad,
        reference_wavelength_A=reference_wavelength_A,
        marker_decisions=frozen_decisions,
        detector_data_hash=_array_hash(supplied),
        detector_mask_hash=_array_hash(valid_mask),
        detector_mask_revision=mask_revision,
        context_hash=_context_hash(instrument, angle_frame, prediction),
        policy=active_policy,
    )


def _cross_image_track_rms_px(
    evidence: tuple[tuple[MeasuredImageIndexingResult, BranchTrackDecision], ...],
    common_l: tuple[int, ...],
) -> float:
    centered_offsets: list[NDArray[np.float64]] = []
    for result, track in evidence:
        by_l = {
            decision.key.integer_L: decision
            for decision in result.marker_decisions
            if decision.track_key == track.track_key
            and decision.status == MarkerIndexingStatus.VISIBLE_CONFIDENT
        }
        offsets = np.asarray(
            [
                (
                    by_l[integer_l].observed_column_px - by_l[integer_l].predicted_column_px,
                    by_l[integer_l].observed_row_px - by_l[integer_l].predicted_row_px,
                )
                for integer_l in common_l
            ]
        )
        centered_offsets.append(offsets - np.mean(offsets, axis=0))
    if len(centered_offsets) < 2:
        return math.inf
    return max(
        float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))
        for left, right in combinations(centered_offsets, 2)
    )


def _has_repeated_track_geometry(
    evidence: tuple[tuple[MeasuredImageIndexingResult, BranchTrackDecision], ...],
    common_l: tuple[int, ...],
) -> bool:
    signatures: list[NDArray[np.float64]] = []
    for result, track in evidence:
        by_l = {
            decision.key.integer_L: decision
            for decision in result.marker_decisions
            if decision.track_key == track.track_key
            and decision.status == MarkerIndexingStatus.VISIBLE_CONFIDENT
        }
        coordinates = np.asarray(
            [
                (by_l[integer_l].predicted_column_px, by_l[integer_l].predicted_row_px)
                for integer_l in common_l
            ],
            dtype=np.float64,
        )
        signatures.append(coordinates)
    return any(
        np.allclose(left, right, rtol=0.0, atol=1.0e-9)
        for left, right in combinations(signatures, 2)
    )


def _qualifying_track_evidence(
    group: tuple[tuple[MeasuredImageIndexingResult, BranchTrackDecision], ...],
    policy: PeakIndexingPolicy,
) -> tuple[
    tuple[tuple[MeasuredImageIndexingResult, BranchTrackDecision], ...],
    tuple[int, ...],
    float | None,
]:
    accepted = tuple(item for item in group if item[1].accepted)
    qualifying_by_image: dict[
        str,
        tuple[MeasuredImageIndexingResult, BranchTrackDecision],
    ] = {}
    qualifying_l: set[int] = set()
    qualifying_rms: list[float] = []
    for size in range(len(accepted), policy.minimum_track_images - 1, -1):
        for evidence in combinations(accepted, size):
            incidences = tuple(item[0].incidence_angle_rad for item in evidence)
            if any(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
                for left, right in combinations(incidences, 2)
            ):
                continue
            detector_hashes = tuple(item[0].detector_data_hash for item in evidence)
            context_hashes = tuple(item[0].context_hash for item in evidence)
            if len(set(detector_hashes)) != size or len(set(context_hashes)) != size:
                continue
            common_l = tuple(
                sorted(set.intersection(*(set(item[1].distinct_integer_L) for item in evidence)))
            )
            if len(common_l) < policy.minimum_common_l_sites:
                continue
            if _has_repeated_track_geometry(tuple(evidence), common_l):
                continue
            cross_rms = _cross_image_track_rms_px(evidence, common_l)
            if cross_rms <= policy.maximum_track_rms_px:
                qualifying_by_image.update((item[0].image_id, item) for item in evidence)
                qualifying_l.update(common_l)
                qualifying_rms.append(cross_rms)
    if not qualifying_by_image:
        return (), (), None
    return (
        tuple(qualifying_by_image[image_id] for image_id in sorted(qualifying_by_image)),
        tuple(sorted(qualifying_l)),
        max(qualifying_rms),
    )


def _cross_image_tracks(
    results: tuple[MeasuredImageIndexingResult, ...],
    policy: PeakIndexingPolicy,
) -> tuple[BranchTrackDecision, ...]:
    groups: dict[
        _TrackKey,
        list[tuple[MeasuredImageIndexingResult, BranchTrackDecision]],
    ] = {}
    for result in results:
        for track in result.branch_tracks:
            groups.setdefault(track.track_key, []).append((result, track))
    tracks: list[BranchTrackDecision] = []
    for key, group in sorted(groups.items()):
        frozen_group = tuple(group)
        evidence, common_l, cross_rms = _qualifying_track_evidence(
            frozen_group,
            policy,
        )
        accepted = bool(evidence)
        selected = evidence or tuple(item for item in frozen_group if item[1].accepted)
        if accepted:
            integer_l = common_l
            site_count = sum(
                decision.status == MarkerIndexingStatus.VISIBLE_CONFIDENT
                and decision.track_key == key
                and decision.key.integer_L in common_l
                for result, _ in selected
                for decision in result.marker_decisions
            )
        else:
            site_count = sum(item[1].confident_site_count for item in selected)
            integer_l = tuple(
                sorted({value for item in selected for value in item[1].distinct_integer_L})
            )
        rms_values = tuple(item[1].rms_px for item in selected if item[1].rms_px is not None)
        if cross_rms is not None:
            rms_values = (*rms_values, cross_rms)
        tracks.append(
            BranchTrackDecision(
                family_m=key[0],
                branch=key[1],
                root_sign=key[2],
                representative_rod_hk=key[3],
                image_ids=tuple(item[0].image_id for item in (selected or frozen_group)),
                accepted=accepted,
                confident_site_count=site_count,
                distinct_integer_L=integer_l,
                rms_px=max(rms_values) if rms_values else None,
                reason=(
                    "coherent measured branch track repeats at distinct incidences "
                    f"with shared L={common_l}"
                    if accepted
                    else "branch track lacks distinct-incidence replication with shared L anchors"
                ),
            )
        )
    return tuple(tracks)


def select_confident_branch_tracks(
    image_results: tuple[MeasuredImageIndexingResult, ...],
    *,
    policy: PeakIndexingPolicy | None = None,
) -> MeasuredIndexingResult:
    """Keep only coherent local tracks replicated across the measured angle series."""

    results = tuple(image_results)
    if not results:
        raise ValueError("image_results must not be empty")
    active_policy = results[0].policy if policy is None else policy
    if not isinstance(active_policy, PeakIndexingPolicy):
        raise TypeError("policy must be a PeakIndexingPolicy")
    if any(item.policy != active_policy for item in results):
        raise ValueError("all image results must share the selected policy")
    return MeasuredIndexingResult(results, active_policy)
