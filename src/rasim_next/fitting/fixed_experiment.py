"""Immutable position handoff and exact fixed-experiment series rebuild."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np

from painted_ewald import MosaicBraggSpace, MosaicParameters
from rasim_next.core.contracts import (
    canonical_revision_sha256,
    incidence_scan_calibration_binding_revision,
)
from rasim_next.fitting.fixed_lattice import FixedLatticeState
from rasim_next.fitting.indexed_series import (
    DETECTOR_CALIBRATION_PARAMETER_NAMES,
    SHARED_GEOMETRY_PARAMETER_NAMES,
    DetectorCalibrationCorrections,
    SharedGeometryCorrections,
    apply_detector_calibration_corrections,
    apply_shared_geometry_corrections,
    zero_sum_helmert_basis,
)
from rasim_next.geometry import build_incident_states
from rasim_next.materials import read_crystal
from rasim_next.pipeline.configured_simulation import (
    ConfiguredSimulationInputs,
    SimulationConfiguration,
    build_configured_simulation_inputs,
    rebind_configured_simulation_instrument,
)
from rasim_next.pipeline.source_averaged_detector import _detector_native_chart_revision

FIXED_EXPERIMENT_STATE_SCHEMA_VERSION = "rasim-fixed-experiment-state-v2"
FIXED_MOSAIC_STATE_SCHEMA_VERSION = "rasim-fixed-mosaic-state-v1"
POSITION_FIT_RESULT_SCHEMA_VERSION = "rasim-osc-geometry-fit-result-v6"

_REQUIRED_POSITION_FIELDS = frozenset(
    {
        "position_artifact_revision",
        "corrections",
        "incidence_angle_model_id",
        "incidence_angle_delta_rad",
        "commanded_incidence_angles_deg",
        "effective_incidence_angles_deg",
        "beam_center_column_row_px",
        "geometry_parameters_fitted_here",
    }
)
_TRIM_POSITION_FIELDS = frozenset(
    {
        "incidence_angle_image_ids",
        "incidence_angle_trim_by_image_id_rad",
        "incidence_angle_trim_contrast_rad",
        "incidence_angle_trim_prior_sigma_rad",
        "incidence_angle_trim_contrast_half_span_rad",
    }
)
_CALIBRATION_POSITION_FIELDS = frozenset(
    {"detector_calibration_active", "detector_plane_normal_offset_m"}
)


@dataclass(frozen=True, slots=True)
class FixedMosaicState:
    """One explicitly provided continuous mosaic distribution."""

    gaussian_sigma_deg: float
    lorentzian_hwhm_deg: float
    lorentzian_probability: float
    provenance: str

    def __post_init__(self) -> None:
        gaussian = float(self.gaussian_sigma_deg)
        lorentzian = float(self.lorentzian_hwhm_deg)
        probability = float(self.lorentzian_probability)
        provenance = self.provenance
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("mosaic provenance must be a nonempty string")
        parameters = MosaicParameters(
            gaussian_sigma_rad=math.radians(gaussian),
            lorentzian_half_width_rad=math.radians(lorentzian),
            lorentzian_probability=probability,
        )
        if parameters.zero_tilt_probability_mass != 0.0:
            raise ValueError("the continuous detector requires zero mosaic atom mass")
        object.__setattr__(self, "gaussian_sigma_deg", gaussian)
        object.__setattr__(self, "lorentzian_hwhm_deg", lorentzian)
        object.__setattr__(self, "lorentzian_probability", probability)
        object.__setattr__(self, "provenance", provenance.strip())

    @classmethod
    def from_record(cls, record: object) -> Self:
        if not isinstance(record, Mapping) or set(record) != {
            "schema_version",
            "status",
            "gaussian_sigma_deg",
            "lorentzian_hwhm_deg",
            "lorentzian_probability",
            "provenance",
        }:
            raise ValueError("provided mosaic record has an invalid shape")
        if (
            record["schema_version"] != FIXED_MOSAIC_STATE_SCHEMA_VERSION
            or record["status"] != "PROVIDED_MOSAIC_PRIOR"
        ):
            raise ValueError("unsupported provided mosaic state")
        return cls(
            gaussian_sigma_deg=float(record["gaussian_sigma_deg"]),
            lorentzian_hwhm_deg=float(record["lorentzian_hwhm_deg"]),
            lorentzian_probability=float(record["lorentzian_probability"]),
            provenance=record["provenance"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": FIXED_MOSAIC_STATE_SCHEMA_VERSION,
            "status": "PROVIDED_MOSAIC_PRIOR",
            "gaussian_sigma_deg": self.gaussian_sigma_deg,
            "lorentzian_hwhm_deg": self.lorentzian_hwhm_deg,
            "lorentzian_probability": self.lorentzian_probability,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class FixedPositionState:
    """One immutable position fit shared by every downstream incidence view."""

    artifact_revision: str
    corrections: SharedGeometryCorrections
    incidence_angle_delta_rad: float
    commanded_incidence_angles_rad: tuple[float, ...]
    beam_center_column_row_px: tuple[float, float]
    detector_calibration_active: bool = False
    detector_plane_normal_offset_m: float = 0.0
    incidence_angle_image_ids: tuple[str, ...] = ()
    incidence_angle_trim_rad: tuple[float, ...] = ()
    incidence_angle_trim_contrast_rad: tuple[float, ...] = ()
    incidence_angle_trim_prior_sigma_rad: float | None = None
    incidence_angle_trim_contrast_half_span_rad: float | None = None

    def __post_init__(self) -> None:
        revision = str(self.artifact_revision)
        if (
            not revision.startswith("sha256-")
            or len(revision) != 71
            or any(character not in "0123456789abcdef" for character in revision[7:])
        ):
            raise ValueError("artifact_revision must be a sha256-prefixed digest")
        if not isinstance(self.corrections, SharedGeometryCorrections):
            raise TypeError("corrections must be SharedGeometryCorrections")

        delta = float(self.incidence_angle_delta_rad)
        commanded = tuple(float(value) for value in self.commanded_incidence_angles_rad)
        beam_center = tuple(float(value) for value in self.beam_center_column_row_px)
        calibration_active = self.detector_calibration_active
        detector_offset = float(self.detector_plane_normal_offset_m)
        supplied_image_ids = tuple(self.incidence_angle_image_ids)
        if any(not isinstance(value, str) or not value.strip() for value in supplied_image_ids):
            raise ValueError("incidence-angle image IDs must be nonempty strings")
        image_ids = tuple(supplied_image_ids)
        trims = tuple(float(value) for value in self.incidence_angle_trim_rad)
        contrasts = tuple(float(value) for value in self.incidence_angle_trim_contrast_rad)
        prior_sigma = (
            None
            if self.incidence_angle_trim_prior_sigma_rad is None
            else float(self.incidence_angle_trim_prior_sigma_rad)
        )
        contrast_span = (
            None
            if self.incidence_angle_trim_contrast_half_span_rad is None
            else float(self.incidence_angle_trim_contrast_half_span_rad)
        )

        if not math.isfinite(delta):
            raise ValueError("incidence_angle_delta_rad must be finite")
        if not commanded or any(not math.isfinite(value) for value in commanded):
            raise ValueError("commanded incidence angles must be finite and nonempty")
        if len(beam_center) != 2 or any(not math.isfinite(value) for value in beam_center):
            raise ValueError("beam center must contain two finite detector coordinates")
        if not isinstance(calibration_active, bool):
            raise TypeError("detector_calibration_active must be bool")
        if not math.isfinite(detector_offset):
            raise ValueError("detector plane-normal offset must be finite")
        if not calibration_active and detector_offset != 0.0:
            raise ValueError("detector distance requires explicit detector-calibration provenance")

        if image_ids:
            if (
                len(image_ids) != len(commanded)
                or len(set(image_ids)) != len(image_ids)
                or any(not value for value in image_ids)
                or len(trims) != len(commanded)
                or len(contrasts) != len(commanded) - 1
                or any(not math.isfinite(value) for value in (*trims, *contrasts))
                or not math.isclose(math.fsum(trims), 0.0, rel_tol=0.0, abs_tol=1.0e-14)
                or prior_sigma is None
                or not math.isfinite(prior_sigma)
                or prior_sigma <= 0.0
                or contrast_span is None
                or not math.isfinite(contrast_span)
                or contrast_span <= 0.0
            ):
                raise ValueError("incidence-angle trim state is invalid")
            expected_canonical_trims = zero_sum_helmert_basis(len(image_ids)) @ np.asarray(
                contrasts,
                dtype=np.float64,
            )
            trim_by_image_id = dict(zip(image_ids, trims, strict=True))
            canonical_trims = tuple(trim_by_image_id[image_id] for image_id in sorted(image_ids))
            if not np.allclose(
                canonical_trims,
                expected_canonical_trims,
                rtol=0.0,
                atol=2.0e-14,
            ):
                raise ValueError("incidence-angle trims contradict their Helmert coordinates")
            if max(map(abs, contrasts)) > contrast_span + 1.0e-15:
                raise ValueError("incidence-angle trim contrast exceeds its fitted bounds")
        elif trims or contrasts or prior_sigma is not None or contrast_span is not None:
            raise ValueError("common-delta position state cannot contain trim fields")

        object.__setattr__(self, "artifact_revision", revision)
        object.__setattr__(self, "incidence_angle_delta_rad", delta)
        object.__setattr__(self, "commanded_incidence_angles_rad", commanded)
        object.__setattr__(self, "beam_center_column_row_px", beam_center)
        object.__setattr__(self, "detector_calibration_active", calibration_active)
        object.__setattr__(self, "detector_plane_normal_offset_m", detector_offset)
        object.__setattr__(self, "incidence_angle_image_ids", image_ids)
        object.__setattr__(self, "incidence_angle_trim_rad", trims)
        object.__setattr__(self, "incidence_angle_trim_contrast_rad", contrasts)
        object.__setattr__(self, "incidence_angle_trim_prior_sigma_rad", prior_sigma)
        object.__setattr__(
            self,
            "incidence_angle_trim_contrast_half_span_rad",
            contrast_span,
        )

    @property
    def effective_incidence_angles_rad(self) -> tuple[float, ...]:
        trims = self.incidence_angle_trim_rad or (0.0,) * len(self.commanded_incidence_angles_rad)
        return tuple(
            commanded + self.incidence_angle_delta_rad + trim
            for commanded, trim in zip(
                self.commanded_incidence_angles_rad,
                trims,
                strict=True,
            )
        )

    @classmethod
    def from_record(cls, record: object) -> Self:
        """Validate the JSON handoff and convert degree fields at the boundary."""

        if not isinstance(record, Mapping):
            raise ValueError("fixed-position record must be a mapping")
        fields = frozenset(record)
        if (
            not _REQUIRED_POSITION_FIELDS.issubset(fields)
            or fields
            - _REQUIRED_POSITION_FIELDS
            - _TRIM_POSITION_FIELDS
            - _CALIBRATION_POSITION_FIELDS
            or record["geometry_parameters_fitted_here"] is not False
        ):
            raise ValueError("fixed-position record has an invalid shape")

        correction_record = record["corrections"]
        if not isinstance(correction_record, Mapping) or set(correction_record) != set(
            SHARED_GEOMETRY_PARAMETER_NAMES
        ):
            raise ValueError("fixed-position corrections are invalid")

        model_id = record["incidence_angle_model_id"]
        if model_id == "commanded_angle_plus_common_delta.v1":
            if fields & _TRIM_POSITION_FIELDS:
                raise ValueError("common-delta record cannot contain trim fields")
            image_ids: tuple[str, ...] = ()
            trims: tuple[float, ...] = ()
            contrasts: tuple[float, ...] = ()
            prior_sigma = None
            contrast_span = None
        elif model_id == "commanded_plus_common_delta_plus_zero_sum_trim.helmert.v1":
            if not _TRIM_POSITION_FIELDS.issubset(fields):
                raise ValueError("trimmed position record is incomplete")
            supplied_image_ids = record["incidence_angle_image_ids"]
            if not isinstance(supplied_image_ids, (list, tuple)) or any(
                not isinstance(value, str) or not value.strip() for value in supplied_image_ids
            ):
                raise ValueError("position image IDs must be nonempty strings")
            image_ids = tuple(supplied_image_ids)
            trim_by_id = record["incidence_angle_trim_by_image_id_rad"]
            if not isinstance(trim_by_id, Mapping) or set(trim_by_id) != set(image_ids):
                raise ValueError("position trim mapping is invalid")
            trims = tuple(float(trim_by_id[image_id]) for image_id in image_ids)
            contrasts = tuple(float(value) for value in record["incidence_angle_trim_contrast_rad"])
            prior_sigma = float(record["incidence_angle_trim_prior_sigma_rad"])
            contrast_span = float(record["incidence_angle_trim_contrast_half_span_rad"])
        else:
            raise ValueError("unsupported incidence-angle model")

        state = cls(
            artifact_revision=str(record["position_artifact_revision"]),
            corrections=SharedGeometryCorrections.from_array(
                [float(correction_record[name]) for name in SHARED_GEOMETRY_PARAMETER_NAMES]
            ),
            incidence_angle_delta_rad=float(record["incidence_angle_delta_rad"]),
            commanded_incidence_angles_rad=tuple(
                math.radians(float(value)) for value in record["commanded_incidence_angles_deg"]
            ),
            beam_center_column_row_px=tuple(
                float(value) for value in record["beam_center_column_row_px"]
            ),
            detector_calibration_active=record.get("detector_calibration_active", False),
            detector_plane_normal_offset_m=float(record.get("detector_plane_normal_offset_m", 0.0)),
            incidence_angle_image_ids=image_ids,
            incidence_angle_trim_rad=trims,
            incidence_angle_trim_contrast_rad=contrasts,
            incidence_angle_trim_prior_sigma_rad=prior_sigma,
            incidence_angle_trim_contrast_half_span_rad=contrast_span,
        )
        recorded_effective = tuple(
            math.radians(float(value)) for value in record["effective_incidence_angles_deg"]
        )
        if len(recorded_effective) != len(state.effective_incidence_angles_rad) or any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=2.0e-14)
            for observed, expected in zip(
                recorded_effective,
                state.effective_incidence_angles_rad,
                strict=True,
            )
        ):
            raise ValueError("effective incidence angles contradict the shared-angle model")
        return state

    def to_record(self) -> dict[str, object]:
        """Serialize the existing artifact contract without changing its schema."""

        record: dict[str, object] = {
            "position_artifact_revision": self.artifact_revision,
            "corrections": {
                name: float(value)
                for name, value in zip(
                    SHARED_GEOMETRY_PARAMETER_NAMES,
                    self.corrections.as_array(),
                    strict=True,
                )
            },
            "incidence_angle_model_id": (
                "commanded_plus_common_delta_plus_zero_sum_trim.helmert.v1"
                if self.incidence_angle_image_ids
                else "commanded_angle_plus_common_delta.v1"
            ),
            "incidence_angle_delta_rad": self.incidence_angle_delta_rad,
            "commanded_incidence_angles_deg": [
                math.degrees(value) for value in self.commanded_incidence_angles_rad
            ],
            "effective_incidence_angles_deg": [
                math.degrees(value) for value in self.effective_incidence_angles_rad
            ],
            "beam_center_column_row_px": list(self.beam_center_column_row_px),
            "geometry_parameters_fitted_here": False,
        }
        if self.incidence_angle_image_ids:
            record.update(
                {
                    "incidence_angle_image_ids": list(self.incidence_angle_image_ids),
                    "incidence_angle_trim_by_image_id_rad": dict(
                        zip(
                            self.incidence_angle_image_ids,
                            self.incidence_angle_trim_rad,
                            strict=True,
                        )
                    ),
                    "incidence_angle_trim_contrast_rad": list(
                        self.incidence_angle_trim_contrast_rad
                    ),
                    "incidence_angle_trim_prior_sigma_rad": (
                        self.incidence_angle_trim_prior_sigma_rad
                    ),
                    "incidence_angle_trim_contrast_half_span_rad": (
                        self.incidence_angle_trim_contrast_half_span_rad
                    ),
                }
            )
        if self.detector_calibration_active:
            record["detector_calibration_active"] = True
        if self.detector_plane_normal_offset_m != 0.0:
            record["detector_plane_normal_offset_m"] = self.detector_plane_normal_offset_m
        return record


def fixed_position_from_fit_record(
    record: object,
    *,
    expected_manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> tuple[FixedPositionState, str, str]:
    """Validate the raw geometry-fit artifact used by every downstream stage."""

    expected_path = Path(expected_manifest_path).resolve()
    try:
        if not isinstance(record, Mapping):
            raise ValueError("artifact is not a mapping")
        fit = record["fit"]
        qualification = record["qualification"]
        if not isinstance(fit, Mapping) or not isinstance(qualification, Mapping):
            raise ValueError("fit evidence is malformed")
        parameter_names = fit["jacobian_parameter_names"]
        if not isinstance(parameter_names, (list, tuple)) or not parameter_names:
            raise ValueError("Jacobian parameter names are missing")
        if (
            record.get("schema") != POSITION_FIT_RESULT_SCHEMA_VERSION
            or Path(str(record.get("manifest_path", ""))).resolve() != expected_path
            or record.get("manifest_sha256") != expected_manifest_sha256
            or record.get("run_completed") is not True
            or record.get("root_audit", {}).get("classification") != "SAME"
            or record.get("outer_audit", {}).get("classification") != "SAME"
            or fit.get("success") is not True
            or int(fit.get("jacobian_rank", -1)) != len(parameter_names)
            or fit.get("incidence_angle_delta_fitted") is not True
        ):
            raise ValueError("position evidence is incomplete or rank deficient")

        requested = qualification.get("requested")
        if requested is True:
            if qualification.get("accepted") is not True:
                raise ValueError("requested position qualification was not accepted")
            status = "POSITION_QUALIFIED"
        elif requested is False:
            status = "POSITION_MODEL_LIMITED"
        else:
            raise ValueError("position qualification request state is invalid")

        position = FixedPositionState.from_record(record["fixed_position"])
        if not math.isclose(
            position.incidence_angle_delta_rad,
            float(fit["incidence_angle_delta_rad"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("fixed position and fitted common incidence delta differ")

        fitted_corrections = fit["corrections"]
        if not isinstance(fitted_corrections, Mapping) or set(fitted_corrections) != set(
            SHARED_GEOMETRY_PARAMETER_NAMES
        ):
            raise ValueError("fitted geometry corrections are malformed")
        if not np.array_equal(
            position.corrections.as_array(),
            np.asarray(
                [float(fitted_corrections[name]) for name in SHARED_GEOMETRY_PARAMETER_NAMES],
                dtype=np.float64,
            ),
        ):
            raise ValueError("fixed position and fitted geometry corrections differ")

        fitted_calibration = fit.get("detector_calibration_corrections")
        if fitted_calibration is not None:
            if not position.detector_calibration_active:
                raise ValueError("fixed position omitted detector-calibration provenance")
            if not isinstance(fitted_calibration, Mapping) or set(fitted_calibration) != set(
                DETECTOR_CALIBRATION_PARAMETER_NAMES
            ):
                raise ValueError("fitted detector calibration corrections are malformed")
            calibration = DetectorCalibrationCorrections.from_array(
                [float(fitted_calibration[name]) for name in DETECTOR_CALIBRATION_PARAMETER_NAMES]
            )
            configured_center = record.get("configured_detector_reference_coordinate_px")
            if not isinstance(configured_center, (list, tuple)) or len(configured_center) != 2:
                raise ValueError("configured detector reference is missing")
            expected_center = tuple(
                float(value) + offset
                for value, offset in zip(
                    configured_center,
                    calibration.as_array()[:2],
                    strict=True,
                )
            )
            if position.beam_center_column_row_px != expected_center or (
                position.detector_plane_normal_offset_m
                != calibration.detector_plane_normal_offset_m
            ):
                raise ValueError("fixed position and fitted detector calibration differ")
        else:
            if position.detector_calibration_active:
                raise ValueError("legacy fit cannot authorize detector calibration")
            if position.detector_plane_normal_offset_m != 0.0:
                raise ValueError("legacy fixed position cannot introduce detector distance")
            configured_center = record.get("configured_detector_reference_coordinate_px")
            if configured_center is not None:
                if not isinstance(configured_center, (list, tuple)) or len(configured_center) != 2:
                    raise ValueError("configured detector reference is malformed")
                if position.beam_center_column_row_px != tuple(
                    float(value) for value in configured_center
                ):
                    raise ValueError("legacy fixed position changed the detector reference")

        fitted_trims = fit["incidence_angle_trim_by_image_id_rad"]
        if not isinstance(fitted_trims, Mapping):
            raise ValueError("fitted incidence trims are malformed")
        if position.incidence_angle_image_ids:
            expected_trims = dict(
                zip(
                    position.incidence_angle_image_ids,
                    position.incidence_angle_trim_rad,
                    strict=True,
                )
            )
            if fitted_trims != expected_trims:
                raise ValueError("fixed position and fitted incidence trims differ")
        elif any(float(value) != 0.0 for value in fitted_trims.values()):
            raise ValueError("common-delta position contains nonzero incidence trims")

        selection = record["indexed_manifest_hash"]
        if not isinstance(selection, str) or not selection:
            raise ValueError("indexed selection identity is missing")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"position result is not admissible: {exc}") from exc
    return position, status, selection


INCIDENCE_SCAN_CALIBRATION_MODEL_ID = "common_delta_only_incidence_scan_calibration.v1"
_FIXED_INCIDENCE_SCAN_BUILDER_TOKEN = object()


@dataclass(frozen=True, slots=True)
class FixedIncidenceScanSeries:
    """Calibrated incidence nodes and their complete corrected experiments."""

    inputs: tuple[ConfiguredSimulationInputs, ...]
    commanded_incidence_angles_rad: tuple[float, ...]
    effective_incidence_angles_rad: tuple[float, ...]
    calibration_model_id: str
    position_artifact_revision: str
    incidence_angle_delta_rad: float
    _builder_token: InitVar[object | None] = None
    component_sample_geometry_revision: tuple[str, ...] = field(init=False)
    detector_panel_revision: str = field(init=False)
    source_revision: str = field(init=False)
    source_state_count: int = field(init=False)
    scan_calibration_revision: str = field(init=False)
    scan_calibration_binding_revision: str = field(init=False)

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _FIXED_INCIDENCE_SCAN_BUILDER_TOKEN:
            raise TypeError("FixedIncidenceScanSeries must be built by its calibrated builder")
        inputs = tuple(self.inputs)
        commanded = tuple(float(value) for value in self.commanded_incidence_angles_rad)
        effective = tuple(float(value) for value in self.effective_incidence_angles_rad)
        if (
            not inputs
            or len(inputs) != len(commanded)
            or len(inputs) != len(effective)
            or any(not math.isfinite(value) for value in (*commanded, *effective))
            or any(stop <= start for start, stop in pairwise(commanded))
            or any(stop <= start for start, stop in pairwise(effective))
        ):
            raise ValueError("fixed incidence scan nodes must be finite, aligned, and increasing")
        if self.calibration_model_id != INCIDENCE_SCAN_CALIBRATION_MODEL_ID:
            raise ValueError("unsupported incidence-scan calibration model")
        position_revision = self.position_artifact_revision
        if (
            not isinstance(position_revision, str)
            or not position_revision.startswith("sha256-")
            or len(position_revision) != 71
            or any(character not in "0123456789abcdef" for character in position_revision[7:])
        ):
            raise ValueError("position_artifact_revision must identify the fixed position")
        delta = float(self.incidence_angle_delta_rad)
        if not math.isfinite(delta):
            raise ValueError("incidence_angle_delta_rad must be finite")
        calibrated_effective = tuple(value + delta for value in commanded)
        if effective != calibrated_effective:
            raise ValueError("the common-delta calibration requires one shared angle offset")
        if any(len(item.config.instrument.axis_rotations) != 1 for item in inputs):
            raise ValueError("fixed incidence scan nodes require one incidence axis")
        configured_effective = tuple(
            math.radians(item.config.instrument.axis_rotations[0].angle_deg) for item in inputs
        )
        if not np.allclose(configured_effective, effective, rtol=0.0, atol=2.0e-14):
            raise ValueError("configured scan poses do not match the effective angle axis")
        for item in inputs:
            if (
                item.incident.states.sample_geometry_revision
                != item.instrument.sample_geometry_revision
            ):
                raise ValueError("configured scan incident and instrument geometry disagree")
        source_revision = inputs[0].samples.source_revision
        source_state_count = int(inputs[0].samples.incident_sample_id.size)
        if any(item.samples.source_revision != source_revision for item in inputs[1:]):
            raise ValueError("fixed incidence scan nodes must share one source realization")
        if any(item.samples.incident_sample_id.size != source_state_count for item in inputs[1:]):
            raise ValueError("fixed incidence scan nodes must share one source-state count")
        component_geometry_revision = tuple(
            item.incident.states.sample_geometry_revision for item in inputs
        )
        if len(set(component_geometry_revision)) != len(component_geometry_revision):
            raise ValueError("fixed incidence scan nodes must have distinct sample poses")
        panel_revisions = tuple(_detector_native_chart_revision(item.instrument) for item in inputs)
        if len(set(panel_revisions)) != 1:
            raise ValueError("fixed incidence scan nodes must share one detector-native chart")
        panel_revision = panel_revisions[0]
        if any(item.scan_calibration_binding_revision is not None for item in inputs):
            raise ValueError("fixed incidence scan inputs must not be pre-stamped")
        if any(item.calibrated_incidence_axis_angle_rad is not None for item in inputs):
            raise ValueError("fixed incidence scan inputs must not carry a calibrated angle")
        revision = canonical_revision_sha256(
            ("definition_id", "fixed_incidence_scan_calibration.v1"),
            ("calibration_model_id", self.calibration_model_id),
            ("position_artifact_revision", position_revision),
            ("incidence_angle_delta_rad", delta),
            ("commanded_incidence_angles_rad", np.asarray(commanded, dtype=np.float64)),
            ("effective_incidence_angles_rad", np.asarray(effective, dtype=np.float64)),
            ("source_revision", source_revision),
            ("source_state_count", source_state_count),
            ("component_sample_geometry_revision", component_geometry_revision),
            ("detector_panel_revision", panel_revision),
        )
        binding_revision = incidence_scan_calibration_binding_revision(
            scan_calibration_revision=revision,
            component_sample_geometry_revision=component_geometry_revision,
            component_incidence_axis_angle_rad=configured_effective,
            effective_incidence_angle_rad=effective,
            detector_panel_revision=panel_revision,
            source_revision=source_revision,
            source_state_count=source_state_count,
        )
        stamped_inputs_list: list[ConfiguredSimulationInputs] = []
        for item, configured_angle in zip(inputs, configured_effective, strict=True):
            stamped = replace(item)
            object.__setattr__(stamped, "scan_calibration_binding_revision", binding_revision)
            object.__setattr__(
                stamped,
                "calibrated_incidence_axis_angle_rad",
                configured_angle,
            )
            stamped_inputs_list.append(stamped)
        stamped_inputs = tuple(stamped_inputs_list)
        object.__setattr__(self, "inputs", stamped_inputs)
        object.__setattr__(self, "commanded_incidence_angles_rad", commanded)
        object.__setattr__(self, "effective_incidence_angles_rad", effective)
        object.__setattr__(self, "incidence_angle_delta_rad", delta)
        object.__setattr__(
            self,
            "component_sample_geometry_revision",
            component_geometry_revision,
        )
        object.__setattr__(self, "scan_calibration_revision", revision)
        object.__setattr__(self, "detector_panel_revision", panel_revision)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_state_count", source_state_count)
        object.__setattr__(
            self,
            "scan_calibration_binding_revision",
            binding_revision,
        )


def _build_fixed_experiment_series_at_effective_angles(
    config: SimulationConfiguration,
    *,
    position: FixedPositionState,
    fixed_lattice: FixedLatticeState,
    effective_incidence_angles_rad: tuple[float, ...],
    source_sample_count: int,
    gaussian_sigma_rad: float,
    lorentzian_half_width_rad: float,
    lorentzian_probability: float,
) -> tuple[ConfiguredSimulationInputs, ...]:
    """Build corrected configured models at explicit calibrated incidence angles."""

    if not isinstance(position, FixedPositionState):
        raise TypeError("position must be FixedPositionState")
    if not isinstance(fixed_lattice, FixedLatticeState):
        raise TypeError("fixed_lattice must be FixedLatticeState")
    if (
        isinstance(source_sample_count, bool)
        or not isinstance(source_sample_count, int)
        or source_sample_count < 1
    ):
        raise ValueError("source_sample_count must be a positive integer")
    effective_angles = tuple(float(value) for value in effective_incidence_angles_rad)
    if not effective_angles or any(not math.isfinite(value) for value in effective_angles):
        raise ValueError("effective incidence angles must be finite and nonempty")
    if len(config.instrument.axis_rotations) != 1:
        raise ValueError("fixed-experiment series currently requires one incidence axis")
    reference_crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    if not np.array_equal(
        fixed_lattice.reference_direct_basis_A,
        reference_crystal.direct_basis_A,
    ):
        raise ValueError("fixed lattice reference differs from the configured CIF")

    mosaic = MosaicParameters(
        gaussian_sigma_rad=float(gaussian_sigma_rad),
        lorentzian_half_width_rad=float(lorentzian_half_width_rad),
        lorentzian_probability=float(lorentzian_probability),
        alpha_panel_count=config.mosaic.alpha_panel_count,
        alpha_gauss_order=config.mosaic.alpha_gauss_order,
        azimuth_count=config.mosaic.azimuth_count,
        azimuth_phase_rad=math.radians(config.mosaic.azimuth_phase_deg),
    )
    base_config = replace(
        config,
        source=replace(config.source, sample_count=source_sample_count),
        mosaic=replace(
            config.mosaic,
            gaussian_sigma_deg=math.degrees(mosaic.gaussian_sigma_rad),
            lorentzian_hwhm_deg=math.degrees(mosaic.lorentzian_half_width_rad),
            lorentzian_probability=mosaic.lorentzian_probability,
        ),
    )
    base_inputs = build_configured_simulation_inputs(
        base_config,
        direct_basis_A=fixed_lattice.direct_basis_override_A,
    )
    if not np.array_equal(
        base_inputs.crystal.direct_basis_A,
        fixed_lattice.active_direct_basis_A,
    ):
        raise ValueError("rebuilt crystal differs from the adopted lattice")

    configured_center = tuple(
        float(value) for value in base_inputs.instrument.detector_reference_coordinate_px
    )
    if not position.detector_calibration_active and (
        position.beam_center_column_row_px != configured_center
        or position.detector_plane_normal_offset_m != 0.0
    ):
        raise ValueError(
            "legacy fixed position cannot change detector calibration without provenance"
        )

    series: list[ConfiguredSimulationInputs] = []
    for incidence_rad in effective_angles:
        angle_config = replace(
            base_config,
            instrument=replace(
                base_config.instrument,
                axis_rotations=(
                    replace(
                        base_config.instrument.axis_rotations[0],
                        angle_deg=math.degrees(incidence_rad),
                    ),
                ),
            ),
        )
        inputs = rebind_configured_simulation_instrument(base_inputs, angle_config)

        configured_center = tuple(
            float(value) for value in inputs.instrument.detector_reference_coordinate_px
        )
        calibrated_instrument = apply_detector_calibration_corrections(
            inputs.instrument,
            DetectorCalibrationCorrections(
                detector_reference_column_offset_px=(
                    position.beam_center_column_row_px[0] - configured_center[0]
                ),
                detector_reference_row_offset_px=(
                    position.beam_center_column_row_px[1] - configured_center[1]
                ),
                detector_plane_normal_offset_m=position.detector_plane_normal_offset_m,
            ),
        )
        instrument = apply_shared_geometry_corrections(
            calibrated_instrument,
            angle_config.instrument.axis_rotations,
            position.corrections,
        )
        incident = build_incident_states(inputs.samples, inputs.material, instrument)
        if incident.states.incident_state_id.size != source_sample_count or not bool(
            np.all(incident.states.valid)
        ):
            raise RuntimeError("every fixed-experiment source state must be valid")

        bragg_config = replace(
            inputs.bragg_space.config,
            crystal_to_sample=instrument.sample_from_crystal.rotation,
        )
        series.append(
            replace(
                inputs,
                instrument=instrument,
                incident=incident,
                bragg_space=MosaicBraggSpace(bragg_config, inputs.strength),
                commanded_instrument_rebindable=False,
            )
        )

    result = tuple(series)
    reference_samples = result[0].samples
    for inputs in result[1:]:
        if (
            inputs.samples.source_revision != reference_samples.source_revision
            or not np.array_equal(inputs.samples.origin_lab_m, reference_samples.origin_lab_m)
            or not np.array_equal(inputs.samples.direction_lab, reference_samples.direction_lab)
            or not np.array_equal(inputs.samples.wavelength_A, reference_samples.wavelength_A)
            or not np.array_equal(inputs.samples.source_weight, reference_samples.source_weight)
        ):
            raise RuntimeError("incidence views do not share one source realization")
    return result


def build_fixed_experiment_series(
    config: SimulationConfiguration,
    *,
    position: FixedPositionState,
    fixed_lattice: FixedLatticeState,
    source_sample_count: int,
    gaussian_sigma_rad: float,
    lorentzian_half_width_rad: float,
    lorentzian_probability: float,
) -> tuple[ConfiguredSimulationInputs, ...]:
    """Build one corrected configured model per fixed effective incidence angle."""

    return _build_fixed_experiment_series_at_effective_angles(
        config,
        position=position,
        fixed_lattice=fixed_lattice,
        effective_incidence_angles_rad=position.effective_incidence_angles_rad,
        source_sample_count=source_sample_count,
        gaussian_sigma_rad=gaussian_sigma_rad,
        lorentzian_half_width_rad=lorentzian_half_width_rad,
        lorentzian_probability=lorentzian_probability,
    )


def build_fixed_incidence_scan_series(
    config: SimulationConfiguration,
    *,
    position: FixedPositionState,
    fixed_lattice: FixedLatticeState,
    commanded_incidence_angles_rad: tuple[float, ...],
    source_sample_count: int,
    gaussian_sigma_rad: float,
    lorentzian_half_width_rad: float,
    lorentzian_probability: float,
    calibration_model_id: str = INCIDENCE_SCAN_CALIBRATION_MODEL_ID,
) -> FixedIncidenceScanSeries:
    """Build scan nodes using the fitted common delta, never image-specific trims.

    The discrete fitted-image trims have no continuous interpolation.  This
    explicit calibration model therefore applies only the shared commanded-
    angle offset while retaining every detector and shared-geometry correction
    from the fixed position artifact.
    """

    if not isinstance(position, FixedPositionState):
        raise TypeError("position must be FixedPositionState")
    if calibration_model_id != INCIDENCE_SCAN_CALIBRATION_MODEL_ID:
        raise ValueError("unsupported incidence-scan calibration model")
    commanded = tuple(float(value) for value in commanded_incidence_angles_rad)
    if (
        not commanded
        or any(not math.isfinite(value) for value in commanded)
        or any(stop <= start for start, stop in pairwise(commanded))
    ):
        raise ValueError("commanded scan angles must be finite and strictly increasing")
    effective = tuple(value + position.incidence_angle_delta_rad for value in commanded)
    inputs = _build_fixed_experiment_series_at_effective_angles(
        config,
        position=position,
        fixed_lattice=fixed_lattice,
        effective_incidence_angles_rad=effective,
        source_sample_count=source_sample_count,
        gaussian_sigma_rad=gaussian_sigma_rad,
        lorentzian_half_width_rad=lorentzian_half_width_rad,
        lorentzian_probability=lorentzian_probability,
    )
    return FixedIncidenceScanSeries(
        inputs=inputs,
        commanded_incidence_angles_rad=commanded,
        effective_incidence_angles_rad=effective,
        calibration_model_id=calibration_model_id,
        position_artifact_revision=position.artifact_revision,
        incidence_angle_delta_rad=position.incidence_angle_delta_rad,
        _builder_token=_FIXED_INCIDENCE_SCAN_BUILDER_TOKEN,
    )


__all__ = [
    "FIXED_EXPERIMENT_STATE_SCHEMA_VERSION",
    "FIXED_MOSAIC_STATE_SCHEMA_VERSION",
    "INCIDENCE_SCAN_CALIBRATION_MODEL_ID",
    "FixedIncidenceScanSeries",
    "FixedMosaicState",
    "FixedPositionState",
    "build_fixed_experiment_series",
    "build_fixed_incidence_scan_series",
]
