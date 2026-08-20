"""Deterministic panelwise integration for continuous-incidence finite-region observables."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike

from rasim_next.fitting._scan_validation import (
    BoolArray,
    FloatArray,
    IntArray,
)
from rasim_next.fitting._scan_validation import (
    bool_array as _bool_array,
)
from rasim_next.fitting._scan_validation import (
    float_array as _float_array,
)
from rasim_next.fitting._scan_validation import (
    int_array as _int_array,
)
from rasim_next.pipeline.incidence_acquisition import (
    ContinuousIncidenceAcquisition,
    compile_uniform_incidence_angle_legendre_rule,
)


@dataclass(frozen=True, slots=True)
class ScanNodeEvaluation:
    """Finite-ROI values and refinement events at one commanded-angle batch."""

    effective_incidence_angle_rad: ArrayLike
    roi_contrast_A2: ArrayLike
    root_topology_code: ArrayLike
    branch_near_fold: ArrayLike
    signal_window_crossing: ArrayLike
    background_window_crossing: ArrayLike
    rapid_roi_change: ArrayLike
    source_revision: str
    evaluation_revision: str

    def __post_init__(self) -> None:
        effective = _float_array(
            self.effective_incidence_angle_rad,
            "effective_incidence_angle_rad",
            ndim=1,
        )
        mass = _float_array(self.roi_contrast_A2, "roi_contrast_A2", ndim=2)
        topology = _int_array(self.root_topology_code, "root_topology_code", ndim=2)
        shape = mass.shape
        flags: list[BoolArray] = []
        for name in (
            "branch_near_fold",
            "signal_window_crossing",
            "background_window_crossing",
            "rapid_roi_change",
        ):
            supplied = np.asarray(getattr(self, name))
            if supplied.dtype.kind != "b":
                raise ValueError(f"{name} must have boolean dtype")
            value = np.array(supplied, dtype=np.bool_, copy=True, order="C")
            if value.shape != shape:
                raise ValueError(f"{name} must align with finite-ROI masses")
            value.setflags(write=False)
            flags.append(value)
        if (
            mass.shape[0] != effective.size
            or topology.shape != shape
            or not isinstance(self.source_revision, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_revision) is None
            or not isinstance(self.evaluation_revision, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.evaluation_revision) is None
        ):
            raise ValueError("scan node evaluation is invalid")
        object.__setattr__(self, "effective_incidence_angle_rad", effective)
        object.__setattr__(self, "roi_contrast_A2", mass)
        object.__setattr__(self, "root_topology_code", topology)
        object.__setattr__(self, "branch_near_fold", flags[0])
        object.__setattr__(self, "signal_window_crossing", flags[1])
        object.__setattr__(self, "background_window_crossing", flags[2])
        object.__setattr__(self, "rapid_roi_change", flags[3])


@dataclass(frozen=True, slots=True)
class ScanPanelRefinementRecord:
    physical_panel_index: int
    leaf_path: str
    refinement_depth: int
    nominal_lower_deg: float
    nominal_upper_deg: float
    coarse_mass_A2: FloatArray
    fine_mass_A2: FloatArray
    whitened_difference_wrms: float
    event_refinement_required: bool
    accepted: bool

    def __post_init__(self) -> None:
        panel = self.physical_panel_index
        depth = self.refinement_depth
        path = self.leaf_path
        if isinstance(panel, bool) or not isinstance(panel, int) or panel < 0:
            raise ValueError("physical_panel_index must be a nonnegative integer")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("refinement_depth must be a nonnegative integer")
        if not isinstance(path, str) or len(path) != depth or set(path) - {"0", "1"}:
            raise ValueError("leaf_path must encode the refinement depth")
        lower = float(self.nominal_lower_deg)
        upper = float(self.nominal_upper_deg)
        wrms = float(self.whitened_difference_wrms)
        coarse = _float_array(self.coarse_mass_A2, "coarse_mass_A2", ndim=1)
        fine = _float_array(self.fine_mass_A2, "fine_mass_A2", ndim=1)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            or coarse.shape != fine.shape
            or not math.isfinite(wrms)
            or wrms < 0.0
        ):
            raise ValueError("scan panel refinement record is invalid")
        if not isinstance(self.event_refinement_required, bool) or not isinstance(
            self.accepted, bool
        ):
            raise TypeError("scan refinement flags must be booleans")
        if self.accepted and self.event_refinement_required:
            raise ValueError("an event-triggered panel cannot be accepted")
        object.__setattr__(self, "nominal_lower_deg", lower)
        object.__setattr__(self, "nominal_upper_deg", upper)
        object.__setattr__(self, "coarse_mass_A2", coarse)
        object.__setattr__(self, "fine_mass_A2", fine)
        object.__setattr__(self, "whitened_difference_wrms", wrms)


@dataclass(frozen=True, slots=True)
class AdaptiveScanOracleResult:
    """Converged panelwise angular plan and absolute coarse/fine comparison."""

    initial_commanded_incidence_angle_rad: FloatArray
    initial_effective_incidence_angle_rad: FloatArray
    initial_exposure_probability_mass: FloatArray
    initial_physical_panel_index: IntArray
    initial_roi_contrast_A2: FloatArray
    initial_root_topology_code: IntArray
    initial_scan_contrast_A2: FloatArray
    initial_physical_panel_contrast_A2: FloatArray
    commanded_incidence_angle_rad: FloatArray
    effective_incidence_angle_rad: FloatArray
    exposure_probability_mass: FloatArray
    physical_panel_index: IntArray
    leaf_index: IntArray
    roi_contrast_A2: FloatArray
    root_topology_code: IntArray
    branch_near_fold: BoolArray
    signal_window_crossing: BoolArray
    background_window_crossing: BoolArray
    rapid_roi_change: BoolArray
    coarse_scan_contrast_A2: FloatArray
    fine_scan_contrast_A2: FloatArray
    coarse_physical_panel_contrast_A2: FloatArray
    physical_panel_contrast_A2: FloatArray
    working_covariance_count2: FloatArray
    convergence_scale_count_per_A2: float
    convergence_gate_wrms: float
    absolute_convergence_wrms: float
    records: tuple[ScanPanelRefinementRecord, ...]
    source_revision: str
    evaluation_revision: str
    angle_evaluation_count: int
    converged: bool

    def __post_init__(self) -> None:
        initial_commanded = _float_array(
            self.initial_commanded_incidence_angle_rad,
            "initial_commanded_incidence_angle_rad",
            ndim=1,
        )
        initial_effective = _float_array(
            self.initial_effective_incidence_angle_rad,
            "initial_effective_incidence_angle_rad",
            ndim=1,
        )
        initial_mass = _float_array(
            self.initial_exposure_probability_mass,
            "initial_exposure_probability_mass",
            ndim=1,
        )
        initial_panel = _int_array(
            self.initial_physical_panel_index,
            "initial_physical_panel_index",
            ndim=1,
        )
        initial_roi = _float_array(
            self.initial_roi_contrast_A2,
            "initial_roi_contrast_A2",
            ndim=2,
        )
        initial_topology = _int_array(
            self.initial_root_topology_code,
            "initial_root_topology_code",
            ndim=2,
        )
        initial_scan = _float_array(
            self.initial_scan_contrast_A2,
            "initial_scan_contrast_A2",
            ndim=1,
        )
        initial_panel_mass = _float_array(
            self.initial_physical_panel_contrast_A2,
            "initial_physical_panel_contrast_A2",
            ndim=2,
        )
        commanded = _float_array(
            self.commanded_incidence_angle_rad,
            "commanded_incidence_angle_rad",
            ndim=1,
        )
        effective = _float_array(
            self.effective_incidence_angle_rad,
            "effective_incidence_angle_rad",
            ndim=1,
        )
        mass = _float_array(
            self.exposure_probability_mass,
            "exposure_probability_mass",
            ndim=1,
        )
        panel = _int_array(self.physical_panel_index, "physical_panel_index", ndim=1)
        leaf = _int_array(self.leaf_index, "leaf_index", ndim=1)
        roi = _float_array(self.roi_contrast_A2, "roi_contrast_A2", ndim=2)
        topology = _int_array(self.root_topology_code, "root_topology_code", ndim=2)
        branch_fold = _bool_array(self.branch_near_fold, "branch_near_fold", ndim=2)
        signal_crossing = _bool_array(
            self.signal_window_crossing,
            "signal_window_crossing",
            ndim=2,
        )
        background_crossing = _bool_array(
            self.background_window_crossing,
            "background_window_crossing",
            ndim=2,
        )
        rapid = _bool_array(self.rapid_roi_change, "rapid_roi_change", ndim=2)
        coarse_scan = _float_array(
            self.coarse_scan_contrast_A2,
            "coarse_scan_contrast_A2",
            ndim=1,
        )
        fine_scan = _float_array(
            self.fine_scan_contrast_A2,
            "fine_scan_contrast_A2",
            ndim=1,
        )
        coarse_panel = _float_array(
            self.coarse_physical_panel_contrast_A2,
            "coarse_physical_panel_contrast_A2",
            ndim=2,
        )
        panel_mass = _float_array(
            self.physical_panel_contrast_A2,
            "physical_panel_contrast_A2",
            ndim=2,
        )
        covariance = _float_array(
            self.working_covariance_count2,
            "working_covariance_count2",
            ndim=2,
        )
        initial_node_count = initial_commanded.size
        node_count = commanded.size
        roi_count = roi.shape[1]
        panel_count = panel_mass.shape[0]
        if (
            initial_effective.shape != (initial_node_count,)
            or initial_mass.shape != (initial_node_count,)
            or initial_panel.shape != (initial_node_count,)
            or initial_roi.shape != (initial_node_count, roi_count)
            or initial_topology.shape != initial_roi.shape
            or initial_scan.shape != (roi_count,)
            or initial_panel_mass.shape != (panel_count, roi_count)
            or effective.shape != (node_count,)
            or mass.shape != (node_count,)
            or panel.shape != (node_count,)
            or leaf.shape != (node_count,)
            or topology.shape != roi.shape
            or branch_fold.shape != roi.shape
            or signal_crossing.shape != roi.shape
            or background_crossing.shape != roi.shape
            or rapid.shape != roi.shape
            or coarse_scan.shape != (roi_count,)
            or fine_scan.shape != (roi_count,)
            or coarse_panel.shape != (panel_count, roi_count)
            or covariance.shape != (roi_count, roi_count)
            or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError("adaptive scan result arrays do not align")
        root_covariance = np.linalg.cholesky(covariance)
        if (
            np.any(np.diff(initial_commanded) <= 0.0)
            or np.any(np.diff(initial_effective) <= 0.0)
            or np.any(np.diff(commanded) <= 0.0)
            or np.any(np.diff(effective) <= 0.0)
            or np.any(initial_mass <= 0.0)
            or np.any(mass <= 0.0)
            or abs(math.fsum(initial_mass) - 1.0) > 64.0 * np.finfo(np.float64).eps
            or abs(math.fsum(mass) - 1.0) > 64.0 * np.finfo(np.float64).eps
            or not np.array_equal(np.unique(initial_panel), np.arange(panel_count))
            or not np.array_equal(np.unique(panel), np.arange(panel_count))
            or not np.array_equal(np.unique(leaf), np.arange(int(leaf[-1]) + 1))
        ):
            raise ValueError("adaptive scan nodes do not form a normalized panel partition")
        scale = float(self.convergence_scale_count_per_A2)
        gate = float(self.convergence_gate_wrms)
        convergence = float(self.absolute_convergence_wrms)
        evaluations = self.angle_evaluation_count
        records = tuple(self.records)
        if (
            not math.isfinite(scale)
            or scale <= 0.0
            or not math.isfinite(gate)
            or gate <= 0.0
            or not math.isfinite(convergence)
            or convergence < 0.0
            or isinstance(evaluations, bool)
            or not isinstance(evaluations, int)
            or evaluations < node_count
            or not isinstance(self.converged, bool)
            or not records
            or any(not isinstance(record, ScanPanelRefinementRecord) for record in records)
            or any(
                record.physical_panel_index >= panel_count
                or record.coarse_mass_A2.shape != (roi_count,)
                for record in records
            )
            or not isinstance(self.source_revision, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_revision) is None
            or not isinstance(self.evaluation_revision, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.evaluation_revision) is None
        ):
            raise ValueError("adaptive scan result provenance is invalid")
        record_by_key = {
            (record.physical_panel_index, record.leaf_path): record for record in records
        }
        if len(record_by_key) != len(records) or {
            panel_index for panel_index, path in record_by_key if not path
        } != set(range(panel_count)):
            raise ValueError("adaptive scan refinement records do not form one tree per panel")
        for key, record in record_by_key.items():
            panel_index, path = key
            if not path:
                continue
            parent = record_by_key.get((panel_index, path[:-1]))
            if parent is None:
                raise ValueError("adaptive scan refinement record has no active parent")
            midpoint = 0.5 * (parent.nominal_lower_deg + parent.nominal_upper_deg)
            expected_bounds = (
                (parent.nominal_lower_deg, midpoint)
                if path[-1] == "0"
                else (midpoint, parent.nominal_upper_deg)
            )
            if (
                record.nominal_lower_deg != expected_bounds[0]
                or record.nominal_upper_deg != expected_bounds[1]
            ):
                raise ValueError("adaptive scan refinement child has inconsistent bounds")
        terminal_records: list[ScanPanelRefinementRecord] = []
        for panel_index, path in record_by_key:
            has_left = (panel_index, f"{path}0") in record_by_key
            has_right = (panel_index, f"{path}1") in record_by_key
            if has_left != has_right:
                raise ValueError("adaptive scan refinement records require paired children")
            if not has_left:
                terminal_records.append(record_by_key[(panel_index, path)])
        root_records = [record_by_key[(panel_index, "")] for panel_index in range(panel_count)]
        if any(
            left.nominal_upper_deg != right.nominal_lower_deg
            for left, right in pairwise(root_records)
        ):
            raise ValueError("adaptive scan root records do not partition one support")
        full_width_deg = root_records[-1].nominal_upper_deg - root_records[0].nominal_lower_deg
        if full_width_deg <= 0.0:
            raise ValueError("adaptive scan refinement support must have positive width")
        for record in records:
            whitened = np.linalg.solve(
                root_covariance,
                scale * (record.fine_mass_A2 - record.coarse_mass_A2),
            )
            expected_wrms = float(np.sqrt(np.mean(whitened * whitened)))
            wrms_tolerance = 64.0 * np.finfo(np.float64).eps * max(expected_wrms, 1.0)
            if abs(record.whitened_difference_wrms - expected_wrms) > wrms_tolerance:
                raise ValueError("adaptive scan refinement metric is inconsistent")
            leaf_gate = gate * (
                (record.nominal_upper_deg - record.nominal_lower_deg) / full_width_deg
            )
            expected_accepted = expected_wrms < leaf_gate and not record.event_refinement_required
            if record.accepted != expected_accepted:
                raise ValueError("adaptive scan refinement acceptance is inconsistent")
        initial_nodes_per_panel = np.bincount(initial_panel, minlength=panel_count)
        if (
            np.any(initial_nodes_per_panel != initial_nodes_per_panel[0])
            or node_count != 2 * int(initial_nodes_per_panel[0]) * len(terminal_records)
            or evaluations
            != initial_node_count + 2 * int(initial_nodes_per_panel[0]) * len(records)
        ):
            raise ValueError("adaptive scan evaluation counts do not match its refinement tree")
        expected_converged = convergence < gate and all(
            record.accepted for record in terminal_records
        )
        if self.converged != expected_converged:
            raise ValueError("adaptive scan convergence flag disagrees with its terminal records")
        tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                float(np.max(np.abs(initial_scan))),
                float(np.max(np.abs(fine_scan))),
                float(np.max(np.abs(coarse_scan))),
            )
        )
        if (
            not np.allclose(
                np.sum(initial_panel_mass, axis=0), initial_scan, rtol=0.0, atol=tolerance
            )
            or not np.allclose(np.sum(panel_mass, axis=0), fine_scan, rtol=0.0, atol=tolerance)
            or not np.allclose(np.sum(coarse_panel, axis=0), coarse_scan, rtol=0.0, atol=tolerance)
        ):
            raise ValueError("adaptive scan panel masses do not equal their scan totals")
        initial_panel_from_nodes = np.zeros_like(initial_panel_mass)
        final_panel_from_nodes = np.zeros_like(panel_mass)
        for panel_index in range(panel_count):
            initial_selected = initial_panel == panel_index
            final_selected = panel == panel_index
            initial_panel_from_nodes[panel_index] = _weighted_node_sum(
                initial_mass[initial_selected], initial_roi[initial_selected]
            )
            final_panel_from_nodes[panel_index] = _weighted_node_sum(
                mass[final_selected], roi[final_selected]
            )
        terminal_coarse = np.zeros_like(coarse_scan)
        terminal_fine = np.zeros_like(fine_scan)
        terminal_coarse_panel = np.zeros_like(coarse_panel)
        terminal_fine_panel = np.zeros_like(panel_mass)
        for record in terminal_records:
            terminal_coarse += record.coarse_mass_A2
            terminal_fine += record.fine_mass_A2
            terminal_coarse_panel[record.physical_panel_index] += record.coarse_mass_A2
            terminal_fine_panel[record.physical_panel_index] += record.fine_mass_A2
        if (
            not np.allclose(
                _weighted_node_sum(initial_mass, initial_roi),
                initial_scan,
                rtol=0.0,
                atol=tolerance,
            )
            or not np.allclose(
                _weighted_node_sum(mass, roi),
                fine_scan,
                rtol=0.0,
                atol=tolerance,
            )
            or not np.allclose(
                initial_panel_from_nodes,
                initial_panel_mass,
                rtol=0.0,
                atol=tolerance,
            )
            or not np.allclose(
                final_panel_from_nodes,
                panel_mass,
                rtol=0.0,
                atol=tolerance,
            )
            or not np.allclose(terminal_coarse, coarse_scan, rtol=0.0, atol=tolerance)
            or not np.allclose(terminal_fine, fine_scan, rtol=0.0, atol=tolerance)
            or not np.allclose(
                terminal_coarse_panel,
                coarse_panel,
                rtol=0.0,
                atol=tolerance,
            )
            or not np.allclose(
                terminal_fine_panel,
                panel_mass,
                rtol=0.0,
                atol=tolerance,
            )
        ):
            raise ValueError("adaptive scan node masses do not equal their scan totals")
        initial_delta = initial_effective - initial_commanded
        final_delta = effective - commanded
        calibration_tolerance = 32.0 * np.finfo(np.float64).eps
        if (
            np.max(np.abs(initial_delta - initial_delta[0])) > calibration_tolerance
            or np.max(np.abs(final_delta - final_delta[0])) > calibration_tolerance
            or abs(float(initial_delta[0] - final_delta[0])) > calibration_tolerance
        ):
            raise ValueError("adaptive scan nodes do not share one common calibration")
        normalized = {
            "initial_commanded_incidence_angle_rad": initial_commanded,
            "initial_effective_incidence_angle_rad": initial_effective,
            "initial_exposure_probability_mass": initial_mass,
            "initial_physical_panel_index": initial_panel,
            "initial_roi_contrast_A2": initial_roi,
            "initial_root_topology_code": initial_topology,
            "initial_scan_contrast_A2": initial_scan,
            "initial_physical_panel_contrast_A2": initial_panel_mass,
            "commanded_incidence_angle_rad": commanded,
            "effective_incidence_angle_rad": effective,
            "exposure_probability_mass": mass,
            "physical_panel_index": panel,
            "leaf_index": leaf,
            "roi_contrast_A2": roi,
            "root_topology_code": topology,
            "branch_near_fold": branch_fold,
            "signal_window_crossing": signal_crossing,
            "background_window_crossing": background_crossing,
            "rapid_roi_change": rapid,
            "coarse_scan_contrast_A2": coarse_scan,
            "fine_scan_contrast_A2": fine_scan,
            "coarse_physical_panel_contrast_A2": coarse_panel,
            "physical_panel_contrast_A2": panel_mass,
            "working_covariance_count2": covariance,
            "records": records,
            "convergence_scale_count_per_A2": scale,
            "convergence_gate_wrms": gate,
            "absolute_convergence_wrms": convergence,
        }
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class _EvaluatedLeaf:
    physical_panel_index: int
    leaf_path: str
    depth: int
    lower_deg: float
    upper_deg: float
    commanded_rad: FloatArray
    full_support_mass: FloatArray
    evaluation: ScanNodeEvaluation


def _weighted_node_sum(mass: FloatArray, values: FloatArray) -> FloatArray:
    result = np.empty(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        result[index] = math.fsum(
            float(weight) * float(value)
            for weight, value in zip(mass, values[:, index], strict=True)
        )
    result.setflags(write=False)
    return result


def evaluate_adaptive_scan_oracle(
    *,
    acquisition: ContinuousIncidenceAcquisition,
    panel_edges_deg: ArrayLike,
    evaluate_nodes: Callable[[FloatArray], ScanNodeEvaluation],
    working_covariance: ArrayLike,
    model_scale_count_per_A2: float | None = None,
    profile_scale: Callable[[FloatArray], float] | None = None,
    gauss_order: int = 16,
    maximum_refinement_depth: int = 3,
    convergence_gate_wrms: float = 0.25,
) -> AdaptiveScanOracleResult:
    """Refine only failing physical panels using deterministic h-bisection."""

    if not isinstance(acquisition, ContinuousIncidenceAcquisition):
        raise TypeError("acquisition must be ContinuousIncidenceAcquisition")
    if not callable(evaluate_nodes):
        raise TypeError("evaluate_nodes must be callable")
    if (
        isinstance(maximum_refinement_depth, bool)
        or not isinstance(maximum_refinement_depth, int)
        or maximum_refinement_depth < 1
    ):
        raise ValueError("maximum_refinement_depth must be a positive integer")
    gate = float(convergence_gate_wrms)
    if not math.isfinite(gate) or gate <= 0.0:
        raise ValueError("convergence_gate_wrms must be positive")
    if (model_scale_count_per_A2 is None) == (profile_scale is None):
        raise ValueError("provide exactly one fixed model scale or scale profiler")
    model_scale = 1.0 if model_scale_count_per_A2 is None else float(model_scale_count_per_A2)
    if not math.isfinite(model_scale) or model_scale <= 0.0:
        raise ValueError("model_scale_count_per_A2 must be finite and positive")
    covariance = np.asarray(working_covariance, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not covariance.size
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("working_covariance must be a finite symmetric square matrix")
    root_covariance = np.linalg.cholesky(covariance)
    roi_count = covariance.shape[0]
    physical_edges_deg = _float_array(panel_edges_deg, "panel_edges_deg", ndim=1)
    if (
        physical_edges_deg.size < 2
        or np.any(np.diff(physical_edges_deg) <= 0.0)
        or physical_edges_deg[0] != acquisition.nominal_start_deg
        or physical_edges_deg[-1] != acquisition.nominal_stop_deg
    ):
        raise ValueError("panel edges must partition the canonical acquisition support")
    panel_count = physical_edges_deg.size - 1
    full_width_deg = acquisition.nominal_stop_deg - acquisition.nominal_start_deg
    source_revision: str | None = None
    evaluation_revision: str | None = None
    common_delta_rad: float | None = None
    evaluation_count = 0

    def evaluate_leaf(
        physical_panel: int,
        leaf_path: str,
        depth: int,
        lower_deg: float,
        upper_deg: float,
    ) -> _EvaluatedLeaf:
        nonlocal source_revision, evaluation_revision, common_delta_rad, evaluation_count
        commanded, local_mass = compile_uniform_incidence_angle_legendre_rule(
            math.radians(lower_deg),
            math.radians(upper_deg),
            gauss_order=gauss_order,
        )
        full_mass = np.asarray(local_mass * ((upper_deg - lower_deg) / full_width_deg))
        evaluated = evaluate_nodes(commanded)
        if not isinstance(evaluated, ScanNodeEvaluation):
            raise TypeError("evaluate_nodes must return ScanNodeEvaluation")
        if evaluated.roi_contrast_A2.shape != (commanded.size, roi_count):
            raise ValueError("scan node ROI count does not match working covariance")
        delta = np.asarray(evaluated.effective_incidence_angle_rad) - commanded
        tolerance = 32.0 * np.finfo(np.float64).eps
        if np.max(np.abs(delta - delta[0])) > tolerance:
            raise ValueError("scan node evaluator did not apply one common calibration")
        if common_delta_rad is None:
            common_delta_rad = float(delta[0])
        elif abs(float(delta[0]) - common_delta_rad) > tolerance:
            raise ValueError("scan node batches do not share one common calibration")
        if source_revision is None:
            source_revision = evaluated.source_revision
        elif evaluated.source_revision != source_revision:
            raise ValueError("scan node batches must reuse one source realization")
        if evaluation_revision is None:
            evaluation_revision = evaluated.evaluation_revision
        elif evaluated.evaluation_revision != evaluation_revision:
            raise ValueError("scan node batches must share one evaluation contract")
        evaluation_count += commanded.size
        return _EvaluatedLeaf(
            physical_panel,
            leaf_path,
            depth,
            lower_deg,
            upper_deg,
            commanded,
            full_mass,
            evaluated,
        )

    def contribution(leaf: _EvaluatedLeaf) -> FloatArray:
        return _weighted_node_sum(leaf.full_support_mass, leaf.evaluation.roi_contrast_A2)

    def whitened_wrms(value: FloatArray) -> float:
        whitened = np.linalg.solve(root_covariance, model_scale * value)
        return float(np.sqrt(np.mean(whitened * whitened)))

    records: list[ScanPanelRefinementRecord] = []
    final_leaves: list[_EvaluatedLeaf] = []
    initial_leaves: list[_EvaluatedLeaf] = []

    def refine(coarse: _EvaluatedLeaf) -> None:
        midpoint = 0.5 * (coarse.lower_deg + coarse.upper_deg)
        left = evaluate_leaf(
            coarse.physical_panel_index,
            f"{coarse.leaf_path}0",
            coarse.depth + 1,
            coarse.lower_deg,
            midpoint,
        )
        right = evaluate_leaf(
            coarse.physical_panel_index,
            f"{coarse.leaf_path}1",
            coarse.depth + 1,
            midpoint,
            coarse.upper_deg,
        )
        coarse_mass = contribution(coarse)
        fine_mass = contribution(left) + contribution(right)
        difference_wrms = whitened_wrms(fine_mass - coarse_mass)
        combined_angle = np.concatenate(
            (
                coarse.evaluation.effective_incidence_angle_rad,
                left.evaluation.effective_incidence_angle_rad,
                right.evaluation.effective_incidence_angle_rad,
            )
        )
        combined_topology = np.concatenate(
            (
                coarse.evaluation.root_topology_code,
                left.evaluation.root_topology_code,
                right.evaluation.root_topology_code,
            ),
            axis=0,
        )[np.argsort(combined_angle)]
        combined_flags = (
            np.any(coarse.evaluation.branch_near_fold)
            or np.any(left.evaluation.branch_near_fold)
            or np.any(right.evaluation.branch_near_fold)
            or np.any(coarse.evaluation.signal_window_crossing)
            or np.any(left.evaluation.signal_window_crossing)
            or np.any(right.evaluation.signal_window_crossing)
            or np.any(coarse.evaluation.background_window_crossing)
            or np.any(left.evaluation.background_window_crossing)
            or np.any(right.evaluation.background_window_crossing)
            or np.any(coarse.evaluation.rapid_roi_change)
            or np.any(left.evaluation.rapid_roi_change)
            or np.any(right.evaluation.rapid_roi_change)
            or np.any(np.diff(combined_topology, axis=0) != 0)
        )
        event_requires_refinement = bool(combined_flags)
        leaf_gate = gate * (coarse.upper_deg - coarse.lower_deg) / full_width_deg
        passed = difference_wrms < leaf_gate and not event_requires_refinement
        terminal = coarse.depth + 1 >= maximum_refinement_depth
        records.append(
            ScanPanelRefinementRecord(
                physical_panel_index=coarse.physical_panel_index,
                leaf_path=coarse.leaf_path,
                refinement_depth=coarse.depth,
                nominal_lower_deg=coarse.lower_deg,
                nominal_upper_deg=coarse.upper_deg,
                coarse_mass_A2=coarse_mass,
                fine_mass_A2=fine_mass,
                whitened_difference_wrms=difference_wrms,
                event_refinement_required=event_requires_refinement,
                accepted=passed,
            )
        )
        if passed or terminal:
            final_leaves.extend((left, right))
        else:
            refine(left)
            refine(right)

    for physical_panel, (lower_deg, upper_deg) in enumerate(pairwise(physical_edges_deg)):
        initial = evaluate_leaf(
            physical_panel,
            "",
            0,
            float(lower_deg),
            float(upper_deg),
        )
        initial_leaves.append(initial)

    initial_commanded = np.concatenate([leaf.commanded_rad for leaf in initial_leaves])
    initial_effective = np.concatenate(
        [leaf.evaluation.effective_incidence_angle_rad for leaf in initial_leaves]
    )
    initial_mass = np.concatenate([leaf.full_support_mass for leaf in initial_leaves])
    initial_roi_mass = np.concatenate(
        [leaf.evaluation.roi_contrast_A2 for leaf in initial_leaves],
        axis=0,
    )
    initial_topology = np.concatenate(
        [leaf.evaluation.root_topology_code for leaf in initial_leaves],
        axis=0,
    )
    initial_panel_index = np.concatenate(
        [
            np.full(leaf.commanded_rad.size, leaf.physical_panel_index, dtype=np.int64)
            for leaf in initial_leaves
        ]
    )
    initial_scan_mass = _weighted_node_sum(initial_mass, initial_roi_mass)
    initial_panel_mass = np.zeros((panel_count, roi_count), dtype=np.float64)
    for panel in range(panel_count):
        selected = initial_panel_index == panel
        initial_panel_mass[panel] = _weighted_node_sum(
            initial_mass[selected],
            initial_roi_mass[selected],
        )
    if profile_scale is not None:
        model_scale = float(profile_scale(initial_scan_mass))
        if not math.isfinite(model_scale) or model_scale <= 0.0:
            raise ValueError("profile_scale must return a finite positive scale")
    for initial in initial_leaves:
        refine(initial)
    final_leaves.sort(key=lambda leaf: leaf.lower_deg)
    commanded = np.concatenate([leaf.commanded_rad for leaf in final_leaves])
    effective = np.concatenate(
        [leaf.evaluation.effective_incidence_angle_rad for leaf in final_leaves]
    )
    mass = np.concatenate([leaf.full_support_mass for leaf in final_leaves])
    roi_mass = np.concatenate(
        [leaf.evaluation.roi_contrast_A2 for leaf in final_leaves],
        axis=0,
    )
    topology = np.concatenate(
        [leaf.evaluation.root_topology_code for leaf in final_leaves],
        axis=0,
    )
    branch_fold = np.concatenate(
        [leaf.evaluation.branch_near_fold for leaf in final_leaves],
        axis=0,
    )
    signal_crossing = np.concatenate(
        [leaf.evaluation.signal_window_crossing for leaf in final_leaves],
        axis=0,
    )
    background_crossing = np.concatenate(
        [leaf.evaluation.background_window_crossing for leaf in final_leaves],
        axis=0,
    )
    rapid = np.concatenate(
        [leaf.evaluation.rapid_roi_change for leaf in final_leaves],
        axis=0,
    )
    physical_panel_index = np.concatenate(
        [
            np.full(leaf.commanded_rad.size, leaf.physical_panel_index, dtype=np.int64)
            for leaf in final_leaves
        ]
    )
    leaf_index = np.concatenate(
        [
            np.full(leaf.commanded_rad.size, index, dtype=np.int64)
            for index, leaf in enumerate(final_leaves)
        ]
    )
    integrated_final_scan_mass = _weighted_node_sum(mass, roi_mass)
    panel_mass = np.zeros((panel_count, roi_count), dtype=np.float64)
    for panel in range(panel_count):
        selected = physical_panel_index == panel
        panel_mass[panel] = _weighted_node_sum(mass[selected], roi_mass[selected])
    final_scale = model_scale
    if profile_scale is not None:
        final_scale = float(profile_scale(integrated_final_scan_mass))
        if not math.isfinite(final_scale) or final_scale <= 0.0:
            raise ValueError("profile_scale must return a finite positive scale")
    model_scale = final_scale
    final_leaf_passed = True
    terminal_records: list[ScanPanelRefinementRecord] = []
    rescored_records: list[ScanPanelRefinementRecord] = []
    for record in records:
        rescored_wrms = whitened_wrms(record.fine_mass_A2 - record.coarse_mass_A2)
        leaf_gate = gate * (record.nominal_upper_deg - record.nominal_lower_deg) / full_width_deg
        rescored_passed = rescored_wrms < leaf_gate and not record.event_refinement_required
        is_final_record = record.accepted or (
            record.refinement_depth + 1 >= maximum_refinement_depth
        )
        if is_final_record:
            terminal_records.append(record)
        if is_final_record and not rescored_passed:
            final_leaf_passed = False
        rescored_records.append(
            replace(
                record,
                whitened_difference_wrms=rescored_wrms,
                accepted=rescored_passed,
            )
        )
    coarse_scan_mass = np.zeros(roi_count, dtype=np.float64)
    fine_scan_mass = np.zeros(roi_count, dtype=np.float64)
    coarse_panel_mass = np.zeros((panel_count, roi_count), dtype=np.float64)
    terminal_panel_mass = np.zeros((panel_count, roi_count), dtype=np.float64)
    for record in terminal_records:
        coarse_scan_mass += record.coarse_mass_A2
        fine_scan_mass += record.fine_mass_A2
        coarse_panel_mass[record.physical_panel_index] += record.coarse_mass_A2
        terminal_panel_mass[record.physical_panel_index] += record.fine_mass_A2
    if not np.allclose(
        fine_scan_mass,
        integrated_final_scan_mass,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(fine_scan_mass))), 1.0),
    ) or not np.allclose(
        terminal_panel_mass,
        panel_mass,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(panel_mass))), 1.0),
    ):
        raise RuntimeError("terminal refinement records do not partition the scan support")
    absolute_convergence = whitened_wrms(fine_scan_mass - coarse_scan_mass)
    converged = bool(final_leaf_passed and absolute_convergence < gate)
    for value in (
        initial_commanded,
        initial_effective,
        initial_mass,
        initial_panel_index,
        initial_roi_mass,
        initial_topology,
        initial_panel_mass,
        coarse_scan_mass,
        coarse_panel_mass,
        commanded,
        effective,
        mass,
        roi_mass,
        topology,
        branch_fold,
        signal_crossing,
        background_crossing,
        rapid,
        physical_panel_index,
        leaf_index,
        initial_scan_mass,
        fine_scan_mass,
        panel_mass,
    ):
        value.setflags(write=False)
    assert source_revision is not None and evaluation_revision is not None
    return AdaptiveScanOracleResult(
        initial_commanded_incidence_angle_rad=initial_commanded,
        initial_effective_incidence_angle_rad=initial_effective,
        initial_exposure_probability_mass=initial_mass,
        initial_physical_panel_index=initial_panel_index,
        initial_roi_contrast_A2=initial_roi_mass,
        initial_root_topology_code=initial_topology,
        initial_scan_contrast_A2=initial_scan_mass,
        initial_physical_panel_contrast_A2=initial_panel_mass,
        commanded_incidence_angle_rad=commanded,
        effective_incidence_angle_rad=effective,
        exposure_probability_mass=mass,
        physical_panel_index=physical_panel_index,
        leaf_index=leaf_index,
        roi_contrast_A2=roi_mass,
        root_topology_code=topology,
        branch_near_fold=branch_fold,
        signal_window_crossing=signal_crossing,
        background_window_crossing=background_crossing,
        rapid_roi_change=rapid,
        coarse_scan_contrast_A2=coarse_scan_mass,
        fine_scan_contrast_A2=fine_scan_mass,
        coarse_physical_panel_contrast_A2=coarse_panel_mass,
        physical_panel_contrast_A2=panel_mass,
        working_covariance_count2=covariance,
        convergence_scale_count_per_A2=model_scale,
        convergence_gate_wrms=gate,
        absolute_convergence_wrms=absolute_convergence,
        records=tuple(rescored_records),
        source_revision=source_revision,
        evaluation_revision=evaluation_revision,
        angle_evaluation_count=evaluation_count,
        converged=converged,
    )


__all__ = [
    "AdaptiveScanOracleResult",
    "ScanNodeEvaluation",
    "ScanPanelRefinementRecord",
    "evaluate_adaptive_scan_oracle",
]
