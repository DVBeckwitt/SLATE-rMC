"""Joint fitting of heterogeneous detector-region observations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.measurement.continuous_regions import ContinuousRegionQuadrature
from rasim_next.pipeline.bragg_space import StructureStrengthParameterization
from rasim_next.pipeline.source_averaged_structure import (
    SourceAveragedDetectorStructureResponse,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FixedMatchedRegionBackground:
    """Frozen background correction, uncertainty, and anchor interpolation operator.

    The correction may be signed after dark subtraction and adjacent-anchor
    conditioning; it is never clipped before subtraction from the observation.
    """

    count_mass: ArrayLike
    covariance_count2: ArrayLike
    revision: str
    anchor_projection: ArrayLike | None = None

    def __post_init__(self) -> None:
        mass = np.array(self.count_mass, dtype=np.float64, copy=True, order="C")
        covariance = np.array(
            self.covariance_count2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        projection = (
            np.zeros((mass.size, mass.size), dtype=np.float64)
            if self.anchor_projection is None
            else np.array(self.anchor_projection, dtype=np.float64, copy=True, order="C")
        )
        covariance_eigenvalue = (
            np.linalg.eigvalsh(covariance)
            if covariance.shape == (mass.size, mass.size)
            else np.asarray((-math.inf,))
        )
        if (
            mass.ndim != 1
            or not mass.size
            or covariance.shape != (mass.size, mass.size)
            or projection.shape != (mass.size, mass.size)
            or np.any(~np.isfinite(mass))
            or np.any(~np.isfinite(covariance))
            or np.any(~np.isfinite(projection))
            or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-10)
            or np.min(covariance_eigenvalue)
            < -1.0e-10 * max(float(np.max(np.abs(covariance_eigenvalue))), 1.0)
            or not isinstance(self.revision, str)
            or not self.revision
        ):
            raise ValueError("fixed matched-region background is invalid")
        mass.setflags(write=False)
        covariance.setflags(write=False)
        projection.setflags(write=False)
        object.__setattr__(self, "count_mass", mass)
        object.__setattr__(self, "covariance_count2", covariance)
        object.__setattr__(self, "anchor_projection", projection)


@dataclass(frozen=True, slots=True)
class MatchedRegionObservations:
    """Raw count masses grouped into complete signal/background blocks."""

    dataset_ids: tuple[str, ...]
    dataset_index: ArrayLike
    block_index: ArrayLike
    signal_family: ArrayLike
    is_background: ArrayLike
    count_mass: ArrayLike
    support_px2: ArrayLike
    background_coordinate: ArrayLike
    required_signal_families: tuple[int, ...]
    count_covariance_count2: ArrayLike | None = None

    def __post_init__(self) -> None:
        dataset_ids = tuple(self.dataset_ids)
        required = tuple(int(value) for value in self.required_signal_families)
        if (
            not dataset_ids
            or any(not isinstance(value, str) or not value for value in dataset_ids)
            or len(set(dataset_ids)) != len(dataset_ids)
        ):
            raise ValueError("dataset_ids must contain unique nonempty strings")
        if (
            not required
            or len(set(required)) != len(required)
            or any(value < 0 for value in required)
        ):
            raise ValueError("required_signal_families must contain unique nonnegative values")
        dataset = np.array(self.dataset_index, dtype=np.int64, copy=True, order="C")
        block = np.array(self.block_index, dtype=np.int64, copy=True, order="C")
        family = np.array(self.signal_family, dtype=np.int64, copy=True, order="C")
        background = np.array(self.is_background, dtype=np.bool_, copy=True, order="C")
        count = np.array(self.count_mass, dtype=np.float64, copy=True, order="C")
        support = np.array(self.support_px2, dtype=np.float64, copy=True, order="C")
        coordinate = np.array(
            self.background_coordinate,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        shape = count.shape
        covariance_was_supplied = self.count_covariance_count2 is not None
        count_covariance = (
            np.diag(np.maximum(count, 1.0))
            if self.count_covariance_count2 is None
            else np.array(
                self.count_covariance_count2,
                dtype=np.float64,
                copy=True,
                order="C",
            )
        )
        covariance_eigenvalue = (
            np.linalg.eigvalsh(count_covariance)
            if count_covariance.shape == (count.size, count.size)
            else np.asarray((-math.inf,))
        )
        if (
            count.ndim != 1
            or not count.size
            or dataset.shape != shape
            or block.shape != shape
            or family.shape != shape
            or background.shape != shape
            or support.shape != shape
            or coordinate.shape != shape
            or count_covariance.shape != (count.size, count.size)
        ):
            raise ValueError("matched-region observation arrays must be aligned nonempty vectors")
        if (
            np.any((dataset < 0) | (dataset >= len(dataset_ids)))
            or np.any(block < 0)
            or np.any(~np.isfinite(count))
            or np.any(~np.isfinite(support))
            or np.any(support <= 0.0)
            or np.any(~np.isfinite(coordinate))
            or np.any(~np.isfinite(count_covariance))
            or not np.allclose(count_covariance, count_covariance.T, rtol=0.0, atol=1.0e-10)
            or np.min(covariance_eigenvalue)
            < -1.0e-10 * max(float(np.max(np.abs(covariance_eigenvalue))), 1.0)
            or np.any(background & (family != -1))
            or np.any((~background) & ~np.isin(family, required))
            or (not covariance_was_supplied and np.any(count < 0.0))
        ):
            raise ValueError("matched-region observations contain invalid values")
        unique_blocks = np.unique(block)
        if not np.array_equal(unique_blocks, np.arange(unique_blocks.size)):
            raise ValueError("block_index must be contiguous from zero")
        for block_id in unique_blocks:
            selected = block == block_id
            if np.unique(dataset[selected]).size != 1:
                raise ValueError("one background block cannot cross datasets")
            if np.count_nonzero(background[selected]) != 2 or not np.any(~background[selected]):
                raise ValueError("every block requires signals and exactly two background anchors")
            design = np.column_stack((support[selected], support[selected] * coordinate[selected]))
            if np.linalg.matrix_rank(design) != 2:
                raise ValueError("every block requires a rank-two affine background design")
        represented_globally = set(family[~background].tolist())
        missing = set(required) - represented_globally
        if missing:
            raise ValueError(f"joint observations lack signal families {sorted(missing)}")
        for value in (
            dataset,
            block,
            family,
            background,
            count,
            support,
            coordinate,
            count_covariance,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "dataset_index", dataset)
        object.__setattr__(self, "block_index", block)
        object.__setattr__(self, "signal_family", family)
        object.__setattr__(self, "is_background", background)
        object.__setattr__(self, "count_mass", count)
        object.__setattr__(self, "support_px2", support)
        object.__setattr__(self, "background_coordinate", coordinate)
        object.__setattr__(self, "required_signal_families", required)
        object.__setattr__(self, "count_covariance_count2", count_covariance)


@dataclass(frozen=True, slots=True)
class IntegratedPeakAreaProjection:
    """Fixed sum from conditioned signal bins to trusted detector-peak areas."""

    peak_ids: tuple[str, ...]
    peak_dataset_index: ArrayLike
    peak_signal_family: ArrayLike
    source_signal_peak_index: ArrayLike
    revision: str

    def __post_init__(self) -> None:
        peak_ids = tuple(self.peak_ids)
        dataset = np.array(self.peak_dataset_index, dtype=np.int64, copy=True, order="C")
        family = np.array(self.peak_signal_family, dtype=np.int64, copy=True, order="C")
        source_peak = np.array(
            self.source_signal_peak_index,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        peak_count = len(peak_ids)
        if (
            not peak_ids
            or len(set(peak_ids)) != peak_count
            or any(not isinstance(value, str) or not value for value in peak_ids)
            or dataset.shape != (peak_count,)
            or family.shape != (peak_count,)
            or source_peak.ndim != 1
            or not source_peak.size
            or np.any(dataset < 0)
            or np.any(family < 0)
            or np.any((source_peak < 0) | (source_peak >= peak_count))
            or not np.array_equal(np.unique(source_peak), np.arange(peak_count))
            or not isinstance(self.revision, str)
            or not self.revision
        ):
            raise ValueError("integrated peak-area projection is invalid")
        for value in (dataset, family, source_peak):
            value.setflags(write=False)
        object.__setattr__(self, "peak_ids", peak_ids)
        object.__setattr__(self, "peak_dataset_index", dataset)
        object.__setattr__(self, "peak_signal_family", family)
        object.__setattr__(self, "source_signal_peak_index", source_peak)

    def aggregation_matrix(self, observations: MatchedRegionObservations) -> FloatArray:
        """Return the validated Boolean sum matrix in canonical signal-row order."""

        if not isinstance(observations, MatchedRegionObservations):
            raise TypeError("observations must be MatchedRegionObservations")
        signal = ~np.asarray(observations.is_background)
        source_peak = np.asarray(self.source_signal_peak_index)
        if source_peak.size != int(np.count_nonzero(signal)):
            raise ValueError("peak-area mapping does not cover every signal row exactly once")
        source_dataset = np.asarray(observations.dataset_index)[signal]
        source_family = np.asarray(observations.signal_family)[signal]
        peak_count = len(self.peak_ids)
        if np.any(np.asarray(self.peak_dataset_index) >= len(observations.dataset_ids)):
            raise ValueError("peak-area dataset index lies outside the observations")
        for peak_index in range(peak_count):
            selected = source_peak == peak_index
            if np.any(source_dataset[selected] != self.peak_dataset_index[peak_index]) or np.any(
                source_family[selected] != self.peak_signal_family[peak_index]
            ):
                raise ValueError("one integrated peak cannot cross datasets or signal families")
        matrix = np.zeros((peak_count, source_peak.size), dtype=np.float64)
        matrix[source_peak, np.arange(source_peak.size)] = 1.0
        matrix.setflags(write=False)
        return matrix


@dataclass(frozen=True, slots=True)
class StructureRegionResponseBlock:
    """One dataset's sparse detector transfer and exact region quadrature."""

    dataset_id: str
    response: SourceAveragedDetectorStructureResponse
    quadrature: ContinuousRegionQuadrature
    block_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a nonempty string")
        if not isinstance(self.response, SourceAveragedDetectorStructureResponse):
            raise TypeError("response must be SourceAveragedDetectorStructureResponse")
        if not isinstance(self.quadrature, ContinuousRegionQuadrature):
            raise TypeError("quadrature must be ContinuousRegionQuadrature")
        if not np.array_equal(
            self.response.column_px, self.quadrature.column_px
        ) or not np.array_equal(
            self.response.row_px,
            self.quadrature.row_px,
        ):
            raise ValueError("response coordinates and region quadrature must align exactly")
        if np.any(self.response.per_rod_caustic):
            raise ValueError("structure-region response cannot contain detector caustics")
        object.__setattr__(
            self,
            "block_revision",
            canonical_revision_sha256(
                ("definition_id", "structure_region_response_block.v1"),
                ("dataset_id", self.dataset_id),
                ("response_revision", self.response.response_revision),
                ("quadrature_revision", self.quadrature.quadrature_revision),
            ),
        )


@dataclass(frozen=True, slots=True)
class ParameterizedStructureRegionModel:
    """Shared structure parameters applied to fixed detector-region transfers."""

    parameterization: StructureStrengthParameterization
    blocks: tuple[StructureRegionResponseBlock, ...]
    parameter_names: tuple[str, ...] = field(init=False)
    parameter_units: tuple[str, ...] = field(init=False)
    reference_parameters: FloatArray = field(init=False)
    dataset_ids: tuple[str, ...] = field(init=False)
    observation_count: int = field(init=False)
    model_revision: str = field(init=False)

    def __post_init__(self) -> None:
        parameterization = self.parameterization
        names = tuple(getattr(parameterization, "parameter_names", ()))
        units = tuple(getattr(parameterization, "parameter_units", ()))
        reference = np.asarray(
            getattr(parameterization, "reference_parameters", ()), dtype=np.float64
        )
        parameterization_revision = getattr(parameterization, "parameterization_revision", None)
        reference_strength = getattr(parameterization, "reference_strength", None)
        reference_structure_revision = getattr(reference_strength, "structure_model_revision", None)
        if (
            not names
            or len(set(names)) != len(names)
            or len(units) != len(names)
            or reference.shape != (len(names),)
            or np.any(~np.isfinite(reference))
            or not isinstance(parameterization_revision, str)
            or len(parameterization_revision) != 64
            or not callable(getattr(parameterization, "bind_strength", None))
            or not isinstance(reference_structure_revision, str)
            or len(reference_structure_revision) != 64
        ):
            raise ValueError("structure-strength parameterization contract is incomplete")
        blocks = tuple(self.blocks)
        if not blocks or any(
            not isinstance(block, StructureRegionResponseBlock) for block in blocks
        ):
            raise ValueError("blocks must contain structure-region response blocks")
        observation_counts = {block.quadrature.observation_count for block in blocks}
        if len(observation_counts) != 1:
            raise ValueError("all structure-region blocks must share one observation row space")
        if any(
            block.response.reference_structure_model_revision != reference_structure_revision
            for block in blocks
        ):
            raise ValueError("response was compiled from a different reference structure model")
        dataset_ids = tuple(dict.fromkeys(block.dataset_id for block in blocks))
        frozen_reference = np.array(reference, copy=True)
        frozen_reference.setflags(write=False)
        model_revision = canonical_revision_sha256(
            ("definition_id", "parameterized_structure_region_model.v1"),
            ("parameterization_revision", parameterization_revision),
            ("block_revisions", tuple(block.block_revision for block in blocks)),
        )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", units)
        object.__setattr__(self, "reference_parameters", frozen_reference)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "observation_count", observation_counts.pop())
        object.__setattr__(self, "model_revision", model_revision)

    def predict_mass_A2(self, parameters: ArrayLike) -> FloatArray:
        """Integrate one candidate structure over every fixed observation region."""

        candidate = self.parameterization.bind_strength(parameters)
        result = np.zeros(self.observation_count, dtype=np.float64)
        for block in self.blocks:
            density = block.response.apply_strength(candidate).density_A2_per_px2
            result += block.quadrature.integrate_density(density)
        if np.any(~np.isfinite(result)) or np.any(result < 0.0):
            raise FloatingPointError("parameterized structure model returned invalid mass")
        result.setflags(write=False)
        return result


def condition_matched_region_background_from_anchors(
    observations: MatchedRegionObservations,
    baseline: FixedMatchedRegionBackground,
) -> FixedMatchedRegionBackground:
    """Condition a frozen baseline on the two measured anchors in every block.

    The anchor residual above the baseline is treated as an affine density in
    the block coordinate.  Its prediction is frozen before structure fitting;
    anchor counting noise and baseline covariance are propagated into the
    resulting signal-background covariance.
    """

    if not isinstance(observations, MatchedRegionObservations):
        raise TypeError("observations must be MatchedRegionObservations")
    if not isinstance(baseline, FixedMatchedRegionBackground):
        raise TypeError("baseline must be FixedMatchedRegionBackground")
    count = np.asarray(observations.count_mass)
    if baseline.count_mass.shape != count.shape:
        raise ValueError("baseline background and observations must align")

    row_count = count.size
    conditioned_mass = np.array(baseline.count_mass, copy=True)
    if np.any(np.asarray(baseline.anchor_projection) != 0.0):
        raise ValueError("background baseline is already conditioned on anchors")
    baseline_transform = np.eye(row_count, dtype=np.float64)
    anchor_projection = np.zeros((row_count, row_count), dtype=np.float64)
    support = np.asarray(observations.support_px2)
    coordinate = np.asarray(observations.background_coordinate)
    is_background = np.asarray(observations.is_background)
    for block_id in range(int(np.max(observations.block_index)) + 1):
        index = np.flatnonzero(observations.block_index == block_id)
        anchor_local = np.flatnonzero(is_background[index])
        anchor_index = index[anchor_local]
        anchor_coordinate = coordinate[anchor_index]
        signal_coordinate = coordinate[index[~is_background[index]]]
        if (
            anchor_coordinate[0] == anchor_coordinate[1]
            or np.any(signal_coordinate < np.min(anchor_coordinate))
            or np.any(signal_coordinate > np.max(anchor_coordinate))
        ):
            raise ValueError("adjacent background anchors must bracket every signal coordinate")
        design = np.column_stack((support[index], support[index] * coordinate[index]))
        anchor_design = design[anchor_local]
        interpolation = np.linalg.solve(anchor_design.T, design.T).T
        residual_anchor_mass = count[anchor_index] - baseline.count_mass[anchor_index]
        conditioned_mass[index] += interpolation @ residual_anchor_mass

        selector = np.zeros((2, index.size), dtype=np.float64)
        selector[np.arange(2), anchor_local] = 1.0
        block_projection = interpolation @ selector
        anchor_projection[np.ix_(index, index)] = block_projection
        baseline_transform[np.ix_(index, index)] -= block_projection
    count_covariance = np.asarray(observations.count_covariance_count2)
    covariance = (
        baseline_transform @ np.asarray(baseline.covariance_count2) @ baseline_transform.T
        + anchor_projection @ count_covariance @ anchor_projection.T
    )
    covariance = 0.5 * (covariance + covariance.T)
    digest = hashlib.sha256()
    digest.update(b"matched-region-adjacent-affine-background.v1\0")
    digest.update(baseline.revision.encode("utf-8"))
    digest.update(np.ascontiguousarray(conditioned_mass).tobytes())
    digest.update(np.ascontiguousarray(covariance).tobytes())
    digest.update(np.ascontiguousarray(anchor_projection).tobytes())
    return FixedMatchedRegionBackground(
        count_mass=conditioned_mass,
        covariance_count2=covariance,
        revision=f"sha256-{digest.hexdigest()}.adjacent-affine.v1",
        anchor_projection=anchor_projection,
    )


def condition_matched_region_model_from_anchors(
    background: FixedMatchedRegionBackground,
    model_mass: ArrayLike,
) -> FloatArray:
    """Apply the data-side anchor subtraction operator to a diffraction model."""

    if not isinstance(background, FixedMatchedRegionBackground):
        raise TypeError("background must be FixedMatchedRegionBackground")
    model = np.asarray(model_mass, dtype=np.float64)
    if (
        model.shape != background.count_mass.shape
        or np.any(~np.isfinite(model))
        or np.any(model < 0.0)
    ):
        raise ValueError("model_mass must be a finite aligned nonnegative vector")
    return model - np.asarray(background.anchor_projection) @ model


def prepare_matched_region_objective(
    observations: MatchedRegionObservations,
    background: FixedMatchedRegionBackground,
    *,
    peak_area_projection: IntegratedPeakAreaProjection | None = None,
):
    """Return shared count objective and raw-model operator with full covariance.

    The operator conditions diffraction on the same anchors as the data, selects
    signal rows and optionally integrates peaks. Each acquisition keeps its own
    exposure. Cross-acquisition covariance remains in the joint NNLS scale fit.
    """
    from rasim_next.fitting.native_observations import NativeFitObservations

    if not isinstance(observations, MatchedRegionObservations) or not isinstance(
        background, FixedMatchedRegionBackground
    ):
        raise TypeError("matched observations and frozen background are required")
    count = observations.count_mass
    if background.count_mass.shape != count.shape:
        raise ValueError("background and observations must align")
    h = background.anchor_projection
    dataset = observations.dataset_index
    if np.any(h[dataset[:, None] != dataset[None, :]] != 0):
        raise ValueError("anchor conditioning cannot cross acquisitions")
    signal = ~observations.is_background
    selection = np.eye(len(count))[signal]
    objective_dataset = dataset[signal]
    if peak_area_projection is not None:
        if not isinstance(peak_area_projection, IntegratedPeakAreaProjection):
            raise TypeError("peak aggregation must be IntegratedPeakAreaProjection")
        selection = peak_area_projection.aggregation_matrix(observations) @ selection
        objective_dataset = peak_area_projection.peak_dataset_index
    covariance = observations.count_covariance_count2
    covariance = (
        selection
        @ (covariance + background.covariance_count2 - h @ covariance - covariance @ h.T)
        @ selection.T
    )
    covariance = (covariance + covariance.T) * 0.5
    target = selection @ (count - background.count_mass)
    operator = selection @ (np.eye(len(count)) - h)
    operator.setflags(write=False)
    revision = canonical_revision_sha256(
        ("definition_id", "matched_native_count_objective.v1"),
        ("dataset_ids", observations.dataset_ids),
        ("dataset_index", objective_dataset),
        ("support_px2", observations.support_px2),
        ("operator", operator),
        ("net_count", target),
        ("covariance", covariance),
        ("background", background.revision),
    )
    n = len(target)
    objective = NativeFitObservations(
        projection=None,
        net_count=target,
        valid=np.ones(n, dtype=bool),
        covariance_count2=covariance,
        fit_operator=np.empty((0, n)),
        fit_target=np.empty(0),
        guard_operator=np.empty((0, n)),
        guard_pointer=np.array([0]),
        guard_limit=np.empty(0),
        input_revision=revision,
        allow_guard_constraints=False,
        exposure_index=objective_dataset,
    )
    return objective, operator


class StructureRegionIdentifiabilityError(RuntimeError):
    """A failed or unresolved fit retains its shared-search result for inspection."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def fit_structure_regions(
    observations,
    model,
    *,
    fixed_background,
    parameters,
    starts,
    peak_area_projection=None,
    calibration=(),
    maximum_function_evaluations=200,
    sensitivity_relative_tolerance=1e-5,
    maximum_sensitivity_condition=1e5,
):
    """Fit generic-CIF or supported stacking responses through the native search.

    Gaussian calibration blocks are explicit. The retired arbitrary prior callback
    is not supported. Data-only rank is assessed after exposure profiling; priors
    cannot turn a non-identifiable physical result into an accepted result.
    """
    from rasim_next.fitting.native_accuracy import native_sensitivity
    from rasim_next.fitting.native_search import fit_native_parameters

    if not isinstance(model, ParameterizedStructureRegionModel):
        raise TypeError("model must be a ParameterizedStructureRegionModel")
    parameters = tuple(parameters)
    if len({p.owner for p in parameters}) != 1 or any(
        not p.owner.startswith("specimen:") for p in parameters
    ):
        raise ValueError("shared structure coordinates must belong to one named specimen")
    if (
        tuple(p.name for p in parameters) != model.parameter_names
        or tuple(p.unit for p in parameters) != model.parameter_units
    ):
        raise ValueError("parameter names and units must match the physical structure binding")
    if (
        not 0 < sensitivity_relative_tolerance < 1
        or not np.isfinite(maximum_sensitivity_condition)
        or maximum_sensitivity_condition < 1
    ):
        raise ValueError("invalid data-sensitivity rank or conditioning limits")
    if observations.count_mass.size != model.observation_count or set(
        observations.dataset_ids
    ) != set(model.dataset_ids):
        raise ValueError("model and observations must share rows and acquisition IDs")
    by_id = {key: i for i, key in enumerate(observations.dataset_ids)}
    for block in model.blocks:
        if np.any(
            observations.dataset_index[block.quadrature.observation_covered]
            != by_id[block.dataset_id]
        ):
            raise ValueError("a structure response cannot cross acquisitions")
    objective, operator = prepare_matched_region_objective(
        observations,
        fixed_background,
        peak_area_projection=peak_area_projection,
    )

    def predict(values):
        return operator @ model.predict_mass_A2(values)

    result = fit_native_parameters(
        predict,
        objective,
        parameters,
        starts,
        calibration=calibration,
        method="trf",
        maximum_function_evaluations=maximum_function_evaluations,
    )
    result.acquisition_ids = observations.dataset_ids
    result.model_revision = model.model_revision
    result.parameterization_revision = model.parameterization.parameterization_revision
    if not result.minimum_resolved or result.best_converged is None:
        raise StructureRegionIdentifiabilityError(
            "structure optimization is unfinished or unresolved", result
        )
    point = result.best_converged
    sensitivity = native_sensitivity(predict, objective, parameters, point.parameter_values)
    singular = sensitivity["singular_values"]
    count = len(parameters)
    numerical_limit = 64 * np.finfo(float).eps * max(sensitivity["jacobian"].shape) * singular[0]
    practical_limit = max(numerical_limit, sensitivity_relative_tolerance * singular[0])
    numerical_rank = int(np.count_nonzero(singular > numerical_limit))
    rank = int(np.count_nonzero(singular > practical_limit))
    condition = (
        float(singular[0] / singular[-1])
        if len(singular) == count and singular[-1] > 0
        else math.inf
    )
    result.sensitivity = dict(
        **sensitivity,
        rank=rank,
        numerical_rank=numerical_rank,
        condition=condition,
        relative_rank_tolerance=sensitivity_relative_tolerance,
        maximum_condition=maximum_sensitivity_condition,
    )
    if rank != count or numerical_rank != count or condition > maximum_sensitivity_condition:
        raise StructureRegionIdentifiabilityError(
            "data sensitivity is rank deficient or ill-conditioned", result
        )
    strength = model.parameterization.bind_strength(point.parameter_values)
    revision = getattr(strength, "structure_model_revision", None)
    if not isinstance(revision, str) or len(revision) != 64:
        raise RuntimeError("fitted strength omitted its model revision")
    result.fitted_structure_model_revision = revision
    result.fitted_native_count = point.scale[observations.dataset_index] * model.predict_mass_A2(
        point.parameter_values
    )
    result.identification_status = "local_data_rank_passed_not_numerically_qualified"
    return result


__all__ = [
    "FixedMatchedRegionBackground",
    "IntegratedPeakAreaProjection",
    "MatchedRegionObservations",
    "ParameterizedStructureRegionModel",
    "StructureRegionIdentifiabilityError",
    "StructureRegionResponseBlock",
    "condition_matched_region_background_from_anchors",
    "condition_matched_region_model_from_anchors",
    "fit_structure_regions",
    "prepare_matched_region_objective",
]
