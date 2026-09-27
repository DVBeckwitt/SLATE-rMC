"""Frozen mosaic profile definitions and continuous finite-bin evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.types import Rod
from rasim_next.core.layer_order import CommensurateLayerOrder
from rasim_next.fitting.geometry import LayerLMarkerObservations
from rasim_next.geometry.angles import (
    AngleFrame,
    detector_coordinates_to_angles,
)
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.measurement import evaluate_continuous_per_rod_angle_signal

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _readonly_float(
    value: ArrayLike,
    shape: tuple[int | None, ...],
    name: str,
    *,
    nonnegative: bool = False,
) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.ndim != len(shape) or any(
        expected is not None and array.shape[axis] != expected
        for axis, expected in enumerate(shape)
    ):
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)) or (nonnegative and np.any(array < 0.0)):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    array.setflags(write=False)
    return array


def _readonly_bool(value: ArrayLike, shape: tuple[int, ...], name: str) -> BoolArray:
    supplied = np.asarray(value)
    if supplied.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must contain boolean values")
    array = np.array(supplied, dtype=np.bool_, copy=True, order="C")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MosaicReflectionGroupKey:
    """Material-neutral physical-rod membership and reflection identity."""

    group_id: str
    rod_catalog_revision: str
    member_rod_hk: tuple[tuple[int, int], ...]
    branch_mode: str
    layered_family_m: int | None = None
    layered_integer_L: int | None = None
    layered_layer_order: CommensurateLayerOrder | None = None
    layered_reciprocal_basis_revision: str | None = None

    def __post_init__(self) -> None:
        for name in ("group_id", "rod_catalog_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        rods = tuple(tuple(item) for item in self.member_rod_hk)
        canonical: list[tuple[int, int]] = []
        for item in rods:
            if len(item) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in item
            ):
                raise ValueError("each member rod must be an integer (h, k) pair")
            canonical.append((int(item[0]), int(item[1])))
        canonical_rods = tuple(sorted(canonical))
        if not canonical_rods or len(set(canonical_rods)) != len(canonical_rods):
            raise ValueError("member_rod_hk must contain unique physical rods")
        if self.branch_mode not in {"EXPLICIT_NONZERO", "COLLAPSED_00L"}:
            raise ValueError("branch_mode must be EXPLICIT_NONZERO or COLLAPSED_00L")
        family_m = self.layered_family_m
        integer_l = self.layered_integer_L
        layer_order = self.layered_layer_order
        basis_revision = self.layered_reciprocal_basis_revision
        has_integer_metadata = integer_l is not None
        has_exact_metadata = layer_order is not None or basis_revision is not None
        if family_m is None and (has_integer_metadata or has_exact_metadata):
            raise ValueError("layered display metadata must be supplied together")
        if family_m is not None:
            if (
                isinstance(family_m, bool)
                or not isinstance(family_m, (int, np.integer))
                or int(family_m) < 0
            ):
                raise ValueError("layered_family_m must be a nonnegative integer")
            object.__setattr__(self, "layered_family_m", int(family_m))
            if (self.branch_mode == "COLLAPSED_00L") != (int(family_m) == 0):
                raise ValueError("layered_family_m disagrees with the branch mode")
            if has_integer_metadata == has_exact_metadata:
                raise ValueError(
                    "layered metadata requires exactly one integer-L or exact layer-order identity"
                )
            if has_integer_metadata:
                if isinstance(integer_l, bool) or not isinstance(integer_l, (int, np.integer)):
                    raise ValueError("layered_integer_L must be an integer")
                object.__setattr__(self, "layered_integer_L", int(integer_l))
                if self.branch_mode == "COLLAPSED_00L" and int(integer_l) <= 0:
                    raise ValueError("collapsed m=0 groups require one positive |L| identity")
            else:
                if not isinstance(layer_order, CommensurateLayerOrder):
                    raise TypeError("layered_layer_order must be CommensurateLayerOrder")
                if (
                    not isinstance(basis_revision, str)
                    or len(basis_revision) != 64
                    or any(character not in "0123456789abcdef" for character in basis_revision)
                ):
                    raise ValueError(
                        "layered_reciprocal_basis_revision must be a lowercase SHA-256 revision"
                    )
                if self.branch_mode != "EXPLICIT_NONZERO":
                    raise ValueError("exact layer-order metadata is restricted to nonzero profiles")
        object.__setattr__(self, "member_rod_hk", canonical_rods)


@dataclass(frozen=True, slots=True)
class MosaicProfileIdentity:
    """One fixed incidence/group/branch profile in the global residual."""

    dataset_id: str
    incidence_angle_rad: float
    group_key: MosaicReflectionGroupKey
    branch_id: int | None
    analytic_branch_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a nonempty string")
        incidence = float(self.incidence_angle_rad)
        if not math.isfinite(incidence):
            raise ValueError("incidence_angle_rad must be finite")
        if not isinstance(self.group_key, MosaicReflectionGroupKey):
            raise TypeError("group_key must be a MosaicReflectionGroupKey")
        branch = self.branch_id
        analytic_branch = self.analytic_branch_id
        if self.group_key.branch_mode == "COLLAPSED_00L":
            if branch is not None or analytic_branch != 0:
                raise ValueError(
                    "collapsed profiles require branch_id=None and analytic_branch_id=0"
                )
        elif (
            isinstance(branch, bool)
            or branch not in {1, 2}
            or isinstance(analytic_branch, bool)
            or analytic_branch not in {1, 2}
        ):
            raise ValueError(
                "explicit nonzero profiles require branch_id and analytic_branch_id in {1, 2}"
            )
        object.__setattr__(self, "incidence_angle_rad", incidence)
        object.__setattr__(self, "branch_id", None if branch is None else int(branch))
        object.__setattr__(
            self,
            "analytic_branch_id",
            int(analytic_branch),
        )


@dataclass(frozen=True, slots=True)
class MosaicProfileDefinition:
    """Fixed finite angular bins for one physical peak profile."""

    identity: MosaicProfileIdentity
    center_two_theta_rad: float
    center_phi_rad: float
    two_theta_half_width_rad: float
    phi_half_width_rad: float
    phi_bin_count: int
    two_theta_gauss_order: int = 2
    phi_gauss_order: int = 2
    excluded_phi_bin_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, MosaicProfileIdentity):
            raise TypeError("identity must be a MosaicProfileIdentity")
        for name in (
            "center_two_theta_rad",
            "center_phi_rad",
            "two_theta_half_width_rad",
            "phi_half_width_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if (
            self.two_theta_half_width_rad <= 0.0
            or self.center_two_theta_rad - self.two_theta_half_width_rad < 0.0
            or self.center_two_theta_rad + self.two_theta_half_width_rad > math.pi
        ):
            raise ValueError("the two-theta profile band must lie inside [0, pi]")
        if not 0.0 < self.phi_half_width_rad <= math.pi:
            raise ValueError("phi_half_width_rad must lie in (0, pi]")
        for name in ("phi_bin_count", "two_theta_gauss_order", "phi_gauss_order"):
            value = getattr(self, name)
            minimum = 3 if name == "phi_bin_count" else 2
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            if name != "phi_bin_count" and int(value) % 2:
                raise ValueError(f"{name} must be even to avoid center caustics")
            object.__setattr__(self, name, int(value))
        excluded = tuple(self.excluded_phi_bin_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, (int, np.integer))
            for index in excluded
        ):
            raise TypeError("excluded_phi_bin_indices must contain integers")
        excluded = tuple(int(index) for index in excluded)
        if (
            len(set(excluded)) != len(excluded)
            or excluded != tuple(sorted(excluded))
            or any(index < 0 or index >= self.phi_bin_count for index in excluded)
        ):
            raise ValueError(
                "excluded_phi_bin_indices must be sorted, unique, and inside the profile"
            )
        if self.phi_bin_count - len(excluded) < 3:
            raise ValueError("each profile must retain at least three included phi bins")
        object.__setattr__(self, "excluded_phi_bin_indices", excluded)


def build_layer_l_mosaic_profile_definitions(
    observations: LayerLMarkerObservations | None,
    *,
    required_reciprocal_basis_revision: str,
    dataset_id: str,
    incidence_angle_rad: float,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    rod_catalog_revision: str,
    two_theta_half_width_rad: float,
    phi_half_width_rad: float,
    phi_bin_count: int,
    two_theta_gauss_order: int = 2,
    phi_gauss_order: int = 2,
) -> tuple[MosaicProfileDefinition, ...]:
    """Map frozen exact-layer centroids into fixed mosaic-profile definitions.

    The input is an already selected geometry observation pack. Parent support remains upstream
    provenance and is never expanded into extra profiles or used as an intensity weight here.
    ``None`` represents an absent optional pack and contributes no mosaic residual.
    """

    if observations is None:
        return ()
    if not isinstance(observations, LayerLMarkerObservations):
        raise TypeError("observations must be LayerLMarkerObservations or None")
    if any(
        key.reciprocal_basis_revision != required_reciprocal_basis_revision
        for key in observations.keys
    ):
        raise ValueError("layer-L observations changed the required reciprocal-basis revision")
    angles = detector_coordinates_to_angles(
        observations.coordinates_px[:, 0],
        observations.coordinates_px[:, 1],
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not np.all(angles.valid & angles.azimuth_valid):
        raise ValueError("a layer-L mosaic centroid has no valid nonpolar angle coordinate")

    definitions: list[MosaicProfileDefinition] = []
    for index, marker in enumerate(observations.definitions):
        key = marker.key
        order = key.layer_order
        group_id = (
            f"layer-L:m={key.family_m}:L={order.numerator}/{order.denominator}:"
            f"basis={key.reciprocal_basis_revision}"
        )
        definitions.append(
            MosaicProfileDefinition(
                identity=MosaicProfileIdentity(
                    dataset_id=dataset_id,
                    incidence_angle_rad=incidence_angle_rad,
                    group_key=MosaicReflectionGroupKey(
                        group_id=group_id,
                        rod_catalog_revision=rod_catalog_revision,
                        member_rod_hk=marker.contributing_rod_hk,
                        branch_mode="EXPLICIT_NONZERO",
                        layered_family_m=key.family_m,
                        layered_layer_order=order,
                        layered_reciprocal_basis_revision=key.reciprocal_basis_revision,
                    ),
                    branch_id=key.tag_branch,
                    analytic_branch_id=key.branch,
                ),
                center_two_theta_rad=float(angles.two_theta_rad[index]),
                center_phi_rad=float(angles.phi_rad[index]),
                two_theta_half_width_rad=two_theta_half_width_rad,
                phi_half_width_rad=phi_half_width_rad,
                phi_bin_count=phi_bin_count,
                two_theta_gauss_order=two_theta_gauss_order,
                phi_gauss_order=phi_gauss_order,
            )
        )
    return tuple(definitions)


class _AllRootDetector(Protocol):
    @property
    def instrument(self) -> CompiledInstrument: ...

    @property
    def rods(self) -> tuple[Rod, ...]: ...

    @property
    def rod_catalog_revision(self) -> str: ...

    def evaluate_detector_coordinates_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
    ) -> object: ...


def _mosaic_profile_quadrature(
    definitions: tuple[MosaicProfileDefinition, ...],
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray, FloatArray, FloatArray]:
    """Build the one authoritative tensor rule for fixed angular profile bins."""

    layouts = {
        (item.phi_bin_count, item.two_theta_gauss_order, item.phi_gauss_order)
        for item in definitions
    }
    if len(layouts) != 1:
        raise ValueError("profile quadrature requires one shared layout")
    phi_bin_count, theta_order, phi_order = next(iter(layouts))
    theta_node, theta_weight = np.polynomial.legendre.leggauss(theta_order)
    phi_node, phi_weight = np.polynomial.legendre.leggauss(phi_order)
    shape = (len(definitions), phi_bin_count, theta_order, phi_order)
    two_theta = np.empty(shape, dtype=np.float64)
    phi = np.empty(shape, dtype=np.float64)
    integration_weight = np.empty(shape, dtype=np.float64)
    included_node = np.ones(shape, dtype=np.bool_)
    phi_bin_edges = np.empty((len(definitions), phi_bin_count + 1), dtype=np.float64)
    two_theta_bounds = np.empty((len(definitions), 2), dtype=np.float64)
    for profile_index, definition in enumerate(definitions):
        mapped_theta = (
            definition.center_two_theta_rad + definition.two_theta_half_width_rad * theta_node
        )
        mapped_theta_weight = definition.two_theta_half_width_rad * theta_weight
        phi_edges = np.linspace(
            definition.center_phi_rad - definition.phi_half_width_rad,
            definition.center_phi_rad + definition.phi_half_width_rad,
            phi_bin_count + 1,
        )
        phi_bin_edges[profile_index] = phi_edges
        two_theta_bounds[profile_index] = (
            definition.center_two_theta_rad - definition.two_theta_half_width_rad,
            definition.center_two_theta_rad + definition.two_theta_half_width_rad,
        )
        phi_midpoint = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        phi_bin_half_width = 0.5 * (phi_edges[1] - phi_edges[0])
        mapped_phi = phi_midpoint[:, None] + phi_bin_half_width * phi_node[None, :]
        mapped_phi_weight = phi_bin_half_width * phi_weight
        two_theta[profile_index] = np.broadcast_to(
            mapped_theta[None, :, None],
            shape[1:],
        )
        phi[profile_index] = np.broadcast_to(
            mapped_phi[:, None, :],
            shape[1:],
        )
        integration_weight[profile_index] = np.broadcast_to(
            mapped_theta_weight[None, :, None] * mapped_phi_weight[None, None, :],
            shape[1:],
        )
        if definition.excluded_phi_bin_indices:
            included_node[profile_index, definition.excluded_phi_bin_indices, :, :] = False
    return (
        two_theta,
        phi,
        integration_weight,
        included_node,
        phi_bin_edges,
        two_theta_bounds,
    )


def evaluate_continuous_mosaic_profiles(
    detector: _AllRootDetector,
    *,
    angle_frame: AngleFrame,
    definitions: tuple[MosaicProfileDefinition, ...],
    profile_revision: str,
    execution_backend: str = "cpu",
) -> MosaicProfileSet:
    """Integrate selected angle bins directly from the continuous all-root detector.

    Detector signal and detector-area normalization are integrated independently over every
    finite bin. The returned intensity is therefore ``sum(S) / sum(N)`` and no detector raster
    or pointwise pre-normalized image is used.
    """

    instrument = getattr(detector, "instrument", None)
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("detector must expose a CompiledInstrument")
    rods = tuple(getattr(detector, "rods", ()))
    if not rods or any(not isinstance(rod, Rod) for rod in rods):
        raise TypeError("detector must expose its physical rods")
    rod_catalog_revision = getattr(detector, "rod_catalog_revision", None)
    if not isinstance(rod_catalog_revision, str) or not rod_catalog_revision:
        raise TypeError("detector must expose a nonempty rod catalog revision")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    frozen = tuple(definitions)
    if not frozen or any(not isinstance(item, MosaicProfileDefinition) for item in frozen):
        raise ValueError("definitions must contain MosaicProfileDefinition values")
    if len({item.identity for item in frozen}) != len(frozen):
        raise ValueError("mosaic profile definitions must have unique identities")
    dataset_ids = {item.identity.dataset_id for item in frozen}
    if len(dataset_ids) != 1:
        raise ValueError("one continuous profile evaluation may contain only one dataset")
    if len({item.identity.incidence_angle_rad for item in frozen}) != 1:
        raise ValueError("one dataset must contain exactly one incidence angle")
    definition_catalog_revisions = {item.identity.group_key.rod_catalog_revision for item in frozen}
    if definition_catalog_revisions != {rod_catalog_revision}:
        raise ValueError("profile definitions do not match the detector rod catalog revision")
    layouts = {
        (item.phi_bin_count, item.two_theta_gauss_order, item.phi_gauss_order) for item in frozen
    }
    if len(layouts) != 1:
        bin_counts = {layout[0] for layout in layouts}
        if len(bin_counts) != 1:
            raise ValueError("batched mosaic profiles must share one phi bin count")
        grouped_indices: dict[tuple[int, int, int], list[int]] = {}
        for index, definition in enumerate(frozen):
            layout = (
                definition.phi_bin_count,
                definition.two_theta_gauss_order,
                definition.phi_gauss_order,
            )
            grouped_indices.setdefault(layout, []).append(index)
        partitions = tuple(
            (
                np.asarray(indices, dtype=np.int64),
                evaluate_continuous_mosaic_profiles(
                    detector,
                    angle_frame=angle_frame,
                    definitions=tuple(frozen[index] for index in indices),
                    profile_revision=profile_revision,
                    execution_backend=execution_backend,
                ),
            )
            for _, indices in sorted(grouped_indices.items())
        )
        source_revisions = {profile.source_revision for _, profile in partitions}
        execution = {
            (profile.execution_backend, profile.execution_device) for _, profile in partitions
        }
        if len(source_revisions) != 1 or len(execution) != 1:
            raise ValueError("quadrature partitions changed source or execution provenance")
        phi_bin_count = next(iter(bin_counts))
        signal = np.empty((len(frozen), phi_bin_count), dtype=np.float64)
        normalization = np.empty_like(signal)
        valid = np.empty(signal.shape, dtype=np.bool_)
        phi_bin_edges = np.empty((len(frozen), phi_bin_count + 1), dtype=np.float64)
        two_theta_bounds = np.empty((len(frozen), 2), dtype=np.float64)
        angle_frame_revisions = [""] * len(frozen)
        for indices, profile in partitions:
            signal[indices] = profile.signal
            normalization[indices] = profile.normalization
            valid[indices] = profile.valid
            phi_bin_edges[indices] = profile.phi_bin_edges_rad
            two_theta_bounds[indices] = profile.two_theta_bounds_rad
            for index, revision in zip(indices, profile.angle_frame_revisions, strict=True):
                angle_frame_revisions[int(index)] = revision
        execution_backend_id, execution_device = next(iter(execution))
        return MosaicProfileSet(
            identities=tuple(item.identity for item in frozen),
            signal=signal,
            normalization=normalization,
            valid=valid,
            profile_revision=profile_revision,
            phi_bin_edges_rad=phi_bin_edges,
            two_theta_bounds_rad=two_theta_bounds,
            angle_frame_revisions=tuple(angle_frame_revisions),
            source_revision=next(iter(source_revisions)),
            execution_backend=execution_backend_id,
            execution_device=execution_device,
        )
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    if not isinstance(profile_revision, str) or not profile_revision:
        raise ValueError("profile_revision must be a nonempty string")

    requested_hk = {
        rod_hk for definition in frozen for rod_hk in definition.identity.group_key.member_rod_hk
    }
    configured_by_hk = {(rod.h, rod.k): rod for rod in rods}
    missing_hk = requested_hk - configured_by_hk.keys()
    if missing_hk:
        raise ValueError(f"profile references unconfigured rods {sorted(missing_hk)}")
    evaluation_rods = tuple(rod for rod in rods if (rod.h, rod.k) in requested_hk)
    restrict_rods = getattr(detector, "restrict_rods", None)
    if callable(restrict_rods):
        evaluation_detector = restrict_rods(evaluation_rods)
        evaluated_rods = evaluation_rods
    else:
        evaluation_detector = detector
        evaluated_rods = rods

    phi_bin_count, theta_order, phi_order = next(iter(layouts))
    (
        two_theta,
        phi,
        integration_weight,
        included_node,
        phi_bin_edges,
        two_theta_bounds,
    ) = _mosaic_profile_quadrature(frozen)
    shape = (len(frozen), phi_bin_count, theta_order, phi_order)

    included_flat = np.flatnonzero(included_node.ravel())
    angle_values = evaluate_continuous_per_rod_angle_signal(
        evaluation_detector,
        angle_frame=angle_frame,
        two_theta_rad=two_theta.ravel()[included_flat],
        phi_rad=phi.ravel()[included_flat],
        execution_backend=execution_backend,
    )
    if (
        angle_values.rods != evaluated_rods
        or angle_values.rod_catalog_revision != rod_catalog_revision
        or angle_values.root_policy != "all_retained_roots.v1"
    ):
        raise ValueError("angle values changed the ordered rod catalog or root policy")
    compact_valid = np.flatnonzero(angle_values.valid)
    flat_valid = included_flat[compact_valid]
    normalization_density = np.zeros(shape, dtype=np.float64)
    normalization_density.ravel()[flat_valid] = (
        angle_values.normalization_density_px2_per_rad2.ravel()[compact_valid]
    )
    signal_density = np.zeros(shape, dtype=np.float64)
    if flat_valid.size:
        per_rod_signal = angle_values.per_rod_signal_density_A2_per_rad2.reshape(
            -1,
            len(evaluated_rods),
        )
        caustic = angle_values.caustic.reshape(-1, len(evaluated_rods))
        rod_lookup = {(rod.h, rod.k): index for index, rod in enumerate(angle_values.rods)}
        if len(rod_lookup) != len(evaluated_rods):
            raise ValueError("detector rods must have unique (h, k) identities")
        profile_at_node = np.broadcast_to(
            np.arange(len(frozen), dtype=np.int64)[:, None, None, None],
            shape,
        ).ravel()[flat_valid]
        flat_signal_density = signal_density.ravel()
        for profile_index, definition in enumerate(frozen):
            try:
                rod_indices = np.asarray(
                    [rod_lookup[rod_hk] for rod_hk in definition.identity.group_key.member_rod_hk],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(f"profile references unconfigured rod {error.args[0]}") from error
            selected_rows = np.flatnonzero(profile_at_node == profile_index)
            selected_nodes = flat_valid[selected_rows]
            selected_compact_nodes = compact_valid[selected_rows]
            if np.any(caustic[np.ix_(selected_compact_nodes, rod_indices)]):
                raise FloatingPointError("an angle-profile quadrature node lies on a caustic")
            flat_signal_density[selected_nodes] = np.sum(
                per_rod_signal[np.ix_(selected_compact_nodes, rod_indices)],
                axis=1,
                dtype=np.float64,
            )

    weighted_normalization = normalization_density * integration_weight
    signal = np.sum(
        signal_density * integration_weight,
        axis=(2, 3),
        dtype=np.float64,
    )
    normalization = np.sum(
        weighted_normalization,
        axis=(2, 3),
        dtype=np.float64,
    )
    valid = normalization > 0.0
    for profile_index, definition in enumerate(frozen):
        if definition.excluded_phi_bin_indices:
            excluded = np.asarray(definition.excluded_phi_bin_indices, dtype=np.int64)
            signal[profile_index, excluded] = 0.0
            normalization[profile_index, excluded] = 0.0
            valid[profile_index, excluded] = False
    signal[~valid] = 0.0
    normalization[~valid] = 0.0
    return MosaicProfileSet(
        identities=tuple(item.identity for item in frozen),
        signal=signal,
        normalization=normalization,
        valid=valid,
        profile_revision=profile_revision,
        phi_bin_edges_rad=phi_bin_edges,
        two_theta_bounds_rad=two_theta_bounds,
        angle_frame_revisions=(angle_frame.revision,) * len(frozen),
        source_revision=angle_values.source_revision,
        execution_backend=angle_values.execution_backend,
        execution_device=angle_values.execution_device,
    )


@dataclass(frozen=True, slots=True)
class MosaicProfileSet:
    """Immutable finite-bin signal and normalization profiles."""

    identities: tuple[MosaicProfileIdentity, ...]
    signal: FloatArray
    normalization: FloatArray
    valid: BoolArray
    profile_revision: str
    phi_bin_edges_rad: FloatArray
    two_theta_bounds_rad: FloatArray
    angle_frame_revisions: tuple[str, ...]
    source_revision: str | None
    execution_backend: str | None = None
    execution_device: str | None = None
    observation_revision: str | None = None

    def __post_init__(self) -> None:
        identities = tuple(self.identities)
        if not identities or any(
            not isinstance(identity, MosaicProfileIdentity) for identity in identities
        ):
            raise ValueError("identities must contain MosaicProfileIdentity values")
        if len(set(identities)) != len(identities):
            raise ValueError("profile identities must be unique")
        incidence_by_dataset: dict[str, set[float]] = {}
        branches_by_dataset_group: dict[
            tuple[str, float, MosaicReflectionGroupKey, int], set[int | None]
        ] = {}
        for identity in identities:
            incidence_by_dataset.setdefault(identity.dataset_id, set()).add(
                identity.incidence_angle_rad
            )
            branches_by_dataset_group.setdefault(
                (
                    identity.dataset_id,
                    identity.incidence_angle_rad,
                    identity.group_key,
                    identity.analytic_branch_id,
                ),
                set(),
            ).add(identity.branch_id)
        if any(len(incidences) != 1 for incidences in incidence_by_dataset.values()):
            raise ValueError("each dataset_id must own exactly one incidence angle")
        for (_, _, group_key, analytic_branch), branches in branches_by_dataset_group.items():
            if group_key.branch_mode == "COLLAPSED_00L" and branches != {None}:
                raise ValueError("collapsed groups must contain exactly one branchless profile")
            if group_key.branch_mode == "EXPLICIT_NONZERO" and not branches <= {1, 2}:
                raise ValueError("nonzero groups may contain only explicit visible branches")
            if (group_key.branch_mode == "COLLAPSED_00L") != (analytic_branch == 0):
                raise ValueError("analytic branch identity disagrees with the branch mode")
        signal = _readonly_float(
            self.signal,
            (len(identities), None),
            "signal",
            nonnegative=True,
        )
        if signal.shape[1] < 3:
            raise ValueError("each mosaic profile must contain at least three bins")
        phi_edges = _readonly_float(
            self.phi_bin_edges_rad,
            (len(identities), signal.shape[1] + 1),
            "phi_bin_edges_rad",
        )
        if np.any(np.diff(phi_edges, axis=1) <= 0.0):
            raise ValueError("phi_bin_edges_rad must increase strictly within each profile")
        two_theta_bounds = _readonly_float(
            self.two_theta_bounds_rad,
            (len(identities), 2),
            "two_theta_bounds_rad",
        )
        if np.any(
            (two_theta_bounds[:, 0] < 0.0)
            | (two_theta_bounds[:, 1] > math.pi)
            | (two_theta_bounds[:, 0] >= two_theta_bounds[:, 1])
        ):
            raise ValueError("two_theta_bounds_rad must contain ordered intervals inside [0, pi]")
        frame_revisions = tuple(self.angle_frame_revisions)
        if len(frame_revisions) != len(identities) or any(
            not isinstance(revision, str) or not revision for revision in frame_revisions
        ):
            raise ValueError("angle_frame_revisions must identify every profile frame")
        normalization = _readonly_float(
            self.normalization,
            signal.shape,
            "normalization",
            nonnegative=True,
        )
        valid = _readonly_bool(self.valid, signal.shape, "valid")
        if np.any(valid & (normalization <= 0.0)):
            raise ValueError("valid bins require positive normalization")
        if np.any(~valid & ((signal != 0.0) | (normalization != 0.0))):
            raise ValueError("invalid bins must have zero signal and normalization")
        if np.any(np.sum(valid, axis=1) < 3):
            raise ValueError("every profile requires at least three valid bins")
        if not isinstance(self.profile_revision, str) or not self.profile_revision:
            raise ValueError("profile_revision must be a nonempty string")
        source_revision = self.source_revision
        observation_revision = self.observation_revision
        if source_revision is not None and (
            not isinstance(source_revision, str) or not source_revision
        ):
            raise ValueError("source_revision must be None or a nonempty string")
        if observation_revision is not None and (
            not isinstance(observation_revision, str) or not observation_revision
        ):
            raise ValueError("observation_revision must be None or a nonempty string")
        if (source_revision is None) == (observation_revision is None):
            raise ValueError(
                "exactly one of source_revision and observation_revision must be supplied"
            )
        if self.execution_backend is not None and self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
            "numpy_cpu_sparse_source_averaged.v1",
        }:
            raise ValueError("unsupported profile execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        if (self.execution_backend == "numba_cuda_source_averaged.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("execution_device must identify exactly the CUDA profile backend")
        if observation_revision is not None and (
            self.execution_backend is not None or self.execution_device is not None
        ):
            raise ValueError("measured observations must not claim simulation execution provenance")
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "phi_bin_edges_rad", phi_edges)
        object.__setattr__(self, "two_theta_bounds_rad", two_theta_bounds)
        object.__setattr__(self, "angle_frame_revisions", frame_revisions)

    @property
    def intensity(self) -> FloatArray:
        result = np.zeros(self.signal.shape, dtype=np.float64)
        np.divide(self.signal, self.normalization, out=result, where=self.valid)
        result.setflags(write=False)
        return result

    def reorder(self, order: ArrayLike) -> MosaicProfileSet:
        supplied = np.asarray(order)
        expected = np.arange(len(self.identities), dtype=np.int64)
        if supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.integer):
            raise ValueError("profile order must be a one-dimensional integer permutation")
        indices = np.asarray(supplied, dtype=np.int64)
        if indices.shape != expected.shape or not np.array_equal(np.sort(indices), expected):
            raise ValueError("profile order must be a complete permutation")
        return MosaicProfileSet(
            identities=tuple(self.identities[int(index)] for index in indices),
            signal=self.signal[indices],
            normalization=self.normalization[indices],
            valid=self.valid[indices],
            profile_revision=self.profile_revision,
            phi_bin_edges_rad=self.phi_bin_edges_rad[indices],
            two_theta_bounds_rad=self.two_theta_bounds_rad[indices],
            angle_frame_revisions=tuple(
                self.angle_frame_revisions[int(index)] for index in indices
            ),
            source_revision=self.source_revision,
            execution_backend=self.execution_backend,
            execution_device=self.execution_device,
            observation_revision=self.observation_revision,
        )


__all__ = [
    "MosaicProfileDefinition",
    "MosaicProfileIdentity",
    "MosaicProfileSet",
    "MosaicReflectionGroupKey",
    "build_layer_l_mosaic_profile_definitions",
    "evaluate_continuous_mosaic_profiles",
]
