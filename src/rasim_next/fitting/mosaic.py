"""Response-folded mosaic-profile fitting with unknown reflection amplitudes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, direct, minimize_scalar

from painted_ewald.types import Rod
from rasim_next.geometry.angles import AngleFrame
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.measurement import evaluate_continuous_per_rod_angle_signal

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_MAXIMUM_SENSITIVITY_CONDITION = 1.0e8
_MIXTURE_Z_LIMIT = 32.0
_MIXTURE_Z_GRID_COUNT = 8193
_MIXTURE_Z_LENGTH_TOLERANCE = 1.0e-6
_MIXTURE_Z_LOCAL_HALF_WIDTH = 0.05


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
        if (family_m is None) != (integer_l is None):
            raise ValueError("layered display metadata must be supplied together")
        if family_m is not None:
            if (
                isinstance(family_m, bool)
                or not isinstance(family_m, (int, np.integer))
                or int(family_m) < 0
            ):
                raise ValueError("layered_family_m must be a nonnegative integer")
            if isinstance(integer_l, bool) or not isinstance(integer_l, (int, np.integer)):
                raise ValueError("layered_integer_L must be an integer")
            object.__setattr__(self, "layered_family_m", int(family_m))
            object.__setattr__(self, "layered_integer_L", int(integer_l))
            if (self.branch_mode == "COLLAPSED_00L") != (int(family_m) == 0):
                raise ValueError("layered_family_m disagrees with the branch mode")
            if self.branch_mode == "COLLAPSED_00L" and int(integer_l) <= 0:
                raise ValueError("collapsed m=0 groups require one positive |L| identity")
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
        detector,
        angle_frame=angle_frame,
        two_theta_rad=two_theta.ravel()[included_flat],
        phi_rad=phi.ravel()[included_flat],
        execution_backend=execution_backend,
    )
    if (
        angle_values.rods != rods
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
            len(rods),
        )
        caustic = angle_values.caustic.reshape(-1, len(rods))
        rod_lookup = {(rod.h, rod.k): index for index, rod in enumerate(angle_values.rods)}
        if len(rod_lookup) != len(rods):
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
    source_revision: str
    execution_backend: str | None = None
    execution_device: str | None = None

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
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be a nonempty string")
        if self.execution_backend is not None and self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
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
        )


def _validated_widths(value: ArrayLike, name: str) -> FloatArray:
    widths = _readonly_float(value, (None,), name)
    if widths.size < 2 or np.any(widths <= 0.0) or np.any(np.diff(widths) <= 0.0):
        raise ValueError(f"{name} must contain at least two strictly increasing positive widths")
    return widths


@dataclass(frozen=True, slots=True)
class MosaicComponentProfile:
    """One exact pure-component response with explicit kind and width provenance."""

    component_kind: str
    width_rad: float
    profile: MosaicProfileSet

    def __post_init__(self) -> None:
        if self.component_kind not in {"gaussian", "lorentzian"}:
            raise ValueError("component_kind must be gaussian or lorentzian")
        width = float(self.width_rad)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("width_rad must be finite and positive")
        if not isinstance(self.profile, MosaicProfileSet):
            raise TypeError("profile must be a MosaicProfileSet")
        object.__setattr__(self, "width_rad", width)

    def reorder(self, order: ArrayLike) -> MosaicComponentProfile:
        return MosaicComponentProfile(
            component_kind=self.component_kind,
            width_rad=self.width_rad,
            profile=self.profile.reorder(order),
        )


def _validated_component_profiles(
    value: tuple[MosaicComponentProfile, ...],
    widths: FloatArray,
    observations: MosaicProfileSet,
    name: str,
    component_kind: str,
) -> tuple[MosaicComponentProfile, ...]:
    profiles = tuple(value)
    if len(profiles) != widths.size or any(
        not isinstance(profile, MosaicComponentProfile) for profile in profiles
    ):
        raise ValueError(f"{name} must contain one MosaicComponentProfile per width")
    for expected_width, component in zip(widths, profiles, strict=True):
        profile = component.profile
        if (
            component.component_kind != component_kind
            or component.width_rad != float(expected_width)
            or profile.identities != observations.identities
            or profile.profile_revision != observations.profile_revision
            or profile.angle_frame_revisions != observations.angle_frame_revisions
            or profile.execution_backend != observations.execution_backend
            or profile.execution_device != observations.execution_device
            or not np.array_equal(profile.phi_bin_edges_rad, observations.phi_bin_edges_rad)
            or not np.array_equal(
                profile.two_theta_bounds_rad,
                observations.two_theta_bounds_rad,
            )
            or not np.array_equal(profile.normalization, observations.normalization)
            or not np.array_equal(profile.valid, observations.valid)
        ):
            raise ValueError(f"{name} changed component or frozen profile provenance")
    return profiles


@dataclass(frozen=True, slots=True)
class MosaicComponentProfileBank:
    """Exact pure-component profiles on deterministic positive-width grids."""

    observations: MosaicProfileSet
    gaussian_sigma_rad: FloatArray
    gaussian_profiles: tuple[MosaicComponentProfile, ...]
    lorentzian_half_width_rad: FloatArray
    lorentzian_profiles: tuple[MosaicComponentProfile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observations, MosaicProfileSet):
            raise TypeError("observations must be a MosaicProfileSet")
        gaussian_width = _validated_widths(self.gaussian_sigma_rad, "gaussian_sigma_rad")
        lorentzian_width = _validated_widths(
            self.lorentzian_half_width_rad,
            "lorentzian_half_width_rad",
        )
        gaussian = _validated_component_profiles(
            self.gaussian_profiles,
            gaussian_width,
            self.observations,
            "gaussian_profiles",
            "gaussian",
        )
        lorentzian = _validated_component_profiles(
            self.lorentzian_profiles,
            lorentzian_width,
            self.observations,
            "lorentzian_profiles",
            "lorentzian",
        )
        source_revisions = {
            self.observations.source_revision,
            *(component.profile.source_revision for component in (*gaussian, *lorentzian)),
        }
        if len(source_revisions) != 1:
            raise ValueError("component profiles changed source-profile provenance")
        object.__setattr__(self, "gaussian_sigma_rad", gaussian_width)
        object.__setattr__(self, "gaussian_profiles", gaussian)
        object.__setattr__(self, "lorentzian_half_width_rad", lorentzian_width)
        object.__setattr__(self, "lorentzian_profiles", lorentzian)

    def reorder_profiles(self, order: ArrayLike) -> MosaicComponentProfileBank:
        supplied = np.asarray(order)
        expected = np.arange(len(self.observations.identities), dtype=np.int64)
        if supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.integer):
            raise ValueError("profile order must be a one-dimensional integer permutation")
        indices = np.asarray(supplied, dtype=np.int64)
        if indices.shape != expected.shape or not np.array_equal(np.sort(indices), expected):
            raise ValueError("profile order must be a complete permutation")
        return MosaicComponentProfileBank(
            observations=self.observations.reorder(indices),
            gaussian_sigma_rad=self.gaussian_sigma_rad,
            gaussian_profiles=tuple(profile.reorder(indices) for profile in self.gaussian_profiles),
            lorentzian_half_width_rad=self.lorentzian_half_width_rad,
            lorentzian_profiles=tuple(
                profile.reorder(indices) for profile in self.lorentzian_profiles
            ),
        )


class MosaicIdentifiabilityError(ValueError):
    """Raised when nuisance-projected profiles cannot constrain the active model."""

    def __init__(
        self,
        rank: int,
        singular_values: ArrayLike,
        condition: float,
        active_parameter_names: tuple[str, ...],
        *,
        reason: str = "local_sensitivity",
        competing_solution_keys: tuple[tuple[str, int | None, int | None, float], ...] = (),
    ) -> None:
        self.rank = int(rank)
        self.active_parameter_names = tuple(active_parameter_names)
        if not self.active_parameter_names:
            raise ValueError("active_parameter_names must not be empty")
        self.singular_values = _readonly_float(
            singular_values,
            (len(self.active_parameter_names),),
            "singular_values",
            nonnegative=True,
        )
        self.condition = float(condition)
        if reason not in {"local_sensitivity", "global_alias"}:
            raise ValueError("reason must be 'local_sensitivity' or 'global_alias'")
        self.reason = reason
        self.competing_solution_keys = tuple(competing_solution_keys)
        if self.reason == "global_alias" and len(self.competing_solution_keys) < 2:
            raise ValueError("global_alias requires at least two competing solution keys")
        detail = (
            f"global_aliases={self.competing_solution_keys}"
            if self.reason == "global_alias"
            else f"rank={self.rank}/{len(self.active_parameter_names)} "
            f"condition={self.condition:.6g}"
        )
        super().__init__(
            f"mosaic distribution is not identifiable after profile-scale projection: {detail}"
        )


@dataclass(frozen=True, slots=True)
class MosaicProfileFitResult:
    """Best exact-bank widths, fitted probability, profile scales, and rank evidence."""

    gaussian_sigma_rad: float | None
    lorentzian_half_width_rad: float | None
    lorentzian_probability: float
    active_parameter_names: tuple[str, ...]
    profile_identities: tuple[MosaicProfileIdentity, ...]
    profile_scales: FloatArray
    profile_relative_l2_residual: FloatArray
    objective: float
    predicted_intensity: FloatArray
    sensitivity_singular_values: FloatArray
    sensitivity_rank: int
    sensitivity_condition: float
    width_pair_objective: FloatArray = field(repr=False)
    width_pair_eta: FloatArray = field(repr=False)
    gaussian_bank_index: int | None = field(repr=False)
    lorentzian_bank_index: int | None = field(repr=False)
    gaussian_activation_probe_width_rad: float | None = None
    lorentzian_activation_probe_width_rad: float | None = None

    weighting_id: ClassVar[str] = "profile_shape_profiled_scale_angle_intensity_l2.v3"
    eta_search_id: ClassVar[str] = (
        "eta_centered_logit_direct_finite_8193_stationary_audit_exact_faces.v1"
    )

    def __post_init__(self) -> None:
        for name in (
            "lorentzian_probability",
            "objective",
            "sensitivity_condition",
        ):
            value = float(getattr(self, name))
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.lorentzian_probability <= 1.0 or self.objective < 0.0:
            raise ValueError("invalid fitted probability or objective")
        expected_active: tuple[str, ...]
        if self.lorentzian_probability == 0.0:
            expected_active = (
                "log_gaussian_sigma",
                "lorentzian_probability_from_zero",
            )
            if self.gaussian_sigma_rad is None or self.lorentzian_half_width_rad is not None:
                raise ValueError("the pure-Gaussian face must deactivate Lorentzian width")
            if (
                self.gaussian_activation_probe_width_rad is not None
                or self.lorentzian_activation_probe_width_rad is None
            ):
                raise ValueError("the pure-Gaussian face must report its Lorentzian probe width")
        elif self.lorentzian_probability == 1.0:
            expected_active = (
                "log_lorentzian_half_width",
                "gaussian_probability_from_one",
            )
            if self.gaussian_sigma_rad is not None or self.lorentzian_half_width_rad is None:
                raise ValueError("the pure-Lorentzian face must deactivate Gaussian width")
            if (
                self.gaussian_activation_probe_width_rad is None
                or self.lorentzian_activation_probe_width_rad is not None
            ):
                raise ValueError("the pure-Lorentzian face must report its Gaussian probe width")
        else:
            expected_active = (
                "log_gaussian_sigma",
                "log_lorentzian_half_width",
                "logit_lorentzian_probability",
            )
            if self.gaussian_sigma_rad is None or self.lorentzian_half_width_rad is None:
                raise ValueError("the interior mixture requires both component widths")
            if (
                self.gaussian_activation_probe_width_rad is not None
                or self.lorentzian_activation_probe_width_rad is not None
            ):
                raise ValueError("interior fits do not use boundary activation probes")
        for name in (
            "gaussian_sigma_rad",
            "lorentzian_half_width_rad",
            "gaussian_activation_probe_width_rad",
            "lorentzian_activation_probe_width_rad",
        ):
            width = getattr(self, name)
            if width is not None:
                width = float(width)
                if not math.isfinite(width) or width <= 0.0:
                    raise ValueError("active fitted mosaic widths must be finite and positive")
                object.__setattr__(self, name, width)
        active = tuple(self.active_parameter_names)
        if active != expected_active:
            raise ValueError("active_parameter_names do not match the selected mixture face")
        identities = tuple(self.profile_identities)
        if (
            not identities
            or any(not isinstance(identity, MosaicProfileIdentity) for identity in identities)
            or len(set(identities)) != len(identities)
        ):
            raise ValueError("profile_identities must contain unique profile identities")
        scales = _readonly_float(
            self.profile_scales,
            (len(identities),),
            "profile_scales",
            nonnegative=True,
        )
        profile_residual = _readonly_float(
            self.profile_relative_l2_residual,
            (len(identities),),
            "profile_relative_l2_residual",
            nonnegative=True,
        )
        residual_objective = float(profile_residual @ profile_residual)
        if not math.isclose(
            self.objective,
            residual_objective,
            rel_tol=2.0e-12,
            abs_tol=2.0e-14 * max(1.0, residual_objective),
        ):
            raise ValueError("objective must equal the summed squared profile residuals")
        predicted = _readonly_float(
            self.predicted_intensity,
            (len(identities), None),
            "predicted_intensity",
            nonnegative=True,
        )
        singular = _readonly_float(
            self.sensitivity_singular_values,
            (len(active),),
            "sensitivity_singular_values",
            nonnegative=True,
        )
        if self.sensitivity_rank != len(active) or not math.isfinite(self.sensitivity_condition):
            raise ValueError("an accepted fit result must have full-rank active sensitivity")
        objective_surface = _readonly_float(
            self.width_pair_objective,
            (None, None),
            "width_pair_objective",
            nonnegative=True,
        )
        eta_surface = _readonly_float(
            self.width_pair_eta,
            objective_surface.shape,
            "width_pair_eta",
            nonnegative=True,
        )
        if np.any(eta_surface > 1.0):
            raise ValueError("width_pair_eta must lie in [0, 1]")
        object.__setattr__(self, "profile_identities", identities)
        object.__setattr__(self, "active_parameter_names", active)
        object.__setattr__(self, "profile_scales", scales)
        object.__setattr__(self, "profile_relative_l2_residual", profile_residual)
        object.__setattr__(self, "predicted_intensity", predicted)
        object.__setattr__(self, "sensitivity_singular_values", singular)
        object.__setattr__(self, "width_pair_objective", objective_surface)
        object.__setattr__(self, "width_pair_eta", eta_surface)

    def scale_for_profile(self, identity: MosaicProfileIdentity) -> float:
        if not isinstance(identity, MosaicProfileIdentity):
            raise TypeError("identity must be a MosaicProfileIdentity")
        try:
            index = self.profile_identities.index(identity)
        except ValueError as error:
            raise KeyError(identity) from error
        return float(self.profile_scales[index])


def _reflection_group_sort_key(group: MosaicReflectionGroupKey) -> tuple[object, ...]:
    return (
        group.rod_catalog_revision,
        group.group_id,
        group.member_rod_hk,
        group.branch_mode,
        -1 if group.layered_family_m is None else group.layered_family_m,
        0 if group.layered_integer_L is None else group.layered_integer_L,
    )


def _canonical_profile_order(
    identities: tuple[MosaicProfileIdentity, ...],
) -> NDArray[np.int64]:
    return np.asarray(
        sorted(
            range(len(identities)),
            key=lambda index: (
                _reflection_group_sort_key(identities[index].group_key),
                identities[index].dataset_id,
                identities[index].incidence_angle_rad,
                identities[index].analytic_branch_id,
                -1 if identities[index].branch_id is None else identities[index].branch_id,
            ),
        ),
        dtype=np.int64,
    )


def _valid_fit_vectors(
    observations: MosaicProfileSet,
) -> tuple[NDArray[np.int64], FloatArray, FloatArray, FloatArray]:
    valid_profile = np.broadcast_to(
        np.arange(len(observations.identities), dtype=np.int64)[:, None],
        observations.valid.shape,
    )[observations.valid]
    observed = observations.intensity[observations.valid]
    if not np.all(np.isfinite(observed)):
        raise ValueError("valid observed intensities must be finite")
    observed_scale = np.zeros(len(observations.identities), dtype=np.float64)
    np.maximum.at(observed_scale, valid_profile, observed)
    if np.any(observed_scale <= 0.0):
        raise ValueError("every profile requires nonzero observed signal")
    observed = observed / observed_scale[valid_profile]
    profile_energy = np.bincount(
        valid_profile,
        weights=observed * observed,
        minlength=len(observations.identities),
    )
    if np.any(profile_energy <= np.finfo(np.float64).tiny):
        raise ValueError("every profile requires nonzero observed energy")
    return valid_profile, observed, 1.0 / profile_energy[valid_profile], observed_scale


def _profiled_scales_and_objective(
    model: FloatArray,
    observed: FloatArray,
    weight: FloatArray,
    valid_profile: NDArray[np.int64],
    profile_count: int,
) -> tuple[FloatArray, float]:
    weighted_model = weight * model
    denominator = np.bincount(
        valid_profile,
        weights=weighted_model * model,
        minlength=profile_count,
    )
    if np.any(denominator <= np.finfo(np.float64).tiny):
        return np.zeros(profile_count, dtype=np.float64), math.inf
    numerator = np.bincount(
        valid_profile,
        weights=weighted_model * observed,
        minlength=profile_count,
    )
    scales = np.maximum(0.0, numerator / denominator)
    residual = scales[valid_profile] * model - observed
    return scales, float(weight @ (residual * residual))


def _best_eta(
    gaussian: FloatArray,
    lorentzian: FloatArray,
    observed: FloatArray,
    weight: FloatArray,
    valid_profile: NDArray[np.int64],
    profile_count: int,
    *,
    audit_all_minima: bool = False,
) -> tuple[float, FloatArray, float, tuple[float, ...]]:
    gaussian_reference = float(np.max(np.abs(gaussian)))
    lorentzian_reference = float(np.max(np.abs(lorentzian)))
    if gaussian_reference <= 0.0 or lorentzian_reference <= 0.0:
        raise ValueError("component profiles must contain positive finite intensity")
    normalized_gaussian = gaussian / gaussian_reference
    normalized_lorentzian = lorentzian / lorentzian_reference
    log_component_ratio = math.log(lorentzian_reference) - math.log(gaussian_reference)
    cache: dict[float, tuple[FloatArray, float]] = {}

    def sigmoid(value: float | FloatArray) -> float | FloatArray:
        supplied = np.asarray(value, dtype=np.float64)
        positive = supplied >= 0.0
        result = np.empty(supplied.shape, dtype=np.float64)
        result[positive] = 1.0 / (1.0 + np.exp(-supplied[positive]))
        exponential = np.exp(supplied[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return float(result) if result.ndim == 0 else result

    def physical_eta(centered_logit: float) -> float:
        return float(sigmoid(centered_logit - log_component_ratio))

    def evaluated_centered_logit(centered_logit: float) -> tuple[FloatArray, float]:
        key = float(centered_logit)
        if key not in cache:
            intrinsic_probability = float(sigmoid(key))
            model = (
                1.0 - intrinsic_probability
            ) * normalized_gaussian + intrinsic_probability * normalized_lorentzian
            cache[key] = _profiled_scales_and_objective(
                model,
                observed,
                weight,
                valid_profile,
                profile_count,
            )
        return cache[key]

    def grouped(values: FloatArray) -> FloatArray:
        return np.bincount(valid_profile, weights=values, minlength=profile_count)

    gaussian_observed = grouped(weight * normalized_gaussian * observed)
    lorentzian_observed = grouped(weight * normalized_lorentzian * observed)
    gaussian_gaussian = grouped(weight * normalized_gaussian * normalized_gaussian)
    gaussian_lorentzian = grouped(weight * normalized_gaussian * normalized_lorentzian)
    lorentzian_lorentzian = grouped(weight * normalized_lorentzian * normalized_lorentzian)
    observed_energy = grouped(weight * observed * observed)

    def coefficient_objective(centered_logit_array: FloatArray) -> float:
        intrinsic_probability = float(sigmoid(float(centered_logit_array[0])))
        gaussian_probability = 1.0 - intrinsic_probability
        numerator = np.maximum(
            0.0,
            gaussian_probability * gaussian_observed + intrinsic_probability * lorentzian_observed,
        )
        denominator = (
            gaussian_probability * gaussian_probability * gaussian_gaussian
            + 2.0 * gaussian_probability * intrinsic_probability * gaussian_lorentzian
            + intrinsic_probability * intrinsic_probability * lorentzian_lorentzian
        )
        if np.any(denominator <= np.finfo(np.float64).tiny):
            return math.inf
        group_objective = observed_energy - numerator * numerator / denominator
        return float(np.sum(np.maximum(0.0, group_objective)))

    def scalar_coefficient_objective(centered_logit: float) -> float:
        return coefficient_objective(np.asarray((centered_logit,), dtype=np.float64))

    def scalar_coefficient_derivative(centered_logit: float) -> float:
        intrinsic_probability = float(sigmoid(centered_logit))
        gaussian_probability = 1.0 - intrinsic_probability
        numerator = (
            gaussian_probability * gaussian_observed + intrinsic_probability * lorentzian_observed
        )
        denominator = (
            gaussian_probability * gaussian_probability * gaussian_gaussian
            + 2.0 * gaussian_probability * intrinsic_probability * gaussian_lorentzian
            + intrinsic_probability * intrinsic_probability * lorentzian_lorentzian
        )
        numerator_derivative = lorentzian_observed - gaussian_observed
        denominator_derivative = 2.0 * (
            gaussian_lorentzian
            - gaussian_gaussian
            + intrinsic_probability
            * (gaussian_gaussian - 2.0 * gaussian_lorentzian + lorentzian_lorentzian)
        )
        derivative_in_probability = -np.sum(
            numerator
            * (2.0 * numerator_derivative * denominator - numerator * denominator_derivative)
            / (denominator * denominator)
        )
        return float(
            intrinsic_probability * (1.0 - intrinsic_probability) * derivative_in_probability
        )

    def polish_centered_logit(lower: float, upper: float) -> tuple[float, float]:
        polished = minimize_scalar(
            lambda centered_logit: evaluated_centered_logit(float(centered_logit))[1],
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1.0e-12, "maxiter": 128},
        )
        centered_logit = float(polished.x)
        objective = float(polished.fun)
        radius = max(1.0e-10, 1.0e-6 * (upper - lower))
        for _ in range(32):
            bracket_lower = max(lower, centered_logit - radius)
            bracket_upper = min(upper, centered_logit + radius)
            lower_derivative = scalar_coefficient_derivative(bracket_lower)
            upper_derivative = scalar_coefficient_derivative(bracket_upper)
            if lower_derivative <= 0.0 <= upper_derivative:
                stationary_logit = brentq(
                    scalar_coefficient_derivative,
                    bracket_lower,
                    bracket_upper,
                    xtol=5.0e-15,
                    rtol=4.0 * np.finfo(np.float64).eps,
                    maxiter=128,
                )
                stationary_objective = evaluated_centered_logit(stationary_logit)[1]
                polish_tolerance = (
                    512.0
                    * np.finfo(np.float64).eps
                    * max(
                        1.0,
                        float(profile_count),
                        abs(objective),
                        abs(stationary_objective),
                    )
                )
                if stationary_objective <= objective + polish_tolerance:
                    return stationary_logit, stationary_objective
                break
            if bracket_lower == lower and bracket_upper == upper:
                break
            radius *= 2.0
        return centered_logit, objective

    def globally_relevant_basin(
        objective: float,
        lower: float,
        upper: float,
    ) -> bool:
        face_objective = min(gaussian_objective, lorentzian_objective)
        tolerance = (
            512.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                float(profile_count),
                abs(objective),
                abs(face_objective),
            )
        )
        if objective < face_objective - tolerance:
            return True
        if objective > face_objective + tolerance:
            return False
        endpoint_objective = min(
            evaluated_centered_logit(lower)[1],
            evaluated_centered_logit(upper)[1],
        )
        return objective < endpoint_objective - tolerance

    gaussian_scales, gaussian_objective = _profiled_scales_and_objective(
        normalized_gaussian,
        observed,
        weight,
        valid_profile,
        profile_count,
    )
    lorentzian_scales, lorentzian_objective = _profiled_scales_and_objective(
        normalized_lorentzian,
        observed,
        weight,
        valid_profile,
        profile_count,
    )

    direction_determinant = (
        gaussian_gaussian * lorentzian_lorentzian - gaussian_lorentzian * gaussian_lorentzian
    )
    direction_scale = gaussian_gaussian * lorentzian_lorentzian
    direction_tolerance = 512.0 * np.finfo(np.float64).eps * direction_scale
    candidate_centered_logit: set[float] = set()
    if np.all(np.abs(direction_determinant) <= direction_tolerance):
        pass
    else:
        global_result = direct(
            coefficient_objective,
            ((-_MIXTURE_Z_LIMIT, _MIXTURE_Z_LIMIT),),
            eps=1.0e-10,
            maxfun=2048,
            maxiter=2048,
            locally_biased=True,
            len_tol=_MIXTURE_Z_LENGTH_TOLERANCE,
            vol_tol=1.0e-16,
        )
        if not np.isfinite(global_result.fun) or not np.all(np.isfinite(global_result.x)):
            raise RuntimeError(
                f"adaptive eta search produced no finite candidate: {global_result.message}"
            )
        global_centered_logit = float(global_result.x[0])
        local_lower = max(
            -_MIXTURE_Z_LIMIT,
            global_centered_logit - _MIXTURE_Z_LOCAL_HALF_WIDTH,
        )
        local_upper = min(
            _MIXTURE_Z_LIMIT,
            global_centered_logit + _MIXTURE_Z_LOCAL_HALF_WIDTH,
        )
        local_centered_logit, local_objective = polish_centered_logit(
            local_lower,
            local_upper,
        )
        if abs(
            local_centered_logit
        ) < _MIXTURE_Z_LIMIT - _MIXTURE_Z_LENGTH_TOLERANCE and globally_relevant_basin(
            local_objective, local_lower, local_upper
        ):
            candidate_centered_logit.add(local_centered_logit)
        if audit_all_minima:
            centered_logit_grid = np.linspace(
                -_MIXTURE_Z_LIMIT,
                _MIXTURE_Z_LIMIT,
                _MIXTURE_Z_GRID_COUNT,
            )
            intrinsic_grid = np.asarray(sigmoid(centered_logit_grid))
            gaussian_probability_grid = 1.0 - intrinsic_grid
            numerator_grid = (
                gaussian_probability_grid[:, None] * gaussian_observed
                + intrinsic_grid[:, None] * lorentzian_observed
            )
            denominator_grid = (
                gaussian_probability_grid[:, None] ** 2 * gaussian_gaussian
                + 2.0
                * gaussian_probability_grid[:, None]
                * intrinsic_grid[:, None]
                * gaussian_lorentzian
                + intrinsic_grid[:, None] ** 2 * lorentzian_lorentzian
            )
            objective_grid = np.sum(
                np.maximum(
                    0.0,
                    observed_energy - numerator_grid * numerator_grid / denominator_grid,
                ),
                axis=1,
            )
            numerator_derivative = lorentzian_observed - gaussian_observed
            denominator_linear = 2.0 * (gaussian_lorentzian - gaussian_gaussian)
            denominator_quadratic = (
                gaussian_gaussian - 2.0 * gaussian_lorentzian + lorentzian_lorentzian
            )
            denominator_derivative_grid = (
                denominator_linear + 2.0 * intrinsic_grid[:, None] * denominator_quadratic
            )
            objective_derivative_grid = -np.sum(
                numerator_grid
                * (
                    2.0 * numerator_derivative * denominator_grid
                    - numerator_grid * denominator_derivative_grid
                )
                / (denominator_grid * denominator_grid),
                axis=1,
            )
            brackets: set[tuple[int, int]] = set()
            for index in np.flatnonzero(
                ((objective_derivative_grid[:-1] < 0.0) & (objective_derivative_grid[1:] >= 0.0))
                | ((objective_derivative_grid[:-1] <= 0.0) & (objective_derivative_grid[1:] > 0.0))
            ):
                brackets.add((int(index), int(index + 1)))
            for index in (
                np.flatnonzero(
                    (objective_grid[1:-1] < objective_grid[:-2])
                    & (objective_grid[1:-1] < objective_grid[2:])
                )
                + 1
            ):
                brackets.add((int(index - 1), int(index + 1)))
            for lower_index, upper_index in brackets:
                basin_centered_logit, basin_objective = polish_centered_logit(
                    float(centered_logit_grid[lower_index]),
                    float(centered_logit_grid[upper_index]),
                )
                if abs(
                    basin_centered_logit
                ) < _MIXTURE_Z_LIMIT - _MIXTURE_Z_LENGTH_TOLERANCE and globally_relevant_basin(
                    basin_objective,
                    float(centered_logit_grid[lower_index]),
                    float(centered_logit_grid[upper_index]),
                ):
                    candidate_centered_logit.add(basin_centered_logit)

    results: list[tuple[float, FloatArray, float, float]] = []
    results.extend(
        (
            (0.0, gaussian_scales / gaussian_reference, gaussian_objective, -math.inf),
            (1.0, lorentzian_scales / lorentzian_reference, lorentzian_objective, math.inf),
        )
    )
    for centered_logit in sorted(candidate_centered_logit):
        eta = physical_eta(centered_logit)
        normalized_scales, objective = evaluated_centered_logit(centered_logit)
        physical_normalization = (1.0 - eta) * gaussian_reference + eta * lorentzian_reference
        results.append(
            (
                eta,
                normalized_scales / physical_normalization,
                objective,
                centered_logit,
            )
        )
    minimum_objective = min(item[2] for item in results)
    tie_tolerance = (
        512.0 * np.finfo(np.float64).eps * max(1.0, float(profile_count), abs(minimum_objective))
    )
    indistinguishable = sorted(
        (item for item in results if item[2] <= minimum_objective + tie_tolerance),
        key=lambda item: item[3],
    )
    clusters: list[list[tuple[float, FloatArray, float, float]]] = []
    for item in indistinguishable:
        if (
            not clusters
            or not math.isfinite(item[3])
            or not math.isfinite(clusters[-1][-1][3])
            or item[3] - clusters[-1][-1][3] > _MIXTURE_Z_LENGTH_TOLERANCE
        ):
            clusters.append([item])
        else:
            clusters[-1].append(item)
    indistinguishable = [min(cluster, key=lambda item: (item[2], item[3])) for cluster in clusters]
    eta, scales, objective, _ = min(
        indistinguishable,
        key=lambda item: (item[0] not in {0.0, 1.0}, item[0]),
    )
    tied_eta = tuple(sorted(item[0] for item in indistinguishable))
    return eta, scales, objective, tied_eta


def _log_width_derivative(templates: FloatArray, widths: FloatArray, index: int) -> FloatArray:
    if index == 0:
        lower, upper = 0, 1
    elif index == widths.size - 1:
        lower, upper = widths.size - 2, widths.size - 1
    else:
        lower, upper = index - 1, index + 1
    return (templates[upper] - templates[lower]) / math.log(widths[upper] / widths[lower])


def _sensitivity_diagnostics(
    *,
    gaussian: FloatArray,
    lorentzian: FloatArray,
    gaussian_derivative: FloatArray,
    lorentzian_derivative: FloatArray,
    eta: float,
    scales: FloatArray,
    weight: FloatArray,
    valid_profile: NDArray[np.int64],
) -> tuple[tuple[str, ...], FloatArray, int, float]:
    profile_scale = scales[valid_profile]
    model = (1.0 - eta) * gaussian + eta * lorentzian
    if eta == 0.0:
        active_parameter_names = (
            "log_gaussian_sigma",
            "lorentzian_probability_from_zero",
        )
        jacobian = np.stack(
            (
                profile_scale * gaussian_derivative,
                profile_scale * (lorentzian - gaussian),
            ),
            axis=-1,
        )
    elif eta == 1.0:
        active_parameter_names = (
            "log_lorentzian_half_width",
            "gaussian_probability_from_one",
        )
        jacobian = np.stack(
            (
                profile_scale * lorentzian_derivative,
                profile_scale * (gaussian - lorentzian),
            ),
            axis=-1,
        )
    else:
        active_parameter_names = (
            "log_gaussian_sigma",
            "log_lorentzian_half_width",
            "logit_lorentzian_probability",
        )
        jacobian = np.stack(
            (
                profile_scale * (1.0 - eta) * gaussian_derivative,
                profile_scale * eta * lorentzian_derivative,
                profile_scale * eta * (1.0 - eta) * (lorentzian - gaussian),
            ),
            axis=-1,
        )
    weighted_nuisance = weight * model
    denominator = np.bincount(
        valid_profile,
        weights=weighted_nuisance * model,
        minlength=scales.size,
    )
    for column in range(len(active_parameter_names)):
        numerator = np.bincount(
            valid_profile,
            weights=weighted_nuisance * jacobian[:, column],
            minlength=scales.size,
        )
        jacobian[:, column] -= (numerator / denominator)[valid_profile] * model
    whitened = np.sqrt(weight)[:, None] * jacobian
    singular = np.linalg.svd(whitened, compute_uv=False)
    if singular.size < len(active_parameter_names):
        singular = np.pad(singular, (0, len(active_parameter_names) - singular.size))
    tolerance = (
        64.0 * np.finfo(np.float64).eps * max(whitened.shape, default=1) * float(singular[0])
    )
    rank = int(np.count_nonzero(singular > tolerance)) if singular[0] > 0.0 else 0
    condition = (
        float(singular[0] / singular[-1])
        if rank == len(active_parameter_names) and singular[-1] > 0.0
        else math.inf
    )
    singular.setflags(write=False)
    return active_parameter_names, singular, rank, condition


def fit_mosaic_component_profiles(
    bank: MosaicComponentProfileBank,
    *,
    maximum_sensitivity_condition: float = _MAXIMUM_SENSITIVITY_CONDITION,
    _pair_result_cache: dict[tuple[float, float], tuple[float, float, tuple[float, ...]]]
    | None = None,
) -> MosaicProfileFitResult:
    """Fit exact component-bank widths and eta after profiling peak amplitudes.

    Widths are selected only from the supplied exact response bank. Eta is optimized on
    ``[0, 1]`` for every width pair, including both boundary faces. The residual uses one
    nonnegative scale per profile. Absolute and relative peak heights therefore do not enter the
    mosaic objective; only each profile's finite-bin shape is shared across the joint fit.
    """

    if not isinstance(bank, MosaicComponentProfileBank):
        raise TypeError("bank must be a MosaicComponentProfileBank")
    maximum_condition = float(maximum_sensitivity_condition)
    if not math.isfinite(maximum_condition) or maximum_condition <= 1.0:
        raise ValueError("maximum_sensitivity_condition must be finite and greater than one")
    order = _canonical_profile_order(bank.observations.identities)
    inverse_order = np.argsort(order)
    canonical_bank = bank.reorder_profiles(order)
    observations = canonical_bank.observations
    profile_count = len(observations.identities)
    valid_profile, observed, weight, observed_scale = _valid_fit_vectors(observations)
    gaussian_intensity = np.asarray(
        [component.profile.intensity for component in canonical_bank.gaussian_profiles]
    )
    lorentzian_intensity = np.asarray(
        [component.profile.intensity for component in canonical_bank.lorentzian_profiles]
    )
    gaussian_valid = gaussian_intensity[:, observations.valid]
    lorentzian_valid = lorentzian_intensity[:, observations.valid]
    point_reference = np.maximum(
        np.max(gaussian_valid, axis=0),
        np.max(lorentzian_valid, axis=0),
    )
    component_reference = np.zeros(profile_count, dtype=np.float64)
    np.maximum.at(component_reference, valid_profile, point_reference)
    if np.any(component_reference <= 0.0):
        raise ValueError("every profile requires positive component intensity")
    gaussian_valid = gaussian_valid / component_reference[valid_profile]
    lorentzian_valid = lorentzian_valid / component_reference[valid_profile]
    objective_surface = np.empty(
        (gaussian_valid.shape[0], lorentzian_valid.shape[0]),
        dtype=np.float64,
    )
    eta_surface = np.empty(objective_surface.shape, dtype=np.float64)
    tied_eta_surface: list[list[tuple[float, ...]]] = [
        [() for _ in range(lorentzian_valid.shape[0])] for _ in range(gaussian_valid.shape[0])
    ]
    best: tuple[float, int, int, float] | None = None
    for gaussian_index, gaussian in enumerate(gaussian_valid):
        for lorentzian_index, lorentzian in enumerate(lorentzian_valid):
            pair_key = (
                float(canonical_bank.gaussian_sigma_rad[gaussian_index]),
                float(canonical_bank.lorentzian_half_width_rad[lorentzian_index]),
            )
            cached_pair = None if _pair_result_cache is None else _pair_result_cache.get(pair_key)
            if cached_pair is None:
                eta, _, objective, tied_eta = _best_eta(
                    gaussian,
                    lorentzian,
                    observed,
                    weight,
                    valid_profile,
                    profile_count,
                    audit_all_minima=True,
                )
                if _pair_result_cache is not None:
                    _pair_result_cache[pair_key] = (eta, objective, tied_eta)
            else:
                eta, objective, tied_eta = cached_pair
            objective_surface[gaussian_index, lorentzian_index] = objective
            eta_surface[gaussian_index, lorentzian_index] = eta
            tied_eta_surface[gaussian_index][lorentzian_index] = tied_eta
            candidate = (objective, gaussian_index, lorentzian_index, eta)
            if best is None or candidate < best:
                best = candidate
    if best is None or not math.isfinite(best[0]):
        raise ValueError("no finite mosaic-profile solution exists on the supplied bank")
    objective, gaussian_index, lorentzian_index, eta = best
    best_gaussian = gaussian_valid[gaussian_index]
    best_lorentzian = lorentzian_valid[lorentzian_index]
    best_model = (1.0 - eta) * best_gaussian + eta * best_lorentzian
    normalized_scales, objective = _profiled_scales_and_objective(
        best_model,
        observed,
        weight,
        valid_profile,
        profile_count,
    )
    prediction_scales = normalized_scales * observed_scale
    scales = prediction_scales / component_reference
    canonical_predicted = np.zeros(observations.signal.shape, dtype=np.float64)
    canonical_predicted[observations.valid] = prediction_scales[valid_profile] * best_model
    predicted = canonical_predicted[inverse_order]
    normalized_residual = normalized_scales[valid_profile] * best_model - observed
    canonical_profile_residual = np.sqrt(
        np.bincount(
            valid_profile,
            weights=weight * normalized_residual * normalized_residual,
            minlength=profile_count,
        )
    )
    gaussian_derivative = _log_width_derivative(
        gaussian_valid,
        canonical_bank.gaussian_sigma_rad,
        gaussian_index,
    )
    lorentzian_derivative = _log_width_derivative(
        lorentzian_valid,
        canonical_bank.lorentzian_half_width_rad,
        lorentzian_index,
    )
    gaussian_probe_index = gaussian_index
    lorentzian_probe_index = lorentzian_index
    if eta == 0.0:
        boundary_diagnostics = [
            (
                index,
                *_sensitivity_diagnostics(
                    gaussian=best_gaussian,
                    lorentzian=component,
                    gaussian_derivative=gaussian_derivative,
                    lorentzian_derivative=lorentzian_derivative,
                    eta=eta,
                    scales=normalized_scales,
                    weight=weight,
                    valid_profile=valid_profile,
                ),
            )
            for index, component in enumerate(lorentzian_valid)
        ]
        (
            lorentzian_probe_index,
            active_parameter_names,
            singular,
            rank,
            condition,
        ) = min(
            boundary_diagnostics,
            key=lambda item: (item[3] < len(item[1]), item[4]),
        )
    elif eta == 1.0:
        boundary_diagnostics = [
            (
                index,
                *_sensitivity_diagnostics(
                    gaussian=component,
                    lorentzian=best_lorentzian,
                    gaussian_derivative=gaussian_derivative,
                    lorentzian_derivative=lorentzian_derivative,
                    eta=eta,
                    scales=normalized_scales,
                    weight=weight,
                    valid_profile=valid_profile,
                ),
            )
            for index, component in enumerate(gaussian_valid)
        ]
        (
            gaussian_probe_index,
            active_parameter_names,
            singular,
            rank,
            condition,
        ) = min(
            boundary_diagnostics,
            key=lambda item: (item[3] < len(item[1]), item[4]),
        )
    else:
        active_parameter_names, singular, rank, condition = _sensitivity_diagnostics(
            gaussian=best_gaussian,
            lorentzian=best_lorentzian,
            gaussian_derivative=gaussian_derivative,
            lorentzian_derivative=lorentzian_derivative,
            eta=eta,
            scales=normalized_scales,
            weight=weight,
            valid_profile=valid_profile,
        )
    tie_tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(profile_count),
            abs(objective),
        )
    )
    solution_keys: set[tuple[str, int | None, int | None, float]] = set()
    for tied_gaussian_index, tied_lorentzian_index in np.argwhere(
        objective_surface <= objective + tie_tolerance
    ):
        for tied_eta in tied_eta_surface[tied_gaussian_index][tied_lorentzian_index]:
            if tied_eta == 0.0:
                solution_key = ("G", int(tied_gaussian_index), None, 0.0)
            elif tied_eta == 1.0:
                solution_key = ("L", None, int(tied_lorentzian_index), 1.0)
            else:
                solution_key = (
                    "GL",
                    int(tied_gaussian_index),
                    int(tied_lorentzian_index),
                    tied_eta,
                )
            solution_keys.add(solution_key)
    if len(solution_keys) > 1:
        raise MosaicIdentifiabilityError(
            rank,
            singular,
            condition,
            active_parameter_names,
            reason="global_alias",
            competing_solution_keys=tuple(sorted(solution_keys, key=repr)),
        )
    if rank < len(active_parameter_names) or condition > maximum_condition:
        raise MosaicIdentifiabilityError(
            rank,
            singular,
            condition,
            active_parameter_names,
        )
    gaussian_width = (
        None if eta == 1.0 else float(canonical_bank.gaussian_sigma_rad[gaussian_index])
    )
    lorentzian_width = (
        None if eta == 0.0 else float(canonical_bank.lorentzian_half_width_rad[lorentzian_index])
    )
    return MosaicProfileFitResult(
        gaussian_sigma_rad=gaussian_width,
        lorentzian_half_width_rad=lorentzian_width,
        lorentzian_probability=eta,
        active_parameter_names=active_parameter_names,
        profile_identities=bank.observations.identities,
        profile_scales=scales[inverse_order],
        profile_relative_l2_residual=canonical_profile_residual[inverse_order],
        objective=objective,
        predicted_intensity=predicted,
        sensitivity_singular_values=singular,
        sensitivity_rank=rank,
        sensitivity_condition=condition,
        width_pair_objective=objective_surface,
        width_pair_eta=eta_surface,
        gaussian_bank_index=None if eta == 1.0 else gaussian_index,
        lorentzian_bank_index=None if eta == 0.0 else lorentzian_index,
        gaussian_activation_probe_width_rad=(
            float(canonical_bank.gaussian_sigma_rad[gaussian_probe_index]) if eta == 1.0 else None
        ),
        lorentzian_activation_probe_width_rad=(
            float(canonical_bank.lorentzian_half_width_rad[lorentzian_probe_index])
            if eta == 0.0
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class MosaicProfileRefinementStep:
    """One deterministic exact-bank refinement level."""

    level: int
    gaussian_width_count: int
    lorentzian_width_count: int
    retained_cell_count: int
    best_objective: float
    best_gaussian_sigma_rad: float | None
    best_lorentzian_half_width_rad: float | None
    best_lorentzian_probability: float


@dataclass(frozen=True, slots=True)
class MosaicProfileSearchResult:
    """Final fit, exact response bank, and global-grid refinement evidence."""

    fit: MosaicProfileFitResult
    bank: MosaicComponentProfileBank
    refinement_steps: tuple[MosaicProfileRefinementStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fit, MosaicProfileFitResult):
            raise TypeError("fit must be a MosaicProfileFitResult")
        if not isinstance(self.bank, MosaicComponentProfileBank):
            raise TypeError("bank must be a MosaicComponentProfileBank")
        steps = tuple(self.refinement_steps)
        if not steps or any(not isinstance(step, MosaicProfileRefinementStep) for step in steps):
            raise ValueError("refinement_steps must contain typed refinement evidence")
        object.__setattr__(self, "refinement_steps", steps)


def _refinement_candidates(widths: FloatArray, indices: set[int], count: int) -> FloatArray:
    candidates = [widths]
    for index in sorted(indices):
        lower = max(index - 1, 0)
        upper = min(index + 1, widths.size - 1)
        if lower != upper:
            candidates.append(np.geomspace(widths[lower], widths[upper], count))
    return np.unique(np.concatenate(candidates))


def _retained_width_cells(
    objective: FloatArray,
    objective_delta: float,
) -> tuple[tuple[int, int], ...]:
    best = float(np.min(objective))
    retained: list[tuple[float, int, int]] = []
    for gaussian_index in range(objective.shape[0]):
        for lorentzian_index in range(objective.shape[1]):
            value = float(objective[gaussian_index, lorentzian_index])
            if value > best + objective_delta:
                continue
            neighborhood = objective[
                max(0, gaussian_index - 1) : gaussian_index + 2,
                max(0, lorentzian_index - 1) : lorentzian_index + 2,
            ]
            if value <= float(np.min(neighborhood)):
                retained.append((value, gaussian_index, lorentzian_index))
    if not retained:
        gaussian_index, lorentzian_index = np.unravel_index(
            int(np.argmin(objective)),
            objective.shape,
        )
        retained.append((best, int(gaussian_index), int(lorentzian_index)))
    return tuple((item[1], item[2]) for item in sorted(retained))


def fit_refined_mosaic_component_profiles(
    observations: MosaicProfileSet,
    *,
    evaluate_gaussian_profile: Callable[[float], MosaicProfileSet],
    evaluate_lorentzian_profile: Callable[[float], MosaicProfileSet],
    gaussian_sigma_bounds_rad: ArrayLike,
    lorentzian_half_width_bounds_rad: ArrayLike,
    coarse_width_count: int,
    refinement_width_count: int,
    refinement_levels: int,
    near_optimal_objective_delta: float,
    maximum_sensitivity_condition: float = _MAXIMUM_SENSITIVITY_CONDITION,
) -> MosaicProfileSearchResult:
    """Globally scan eta and refine every near-optimal exact width-grid basin."""

    if not isinstance(observations, MosaicProfileSet):
        raise TypeError("observations must be a MosaicProfileSet")
    if not callable(evaluate_gaussian_profile) or not callable(evaluate_lorentzian_profile):
        raise TypeError("component profile evaluators must be callable")
    gaussian_bounds = _readonly_float(
        gaussian_sigma_bounds_rad,
        (2,),
        "gaussian_sigma_bounds_rad",
    )
    lorentzian_bounds = _readonly_float(
        lorentzian_half_width_bounds_rad,
        (2,),
        "lorentzian_half_width_bounds_rad",
    )
    if (
        gaussian_bounds[0] <= 0.0
        or gaussian_bounds[1] <= gaussian_bounds[0]
        or lorentzian_bounds[0] <= 0.0
        or lorentzian_bounds[1] <= lorentzian_bounds[0]
    ):
        raise ValueError("component width bounds must be strictly increasing and positive")
    for name, value, minimum in (
        ("coarse_width_count", coarse_width_count, 2),
        ("refinement_width_count", refinement_width_count, 3),
        ("refinement_levels", refinement_levels, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    objective_delta = float(near_optimal_objective_delta)
    if not math.isfinite(objective_delta) or objective_delta < 0.0:
        raise ValueError("near_optimal_objective_delta must be finite and nonnegative")

    gaussian_cache: dict[float, MosaicProfileSet] = {}
    lorentzian_cache: dict[float, MosaicProfileSet] = {}
    pair_result_cache: dict[tuple[float, float], tuple[float, float, tuple[float, ...]]] = {}
    gaussian_widths = np.geomspace(*gaussian_bounds, int(coarse_width_count))
    lorentzian_widths = np.geomspace(*lorentzian_bounds, int(coarse_width_count))
    steps: list[MosaicProfileRefinementStep] = []
    bank: MosaicComponentProfileBank | None = None
    result: MosaicProfileFitResult | None = None
    for level in range(int(refinement_levels) + 1):
        for width in gaussian_widths:
            key = float(width)
            if key not in gaussian_cache:
                gaussian_cache[key] = evaluate_gaussian_profile(key)
        for width in lorentzian_widths:
            key = float(width)
            if key not in lorentzian_cache:
                lorentzian_cache[key] = evaluate_lorentzian_profile(key)
        gaussian_widths = np.asarray(sorted(gaussian_cache), dtype=np.float64)
        lorentzian_widths = np.asarray(sorted(lorentzian_cache), dtype=np.float64)
        bank = MosaicComponentProfileBank(
            observations=observations,
            gaussian_sigma_rad=gaussian_widths,
            gaussian_profiles=tuple(
                MosaicComponentProfile("gaussian", float(width), gaussian_cache[float(width)])
                for width in gaussian_widths
            ),
            lorentzian_half_width_rad=lorentzian_widths,
            lorentzian_profiles=tuple(
                MosaicComponentProfile(
                    "lorentzian",
                    float(width),
                    lorentzian_cache[float(width)],
                )
                for width in lorentzian_widths
            ),
        )
        try:
            result = fit_mosaic_component_profiles(
                bank,
                maximum_sensitivity_condition=maximum_sensitivity_condition,
                _pair_result_cache=pair_result_cache,
            )
        except MosaicIdentifiabilityError:
            if level == int(refinement_levels):
                raise
            gaussian_widths = _refinement_candidates(
                gaussian_widths,
                set(range(gaussian_widths.size)),
                3,
            )
            lorentzian_widths = _refinement_candidates(
                lorentzian_widths,
                set(range(lorentzian_widths.size)),
                3,
            )
            continue
        retained = _retained_width_cells(
            result.width_pair_objective,
            objective_delta,
        )
        steps.append(
            MosaicProfileRefinementStep(
                level=level,
                gaussian_width_count=gaussian_widths.size,
                lorentzian_width_count=lorentzian_widths.size,
                retained_cell_count=len(retained),
                best_objective=result.objective,
                best_gaussian_sigma_rad=result.gaussian_sigma_rad,
                best_lorentzian_half_width_rad=result.lorentzian_half_width_rad,
                best_lorentzian_probability=result.lorentzian_probability,
            )
        )
        if level == int(refinement_levels):
            break
        active_gaussian_indices = {
            gaussian_index
            for gaussian_index, lorentzian_index in retained
            if result.width_pair_eta[gaussian_index, lorentzian_index] != 1.0
        }
        active_lorentzian_indices = {
            lorentzian_index
            for gaussian_index, lorentzian_index in retained
            if result.width_pair_eta[gaussian_index, lorentzian_index] != 0.0
        }
        gaussian_candidates = gaussian_widths
        lorentzian_candidates = lorentzian_widths
        if active_gaussian_indices:
            gaussian_candidates = _refinement_candidates(
                gaussian_widths,
                active_gaussian_indices,
                int(refinement_width_count),
            )
        if any(
            result.width_pair_eta[gaussian_index, lorentzian_index] == 1.0
            for gaussian_index, lorentzian_index in retained
        ):
            gaussian_candidates = np.unique(
                np.concatenate(
                    (
                        gaussian_candidates,
                        _refinement_candidates(
                            gaussian_widths,
                            set(range(gaussian_widths.size)),
                            3,
                        ),
                    )
                )
            )
        if active_lorentzian_indices:
            lorentzian_candidates = _refinement_candidates(
                lorentzian_widths,
                active_lorentzian_indices,
                int(refinement_width_count),
            )
        if any(
            result.width_pair_eta[gaussian_index, lorentzian_index] == 0.0
            for gaussian_index, lorentzian_index in retained
        ):
            lorentzian_candidates = np.unique(
                np.concatenate(
                    (
                        lorentzian_candidates,
                        _refinement_candidates(
                            lorentzian_widths,
                            set(range(lorentzian_widths.size)),
                            3,
                        ),
                    )
                )
            )
        gaussian_widths = gaussian_candidates
        lorentzian_widths = lorentzian_candidates
    if result is None or bank is None:
        raise RuntimeError("mosaic profile refinement produced no result")
    return MosaicProfileSearchResult(result, bank, tuple(steps))


__all__ = [
    "MosaicComponentProfile",
    "MosaicComponentProfileBank",
    "MosaicIdentifiabilityError",
    "MosaicProfileDefinition",
    "MosaicProfileFitResult",
    "MosaicProfileIdentity",
    "MosaicProfileRefinementStep",
    "MosaicProfileSearchResult",
    "MosaicProfileSet",
    "MosaicReflectionGroupKey",
    "evaluate_continuous_mosaic_profiles",
    "fit_mosaic_component_profiles",
    "fit_refined_mosaic_component_profiles",
]
