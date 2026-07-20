"""Orchestration and explicit mass accounting for mosaic-painted Ewald spheres."""

from __future__ import annotations

from math import fsum, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.ewald import (
    _RootBatch,
    _solve_batched_infinite_rod_ewald,
    _solve_validated_infinite_rod_ewald,
    _stable_line_components,
)
from painted_ewald.mosaic import MosaicSpace, build_mosaic_space
from painted_ewald.raster import (
    _rasterize_arrays_equal_solid_angle,
    _sphere_texture_from_mass,
    rasterize_equal_solid_angle,
)
from painted_ewald.rotations import mosaic_axes
from painted_ewald.types import (
    BranchCoatingSummary,
    FamilyCoatingSummary,
    MassLedger,
    MosaicSlice,
    PaintedEwaldCoating,
    PaintedEwaldSphere,
    PaintedPoint,
    PainterConfig,
    PaintMeasure,
    Rod,
    RodCoatingSummary,
    RootStatus,
    StrengthModel,
    _hexagonal_family_indices,
)
from painted_ewald.validation import finite_scalar, readonly_float_array

FloatArray = NDArray[np.float64]


class _UnitStrength:
    __slots__ = ()

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        return 1.0


def _is_complete_hexagonal_family(
    *,
    reciprocal_basis_Ainv: FloatArray,
    family_m: int,
    rods: list[Rod],
) -> bool:
    mean_axis = reciprocal_basis_Ainv[:, 2]
    in_plane = np.column_stack(
        (
            _stable_line_components(reciprocal_basis_Ainv[:, 0], mean_axis)[1],
            _stable_line_components(reciprocal_basis_Ainv[:, 1], mean_axis)[1],
        )
    )
    metric = in_plane.T @ in_plane
    metric_scale = float(np.max(np.abs(metric)))
    if not np.isfinite(metric_scale) or metric_scale <= 0.0:
        return False
    normalized_metric = metric / metric_scale
    in_plane_scale = sqrt(metric_scale)
    projection_condition = (
        max(
            float(np.linalg.norm(reciprocal_basis_Ainv[:, 0])),
            float(np.linalg.norm(reciprocal_basis_Ainv[:, 1])),
        )
        / in_plane_scale
    )
    machine_epsilon = np.finfo(np.float64).eps
    projection_uncertainty = 32.0 * machine_epsilon * projection_condition
    if not np.isfinite(projection_uncertainty) or projection_uncertainty > sqrt(machine_epsilon):
        return False
    tolerance = max(2048.0 * machine_epsilon, projection_uncertainty)
    if not (
        np.isclose(normalized_metric[0, 0], normalized_metric[1, 1], rtol=0.0, atol=tolerance)
        and np.isclose(
            normalized_metric[0, 1],
            0.5 * normalized_metric[0, 0],
            rtol=0.0,
            atol=tolerance,
        )
    ):
        return False
    return {(rod.h, rod.k) for rod in rods} == _hexagonal_family_indices(family_m)


class EwaldSpherePainter:
    """Compile one mosaic quadrature and paint analytic rod intersections for each ``ki``."""

    __slots__ = (
        "_b3_norm_Ainv",
        "_config",
        "_d_hat_sample",
        "_mean_axis_crystal",
        "_mosaic_space",
        "_strength_model",
    )

    def __init__(
        self,
        config: PainterConfig,
        strength_model: StrengthModel | None = None,
    ) -> None:
        if not isinstance(config, PainterConfig):
            raise TypeError("config must be PainterConfig")
        selected_strength_model: StrengthModel = (
            _UnitStrength() if strength_model is None else strength_model
        )
        if not callable(getattr(selected_strength_model, "evaluate", None)):
            raise TypeError("strength_model must provide an evaluate method")
        mosaic_space = build_mosaic_space(
            reciprocal_basis_Ainv=config.reciprocal_basis_Ainv,
            crystal_to_sample=config.crystal_to_sample,
            parameters=config.mosaic,
        )
        mean_axis, _ = mosaic_axes(config.reciprocal_basis_Ainv)
        direction_crystal = np.einsum(
            "nij,j->ni", mosaic_space.rotation_crystal, mean_axis, optimize=True
        )
        direction_sample = direction_crystal @ config.crystal_to_sample.T
        direction_sample /= np.linalg.norm(direction_sample, axis=1)[:, None]
        direction_sample = np.ascontiguousarray(direction_sample, dtype=np.float64)
        direction_sample.setflags(write=False)
        self._config = config
        self._strength_model = selected_strength_model
        self._mosaic_space = mosaic_space
        self._b3_norm_Ainv = float(np.linalg.norm(config.reciprocal_basis_Ainv[:, 2]))
        self._d_hat_sample = direction_sample
        self._mean_axis_crystal = mean_axis

    @property
    def config(self) -> PainterConfig:
        return self._config

    @property
    def mosaic_space(self) -> MosaicSpace:
        return self._mosaic_space

    def mosaic_slice(self, *, rod: Rod, u_Ainv: float) -> MosaicSlice:
        """Return one complete fixed-u mosaic cap or ring before Ewald conditioning."""

        return self._mosaic_space.mosaic_slice(rod=rod, u_Ainv=u_Ainv)

    def _q0_sample_for_rod(self, rod: Rod) -> tuple[FloatArray, float] | None:
        if rod.family_m == 0:
            return None
        basis = self._config.reciprocal_basis_Ainv
        q_parallel_crystal = rod.h * basis[:, 0] + rod.k * basis[:, 1]
        q0_parallel, q0_perpendicular_crystal = _stable_line_components(
            q_parallel_crystal,
            self._mean_axis_crystal,
        )
        q0_crystal = np.einsum(
            "nij,j->ni",
            self._mosaic_space.rotation_crystal,
            q0_perpendicular_crystal,
            optimize=True,
        )
        return (
            np.ascontiguousarray(q0_crystal @ self._config.crystal_to_sample.T),
            q0_parallel,
        )

    def paint(self, ki_sample_Ainv: ArrayLike) -> PaintedEwaldSphere:
        """Paint all configured infinite rods onto one elastic q-space Ewald sphere."""

        incident = readonly_float_array(ki_sample_Ainv, (3,), "ki_sample_Ainv")
        incident_norm = sqrt(fsum(float(value) * float(value) for value in incident))
        if incident_norm == 0.0:
            raise ValueError("ki_sample_Ainv must be nonzero")

        points: list[PaintedPoint] = []
        collapsed_mass: list[float] = []
        tangent_mass: list[float] = []
        no_root_mass: list[float] = []
        forward_excluded_mass: list[float] = []
        suppressed_direct_count = 0
        zero = np.zeros(3, dtype=np.float64)
        space = self._mosaic_space

        for rod in self._config.rods:
            q0_components = self._q0_sample_for_rod(rod)
            q0_sample = None if q0_components is None else q0_components[0]
            rod_is_m0 = rod.family_m == 0
            for row in range(space.orientation_id.size):
                root_result = _solve_validated_infinite_rod_ewald(
                    incident=incident,
                    incident_norm=incident_norm,
                    q0=zero if q0_sample is None else q0_sample[row],
                    q0_parallel_Ainv=(None if q0_components is None else q0_components[1]),
                    direction=self._d_hat_sample[row],
                    b3_norm_Ainv=self._b3_norm_Ainv,
                    rod_is_m0=rod_is_m0,
                    root_tolerance_rel=self._config.root_tolerance_rel,
                    residual_tolerance_rel=self._config.residual_tolerance_rel,
                )
                orientation_mass = float(space.probability_mass[row])
                base_orientation_mass = rod.population * orientation_mass
                suppressed_direct_count += root_result.direct_root_count
                if root_result.status is RootStatus.NO_ROOT:
                    no_root_mass.append(base_orientation_mass)
                    continue
                if root_result.status is RootStatus.COLLAPSED_DIRECT:
                    collapsed_mass.append(base_orientation_mass)
                    continue
                if root_result.status is RootStatus.TANGENT:
                    tangent_mass.append(base_orientation_mass)
                    continue

                for root in root_result.emittable_roots:
                    strength = finite_scalar(
                        self._strength_model.evaluate(
                            rod=rod,
                            L=root.L,
                            k_norm_Ainv=incident_norm,
                        ),
                        "rod strength",
                    )
                    if strength < 0.0:
                        raise ValueError("rod strength must be nonnegative")
                    base_weight = base_orientation_mass * strength
                    jacobian: float | None = None
                    weight = base_weight
                    if self._config.measure is PaintMeasure.COAREA_INTENSITY:
                        q_norm = sqrt(
                            fsum(float(value) * float(value) for value in root.q_sample_Ainv)
                        )
                        q_min = self._config.forward_policy.q_min_Ainv
                        if rod_is_m0 and q_min is not None and q_norm < q_min:
                            forward_excluded_mass.append(base_weight)
                            continue
                        jacobian = root.coarea_jacobian
                        weight = base_weight * jacobian
                    points.append(
                        PaintedPoint(
                            rod_h=rod.h,
                            rod_k=rod.k,
                            family_m=rod.family_m,
                            branch=root.branch,
                            orientation_id=int(space.orientation_id[row]),
                            alpha_rad=float(space.alpha_rad[row]),
                            beta_rad=float(space.beta_rad[row]),
                            u_Ainv=root.u_Ainv,
                            L=root.L,
                            q_sample_Ainv=root.q_sample_Ainv,
                            kf_sample_Ainv=root.kf_sample_Ainv,
                            orientation_mass=orientation_mass,
                            rod_population=rod.population,
                            rod_strength=strength,
                            base_weight=base_weight,
                            coarea_jacobian=jacobian,
                            weight=weight,
                            ewald_residual_Ainv=root.ewald_residual_Ainv,
                        )
                    )

        point_tuple = tuple(points)
        painted_weight = fsum(point.weight for point in point_tuple)
        texture = rasterize_equal_solid_angle(
            points=point_tuple,
            radius_Ainv=incident_norm,
            parameters=self._config.raster,
        )
        ledger = MassLedger(
            painted_weight=painted_weight,
            collapsed_direct_base_mass=fsum(collapsed_mass),
            tangent_base_mass=fsum(tangent_mass),
            no_root_base_mass=fsum(no_root_mass),
            forward_excluded_base_mass=fsum(forward_excluded_mass),
            suppressed_algebraic_direct_root_count=suppressed_direct_count,
        )
        return PaintedEwaldSphere(
            ki_sample_Ainv=incident,
            center_q_sample_Ainv=-incident,
            radius_Ainv=incident_norm,
            points=point_tuple,
            texture=texture,
            ledger=ledger,
        )

    def _root_strengths(
        self,
        *,
        rod: Rod,
        root: _RootBatch,
        incident_norm: float,
    ) -> FloatArray:
        strength = np.fromiter(
            (
                finite_scalar(
                    self._strength_model.evaluate(
                        rod=rod,
                        L=float(l_coordinate),
                        k_norm_Ainv=incident_norm,
                    ),
                    "rod strength",
                )
                for l_coordinate in root.L
            ),
            dtype=np.float64,
            count=root.L.size,
        )
        if np.any(strength < 0.0):
            raise ValueError("rod strength must be nonnegative")
        return strength

    def _paint_rod_coating(
        self,
        *,
        rod: Rod,
        incident: FloatArray,
        incident_norm: float,
        texture_mass: FloatArray,
        texture_compensation: FloatArray,
    ) -> RodCoatingSummary:
        q0_components = self._q0_sample_for_rod(rod)
        q0_sample = None if q0_components is None else q0_components[0]
        q0_parallel = None if q0_components is None else q0_components[1]
        root_result = _solve_batched_infinite_rod_ewald(
            incident=incident,
            incident_norm=incident_norm,
            q0=q0_sample,
            q0_parallel_Ainv=q0_parallel,
            direction=self._d_hat_sample,
            b3_norm_Ainv=self._b3_norm_Ainv,
            rod_is_m0=rod.family_m == 0,
            root_tolerance_rel=self._config.root_tolerance_rel,
            residual_tolerance_rel=self._config.residual_tolerance_rel,
        )
        orientation_base_mass = rod.population * self._mosaic_space.probability_mass
        collapsed_mass = float(
            np.sum(
                orientation_base_mass,
                where=root_result.collapsed_direct,
                initial=0.0,
                dtype=np.float64,
            )
        )
        tangent_mass = float(
            np.sum(
                orientation_base_mass,
                where=root_result.tangent,
                initial=0.0,
                dtype=np.float64,
            )
        )
        no_root_mass = float(
            np.sum(
                orientation_base_mass,
                where=root_result.no_root,
                initial=0.0,
                dtype=np.float64,
            )
        )
        forward_excluded_mass = 0.0
        branch_summaries: list[BranchCoatingSummary] = []
        maximum_residual = 0.0

        for root in root_result.roots:
            selected = root.orientation_index
            base_weight = orientation_base_mass[selected]
            if not isinstance(self._strength_model, _UnitStrength):
                base_weight = base_weight * self._root_strengths(
                    rod=rod,
                    root=root,
                    incident_norm=incident_norm,
                )
            retained_base_weight = base_weight
            retained_kf = root.kf_sample_Ainv
            retained_residual = root.ewald_residual_Ainv
            retained_jacobian = root.coarea_jacobian
            q_min = self._config.forward_policy.q_min_Ainv
            if (
                self._config.measure is PaintMeasure.COAREA_INTENSITY
                and rod.family_m == 0
                and q_min is not None
            ):
                keep = np.linalg.norm(root.q_sample_Ainv, axis=1) >= q_min
                forward_excluded_mass += float(
                    np.sum(
                        base_weight,
                        where=~keep,
                        initial=0.0,
                        dtype=np.float64,
                    )
                )
                retained_base_weight = base_weight[keep]
                retained_kf = retained_kf[keep]
                retained_residual = retained_residual[keep]
                retained_jacobian = retained_jacobian[keep]
            retained_weight = retained_base_weight
            if self._config.measure is PaintMeasure.COAREA_INTENSITY:
                retained_weight = retained_base_weight * retained_jacobian
            if retained_residual.size:
                maximum_residual = max(maximum_residual, float(np.max(retained_residual)))
            increment = _rasterize_arrays_equal_solid_angle(
                kf_sample_Ainv=retained_kf,
                weights=retained_weight,
                radius_Ainv=incident_norm,
                parameters=self._config.raster,
            )
            adjusted_increment = increment - texture_compensation
            updated_mass = texture_mass + adjusted_increment
            texture_compensation[:] = updated_mass - texture_mass - adjusted_increment
            texture_mass[:] = updated_mass
            branch_summaries.append(
                BranchCoatingSummary(
                    branch=root.branch,
                    retained_root_count=retained_weight.size,
                    painted_weight=float(np.sum(retained_weight, dtype=np.float64)),
                )
            )

        ledger = MassLedger(
            painted_weight=fsum(branch.painted_weight for branch in branch_summaries),
            collapsed_direct_base_mass=collapsed_mass,
            tangent_base_mass=tangent_mass,
            no_root_base_mass=no_root_mass,
            forward_excluded_base_mass=forward_excluded_mass,
            suppressed_algebraic_direct_root_count=(
                root_result.suppressed_algebraic_direct_root_count
            ),
        )
        return RodCoatingSummary(
            rod=rod,
            branches=tuple(branch_summaries),
            ledger=ledger,
            maximum_ewald_residual_Ainv=maximum_residual,
        )

    def paint_coating(self, ki_sample_Ainv: ArrayLike) -> PaintedEwaldCoating:
        """Stream exact rod strengths into configured ``m`` subtotals.

        Every physical rod retains its own Ewald roots and exact-``L`` strength.
        Each subtotal is their incoherent selected-measure sum. Its completeness
        flag is true only for a full shell under a validated hexagonal in-plane
        metric; no representative rod replaces member-specific geometry.
        """

        incident = readonly_float_array(ki_sample_Ainv, (3,), "ki_sample_Ainv")
        incident_norm = sqrt(fsum(float(value) * float(value) for value in incident))
        if incident_norm == 0.0:
            raise ValueError("ki_sample_Ainv must be nonzero")
        texture_mass = np.zeros(
            (
                self._config.raster.mu_bin_count,
                self._config.raster.phi_bin_count,
            ),
            dtype=np.float64,
        )
        texture_compensation = np.zeros_like(texture_mass)
        rods_by_family: dict[int, list[Rod]] = {}
        for rod in self._config.rods:
            rods_by_family.setdefault(rod.family_m, []).append(rod)

        family_summaries: list[FamilyCoatingSummary] = []
        for family_m in sorted(rods_by_family):
            rod_summaries = tuple(
                self._paint_rod_coating(
                    rod=rod,
                    incident=incident,
                    incident_norm=incident_norm,
                    texture_mass=texture_mass,
                    texture_compensation=texture_compensation,
                )
                for rod in rods_by_family[family_m]
            )
            family_ledger = MassLedger.combine(tuple(summary.ledger for summary in rod_summaries))
            branch_ids = (0,) if family_m == 0 else (1, 2)
            branch_summaries = tuple(
                BranchCoatingSummary(
                    branch=branch,
                    retained_root_count=sum(
                        rod_summary.branches[index].retained_root_count
                        for rod_summary in rod_summaries
                    ),
                    painted_weight=fsum(
                        rod_summary.branches[index].painted_weight for rod_summary in rod_summaries
                    ),
                )
                for index, branch in enumerate(branch_ids)
            )
            family_summaries.append(
                FamilyCoatingSummary(
                    family_m=family_m,
                    rod_summaries=rod_summaries,
                    branches=branch_summaries,
                    ledger=family_ledger,
                    is_complete_hexagonal_family=_is_complete_hexagonal_family(
                        reciprocal_basis_Ainv=self._config.reciprocal_basis_Ainv,
                        family_m=family_m,
                        rods=rods_by_family[family_m],
                    ),
                    maximum_ewald_residual_Ainv=max(
                        summary.maximum_ewald_residual_Ainv for summary in rod_summaries
                    ),
                )
            )

        family_tuple = tuple(family_summaries)
        ledger = MassLedger.combine(tuple(family.ledger for family in family_tuple))
        texture = _sphere_texture_from_mass(
            mass=texture_mass,
            parameters=self._config.raster,
        )
        return PaintedEwaldCoating(
            ki_sample_Ainv=incident,
            center_q_sample_Ainv=-incident,
            radius_Ainv=incident_norm,
            measure=self._config.measure,
            forward_policy=self._config.forward_policy,
            texture=texture,
            ledger=ledger,
            family_summaries=family_tuple,
        )


__all__ = ["EwaldSpherePainter"]
