"""Continuous layered-reciprocal charts on one physical detector side."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from painted_ewald import EwaldLatentGeometry
from painted_ewald.rotations import mosaic_axes
from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.measurement.continuous_regions import ContinuousDetectorChartAreaMeasure
from rasim_next.measurement.reciprocal_profiles import LayeredReciprocalFrame
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure


@dataclass(frozen=True, slots=True)
class LayeredReciprocalDetectorAreaChart:
    """Map ``(Qr, L)`` to one detector-side preimage and its area measure.

    The inverse is analytic for a layered reciprocal basis whose axial vector
    is perpendicular to its basal plane. A nonorthogonal axial basis fails
    closed because it may have more than two elastic preimages.
    """

    detector_measure: DetectorEwaldMeasure
    reciprocal_frame: LayeredReciprocalFrame
    detector_column_interval_px: tuple[float, float]
    air_exit_guard_rad: float = 0.0
    revision: str = field(init=False)
    _radial_axis_1_sample: np.ndarray = field(init=False, repr=False)
    _radial_axis_2_sample: np.ndarray = field(init=False, repr=False)
    _axial_axis_sample: np.ndarray = field(init=False, repr=False)
    _axial_basis_norm_Ainv: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.detector_measure, DetectorEwaldMeasure):
            raise TypeError("detector_measure must be DetectorEwaldMeasure")
        if not isinstance(self.reciprocal_frame, LayeredReciprocalFrame):
            raise TypeError("reciprocal_frame must be LayeredReciprocalFrame")
        if self.reciprocal_frame.axial_basis_index != 2:
            raise ValueError("the current analytic chart requires axial_basis_index=2")
        config = self.detector_measure.coating.bragg_space.config
        basis = np.asarray(config.reciprocal_basis_Ainv, dtype=np.float64)
        rotation = np.asarray(config.crystal_to_sample, dtype=np.float64)
        if not np.array_equal(basis, self.reciprocal_frame.reciprocal_basis_Ainv):
            raise ValueError("reciprocal chart and detector measure bases differ")
        if not np.array_equal(rotation, self.reciprocal_frame.sample_from_crystal_rotation):
            raise ValueError("reciprocal chart and detector measure rotations differ")
        axial_axis_crystal, radial_axis_2_crystal = mosaic_axes(basis)
        radial_axis_1_crystal = np.cross(radial_axis_2_crystal, axial_axis_crystal)
        basis_scale = max(float(np.linalg.norm(basis)), 1.0)
        orthogonality_tolerance = 512.0 * np.finfo(np.float64).eps * basis_scale
        if np.any(np.abs(basis[:, :2].T @ axial_axis_crystal) > orthogonality_tolerance):
            raise ValueError(
                "analytic layered reciprocal chart requires an axial vector perpendicular "
                "to the basal reciprocal plane"
            )
        rows, columns = self.detector_measure.instrument.detector_shape_rc
        interval = tuple(float(value) for value in self.detector_column_interval_px)
        if (
            len(interval) != 2
            or not np.all(np.isfinite(interval))
            or interval[0] < -0.5
            or interval[1] > columns - 0.5
            or interval[0] >= interval[1]
        ):
            raise ValueError("detector column interval must lie on the active panel")
        guard = float(self.air_exit_guard_rad)
        if not isfinite(guard) or not -0.5 * np.pi < guard < 0.5 * np.pi:
            raise ValueError("air_exit_guard_rad must be a finite physical elevation")
        radial_axis_1_sample = rotation @ radial_axis_1_crystal
        radial_axis_2_sample = rotation @ radial_axis_2_crystal
        axial_axis_sample = rotation @ axial_axis_crystal
        for value in (radial_axis_1_sample, radial_axis_2_sample, axial_axis_sample):
            value.setflags(write=False)
        axial_norm = float(np.linalg.norm(basis[:, 2]))
        object.__setattr__(self, "detector_column_interval_px", interval)
        object.__setattr__(self, "air_exit_guard_rad", guard)
        object.__setattr__(self, "_radial_axis_1_sample", radial_axis_1_sample)
        object.__setattr__(self, "_radial_axis_2_sample", radial_axis_2_sample)
        object.__setattr__(self, "_axial_axis_sample", axial_axis_sample)
        object.__setattr__(self, "_axial_basis_norm_Ainv", axial_norm)
        states = self.detector_measure.incident.states
        object.__setattr__(
            self,
            "revision",
            canonical_revision_sha256(
                ("definition_id", "orthogonal_layered_reciprocal_detector_area_chart.v1"),
                ("reciprocal_basis_Ainv", basis),
                ("crystal_to_sample", rotation),
                ("ki_sample_Ainv", self.detector_measure.coating.ki_sample_Ainv),
                ("source_revision", states.source_revision),
                ("material_revision", states.material_revision),
                (
                    "sample_geometry_revision",
                    self.detector_measure.instrument.sample_geometry_revision,
                ),
                ("detector_shape_rc", (rows, columns)),
                ("detector_column_interval_px", interval),
                ("air_exit_guard_rad", guard),
                ("jacobian", "q_chart_surface_area_over_q_detector_surface_area.v1"),
            ),
        )

    def map_detector_area(
        self,
        first_coordinate: ArrayLike,
        second_coordinate: ArrayLike,
    ) -> ContinuousDetectorChartAreaMeasure:
        """Map nonnegative radial ``Qr`` and fractional ``L`` to detector area."""

        supplied_qr = np.asarray(first_coordinate)
        supplied_l = np.asarray(second_coordinate)
        if (np.iscomplexobj(supplied_qr) and np.any(supplied_qr.imag != 0.0)) or (
            np.iscomplexobj(supplied_l) and np.any(supplied_l.imag != 0.0)
        ):
            raise ValueError("reciprocal chart coordinates must be real")
        qr, ell = np.broadcast_arrays(
            np.asarray(supplied_qr.real, dtype=np.float64),
            np.asarray(supplied_l.real, dtype=np.float64),
        )
        if not np.all(np.isfinite(qr)) or not np.all(np.isfinite(ell)):
            raise ValueError("reciprocal chart coordinates must be finite")
        shape = qr.shape
        flat_qr = qr.reshape(-1)
        flat_l = ell.reshape(-1)
        size = flat_qr.size
        ki = np.asarray(self.detector_measure.coating.ki_sample_Ainv, dtype=np.float64)
        e1 = self._radial_axis_1_sample
        e2 = self._radial_axis_2_sample
        normal = self._axial_axis_sample
        axial_q = self._axial_basis_norm_Ainv * flat_l
        coefficient_cos = 2.0 * flat_qr * float(ki @ e1)
        coefficient_sin = 2.0 * flat_qr * float(ki @ e2)
        constant = flat_qr * flat_qr + axial_q * axial_q + 2.0 * axial_q * float(ki @ normal)
        radius = np.hypot(coefficient_cos, coefficient_sin)
        scale = np.maximum.reduce(
            (
                np.abs(coefficient_cos),
                np.abs(coefficient_sin),
                np.abs(constant),
                np.ones(size, dtype=np.float64),
            )
        )
        tolerance = 512.0 * np.finfo(np.float64).eps * scale
        root_exists = (
            (flat_qr > 0.0) & (radius > tolerance) & (np.abs(constant) < radius - tolerance)
        )
        cosine = np.zeros(size, dtype=np.float64)
        cosine[root_exists] = np.clip(
            -constant[root_exists] / radius[root_exists],
            -1.0,
            1.0,
        )
        offset = np.arctan2(coefficient_sin, coefficient_cos)
        opening = np.arccos(cosine)
        gamma = np.stack((offset + opening, offset - opening), axis=0)
        radial_direction = (
            np.cos(gamma)[..., None] * e1[None, None, :]
            + np.sin(gamma)[..., None] * e2[None, None, :]
        )
        tangent_direction = (
            -np.sin(gamma)[..., None] * e1[None, None, :]
            + np.cos(gamma)[..., None] * e2[None, None, :]
        )
        q_sample = (
            flat_qr[None, :, None] * radial_direction
            + axial_q[None, :, None] * normal[None, None, :]
        )
        kf_sample = ki[None, None, :] + q_sample
        elastic_scale = max(float(np.linalg.norm(ki)), 1.0)
        elastic = (
            np.abs(np.linalg.norm(kf_sample, axis=-1) - np.linalg.norm(ki))
            <= 1024.0 * np.finfo(np.float64).eps * elastic_scale
        )
        root_valid = root_exists[None, :] & elastic & (kf_sample[..., 2] > 0.0)
        rods = self.detector_measure.coating.bragg_space.config.rods
        if not rods:
            raise ValueError("detector measure must retain at least one physical rod")
        reference_rod = rods[0]
        status = np.where(root_valid, "regular", "no_root")
        geometry = EwaldLatentGeometry(
            rod=reference_rod,
            branch=0 if reference_rod.family_m == 0 else 1,
            alpha_rad=np.zeros((2, size), dtype=np.float64),
            beta_rad=np.zeros((2, size), dtype=np.float64),
            u_Ainv=np.zeros((2, size), dtype=np.float64),
            L=np.broadcast_to(flat_l, (2, size)),
            q_sample_Ainv=q_sample,
            kf_sample_Ainv=kf_sample,
            ewald_residual_Ainv=np.zeros((2, size), dtype=np.float64),
            status=status,
        )
        mapped = self.detector_measure.map_ewald_geometry(geometry)
        column = np.asarray(mapped.column_px)
        row = np.asarray(mapped.row_px)
        kf_air = np.asarray(mapped.kf_air_sample_Ainv)
        air_exit = np.arctan2(kf_air[..., 2], np.hypot(kf_air[..., 0], kf_air[..., 1]))
        lower_column, upper_column = self.detector_column_interval_px
        side_valid = (column >= lower_column) & (column <= upper_column)
        eligible = root_valid & mapped.valid & side_valid & (air_exit >= self.air_exit_guard_rad)
        eligible_count = np.sum(eligible, axis=0)
        selected_root = np.argmax(eligible, axis=0)
        node = np.arange(size, dtype=np.int64)
        selected_column = column[selected_root, node]
        selected_row = row[selected_root, node]
        selected_q = q_sample[selected_root, node]
        selected_kf = kf_sample[selected_root, node]
        selected_radial_direction = radial_direction[selected_root, node]
        selected_tangent_direction = tangent_direction[selected_root, node]

        detector_geometry = self.detector_measure.evaluate_detector_geometry(
            selected_column,
            selected_row,
            include_surface_jacobian=True,
        )
        recovered_qr, recovered_l = self.reciprocal_frame.coordinates(
            detector_geometry.q_sample_Ainv
        )
        coordinate_scale_qr = np.maximum(np.abs(flat_qr), 1.0)
        coordinate_scale_l = np.maximum(np.abs(flat_l), 1.0)
        coordinate_valid = (
            np.abs(recovered_qr - flat_qr)
            <= 4096.0 * np.finfo(np.float64).eps * coordinate_scale_qr
        ) & (np.abs(recovered_l - flat_l) <= 4096.0 * np.finfo(np.float64).eps * coordinate_scale_l)
        q_match = (
            np.linalg.norm(
                np.asarray(detector_geometry.q_sample_Ainv) - selected_q,
                axis=-1,
            )
            <= 4096.0 * np.finfo(np.float64).eps * elastic_scale
        )
        kf_dot_tangent = np.einsum(
            "ij,ij->i",
            selected_kf,
            selected_tangent_direction,
            optimize=True,
        )
        tangent_scale = np.maximum(np.linalg.norm(selected_kf, axis=-1), 1.0)
        regular_chart = (
            root_exists
            & (flat_qr > 0.0)
            & (np.abs(kf_dot_tangent) > 1024.0 * np.finfo(np.float64).eps * tangent_scale)
        )
        d_gamma_d_qr = np.zeros(size, dtype=np.float64)
        d_gamma_d_l = np.zeros(size, dtype=np.float64)
        d_gamma_d_qr[regular_chart] = -np.einsum(
            "ij,ij->i",
            selected_kf[regular_chart],
            selected_radial_direction[regular_chart],
            optimize=True,
        ) / (flat_qr[regular_chart] * kf_dot_tangent[regular_chart])
        d_gamma_d_l[regular_chart] = (
            -self._axial_basis_norm_Ainv
            * (selected_kf[regular_chart] @ normal)
            / (flat_qr[regular_chart] * kf_dot_tangent[regular_chart])
        )
        d_q_d_qr = (
            selected_radial_direction
            + (flat_qr * d_gamma_d_qr)[:, None] * selected_tangent_direction
        )
        d_q_d_l = (
            self._axial_basis_norm_Ainv * normal[None, :]
            + (flat_qr * d_gamma_d_l)[:, None] * selected_tangent_direction
        )
        q_chart_surface_jacobian = np.linalg.norm(
            np.cross(d_q_d_qr, d_q_d_l),
            axis=-1,
        )
        q_detector_surface_jacobian = np.asarray(detector_geometry.q_surface_jacobian_Ainv2_per_px2)
        final_valid = (
            (eligible_count == 1)
            & detector_geometry.valid
            & coordinate_valid
            & q_match
            & regular_chart
            & np.isfinite(q_chart_surface_jacobian)
            & (q_chart_surface_jacobian > 0.0)
            & np.isfinite(q_detector_surface_jacobian)
            & (q_detector_surface_jacobian > 0.0)
        )
        area_jacobian = np.zeros(size, dtype=np.float64)
        area_jacobian[final_valid] = (
            q_chart_surface_jacobian[final_valid] / q_detector_surface_jacobian[final_valid]
        )
        output_column = np.zeros(size, dtype=np.float64)
        output_row = np.zeros(size, dtype=np.float64)
        output_column[final_valid] = selected_column[final_valid]
        output_row[final_valid] = selected_row[final_valid]
        return ContinuousDetectorChartAreaMeasure(
            column_px=output_column.reshape(shape),
            row_px=output_row.reshape(shape),
            detector_area_jacobian_px2_per_chart2=area_jacobian.reshape(shape),
            valid=final_valid.reshape(shape),
        )


__all__ = ["LayeredReciprocalDetectorAreaChart"]
