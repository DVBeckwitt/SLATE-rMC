"""Sparse simultaneous ordered-structure fitting at fixed geometry and mosaic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from painted_ewald import MosaicParameters, Rod
from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.fitting.mosaic import (
    MosaicProfileDefinition,
    MosaicProfileIdentity,
    _mosaic_profile_quadrature,
)
from rasim_next.geometry.angles import AngleFrame, angles_to_detector_coordinate_area_measure
from rasim_next.measurement.continuous_angle import evaluate_continuous_per_rod_angle_signal
from rasim_next.ordered import Bi2X3QuintupleLayerParameters
from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure
from rasim_next.pipeline.source_averaged_detector import (
    SourceAveragedDetectorEwaldMeasure,
    source_averaged_detector_instrument_revision,
)
from rasim_next.pipeline.source_averaged_structure import SourceAveragedStructureDetector

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

STRUCTURE_FACTOR_PARAMETER_NAMES = (
    "bi_delta_z_fractional",
    "se2_delta_z_fractional",
    "bi_occupancy",
    "se1_occupancy",
    "se2_occupancy",
    "u_radial_A2",
    "u_normal_A2",
)
_OCCUPANCY_PARAMETER_NAMES = (
    "bi_occupancy",
    "se1_occupancy",
    "se2_occupancy",
)
_POSITION_PARAMETER_NAMES = (
    "bi_delta_z_fractional",
    "se2_delta_z_fractional",
)
ORDERED_INTENSITY_TOPOLOGY_PROBE_REVISION = "source_resolved_inverse_root_signature_grid_17x17.v2"
_RESPONSE_COORDINATE_BLOCK_SIZE = 32_768
_STRUCTURE_KERNEL_TERM_BLOCK_SIZE = 131_072
_POINT_RESPONSE_U_NORMAL_MAX_A2 = 0.1
_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT = 13
_POINT_RESPONSE_VALIDATION_NODE_COUNT = 12
_POINT_RESPONSE_MAXIMUM_INTERPOLATION_RELATIVE_ERROR = 2.0e-3
SOURCE_AVERAGED_ORDERED_INTENSITY_RESPONSE_CONTRACT_REVISION = (
    "source-averaged-selected-center-occ-quadratic-chebyshev-qz-spectral.v2"
)
SOURCE_AVERAGED_ORDERED_INTENSITY_SIGNAL_CERTIFICATE_RELATIVE_FLOOR = 1.0e-12


def _profile_identity_revision(identity: MosaicProfileIdentity) -> str:
    group = identity.group_key
    member_rods = np.asarray(group.member_rod_hk, dtype=np.int64)
    common_fields: tuple[tuple[str, object], ...] = (
        ("dataset_id", identity.dataset_id),
        ("incidence_angle_rad", identity.incidence_angle_rad),
        ("group_id", group.group_id),
        ("rod_catalog_revision", group.rod_catalog_revision),
        ("member_rod_h", member_rods[:, 0]),
        ("member_rod_k", member_rods[:, 1]),
        ("branch_mode", group.branch_mode),
        ("layered_family_m", -1 if group.layered_family_m is None else group.layered_family_m),
        ("layered_integer_L", 0 if group.layered_integer_L is None else group.layered_integer_L),
        ("branch_id", 0 if identity.branch_id is None else identity.branch_id),
        ("analytic_branch_id", identity.analytic_branch_id),
    )
    order = group.layered_layer_order
    if order is None:
        return canonical_revision_sha256(
            ("definition_id", "ordered_intensity_profile_identity.v1"),
            *common_fields,
        )
    return canonical_revision_sha256(
        ("definition_id", "ordered_intensity_profile_identity.v2"),
        *common_fields,
        ("layered_coordinate_kind", "exact-rational"),
        ("layered_order_numerator", order.numerator),
        ("layered_order_denominator", order.denominator),
        ("layered_reciprocal_basis_revision", group.layered_reciprocal_basis_revision),
    )


def ordered_intensity_profile_catalog_revision(
    definitions: tuple[MosaicProfileDefinition, ...],
) -> str:
    """Hash the ordered identity and membership of one frozen profile catalog."""

    frozen = tuple(definitions)
    if not frozen or any(not isinstance(item, MosaicProfileDefinition) for item in frozen):
        raise ValueError("definitions must contain MosaicProfileDefinition values")
    return canonical_revision_sha256(
        ("definition_id", "ordered_intensity_profile_catalog.v1"),
        (
            "profile_identity_revision",
            tuple(_profile_identity_revision(item.identity) for item in frozen),
        ),
    )


def _ordered_intensity_observable_revision(
    definitions: tuple[MosaicProfileDefinition, ...],
    *,
    angle_frame_revision: str,
) -> str:
    """Hash the selected-component angular observable without numerical quadrature order."""

    return canonical_revision_sha256(
        ("definition_id", "selected_group_angle_roi_mass.v1"),
        ("measure", "integrated_selected_group_detector_density_A2.v1"),
        ("angle_frame_revision", angle_frame_revision),
        ("topology_probe_revision", ORDERED_INTENSITY_TOPOLOGY_PROBE_REVISION),
        (
            "profile_identity_revision",
            ordered_intensity_profile_catalog_revision(definitions),
        ),
        (
            "center_two_theta_rad",
            np.asarray([item.center_two_theta_rad for item in definitions], dtype=np.float64),
        ),
        (
            "center_phi_rad",
            np.asarray([item.center_phi_rad for item in definitions], dtype=np.float64),
        ),
        (
            "two_theta_half_width_rad",
            np.asarray([item.two_theta_half_width_rad for item in definitions], dtype=np.float64),
        ),
        (
            "phi_half_width_rad",
            np.asarray([item.phi_half_width_rad for item in definitions], dtype=np.float64),
        ),
        ("phi_bin_count", np.asarray([item.phi_bin_count for item in definitions], dtype=np.int64)),
        (
            "excluded_phi_bin_indices",
            tuple(
                ",".join(str(index) for index in item.excluded_phi_bin_indices)
                for item in definitions
            ),
        ),
    )


def _ordered_intensity_point_observable_revision(
    definitions: tuple[MosaicProfileDefinition, ...],
    *,
    angle_frame_revision: str,
) -> str:
    """Hash selected-group signal density at frozen angular peak anchors."""

    return canonical_revision_sha256(
        ("definition_id", "selected_group_angle_anchor_signal.v1"),
        ("measure", "selected_group_angular_signal_density_A2_per_rad2.v1"),
        ("angle_frame_revision", angle_frame_revision),
        (
            "profile_identity_revision",
            ordered_intensity_profile_catalog_revision(definitions),
        ),
        (
            "center_two_theta_rad",
            np.asarray([item.center_two_theta_rad for item in definitions], dtype=np.float64),
        ),
        (
            "center_phi_rad",
            np.asarray([item.center_phi_rad for item in definitions], dtype=np.float64),
        ),
    )


def ordered_intensity_structure_model_revision(strength: Bi2X3FiniteStackStrength) -> str:
    """Hash coefficient-generating structure physics, excluding fitted occupancies/U and wavelength."""

    if not isinstance(strength, Bi2X3FiniteStackStrength):
        raise TypeError("strength must be Bi2X3FiniteStackStrength")
    parameters = strength.structure_parameters
    if not isinstance(parameters, Bi2X3QuintupleLayerParameters):
        raise TypeError("layered-quintuple strength requires resolved structure parameters")
    crystal = strength.crystal
    sites = crystal.sites
    return canonical_revision_sha256(
        ("definition_id", "bi2se3_fixed_position_occ_directional_u.v1"),
        ("phase_id", crystal.phase_id),
        ("spacegroup_hm", crystal.spacegroup_hm),
        ("direct_basis_A", crystal.direct_basis_A),
        ("site_source_label", tuple(site.source_label for site in sites)),
        ("site_species", tuple(site.species for site in sites)),
        ("site_element", tuple(site.element for site in sites)),
        ("site_charge", np.asarray([site.charge for site in sites], dtype=np.int64)),
        ("site_fractional", np.asarray([site.fractional for site in sites], dtype=np.float64)),
        (
            "site_source_multiplicity",
            np.asarray([site.source_multiplicity for site in sites], dtype=np.int64),
        ),
        ("bi_fractional_z", parameters.bi_fractional_z),
        ("se2_fractional_z", parameters.se2_fractional_z),
        ("stacking_parent", strength.parent.value),
        ("stacking_layers", strength.layers),
        ("stacking_normalization", strength.normalization.value),
        ("shared_disorder_epsilon", strength.shared_disorder_epsilon),
        ("initial_population", "plus_only"),
        ("form_factor_authority", "xraydb_f0_f1_if2.q_abs_over_4pi.v1"),
        ("finite_strength_conversion", "classical_electron_radius_squared_A2.v1"),
    )


def _mosaic_parameters_revision(mosaic: MosaicParameters) -> str:
    return canonical_revision_sha256(
        ("definition_id", "folded_alpha_full_azimuth_wrapped_mosaic.v1"),
        ("gaussian_sigma_rad", mosaic.gaussian_sigma_rad),
        ("lorentzian_half_width_rad", mosaic.lorentzian_half_width_rad),
        ("lorentzian_probability", mosaic.lorentzian_probability),
        ("measure", "dalpha_dbeta.folded_alpha_probability.v1"),
    )


def probe_ordered_intensity_inverse_boundary_bins(
    detector: DetectorEwaldMeasure | SourceAveragedStructureDetector,
    *,
    angle_frame: AngleFrame,
    definitions: tuple[MosaicProfileDefinition, ...],
) -> tuple[MosaicProfileDefinition, ...]:
    """Omit phi bins whose source-resolved inverse-root topology changes.

    The fixed 17 by 17 geometry probe includes cell edges and interior points. Classification uses
    detector validity, source-state/rod/root-sign multiplicity, and exact caustic flags; structure
    strength and observed intensity never enter the decision. Finite probing is not a general
    topology certificate, so response-order convergence remains required for each material/ROI.
    """

    if not isinstance(detector, (DetectorEwaldMeasure, SourceAveragedStructureDetector)):
        raise TypeError("detector must expose one-state or source-resolved structure geometry")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    frozen = tuple(definitions)
    if not frozen or any(not isinstance(item, MosaicProfileDefinition) for item in frozen):
        raise ValueError("definitions must contain MosaicProfileDefinition values")
    if isinstance(detector, DetectorEwaldMeasure):
        configured = detector.coating.bragg_space.config.rods
    else:
        configured = detector.rods
    configured_rods = {(rod.h, rod.k): rod for rod in configured}
    probe_count = 17
    unit_probe = np.linspace(0.0, 1.0, probe_count)
    classified: list[MosaicProfileDefinition] = []
    for definition in frozen:
        try:
            rods = tuple(
                configured_rods[rod_hk] for rod_hk in definition.identity.group_key.member_rod_hk
            )
        except KeyError as error:
            raise ValueError(f"profile references unconfigured rod {error.args[0]}") from error
        theta_probe = (
            definition.center_two_theta_rad
            - definition.two_theta_half_width_rad
            + 2.0 * definition.two_theta_half_width_rad * unit_probe
        )
        phi_edges = np.linspace(
            definition.center_phi_rad - definition.phi_half_width_rad,
            definition.center_phi_rad + definition.phi_half_width_rad,
            definition.phi_bin_count + 1,
        )
        phi_probe = phi_edges[:-1, None] + (phi_edges[1:] - phi_edges[:-1])[:, None] * unit_probe
        two_theta = np.broadcast_to(
            theta_probe[None, :, None],
            (definition.phi_bin_count, probe_count, probe_count),
        )
        phi = np.broadcast_to(
            phi_probe[:, None, :],
            (definition.phi_bin_count, probe_count, probe_count),
        )
        coordinate_measure = angles_to_detector_coordinate_area_measure(
            two_theta,
            phi,
            instrument=detector.instrument,
            angle_frame=angle_frame,
        )
        column = coordinate_measure.coordinates.column_px.ravel()
        row = coordinate_measure.coordinates.row_px.ravel()
        if isinstance(detector, DetectorEwaldMeasure):
            response = detector.evaluate_detector_structure_response(column, row, rods=rods)
            source_count = 1
            term_source = np.zeros(response.term_coordinate_index.size, dtype=np.int64)
            valid = (
                coordinate_measure.coordinates.valid.ravel() & response.coordinate_valid
            ).reshape(definition.phi_bin_count, probe_count * probe_count)
        else:
            response = detector.restrict_rods(rods).compile_structure_response(column, row)
            source_count = response.source_state_count
            term_source = response.term_source_state_index
            valid = np.column_stack(
                (
                    coordinate_measure.coordinates.valid.ravel(),
                    response.valid_source_count.ravel(),
                )
            ).reshape(definition.phi_bin_count, probe_count * probe_count, 2)
        signature = np.zeros(
            (two_theta.size, source_count, len(rods), 3),
            dtype=np.uint16,
        )
        np.add.at(
            signature,
            (
                response.term_coordinate_index,
                term_source,
                response.term_rod_index,
                response.term_root_sign.astype(np.int64) + 1,
            ),
            1,
        )
        signature = signature.reshape(
            definition.phi_bin_count,
            probe_count * probe_count,
            source_count,
            len(rods),
            3,
        )
        caustic = response.per_rod_caustic.reshape(
            definition.phi_bin_count,
            probe_count * probe_count,
            len(rods),
        )
        topology_change = np.any(signature != signature[:, :1, :, :, :], axis=(1, 2, 3, 4))
        validity_change = np.any(valid != valid[:, :1, ...], axis=tuple(range(1, valid.ndim)))
        exact_caustic = np.any(caustic, axis=(1, 2))
        excluded = tuple(
            sorted(
                {
                    *definition.excluded_phi_bin_indices,
                    *np.flatnonzero(topology_change | validity_change | exact_caustic).tolist(),
                }
            )
        )
        classified.append(replace(definition, excluded_phi_bin_indices=excluded))
    return tuple(classified)


class OrderedIntensityIdentifiabilityError(ValueError):
    """The requested active structure coordinates are not independently observable."""


@dataclass(frozen=True, slots=True)
class OrderedIntensityDatasetResponse:
    """One fixed-incidence sparse response from structure strength to ROI mass."""

    dataset_id: str
    incidence_angle_rad: float
    identities: tuple[MosaicProfileIdentity, ...]
    rods: tuple[Rod, ...]
    term_observation_index: IntArray
    term_rod_index: IntArray
    term_L: FloatArray
    term_fixed_mass_per_strength: FloatArray
    term_root_sign: NDArray[np.int8]
    term_occupancy_quadratic_strength_A2: FloatArray
    term_q_radial_squared_Ainv2: FloatArray
    term_q_normal_squared_Ainv2: FloatArray
    normalization_mass_px2: FloatArray
    reciprocal_basis_Ainv: FloatArray
    k_norm_Ainv: float
    fixed_structure_parameters: Bi2X3QuintupleLayerParameters
    rod_catalog_revision: str
    structure_model_revision: str
    mosaic_model_revision: str
    angle_frame_revision: str
    source_revision: str
    sample_geometry_revision: str
    material_revision: str
    observable_revision: str
    excluded_phi_bin_indices: tuple[tuple[int, ...], ...]
    topology_probe_revision: str
    response_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be nonempty")
        for name in (
            "rod_catalog_revision",
            "structure_model_revision",
            "mosaic_model_revision",
            "angle_frame_revision",
            "source_revision",
            "sample_geometry_revision",
            "material_revision",
            "observable_revision",
            "topology_probe_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")
        incidence = float(self.incidence_angle_rad)
        if not math.isfinite(incidence):
            raise ValueError("incidence_angle_rad must be finite")
        identities = tuple(self.identities)
        if (
            not identities
            or any(not isinstance(identity, MosaicProfileIdentity) for identity in identities)
            or len(set(identities)) != len(identities)
            or any(identity.dataset_id != self.dataset_id for identity in identities)
            or any(identity.incidence_angle_rad != incidence for identity in identities)
            or any(
                identity.group_key.rod_catalog_revision != self.rod_catalog_revision
                for identity in identities
            )
        ):
            raise ValueError("identities must be unique and match this dataset and rod catalog")
        excluded = tuple(tuple(indices) for indices in self.excluded_phi_bin_indices)
        if len(excluded) != len(identities) or any(
            tuple(sorted(set(indices))) != indices or any(index < 0 for index in indices)
            for indices in excluded
        ):
            raise ValueError("excluded_phi_bin_indices must identify sorted masks per profile")
        rods = tuple(self.rods)
        if not rods or any(not isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain physical Rod values")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical line")
        observation = np.array(self.term_observation_index, dtype=np.int64, copy=True, order="C")
        rod_index = np.array(self.term_rod_index, dtype=np.int64, copy=True, order="C")
        ell = np.array(self.term_L, dtype=np.float64, copy=True, order="C")
        fixed = np.array(
            self.term_fixed_mass_per_strength,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        root_sign = np.array(self.term_root_sign, dtype=np.int8, copy=True, order="C")
        occupancy_quadratic = np.array(
            self.term_occupancy_quadratic_strength_A2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        q_radial_squared = np.array(
            self.term_q_radial_squared_Ainv2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        q_normal_squared = np.array(
            self.term_q_normal_squared_Ainv2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        term_shape = observation.shape
        if (
            observation.ndim != 1
            or rod_index.shape != term_shape
            or ell.shape != term_shape
            or fixed.shape != term_shape
            or root_sign.shape != term_shape
            or occupancy_quadratic.shape != (*term_shape, 6)
            or q_radial_squared.shape != term_shape
            or q_normal_squared.shape != term_shape
        ):
            raise ValueError("sparse response term arrays must align")
        if (
            not observation.size
            or np.any((observation < 0) | (observation >= len(identities)))
            or np.any((rod_index < 0) | (rod_index >= len(rods)))
            or not np.all(np.isfinite(ell))
            or not np.all(np.isfinite(fixed))
            or np.any(fixed <= 0.0)
            or np.any(~np.isin(root_sign, (-1, 0, 1)))
            or not np.all(np.isfinite(occupancy_quadratic))
            or not np.all(np.isfinite(q_radial_squared))
            or np.any(q_radial_squared < 0.0)
            or not np.all(np.isfinite(q_normal_squared))
            or np.any(q_normal_squared < 0.0)
        ):
            raise ValueError("sparse response terms must be finite, positive, and in range")
        if np.unique(observation).size != len(identities):
            raise ValueError("every observation requires at least one positive response term")
        normalization = np.array(
            self.normalization_mass_px2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if (
            normalization.shape != (len(identities),)
            or not np.all(np.isfinite(normalization))
            or np.any(normalization <= 0.0)
        ):
            raise ValueError("every observation requires positive finite detector-area support")
        basis = np.array(self.reciprocal_basis_Ainv, dtype=np.float64, copy=True, order="C")
        if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
            raise ValueError("reciprocal_basis_Ainv must be finite with shape (3, 3)")
        k_norm = float(self.k_norm_Ainv)
        if not math.isfinite(k_norm) or k_norm <= 0.0:
            raise ValueError("k_norm_Ainv must be finite and positive")
        if not isinstance(self.fixed_structure_parameters, Bi2X3QuintupleLayerParameters):
            raise TypeError("fixed_structure_parameters must be Bi2X3QuintupleLayerParameters")
        for value in (
            observation,
            rod_index,
            ell,
            fixed,
            root_sign,
            occupancy_quadratic,
            q_radial_squared,
            q_normal_squared,
            normalization,
            basis,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "incidence_angle_rad", incidence)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "term_observation_index", observation)
        object.__setattr__(self, "term_rod_index", rod_index)
        object.__setattr__(self, "term_L", ell)
        object.__setattr__(self, "term_fixed_mass_per_strength", fixed)
        object.__setattr__(self, "term_root_sign", root_sign)
        object.__setattr__(
            self,
            "term_occupancy_quadratic_strength_A2",
            occupancy_quadratic,
        )
        object.__setattr__(self, "term_q_radial_squared_Ainv2", q_radial_squared)
        object.__setattr__(self, "term_q_normal_squared_Ainv2", q_normal_squared)
        object.__setattr__(self, "normalization_mass_px2", normalization)
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "k_norm_Ainv", k_norm)
        object.__setattr__(self, "excluded_phi_bin_indices", excluded)
        object.__setattr__(
            self,
            "response_revision",
            canonical_revision_sha256(
                ("definition_id", "fixed_detector_ordered_intensity_response.bounded.v3"),
                ("dataset_id", self.dataset_id),
                ("incidence_angle_rad", incidence),
                (
                    "profile_identity_revision",
                    tuple(_profile_identity_revision(identity) for identity in identities),
                ),
                ("rod_h", np.asarray([rod.h for rod in rods], dtype=np.int64)),
                ("rod_k", np.asarray([rod.k for rod in rods], dtype=np.int64)),
                ("rod_population", np.asarray([rod.population for rod in rods], dtype=np.float64)),
                ("term_observation_index", observation),
                ("term_rod_index", rod_index),
                ("term_L", ell),
                ("term_fixed_mass_per_strength", fixed),
                ("term_root_sign", root_sign),
                ("term_occupancy_quadratic_strength_A2", occupancy_quadratic),
                ("term_q_radial_squared_Ainv2", q_radial_squared),
                ("term_q_normal_squared_Ainv2", q_normal_squared),
                ("normalization_mass_px2", normalization),
                ("reciprocal_basis_Ainv", basis),
                ("k_norm_Ainv", k_norm),
                ("rod_catalog_revision", self.rod_catalog_revision),
                ("structure_model_revision", self.structure_model_revision),
                ("mosaic_model_revision", self.mosaic_model_revision),
                ("angle_frame_revision", self.angle_frame_revision),
                ("source_revision", self.source_revision),
                ("sample_geometry_revision", self.sample_geometry_revision),
                ("material_revision", self.material_revision),
                ("observable_revision", self.observable_revision),
                (
                    "excluded_phi_bin_indices",
                    tuple(",".join(str(index) for index in indices) for indices in excluded),
                ),
                ("topology_probe_revision", self.topology_probe_revision),
            ),
        )

    def predict_mass_A2(self, structure_parameters: Bi2X3QuintupleLayerParameters) -> FloatArray:
        """Apply occupancies and directional damping to the frozen response."""

        if not isinstance(structure_parameters, Bi2X3QuintupleLayerParameters):
            raise TypeError("structure_parameters must be Bi2X3QuintupleLayerParameters")
        fixed = self.fixed_structure_parameters
        if (
            structure_parameters.bi_fractional_z != fixed.bi_fractional_z
            or structure_parameters.se2_fractional_z != fixed.se2_fractional_z
        ):
            raise ValueError("atomic positions differ from the compiled fixed response")
        return self.predict_mass_coordinates_A2(
            occupancy_coefficients=(
                structure_parameters.bi_occupancy,
                structure_parameters.se1_occupancy,
                structure_parameters.se2_occupancy,
            ),
            u_radial_A2=structure_parameters.u_radial_A2,
            u_normal_A2=structure_parameters.u_normal_A2,
        )

    def predict_mass_coordinates_A2(
        self,
        *,
        occupancy_coefficients: ArrayLike,
        u_radial_A2: float,
        u_normal_A2: float,
    ) -> FloatArray:
        """Contract nonnegative occupancy coordinates and directional damping."""

        occupancy = np.asarray(occupancy_coefficients, dtype=np.float64)
        if occupancy.shape != (3,) or not np.all(np.isfinite(occupancy)) or np.any(occupancy < 0.0):
            raise ValueError("occupancy_coefficients must be three finite nonnegative values")
        u_radial = float(u_radial_A2)
        u_normal = float(u_normal_A2)
        if (
            not math.isfinite(u_radial)
            or not math.isfinite(u_normal)
            or u_radial < 0.0
            or u_normal < 0.0
        ):
            raise ValueError("directional displacement coordinates must be finite and nonnegative")
        bi, se1, se2 = occupancy
        occupancy_products = np.asarray(
            (bi * bi, se1 * se1, se2 * se2, bi * se1, bi * se2, se1 * se2),
            dtype=np.float64,
        )
        undamped_strength = self.term_occupancy_quadratic_strength_A2 @ occupancy_products
        negative = undamped_strength < 0.0
        if np.any(negative):
            rounding_scale = np.sum(
                np.abs(self.term_occupancy_quadratic_strength_A2[negative])
                * np.abs(occupancy_products)[None, :],
                axis=1,
            )
            tolerance = 2048.0 * np.finfo(np.float64).eps * rounding_scale
            if np.any(undamped_strength[negative] < -tolerance):
                raise FloatingPointError("occupancy quadratic produced negative rod strength")
            undamped_strength = undamped_strength.copy()
            undamped_strength[negative] = 0.0
        damping = np.exp(
            -u_radial * self.term_q_radial_squared_Ainv2
            - u_normal * self.term_q_normal_squared_Ainv2
        )
        term_strength = undamped_strength * damping
        predicted = np.bincount(
            self.term_observation_index,
            weights=self.term_fixed_mass_per_strength * term_strength,
            minlength=len(self.identities),
        )
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
            raise FloatingPointError("candidate structure produced invalid ROI mass")
        predicted.setflags(write=False)
        return predicted

    def predict_mass_direct_A2(self, strength: Bi2X3FiniteStackStrength) -> FloatArray:
        """Evaluate the full strength model once as an independent response oracle."""

        if not isinstance(strength, Bi2X3FiniteStackStrength):
            raise TypeError("strength must be Bi2X3FiniteStackStrength")
        if ordered_intensity_structure_model_revision(strength) != self.structure_model_revision:
            raise ValueError("candidate strength changed the compiled structure model")
        scale = max(float(np.linalg.norm(self.reciprocal_basis_Ainv)), 1.0)
        if not np.allclose(
            strength.reciprocal_basis_Ainv,
            self.reciprocal_basis_Ainv,
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * scale,
        ):
            raise ValueError("candidate strength changed the frozen reciprocal lattice")
        parameters = strength.structure_parameters
        fixed = self.fixed_structure_parameters
        if not isinstance(parameters, Bi2X3QuintupleLayerParameters) or (
            parameters.bi_fractional_z != fixed.bi_fractional_z
            or parameters.se2_fractional_z != fixed.se2_fractional_z
        ):
            raise ValueError("atomic positions differ from the compiled fixed response")
        rod_h = np.fromiter((rod.h for rod in self.rods), dtype=np.int32, count=len(self.rods))
        rod_k = np.fromiter((rod.k for rod in self.rods), dtype=np.int32, count=len(self.rods))
        term_strength = strength.evaluate_hkl(
            h=rod_h[self.term_rod_index],
            k=rod_k[self.term_rod_index],
            L=self.term_L,
            k_norm_Ainv=self.k_norm_Ainv,
        )
        predicted = np.bincount(
            self.term_observation_index,
            weights=self.term_fixed_mass_per_strength * term_strength,
            minlength=len(self.identities),
        )
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
            raise FloatingPointError("candidate structure produced invalid direct ROI mass")
        predicted.setflags(write=False)
        return predicted


@dataclass(frozen=True, slots=True)
class SourceAveragedOrderedIntensityDatasetResponse:
    """Peak-center angular signal after one incoherent source-state sum."""

    dataset_id: str
    incidence_angle_rad: float
    identities: tuple[MosaicProfileIdentity, ...]
    occupancy_quadratic_chebyshev_signal_density_A2_per_rad2: FloatArray
    q_radial_squared_Ainv2: FloatArray
    u_normal_nodes_A2: FloatArray
    center_two_theta_rad: FloatArray
    center_phi_rad: FloatArray
    normalization_density_px2_per_rad2: FloatArray
    reciprocal_basis_Ainv: FloatArray
    fixed_structure_parameters: Bi2X3QuintupleLayerParameters
    rod_catalog_revision: str
    structure_model_revision: str
    mosaic_model_revision: str
    angle_frame_revision: str
    source_revision: str
    source_state_count: int
    sample_geometry_revision: str
    material_revision: str
    interpolation_validation_maximum_relative_error: float
    interpolation_validation_node_count: int
    instrument_revision: str
    execution_backend: str
    execution_device: str | None
    observable_revision: str
    measure_id: str = "selected_group_angular_signal_density_A2_per_rad2.v1"
    root_policy: str = "all_retained_roots.v1"
    response_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be nonempty")
        incidence = float(self.incidence_angle_rad)
        if not math.isfinite(incidence):
            raise ValueError("incidence_angle_rad must be finite")
        for name in (
            "rod_catalog_revision",
            "structure_model_revision",
            "mosaic_model_revision",
            "angle_frame_revision",
            "source_revision",
            "sample_geometry_revision",
            "material_revision",
            "instrument_revision",
            "observable_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")
        state_count = self.source_state_count
        if (
            isinstance(state_count, bool)
            or not isinstance(state_count, (int, np.integer))
            or int(state_count) < 1
        ):
            raise ValueError("source_state_count must be a positive integer")
        validation_error = float(self.interpolation_validation_maximum_relative_error)
        validation_count = self.interpolation_validation_node_count
        if (
            not math.isfinite(validation_error)
            or validation_error < 0.0
            or validation_error > _POINT_RESPONSE_MAXIMUM_INTERPOLATION_RELATIVE_ERROR
            or isinstance(validation_count, bool)
            or not isinstance(validation_count, (int, np.integer))
            or int(validation_count) != _POINT_RESPONSE_VALIDATION_NODE_COUNT
        ):
            raise ValueError(
                "peak-center interpolation certificate is missing or outside tolerance"
            )
        identities = tuple(self.identities)
        if (
            not identities
            or any(not isinstance(identity, MosaicProfileIdentity) for identity in identities)
            or len(set(identities)) != len(identities)
            or any(identity.dataset_id != self.dataset_id for identity in identities)
            or any(identity.incidence_angle_rad != incidence for identity in identities)
            or any(
                identity.group_key.rod_catalog_revision != self.rod_catalog_revision
                for identity in identities
            )
        ):
            raise ValueError("identities must be unique and match this dataset and rod catalog")
        coefficients = np.array(
            self.occupancy_quadratic_chebyshev_signal_density_A2_per_rad2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        q_radial_squared = np.array(
            self.q_radial_squared_Ainv2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        nodes = np.array(
            self.u_normal_nodes_A2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        center_two_theta = np.array(
            self.center_two_theta_rad,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        center_phi = np.array(
            self.center_phi_rad,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        normalization = np.array(
            self.normalization_density_px2_per_rad2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        profile_count = len(identities)
        if (
            coefficients.shape != (profile_count, 6, _POINT_RESPONSE_CHEBYSHEV_NODE_COUNT)
            or q_radial_squared.shape != (profile_count,)
            or nodes.shape != (_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT,)
            or center_two_theta.shape != (profile_count,)
            or center_phi.shape != (profile_count,)
            or normalization.shape != (profile_count,)
            or not np.all(np.isfinite(coefficients))
            or not np.all(np.isfinite(q_radial_squared))
            or np.any(q_radial_squared < 0.0)
            or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(center_two_theta))
            or np.any((center_two_theta < 0.0) | (center_two_theta > math.pi))
            or not np.all(np.isfinite(center_phi))
            or not np.all(np.isfinite(normalization))
            or np.any(normalization <= 0.0)
        ):
            raise ValueError("peak-center response arrays are invalid or misaligned")
        expected_scaled_nodes = np.sort(
            np.cos(
                np.pi
                * np.arange(_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, dtype=np.float64)
                / (_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT - 1)
            )
        )
        expected_nodes = 0.5 * _POINT_RESPONSE_U_NORMAL_MAX_A2 * (expected_scaled_nodes + 1.0)
        expected_nodes[0] = 0.0
        expected_nodes[-1] = _POINT_RESPONSE_U_NORMAL_MAX_A2
        if not np.array_equal(nodes, expected_nodes):
            raise ValueError("u_normal_nodes_A2 must equal the canonical Chebyshev-Lobatto grid")
        if self.measure_id != "selected_group_angular_signal_density_A2_per_rad2.v1":
            raise ValueError("unsupported source-averaged ordered-intensity measure")
        if self.root_policy != "all_retained_roots.v1":
            raise ValueError("source-averaged point responses require all retained roots")
        if self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
        }:
            raise ValueError("unsupported source-averaged response execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        if (self.execution_backend == "numba_cuda_source_averaged.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("execution_device must identify exactly the CUDA backend")
        basis = np.array(self.reciprocal_basis_Ainv, dtype=np.float64, copy=True, order="C")
        if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
            raise ValueError("reciprocal_basis_Ainv must be finite with shape (3, 3)")
        if not isinstance(self.fixed_structure_parameters, Bi2X3QuintupleLayerParameters):
            raise TypeError("fixed_structure_parameters must be Bi2X3QuintupleLayerParameters")
        for value in (
            coefficients,
            q_radial_squared,
            nodes,
            center_two_theta,
            center_phi,
            normalization,
            basis,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "incidence_angle_rad", incidence)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(
            self,
            "occupancy_quadratic_chebyshev_signal_density_A2_per_rad2",
            coefficients,
        )
        object.__setattr__(self, "q_radial_squared_Ainv2", q_radial_squared)
        object.__setattr__(self, "u_normal_nodes_A2", nodes)
        object.__setattr__(self, "center_two_theta_rad", center_two_theta)
        object.__setattr__(self, "center_phi_rad", center_phi)
        object.__setattr__(self, "normalization_density_px2_per_rad2", normalization)
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "source_state_count", int(state_count))
        object.__setattr__(
            self,
            "interpolation_validation_maximum_relative_error",
            validation_error,
        )
        object.__setattr__(self, "interpolation_validation_node_count", int(validation_count))
        object.__setattr__(
            self,
            "response_revision",
            canonical_revision_sha256(
                (
                    "definition_id",
                    SOURCE_AVERAGED_ORDERED_INTENSITY_RESPONSE_CONTRACT_REVISION,
                ),
                ("dataset_id", self.dataset_id),
                ("incidence_angle_rad", incidence),
                (
                    "profile_identity_revision",
                    tuple(_profile_identity_revision(identity) for identity in identities),
                ),
                (
                    "occupancy_quadratic_chebyshev_signal_density_A2_per_rad2",
                    coefficients,
                ),
                ("q_radial_squared_Ainv2", q_radial_squared),
                ("u_normal_nodes_A2", nodes),
                ("center_two_theta_rad", center_two_theta),
                ("center_phi_rad", center_phi),
                ("normalization_density_px2_per_rad2", normalization),
                ("reciprocal_basis_Ainv", basis),
                ("rod_catalog_revision", self.rod_catalog_revision),
                ("structure_model_revision", self.structure_model_revision),
                ("mosaic_model_revision", self.mosaic_model_revision),
                ("angle_frame_revision", self.angle_frame_revision),
                ("source_revision", self.source_revision),
                ("source_state_count", int(state_count)),
                ("sample_geometry_revision", self.sample_geometry_revision),
                ("material_revision", self.material_revision),
                (
                    "interpolation_validation_maximum_relative_error",
                    validation_error,
                ),
                ("interpolation_validation_node_count", int(validation_count)),
                ("instrument_revision", self.instrument_revision),
                ("execution_backend", self.execution_backend),
                ("execution_device_present", int(self.execution_device is not None)),
                (
                    "execution_device",
                    "" if self.execution_device is None else self.execution_device,
                ),
                ("observable_revision", self.observable_revision),
                ("measure_id", self.measure_id),
                ("root_policy", self.root_policy),
                ("directional_damping", "exact_qr_chebyshev_qz.v2"),
                (
                    "signal_certificate_relative_floor",
                    SOURCE_AVERAGED_ORDERED_INTENSITY_SIGNAL_CERTIFICATE_RELATIVE_FLOOR,
                ),
            ),
        )

    def predict_signal_density_A2_per_rad2(
        self,
        structure_parameters: Bi2X3QuintupleLayerParameters,
    ) -> FloatArray:
        if not isinstance(structure_parameters, Bi2X3QuintupleLayerParameters):
            raise TypeError("structure_parameters must be Bi2X3QuintupleLayerParameters")
        fixed = self.fixed_structure_parameters
        if (
            structure_parameters.bi_fractional_z != fixed.bi_fractional_z
            or structure_parameters.se2_fractional_z != fixed.se2_fractional_z
        ):
            raise ValueError("atomic positions differ from the compiled fixed response")
        return self.predict_signal_density_coordinates_A2_per_rad2(
            occupancy_coefficients=(
                structure_parameters.bi_occupancy,
                structure_parameters.se1_occupancy,
                structure_parameters.se2_occupancy,
            ),
            u_radial_A2=structure_parameters.u_radial_A2,
            u_normal_A2=structure_parameters.u_normal_A2,
        )

    def predict_signal_density_coordinates_A2_per_rad2(
        self,
        *,
        occupancy_coefficients: ArrayLike,
        u_radial_A2: float,
        u_normal_A2: float,
    ) -> FloatArray:
        occupancy = np.asarray(occupancy_coefficients, dtype=np.float64)
        if occupancy.shape != (3,) or not np.all(np.isfinite(occupancy)) or np.any(occupancy < 0.0):
            raise ValueError("occupancy_coefficients must be three finite nonnegative values")
        u_radial = float(u_radial_A2)
        u_normal = float(u_normal_A2)
        if (
            not math.isfinite(u_radial)
            or not math.isfinite(u_normal)
            or u_radial < 0.0
            or u_normal < 0.0
            or u_normal > _POINT_RESPONSE_U_NORMAL_MAX_A2
        ):
            raise ValueError("directional displacement coordinates lie outside the response domain")
        bi, se1, se2 = occupancy
        occupancy_products = np.asarray(
            (bi * bi, se1 * se1, se2 * se2, bi * se1, bi * se2, se1 * se2),
            dtype=np.float64,
        )
        scaled_u_normal = 2.0 * u_normal / _POINT_RESPONSE_U_NORMAL_MAX_A2 - 1.0
        coefficient_axis_first = np.moveaxis(
            self.occupancy_quadratic_chebyshev_signal_density_A2_per_rad2,
            -1,
            0,
        )
        occupancy_basis = np.polynomial.chebyshev.chebval(
            scaled_u_normal,
            coefficient_axis_first,
        )
        undamped_signal = occupancy_basis @ occupancy_products
        negative = undamped_signal < 0.0
        if np.any(negative):
            scale = np.sum(
                np.abs(occupancy_basis[negative]) * np.abs(occupancy_products)[None, :],
                axis=1,
            )
            tolerance = 2048.0 * np.finfo(np.float64).eps * scale
            if np.any(undamped_signal[negative] < -tolerance):
                raise FloatingPointError("occupancy quadratic produced negative peak-center signal")
            undamped_signal = undamped_signal.copy()
            undamped_signal[negative] = 0.0
        predicted = undamped_signal * np.exp(-u_radial * self.q_radial_squared_Ainv2)
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
            raise FloatingPointError("candidate structure produced invalid peak-center signal")
        predicted.setflags(write=False)
        return predicted


def _compile_fixed_position_structure_kernel(
    strength: Bi2X3FiniteStackStrength,
    *,
    rods: tuple[Rod, ...],
    term_rod_index: IntArray,
    term_L: FloatArray,
    k_norm_Ainv: float,
) -> tuple[FloatArray, FloatArray, FloatArray, Bi2X3QuintupleLayerParameters]:
    """Compile the occupancy quadratic and reciprocal damping coordinates once."""

    fixed = strength.structure_parameters
    if not isinstance(fixed, Bi2X3QuintupleLayerParameters):
        raise TypeError("layered-quintuple strength requires resolved structure parameters")
    rod_h = np.fromiter((rod.h for rod in rods), dtype=np.int32, count=len(rods))
    rod_k = np.fromiter((rod.k for rod in rods), dtype=np.int32, count=len(rods))
    term_h = rod_h[term_rod_index]
    term_k = rod_k[term_rod_index]

    quadratic = strength.fixed_position_occupancy_quadratic(
        h=term_h,
        k=term_k,
        L=term_L,
        k_norm_Ainv=k_norm_Ainv,
    )
    hkl = np.column_stack((term_h, term_k, term_L))
    q_crystal = hkl @ strength.reciprocal_basis_Ainv.T
    layer_normal = np.cross(
        strength.crystal.direct_basis_A[:, 0],
        strength.crystal.direct_basis_A[:, 1],
    )
    layer_normal /= np.linalg.norm(layer_normal)
    if np.dot(layer_normal, strength.crystal.direct_basis_A[:, 2]) < 0.0:
        layer_normal = -layer_normal
    q_normal = q_crystal @ layer_normal
    q_normal_squared = q_normal * q_normal
    q_radial = q_crystal - q_normal[:, None] * layer_normal[None, :]
    q_radial_squared = np.einsum(
        "ij,ij->i",
        q_radial,
        q_radial,
        optimize=True,
    )
    q_radial_squared[(term_h == 0) & (term_k == 0)] = 0.0
    return quadratic, q_radial_squared, q_normal_squared, fixed


def compile_ordered_intensity_response(
    detector: DetectorEwaldMeasure,
    *,
    angle_frame: AngleFrame,
    definitions: tuple[MosaicProfileDefinition, ...],
) -> OrderedIntensityDatasetResponse:
    """Compile fixed geometry, mosaic, optics, and finite angular ROIs exactly once."""

    if not isinstance(detector, DetectorEwaldMeasure):
        raise TypeError("detector must be a one-state DetectorEwaldMeasure")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    frozen = tuple(definitions)
    if not frozen or any(not isinstance(item, MosaicProfileDefinition) for item in frozen):
        raise ValueError("definitions must contain MosaicProfileDefinition values")
    if len({item.identity for item in frozen}) != len(frozen):
        raise ValueError("definitions must have unique identities")
    dataset_ids = {item.identity.dataset_id for item in frozen}
    incidences = {item.identity.incidence_angle_rad for item in frozen}
    if len(dataset_ids) != 1 or len(incidences) != 1:
        raise ValueError("one response must contain exactly one dataset and incidence")
    rod_catalog_revision = detector.rod_catalog_revision
    if not isinstance(rod_catalog_revision, str) or not rod_catalog_revision:
        raise ValueError("detector must expose its configured rod catalog revision")
    definition_catalog_revisions = {item.identity.group_key.rod_catalog_revision for item in frozen}
    if definition_catalog_revisions != {rod_catalog_revision}:
        raise ValueError("profile definitions do not match the detector rod catalog revision")
    frozen = probe_ordered_intensity_inverse_boundary_bins(
        detector,
        angle_frame=angle_frame,
        definitions=frozen,
    )

    configured_rods = {(rod.h, rod.k): rod for rod in detector.coating.bragg_space.config.rods}
    requested_hk = tuple(
        sorted(
            {
                rod_hk
                for definition in frozen
                for rod_hk in definition.identity.group_key.member_rod_hk
            }
        )
    )
    try:
        rods = tuple(configured_rods[rod_hk] for rod_hk in requested_hk)
    except KeyError as error:
        raise ValueError(f"profile references unconfigured rod {error.args[0]}") from error
    rod_index_by_hk = {(rod.h, rod.k): rod_index for rod_index, rod in enumerate(rods)}

    two_theta, phi, integration_weight, included_node, _, _ = _mosaic_profile_quadrature(frozen)
    included_flat = np.flatnonzero(included_node.ravel())
    coordinate_measure = angles_to_detector_coordinate_area_measure(
        two_theta.ravel()[included_flat],
        phi.ravel()[included_flat],
        instrument=detector.instrument,
        angle_frame=angle_frame,
    )
    profile_at_node = np.broadcast_to(
        np.arange(len(frozen), dtype=np.int64)[:, None, None, None],
        included_node.shape,
    ).ravel()[included_flat]
    node_weight = integration_weight.ravel()[included_flat]
    angle_jacobian = coordinate_measure.detector_area_jacobian_px2_per_rad2
    normalization = np.bincount(
        profile_at_node,
        weights=node_weight * angle_jacobian,
        minlength=len(frozen),
    )

    profiles_by_rod_group: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for profile_index, definition in enumerate(frozen):
        group_hk = definition.identity.group_key.member_rod_hk
        profiles_by_rod_group.setdefault(group_hk, []).append(profile_index)

    term_observation_blocks: list[NDArray[np.int64]] = []
    term_rod_blocks: list[NDArray[np.int64]] = []
    term_l_blocks: list[FloatArray] = []
    term_fixed_blocks: list[FloatArray] = []
    term_root_sign_blocks: list[NDArray[np.int8]] = []
    for group_hk, profile_indices in profiles_by_rod_group.items():
        group_coordinate = np.flatnonzero(np.isin(profile_at_node, profile_indices))
        group_rods = tuple(configured_rods[rod_hk] for rod_hk in group_hk)
        group_rod_index = np.asarray(
            [rod_index_by_hk[rod_hk] for rod_hk in group_hk], dtype=np.int64
        )
        for start in range(0, group_coordinate.size, _RESPONSE_COORDINATE_BLOCK_SIZE):
            coordinate_block = group_coordinate[start : start + _RESPONSE_COORDINATE_BLOCK_SIZE]
            response = detector.evaluate_detector_structure_response(
                coordinate_measure.coordinates.column_px[coordinate_block],
                coordinate_measure.coordinates.row_px[coordinate_block],
                rods=group_rods,
            )
            if np.any(response.per_rod_caustic):
                raise FloatingPointError("an ordered-intensity quadrature node lies on a caustic")
            coordinate_index = coordinate_block[response.term_coordinate_index]
            fixed_mass = (
                response.term_fixed_density_per_strength_px2_inv
                * angle_jacobian[coordinate_index]
                * node_weight[coordinate_index]
            )
            positive = fixed_mass > 0.0
            term_observation_blocks.append(profile_at_node[coordinate_index][positive])
            term_rod_blocks.append(group_rod_index[response.term_rod_index[positive]])
            term_l_blocks.append(response.term_L[positive])
            term_fixed_blocks.append(fixed_mass[positive])
            term_root_sign_blocks.append(response.term_root_sign[positive])
    if not term_fixed_blocks or not any(block.size for block in term_fixed_blocks):
        raise ValueError("profile response contains no positive regular-root terms")
    term_observation = np.concatenate(term_observation_blocks)
    term_rod = np.concatenate(term_rod_blocks)
    term_l = np.concatenate(term_l_blocks)
    term_fixed = np.concatenate(term_fixed_blocks)
    term_root_sign = np.concatenate(term_root_sign_blocks)
    strength = detector.coating.bragg_space.strength_model
    if not isinstance(strength, Bi2X3FiniteStackStrength):
        raise TypeError("ordered-intensity fitting currently requires Bi2X3FiniteStackStrength")
    occupancy_quadratic_blocks: list[FloatArray] = []
    q_radial_squared_blocks: list[FloatArray] = []
    q_normal_squared_blocks: list[FloatArray] = []
    fixed_structure: Bi2X3QuintupleLayerParameters | None = None
    for start in range(0, term_l.size, _STRUCTURE_KERNEL_TERM_BLOCK_SIZE):
        stop = min(start + _STRUCTURE_KERNEL_TERM_BLOCK_SIZE, term_l.size)
        quadratic_block, q_radial_block, q_normal_block, block_structure = (
            _compile_fixed_position_structure_kernel(
                strength,
                rods=rods,
                term_rod_index=term_rod[start:stop],
                term_L=term_l[start:stop],
                k_norm_Ainv=detector.coating.bragg_space.config.k_norm_Ainv,
            )
        )
        if fixed_structure is None:
            fixed_structure = block_structure
        elif block_structure != fixed_structure:
            raise RuntimeError("structure-kernel blocks changed the fixed structure state")
        occupancy_quadratic_blocks.append(quadratic_block)
        q_radial_squared_blocks.append(q_radial_block)
        q_normal_squared_blocks.append(q_normal_block)
    if fixed_structure is None:
        raise RuntimeError("structure-kernel compilation produced no blocks")
    occupancy_quadratic = np.concatenate(occupancy_quadratic_blocks, axis=0)
    q_radial_squared = np.concatenate(q_radial_squared_blocks)
    q_normal_squared = np.concatenate(q_normal_squared_blocks)
    return OrderedIntensityDatasetResponse(
        dataset_id=next(iter(dataset_ids)),
        incidence_angle_rad=next(iter(incidences)),
        identities=tuple(item.identity for item in frozen),
        rods=rods,
        term_observation_index=term_observation,
        term_rod_index=term_rod,
        term_L=term_l,
        term_fixed_mass_per_strength=term_fixed,
        term_root_sign=term_root_sign,
        term_occupancy_quadratic_strength_A2=occupancy_quadratic,
        term_q_radial_squared_Ainv2=q_radial_squared,
        term_q_normal_squared_Ainv2=q_normal_squared,
        normalization_mass_px2=normalization,
        reciprocal_basis_Ainv=detector.coating.bragg_space.config.reciprocal_basis_Ainv,
        k_norm_Ainv=detector.coating.bragg_space.config.k_norm_Ainv,
        fixed_structure_parameters=fixed_structure,
        rod_catalog_revision=rod_catalog_revision,
        structure_model_revision=ordered_intensity_structure_model_revision(strength),
        mosaic_model_revision=_mosaic_parameters_revision(
            detector.coating.bragg_space.config.mosaic
        ),
        angle_frame_revision=angle_frame.revision,
        source_revision=detector.incident.states.source_revision,
        sample_geometry_revision=detector.incident.states.sample_geometry_revision,
        material_revision=detector.incident.states.material_revision,
        observable_revision=_ordered_intensity_observable_revision(
            frozen,
            angle_frame_revision=angle_frame.revision,
        ),
        excluded_phi_bin_indices=tuple(item.excluded_phi_bin_indices for item in frozen),
        topology_probe_revision=ORDERED_INTENSITY_TOPOLOGY_PROBE_REVISION,
    )


def _profile_q_radial_squared(
    strength: Bi2X3FiniteStackStrength,
    definitions: tuple[MosaicProfileDefinition, ...],
) -> FloatArray:
    layer_normal = np.cross(
        strength.crystal.direct_basis_A[:, 0],
        strength.crystal.direct_basis_A[:, 1],
    )
    layer_normal /= np.linalg.norm(layer_normal)
    if np.dot(layer_normal, strength.crystal.direct_basis_A[:, 2]) < 0.0:
        layer_normal = -layer_normal
    q_radial_squared = np.empty(len(definitions), dtype=np.float64)
    for profile_index, definition in enumerate(definitions):
        member_hk = definition.identity.group_key.member_rod_hk
        hkl = np.asarray(
            [(h, k, 0.0) for h, k in member_hk],
            dtype=np.float64,
        )
        q_crystal = hkl @ strength.reciprocal_basis_Ainv.T
        q_normal = q_crystal @ layer_normal
        q_radial = q_crystal - q_normal[:, None] * layer_normal[None, :]
        member_q_radial_squared = np.einsum("ij,ij->i", q_radial, q_radial, optimize=True)
        scale = max(float(np.max(member_q_radial_squared)), 1.0)
        tolerance = 1024.0 * np.finfo(np.float64).eps * scale
        if np.ptp(member_q_radial_squared) > tolerance:
            raise ValueError("profile rod members do not share one radial damping coordinate")
        q_radial_squared[profile_index] = float(member_q_radial_squared[0])
    q_radial_squared[np.abs(q_radial_squared) <= 1024.0 * np.finfo(np.float64).eps] = 0.0
    q_radial_squared.setflags(write=False)
    return q_radial_squared


def _occupancy_quadratic_matrices(basis: FloatArray) -> FloatArray:
    """Return symmetric matrices for ``basis @ (b², s1², s2², b*s1, ...)``."""

    matrices = np.zeros((*basis.shape[:-1], 3, 3), dtype=np.float64)
    matrices[..., 0, 0] = basis[..., 0]
    matrices[..., 1, 1] = basis[..., 1]
    matrices[..., 2, 2] = basis[..., 2]
    matrices[..., 0, 1] = matrices[..., 1, 0] = 0.5 * basis[..., 3]
    matrices[..., 0, 2] = matrices[..., 2, 0] = 0.5 * basis[..., 4]
    matrices[..., 1, 2] = matrices[..., 2, 1] = 0.5 * basis[..., 5]
    return matrices


def compile_source_averaged_ordered_intensity_response(
    detector: SourceAveragedDetectorEwaldMeasure,
    *,
    angle_frame: AngleFrame,
    definitions: tuple[MosaicProfileDefinition, ...],
    execution_backend: str = "cpu",
) -> SourceAveragedOrderedIntensityDatasetResponse:
    """Compile a Chebyshev peak-center response from one combined source detector function."""

    if not isinstance(detector, SourceAveragedDetectorEwaldMeasure):
        raise TypeError("detector must be SourceAveragedDetectorEwaldMeasure")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be cpu or cuda")
    frozen = tuple(definitions)
    if not frozen or any(not isinstance(item, MosaicProfileDefinition) for item in frozen):
        raise ValueError("definitions must contain MosaicProfileDefinition values")
    if len({item.identity for item in frozen}) != len(frozen):
        raise ValueError("definitions must have unique identities")
    dataset_ids = {item.identity.dataset_id for item in frozen}
    incidences = {item.identity.incidence_angle_rad for item in frozen}
    if len(dataset_ids) != 1 or len(incidences) != 1:
        raise ValueError("one response must contain exactly one dataset and incidence")
    if {item.identity.group_key.rod_catalog_revision for item in frozen} != {
        detector.rod_catalog_revision
    }:
        raise ValueError("profile definitions do not match the detector rod catalog revision")
    configured_rods = {(rod.h, rod.k): rod for rod in detector.rods}
    strength = detector.strength_model
    if not isinstance(strength, Bi2X3FiniteStackStrength):
        raise TypeError("ordered-intensity fitting currently requires Bi2X3FiniteStackStrength")
    fixed = strength.structure_parameters
    if not isinstance(fixed, Bi2X3QuintupleLayerParameters):
        raise TypeError("layered-quintuple strength requires resolved structure parameters")
    q_radial_squared = _profile_q_radial_squared(strength, frozen)
    occupancy_probe = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
    )
    scaled_u_nodes = np.sort(
        np.cos(
            np.pi
            * np.arange(_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, dtype=np.float64)
            / (_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT - 1)
        )
    )
    u_normal_nodes = 0.5 * _POINT_RESPONSE_U_NORMAL_MAX_A2 * (scaled_u_nodes + 1.0)
    u_normal_nodes[0] = 0.0
    u_normal_nodes[-1] = _POINT_RESPONSE_U_NORMAL_MAX_A2
    validation_u_normal_nodes = np.sort(
        0.5
        * _POINT_RESPONSE_U_NORMAL_MAX_A2
        * (
            1.0
            + np.cos(
                np.pi
                * (np.arange(_POINT_RESPONSE_VALIDATION_NODE_COUNT, dtype=np.float64) + 0.5)
                / _POINT_RESPONSE_VALIDATION_NODE_COUNT
            )
        )
    )
    evaluated_u_normal_nodes = np.concatenate((u_normal_nodes, validation_u_normal_nodes))
    probe_signal = np.empty(
        (evaluated_u_normal_nodes.size, 6, len(frozen)),
        dtype=np.float64,
    )
    normalization = np.empty(len(frozen), dtype=np.float64)
    normalization.fill(np.nan)
    result_backend: str | None = None
    result_device: str | None = None
    profiles_by_rod_group: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for profile_index, definition in enumerate(frozen):
        profiles_by_rod_group.setdefault(
            definition.identity.group_key.member_rod_hk,
            [],
        ).append(profile_index)
    observable_revision = _ordered_intensity_point_observable_revision(
        frozen,
        angle_frame_revision=angle_frame.revision,
    )
    for group_hk, profile_indices in profiles_by_rod_group.items():
        try:
            group_rods = tuple(configured_rods[rod_hk] for rod_hk in group_hk)
        except KeyError as error:
            raise ValueError(f"profile references unconfigured rod {error.args[0]}") from error
        group_detector = detector.restrict_rods(group_rods)
        group_normalization: FloatArray | None = None
        group_two_theta = np.asarray(
            [frozen[index].center_two_theta_rad for index in profile_indices],
            dtype=np.float64,
        )
        group_phi = np.asarray(
            [frozen[index].center_phi_rad for index in profile_indices],
            dtype=np.float64,
        )
        for node_index, u_normal_A2 in enumerate(evaluated_u_normal_nodes):
            for probe_index, occupancies in enumerate(occupancy_probe):
                parameters = replace(
                    fixed,
                    bi_occupancy=occupancies[0],
                    se1_occupancy=occupancies[1],
                    se2_occupancy=occupancies[2],
                    u_radial_A2=0.0,
                    u_normal_A2=float(u_normal_A2),
                )
                evaluated = evaluate_continuous_per_rod_angle_signal(
                    group_detector.rebind_physics(
                        strength_model=replace(strength, structure_parameters=parameters)
                    ),
                    angle_frame=angle_frame,
                    two_theta_rad=group_two_theta,
                    phi_rad=group_phi,
                    execution_backend=execution_backend,
                )
                if evaluated.source_revision != detector.incident.states.source_revision:
                    raise RuntimeError("source-averaged response changed the source realization")
                if result_backend is None:
                    result_backend = evaluated.execution_backend
                    result_device = evaluated.execution_device
                elif (
                    evaluated.execution_backend != result_backend
                    or evaluated.execution_device != result_device
                ):
                    raise RuntimeError("source-averaged response changed execution backend")
                if not np.all(evaluated.valid) or np.any(evaluated.caustic):
                    raise FloatingPointError("a frozen peak center is invalid or caustic")
                current_normalization = evaluated.normalization_density_px2_per_rad2
                if group_normalization is None:
                    group_normalization = current_normalization
                elif not np.array_equal(current_normalization, group_normalization):
                    raise RuntimeError("structure probes changed detector-area normalization")
                probe_signal[node_index, probe_index, profile_indices] = np.sum(
                    evaluated.per_rod_signal_density_A2_per_rad2,
                    axis=1,
                    dtype=np.float64,
                )
        if group_normalization is None:
            raise RuntimeError("source-averaged structure probes produced no peak centers")
        normalization[profile_indices] = group_normalization
    if not np.all(np.isfinite(probe_signal)) or np.any(probe_signal < 0.0):
        raise FloatingPointError("source-averaged occupancy probes produced invalid signal")
    quadratic_at_nodes = np.stack(
        (
            probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 0],
            probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 1],
            probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 2],
            probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 3]
            - probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 0]
            - probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 1],
            probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 4]
            - probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 0]
            - probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 2],
            probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 5]
            - probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 1]
            - probe_signal[:_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, 2],
        ),
        axis=2,
    )
    fitted_coefficients = np.polynomial.chebyshev.chebfit(
        scaled_u_nodes,
        quadratic_at_nodes.reshape(_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT, -1),
        deg=_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT - 1,
    )
    coefficients = fitted_coefficients.T.reshape(
        len(frozen),
        6,
        _POINT_RESPONSE_CHEBYSHEV_NODE_COUNT,
    )
    validation_probe = probe_signal[_POINT_RESPONSE_CHEBYSHEV_NODE_COUNT:]
    validation_basis = np.stack(
        (
            validation_probe[:, 0],
            validation_probe[:, 1],
            validation_probe[:, 2],
            validation_probe[:, 3] - validation_probe[:, 0] - validation_probe[:, 1],
            validation_probe[:, 4] - validation_probe[:, 0] - validation_probe[:, 2],
            validation_probe[:, 5] - validation_probe[:, 1] - validation_probe[:, 2],
        ),
        axis=2,
    )
    coefficient_axis_first = np.moveaxis(coefficients, -1, 0)
    interpolated_validation_basis = np.stack(
        tuple(
            np.polynomial.chebyshev.chebval(
                2.0 * u_normal_A2 / _POINT_RESPONSE_U_NORMAL_MAX_A2 - 1.0,
                coefficient_axis_first,
            )
            for u_normal_A2 in validation_u_normal_nodes
        )
    )
    exact_quadratic = _occupancy_quadratic_matrices(validation_basis)
    interpolation_error_quadratic = _occupancy_quadratic_matrices(
        interpolated_validation_basis - validation_basis
    )
    exact_eigenvalue, exact_eigenvector = np.linalg.eigh(exact_quadratic)
    profile_scale = np.maximum(
        np.max(np.abs(exact_eigenvalue), axis=(0, 2)),
        np.finfo(np.float64).tiny,
    )
    positivity_tolerance = 4096.0 * np.finfo(np.float64).eps * profile_scale[None, :]
    if np.any(exact_eigenvalue[..., 0] < -positivity_tolerance):
        raise FloatingPointError("full-detector occupancy quadratic is not positive semidefinite")
    certified_eigenvalue = exact_eigenvalue + (
        SOURCE_AVERAGED_ORDERED_INTENSITY_SIGNAL_CERTIFICATE_RELATIVE_FLOOR
        * profile_scale[None, :, None]
    )
    if np.any(certified_eigenvalue <= 0.0):
        raise FloatingPointError("occupancy-signal certificate floor did not regularize extinction")
    inverse_square_root = np.einsum(
        "...ik,...k,...jk->...ij",
        exact_eigenvector,
        1.0 / np.sqrt(certified_eigenvalue),
        exact_eigenvector,
        optimize=True,
    )
    relative_error_quadratic = np.einsum(
        "...ik,...kl,...lj->...ij",
        inverse_square_root,
        interpolation_error_quadratic,
        inverse_square_root,
        optimize=True,
    )
    relative_error_quadratic = 0.5 * (
        relative_error_quadratic + np.swapaxes(relative_error_quadratic, -1, -2)
    )
    maximum_interpolation_relative_error = float(
        np.max(np.abs(np.linalg.eigvalsh(relative_error_quadratic)))
    )
    if maximum_interpolation_relative_error > (
        _POINT_RESPONSE_MAXIMUM_INTERPOLATION_RELATIVE_ERROR
    ):
        raise FloatingPointError(
            "peak-center Chebyshev interpolation did not meet its certified tolerance"
        )
    if result_backend is None:
        raise RuntimeError("source-averaged response produced no detector evaluations")
    return SourceAveragedOrderedIntensityDatasetResponse(
        dataset_id=next(iter(dataset_ids)),
        incidence_angle_rad=next(iter(incidences)),
        identities=tuple(item.identity for item in frozen),
        occupancy_quadratic_chebyshev_signal_density_A2_per_rad2=coefficients,
        q_radial_squared_Ainv2=q_radial_squared,
        u_normal_nodes_A2=u_normal_nodes,
        center_two_theta_rad=np.asarray(
            [item.center_two_theta_rad for item in frozen],
            dtype=np.float64,
        ),
        center_phi_rad=np.asarray([item.center_phi_rad for item in frozen], dtype=np.float64),
        normalization_density_px2_per_rad2=normalization,
        reciprocal_basis_Ainv=strength.reciprocal_basis_Ainv,
        fixed_structure_parameters=fixed,
        rod_catalog_revision=detector.rod_catalog_revision,
        structure_model_revision=ordered_intensity_structure_model_revision(strength),
        mosaic_model_revision=_mosaic_parameters_revision(detector.mosaic),
        angle_frame_revision=angle_frame.revision,
        source_revision=detector.incident.states.source_revision,
        source_state_count=int(detector.incident.states.incident_state_id.size),
        sample_geometry_revision=detector.incident.states.sample_geometry_revision,
        material_revision=detector.incident.states.material_revision,
        interpolation_validation_maximum_relative_error=(maximum_interpolation_relative_error),
        interpolation_validation_node_count=_POINT_RESPONSE_VALIDATION_NODE_COUNT,
        instrument_revision=source_averaged_detector_instrument_revision(detector),
        execution_backend=result_backend,
        execution_device=result_device,
        observable_revision=observable_revision,
    )


def evaluate_source_averaged_ordered_intensity_point_signal(
    detector: SourceAveragedDetectorEwaldMeasure,
    *,
    angle_frame: AngleFrame,
    definitions: tuple[MosaicProfileDefinition, ...],
    structure_parameters: Bi2X3QuintupleLayerParameters,
    execution_backend: str = "cpu",
) -> FloatArray:
    """Evaluate selected-group peak-center signals from a fresh combined detector oracle."""

    if not isinstance(detector, SourceAveragedDetectorEwaldMeasure):
        raise TypeError("detector must be SourceAveragedDetectorEwaldMeasure")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    if not isinstance(structure_parameters, Bi2X3QuintupleLayerParameters):
        raise TypeError("structure_parameters must be Bi2X3QuintupleLayerParameters")
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be cpu or cuda")
    frozen = tuple(definitions)
    if not frozen or any(not isinstance(item, MosaicProfileDefinition) for item in frozen):
        raise ValueError("definitions must contain MosaicProfileDefinition values")
    if len({item.identity for item in frozen}) != len(frozen):
        raise ValueError("definitions must have unique identities")
    dataset_ids = {item.identity.dataset_id for item in frozen}
    incidences = {item.identity.incidence_angle_rad for item in frozen}
    if len(dataset_ids) != 1 or len(incidences) != 1:
        raise ValueError("one oracle call must contain exactly one dataset and incidence")
    if {item.identity.group_key.rod_catalog_revision for item in frozen} != {
        detector.rod_catalog_revision
    }:
        raise ValueError("definitions do not match the detector rod catalog revision")
    configured_rods = {(rod.h, rod.k): rod for rod in detector.rods}
    strength = detector.strength_model
    if not isinstance(strength, Bi2X3FiniteStackStrength):
        raise TypeError("ordered-intensity fitting currently requires Bi2X3FiniteStackStrength")
    fixed = strength.structure_parameters
    if (
        structure_parameters.bi_fractional_z != fixed.bi_fractional_z
        or structure_parameters.se2_fractional_z != fixed.se2_fractional_z
    ):
        raise ValueError("atomic positions differ from the fixed point response")
    candidate_strength = replace(strength, structure_parameters=structure_parameters)
    signal = np.empty(len(frozen), dtype=np.float64)
    profiles_by_rod_group: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for profile_index, definition in enumerate(frozen):
        profiles_by_rod_group.setdefault(
            definition.identity.group_key.member_rod_hk,
            [],
        ).append(profile_index)
    for group_hk, profile_indices in profiles_by_rod_group.items():
        try:
            group_rods = tuple(configured_rods[rod_hk] for rod_hk in group_hk)
        except KeyError as error:
            raise ValueError(f"profile references unconfigured rod {error.args[0]}") from error
        evaluated = evaluate_continuous_per_rod_angle_signal(
            detector.restrict_rods(group_rods).rebind_physics(strength_model=candidate_strength),
            angle_frame=angle_frame,
            two_theta_rad=np.asarray(
                [frozen[index].center_two_theta_rad for index in profile_indices],
                dtype=np.float64,
            ),
            phi_rad=np.asarray(
                [frozen[index].center_phi_rad for index in profile_indices],
                dtype=np.float64,
            ),
            execution_backend=execution_backend,
        )
        if not np.all(evaluated.valid) or np.any(evaluated.caustic):
            raise FloatingPointError("a frozen peak center is invalid or caustic")
        signal[profile_indices] = np.sum(
            evaluated.per_rod_signal_density_A2_per_rad2,
            axis=1,
            dtype=np.float64,
        )
    if not np.all(np.isfinite(signal)) or np.any(signal < 0.0):
        raise FloatingPointError("source-averaged ordered-intensity signal is invalid")
    signal.setflags(write=False)
    return signal


@dataclass(frozen=True, slots=True)
class OrderedIntensityObservations:
    """Positive selected-component ROI masses bound to one frozen observable layout."""

    dataset_id: str
    observable_revision: str
    mass_A2: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be nonempty")
        if not isinstance(self.observable_revision, str) or not self.observable_revision:
            raise ValueError("observable_revision must be nonempty")
        mass = np.array(self.mass_A2, dtype=np.float64, copy=True, order="C")
        if mass.ndim != 1 or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("mass_A2 must be a positive finite vector")
        mass.setflags(write=False)
        object.__setattr__(self, "mass_A2", mass)


@dataclass(frozen=True, slots=True)
class OrderedIntensityPeakCenterObservations:
    """Positive selected-group angular signal densities at frozen peak centers."""

    dataset_id: str
    observable_revision: str
    signal_density_A2_per_rad2: FloatArray
    measure_id: str = "selected_group_angular_signal_density_A2_per_rad2.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be nonempty")
        if not isinstance(self.observable_revision, str) or not self.observable_revision:
            raise ValueError("observable_revision must be nonempty")
        signal = np.array(
            self.signal_density_A2_per_rad2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if signal.ndim != 1 or not np.all(np.isfinite(signal)) or np.any(signal <= 0.0):
            raise ValueError("signal_density_A2_per_rad2 must be a positive finite vector")
        if self.measure_id != "selected_group_angular_signal_density_A2_per_rad2.v1":
            raise ValueError("unsupported peak-center ordered-intensity measure")
        signal.setflags(write=False)
        object.__setattr__(self, "signal_density_A2_per_rad2", signal)


@dataclass(frozen=True, slots=True)
class OrderedIntensityFitResult:
    """Joint result with explicit ratio gauge and an admissible structure representative.

    In absolute mode the representative is the fitted physical structure. In relative-scale mode
    occupancy coordinates are ratios to the frozen reference; the representative rescales all
    ratios so their maximum is one. Frozen occupancy names therefore refer to ratio coordinates in
    relative mode, not to numeric fields of the representative.
    """

    structure_representative: Bi2X3QuintupleLayerParameters
    active_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    relative_scale_mode: bool
    occupancy_ratio_reference: str | None
    occupancy_ratios: FloatArray
    dataset_ids: tuple[str, ...]
    dataset_scales: FloatArray
    predicted_mass_A2: tuple[FloatArray, ...]
    relative_residual: FloatArray
    objective: float
    sensitivity_singular_values: FloatArray
    sensitivity_rank: int
    sensitivity_condition: float
    parameter_correlation: FloatArray
    active_bounds: BoolArray
    function_evaluations: int
    predicted_signal_density_A2_per_rad2: tuple[FloatArray, ...] = ()
    observable_measure_id: str = "selected_group_angle_roi_mass_A2.v1"

    def __post_init__(self) -> None:
        if self.observable_measure_id == "selected_group_angle_roi_mass_A2.v1":
            predictions = self.predicted_mass_A2
            if self.predicted_signal_density_A2_per_rad2:
                raise ValueError("ROI-mass fits cannot contain peak-center predictions")
        elif self.observable_measure_id == "selected_group_angular_signal_density_A2_per_rad2.v1":
            predictions = self.predicted_signal_density_A2_per_rad2
            if self.predicted_mass_A2:
                raise ValueError("peak-center fits cannot contain ROI-mass predictions")
        else:
            raise ValueError("unsupported ordered-intensity fit observable measure")
        if (
            len(self.dataset_ids) != len(predictions)
            or self.dataset_scales.shape != (len(self.dataset_ids),)
            or any(value.ndim != 1 or not np.all(np.isfinite(value)) for value in predictions)
        ):
            raise ValueError("fit predictions and dataset scales are misaligned")


def _parameter_vector(
    parameters: Bi2X3QuintupleLayerParameters,
    baseline: Bi2X3QuintupleLayerParameters,
) -> FloatArray:
    return np.asarray(
        (
            parameters.bi_fractional_z - baseline.bi_fractional_z,
            parameters.se2_fractional_z - baseline.se2_fractional_z,
            parameters.bi_occupancy,
            parameters.se1_occupancy,
            parameters.se2_occupancy,
            parameters.u_radial_A2,
            parameters.u_normal_A2,
        ),
        dtype=np.float64,
    )


def _structure_from_vector(
    vector: FloatArray,
    baseline: Bi2X3QuintupleLayerParameters,
) -> Bi2X3QuintupleLayerParameters:
    return Bi2X3QuintupleLayerParameters(
        bi_fractional_z=baseline.bi_fractional_z + float(vector[0]),
        se2_fractional_z=baseline.se2_fractional_z + float(vector[1]),
        bi_occupancy=float(vector[2]),
        se1_occupancy=float(vector[3]),
        se2_occupancy=float(vector[4]),
        u_radial_A2=float(vector[5]),
        u_normal_A2=float(vector[6]),
    )


def _profile_scales_and_residual(
    predictions: tuple[FloatArray, ...],
    observations: tuple[FloatArray, ...],
    *,
    relative_scale_mode: bool,
) -> tuple[FloatArray, FloatArray]:
    scales = np.ones(len(predictions), dtype=np.float64)
    residuals: list[FloatArray] = []
    for dataset_index, (predicted, observed) in enumerate(
        zip(predictions, observations, strict=True)
    ):
        if relative_scale_mode:
            ratio = predicted / observed
            denominator = float(ratio @ ratio)
            if denominator <= np.finfo(np.float64).tiny:
                raise FloatingPointError("candidate structure has zero predicted observable")
            scales[dataset_index] = max(0.0, float(np.sum(ratio)) / denominator)
            residuals.append(scales[dataset_index] * ratio - 1.0)
        else:
            residuals.append(predicted / observed - 1.0)
    return scales, np.concatenate(residuals)


def fit_ordered_intensity_series(
    responses: tuple[
        OrderedIntensityDatasetResponse | SourceAveragedOrderedIntensityDatasetResponse,
        ...,
    ],
    observations: tuple[
        OrderedIntensityObservations | OrderedIntensityPeakCenterObservations,
        ...,
    ],
    *,
    base_strength: Bi2X3FiniteStackStrength,
    active_parameter_names: tuple[str, ...],
    initial_parameters: Bi2X3QuintupleLayerParameters | None = None,
    relative_scale_mode: bool = True,
    active_parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    maximum_function_evaluations: int = 400,
    required_source_state_count: int | None = None,
    required_source_revision: str | None = None,
) -> OrderedIntensityFitResult:
    """Fit a declared occupancy/U subset with atomic positions held fixed.

    ``active_parameter_bounds`` optionally narrows the canonical physical bounds for named active
    coordinates. Omitted coordinates retain their default bounds.
    """

    datasets = tuple(responses)
    if (
        not datasets
        or any(
            not isinstance(
                item,
                (OrderedIntensityDatasetResponse, SourceAveragedOrderedIntensityDatasetResponse),
            )
            for item in datasets
        )
        or len({item.dataset_id for item in datasets}) != len(datasets)
    ):
        raise ValueError("responses must contain unique ordered-intensity datasets")
    if not isinstance(base_strength, Bi2X3FiniteStackStrength):
        raise TypeError("base_strength must be Bi2X3FiniteStackStrength")
    expected_structure_revision = ordered_intensity_structure_model_revision(base_strength)
    if any(
        response.structure_model_revision != expected_structure_revision for response in datasets
    ):
        raise ValueError("responses and base_strength do not share one fixed structure model")
    if len({response.mosaic_model_revision for response in datasets}) != 1:
        raise ValueError("simultaneous responses must share one frozen physical mosaic model")
    response_measure_ids = {
        (
            response.measure_id
            if isinstance(response, SourceAveragedOrderedIntensityDatasetResponse)
            else "selected_group_angle_roi_mass_A2.v1"
        )
        for response in datasets
    }
    if len(response_measure_ids) != 1:
        raise ValueError("simultaneous responses must share one observable measure")
    observable_measure_id = next(iter(response_measure_ids))
    source_averaged = tuple(
        response
        for response in datasets
        if isinstance(response, SourceAveragedOrderedIntensityDatasetResponse)
    )
    if source_averaged and (
        len({response.execution_backend for response in source_averaged}) != 1
        or len({response.execution_device for response in source_averaged}) != 1
    ):
        raise ValueError("simultaneous source-averaged responses changed execution backend")
    if required_source_state_count is not None:
        if (
            isinstance(required_source_state_count, bool)
            or not isinstance(required_source_state_count, (int, np.integer))
            or int(required_source_state_count) < 1
        ):
            raise ValueError("required_source_state_count must be a positive integer")
        if len(source_averaged) != len(datasets) or any(
            response.source_state_count != int(required_source_state_count)
            for response in source_averaged
        ):
            raise ValueError("a response does not match the required source-state count")
    if required_source_revision is not None:
        if not isinstance(required_source_revision, str) or not required_source_revision:
            raise ValueError("required_source_revision must be a nonempty string")
        if len(source_averaged) != len(datasets) or any(
            response.source_revision != required_source_revision for response in source_averaged
        ):
            raise ValueError("a response does not match the required source revision")
    reciprocal_scale = max(float(np.linalg.norm(base_strength.reciprocal_basis_Ainv)), 1.0)
    if any(
        not np.allclose(
            response.reciprocal_basis_Ainv,
            base_strength.reciprocal_basis_Ainv,
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * reciprocal_scale,
        )
        for response in datasets
    ):
        raise ValueError("a response changed the fixed reciprocal lattice")
    active = tuple(active_parameter_names)
    if len(set(active)) != len(active) or any(
        name not in STRUCTURE_FACTOR_PARAMETER_NAMES for name in active
    ):
        raise ValueError("active_parameter_names must be unique canonical parameter names")
    active = tuple(name for name in STRUCTURE_FACTOR_PARAMETER_NAMES if name in active)
    if any(name in active for name in _POSITION_PARAMETER_NAMES):
        raise ValueError(
            "atomic positions are fixed in the compiled occupancy/Debye-Waller response"
        )
    if relative_scale_mode and all(name in active for name in _OCCUPANCY_PARAMETER_NAMES):
        raise OrderedIntensityIdentifiabilityError(
            "three active occupancies and free image scales have an exact common-scale gauge"
        )
    if (
        isinstance(maximum_function_evaluations, bool)
        or not isinstance(maximum_function_evaluations, (int, np.integer))
        or int(maximum_function_evaluations) < 1
    ):
        raise ValueError("maximum_function_evaluations must be a positive integer")
    records = tuple(observations)
    if any(
        not isinstance(
            item,
            (OrderedIntensityObservations, OrderedIntensityPeakCenterObservations),
        )
        for item in records
    ):
        raise TypeError("observations contain an unsupported ordered-intensity measure")
    observation_by_dataset = {item.dataset_id: item for item in records}
    dataset_ids = {response.dataset_id for response in datasets}
    if len(observation_by_dataset) != len(records) or set(observation_by_dataset) != dataset_ids:
        raise ValueError("observations must match the response dataset IDs exactly")
    frozen_observations_list: list[FloatArray] = []
    for response in datasets:
        record = observation_by_dataset[response.dataset_id]
        if record.observable_revision != response.observable_revision:
            raise ValueError(
                f"observation {record.dataset_id!r} does not match its frozen observable revision"
            )
        if observable_measure_id == "selected_group_angle_roi_mass_A2.v1":
            if not isinstance(record, OrderedIntensityObservations):
                raise ValueError("ROI-mass responses require ROI-mass observations")
            values = record.mass_A2
        else:
            if not isinstance(record, OrderedIntensityPeakCenterObservations):
                raise ValueError("peak-center responses require peak-center observations")
            if record.measure_id != observable_measure_id:
                raise ValueError("peak-center observation measure does not match its response")
            values = record.signal_density_A2_per_rad2
        if values.shape != (len(response.identities),):
            raise ValueError("observation shape does not match its compiled response")
        frozen_observations_list.append(values)
    frozen_observations = tuple(frozen_observations_list)

    baseline = Bi2X3QuintupleLayerParameters.from_crystal(base_strength.crystal)
    initial = (
        base_strength.structure_parameters if initial_parameters is None else initial_parameters
    )
    if not isinstance(initial, Bi2X3QuintupleLayerParameters):
        raise TypeError("initial_parameters must be Bi2X3QuintupleLayerParameters")
    for response in datasets:
        fixed = response.fixed_structure_parameters
        if (
            fixed.bi_fractional_z != initial.bi_fractional_z
            or fixed.se2_fractional_z != initial.se2_fractional_z
        ):
            raise ValueError("initial atomic positions differ from a compiled response")
    initial_vector = _parameter_vector(initial, baseline)
    lower = np.asarray((-0.02, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0))
    upper = np.asarray((0.02, 0.02, 1.0, 1.0, 1.0, 0.1, 0.1))
    parameter_scale = np.asarray((0.04, 0.04, 1.0, 1.0, 1.0, 0.1, 0.1))
    active_index = np.asarray(
        [STRUCTURE_FACTOR_PARAMETER_NAMES.index(name) for name in active],
        dtype=np.int64,
    )
    occupancy_ratio_reference: str | None = None
    occupancy_reference_index: int | None = None
    if relative_scale_mode:
        inactive_occupancy = tuple(
            (name, STRUCTURE_FACTOR_PARAMETER_NAMES.index(name))
            for name in _OCCUPANCY_PARAMETER_NAMES
            if name not in active
        )
        positive_reference = tuple(
            (name, index) for name, index in inactive_occupancy if initial_vector[index] > 0.0
        )
        if not positive_reference:
            raise OrderedIntensityIdentifiabilityError(
                "relative image scales require one positive frozen occupancy as the common-scale gauge"
            )
        occupancy_ratio_reference, occupancy_reference_index = max(
            positive_reference,
            key=lambda item: initial_vector[item[1]],
        )
        reference_value = initial_vector[occupancy_reference_index]
        initial_vector[2:5] /= reference_value
        if not np.all(np.isfinite(initial_vector[2:5])):
            raise ValueError("initial occupancy ratios are not finite")
        upper[2:5] = np.inf
    if active_parameter_bounds is not None:
        if not isinstance(active_parameter_bounds, Mapping):
            raise TypeError("active_parameter_bounds must be a mapping")
        inactive_bound_names = set(active_parameter_bounds) - set(active)
        if inactive_bound_names:
            raise ValueError(
                "bounds may only be declared for an active parameter: "
                f"{sorted(inactive_bound_names)[0]!r}"
            )
        for name, raw_bounds in active_parameter_bounds.items():
            if not isinstance(raw_bounds, (tuple, list)) or len(raw_bounds) != 2:
                raise ValueError(f"active parameter bounds for {name!r} must contain two values")
            lower_value, upper_value = (float(value) for value in raw_bounds)
            if (
                not math.isfinite(lower_value)
                or not math.isfinite(upper_value)
                or lower_value >= upper_value
            ):
                raise ValueError(
                    f"active parameter bounds for {name!r} must be finite and increasing"
                )
            parameter_index = STRUCTURE_FACTOR_PARAMETER_NAMES.index(name)
            if lower_value < lower[parameter_index] or upper_value > upper[parameter_index]:
                raise ValueError(
                    f"active parameter bounds for {name!r} must narrow the canonical bounds"
                )
            lower[parameter_index] = lower_value
            upper[parameter_index] = upper_value
    if np.any(initial_vector[active_index] < lower[active_index]) or np.any(
        initial_vector[active_index] > upper[active_index]
    ):
        raise ValueError("initial structure parameters lie outside the accepted local bounds")

    def evaluated(
        active_value: FloatArray,
    ) -> tuple[
        FloatArray,
        tuple[FloatArray, ...],
        FloatArray,
        FloatArray,
    ]:
        full = initial_vector.copy()
        full[active_index] = active_value
        if relative_scale_mode:
            predictions = tuple(
                (
                    response.predict_signal_density_coordinates_A2_per_rad2(
                        occupancy_coefficients=full[2:5],
                        u_radial_A2=float(full[5]),
                        u_normal_A2=float(full[6]),
                    )
                    if isinstance(response, SourceAveragedOrderedIntensityDatasetResponse)
                    else response.predict_mass_coordinates_A2(
                        occupancy_coefficients=full[2:5],
                        u_radial_A2=float(full[5]),
                        u_normal_A2=float(full[6]),
                    )
                )
                for response in datasets
            )
        else:
            structure = _structure_from_vector(full, baseline)
            predictions = tuple(
                (
                    response.predict_signal_density_A2_per_rad2(structure)
                    if isinstance(response, SourceAveragedOrderedIntensityDatasetResponse)
                    else response.predict_mass_A2(structure)
                )
                for response in datasets
            )
        scales, residual = _profile_scales_and_residual(
            predictions,
            frozen_observations,
            relative_scale_mode=relative_scale_mode,
        )
        return full, predictions, scales, residual

    if active:
        optimization = least_squares(
            lambda value: evaluated(value)[3],
            initial_vector[active_index],
            bounds=(lower[active_index], upper[active_index]),
            x_scale=parameter_scale[active_index],
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=int(maximum_function_evaluations),
        )
        if not optimization.success:
            raise RuntimeError(f"ordered-intensity optimization failed: {optimization.message}")
        terminal_vector, predictions, scales, residual = evaluated(optimization.x)
        natural_scale = parameter_scale[active_index]
        scaled_jacobian = np.asarray(optimization.jac, dtype=np.float64) * natural_scale[None, :]
        _, singular, right_singular = np.linalg.svd(
            scaled_jacobian,
            full_matrices=False,
        )
        tolerance = np.sqrt(np.finfo(np.float64).eps) * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        if rank != len(active):
            raise OrderedIntensityIdentifiabilityError(
                f"active structure sensitivity has rank {rank}/{len(active)}"
            )
        condition = float(singular[0] / singular[-1])
        covariance = (right_singular.T * (1.0 / (singular * singular))[None, :]) @ (right_singular)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        correlation = covariance / np.outer(standard_deviation, standard_deviation)
        if not np.all(np.isfinite(correlation)):
            raise FloatingPointError("active structure correlation is not finite")
        function_evaluations = int(optimization.nfev)
        bound_tolerance = 1.0e-8 * parameter_scale[active_index]
        active_bounds = (np.abs(optimization.x - lower[active_index]) <= bound_tolerance) | (
            np.isfinite(upper[active_index])
            & (np.abs(optimization.x - upper[active_index]) <= bound_tolerance)
        )
    else:
        terminal_vector, predictions, scales, residual = evaluated(np.empty(0, dtype=np.float64))
        singular = np.empty(0, dtype=np.float64)
        rank = 0
        condition = 1.0
        correlation = np.empty((0, 0), dtype=np.float64)
        active_bounds = np.empty(0, dtype=np.bool_)
        function_evaluations = 1

    if relative_scale_mode:
        if occupancy_reference_index is None or occupancy_ratio_reference is None:
            raise RuntimeError("relative occupancy gauge was not resolved")
        occupancy_ratios = terminal_vector[2:5] / terminal_vector[occupancy_reference_index]
        representative_scale = float(np.max(occupancy_ratios))
        if not math.isfinite(representative_scale) or representative_scale <= 0.0:
            raise FloatingPointError("relative occupancy ratios have no positive representative")
        representative_vector = terminal_vector.copy()
        representative_vector[2:5] = occupancy_ratios / representative_scale
        structure = _structure_from_vector(representative_vector, baseline)
        intensity_scale = representative_scale * representative_scale
        predictions = tuple(predicted / intensity_scale for predicted in predictions)
        scales = scales * intensity_scale
    else:
        structure = _structure_from_vector(terminal_vector, baseline)
        occupancy_ratios = np.empty(0, dtype=np.float64)

    frozen = tuple(name for name in STRUCTURE_FACTOR_PARAMETER_NAMES if name not in active)
    for value in (
        *predictions,
        occupancy_ratios,
        scales,
        residual,
        singular,
        correlation,
        active_bounds,
    ):
        value.setflags(write=False)
    return OrderedIntensityFitResult(
        structure_representative=structure,
        active_parameter_names=active,
        frozen_parameter_names=frozen,
        relative_scale_mode=bool(relative_scale_mode),
        occupancy_ratio_reference=occupancy_ratio_reference,
        occupancy_ratios=occupancy_ratios,
        dataset_ids=tuple(response.dataset_id for response in datasets),
        dataset_scales=scales,
        predicted_mass_A2=(
            predictions if observable_measure_id == "selected_group_angle_roi_mass_A2.v1" else ()
        ),
        relative_residual=residual,
        objective=float(residual @ residual),
        sensitivity_singular_values=singular,
        sensitivity_rank=rank,
        sensitivity_condition=condition,
        parameter_correlation=correlation,
        active_bounds=active_bounds,
        function_evaluations=function_evaluations,
        predicted_signal_density_A2_per_rad2=(
            predictions
            if observable_measure_id == "selected_group_angular_signal_density_A2_per_rad2.v1"
            else ()
        ),
        observable_measure_id=observable_measure_id,
    )


__all__ = [
    "ORDERED_INTENSITY_TOPOLOGY_PROBE_REVISION",
    "STRUCTURE_FACTOR_PARAMETER_NAMES",
    "OrderedIntensityDatasetResponse",
    "OrderedIntensityFitResult",
    "OrderedIntensityIdentifiabilityError",
    "OrderedIntensityObservations",
    "OrderedIntensityPeakCenterObservations",
    "SourceAveragedOrderedIntensityDatasetResponse",
    "compile_ordered_intensity_response",
    "compile_source_averaged_ordered_intensity_response",
    "evaluate_source_averaged_ordered_intensity_point_signal",
    "fit_ordered_intensity_series",
    "ordered_intensity_profile_catalog_revision",
    "ordered_intensity_structure_model_revision",
    "probe_ordered_intensity_inverse_boundary_bins",
    "source_averaged_detector_instrument_revision",
]
