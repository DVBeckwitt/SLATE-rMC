"""Compiled point evaluator for the accepted finite parent-2H Bi2Se3 fixture."""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import numba
import numpy as np
import xraydb
from numpy.typing import ArrayLike, NDArray

from painted_ewald import MosaicParameters
from rasim_next.core.contracts import EventIntensityNormalization
from rasim_next.core.scattering import CLASSICAL_ELECTRON_RADIUS_A
from rasim_next.geometry.detector import _DETECTOR_INCIDENCE_COSINE_TOL
from rasim_next.materials.optics import HC_EV_A, _f0_species
from rasim_next.ordered.motifs import _parameterized_bi2se3_quintuple_layer
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class CompiledPixelIntegral(NamedTuple):
    """Private fused pixel result retained at pixel rather than node scale."""

    per_rod_mass_A2: FloatArray
    center_per_rod_density_A2_per_px2: FloatArray
    per_rod_inverse_count_min: IntArray
    per_rod_inverse_count_max: IntArray
    per_rod_caustic: BoolArray
    valid_any: BoolArray
    valid_all: BoolArray
    center_valid: BoolArray


class _ForwardDetectorState(NamedTuple):
    detector_shape_rc: tuple[int, int]
    detector_column_row_covectors_sample_per_m: FloatArray
    detector_normal_sample: FloatArray
    ray_origin_detector_column_row_px: FloatArray
    ray_origin_detector_normal_m: float
    ki_film_sample_Ainv: FloatArray
    internal_k_Ainv: float
    air_k0_Ainv: float
    refractive_index: complex
    entrance_power: float
    incident_decay_Ainv: float
    film_thickness_A: float
    source_phase_weight: float
    sample_from_local: FloatArray
    rod_hk_population: FloatArray
    rod_parallel_local_Ainv: FloatArray
    rod_u_bounds_Ainv: FloatArray
    rod_inverse_constants: FloatArray
    b3_norm_Ainv: float
    atom_fractional_offset: FloatArray
    atom_occupancy_element: FloatArray
    rod_atom_inplane_factor: NDArray[np.complex128]
    u_radial_A2: float
    u_normal_A2: float
    f0_parameters: FloatArray
    anomalous_factor_e: NDArray[np.complex128]
    layers: int
    shared_disorder_epsilon: float
    normalization_divisor: float


class _DetectorProjection(NamedTuple):
    """Batched ray-to-detector projection compiled in the sample frame."""

    detector_column_row_covectors_sample_per_m: FloatArray
    detector_normal_sample: FloatArray
    ray_origin_detector_column_row_px: FloatArray
    ray_origin_detector_normal_m: FloatArray


@dataclass(frozen=True, slots=True)
class CompiledDetectorState:
    """Immutable numeric state consumed by the no-GIL point kernel."""

    detector_zero_lab_m: FloatArray
    detector_column_step_lab_m: FloatArray
    detector_row_step_lab_m: FloatArray
    detector_pixel_area_vector_lab_m2: FloatArray
    ray_origin_lab_m: FloatArray
    sample_from_lab: FloatArray
    ki_film_sample_Ainv: FloatArray
    internal_k_Ainv: float
    air_k0_Ainv: float
    refractive_index: complex
    entrance_amplitude: complex
    incident_decay_Ainv: float
    film_thickness_A: float
    source_phase_weight: float
    sample_from_local: FloatArray
    rod_hk_population: FloatArray
    rod_parallel_local_Ainv: FloatArray
    rod_u_bounds_Ainv: FloatArray
    rod_inverse_constants: FloatArray
    b3_norm_Ainv: float
    gaussian_sigma_rad: float
    lorentzian_hwhm_rad: float
    lorentzian_probability: float
    atom_fractional_offset: FloatArray
    atom_occupancy_element: FloatArray
    rod_atom_inplane_factor: NDArray[np.complex128]
    u_radial_A2: float
    u_normal_A2: float
    f0_parameters: FloatArray
    anomalous_factor_e: NDArray[np.complex128]
    layers: int
    shared_disorder_epsilon: float
    normalization_divisor: float

    def __post_init__(self) -> None:
        rod_count = np.asarray(self.rod_hk_population).shape[0]
        atom_count = np.asarray(self.atom_fractional_offset).shape[0]
        shapes = {
            "detector_zero_lab_m": (3,),
            "detector_column_step_lab_m": (3,),
            "detector_row_step_lab_m": (3,),
            "detector_pixel_area_vector_lab_m2": (3,),
            "ray_origin_lab_m": (3,),
            "sample_from_lab": (3, 3),
            "ki_film_sample_Ainv": (3,),
            "sample_from_local": (3, 3),
            "rod_hk_population": (rod_count, 3),
            "rod_parallel_local_Ainv": (rod_count, 3),
            "rod_u_bounds_Ainv": (rod_count, 2),
            "rod_inverse_constants": (rod_count, 4),
            "atom_fractional_offset": (atom_count, 3),
            "atom_occupancy_element": (atom_count, 2),
            "f0_parameters": (2, 11),
        }
        if rod_count == 0 or atom_count == 0:
            raise ValueError("compiled detector state requires rods and motif atoms")
        for name, shape in shapes.items():
            value = np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        anomalous = np.array(self.anomalous_factor_e, dtype=np.complex128, copy=True, order="C")
        if anomalous.shape != (2,) or not np.all(np.isfinite(anomalous)):
            raise ValueError("anomalous_factor_e must be finite with shape (2,)")
        anomalous.setflags(write=False)
        object.__setattr__(self, "anomalous_factor_e", anomalous)
        inplane_factor = np.array(
            self.rod_atom_inplane_factor,
            dtype=np.complex128,
            copy=True,
            order="C",
        )
        if inplane_factor.shape != (rod_count, atom_count) or not np.all(
            np.isfinite(inplane_factor)
        ):
            raise ValueError(
                "rod_atom_inplane_factor must be finite with shape (rod_count, atom_count)"
            )
        inplane_factor.setflags(write=False)
        object.__setattr__(self, "rod_atom_inplane_factor", inplane_factor)
        scalar_names = (
            "internal_k_Ainv",
            "air_k0_Ainv",
            "incident_decay_Ainv",
            "film_thickness_A",
            "source_phase_weight",
            "b3_norm_Ainv",
            "gaussian_sigma_rad",
            "lorentzian_hwhm_rad",
            "lorentzian_probability",
            "u_radial_A2",
            "u_normal_A2",
            "shared_disorder_epsilon",
            "normalization_divisor",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.internal_k_Ainv == 0.0 or self.air_k0_Ainv == 0.0 or self.b3_norm_Ainv == 0.0:
            raise ValueError("compiled wavevector and reciprocal scales must be positive")
        if self.normalization_divisor == 0.0:
            raise ValueError("normalization_divisor must be positive")
        if not 0.0 <= self.lorentzian_probability <= 1.0:
            raise ValueError("lorentzian_probability must lie in [0, 1]")
        if not 0.0 <= self.shared_disorder_epsilon <= 1.0:
            raise ValueError("shared_disorder_epsilon must lie in [0, 1]")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        for name in ("refractive_index", "entrance_amplitude"):
            value = complex(getattr(self, name))
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def rebind_geometry(
        self,
        *,
        detector_zero_lab_m: ArrayLike,
        detector_column_step_lab_m: ArrayLike,
        detector_row_step_lab_m: ArrayLike,
        detector_pixel_area_vector_lab_m2: ArrayLike,
        ray_origin_lab_m: ArrayLike,
        sample_from_lab: ArrayLike,
        ki_film_sample_Ainv: ArrayLike,
        internal_k_Ainv: float,
        entrance_amplitude: complex,
        incident_decay_Ainv: float,
        source_phase_weight: float,
    ) -> CompiledDetectorState:
        """Replace dynamic geometry while sharing validated read-only physics arrays."""

        array_values = {
            "detector_zero_lab_m": (detector_zero_lab_m, (3,)),
            "detector_column_step_lab_m": (detector_column_step_lab_m, (3,)),
            "detector_row_step_lab_m": (detector_row_step_lab_m, (3,)),
            "detector_pixel_area_vector_lab_m2": (
                detector_pixel_area_vector_lab_m2,
                (3,),
            ),
            "ray_origin_lab_m": (ray_origin_lab_m, (3,)),
            "sample_from_lab": (sample_from_lab, (3, 3)),
            "ki_film_sample_Ainv": (ki_film_sample_Ainv, (3,)),
        }
        validated_arrays: dict[str, FloatArray] = {}
        for name, (supplied, shape) in array_values.items():
            value = np.array(supplied, dtype=np.float64, copy=True, order="C")
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            value.setflags(write=False)
            validated_arrays[name] = value
        scalar_values = {
            "internal_k_Ainv": float(internal_k_Ainv),
            "incident_decay_Ainv": float(incident_decay_Ainv),
            "source_phase_weight": float(source_phase_weight),
        }
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in scalar_values.values())
            or scalar_values["internal_k_Ainv"] == 0.0
        ):
            raise ValueError("rebound geometry scalars must be finite and physically nonnegative")
        entrance = complex(entrance_amplitude)
        if not math.isfinite(entrance.real) or not math.isfinite(entrance.imag):
            raise ValueError("entrance_amplitude must be finite")

        rebound = object.__new__(type(self))
        for descriptor in fields(self):
            object.__setattr__(rebound, descriptor.name, getattr(self, descriptor.name))
        for name, value in validated_arrays.items():
            object.__setattr__(rebound, name, value)
        for name, value in scalar_values.items():
            object.__setattr__(rebound, name, value)
        object.__setattr__(rebound, "entrance_amplitude", entrance)
        return rebound

    def _rebind_validated_geometry(
        self,
        *,
        detector_zero_lab_m: FloatArray,
        detector_column_step_lab_m: FloatArray,
        detector_row_step_lab_m: FloatArray,
        detector_pixel_area_vector_lab_m2: FloatArray,
        ray_origin_lab_m: FloatArray,
        sample_from_lab: FloatArray,
        ki_film_sample_Ainv: FloatArray,
        internal_k_Ainv: float,
        entrance_amplitude: complex,
        incident_decay_Ainv: float,
        source_phase_weight: float,
    ) -> CompiledDetectorState:
        """Reuse geometry arrays validated by the owning source-average model."""

        rebound = object.__new__(type(self))
        for descriptor in fields(self):
            object.__setattr__(rebound, descriptor.name, getattr(self, descriptor.name))
        replacements = {
            "detector_zero_lab_m": detector_zero_lab_m,
            "detector_column_step_lab_m": detector_column_step_lab_m,
            "detector_row_step_lab_m": detector_row_step_lab_m,
            "detector_pixel_area_vector_lab_m2": detector_pixel_area_vector_lab_m2,
            "ray_origin_lab_m": ray_origin_lab_m,
            "sample_from_lab": sample_from_lab,
            "ki_film_sample_Ainv": ki_film_sample_Ainv,
            "internal_k_Ainv": internal_k_Ainv,
            "entrance_amplitude": entrance_amplitude,
            "incident_decay_Ainv": incident_decay_Ainv,
            "source_phase_weight": source_phase_weight,
        }
        for name, value in replacements.items():
            object.__setattr__(rebound, name, value)
        return rebound

    def rebind_physics(
        self,
        *,
        mosaic: MosaicParameters,
        atom_fractional_offset: ArrayLike,
        atom_occupancy_element: ArrayLike,
        f0_parameters: ArrayLike,
        anomalous_factor_e: ArrayLike,
        layers: int,
        normalization_divisor: float,
        u_radial_A2: float,
        u_normal_A2: float,
        shared_disorder_epsilon: float,
    ) -> CompiledDetectorState:
        """Replace mosaic and fixed-geometry structure state without rebuilding geometry."""

        if not isinstance(mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        if mosaic.zero_tilt_probability_mass != 0.0:
            raise ValueError("compiled integration does not support zero-tilt atoms")
        offsets = np.array(atom_fractional_offset, dtype=np.float64, copy=True, order="C")
        properties = np.array(atom_occupancy_element, dtype=np.float64, copy=True, order="C")
        parameters = np.array(f0_parameters, dtype=np.float64, copy=True, order="C")
        anomalous = np.array(anomalous_factor_e, dtype=np.complex128, copy=True, order="C")
        if offsets.shape != self.atom_fractional_offset.shape or not np.all(np.isfinite(offsets)):
            raise ValueError("atom_fractional_offset changed shape or contains nonfinite values")
        if not np.array_equal(offsets[:, :2], self.atom_fractional_offset[:, :2]):
            raise ValueError("structure rebinding cannot change in-plane atomic coordinates")
        if properties.shape != self.atom_occupancy_element.shape or not np.all(
            np.isfinite(properties)
        ):
            raise ValueError("atom_occupancy_element changed shape or contains nonfinite values")
        if parameters.shape != self.f0_parameters.shape or not np.all(np.isfinite(parameters)):
            raise ValueError("f0_parameters changed shape or contains nonfinite values")
        if anomalous.shape != self.anomalous_factor_e.shape or not np.all(np.isfinite(anomalous)):
            raise ValueError("anomalous_factor_e changed shape or contains nonfinite values")
        scalar_values = {
            "normalization_divisor": float(normalization_divisor),
            "u_radial_A2": float(u_radial_A2),
            "u_normal_A2": float(u_normal_A2),
            "shared_disorder_epsilon": float(shared_disorder_epsilon),
        }
        if not all(math.isfinite(value) and value >= 0.0 for value in scalar_values.values()):
            raise ValueError("rebound structure scalars must be finite and nonnegative")
        if scalar_values["normalization_divisor"] == 0.0:
            raise ValueError("normalization_divisor must be positive")
        if not 0.0 <= scalar_values["shared_disorder_epsilon"] <= 1.0:
            raise ValueError("shared_disorder_epsilon must lie in [0, 1]")
        rebound_layers = int(layers)
        if isinstance(layers, bool) or rebound_layers < 1 or rebound_layers != layers:
            raise ValueError("layers must be a positive integer")
        for value in (offsets, properties, parameters, anomalous):
            value.setflags(write=False)

        rebound = object.__new__(type(self))
        for descriptor in fields(self):
            object.__setattr__(rebound, descriptor.name, getattr(self, descriptor.name))
        object.__setattr__(rebound, "gaussian_sigma_rad", mosaic.gaussian_sigma_rad)
        object.__setattr__(rebound, "lorentzian_hwhm_rad", mosaic.lorentzian_half_width_rad)
        object.__setattr__(rebound, "lorentzian_probability", mosaic.lorentzian_probability)
        object.__setattr__(rebound, "atom_fractional_offset", offsets)
        object.__setattr__(rebound, "atom_occupancy_element", properties)
        object.__setattr__(rebound, "f0_parameters", parameters)
        object.__setattr__(rebound, "anomalous_factor_e", anomalous)
        object.__setattr__(rebound, "layers", rebound_layers)
        for name, value in scalar_values.items():
            object.__setattr__(rebound, name, value)
        return rebound


def pack_bi2se3_two_h_structure(
    strength: Bi2Se3TwoHStrength,
    *,
    wavelength_A: float,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.complex128], int, float, float, float]:
    """Pack the existing CIF/XrayDB authorities once for exact compiled evaluation."""

    packed = pack_bi2se3_two_h_structures(
        strength,
        wavelength_A=np.asarray([wavelength_A], dtype=np.float64),
    )
    offsets, properties, parameters, anomalous, layers, divisor, u_radial, u_normal = packed
    return offsets, properties, parameters, anomalous[0], layers, divisor, u_radial, u_normal


def pack_bi2se3_two_h_structures(
    strength: Bi2Se3TwoHStrength,
    *,
    wavelength_A: NDArray[np.float64],
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    NDArray[np.complex128],
    int,
    float,
    float,
    float,
]:
    """Pack shared structure data and vectorized anomalous factors for many wavelengths."""

    if not isinstance(strength, Bi2Se3TwoHStrength):
        raise TypeError("compiled detector integration requires Bi2Se3TwoHStrength")
    wavelength = np.asarray(wavelength_A, dtype=np.float64)
    if wavelength.ndim != 1 or not wavelength.size or not np.all(np.isfinite(wavelength)):
        raise ValueError("wavelength_A must be a finite nonempty one-dimensional array")
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_A must be positive")
    structure = strength.structure_parameters
    if structure is None:
        raise ValueError("compiled quintuple-layer integration requires structure parameters")
    atoms = _parameterized_bi2se3_quintuple_layer(strength.crystal, structure)
    elements = tuple(
        sorted({atom.element for atom in atoms}, key=lambda value: (value != "Bi", value))
    )
    if len(elements) != 2 or elements[0] != "Bi":
        raise ValueError("compiled quintuple-layer integration requires Bi and one chalcogen")
    element_index = {element: position for position, element in enumerate(elements)}
    offsets = np.asarray([atom.fractional_offset for atom in atoms], dtype=np.float64)
    properties = np.asarray(
        [(atom.occupancy, element_index[atom.element]) for atom in atoms],
        dtype=np.float64,
    )

    waasmaier = xraydb.get_xraydb().get_cache("Waasmaier")
    parameters = np.empty((len(elements), 11), dtype=np.float64)
    anomalous = np.empty((wavelength.size, len(elements)), dtype=np.complex128)
    energy_eV = HC_EV_A / wavelength
    for position, element in enumerate(elements):
        charges = {atom.charge for atom in atoms if atom.element == element}
        if len(charges) != 1:
            raise ValueError(
                f"compiled quintuple-layer integration requires one charge for {element}"
            )
        species = _f0_species(element, charges.pop())
        row = next((candidate for candidate in waasmaier if candidate.ion == species), None)
        if row is None:
            raise ValueError(f"XrayDB has no Waasmaier coefficients for {species}")
        parameters[position, 0] = float(row.offset)
        parameters[position, 1:6] = json.loads(row.scale)
        parameters[position, 6:11] = json.loads(row.exponents)
        chantler_row = xraydb.get_xraydb().get_cache(
            "Chantler",
            column="element",
            value=element,
        )[0]
        chantler_energy_eV = np.asarray(json.loads(chantler_row.energy), dtype=np.float64)
        interval = np.searchsorted(chantler_energy_eV, energy_eV, side="right") - 1
        for interval_index in np.unique(interval):
            selected = interval == interval_index
            selected_energy = energy_eV[selected]
            anomalous[selected, position] = np.asarray(
                xraydb.f1_chantler(element, selected_energy),
                dtype=np.float64,
            ) + 1j * np.asarray(
                xraydb.f2_chantler(element, selected_energy),
                dtype=np.float64,
            )

    divisor = (
        float(strength.layers)
        if strength.normalization is EventIntensityNormalization.FINITE_PER_LAYER
        else 1.0
    )
    for value in (offsets, properties, parameters, anomalous):
        value.setflags(write=False)
    return (
        offsets,
        properties,
        parameters,
        anomalous,
        strength.layers,
        divisor,
        structure.u_radial_A2,
        structure.u_normal_A2,
    )


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _positive_normal_root(radicand: complex) -> complex:
    root = cmath.sqrt(radicand)
    if root.imag != 0.0:
        if root.imag < 0.0:
            root = -root
    elif root.real < 0.0:
        root = -root
    return root


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _exit_optical_weight(
    film_normal_radicand_Ainv2: complex,
    kf_air_z_Ainv: float,
    entrance_power: float,
    incident_decay_Ainv: float,
    film_thickness_A: float,
) -> float:
    kz_film = _positive_normal_root(film_normal_radicand_Ainv2)
    denominator = kz_film + complex(kf_air_z_Ainv, 0.0)
    if denominator == 0.0:
        return 0.0
    exit_amplitude = 2.0 * kz_film / denominator
    exponent = 2.0 * (incident_decay_Ainv + max(kz_film.imag, 0.0)) * film_thickness_A
    attenuation = 1.0 if exponent == 0.0 else -math.expm1(-exponent) / exponent
    return entrance_power * (exit_amplitude.real**2 + exit_amplitude.imag**2) * attenuation


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _element_factors(
    q_norm_squared_Ainv2: float,
    f0_parameters: FloatArray,
    anomalous_factor_e: NDArray[np.complex128],
) -> tuple[complex, complex]:
    q_xraydb_squared = q_norm_squared_Ainv2 / (16.0 * math.pi * math.pi)
    f0_0 = f0_parameters[0, 0]
    f0_1 = f0_parameters[1, 0]
    for coefficient in range(5):
        f0_0 += f0_parameters[0, 1 + coefficient] * math.exp(
            -f0_parameters[0, 6 + coefficient] * q_xraydb_squared
        )
        f0_1 += f0_parameters[1, 1 + coefficient] * math.exp(
            -f0_parameters[1, 6 + coefficient] * q_xraydb_squared
        )
    return f0_0 + anomalous_factor_e[0], f0_1 + anomalous_factor_e[1]


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _common_damping(
    q_norm_squared_Ainv2: float,
    parallel_norm_Ainv: float,
    w_value_Ainv: float,
    u_radial_A2: float,
    u_normal_A2: float,
) -> float:
    if u_radial_A2 == u_normal_A2:
        return math.exp(-0.5 * q_norm_squared_Ainv2 * u_radial_A2)
    return math.exp(
        -0.5
        * (
            u_radial_A2 * parallel_norm_Ainv * parallel_norm_Ainv
            + u_normal_A2 * w_value_Ainv * w_value_Ainv
        )
    )


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _wrapped_mosaic_density(
    alpha: float,
    gaussian_sigma: float,
    gaussian_probability: float,
    gaussian_normalization: float,
    lorentzian_probability: float,
    lorentzian_rho: float,
    lorentzian_one_minus_rho: float,
    lorentzian_numerator: float,
) -> float:
    two_pi = 2.0 * math.pi
    wrapped = (alpha + math.pi) % two_pi - math.pi
    density = 0.0
    if gaussian_probability > 0.0 and gaussian_sigma > 0.0:
        if gaussian_sigma >= 1.0:
            gaussian = 1.0
            harmonic = 1
            while harmonic <= 100_000:
                amplitude = 2.0 * math.exp(-0.5 * (harmonic * gaussian_sigma) ** 2)
                gaussian += amplitude * math.cos(harmonic * wrapped)
                next_amplitude = 2.0 * math.exp(-0.5 * ((harmonic + 1) * gaussian_sigma) ** 2)
                if next_amplitude <= 1.0e-15 * gaussian:
                    break
                harmonic += 1
            gaussian /= two_pi
        else:
            scaled = math.exp(-0.5 * (wrapped / gaussian_sigma) ** 2)
            # For sigma < 1 rad and wrapped in [-pi, pi], every |n| >= 2
            # image is below 1e-15 of the retained n=0,+/-1 sum.  Keeping the
            # first pair also handles the exactly wrapped +/-pi seam.
            scaled += math.exp(-0.5 * ((wrapped + two_pi) / gaussian_sigma) ** 2)
            scaled += math.exp(-0.5 * ((wrapped - two_pi) / gaussian_sigma) ** 2)
            gaussian = scaled / gaussian_normalization
        density += gaussian_probability * gaussian
    if lorentzian_probability > 0.0 and lorentzian_numerator > 0.0:
        denominator = two_pi * (
            lorentzian_one_minus_rho * lorentzian_one_minus_rho
            + 4.0 * lorentzian_rho * math.sin(0.5 * wrapped) ** 2
        )
        density += lorentzian_probability * lorentzian_numerator / denominator
    return density / math.pi


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _two_h_strength_A2(
    rod_index: int,
    ell: float,
    common_damping: float,
    element_factor_0: complex,
    element_factor_1: complex,
    rod_atom_inplane_factor: NDArray[np.complex128],
    atom_fractional_offset: FloatArray,
    atom_occupancy_element: FloatArray,
    layers: int,
    shared_disorder_epsilon: float,
    rod_hk_population: FloatArray,
    normalization_divisor: float,
) -> float:
    amplitude_plus = 0.0 + 0.0j
    amplitude_minus = 0.0 + 0.0j
    for atom in range(atom_fractional_offset.shape[0]):
        occupancy = atom_occupancy_element[atom, 0]
        element = int(atom_occupancy_element[atom, 1])
        phase_z = 2.0 * math.pi * ell * atom_fractional_offset[atom, 2]
        inplane_factor = rod_atom_inplane_factor[rod_index, atom]
        phase_plus = inplane_factor * complex(math.cos(phase_z), math.sin(phase_z))
        phase_minus = inplane_factor * complex(math.cos(phase_z), -math.sin(phase_z))
        element_factor = element_factor_0 if element == 0 else element_factor_1
        amplitude_plus += occupancy * element_factor * phase_plus
        amplitude_minus += occupancy * element_factor * phase_minus

    vertical_phase_angle = 2.0 * math.pi * ell / 3.0
    vertical_phase = complex(math.cos(vertical_phase_angle), math.sin(vertical_phase_angle))
    if shared_disorder_epsilon == 0.0:
        phase_power = 1.0 + 0.0j
        stack_sum = 1.0 + 0.0j
        for _ in range(1, layers):
            phase_power *= vertical_phase
            stack_sum += phase_power
        total = amplitude_plus * stack_sum
        intensity_e2 = total.real * total.real + total.imag * total.imag
    else:
        h = int(rod_hk_population[rod_index, 0])
        k = int(rod_hk_population[rod_index, 1])
        registry_index = (h + 2 * k) % 3
        if registry_index == 0:
            omega = 1.0 + 0.0j
        elif registry_index == 1:
            omega = complex(-0.5, 0.5 * math.sqrt(3.0))
        else:
            omega = complex(-0.5, -0.5 * math.sqrt(3.0))
        inverse_omega = omega.conjugate()
        alternative = 0.25 * shared_disorder_epsilon
        parent = 1.0 - shared_disorder_epsilon
        same_probability = parent + 2.0 * alternative
        flip_probability = 2.0 * alternative
        same_gauge = parent + alternative * inverse_omega + alternative * omega
        plus_to_minus_gauge = alternative * inverse_omega + alternative * omega
        minus_to_plus_gauge = alternative * omega + alternative * inverse_omega

        probability_plus = 1.0
        probability_minus = 0.0
        first_moment_plus = amplitude_plus
        first_moment_minus = 0.0 + 0.0j
        second_moment_plus = (
            amplitude_plus.real * amplitude_plus.real + amplitude_plus.imag * amplitude_plus.imag
        )
        second_moment_minus = 0.0
        phase_power = 1.0 + 0.0j
        for _ in range(1, layers):
            phase_power *= vertical_phase
            contribution_plus = phase_power * amplitude_plus
            contribution_minus = phase_power * amplitude_minus
            next_probability_plus = (
                probability_plus * same_probability + probability_minus * flip_probability
            )
            next_probability_minus = (
                probability_minus * same_probability + probability_plus * flip_probability
            )
            propagated_plus = (
                same_gauge * first_moment_plus + minus_to_plus_gauge * first_moment_minus
            )
            propagated_minus = (
                same_gauge * first_moment_minus + plus_to_minus_gauge * first_moment_plus
            )
            next_first_moment_plus = propagated_plus + next_probability_plus * contribution_plus
            next_first_moment_minus = propagated_minus + next_probability_minus * contribution_minus
            contribution_plus_squared = (
                contribution_plus.real * contribution_plus.real
                + contribution_plus.imag * contribution_plus.imag
            )
            contribution_minus_squared = (
                contribution_minus.real * contribution_minus.real
                + contribution_minus.imag * contribution_minus.imag
            )
            cross_plus = contribution_plus * propagated_plus.conjugate()
            cross_minus = contribution_minus * propagated_minus.conjugate()
            next_second_moment_plus = (
                same_probability * second_moment_plus
                + flip_probability * second_moment_minus
                + next_probability_plus * contribution_plus_squared
                + 2.0 * cross_plus.real
            )
            next_second_moment_minus = (
                same_probability * second_moment_minus
                + flip_probability * second_moment_plus
                + next_probability_minus * contribution_minus_squared
                + 2.0 * cross_minus.real
            )
            probability_plus = next_probability_plus
            probability_minus = next_probability_minus
            first_moment_plus = next_first_moment_plus
            first_moment_minus = next_first_moment_minus
            second_moment_plus = next_second_moment_plus
            second_moment_minus = next_second_moment_minus
        intensity_e2 = max(second_moment_plus + second_moment_minus, 0.0)
    return (
        CLASSICAL_ELECTRON_RADIUS_A**2
        * common_damping
        * common_damping
        * intensity_e2
        / normalization_divisor
    )


@numba.njit(nogil=True, fastmath=False, cache=False)
def _evaluate_point_into(
    column: float,
    row: float,
    detector_shape_rc: tuple[int, int],
    detector_zero_lab_m: FloatArray,
    detector_column_step_lab_m: FloatArray,
    detector_row_step_lab_m: FloatArray,
    detector_pixel_area_vector_lab_m2: FloatArray,
    ray_origin_lab_m: FloatArray,
    sample_from_lab: FloatArray,
    ki_film_sample_Ainv: FloatArray,
    internal_k_Ainv: float,
    internal_k_squared_Ainv2: float,
    air_k0_Ainv: float,
    refractive_air_k_squared_Ainv2: complex,
    incident_decay_Ainv: float,
    film_thickness_A: float,
    source_phase_weight: float,
    sample_from_local: FloatArray,
    rod_hk_population: FloatArray,
    rod_parallel_local_Ainv: FloatArray,
    rod_u_bounds_Ainv: FloatArray,
    rod_inverse_constants: FloatArray,
    b3_norm_Ainv: float,
    gaussian_sigma_rad: float,
    gaussian_probability: float,
    gaussian_normalization: float,
    lorentzian_probability: float,
    lorentzian_rho: float,
    lorentzian_one_minus_rho: float,
    lorentzian_numerator: float,
    atom_fractional_offset: FloatArray,
    atom_occupancy_element: FloatArray,
    rod_atom_inplane_factor: NDArray[np.complex128],
    u_radial_A2: float,
    u_normal_A2: float,
    f0_parameters: FloatArray,
    anomalous_factor_e: NDArray[np.complex128],
    layers: int,
    shared_disorder_epsilon: float,
    normalization_divisor: float,
    branch: int,
    entrance_power: float,
    reconstruction_tolerance: float,
    density: FloatArray,
    inverse_count: IntArray,
    caustic: BoolArray,
) -> bool:
    rod_count = rod_hk_population.shape[0]
    for rod_index in range(rod_count):
        density[rod_index] = 0.0
        inverse_count[rod_index] = 0
        caustic[rod_index] = False
    rows, columns = detector_shape_rc
    if column < -0.5 or column > columns - 0.5 or row < -0.5 or row > rows - 0.5:
        return False

    two_pi = 2.0 * math.pi
    angular_tolerance = 2048.0 * np.finfo(np.float64).eps
    displacement_x = (
        detector_zero_lab_m[0]
        + column * detector_column_step_lab_m[0]
        + row * detector_row_step_lab_m[0]
        - ray_origin_lab_m[0]
    )
    displacement_y = (
        detector_zero_lab_m[1]
        + column * detector_column_step_lab_m[1]
        + row * detector_row_step_lab_m[1]
        - ray_origin_lab_m[1]
    )
    displacement_z = (
        detector_zero_lab_m[2]
        + column * detector_column_step_lab_m[2]
        + row * detector_row_step_lab_m[2]
        - ray_origin_lab_m[2]
    )
    distance = math.sqrt(
        displacement_x * displacement_x
        + displacement_y * displacement_y
        + displacement_z * displacement_z
    )
    if distance == 0.0:
        return False
    direction_x = displacement_x / distance
    direction_y = displacement_y / distance
    direction_z = displacement_z / distance
    signed_pixel_area_projection = (
        direction_x * detector_pixel_area_vector_lab_m2[0]
        + direction_y * detector_pixel_area_vector_lab_m2[1]
        + direction_z * detector_pixel_area_vector_lab_m2[2]
    )
    pixel_area = math.sqrt(
        detector_pixel_area_vector_lab_m2[0] ** 2
        + detector_pixel_area_vector_lab_m2[1] ** 2
        + detector_pixel_area_vector_lab_m2[2] ** 2
    )
    if signed_pixel_area_projection <= _DETECTOR_INCIDENCE_COSINE_TOL * pixel_area:
        return False
    scaled_direction_x = air_k0_Ainv * direction_x
    scaled_direction_y = air_k0_Ainv * direction_y
    scaled_direction_z = air_k0_Ainv * direction_z
    kf_air_x = (
        sample_from_lab[0, 0] * scaled_direction_x
        + sample_from_lab[0, 1] * scaled_direction_y
        + sample_from_lab[0, 2] * scaled_direction_z
    )
    kf_air_y = (
        sample_from_lab[1, 0] * scaled_direction_x
        + sample_from_lab[1, 1] * scaled_direction_y
        + sample_from_lab[1, 2] * scaled_direction_z
    )
    kf_air_z = (
        sample_from_lab[2, 0] * scaled_direction_x
        + sample_from_lab[2, 1] * scaled_direction_y
        + sample_from_lab[2, 2] * scaled_direction_z
    )
    if kf_air_z <= 0.0:
        return False
    parallel_squared = kf_air_x * kf_air_x + kf_air_y * kf_air_y
    normal_squared = internal_k_squared_Ainv2 - parallel_squared
    if normal_squared <= 0.0:
        return False
    kf_film_x = kf_air_x
    kf_film_y = kf_air_y
    kf_film_z = math.sqrt(normal_squared)
    q_sample_x = kf_film_x - ki_film_sample_Ainv[0]
    q_sample_y = kf_film_y - ki_film_sample_Ainv[1]
    q_sample_z = kf_film_z - ki_film_sample_Ainv[2]

    pixel_solid_angle = signed_pixel_area_projection / (distance * distance)
    area_jacobian = internal_k_Ainv * air_k0_Ainv * kf_air_z * pixel_solid_angle / kf_film_z

    optical_weight = _exit_optical_weight(
        refractive_air_k_squared_Ainv2 - parallel_squared,
        kf_air_z,
        entrance_power,
        incident_decay_Ainv,
        film_thickness_A,
    )

    q_local_x = (
        q_sample_x * sample_from_local[0, 0]
        + q_sample_y * sample_from_local[1, 0]
        + q_sample_z * sample_from_local[2, 0]
    )
    q_local_y = (
        q_sample_x * sample_from_local[0, 1]
        + q_sample_y * sample_from_local[1, 1]
        + q_sample_z * sample_from_local[2, 1]
    )
    q_local_z = (
        q_sample_x * sample_from_local[0, 2]
        + q_sample_y * sample_from_local[1, 2]
        + q_sample_z * sample_from_local[2, 2]
    )
    q_norm_squared = q_local_x * q_local_x + q_local_y * q_local_y + q_local_z * q_local_z
    q_norm = math.sqrt(q_norm_squared)
    element_factor_0, element_factor_1 = _element_factors(
        q_norm_squared,
        f0_parameters,
        anomalous_factor_e,
    )
    transverse_norm = math.hypot(q_local_x, q_local_y)
    azimuth_q = math.atan2(q_local_y, q_local_x)
    for rod_index in range(rod_count):
        a = rod_parallel_local_Ainv[rod_index, 0]
        b = rod_parallel_local_Ainv[rod_index, 1]
        c0 = rod_parallel_local_Ainv[rod_index, 2]
        abs_b = rod_inverse_constants[rod_index, 0]
        parallel_norm = rod_inverse_constants[rod_index, 1]
        inverse_reference = rod_inverse_constants[rod_index, 2]
        u_tolerance = rod_inverse_constants[rod_index, 3]
        x_squared = (transverse_norm - abs_b) * (transverse_norm + abs_b)
        w_squared = (q_norm - parallel_norm) * (q_norm + parallel_norm)
        inverse_scale = max(q_norm_squared, inverse_reference, 1.0)
        inverse_tolerance = 1024.0 * np.finfo(np.float64).eps * inverse_scale
        if x_squared < -inverse_tolerance or w_squared < -inverse_tolerance:
            continue
        x_magnitude = math.sqrt(max(x_squared, 0.0))
        w_magnitude = math.sqrt(max(w_squared, 0.0))
        lower_u = rod_u_bounds_Ainv[rod_index, 0]
        upper_u = rod_u_bounds_Ainv[rod_index, 1]
        for x_sign in (-1.0, 1.0):
            x_value = x_sign * x_magnitude
            beta = (azimuth_q - math.atan2(b, x_value)) % two_pi
            if beta >= two_pi:
                beta = 0.0
            for w_sign in (-1.0, 1.0):
                w_value = w_sign * w_magnitude
                alpha = (math.atan2(w_value, a) - math.atan2(q_local_z, x_value)) % two_pi
                if alpha >= two_pi - angular_tolerance:
                    alpha = 0.0
                if alpha > math.pi + angular_tolerance:
                    continue
                alpha = min(alpha, math.pi)
                u_value = w_value - c0
                if u_value < lower_u - u_tolerance or u_value > upper_u + u_tolerance:
                    continue
                sin_alpha = math.sin(alpha)
                direction_local_x = sin_alpha * math.cos(beta)
                direction_local_y = sin_alpha * math.sin(beta)
                direction_local_z = math.cos(alpha)
                direction_sample_x = (
                    sample_from_local[0, 0] * direction_local_x
                    + sample_from_local[0, 1] * direction_local_y
                    + sample_from_local[0, 2] * direction_local_z
                )
                direction_sample_y = (
                    sample_from_local[1, 0] * direction_local_x
                    + sample_from_local[1, 1] * direction_local_y
                    + sample_from_local[1, 2] * direction_local_z
                )
                direction_sample_z = (
                    sample_from_local[2, 0] * direction_local_x
                    + sample_from_local[2, 1] * direction_local_y
                    + sample_from_local[2, 2] * direction_local_z
                )
                root_sign = (
                    kf_film_x * direction_sample_x
                    + kf_film_y * direction_sample_y
                    + kf_film_z * direction_sample_z
                )
                if (branch == 2 and root_sign <= 0.0) or (branch == 1 and root_sign >= 0.0):
                    continue
                ell = u_value / b3_norm_Ainv
                common_damping = _common_damping(
                    q_norm_squared,
                    parallel_norm,
                    w_value,
                    u_radial_A2,
                    u_normal_A2,
                )
                strength = _two_h_strength_A2(
                    rod_index,
                    ell,
                    common_damping,
                    element_factor_0,
                    element_factor_1,
                    rod_atom_inplane_factor,
                    atom_fractional_offset,
                    atom_occupancy_element,
                    layers,
                    shared_disorder_epsilon,
                    rod_hk_population,
                    normalization_divisor,
                )
                mosaic_density = _wrapped_mosaic_density(
                    alpha,
                    gaussian_sigma_rad,
                    gaussian_probability,
                    gaussian_normalization,
                    lorentzian_probability,
                    lorentzian_rho,
                    lorentzian_one_minus_rho,
                    lorentzian_numerator,
                )
                jacobian = abs(w_value * x_value)
                if jacobian == 0.0:
                    cos_beta = math.cos(beta)
                    sin_beta = math.sin(beta)
                    reconstructed_local_x = x_value * cos_beta - b * sin_beta
                    reconstructed_local_y = x_value * sin_beta + b * cos_beta
                    reconstructed_local_z = -a * math.sin(alpha) + w_value * math.cos(alpha)
                    reconstructed_sample_x = (
                        sample_from_local[0, 0] * reconstructed_local_x
                        + sample_from_local[0, 1] * reconstructed_local_y
                        + sample_from_local[0, 2] * reconstructed_local_z
                    )
                    reconstructed_sample_y = (
                        sample_from_local[1, 0] * reconstructed_local_x
                        + sample_from_local[1, 1] * reconstructed_local_y
                        + sample_from_local[1, 2] * reconstructed_local_z
                    )
                    reconstructed_sample_z = (
                        sample_from_local[2, 0] * reconstructed_local_x
                        + sample_from_local[2, 1] * reconstructed_local_y
                        + sample_from_local[2, 2] * reconstructed_local_z
                    )
                    reconstruction_error = math.sqrt(
                        (reconstructed_sample_x - q_sample_x) ** 2
                        + (reconstructed_sample_y - q_sample_y) ** 2
                        + (reconstructed_sample_z - q_sample_z) ** 2
                    )
                    if reconstruction_error > reconstruction_tolerance:
                        continue
                    caustic[rod_index] = True
                    if (
                        source_phase_weight > 0.0
                        and rod_hk_population[rod_index, 2] > 0.0
                        and area_jacobian > 0.0
                        and optical_weight > 0.0
                        and strength > 0.0
                        and mosaic_density > 0.0
                    ):
                        density[rod_index] = np.inf
                    continue
                density[rod_index] += (
                    mosaic_density
                    * rod_hk_population[rod_index, 2]
                    * strength
                    * area_jacobian
                    * optical_weight
                    * source_phase_weight
                    / jacobian
                )
                inverse_count[rod_index] += 1
    return True


@numba.njit(nogil=True, fastmath=False, cache=False)
def _evaluate_points_kernel(
    column_px: FloatArray,
    row_px: FloatArray,
    detector_shape_rc: tuple[int, int],
    detector_zero_lab_m: FloatArray,
    detector_column_step_lab_m: FloatArray,
    detector_row_step_lab_m: FloatArray,
    detector_pixel_area_vector_lab_m2: FloatArray,
    ray_origin_lab_m: FloatArray,
    sample_from_lab: FloatArray,
    ki_film_sample_Ainv: FloatArray,
    internal_k_Ainv: float,
    air_k0_Ainv: float,
    refractive_index: complex,
    entrance_amplitude: complex,
    incident_decay_Ainv: float,
    film_thickness_A: float,
    source_phase_weight: float,
    sample_from_local: FloatArray,
    rod_hk_population: FloatArray,
    rod_parallel_local_Ainv: FloatArray,
    rod_u_bounds_Ainv: FloatArray,
    rod_inverse_constants: FloatArray,
    b3_norm_Ainv: float,
    gaussian_sigma_rad: float,
    lorentzian_hwhm_rad: float,
    lorentzian_probability: float,
    atom_fractional_offset: FloatArray,
    atom_occupancy_element: FloatArray,
    rod_atom_inplane_factor: NDArray[np.complex128],
    u_radial_A2: float,
    u_normal_A2: float,
    f0_parameters: FloatArray,
    anomalous_factor_e: NDArray[np.complex128],
    layers: int,
    shared_disorder_epsilon: float,
    normalization_divisor: float,
    branch: int,
) -> tuple[FloatArray, IntArray, BoolArray, BoolArray]:
    size = column_px.size
    rod_count = rod_hk_population.shape[0]
    density = np.zeros((size, rod_count), dtype=np.float64)
    inverse_count = np.zeros((size, rod_count), dtype=np.int64)
    caustic = np.zeros((size, rod_count), dtype=np.bool_)
    valid = np.zeros(size, dtype=np.bool_)
    ki_norm = math.sqrt(
        ki_film_sample_Ainv[0] ** 2 + ki_film_sample_Ainv[1] ** 2 + ki_film_sample_Ainv[2] ** 2
    )
    reconstruction_tolerance = 4096.0 * np.finfo(np.float64).eps * max(ki_norm, 1.0)
    entrance_power = entrance_amplitude.real**2 + entrance_amplitude.imag**2
    internal_k_squared_Ainv2 = internal_k_Ainv * internal_k_Ainv
    refractive_air_k_squared_Ainv2 = (refractive_index * air_k0_Ainv) ** 2
    gaussian_probability = 1.0 - lorentzian_probability
    gaussian_normalization = math.sqrt(2.0 * math.pi) * gaussian_sigma_rad
    lorentzian_rho = math.exp(-lorentzian_hwhm_rad)
    lorentzian_one_minus_rho = -math.expm1(-lorentzian_hwhm_rad)
    lorentzian_numerator = -math.expm1(-2.0 * lorentzian_hwhm_rad)
    for point in range(size):
        valid[point] = _evaluate_point_into(
            column_px[point],
            row_px[point],
            detector_shape_rc,
            detector_zero_lab_m,
            detector_column_step_lab_m,
            detector_row_step_lab_m,
            detector_pixel_area_vector_lab_m2,
            ray_origin_lab_m,
            sample_from_lab,
            ki_film_sample_Ainv,
            internal_k_Ainv,
            internal_k_squared_Ainv2,
            air_k0_Ainv,
            refractive_air_k_squared_Ainv2,
            incident_decay_Ainv,
            film_thickness_A,
            source_phase_weight,
            sample_from_local,
            rod_hk_population,
            rod_parallel_local_Ainv,
            rod_u_bounds_Ainv,
            rod_inverse_constants,
            b3_norm_Ainv,
            gaussian_sigma_rad,
            gaussian_probability,
            gaussian_normalization,
            lorentzian_probability,
            lorentzian_rho,
            lorentzian_one_minus_rho,
            lorentzian_numerator,
            atom_fractional_offset,
            atom_occupancy_element,
            rod_atom_inplane_factor,
            u_radial_A2,
            u_normal_A2,
            f0_parameters,
            anomalous_factor_e,
            layers,
            shared_disorder_epsilon,
            normalization_divisor,
            branch,
            entrance_power,
            reconstruction_tolerance,
            density[point],
            inverse_count[point],
            caustic[point],
        )
    return density, inverse_count, caustic, valid


@numba.njit(nogil=True, fastmath=False, cache=False)
def _forward_root_pixel(
    rod_index: int,
    u_Ainv: float,
    kf_film_x: float,
    kf_film_y: float,
    kf_film_z: float,
    coarea_jacobian: float,
    state: _ForwardDetectorState,
) -> tuple[int, float, bool]:
    """Map one regular latent root and return its exact native-pixel owner and mass."""

    if kf_film_z <= 0.0 or coarea_jacobian <= 0.0:
        return -1, 0.0, False
    kf_norm = math.sqrt(kf_film_x * kf_film_x + kf_film_y * kf_film_y + kf_film_z * kf_film_z)
    residual_limit = 512.0 * np.finfo(np.float64).eps * max(state.internal_k_Ainv, 1.0)
    if abs(kf_norm - state.internal_k_Ainv) > residual_limit:
        return -1, 0.0, True
    parallel_squared = kf_film_x * kf_film_x + kf_film_y * kf_film_y
    air_normal_squared = state.air_k0_Ainv * state.air_k0_Ainv - parallel_squared
    critical_tolerance = (
        16.0
        * np.finfo(np.float64).eps
        * max(
            state.air_k0_Ainv * state.air_k0_Ainv,
            parallel_squared,
            1.0,
        )
    )
    if air_normal_squared < -critical_tolerance:
        return -1, 0.0, False
    kf_air_z = math.sqrt(max(air_normal_squared, 0.0))
    optical_weight = _exit_optical_weight(
        (state.refractive_index * state.air_k0_Ainv) ** 2 - parallel_squared,
        kf_air_z,
        state.entrance_power,
        state.incident_decay_Ainv,
        state.film_thickness_A,
    )
    if optical_weight <= 0.0 or state.source_phase_weight <= 0.0:
        return -1, 0.0, False

    direction_sample_x = kf_film_x / state.air_k0_Ainv
    direction_sample_y = kf_film_y / state.air_k0_Ainv
    direction_sample_z = kf_air_z / state.air_k0_Ainv
    direction_column_per_m = (
        state.detector_column_row_covectors_sample_per_m[0, 0] * direction_sample_x
        + state.detector_column_row_covectors_sample_per_m[0, 1] * direction_sample_y
        + state.detector_column_row_covectors_sample_per_m[0, 2] * direction_sample_z
    )
    direction_row_per_m = (
        state.detector_column_row_covectors_sample_per_m[1, 0] * direction_sample_x
        + state.detector_column_row_covectors_sample_per_m[1, 1] * direction_sample_y
        + state.detector_column_row_covectors_sample_per_m[1, 2] * direction_sample_z
    )
    direction_normal = (
        state.detector_normal_sample[0] * direction_sample_x
        + state.detector_normal_sample[1] * direction_sample_y
        + state.detector_normal_sample[2] * direction_sample_z
    )
    if direction_normal <= _DETECTOR_INCIDENCE_COSINE_TOL:
        return -1, 0.0, False
    ray_distance = -state.ray_origin_detector_normal_m / direction_normal
    if ray_distance <= 0.0:
        return -1, 0.0, False
    column = state.ray_origin_detector_column_row_px[0] + ray_distance * direction_column_per_m
    row = state.ray_origin_detector_column_row_px[1] + ray_distance * direction_row_per_m
    rows, columns = state.detector_shape_rc
    if column < -0.5 or column > columns - 0.5 or row < -0.5 or row > rows - 0.5:
        return -1, 0.0, False

    q_x = kf_film_x - state.ki_film_sample_Ainv[0]
    q_y = kf_film_y - state.ki_film_sample_Ainv[1]
    q_z = kf_film_z - state.ki_film_sample_Ainv[2]
    q_norm_squared = q_x * q_x + q_y * q_y + q_z * q_z
    element_factor_0, element_factor_1 = _element_factors(
        q_norm_squared,
        state.f0_parameters,
        state.anomalous_factor_e,
    )
    parallel_norm = state.rod_inverse_constants[rod_index, 1]
    w_value = u_Ainv + state.rod_parallel_local_Ainv[rod_index, 2]
    common_damping = _common_damping(
        q_norm_squared,
        parallel_norm,
        w_value,
        state.u_radial_A2,
        state.u_normal_A2,
    )
    strength = _two_h_strength_A2(
        rod_index,
        u_Ainv / state.b3_norm_Ainv,
        common_damping,
        element_factor_0,
        element_factor_1,
        state.rod_atom_inplane_factor,
        state.atom_fractional_offset,
        state.atom_occupancy_element,
        state.layers,
        state.shared_disorder_epsilon,
        state.rod_hk_population,
        state.normalization_divisor,
    )
    importance_weight = (
        state.source_phase_weight
        * state.rod_hk_population[rod_index, 2]
        * strength
        * coarea_jacobian
        * optical_weight
    )
    if not math.isfinite(importance_weight):
        return -1, 0.0, True
    if importance_weight <= 0.0:
        return -1, 0.0, False
    pixel_column = min(math.floor(column + 0.5), columns - 1)
    pixel_row = min(math.floor(row + 0.5), rows - 1)
    return pixel_row * columns + pixel_column, importance_weight, False


@numba.njit(nogil=True, fastmath=False, cache=False)
def _accumulate_latent_pixel_mass_kernel(
    alpha_rad: FloatArray,
    beta_rad: FloatArray,
    image_A2: FloatArray,
    replicate_total_mass_A2: FloatArray,
    inverse_draw_count: float,
    state: _ForwardDetectorState,
) -> tuple[int, float]:
    draw_count = alpha_rad.size
    rod_count = state.rod_hk_population.shape[0]
    visible_hit_count = 0
    maximum_root_deposit_A2 = 0.0
    for draw in range(draw_count):
        alpha = alpha_rad[draw]
        beta = beta_rad[draw]
        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)
        sin_beta = math.sin(beta)
        cos_beta = math.cos(beta)
        direction_local_x = sin_alpha * cos_beta
        direction_local_y = sin_alpha * sin_beta
        direction_local_z = cos_alpha
        direction_sample_x = (
            state.sample_from_local[0, 0] * direction_local_x
            + state.sample_from_local[0, 1] * direction_local_y
            + state.sample_from_local[0, 2] * direction_local_z
        )
        direction_sample_y = (
            state.sample_from_local[1, 0] * direction_local_x
            + state.sample_from_local[1, 1] * direction_local_y
            + state.sample_from_local[1, 2] * direction_local_z
        )
        direction_sample_z = (
            state.sample_from_local[2, 0] * direction_local_x
            + state.sample_from_local[2, 1] * direction_local_y
            + state.sample_from_local[2, 2] * direction_local_z
        )
        direction_norm_squared = (
            direction_sample_x * direction_sample_x
            + direction_sample_y * direction_sample_y
            + direction_sample_z * direction_sample_z
        )
        direction_norm = math.sqrt(direction_norm_squared)
        incident_dot_direction = (
            state.ki_film_sample_Ainv[0] * direction_sample_x
            + state.ki_film_sample_Ainv[1] * direction_sample_y
            + state.ki_film_sample_Ainv[2] * direction_sample_z
        )
        incident_parallel = incident_dot_direction / direction_norm_squared
        for rod_index in range(rod_count):
            a = state.rod_parallel_local_Ainv[rod_index, 0]
            b = state.rod_parallel_local_Ainv[rod_index, 1]
            c0 = state.rod_parallel_local_Ainv[rod_index, 2]
            x0 = a * cos_alpha + c0 * sin_alpha
            q0_local_x = x0 * cos_beta - b * sin_beta
            q0_local_y = x0 * sin_beta + b * cos_beta
            q0_local_z = -a * sin_alpha + c0 * cos_alpha
            q0_sample_x = (
                state.sample_from_local[0, 0] * q0_local_x
                + state.sample_from_local[0, 1] * q0_local_y
                + state.sample_from_local[0, 2] * q0_local_z
            )
            q0_sample_y = (
                state.sample_from_local[1, 0] * q0_local_x
                + state.sample_from_local[1, 1] * q0_local_y
                + state.sample_from_local[1, 2] * q0_local_z
            )
            q0_sample_z = (
                state.sample_from_local[2, 0] * q0_local_x
                + state.sample_from_local[2, 1] * q0_local_y
                + state.sample_from_local[2, 2] * q0_local_z
            )
            q0_dot_direction = (
                q0_sample_x * direction_sample_x
                + q0_sample_y * direction_sample_y
                + q0_sample_z * direction_sample_z
            )
            q0_parallel = q0_dot_direction / direction_norm_squared
            is_m0 = (
                int(state.rod_hk_population[rod_index, 0]) == 0
                and int(state.rod_hk_population[rod_index, 1]) == 0
            )
            lower_u = state.rod_u_bounds_Ainv[rod_index, 0]
            upper_u = state.rod_u_bounds_Ainv[rod_index, 1]
            u_tolerance = state.rod_inverse_constants[rod_index, 3]
            if is_m0:
                if incident_dot_direction == 0.0:
                    continue
                u_value = -2.0 * incident_parallel - q0_parallel
                if u_value < lower_u - u_tolerance or u_value > upper_u + u_tolerance:
                    continue
                kf_film_x = state.ki_film_sample_Ainv[0] + u_value * direction_sample_x
                kf_film_y = state.ki_film_sample_Ainv[1] + u_value * direction_sample_y
                kf_film_z = state.ki_film_sample_Ainv[2] + u_value * direction_sample_z
                coarea = state.internal_k_Ainv / abs(incident_dot_direction)
                pixel, weight, numeric_failure = _forward_root_pixel(
                    rod_index,
                    u_value,
                    kf_film_x,
                    kf_film_y,
                    kf_film_z,
                    coarea,
                    state,
                )
                if numeric_failure:
                    raise FloatingPointError("invalid Monte Carlo root numerics")
                if pixel >= 0:
                    deposit = weight * inverse_draw_count
                    image_A2[pixel] += deposit
                    replicate_total_mass_A2[draw] += weight
                    visible_hit_count += 1
                    maximum_root_deposit_A2 = max(maximum_root_deposit_A2, deposit)
                continue

            q0_perpendicular_x = q0_sample_x - q0_parallel * direction_sample_x
            q0_perpendicular_y = q0_sample_y - q0_parallel * direction_sample_y
            q0_perpendicular_z = q0_sample_z - q0_parallel * direction_sample_z
            sphere_perpendicular_x = (
                state.ki_film_sample_Ainv[0]
                - incident_parallel * direction_sample_x
                + q0_perpendicular_x
            )
            sphere_perpendicular_y = (
                state.ki_film_sample_Ainv[1]
                - incident_parallel * direction_sample_y
                + q0_perpendicular_y
            )
            sphere_perpendicular_z = (
                state.ki_film_sample_Ainv[2]
                - incident_parallel * direction_sample_z
                + q0_perpendicular_z
            )
            perpendicular_squared = (
                sphere_perpendicular_x * sphere_perpendicular_x
                + sphere_perpendicular_y * sphere_perpendicular_y
                + sphere_perpendicular_z * sphere_perpendicular_z
            )
            discriminant = state.internal_k_Ainv * state.internal_k_Ainv - perpendicular_squared
            if discriminant <= 0.0:
                continue
            sqrt_discriminant = math.sqrt(discriminant)
            root_coordinate_magnitude = sqrt_discriminant / direction_norm
            parallel_offset = incident_parallel + q0_parallel
            coarea = state.internal_k_Ainv / (sqrt_discriminant * direction_norm)
            for root_slot in range(2):
                signed_root = (
                    -root_coordinate_magnitude if root_slot == 0 else root_coordinate_magnitude
                )
                u_value = -parallel_offset + signed_root
                if u_value < lower_u - u_tolerance or u_value > upper_u + u_tolerance:
                    continue
                kf_film_x = sphere_perpendicular_x + signed_root * direction_sample_x
                kf_film_y = sphere_perpendicular_y + signed_root * direction_sample_y
                kf_film_z = sphere_perpendicular_z + signed_root * direction_sample_z
                pixel, weight, numeric_failure = _forward_root_pixel(
                    rod_index,
                    u_value,
                    kf_film_x,
                    kf_film_y,
                    kf_film_z,
                    coarea,
                    state,
                )
                if numeric_failure:
                    raise FloatingPointError("invalid Monte Carlo root numerics")
                if pixel >= 0:
                    deposit = weight * inverse_draw_count
                    image_A2[pixel] += deposit
                    replicate_total_mass_A2[draw] += weight
                    visible_hit_count += 1
                    maximum_root_deposit_A2 = max(maximum_root_deposit_A2, deposit)
    return visible_hit_count, maximum_root_deposit_A2


@numba.njit(nogil=True, fastmath=False, cache=False)
def _integrate_pixel_boxes_kernel(
    flat_pixel_index: IntArray,
    offset_px: FloatArray,
    one_dimensional_weight: FloatArray,
    include_center_diagnostics: bool,
    detector_shape_rc: tuple[int, int],
    detector_zero_lab_m: FloatArray,
    detector_column_step_lab_m: FloatArray,
    detector_row_step_lab_m: FloatArray,
    detector_pixel_area_vector_lab_m2: FloatArray,
    ray_origin_lab_m: FloatArray,
    sample_from_lab: FloatArray,
    ki_film_sample_Ainv: FloatArray,
    internal_k_Ainv: float,
    air_k0_Ainv: float,
    refractive_index: complex,
    entrance_amplitude: complex,
    incident_decay_Ainv: float,
    film_thickness_A: float,
    source_phase_weight: float,
    sample_from_local: FloatArray,
    rod_hk_population: FloatArray,
    rod_parallel_local_Ainv: FloatArray,
    rod_u_bounds_Ainv: FloatArray,
    rod_inverse_constants: FloatArray,
    b3_norm_Ainv: float,
    gaussian_sigma_rad: float,
    lorentzian_hwhm_rad: float,
    lorentzian_probability: float,
    atom_fractional_offset: FloatArray,
    atom_occupancy_element: FloatArray,
    rod_atom_inplane_factor: NDArray[np.complex128],
    u_radial_A2: float,
    u_normal_A2: float,
    f0_parameters: FloatArray,
    anomalous_factor_e: NDArray[np.complex128],
    layers: int,
    shared_disorder_epsilon: float,
    normalization_divisor: float,
    branch: int,
) -> tuple[
    FloatArray,
    FloatArray,
    IntArray,
    IntArray,
    BoolArray,
    BoolArray,
    BoolArray,
    BoolArray,
]:
    pixel_count = flat_pixel_index.size
    rod_count = rod_hk_population.shape[0]
    _, columns = detector_shape_rc
    mass = np.zeros((pixel_count, rod_count), dtype=np.float64)
    diagnostic_pixel_count = pixel_count if include_center_diagnostics else 0
    center_density = np.zeros((diagnostic_pixel_count, rod_count), dtype=np.float64)
    count_min = np.full(
        (diagnostic_pixel_count, rod_count),
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )
    count_max = np.zeros((diagnostic_pixel_count, rod_count), dtype=np.int64)
    caustic_any = np.zeros((diagnostic_pixel_count, rod_count), dtype=np.bool_)
    valid_any = np.zeros(diagnostic_pixel_count, dtype=np.bool_)
    valid_all = np.ones(diagnostic_pixel_count, dtype=np.bool_)
    center_valid_pixel = np.zeros(diagnostic_pixel_count, dtype=np.bool_)
    density = np.empty(rod_count, dtype=np.float64)
    inverse_count = np.empty(rod_count, dtype=np.int64)
    caustic = np.empty(rod_count, dtype=np.bool_)
    ki_norm = math.sqrt(
        ki_film_sample_Ainv[0] ** 2 + ki_film_sample_Ainv[1] ** 2 + ki_film_sample_Ainv[2] ** 2
    )
    reconstruction_tolerance = 4096.0 * np.finfo(np.float64).eps * max(ki_norm, 1.0)
    entrance_power = entrance_amplitude.real**2 + entrance_amplitude.imag**2
    internal_k_squared_Ainv2 = internal_k_Ainv * internal_k_Ainv
    refractive_air_k_squared_Ainv2 = (refractive_index * air_k0_Ainv) ** 2
    gaussian_probability = 1.0 - lorentzian_probability
    gaussian_normalization = math.sqrt(2.0 * math.pi) * gaussian_sigma_rad
    lorentzian_rho = math.exp(-lorentzian_hwhm_rad)
    lorentzian_one_minus_rho = -math.expm1(-lorentzian_hwhm_rad)
    lorentzian_numerator = -math.expm1(-2.0 * lorentzian_hwhm_rad)

    for pixel in range(pixel_count):
        pixel_row = flat_pixel_index[pixel] // columns
        pixel_column = flat_pixel_index[pixel] - pixel_row * columns
        for row_node in range(offset_px.size):
            for column_node in range(offset_px.size):
                node_valid = _evaluate_point_into(
                    pixel_column + offset_px[column_node],
                    pixel_row + offset_px[row_node],
                    detector_shape_rc,
                    detector_zero_lab_m,
                    detector_column_step_lab_m,
                    detector_row_step_lab_m,
                    detector_pixel_area_vector_lab_m2,
                    ray_origin_lab_m,
                    sample_from_lab,
                    ki_film_sample_Ainv,
                    internal_k_Ainv,
                    internal_k_squared_Ainv2,
                    air_k0_Ainv,
                    refractive_air_k_squared_Ainv2,
                    incident_decay_Ainv,
                    film_thickness_A,
                    source_phase_weight,
                    sample_from_local,
                    rod_hk_population,
                    rod_parallel_local_Ainv,
                    rod_u_bounds_Ainv,
                    rod_inverse_constants,
                    b3_norm_Ainv,
                    gaussian_sigma_rad,
                    gaussian_probability,
                    gaussian_normalization,
                    lorentzian_probability,
                    lorentzian_rho,
                    lorentzian_one_minus_rho,
                    lorentzian_numerator,
                    atom_fractional_offset,
                    atom_occupancy_element,
                    rod_atom_inplane_factor,
                    u_radial_A2,
                    u_normal_A2,
                    f0_parameters,
                    anomalous_factor_e,
                    layers,
                    shared_disorder_epsilon,
                    normalization_divisor,
                    branch,
                    entrance_power,
                    reconstruction_tolerance,
                    density,
                    inverse_count,
                    caustic,
                )
                if include_center_diagnostics:
                    valid_any[pixel] = valid_any[pixel] or node_valid
                    valid_all[pixel] = valid_all[pixel] and node_valid
                node_weight = one_dimensional_weight[column_node] * one_dimensional_weight[row_node]
                for rod_index in range(rod_count):
                    value = 0.0 if caustic[rod_index] else density[rod_index]
                    mass[pixel, rod_index] += node_weight * value
                    if include_center_diagnostics:
                        count_min[pixel, rod_index] = min(
                            count_min[pixel, rod_index], inverse_count[rod_index]
                        )
                        count_max[pixel, rod_index] = max(
                            count_max[pixel, rod_index], inverse_count[rod_index]
                        )
                        caustic_any[pixel, rod_index] = (
                            caustic_any[pixel, rod_index] or caustic[rod_index]
                        )
        if include_center_diagnostics:
            center_valid = _evaluate_point_into(
                float(pixel_column),
                float(pixel_row),
                detector_shape_rc,
                detector_zero_lab_m,
                detector_column_step_lab_m,
                detector_row_step_lab_m,
                detector_pixel_area_vector_lab_m2,
                ray_origin_lab_m,
                sample_from_lab,
                ki_film_sample_Ainv,
                internal_k_Ainv,
                internal_k_squared_Ainv2,
                air_k0_Ainv,
                refractive_air_k_squared_Ainv2,
                incident_decay_Ainv,
                film_thickness_A,
                source_phase_weight,
                sample_from_local,
                rod_hk_population,
                rod_parallel_local_Ainv,
                rod_u_bounds_Ainv,
                rod_inverse_constants,
                b3_norm_Ainv,
                gaussian_sigma_rad,
                gaussian_probability,
                gaussian_normalization,
                lorentzian_probability,
                lorentzian_rho,
                lorentzian_one_minus_rho,
                lorentzian_numerator,
                atom_fractional_offset,
                atom_occupancy_element,
                rod_atom_inplane_factor,
                u_radial_A2,
                u_normal_A2,
                f0_parameters,
                anomalous_factor_e,
                layers,
                shared_disorder_epsilon,
                normalization_divisor,
                branch,
                entrance_power,
                reconstruction_tolerance,
                density,
                inverse_count,
                caustic,
            )
            center_valid_pixel[pixel] = center_valid
            valid_any[pixel] = valid_any[pixel] or center_valid
            valid_all[pixel] = valid_all[pixel] and center_valid
            for rod_index in range(rod_count):
                center_density[pixel, rod_index] = 0.0 if caustic[rod_index] else density[rod_index]
                count_min[pixel, rod_index] = min(
                    count_min[pixel, rod_index], inverse_count[rod_index]
                )
                count_max[pixel, rod_index] = max(
                    count_max[pixel, rod_index], inverse_count[rod_index]
                )
                caustic_any[pixel, rod_index] = caustic_any[pixel, rod_index] or caustic[rod_index]
    return (
        mass,
        center_density,
        count_min,
        count_max,
        caustic_any,
        valid_any,
        valid_all,
        center_valid_pixel,
    )


def _compile_detector_projection(
    *,
    detector_zero_lab_m: FloatArray,
    detector_column_step_lab_m: FloatArray,
    detector_row_step_lab_m: FloatArray,
    detector_pixel_area_vector_lab_m2: FloatArray,
    sample_from_lab: FloatArray,
    ray_origin_lab_m: FloatArray,
) -> _DetectorProjection:
    """Compile the one authoritative batched ray-to-native-pixel projection."""

    ray_origins = np.asarray(ray_origin_lab_m, dtype=np.float64)
    if ray_origins.ndim != 2 or ray_origins.shape[1] != 3:
        raise ValueError("ray_origin_lab_m must have shape (state_count, 3)")
    detector_steps_lab_m = np.stack((detector_column_step_lab_m, detector_row_step_lab_m))
    detector_covectors_lab_per_m = detector_steps_lab_m / np.sum(
        detector_steps_lab_m * detector_steps_lab_m,
        axis=1,
        keepdims=True,
    )
    detector_normal_lab = detector_pixel_area_vector_lab_m2 / np.linalg.norm(
        detector_pixel_area_vector_lab_m2
    )
    relative_origin_lab_m = ray_origins - detector_zero_lab_m
    projection = _DetectorProjection(
        np.ascontiguousarray(detector_covectors_lab_per_m @ sample_from_lab.T),
        np.ascontiguousarray(detector_normal_lab @ sample_from_lab.T),
        np.ascontiguousarray(relative_origin_lab_m @ detector_covectors_lab_per_m.T),
        np.ascontiguousarray(relative_origin_lab_m @ detector_normal_lab),
    )
    for value in projection:
        value.setflags(write=False)
    return projection


class CompiledDetectorEvaluator:
    """Python owner for one packed, reusable compiled detector kernel."""

    __slots__ = (
        "_detector_column_row_covectors_sample_per_m",
        "_detector_normal_sample",
        "_detector_shape_rc",
        "_ray_origin_detector_column_row_px",
        "_ray_origin_detector_normal_m",
        "_state",
    )

    def __init__(self, state: CompiledDetectorState, detector_shape_rc: tuple[int, int]) -> None:
        projection = _compile_detector_projection(
            detector_zero_lab_m=state.detector_zero_lab_m,
            detector_column_step_lab_m=state.detector_column_step_lab_m,
            detector_row_step_lab_m=state.detector_row_step_lab_m,
            detector_pixel_area_vector_lab_m2=state.detector_pixel_area_vector_lab_m2,
            sample_from_lab=state.sample_from_lab,
            ray_origin_lab_m=state.ray_origin_lab_m.reshape(1, 3),
        )
        self._state = state
        self._detector_shape_rc = detector_shape_rc
        self._detector_column_row_covectors_sample_per_m = (
            projection.detector_column_row_covectors_sample_per_m
        )
        self._detector_normal_sample = projection.detector_normal_sample
        self._ray_origin_detector_column_row_px = projection.ray_origin_detector_column_row_px[0]
        self._ray_origin_detector_normal_m = float(projection.ray_origin_detector_normal_m[0])

    @property
    def state(self) -> CompiledDetectorState:
        """Return the immutable numeric state for geometry-only rebinding."""

        return self._state

    def _rebind_validated_projection(
        self,
        state: CompiledDetectorState,
        projection: _DetectorProjection,
        *,
        state_index: int,
    ) -> CompiledDetectorEvaluator:
        """Bind one state to an already compiled batched detector projection."""

        rebound = object.__new__(type(self))
        rebound._state = state
        rebound._detector_shape_rc = self._detector_shape_rc
        rebound._detector_column_row_covectors_sample_per_m = (
            projection.detector_column_row_covectors_sample_per_m
        )
        rebound._detector_normal_sample = projection.detector_normal_sample
        rebound._ray_origin_detector_column_row_px = projection.ray_origin_detector_column_row_px[
            state_index
        ]
        rebound._ray_origin_detector_normal_m = float(
            projection.ray_origin_detector_normal_m[state_index]
        )
        return rebound

    def _evaluate_with_root_selector(
        self,
        column_px: NDArray[np.float64],
        row_px: NDArray[np.float64],
        *,
        root_selector: int,
    ) -> tuple[FloatArray, IntArray, BoolArray, BoolArray]:
        column = np.ascontiguousarray(column_px, dtype=np.float64).reshape(-1)
        row = np.ascontiguousarray(row_px, dtype=np.float64).reshape(-1)
        if column.shape != row.shape:
            raise ValueError("compiled detector coordinates must have equal shapes")
        state = self._state
        return _evaluate_points_kernel(
            column,
            row,
            self._detector_shape_rc,
            state.detector_zero_lab_m,
            state.detector_column_step_lab_m,
            state.detector_row_step_lab_m,
            state.detector_pixel_area_vector_lab_m2,
            state.ray_origin_lab_m,
            state.sample_from_lab,
            state.ki_film_sample_Ainv,
            state.internal_k_Ainv,
            state.air_k0_Ainv,
            state.refractive_index,
            state.entrance_amplitude,
            state.incident_decay_Ainv,
            state.film_thickness_A,
            state.source_phase_weight,
            state.sample_from_local,
            state.rod_hk_population,
            state.rod_parallel_local_Ainv,
            state.rod_u_bounds_Ainv,
            state.rod_inverse_constants,
            state.b3_norm_Ainv,
            state.gaussian_sigma_rad,
            state.lorentzian_hwhm_rad,
            state.lorentzian_probability,
            state.atom_fractional_offset,
            state.atom_occupancy_element,
            state.rod_atom_inplane_factor,
            state.u_radial_A2,
            state.u_normal_A2,
            state.f0_parameters,
            state.anomalous_factor_e,
            state.layers,
            state.shared_disorder_epsilon,
            state.normalization_divisor,
            root_selector,
        )

    def evaluate(
        self,
        column_px: NDArray[np.float64],
        row_px: NDArray[np.float64],
        *,
        branch: int,
    ) -> tuple[FloatArray, IntArray, BoolArray, BoolArray]:
        """Evaluate one ordered nonzero-rod root branch."""

        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        return self._evaluate_with_root_selector(
            column_px,
            row_px,
            root_selector=branch,
        )

    def evaluate_all_roots(
        self,
        column_px: NDArray[np.float64],
        row_px: NDArray[np.float64],
    ) -> tuple[FloatArray, IntArray, BoolArray, BoolArray]:
        """Sum both nonzero roots and both m=0 latent inverse preimages."""

        return self._evaluate_with_root_selector(
            column_px,
            row_px,
            root_selector=0,
        )

    def accumulate_latent_pixel_mass(
        self,
        alpha_rad: NDArray[np.float64],
        beta_rad: NDArray[np.float64],
        image_A2: NDArray[np.float64],
        replicate_total_mass_A2: NDArray[np.float64],
        *,
        image_weight_scale: float | None = None,
    ) -> tuple[int, float]:
        """Accumulate sampled root weights into exact native-pixel owners."""

        alpha = np.ascontiguousarray(alpha_rad, dtype=np.float64).reshape(-1)
        beta = np.ascontiguousarray(beta_rad, dtype=np.float64).reshape(-1)
        if alpha.shape != beta.shape or not alpha.size:
            raise ValueError("sampled alpha and beta must have the same nonzero shape")
        if (
            not np.all(np.isfinite(alpha))
            or not np.all(np.isfinite(beta))
            or np.any((alpha < 0.0) | (alpha > np.pi))
            or np.any((beta < 0.0) | (beta >= 2.0 * np.pi))
        ):
            raise ValueError("sampled mosaic coordinates lie outside their canonical domains")
        image = np.asarray(image_A2)
        replicate = np.asarray(replicate_total_mass_A2)
        if (
            image.dtype != np.float64
            or image.shape != (self._detector_shape_rc[0] * self._detector_shape_rc[1],)
            or not image.flags.c_contiguous
            or not image.flags.writeable
        ):
            raise ValueError("image_A2 must be a writable contiguous flat detector array")
        if (
            replicate.dtype != np.float64
            or replicate.shape != alpha.shape
            or not replicate.flags.c_contiguous
            or not replicate.flags.writeable
        ):
            raise ValueError("replicate_total_mass_A2 must be writable and match the draws")
        if image_weight_scale is None:
            weight_scale = 1.0 / alpha.size
        else:
            weight_scale = float(image_weight_scale)
            if not math.isfinite(weight_scale) or weight_scale <= 0.0:
                raise ValueError("image_weight_scale must be finite and positive")
        state = self._state
        forward_state = _ForwardDetectorState(
            self._detector_shape_rc,
            self._detector_column_row_covectors_sample_per_m,
            self._detector_normal_sample,
            self._ray_origin_detector_column_row_px,
            self._ray_origin_detector_normal_m,
            state.ki_film_sample_Ainv,
            state.internal_k_Ainv,
            state.air_k0_Ainv,
            state.refractive_index,
            state.entrance_amplitude.real**2 + state.entrance_amplitude.imag**2,
            state.incident_decay_Ainv,
            state.film_thickness_A,
            state.source_phase_weight,
            state.sample_from_local,
            state.rod_hk_population,
            state.rod_parallel_local_Ainv,
            state.rod_u_bounds_Ainv,
            state.rod_inverse_constants,
            state.b3_norm_Ainv,
            state.atom_fractional_offset,
            state.atom_occupancy_element,
            state.rod_atom_inplane_factor,
            state.u_radial_A2,
            state.u_normal_A2,
            state.f0_parameters,
            state.anomalous_factor_e,
            state.layers,
            state.shared_disorder_epsilon,
            state.normalization_divisor,
        )
        return _accumulate_latent_pixel_mass_kernel(
            alpha,
            beta,
            image,
            replicate,
            weight_scale,
            forward_state,
        )

    def integrate_pixel_boxes(
        self,
        flat_pixel_index: NDArray[np.int64],
        *,
        offset_px: NDArray[np.float64],
        one_dimensional_weight: NDArray[np.float64],
        branch: int,
        include_center_diagnostics: bool,
    ) -> CompiledPixelIntegral:
        """Integrate exact pixel boxes without materializing node-scale fields."""

        flat_index = np.ascontiguousarray(flat_pixel_index, dtype=np.int64).reshape(-1)
        offset = np.ascontiguousarray(offset_px, dtype=np.float64).reshape(-1)
        weight = np.ascontiguousarray(one_dimensional_weight, dtype=np.float64).reshape(-1)
        if offset.size == 0 or weight.shape != offset.shape:
            raise ValueError("pixel offsets and weights must have the same nonzero shape")
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(weight)):
            raise ValueError("pixel offsets and weights must be finite")
        if np.any(weight < 0.0):
            raise ValueError("pixel quadrature weights must be nonnegative")
        rows, columns = self._detector_shape_rc
        if np.any((flat_index < 0) | (flat_index >= rows * columns)):
            raise ValueError("flat_pixel_index lies outside the active detector")
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        state = self._state
        values = _integrate_pixel_boxes_kernel(
            flat_index,
            offset,
            weight,
            bool(include_center_diagnostics),
            self._detector_shape_rc,
            state.detector_zero_lab_m,
            state.detector_column_step_lab_m,
            state.detector_row_step_lab_m,
            state.detector_pixel_area_vector_lab_m2,
            state.ray_origin_lab_m,
            state.sample_from_lab,
            state.ki_film_sample_Ainv,
            state.internal_k_Ainv,
            state.air_k0_Ainv,
            state.refractive_index,
            state.entrance_amplitude,
            state.incident_decay_Ainv,
            state.film_thickness_A,
            state.source_phase_weight,
            state.sample_from_local,
            state.rod_hk_population,
            state.rod_parallel_local_Ainv,
            state.rod_u_bounds_Ainv,
            state.rod_inverse_constants,
            state.b3_norm_Ainv,
            state.gaussian_sigma_rad,
            state.lorentzian_hwhm_rad,
            state.lorentzian_probability,
            state.atom_fractional_offset,
            state.atom_occupancy_element,
            state.rod_atom_inplane_factor,
            state.u_radial_A2,
            state.u_normal_A2,
            state.f0_parameters,
            state.anomalous_factor_e,
            state.layers,
            state.shared_disorder_epsilon,
            state.normalization_divisor,
            branch,
        )
        return CompiledPixelIntegral(*values)


__all__ = [
    "CompiledDetectorEvaluator",
    "CompiledDetectorState",
    "CompiledPixelIntegral",
    "pack_bi2se3_two_h_structure",
]
