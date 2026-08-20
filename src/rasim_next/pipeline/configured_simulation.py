"""Strict YAML boundary and reusable configured detector calculations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from painted_ewald import (
    BraggSpaceConfig,
    ContinuousEwaldCoating,
    MosaicBraggSpace,
    MosaicParameters,
    Rod,
    enumerate_rods_within_ewald_sphere,
    evaluate_infinite_rod_ewald_geometry,
)
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    IncidentSampleBatch,
    MaterialOptics,
    canonical_revision_sha256,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.layer_order import CommensurateLayerOrder
from rasim_next.core.scattering import polarization_model_code
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry import build_incident_states
from rasim_next.geometry.instrument import (
    AxisRotation,
    CompiledInstrument,
    InstrumentConfiguration,
    compile_instrument,
    compose_intrinsic_xy_rotation,
    validate_detector_path_attenuation_declaration,
)
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.materials import (
    CrystalStructure,
    crystal_with_direct_basis,
    material_optics,
    read_crystal,
)
from rasim_next.optics import (
    DETECTOR_PATH_ATTENUATION_MODEL_ID,
    INCIDENT_ILLUMINATED_PATH_MODEL_ID,
)
from rasim_next.pipeline.bragg_space import (
    Bi2X3FiniteStackStrength,
    CifFiniteStackStrength,
    RevisionedStructureStrengthModel,
)
from rasim_next.pipeline.continuous_detector import (
    DetectorCoordinateGeometry,
    DetectorEwaldMeasure,
    DetectorMappedGeometry,
    evaluate_detector_coordinates_geometry,
    map_ewald_geometry_to_detector,
)
from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure
from rasim_next.pipeline.source_averaged_structure import SourceAveragedStructureDetector
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.sampling.source import (
    SOURCE_QUADRATURE_MODEL_ID,
    require_physical_intensity_source_model,
    sample_discrete_gaussian_line_source_rays,
    sample_gaussian_source_rays,
    sample_nominal_mean_geometry_source_ray,
)
from rasim_next.stacking import Parent

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
CONFIGURED_RESULT_SCHEMA_VERSION = "rasim-configured-result-v2"
_MAXIMUM_MACROBIN_COORDINATES_PER_CALL = 1_500_000


def _readonly_float_array(value: Any, shape: tuple[int | None, ...], name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has the wrong shape")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


class _StrictLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate and merged mapping keys."""


def _construct_mapping(
    loader: _StrictLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    if not isinstance(node, MappingNode):
        raise ValueError("YAML mapping expected")
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("YAML merge keys are not supported")
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("YAML mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True, slots=True)
class TransformConfiguration:
    rotation: tuple[tuple[float, float, float], ...]
    translation_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AxisRotationConfiguration:
    axis_lab: tuple[float, float, float]
    angle_deg: float
    pivot_lab_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MaterialConfiguration:
    cif_path: Path
    phase_id: str


def _source_real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _source_real_tuple(value: object, size: int, name: str) -> tuple[float, ...]:
    try:
        entries = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain {size} real numbers") from exc
    if len(entries) != size:
        raise ValueError(f"{name} must contain {size} real numbers")
    return tuple(_source_real_scalar(item, name) for item in entries)


def _source_real_sequence(value: object, name: str) -> tuple[float, ...]:
    try:
        entries = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    return tuple(_source_real_scalar(item, name) for item in entries)


@dataclass(frozen=True, slots=True)
class SourceConfiguration:
    mean_origin_lab_m: tuple[float, float, float]
    mean_direction_lab: tuple[float, float, float]
    transverse_axes_lab: tuple[tuple[float, float, float], ...]
    spatial_sigma_m: tuple[float, float]
    divergence_sigma_rad: tuple[float, float]
    mean_wavelength_A: float
    wavelength_sigma_A: float
    sample_count: int
    seed: int
    polarization_state_id: str
    wavelength_model_id: str = "gaussian.v1"
    line_wavelength_A: tuple[float, ...] = ()
    line_probability: tuple[float, ...] = ()
    common_line_sigma_A: float = 0.0
    position_divergence_correlation: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.wavelength_model_id not in {"gaussian.v1", "discrete_gaussian_lines.v1"}:
            raise ValueError(
                "wavelength_model_id must be gaussian.v1 or discrete_gaussian_lines.v1"
            )
        object.__setattr__(
            self,
            "mean_origin_lab_m",
            _source_real_tuple(self.mean_origin_lab_m, 3, "mean_origin_lab_m"),
        )
        object.__setattr__(
            self,
            "mean_direction_lab",
            _source_real_tuple(self.mean_direction_lab, 3, "mean_direction_lab"),
        )
        try:
            axes = tuple(self.transverse_axes_lab)
        except TypeError as exc:
            raise ValueError("transverse_axes_lab must contain two 3-vectors") from exc
        if len(axes) != 2:
            raise ValueError("transverse_axes_lab must contain two 3-vectors")
        object.__setattr__(
            self,
            "transverse_axes_lab",
            tuple(_source_real_tuple(axis, 3, "transverse_axes_lab") for axis in axes),
        )
        spatial_sigma_m = _source_real_tuple(self.spatial_sigma_m, 2, "spatial_sigma_m")
        divergence_sigma_rad = _source_real_tuple(
            self.divergence_sigma_rad,
            2,
            "divergence_sigma_rad",
        )
        if any(value < 0.0 for value in (*spatial_sigma_m, *divergence_sigma_rad)):
            raise ValueError("source spatial and divergence widths must be nonnegative")
        object.__setattr__(self, "spatial_sigma_m", spatial_sigma_m)
        object.__setattr__(self, "divergence_sigma_rad", divergence_sigma_rad)
        for name in (
            "mean_wavelength_A",
            "wavelength_sigma_A",
            "common_line_sigma_A",
        ):
            object.__setattr__(self, name, _source_real_scalar(getattr(self, name), name))
        object.__setattr__(
            self,
            "line_wavelength_A",
            _source_real_sequence(self.line_wavelength_A, "source line wavelengths"),
        )
        object.__setattr__(
            self,
            "line_probability",
            _source_real_sequence(self.line_probability, "source line probabilities"),
        )
        if self.mean_wavelength_A <= 0.0:
            raise ValueError("mean_wavelength_A must be positive")
        if self.wavelength_sigma_A < 0.0 or self.common_line_sigma_A < 0.0:
            raise ValueError("source wavelength widths must be nonnegative")
        if (
            isinstance(self.sample_count, (bool, np.bool_))
            or not isinstance(self.sample_count, (int, np.integer))
            or self.sample_count <= 0
        ):
            raise ValueError("sample_count must be a positive integer")
        object.__setattr__(self, "sample_count", int(self.sample_count))
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
            or self.seed > 2**64 - 1
        ):
            raise ValueError("seed must be a nonnegative unsigned 64-bit integer")
        object.__setattr__(self, "seed", int(self.seed))
        if not isinstance(self.polarization_state_id, str) or not self.polarization_state_id:
            raise ValueError("polarization_state_id must be a nonempty string")
        correlation = _source_real_tuple(
            self.position_divergence_correlation,
            2,
            "position_divergence_correlation",
        )
        if any(abs(value) >= 1.0 for value in correlation):
            raise ValueError("position_divergence_correlation must lie within (-1, 1)")
        object.__setattr__(self, "position_divergence_correlation", correlation)
        if self.wavelength_model_id == "gaussian.v1":
            if self.line_wavelength_A or self.line_probability or self.common_line_sigma_A != 0.0:
                raise ValueError("Gaussian source must not declare discrete line parameters")
            return
        if self.wavelength_sigma_A != 0.0:
            raise ValueError("wavelength_sigma_A must be zero for discrete_gaussian_lines.v1")
        if len(self.line_wavelength_A) < 2 or len(self.line_probability) != len(
            self.line_wavelength_A
        ):
            raise ValueError(
                "discrete source must declare matching wavelength and probability lines"
            )
        if self.sample_count < len(self.line_wavelength_A):
            raise ValueError("sample_count must be at least the number of source lines")
        if any(value <= 0.0 for value in self.line_wavelength_A):
            raise ValueError("source line wavelengths must be finite and positive")
        if len(set(self.line_wavelength_A)) != len(self.line_wavelength_A):
            raise ValueError("source line wavelengths must be distinct")
        if any(value <= 0.0 for value in self.line_probability):
            raise ValueError("source line probabilities must be finite and positive")
        if not math.isclose(
            math.fsum(self.line_probability),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-15,
        ):
            raise ValueError("source line probabilities must sum to one")
        line_counts = [self.sample_count // len(self.line_probability)] * len(self.line_probability)
        for index in range(self.sample_count % len(line_counts)):
            line_counts[index] += 1
        for probability, line_count in zip(
            self.line_probability,
            line_counts,
            strict=True,
        ):
            row_mass = probability / line_count
            if row_mass <= 0.0 or not math.isclose(
                math.fsum([row_mass] * line_count),
                probability,
                rel_tol=8.0 * np.finfo(np.float64).eps,
                abs_tol=0.0,
            ):
                raise ValueError("source line probability is too small for its allocated rows")
        centroid_A = math.fsum(
            probability * wavelength_A
            for probability, wavelength_A in zip(
                self.line_probability,
                self.line_wavelength_A,
                strict=True,
            )
        )
        if not math.isclose(self.mean_wavelength_A, centroid_A, rel_tol=0.0, abs_tol=2.0e-15):
            raise ValueError("mean_wavelength_A must equal the discrete line centroid")

    @property
    def minimum_physical_sample_count(self) -> int:
        """Return the smallest source ensemble that realizes the declared spectrum."""

        if self.wavelength_model_id == "discrete_gaussian_lines.v1":
            return len(self.line_wavelength_A)
        return 1


def _source_revision_payload(source: SourceConfiguration) -> dict[str, Any]:
    payload = asdict(source)
    if source.wavelength_model_id == "gaussian.v1":
        for name in (
            "wavelength_model_id",
            "line_wavelength_A",
            "line_probability",
            "common_line_sigma_A",
        ):
            payload.pop(name)
    if source.position_divergence_correlation == (0.0, 0.0):
        payload.pop("position_divergence_correlation")
    return payload


@dataclass(frozen=True, slots=True)
class InstrumentInputConfiguration:
    axis_rotations: tuple[AxisRotationConfiguration, ...]
    lab_from_goniometer_zero: TransformConfiguration
    goniometer_from_sample: TransformConfiguration
    sample_from_crystal: TransformConfiguration
    lab_from_detector: TransformConfiguration
    detector_shape_rc: tuple[int, int]
    detector_row_pitch_m: float
    detector_column_pitch_m: float
    detector_reference_coordinate_px: tuple[float, float]
    sample_support_model_id: str
    sample_width_m: float | None
    sample_length_m: float | None
    film_thickness_A: float
    detector_path_medium_id: str = "vacuum_or_helium_unity.v1"
    detector_path_linear_attenuation_m_inv: float = 0.0
    detector_path_wavelength_A: tuple[float, ...] = ()
    detector_path_linear_attenuation_m_inv_by_wavelength: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        path = validate_detector_path_attenuation_declaration(
            self.detector_path_medium_id,
            self.detector_path_linear_attenuation_m_inv,
            self.detector_path_wavelength_A,
            self.detector_path_linear_attenuation_m_inv_by_wavelength,
        )
        object.__setattr__(self, "detector_path_medium_id", path[0])
        object.__setattr__(self, "detector_path_linear_attenuation_m_inv", path[1])
        object.__setattr__(self, "detector_path_wavelength_A", path[2])
        object.__setattr__(
            self,
            "detector_path_linear_attenuation_m_inv_by_wavelength",
            path[3],
        )


@dataclass(frozen=True, slots=True)
class MosaicInputConfiguration:
    gaussian_sigma_deg: float
    lorentzian_hwhm_deg: float
    lorentzian_probability: float
    alpha_panel_count: int
    alpha_gauss_order: int
    azimuth_count: int
    azimuth_phase_deg: float


@dataclass(frozen=True, slots=True)
class StructureFactorConfiguration:
    model_id: str
    normalization: str
    shared_disorder_epsilon: float
    layers: int | None = None
    repeats: int | None = None
    unknown_u_iso_A2: float | None = None


@dataclass(frozen=True, slots=True)
class BraggConfiguration:
    selection_model: str
    rod_population: float
    include_detector_visible_m0: bool
    root_policy: str


@dataclass(frozen=True, slots=True)
class WeightConfiguration:
    phase_population: float
    polarization: float


@dataclass(frozen=True, slots=True)
class NumericalConfiguration:
    worker_count: int
    detector_macrobin_size_px: int
    detector_gauss_order: int
    reciprocal_alpha_count: int
    reciprocal_beta_count: int
    reciprocal_u_count: int
    reciprocal_alpha_max_deg: float
    ewald_alpha_count: int
    ewald_beta_count: int
    ewald_alpha_max_deg: float
    detector_execution_backend: str = "cpu"


@dataclass(frozen=True, slots=True)
class ArtifactConfiguration:
    enabled: bool
    filename: str


@dataclass(frozen=True, slots=True)
class SimulationOutputConfiguration:
    reciprocal_space: ArtifactConfiguration
    ewald_surface: ArtifactConfiguration
    detector: ArtifactConfiguration


@dataclass(frozen=True, slots=True)
class SimulationConfiguration:
    """Fully validated immutable simulation and rendering declaration."""

    schema_version: str
    config_path: Path
    material: MaterialConfiguration
    source: SourceConfiguration
    instrument: InstrumentInputConfiguration
    mosaic: MosaicInputConfiguration
    structure_factor: StructureFactorConfiguration
    bragg: BraggConfiguration
    weights: WeightConfiguration
    numerics: NumericalConfiguration
    output_directory: Path
    outputs: SimulationOutputConfiguration
    cif_sha256: str = field(init=False)
    physics_revision: str = field(init=False)
    render_revision: str = field(init=False)

    def __post_init__(self) -> None:
        detector_path_wavelengths = self.instrument.detector_path_wavelength_A
        if detector_path_wavelengths:
            if self.source.wavelength_model_id == "gaussian.v1":
                if self.source.wavelength_sigma_A != 0.0:
                    raise ValueError(
                        "exact detector-path attenuation tables require a zero-width source"
                    )
                required_wavelengths = (self.source.mean_wavelength_A,)
            else:
                if self.source.common_line_sigma_A != 0.0:
                    raise ValueError(
                        "exact detector-path attenuation tables require zero-width source lines"
                    )
                required_wavelengths = (
                    *self.source.line_wavelength_A,
                    self.source.mean_wavelength_A,
                )
            if any(value not in detector_path_wavelengths for value in required_wavelengths):
                raise ValueError(
                    "detector-path attenuation table must contain every source line and the "
                    "nominal mean wavelength"
                )
        cif_sha256 = hashlib.sha256(self.material.cif_path.read_bytes()).hexdigest()
        payload = {
            "schema_version": self.schema_version,
            "cif_sha256": cif_sha256,
            "material_phase_id": self.material.phase_id,
            "source": _source_revision_payload(self.source),
            "instrument": asdict(self.instrument),
            "mosaic": asdict(self.mosaic),
            "structure_factor": asdict(self.structure_factor),
            "bragg": asdict(self.bragg),
            "weights": asdict(self.weights),
            "source_quadrature_model_id": SOURCE_QUADRATURE_MODEL_ID,
            "incident_illuminated_path_model_id": INCIDENT_ILLUMINATED_PATH_MODEL_ID,
            "detector_path_attenuation_model_id": DETECTOR_PATH_ATTENUATION_MODEL_ID,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        physics_revision = hashlib.sha256(encoded).hexdigest()
        render_payload = {
            "physics_revision": physics_revision,
            "numerics": asdict(self.numerics),
            "outputs": asdict(self.outputs),
        }
        render_encoded = json.dumps(
            render_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "cif_sha256", cif_sha256)
        object.__setattr__(self, "physics_revision", physics_revision)
        object.__setattr__(self, "render_revision", hashlib.sha256(render_encoded).hexdigest())

    @property
    def enabled_artifact_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("reciprocal_space", "ewald_surface", "detector")
            if getattr(self.outputs, name).enabled
        )


def _mapping(
    value: Any,
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - (optional or set())
    if missing:
        raise ValueError(f"{path}: missing key {sorted(missing)[0]!r}")
    if unknown:
        raise ValueError(f"{path}: unknown key {sorted(unknown)[0]!r}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _nonnegative(value: Any, path: str) -> float:
    result = _finite(value, path)
    if result < 0.0:
        raise ValueError(f"{path} must be nonnegative")
    return result


def _positive(value: Any, path: str) -> float:
    result = _finite(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _integer(value: Any, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if positive and value < 1:
        raise ValueError(f"{path} must be positive")
    return value


def _vector(value: Any, path: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{path} must contain {length} numbers")
    return tuple(_finite(item, f"{path}[{index}]") for index, item in enumerate(value))


def _nonnegative_vector(value: Any, path: str, length: int) -> tuple[float, ...]:
    values = _vector(value, path, length)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{path} entries must be nonnegative")
    return values


def _number_sequence(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a sequence")
    parser = _positive if positive else _nonnegative if nonnegative else _finite
    return tuple(parser(item, f"{path}[{index}]") for index, item in enumerate(value))


def _matrix(value: Any, path: str, rows: int, columns: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{path} must contain {rows} rows")
    return tuple(_vector(row, f"{path}[{index}]", columns) for index, row in enumerate(value))


def _transform(value: Any, path: str) -> TransformConfiguration:
    data = _mapping(value, path, required={"rotation", "translation_m"})
    return TransformConfiguration(
        rotation=_matrix(data["rotation"], f"{path}.rotation", 3, 3),
        translation_m=_vector(data["translation_m"], f"{path}.translation_m", 3),
    )


def _detector_transform(value: Any, tilt_value: Any | None) -> TransformConfiguration:
    """Fold intrinsic detector-axis tilts into the one canonical detector pose."""

    transform = _transform(value, "instrument.lab_from_detector")
    if tilt_value is None:
        return transform
    tilt = _mapping(
        tilt_value,
        "instrument.detector_tilt",
        required={"about_column_axis_deg", "about_row_axis_deg"},
    )
    column_deg = _finite(
        tilt["about_column_axis_deg"],
        "instrument.detector_tilt.about_column_axis_deg",
    )
    row_deg = _finite(
        tilt["about_row_axis_deg"],
        "instrument.detector_tilt.about_row_axis_deg",
    )
    if column_deg == 0.0 and row_deg == 0.0:
        return transform

    rotation = compose_intrinsic_xy_rotation(
        np.asarray(transform.rotation),
        math.radians(column_deg),
        math.radians(row_deg),
    )
    return TransformConfiguration(
        rotation=tuple(tuple(float(entry) for entry in row) for row in rotation),
        translation_m=transform.translation_m,
    )


def _artifact(value: Any, path: str) -> ArtifactConfiguration:
    data = _mapping(value, path, required={"enabled", "filename"})
    filename = _string(data["filename"], f"{path}.filename")
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".png":
        raise ValueError(f"{path}.filename must be one PNG basename")
    return ArtifactConfiguration(
        enabled=_boolean(data["enabled"], f"{path}.enabled"),
        filename=filename,
    )


def load_strict_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load exactly one alias-free YAML mapping with duplicate-key rejection."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if any(isinstance(event, AliasEvent) for event in yaml.parse(text)):
        raise ValueError("YAML aliases are not supported")
    documents = list(yaml.load_all(text, Loader=_StrictLoader))
    if len(documents) != 1:
        raise ValueError("configuration must contain exactly one YAML document")
    if not isinstance(documents[0], dict):
        raise ValueError("configuration root must be a mapping")
    return documents[0]


def load_simulation_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> SimulationConfiguration:
    """Load one strict, config-relative ``rasim-simulation-v2`` document."""

    config_path = Path(path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    document = _mapping(
        load_strict_yaml_mapping(config_path),
        "configuration",
        required={
            "schema_version",
            "material",
            "source",
            "instrument",
            "mosaic",
            "structure_factor",
            "bragg",
            "weights",
            "numerics",
            "outputs",
        },
    )
    schema = _string(document["schema_version"], "schema_version")
    if schema != "rasim-simulation-v2":
        raise ValueError("schema_version must be rasim-simulation-v2")

    material_data = _mapping(
        document["material"],
        "material",
        required={"cif_path", "phase_id"},
    )
    cif_path = (
        config_path.parent / _string(material_data["cif_path"], "material.cif_path")
    ).resolve()
    if not cif_path.is_file():
        raise ValueError(f"material.cif_path does not exist: {cif_path}")
    material = MaterialConfiguration(
        cif_path=cif_path,
        phase_id=_string(material_data["phase_id"], "material.phase_id"),
    )

    source_data = _mapping(
        document["source"],
        "source",
        required={
            "mean_origin_lab_m",
            "mean_direction_lab",
            "transverse_axes_lab",
            "spatial_sigma_m",
            "divergence_sigma_rad",
            "mean_wavelength_A",
            "wavelength_sigma_A",
            "sample_count",
            "seed",
            "polarization_state_id",
        },
        optional={
            "wavelength_model_id",
            "line_wavelength_A",
            "line_probability",
            "common_line_sigma_A",
            "position_divergence_correlation",
        },
    )
    wavelength_model_id = _string(
        source_data.get("wavelength_model_id", "gaussian.v1"),
        "source.wavelength_model_id",
    )
    if wavelength_model_id not in {"gaussian.v1", "discrete_gaussian_lines.v1"}:
        raise ValueError(
            "source.wavelength_model_id must be gaussian.v1 or discrete_gaussian_lines.v1"
        )
    raw_line_wavelength = source_data.get("line_wavelength_A", [])
    raw_line_probability = source_data.get("line_probability", [])
    if not isinstance(raw_line_wavelength, list) or not isinstance(raw_line_probability, list):
        raise ValueError("source line wavelengths and probabilities must be sequences")
    line_wavelength_A = tuple(
        _positive(value, f"source.line_wavelength_A[{index}]")
        for index, value in enumerate(raw_line_wavelength)
    )
    line_probability = tuple(
        _positive(value, f"source.line_probability[{index}]")
        for index, value in enumerate(raw_line_probability)
    )
    if wavelength_model_id == "gaussian.v1":
        if line_wavelength_A or line_probability or "common_line_sigma_A" in source_data:
            raise ValueError("Gaussian source must not declare discrete line parameters")
    else:
        if len(line_wavelength_A) < 2 or len(line_probability) != len(line_wavelength_A):
            raise ValueError(
                "discrete source must declare matching wavelength and probability lines"
            )
        if len(set(line_wavelength_A)) != len(line_wavelength_A):
            raise ValueError("source line wavelengths must be distinct")
        if not math.isclose(sum(line_probability), 1.0, rel_tol=0.0, abs_tol=2.0e-15):
            raise ValueError("source line probabilities must sum to one")
    source = SourceConfiguration(
        mean_origin_lab_m=_vector(source_data["mean_origin_lab_m"], "source.mean_origin_lab_m", 3),
        mean_direction_lab=_vector(
            source_data["mean_direction_lab"], "source.mean_direction_lab", 3
        ),
        transverse_axes_lab=_matrix(
            source_data["transverse_axes_lab"], "source.transverse_axes_lab", 2, 3
        ),
        spatial_sigma_m=_nonnegative_vector(
            source_data["spatial_sigma_m"], "source.spatial_sigma_m", 2
        ),
        divergence_sigma_rad=_nonnegative_vector(
            source_data["divergence_sigma_rad"], "source.divergence_sigma_rad", 2
        ),
        mean_wavelength_A=_positive(source_data["mean_wavelength_A"], "source.mean_wavelength_A"),
        wavelength_sigma_A=_nonnegative(
            source_data["wavelength_sigma_A"], "source.wavelength_sigma_A"
        ),
        sample_count=_integer(source_data["sample_count"], "source.sample_count", positive=True),
        seed=_integer(source_data["seed"], "source.seed"),
        polarization_state_id=_string(
            source_data["polarization_state_id"], "source.polarization_state_id"
        ),
        wavelength_model_id=wavelength_model_id,
        line_wavelength_A=line_wavelength_A,
        line_probability=line_probability,
        common_line_sigma_A=_nonnegative(
            source_data.get("common_line_sigma_A", 0.0),
            "source.common_line_sigma_A",
        ),
        position_divergence_correlation=_vector(
            source_data.get("position_divergence_correlation", [0.0, 0.0]),
            "source.position_divergence_correlation",
            2,
        ),
    )
    polarization_model_code(source.polarization_state_id)
    if source.seed < 0:
        raise ValueError("source.seed must be nonnegative")
    if source.wavelength_model_id == "discrete_gaussian_lines.v1" and source.wavelength_sigma_A:
        raise ValueError("source.wavelength_sigma_A must be zero for discrete_gaussian_lines.v1")
    if any(abs(value) >= 1.0 for value in source.position_divergence_correlation):
        raise ValueError("source.position_divergence_correlation must lie within (-1, 1)")

    instrument_data = _mapping(
        document["instrument"],
        "instrument",
        required={
            "axis_rotations",
            "lab_from_goniometer_zero",
            "goniometer_from_sample",
            "sample_from_crystal",
            "lab_from_detector",
            "detector_shape_rc",
            "detector_row_pitch_m",
            "detector_column_pitch_m",
            "detector_reference_coordinate_px",
            "sample_support_model_id",
            "sample_width_m",
            "sample_length_m",
            "film_thickness_A",
        },
        optional={
            "detector_tilt",
            "detector_path_medium_id",
            "detector_path_linear_attenuation_m_inv",
            "detector_path_wavelength_A",
            "detector_path_linear_attenuation_m_inv_by_wavelength",
        },
    )
    raw_rotations = instrument_data["axis_rotations"]
    if not isinstance(raw_rotations, list):
        raise ValueError("instrument.axis_rotations must be a sequence")
    rotations = []
    for position, raw_rotation in enumerate(raw_rotations):
        path_prefix = f"instrument.axis_rotations[{position}]"
        item = _mapping(
            raw_rotation,
            path_prefix,
            required={"axis_lab", "angle_deg", "pivot_lab_m"},
        )
        rotations.append(
            AxisRotationConfiguration(
                axis_lab=_vector(item["axis_lab"], f"{path_prefix}.axis_lab", 3),
                angle_deg=_finite(item["angle_deg"], f"{path_prefix}.angle_deg"),
                pivot_lab_m=_vector(item["pivot_lab_m"], f"{path_prefix}.pivot_lab_m", 3),
            )
        )
    detector_shape = instrument_data["detector_shape_rc"]
    if not isinstance(detector_shape, list) or len(detector_shape) != 2:
        raise ValueError("instrument.detector_shape_rc must contain row and column counts")
    optional_widths: dict[str, float | None] = {}
    for name in ("sample_width_m", "sample_length_m"):
        supplied = instrument_data[name]
        optional_widths[name] = (
            None if supplied is None else _positive(supplied, f"instrument.{name}")
        )
    instrument = InstrumentInputConfiguration(
        axis_rotations=tuple(rotations),
        lab_from_goniometer_zero=_transform(
            instrument_data["lab_from_goniometer_zero"], "instrument.lab_from_goniometer_zero"
        ),
        goniometer_from_sample=_transform(
            instrument_data["goniometer_from_sample"], "instrument.goniometer_from_sample"
        ),
        sample_from_crystal=_transform(
            instrument_data["sample_from_crystal"], "instrument.sample_from_crystal"
        ),
        lab_from_detector=_detector_transform(
            instrument_data["lab_from_detector"],
            instrument_data.get("detector_tilt"),
        ),
        detector_shape_rc=(
            _integer(detector_shape[0], "instrument.detector_shape_rc[0]", positive=True),
            _integer(detector_shape[1], "instrument.detector_shape_rc[1]", positive=True),
        ),
        detector_row_pitch_m=_positive(
            instrument_data["detector_row_pitch_m"], "instrument.detector_row_pitch_m"
        ),
        detector_column_pitch_m=_positive(
            instrument_data["detector_column_pitch_m"], "instrument.detector_column_pitch_m"
        ),
        detector_reference_coordinate_px=_vector(
            instrument_data["detector_reference_coordinate_px"],
            "instrument.detector_reference_coordinate_px",
            2,
        ),
        sample_support_model_id=_string(
            instrument_data["sample_support_model_id"], "instrument.sample_support_model_id"
        ),
        sample_width_m=optional_widths["sample_width_m"],
        sample_length_m=optional_widths["sample_length_m"],
        film_thickness_A=_nonnegative(
            instrument_data["film_thickness_A"], "instrument.film_thickness_A"
        ),
        detector_path_medium_id=_string(
            instrument_data.get("detector_path_medium_id", "vacuum_or_helium_unity.v1"),
            "instrument.detector_path_medium_id",
        ),
        detector_path_linear_attenuation_m_inv=_nonnegative(
            instrument_data.get("detector_path_linear_attenuation_m_inv", 0.0),
            "instrument.detector_path_linear_attenuation_m_inv",
        ),
        detector_path_wavelength_A=_number_sequence(
            instrument_data.get("detector_path_wavelength_A", []),
            "instrument.detector_path_wavelength_A",
            positive=True,
        ),
        detector_path_linear_attenuation_m_inv_by_wavelength=_number_sequence(
            instrument_data.get(
                "detector_path_linear_attenuation_m_inv_by_wavelength",
                [],
            ),
            "instrument.detector_path_linear_attenuation_m_inv_by_wavelength",
            nonnegative=True,
        ),
    )

    mosaic_data = _mapping(
        document["mosaic"],
        "mosaic",
        required={
            "gaussian_sigma_deg",
            "lorentzian_hwhm_deg",
            "lorentzian_probability",
            "alpha_panel_count",
            "alpha_gauss_order",
            "azimuth_count",
            "azimuth_phase_deg",
        },
    )
    mosaic = MosaicInputConfiguration(
        gaussian_sigma_deg=_nonnegative(
            mosaic_data["gaussian_sigma_deg"], "mosaic.gaussian_sigma_deg"
        ),
        lorentzian_hwhm_deg=_nonnegative(
            mosaic_data["lorentzian_hwhm_deg"], "mosaic.lorentzian_hwhm_deg"
        ),
        lorentzian_probability=_finite(
            mosaic_data["lorentzian_probability"], "mosaic.lorentzian_probability"
        ),
        alpha_panel_count=_integer(
            mosaic_data["alpha_panel_count"], "mosaic.alpha_panel_count", positive=True
        ),
        alpha_gauss_order=_integer(
            mosaic_data["alpha_gauss_order"], "mosaic.alpha_gauss_order", positive=True
        ),
        azimuth_count=_integer(mosaic_data["azimuth_count"], "mosaic.azimuth_count", positive=True),
        azimuth_phase_deg=_finite(mosaic_data["azimuth_phase_deg"], "mosaic.azimuth_phase_deg"),
    )
    if not 0.0 <= mosaic.lorentzian_probability <= 1.0:
        raise ValueError("mosaic.lorentzian_probability must lie in [0, 1]")
    sf_data = _mapping(
        document["structure_factor"],
        "structure_factor",
        required={"model_id", "normalization"},
        optional={"layers", "repeats", "shared_disorder_epsilon", "unknown_u_iso_A2"},
    )
    structure_model_id = _string(sf_data["model_id"], "structure_factor.model_id")
    generic_cif_model = structure_model_id == "cif_conventional_cell_finite_repeat.v1"
    count_key = "repeats" if generic_cif_model else "layers"
    forbidden_count_key = "layers" if generic_cif_model else "repeats"
    if count_key not in sf_data:
        raise ValueError(f"structure_factor: missing key {count_key!r} for the selected model")
    if forbidden_count_key in sf_data:
        raise ValueError(
            f"structure_factor.{forbidden_count_key} does not apply to {structure_model_id}"
        )
    if (
        structure_model_id != "cif_conventional_cell_finite_repeat.v1"
        and "shared_disorder_epsilon" not in sf_data
    ):
        raise ValueError(
            "structure_factor: missing key 'shared_disorder_epsilon' for the stacking model"
        )
    structure_factor = StructureFactorConfiguration(
        model_id=structure_model_id,
        normalization=_string(sf_data["normalization"], "structure_factor.normalization"),
        shared_disorder_epsilon=_finite(
            sf_data.get("shared_disorder_epsilon", 0.0),
            "structure_factor.shared_disorder_epsilon",
        ),
        layers=(
            None
            if generic_cif_model
            else _integer(sf_data["layers"], "structure_factor.layers", positive=True)
        ),
        repeats=(
            _integer(sf_data["repeats"], "structure_factor.repeats", positive=True)
            if generic_cif_model
            else None
        ),
        unknown_u_iso_A2=(
            None
            if sf_data.get("unknown_u_iso_A2") is None
            else _nonnegative(
                sf_data["unknown_u_iso_A2"],
                "structure_factor.unknown_u_iso_A2",
            )
        ),
    )
    bragg_data = _mapping(
        document["bragg"],
        "bragg",
        required={
            "selection_model",
            "rod_population",
            "include_detector_visible_m0",
            "root_policy",
        },
    )
    bragg = BraggConfiguration(
        selection_model=_string(bragg_data["selection_model"], "bragg.selection_model"),
        rod_population=_nonnegative(bragg_data["rod_population"], "bragg.rod_population"),
        include_detector_visible_m0=_boolean(
            bragg_data["include_detector_visible_m0"], "bragg.include_detector_visible_m0"
        ),
        root_policy=_string(bragg_data["root_policy"], "bragg.root_policy"),
    )
    if bragg.selection_model != "all_elastic_reachable.v1":
        raise ValueError("bragg.selection_model must be all_elastic_reachable.v1")
    if bragg.root_policy != "all_retained_roots.v1":
        raise ValueError("bragg.root_policy must be all_retained_roots.v1")

    weights_data = _mapping(
        document["weights"],
        "weights",
        required={"phase_population", "polarization"},
    )
    weights = WeightConfiguration(
        phase_population=_nonnegative(weights_data["phase_population"], "weights.phase_population"),
        polarization=_nonnegative(weights_data["polarization"], "weights.polarization"),
    )

    numerical_data = _mapping(
        document["numerics"],
        "numerics",
        required={
            "worker_count",
            "detector_execution_backend",
            "detector_macrobin_size_px",
            "detector_gauss_order",
            "reciprocal_alpha_count",
            "reciprocal_beta_count",
            "reciprocal_u_count",
            "reciprocal_alpha_max_deg",
            "ewald_alpha_count",
            "ewald_beta_count",
            "ewald_alpha_max_deg",
        },
        optional=set(),
    )
    numerics = NumericalConfiguration(
        worker_count=_integer(
            numerical_data["worker_count"], "numerics.worker_count", positive=True
        ),
        detector_execution_backend=_string(
            numerical_data["detector_execution_backend"],
            "numerics.detector_execution_backend",
        ),
        detector_macrobin_size_px=_integer(
            numerical_data["detector_macrobin_size_px"],
            "numerics.detector_macrobin_size_px",
            positive=True,
        ),
        detector_gauss_order=_integer(
            numerical_data["detector_gauss_order"], "numerics.detector_gauss_order", positive=True
        ),
        reciprocal_alpha_count=_integer(
            numerical_data["reciprocal_alpha_count"],
            "numerics.reciprocal_alpha_count",
            positive=True,
        ),
        reciprocal_beta_count=_integer(
            numerical_data["reciprocal_beta_count"], "numerics.reciprocal_beta_count", positive=True
        ),
        reciprocal_u_count=_integer(
            numerical_data["reciprocal_u_count"], "numerics.reciprocal_u_count", positive=True
        ),
        reciprocal_alpha_max_deg=_positive(
            numerical_data["reciprocal_alpha_max_deg"], "numerics.reciprocal_alpha_max_deg"
        ),
        ewald_alpha_count=_integer(
            numerical_data["ewald_alpha_count"], "numerics.ewald_alpha_count", positive=True
        ),
        ewald_beta_count=_integer(
            numerical_data["ewald_beta_count"], "numerics.ewald_beta_count", positive=True
        ),
        ewald_alpha_max_deg=_positive(
            numerical_data["ewald_alpha_max_deg"], "numerics.ewald_alpha_max_deg"
        ),
    )
    if numerics.detector_execution_backend not in {"cpu", "cuda"}:
        raise ValueError("numerics.detector_execution_backend must be cpu or cuda")
    if numerics.reciprocal_alpha_max_deg > 180.0:
        raise ValueError("numerics.reciprocal_alpha_max_deg must not exceed 180 degrees")
    if numerics.ewald_alpha_max_deg > 180.0:
        raise ValueError("numerics.ewald_alpha_max_deg must not exceed 180 degrees")
    if numerics.detector_gauss_order % 2:
        raise ValueError("numerics.detector_gauss_order must be even to avoid center caustics")
    rows, columns = instrument.detector_shape_rc
    if rows % numerics.detector_macrobin_size_px or columns % numerics.detector_macrobin_size_px:
        raise ValueError("detector_macrobin_size_px must divide both detector dimensions")

    outputs_data = _mapping(
        document["outputs"],
        "outputs",
        required={"output_directory", "reciprocal_space", "ewald_surface", "detector"},
    )
    configured_output = Path(
        _string(outputs_data["output_directory"], "outputs.output_directory")
    ).expanduser()
    output_directory = (
        configured_output
        if configured_output.is_absolute()
        else config_path.parent / configured_output
    ).resolve()
    if output_directory == root or output_directory.is_relative_to(root):
        raise ValueError("outputs.output_directory must be outside the repository")
    outputs = SimulationOutputConfiguration(
        reciprocal_space=_artifact(outputs_data["reciprocal_space"], "outputs.reciprocal_space"),
        ewald_surface=_artifact(outputs_data["ewald_surface"], "outputs.ewald_surface"),
        detector=_artifact(outputs_data["detector"], "outputs.detector"),
    )
    enabled_filenames = [
        artifact.filename.casefold()
        for artifact in (
            outputs.reciprocal_space,
            outputs.ewald_surface,
            outputs.detector,
        )
        if artifact.enabled
    ]
    if len(set(enabled_filenames)) != len(enabled_filenames):
        raise ValueError("enabled output filenames must be unique")
    if not any(
        getattr(outputs, name).enabled for name in ("reciprocal_space", "ewald_surface", "detector")
    ):
        raise ValueError("at least one output must be enabled")

    return SimulationConfiguration(
        schema_version=schema,
        config_path=config_path,
        material=material,
        source=source,
        instrument=instrument,
        mosaic=mosaic,
        structure_factor=structure_factor,
        bragg=bragg,
        weights=weights,
        numerics=numerics,
        output_directory=output_directory,
        outputs=outputs,
    )


def _rigid_transform(
    configured: TransformConfiguration,
    source_frame: FrameId,
    target_frame: FrameId,
) -> RigidTransform:
    return RigidTransform(
        np.asarray(configured.rotation, dtype=np.float64),
        np.asarray(configured.translation_m, dtype=np.float64),
        source_frame,
        target_frame,
    )


def sample_configured_source(
    source: SourceConfiguration,
    *,
    sample_count: int | None = None,
) -> IncidentSampleBatch:
    count = source.sample_count if sample_count is None else sample_count
    if source.wavelength_model_id == "discrete_gaussian_lines.v1":
        return sample_discrete_gaussian_line_source_rays(
            mean_origin_lab_m=np.asarray(source.mean_origin_lab_m),
            mean_direction_lab=np.asarray(source.mean_direction_lab),
            transverse_axes_lab=np.asarray(source.transverse_axes_lab),
            spatial_sigma_m=np.asarray(source.spatial_sigma_m),
            divergence_sigma_rad=np.asarray(source.divergence_sigma_rad),
            line_wavelength_A=np.asarray(source.line_wavelength_A),
            line_probability=np.asarray(source.line_probability),
            common_wavelength_sigma_A=source.common_line_sigma_A,
            sample_count=count,
            seed=source.seed,
            polarization_state_id=source.polarization_state_id,
            position_divergence_correlation=source.position_divergence_correlation,
        )
    if source.wavelength_model_id != "gaussian.v1":
        raise ValueError(f"unsupported wavelength_model_id {source.wavelength_model_id!r}")
    return sample_gaussian_source_rays(
        mean_origin_lab_m=np.asarray(source.mean_origin_lab_m),
        mean_direction_lab=np.asarray(source.mean_direction_lab),
        transverse_axes_lab=np.asarray(source.transverse_axes_lab),
        spatial_sigma_m=np.asarray(source.spatial_sigma_m),
        divergence_sigma_rad=np.asarray(source.divergence_sigma_rad),
        mean_wavelength_A=source.mean_wavelength_A,
        wavelength_sigma_A=source.wavelength_sigma_A,
        sample_count=count,
        seed=source.seed,
        polarization_state_id=source.polarization_state_id,
        position_divergence_correlation=source.position_divergence_correlation,
    )


def sample_configured_nominal_geometry_source(
    source: SourceConfiguration,
) -> IncidentSampleBatch:
    """Build the explicit centroid companion used by geometry and nominal Ewald tags."""

    return sample_nominal_mean_geometry_source_ray(
        mean_origin_lab_m=np.asarray(source.mean_origin_lab_m),
        mean_direction_lab=np.asarray(source.mean_direction_lab),
        transverse_axes_lab=np.asarray(source.transverse_axes_lab),
        spatial_sigma_m=np.asarray(source.spatial_sigma_m),
        divergence_sigma_rad=np.asarray(source.divergence_sigma_rad),
        reference_wavelength_A=source.mean_wavelength_A,
        polarization_state_id=source.polarization_state_id,
        position_divergence_correlation=source.position_divergence_correlation,
    )


def _compile_instrument(configured: InstrumentInputConfiguration) -> CompiledInstrument:
    return compile_instrument(
        InstrumentConfiguration(
            axis_rotations=tuple(
                AxisRotation(
                    axis_lab=np.asarray(rotation.axis_lab),
                    angle_rad=math.radians(rotation.angle_deg),
                    pivot_lab_m=np.asarray(rotation.pivot_lab_m),
                )
                for rotation in configured.axis_rotations
            ),
            lab_from_goniometer_zero=_rigid_transform(
                configured.lab_from_goniometer_zero,
                FrameId.GONIOMETER,
                FrameId.LAB,
            ),
            goniometer_from_sample=_rigid_transform(
                configured.goniometer_from_sample,
                FrameId.SAMPLE,
                FrameId.GONIOMETER,
            ),
            sample_from_crystal=_rigid_transform(
                configured.sample_from_crystal,
                FrameId.CRYSTAL,
                FrameId.SAMPLE,
            ),
            lab_from_detector=_rigid_transform(
                configured.lab_from_detector,
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
            detector_shape_rc=configured.detector_shape_rc,
            detector_row_pitch_m=configured.detector_row_pitch_m,
            detector_column_pitch_m=configured.detector_column_pitch_m,
            detector_reference_coordinate_px=configured.detector_reference_coordinate_px,
            sample_support_model_id=configured.sample_support_model_id,
            sample_width_m=configured.sample_width_m,
            sample_length_m=configured.sample_length_m,
            film_thickness_A=configured.film_thickness_A,
            detector_path_medium_id=configured.detector_path_medium_id,
            detector_path_linear_attenuation_m_inv=(
                configured.detector_path_linear_attenuation_m_inv
            ),
            detector_path_wavelength_A=configured.detector_path_wavelength_A,
            detector_path_linear_attenuation_m_inv_by_wavelength=(
                configured.detector_path_linear_attenuation_m_inv_by_wavelength
            ),
        )
    )


def _mosaic(configured: MosaicInputConfiguration) -> MosaicParameters:
    return MosaicParameters(
        gaussian_sigma_rad=math.radians(configured.gaussian_sigma_deg),
        lorentzian_half_width_rad=math.radians(configured.lorentzian_hwhm_deg),
        lorentzian_probability=configured.lorentzian_probability,
        alpha_panel_count=configured.alpha_panel_count,
        alpha_gauss_order=configured.alpha_gauss_order,
        azimuth_count=configured.azimuth_count,
        azimuth_phase_rad=math.radians(configured.azimuth_phase_deg),
    )


@dataclass(frozen=True, slots=True)
class ConfiguredGeometryInputs:
    """Nominal exact-tag state with no intensity or mosaic numerical objects."""

    config: SimulationConfiguration
    samples: IncidentSampleBatch
    instrument: CompiledInstrument
    crystal: CrystalStructure
    material: MaterialOptics
    reciprocal: ReciprocalLattice
    rods: tuple[Rod, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, SimulationConfiguration):
            raise TypeError("config must be SimulationConfiguration")
        if not isinstance(self.samples, IncidentSampleBatch):
            raise TypeError("samples must be IncidentSampleBatch")
        if self.samples.incident_sample_id.size != 1:
            raise ValueError("configured geometry requires exactly one source sample")
        expected_samples = sample_configured_nominal_geometry_source(self.config.source)
        if self.samples.source_revision != expected_samples.source_revision:
            raise ValueError(
                "configured geometry sample must be the source-center, zero-divergence, "
                "mean-wavelength companion state"
            )
        if not isinstance(self.instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        if not isinstance(self.crystal, CrystalStructure):
            raise TypeError("crystal must be CrystalStructure")
        if not isinstance(self.material, MaterialOptics):
            raise TypeError("material must be MaterialOptics")
        if not isinstance(self.reciprocal, ReciprocalLattice):
            raise TypeError("reciprocal must be ReciprocalLattice")
        rods = tuple(self.rods)
        if not rods or any(not isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat physical (h, k) lines")
        if not np.array_equal(self.material.wavelength_A, self.samples.wavelength_A):
            raise ValueError("geometry material wavelengths must match the source sample")
        object.__setattr__(self, "rods", rods)


@dataclass(frozen=True, slots=True)
class GeometryOnlyEwaldContext:
    """Nominal one-ray Ewald geometry with no strength or mosaic numerical state."""

    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    rods: tuple[Rod, ...]
    incident: IncidentTransportResult
    material: MaterialOptics
    instrument: CompiledInstrument

    def __post_init__(self) -> None:
        basis = _readonly_float_array(
            self.reciprocal_basis_Ainv,
            (3, 3),
            "reciprocal_basis_Ainv",
        )
        crystal_to_sample = _readonly_float_array(
            self.crystal_to_sample,
            (3, 3),
            "crystal_to_sample",
        )
        if not np.allclose(
            crystal_to_sample.T @ crystal_to_sample,
            np.eye(3),
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.isclose(np.linalg.det(crystal_to_sample), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("crystal_to_sample must be a proper rotation")
        rods = tuple(self.rods)
        if not rods or any(not isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if not isinstance(self.incident, IncidentTransportResult):
            raise TypeError("incident must be IncidentTransportResult")
        if not isinstance(self.material, MaterialOptics):
            raise TypeError("material must be MaterialOptics")
        if not isinstance(self.instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        states = self.incident.states
        if states.incident_state_id.size != 1 or not bool(states.valid[0]):
            raise ValueError("geometry-only Ewald context requires one valid incident state")
        if states.sample_geometry_revision != self.instrument.sample_geometry_revision:
            raise ValueError("incident and instrument sample revisions disagree")
        if states.material_revision != self.material.material_revision:
            raise ValueError("incident and material revisions disagree")
        if not np.array_equal(crystal_to_sample, self.instrument.sample_from_crystal.rotation):
            raise ValueError("crystal_to_sample must match the configured instrument")
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "crystal_to_sample", crystal_to_sample)
        object.__setattr__(self, "rods", rods)

    @property
    def ki_sample_Ainv(self) -> FloatArray:
        return self.incident.states.k_film_phase_sample_Ainv[0]

    def map_latent_geometry(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> DetectorMappedGeometry:
        configured = next(
            (candidate for candidate in self.rods if (candidate.h, candidate.k) == (rod.h, rod.k)),
            None,
        )
        if configured is None:
            raise ValueError(f"rod ({rod.h}, {rod.k}) is not configured")
        geometry = evaluate_infinite_rod_ewald_geometry(
            rod=configured,
            branch=branch,
            reciprocal_basis_Ainv=self.reciprocal_basis_Ainv,
            crystal_to_sample=self.crystal_to_sample,
            ki_sample_Ainv=self.ki_sample_Ainv,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )
        return map_ewald_geometry_to_detector(
            geometry,
            incident=self.incident,
            material=self.material,
            instrument=self.instrument,
        )

    def evaluate_detector_geometry(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        include_surface_jacobian: bool = True,
    ) -> DetectorCoordinateGeometry:
        return evaluate_detector_coordinates_geometry(
            column_px,
            row_px,
            incident=self.incident,
            instrument=self.instrument,
            ki_sample_Ainv=self.ki_sample_Ainv,
            include_surface_jacobian=include_surface_jacobian,
        )


def build_geometry_only_ewald_context(
    inputs: ConfiguredGeometryInputs,
    *,
    instrument: CompiledInstrument | None = None,
) -> GeometryOnlyEwaldContext:
    """Build the nominal Ewald indexing context without intensity or mosaic setup."""

    if not isinstance(inputs, ConfiguredGeometryInputs):
        raise TypeError("inputs must be ConfiguredGeometryInputs")
    active_instrument = inputs.instrument if instrument is None else instrument
    if not isinstance(active_instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    incident = build_incident_states(inputs.samples, inputs.material, active_instrument)
    if not bool(incident.states.valid[0]):
        raise ValueError("nominal geometry-only incident state is invalid")
    return GeometryOnlyEwaldContext(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        crystal_to_sample=active_instrument.sample_from_crystal.rotation,
        rods=inputs.rods,
        incident=incident,
        material=inputs.material,
        instrument=active_instrument,
    )


def build_configured_geometry_inputs(
    config: SimulationConfiguration,
    *,
    direct_basis_A: ArrayLike | None = None,
) -> ConfiguredGeometryInputs:
    """Build the one-ray material and reciprocal state needed by exact geometry tags."""

    if not isinstance(config, SimulationConfiguration):
        raise TypeError("config must be SimulationConfiguration")
    samples = sample_configured_nominal_geometry_source(config.source)
    instrument = _compile_instrument(config.instrument)
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    if direct_basis_A is not None:
        crystal = crystal_with_direct_basis(
            crystal,
            np.asarray(direct_basis_A, dtype=np.float64),
            provenance="explicit configured direct-basis override",
        )
    material = material_optics(crystal, samples.wavelength_A)
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    air_k_Ainv = 2.0 * np.pi / float(samples.wavelength_A[0])
    rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=reciprocal.basis_Ainv,
        k_norm_Ainv=air_k_Ainv,
        population=config.bragg.rod_population,
    )
    if not config.bragg.include_detector_visible_m0:
        rods = tuple(rod for rod in rods if rod.family_m != 0)
    return ConfiguredGeometryInputs(
        config=config,
        samples=samples,
        instrument=instrument,
        crystal=crystal,
        material=material,
        reciprocal=reciprocal,
        rods=rods,
    )


def rebind_configured_geometry_direct_basis(
    inputs: ConfiguredGeometryInputs,
    direct_basis_A: ArrayLike,
) -> ConfiguredGeometryInputs:
    """Rebuild every lattice-dependent geometry input for one explicit direct basis."""

    if not isinstance(inputs, ConfiguredGeometryInputs):
        raise TypeError("inputs must be ConfiguredGeometryInputs")
    crystal = crystal_with_direct_basis(
        inputs.crystal,
        np.asarray(direct_basis_A, dtype=np.float64),
        provenance="regularized fitted direct basis",
    )
    material = material_optics(crystal, inputs.samples.wavelength_A)
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    air_k_Ainv = 2.0 * np.pi / float(inputs.samples.wavelength_A[0])
    rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=reciprocal.basis_Ainv,
        k_norm_Ainv=air_k_Ainv,
        population=inputs.config.bragg.rod_population,
    )
    if not inputs.config.bragg.include_detector_visible_m0:
        rods = tuple(rod for rod in rods if rod.family_m != 0)
    return replace(
        inputs,
        crystal=crystal,
        material=material,
        reciprocal=reciprocal,
        rods=rods,
    )


def rebind_configured_geometry_instrument(
    inputs: ConfiguredGeometryInputs,
    config: SimulationConfiguration,
) -> ConfiguredGeometryInputs:
    """Reuse material/reciprocal state when only commanded axis angles change."""

    if not isinstance(inputs, ConfiguredGeometryInputs):
        raise TypeError("inputs must be ConfiguredGeometryInputs")
    if not isinstance(config, SimulationConfiguration):
        raise TypeError("config must be SimulationConfiguration")
    before = inputs.config
    before_instrument = replace(
        before.instrument,
        axis_rotations=tuple(
            replace(axis, angle_deg=0.0) for axis in before.instrument.axis_rotations
        ),
    )
    after_instrument = replace(
        config.instrument,
        axis_rotations=tuple(
            replace(axis, angle_deg=0.0) for axis in config.instrument.axis_rotations
        ),
    )
    if (
        config.material != before.material
        or config.cif_sha256 != before.cif_sha256
        or config.source != before.source
        or config.bragg != before.bragg
        or after_instrument != before_instrument
    ):
        raise ValueError(
            "geometry reuse requires identical material and CIF content, source, Bragg state, "
            "and instrument apart from commanded axis angles"
        )
    return replace(inputs, config=config, instrument=_compile_instrument(config.instrument))


@dataclass(frozen=True, slots=True)
class ConfiguredSimulationInputs:
    config: SimulationConfiguration
    samples: IncidentSampleBatch
    instrument: CompiledInstrument
    crystal: CrystalStructure
    incident: IncidentTransportResult
    reciprocal: ReciprocalLattice
    rods: tuple[Rod, ...]
    mosaic: MosaicParameters
    strength: RevisionedStructureStrengthModel
    bragg_space: MosaicBraggSpace
    material: MaterialOptics
    commanded_instrument_rebindable: bool = True
    scan_calibration_binding_revision: str | None = field(init=False, default=None)
    calibrated_incidence_axis_angle_rad: float | None = field(init=False, default=None)


def configured_rod_catalog_revision(
    inputs: ConfiguredSimulationInputs,
    *,
    rods: tuple[Rod, ...] | None = None,
) -> str:
    """Return the material-, lattice-, and order-bound physical rod catalog revision."""

    if not isinstance(inputs, ConfiguredSimulationInputs):
        raise TypeError("inputs must be ConfiguredSimulationInputs")
    active_rods = inputs.rods if rods is None else tuple(rods)
    if not active_rods or any(rod not in inputs.rods for rod in active_rods):
        raise ValueError("rod revision requires a nonempty configured subset")
    return canonical_revision_sha256(
        ("definition_id", "configured_physical_rods.v1"),
        ("phase_id", inputs.config.material.phase_id),
        ("cif_sha256", inputs.config.cif_sha256),
        ("reciprocal_basis_Ainv", inputs.reciprocal.basis_Ainv),
        ("rod_h", np.asarray([rod.h for rod in active_rods], dtype=np.int64)),
        ("rod_k", np.asarray([rod.k for rod in active_rods], dtype=np.int64)),
        ("rod_population", np.asarray([rod.population for rod in active_rods], dtype=np.float64)),
    )


def build_configured_simulation_inputs(
    config: SimulationConfiguration,
    *,
    direct_basis_A: ArrayLike | None = None,
) -> ConfiguredSimulationInputs:
    """Build shared immutable physics once, without allocating any rendered field."""

    if not isinstance(config, SimulationConfiguration):
        raise TypeError("config must be SimulationConfiguration")
    if config.structure_factor.model_id not in {
        "cif_conventional_cell_finite_repeat.v1",
        "r3m_quintuple_finite_2h.v1",
        "r3m_quintuple_finite_3r.v1",
    }:
        raise ValueError(
            "structure_factor.model_id must select a supported model for intensity: the generic "
            "CIF finite-repeat or finite R-3m quintuple-layer model"
        )
    try:
        normalization = EventIntensityNormalization(config.structure_factor.normalization)
    except ValueError as error:
        raise ValueError(
            "structure_factor.normalization must be FINITE_TOTAL or FINITE_PER_LAYER"
        ) from error
    if normalization is EventIntensityNormalization.UNIT_CELL:
        raise ValueError("structure_factor.normalization must be FINITE_TOTAL or FINITE_PER_LAYER")
    generic_cif = config.structure_factor.model_id == "cif_conventional_cell_finite_repeat.v1"
    if not generic_cif and normalization is not EventIntensityNormalization.FINITE_TOTAL:
        raise ValueError("R-3m quintuple intensity requires FINITE_TOTAL normalization")
    if not 0.0 <= config.structure_factor.shared_disorder_epsilon <= 1.0:
        raise ValueError(
            "structure_factor.shared_disorder_epsilon must lie in [0, 1] for intensity"
        )
    stacking_parent = (
        Parent.THREE_R
        if config.structure_factor.model_id == "r3m_quintuple_finite_3r.v1"
        else Parent.TWO_H
    )
    if generic_cif and config.structure_factor.shared_disorder_epsilon != 0.0:
        raise ValueError("generic CIF finite repeats do not accept a stacking-disorder epsilon")
    if config.mosaic.lorentzian_probability < 1.0 and config.mosaic.gaussian_sigma_deg == 0.0:
        raise ValueError("active Gaussian mosaic width must be nonzero for intensity")
    if config.mosaic.lorentzian_probability > 0.0 and config.mosaic.lorentzian_hwhm_deg == 0.0:
        raise ValueError("active Lorentzian mosaic width must be nonzero for intensity")
    samples = sample_configured_source(config.source)
    instrument = _compile_instrument(config.instrument)
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    if direct_basis_A is not None:
        crystal = crystal_with_direct_basis(
            crystal,
            np.asarray(direct_basis_A, dtype=np.float64),
            provenance="explicit configured direct-basis override",
        )
    material = material_optics(crystal, samples.wavelength_A)
    incident = build_incident_states(samples, material, instrument)
    valid_index = np.flatnonzero(incident.states.valid)
    if not valid_index.size:
        raise ValueError("configured source produced no valid incident state")
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    # Commanded-angle rebinding may change which sampled rays intersect the film.  Build the
    # immutable catalog for every sampled wavelength so a later valid ray cannot require a rod
    # omitted by the base view.
    maximum_air_k = 2.0 * np.pi / float(np.min(incident.states.wavelength_A))
    rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=reciprocal.basis_Ainv,
        k_norm_Ainv=maximum_air_k,
        population=config.bragg.rod_population,
    )
    if not config.bragg.include_detector_visible_m0:
        rods = tuple(rod for rod in rods if rod.family_m != 0)
    mosaic = _mosaic(config.mosaic)
    strength: RevisionedStructureStrengthModel
    if generic_cif:
        if config.structure_factor.repeats is None or config.structure_factor.layers is not None:
            raise ValueError("generic CIF intensity requires a conventional-cell repeat count")
        strength = CifFiniteStackStrength(
            crystal=crystal,
            repeats=config.structure_factor.repeats,
            normalization=normalization,
            unknown_u_iso_A2=config.structure_factor.unknown_u_iso_A2,
        )
    else:
        if config.structure_factor.layers is None or config.structure_factor.repeats is not None:
            raise ValueError("quintuple-layer intensity requires a layer count")
        if config.structure_factor.unknown_u_iso_A2 is not None:
            raise ValueError("R-3m quintuple intensity does not use unknown_u_iso_A2")
        strength = Bi2X3FiniteStackStrength(
            crystal=crystal,
            layers=config.structure_factor.layers,
            normalization=normalization,
            parent=stacking_parent,
            shared_disorder_epsilon=config.structure_factor.shared_disorder_epsilon,
        )
    nominal_air_k = 2.0 * np.pi / config.source.mean_wavelength_A
    nominal_keys = {
        (rod.h, rod.k)
        for rod in enumerate_rods_within_ewald_sphere(
            reciprocal_basis_Ainv=reciprocal.basis_Ainv,
            k_norm_Ainv=nominal_air_k,
            population=config.bragg.rod_population,
        )
    }
    nominal_rods = tuple(rod for rod in rods if (rod.h, rod.k) in nominal_keys)
    bragg_space = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=reciprocal.basis_Ainv,
            crystal_to_sample=instrument.sample_from_crystal.rotation,
            rods=nominal_rods,
            mosaic=mosaic,
            k_norm_Ainv=nominal_air_k,
        ),
        strength,
    )
    return ConfiguredSimulationInputs(
        config=config,
        samples=samples,
        instrument=instrument,
        crystal=crystal,
        incident=incident,
        reciprocal=reciprocal,
        rods=rods,
        mosaic=mosaic,
        strength=strength,
        bragg_space=bragg_space,
        material=material,
    )


def rebind_configured_simulation_instrument(
    inputs: ConfiguredSimulationInputs,
    config: SimulationConfiguration,
) -> ConfiguredSimulationInputs:
    """Reuse immutable physics when only commanded instrument-axis angles change."""

    if not isinstance(inputs, ConfiguredSimulationInputs):
        raise TypeError("inputs must be ConfiguredSimulationInputs")
    if not isinstance(config, SimulationConfiguration):
        raise TypeError("config must be SimulationConfiguration")
    if inputs.commanded_instrument_rebindable is not True:
        raise ValueError("instrument rebinding is disabled after fixed geometry corrections")

    def without_commanded_angles(value: SimulationConfiguration) -> SimulationConfiguration:
        return replace(
            value,
            instrument=replace(
                value.instrument,
                axis_rotations=tuple(
                    replace(axis, angle_deg=0.0) for axis in value.instrument.axis_rotations
                ),
            ),
        )

    if without_commanded_angles(config) != without_commanded_angles(inputs.config):
        raise ValueError(
            "simulation reuse requires identical configuration apart from commanded axis angles"
        )
    instrument = _compile_instrument(config.instrument)
    incident = build_incident_states(inputs.samples, inputs.material, instrument)
    if not bool(np.any(incident.states.valid)):
        raise ValueError("rebound simulation produced no valid incident state")
    bragg_space = MosaicBraggSpace(
        replace(
            inputs.bragg_space.config,
            crystal_to_sample=instrument.sample_from_crystal.rotation,
        ),
        inputs.strength,
    )
    return replace(
        inputs,
        config=config,
        instrument=instrument,
        incident=incident,
        bragg_space=bragg_space,
    )


def _configured_incidence_scan_metadata(
    inputs: ConfiguredSimulationInputs,
) -> tuple[float | None, str | None]:
    """Return one builder-owned calibrated scan stamp, or no stamp at all."""

    angle = inputs.calibrated_incidence_axis_angle_rad
    binding = inputs.scan_calibration_binding_revision
    if (angle is None) != (binding is None):
        raise ValueError("configured incidence-scan metadata must be present as one pair")
    if angle is None:
        return None, None
    if not isinstance(binding, str) or not binding:
        raise ValueError("configured incidence-scan binding must be nonempty")
    if len(inputs.config.instrument.axis_rotations) != 1:
        raise ValueError("configured incidence-scan metadata requires one incidence axis")
    declared_angle = math.radians(inputs.config.instrument.axis_rotations[0].angle_deg)
    if not math.isfinite(float(angle)) or float(angle) != declared_angle:
        raise ValueError("configured incidence-scan angle does not match its calibrated pose")
    return float(angle), binding


def _source_reachable_rods(inputs: ConfiguredSimulationInputs) -> tuple[Rod, ...]:
    """Return the configured catalog subset reachable by at least one valid source state."""

    valid_index = np.flatnonzero(inputs.incident.states.valid)
    maximum_air_k = 2.0 * np.pi / float(np.min(inputs.incident.states.wavelength_A[valid_index]))
    reachable_keys = {
        (rod.h, rod.k)
        for rod in enumerate_rods_within_ewald_sphere(
            reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
            k_norm_Ainv=maximum_air_k,
            population=inputs.config.bragg.rod_population,
        )
    }
    return tuple(rod for rod in inputs.rods if (rod.h, rod.k) in reachable_keys)


def build_source_averaged_structure_detector(
    inputs: ConfiguredSimulationInputs,
    *,
    strength_model: RevisionedStructureStrengthModel | None = None,
) -> SourceAveragedStructureDetector:
    """Build the one material-neutral detector function used by staged fitting."""

    reachable_rods = _source_reachable_rods(inputs)
    active_strength = inputs.strength if strength_model is None else strength_model
    incidence_axis_angle_rad, scan_binding_revision = _configured_incidence_scan_metadata(inputs)
    return SourceAveragedStructureDetector(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
        rods=reachable_rods,
        rod_catalog_revision=configured_rod_catalog_revision(inputs, rods=reachable_rods),
        mosaic=inputs.mosaic,
        strength_model=active_strength,
        incident=inputs.incident,
        material=inputs.material,
        instrument=inputs.instrument,
        phase_population_weight=inputs.config.weights.phase_population,
        polarization_weight=inputs.config.weights.polarization,
        incidence_axis_angle_rad=incidence_axis_angle_rad,
        scan_calibration_binding_revision=scan_binding_revision,
    )


def build_source_averaged_detector(
    inputs: ConfiguredSimulationInputs,
) -> SourceAveragedDetectorEwaldMeasure:
    """Compile the optimized Bi2X3 renderer; fits use the generic structure detector."""

    if not isinstance(inputs.strength, Bi2X3FiniteStackStrength):
        raise TypeError(
            "the optimized renderer requires Bi2X3FiniteStackStrength; use "
            "build_source_averaged_structure_detector for a general CIF"
        )
    reachable_rods = _source_reachable_rods(inputs)
    incidence_axis_angle_rad, scan_binding_revision = _configured_incidence_scan_metadata(inputs)
    return SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
        rods=reachable_rods,
        rod_catalog_revision=configured_rod_catalog_revision(inputs, rods=reachable_rods),
        mosaic=inputs.mosaic,
        strength_model=inputs.strength,
        incident=inputs.incident,
        material=inputs.material,
        instrument=inputs.instrument,
        phase_population_weight=inputs.config.weights.phase_population,
        polarization_weight=inputs.config.weights.polarization,
        worker_count=inputs.config.numerics.worker_count,
        incidence_axis_angle_rad=incidence_axis_angle_rad,
        scan_calibration_binding_revision=scan_binding_revision,
    )


def rebind_source_averaged_detector_incidence_scan(
    template_detector: SourceAveragedDetectorEwaldMeasure,
    scan_inputs: tuple[ConfiguredSimulationInputs, ...],
) -> tuple[SourceAveragedDetectorEwaldMeasure, ...]:
    """Reuse one compiled optimized detector across a calibrated incidence scan.

    ``scan_inputs`` must be the ordered, builder-stamped inputs from a calibrated scan. The
    template may already carry a rod restriction, candidate physics, execution blocking, and a
    specular stitch. Those immutable states are retained while each input's precomputed incident
    transport and rigid detector/sample geometry are rebound in input order. Scan nodes must keep
    the same source-validity topology; callers needing changing validity must build each component
    independently.
    """

    if not isinstance(template_detector, SourceAveragedDetectorEwaldMeasure):
        raise TypeError("template_detector must be SourceAveragedDetectorEwaldMeasure")
    if not isinstance(scan_inputs, tuple):
        raise TypeError("scan_inputs must be a tuple of ConfiguredSimulationInputs")
    if not scan_inputs:
        raise ValueError("scan_inputs must not be empty")
    if not all(isinstance(item, ConfiguredSimulationInputs) for item in scan_inputs):
        raise TypeError("scan_inputs must contain only ConfiguredSimulationInputs")

    first_angle, first_binding = _configured_incidence_scan_metadata(scan_inputs[0])
    if first_angle is None or first_binding is None:
        raise ValueError("scan_inputs must carry calibrated incidence-scan metadata")

    components: list[SourceAveragedDetectorEwaldMeasure] = []
    static_revision = template_detector.incidence_angle_static_physics_revision
    for item in scan_inputs:
        angle, binding = _configured_incidence_scan_metadata(item)
        if angle is None or binding != first_binding:
            raise ValueError("all scan_inputs must carry one calibrated incidence-scan binding")
        rebound = template_detector._rebind_calibrated_incidence_scan_geometry(
            incident=item.incident,
            instrument=item.instrument,
            incidence_axis_angle_rad=angle,
            scan_calibration_binding_revision=binding,
        )
        if rebound.incidence_angle_static_physics_revision != static_revision:
            raise ValueError("only incidence geometry may vary across the scan")
        components.append(rebound)
    return tuple(components)


@dataclass(frozen=True, slots=True)
class NominalEwaldContext:
    geometry: DetectorEwaldMeasure
    incident: IncidentTransportResult

    @property
    def reciprocal_basis_Ainv(self) -> FloatArray:
        return self.geometry.coating.bragg_space.config.reciprocal_basis_Ainv

    @property
    def crystal_to_sample(self) -> FloatArray:
        return self.geometry.coating.bragg_space.config.crystal_to_sample

    @property
    def rods(self) -> tuple[Rod, ...]:
        return self.geometry.coating.bragg_space.config.rods

    @property
    def instrument(self) -> CompiledInstrument:
        return self.geometry.instrument

    @property
    def ki_sample_Ainv(self) -> FloatArray:
        return self.geometry.coating.ki_sample_Ainv

    def map_latent_geometry(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> DetectorMappedGeometry:
        return self.geometry.map_latent_geometry(
            rod=rod,
            branch=branch,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )

    def evaluate_detector_geometry(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        include_surface_jacobian: bool = True,
    ) -> DetectorCoordinateGeometry:
        return self.geometry.evaluate_detector_geometry(
            column_px,
            row_px,
            include_surface_jacobian=include_surface_jacobian,
        )


def _build_single_ewald_context(
    inputs: ConfiguredSimulationInputs,
    samples: IncidentSampleBatch,
) -> NominalEwaldContext:
    """Build one Ewald context from an explicitly declared source row."""

    if samples.wavelength_A.shape != (1,):
        raise ValueError("a single Ewald context requires exactly one source row")

    material = material_optics(inputs.crystal, samples.wavelength_A)
    incident = build_incident_states(samples, material, inputs.instrument)
    if not bool(incident.states.valid[0]):
        raise ValueError("nominal mean source state is invalid")
    nominal_air_k = 2.0 * np.pi / float(samples.wavelength_A[0])
    bragg_config = inputs.bragg_space.config
    tolerance = 256.0 * np.finfo(np.float64).eps * max(nominal_air_k, 1.0)
    if not np.isclose(bragg_config.k_norm_Ainv, nominal_air_k, rtol=0.0, atol=tolerance):
        raise ValueError("nominal Bragg space wavelength does not match the mean source state")
    if not np.array_equal(bragg_config.reciprocal_basis_Ainv, inputs.reciprocal.basis_Ainv):
        raise ValueError(
            "nominal Bragg space reciprocal basis does not match the configured lattice"
        )
    if not np.array_equal(
        bragg_config.crystal_to_sample,
        inputs.instrument.sample_from_crystal.rotation,
    ):
        raise ValueError("nominal Bragg space orientation does not match the configured instrument")
    if bragg_config.mosaic != inputs.mosaic:
        raise ValueError("nominal Bragg space mosaic does not match the configured mosaic")
    if inputs.bragg_space.strength_model is not inputs.strength:
        raise ValueError("nominal Bragg space does not own the configured strength model")
    reachable_keys = {
        (rod.h, rod.k)
        for rod in enumerate_rods_within_ewald_sphere(
            reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
            k_norm_Ainv=nominal_air_k,
            population=inputs.config.bragg.rod_population,
        )
    }
    expected_rods = tuple(rod for rod in inputs.rods if (rod.h, rod.k) in reachable_keys)
    if bragg_config.rods != expected_rods:
        raise ValueError("nominal Bragg space rods do not match nominal elastic reach")
    coating = ContinuousEwaldCoating(
        inputs.bragg_space,
        ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
    )
    geometry = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=inputs.instrument,
        rod_catalog_revision=configured_rod_catalog_revision(inputs),
        phase_population_weight=inputs.config.weights.phase_population,
        polarization_weight=inputs.config.weights.polarization,
    )
    return NominalEwaldContext(geometry=geometry, incident=incident)


def build_nominal_ewald_context(inputs: ConfiguredSimulationInputs) -> NominalEwaldContext:
    """Build the geometry-only nominal mean incident state for Ewald landmarks."""

    return _build_single_ewald_context(
        inputs,
        sample_configured_nominal_geometry_source(inputs.config.source),
    )


def build_single_source_ewald_context(
    inputs: ConfiguredSimulationInputs,
) -> NominalEwaldContext:
    """Build intensity-capable geometry for an exact configured one-row source."""

    require_physical_intensity_source_model(inputs.samples.source_sampling_model_id)
    if inputs.samples.wavelength_A.shape != (1,):
        raise ValueError("configured source must contain exactly one physical row")
    return _build_single_ewald_context(inputs, inputs.samples)


@dataclass(frozen=True, slots=True)
class DetectorIntegerLMarkers:
    """Exact peak-mosaic integer-L references that reach the active detector.

    The coordinates use the nominal mean incident state and the continuous
    ``alpha=0`` mosaic center.  Coincident family sites are stored once for
    display, while every independently evaluated physical rod remains in the
    aligned provenance tuples. Strengths are nonnegative diagnostics and never
    determine which geometry-visible sites are retained.
    """

    column_px: FloatArray
    row_px: FloatArray
    q_sample_Ainv: FloatArray
    family_m: NDArray[np.int64]
    integer_L: NDArray[np.int64]
    branch: NDArray[np.int64]
    root_sign: NDArray[np.int64]
    ewald_residual_Ainv: FloatArray
    family_strength_weight_A2: FloatArray
    contributing_rod_hk: tuple[tuple[tuple[int, int], ...], ...]
    contributing_beta_rad: tuple[tuple[float, ...], ...]
    per_rod_strength_weight_A2: tuple[tuple[float, ...], ...]
    reference_wavelength_A: float
    definition_id: str = "peak_mosaic_alpha0_integer_L_center.v3"
    source_state_policy: str = "mean_source_state.v1"

    def __post_init__(self) -> None:
        column = np.array(self.column_px, dtype=np.float64, copy=True, order="C")
        if column.ndim != 1 or not np.all(np.isfinite(column)):
            raise ValueError("column_px must be a finite one-dimensional array")
        shape = column.shape
        row = _readonly_float_array(self.row_px, shape, "row_px")
        q_sample = _readonly_float_array(self.q_sample_Ainv, (*shape, 3), "q_sample_Ainv")
        residual = _readonly_float_array(
            self.ewald_residual_Ainv,
            shape,
            "ewald_residual_Ainv",
        )
        family_strength = _readonly_float_array(
            self.family_strength_weight_A2,
            shape,
            "family_strength_weight_A2",
        )
        family = np.array(self.family_m, dtype=np.int64, copy=True, order="C")
        integer_l = np.array(self.integer_L, dtype=np.int64, copy=True, order="C")
        branch = np.array(self.branch, dtype=np.int64, copy=True, order="C")
        root_sign = np.array(self.root_sign, dtype=np.int64, copy=True, order="C")
        if (
            family.shape != shape
            or integer_l.shape != shape
            or branch.shape != shape
            or root_sign.shape != shape
            or np.any(family < 0)
            or np.any(~np.isin(branch, (0, 1, 2)))
            or np.any((family == 0) != (branch == 0))
            or np.any((family == 0) & (root_sign != 0))
            or np.any((family != 0) & ~np.isin(root_sign, (-1, 0, 1)))
            or np.any(residual < 0.0)
            or np.any(family_strength < 0.0)
        ):
            raise ValueError("integer-L marker arrays contain invalid values")
        rod_hk = tuple(
            tuple((int(h), int(k)) for h, k in group) for group in self.contributing_rod_hk
        )
        beta = tuple(tuple(float(value) for value in group) for group in self.contributing_beta_rad)
        per_rod_strength = tuple(
            tuple(float(value) for value in group) for group in self.per_rod_strength_weight_A2
        )
        if not (len(rod_hk) == len(beta) == len(per_rod_strength) == column.size):
            raise ValueError("marker provenance must have one group per display site")
        for index, (hk_group, beta_group, strength_group) in enumerate(
            zip(rod_hk, beta, per_rod_strength, strict=True)
        ):
            if (
                not hk_group
                or len(hk_group) != len(beta_group)
                or len(hk_group) != len(strength_group)
                or len(set(hk_group)) != len(hk_group)
                or any(
                    not math.isfinite(value) or not 0.0 <= value < 2.0 * np.pi
                    for value in beta_group
                )
                or any(not math.isfinite(value) or value < 0.0 for value in strength_group)
                or not math.isclose(
                    math.fsum(strength_group),
                    float(family_strength[index]),
                    rel_tol=1024.0 * np.finfo(np.float64).eps,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("integer-L marker provenance is inconsistent")
        wavelength = _positive(self.reference_wavelength_A, "reference_wavelength_A")
        if self.definition_id != "peak_mosaic_alpha0_integer_L_center.v3":
            raise ValueError("unsupported integer-L marker definition")
        if self.source_state_policy != "mean_source_state.v1":
            raise ValueError("unsupported integer-L marker source-state policy")
        for value in (column, family, integer_l, branch, root_sign):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "q_sample_Ainv", q_sample)
        object.__setattr__(self, "family_m", family)
        object.__setattr__(self, "integer_L", integer_l)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "root_sign", root_sign)
        object.__setattr__(self, "ewald_residual_Ainv", residual)
        object.__setattr__(self, "family_strength_weight_A2", family_strength)
        object.__setattr__(self, "contributing_rod_hk", rod_hk)
        object.__setattr__(self, "contributing_beta_rad", beta)
        object.__setattr__(self, "per_rod_strength_weight_A2", per_rod_strength)
        object.__setattr__(self, "reference_wavelength_A", wavelength)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return unambiguous family, integer-L, Ewald-branch, and beta-side labels."""

        return tuple(
            f"m={family}, L={integer_l}, b={branch}, s={root_sign:+d}"
            for family, integer_l, branch, root_sign in zip(
                self.family_m,
                self.integer_L,
                self.branch,
                self.root_sign,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class _IntegerLMarkerContribution:
    family_m: int
    integer_L: int
    branch: int
    root_sign: int
    column_px: float
    row_px: float
    q_sample_Ainv: FloatArray
    ewald_residual_Ainv: float
    rod_hk: tuple[int, int]
    beta_rad: float
    strength_weight_A2: float


@dataclass(frozen=True, slots=True)
class IntegerLEwaldRoots:
    """Isolated alpha-zero beta roots for one exact integer-L rod section."""

    beta_rad: tuple[float, ...]
    root_sign: tuple[int, ...]
    branch: int

    def __post_init__(self) -> None:
        beta = tuple(float(value) for value in self.beta_rad)
        signs = tuple(int(value) for value in self.root_sign)
        if (
            len(beta) not in {1, 2}
            or len(signs) != len(beta)
            or any(not math.isfinite(value) or not 0.0 <= value < 2.0 * np.pi for value in beta)
            or (len(beta) == 1 and signs != (0,))
            or (len(beta) == 2 and signs != (-1, 1))
            or self.branch not in {1, 2}
        ):
            raise ValueError("invalid isolated integer-L Ewald roots")
        object.__setattr__(self, "beta_rad", beta)
        object.__setattr__(self, "root_sign", signs)


@dataclass(frozen=True, slots=True)
class LayerLEwaldRoots:
    """Isolated alpha-zero beta roots for one exact commensurate layer section."""

    beta_rad: tuple[float, ...]
    root_sign: tuple[int, ...]
    branch: int

    def __post_init__(self) -> None:
        beta = tuple(float(value) for value in self.beta_rad)
        signs = tuple(int(value) for value in self.root_sign)
        if (
            len(beta) not in {1, 2}
            or len(signs) != len(beta)
            or any(not math.isfinite(value) or not 0.0 <= value < 2.0 * np.pi for value in beta)
            or (len(beta) == 1 and signs != (0,))
            or (len(beta) == 2 and signs != (-1, 1))
            or self.branch not in {1, 2}
        ):
            raise ValueError("invalid isolated layer-L Ewald roots")
        object.__setattr__(self, "beta_rad", beta)
        object.__setattr__(self, "root_sign", signs)


def _solve_fixed_layer_l_ewald_roots(
    *,
    rod: Rod,
    layer_l: float,
    reciprocal_basis_Ainv: FloatArray,
    crystal_to_sample: FloatArray,
    ki_sample_Ainv: FloatArray,
) -> LayerLEwaldRoots | None:
    """Authoritative alpha-zero fixed-layer-L root equation."""

    b3 = reciprocal_basis_Ainv[:, 2]
    mean_axis = b3 / np.linalg.norm(b3)
    q_parallel = rod.h * reciprocal_basis_Ainv[:, 0] + rod.k * reciprocal_basis_Ainv[:, 1]
    q_axis = float(q_parallel @ mean_axis) * mean_axis
    q_perpendicular = q_parallel - q_axis
    q_quadrature = np.cross(mean_axis, q_perpendicular)
    cosine_coefficient = 2.0 * float(ki_sample_Ainv @ (crystal_to_sample @ q_perpendicular))
    sine_coefficient = 2.0 * float(ki_sample_Ainv @ (crystal_to_sample @ q_quadrature))
    amplitude = math.hypot(cosine_coefficient, sine_coefficient)
    coefficient_scale = max(
        float(np.linalg.norm(ki_sample_Ainv) * np.linalg.norm(q_perpendicular)),
        1.0,
    )
    tolerance = 4096.0 * np.finfo(np.float64).eps
    u_Ainv = layer_l * float(np.linalg.norm(b3))
    q_base = q_axis + u_Ainv * mean_axis
    q_unrotated = q_parallel + u_Ainv * mean_axis
    constant = float(
        q_unrotated @ q_unrotated + 2.0 * ki_sample_Ainv @ (crystal_to_sample @ q_base)
    )
    equation_scale = max(
        float(q_unrotated @ q_unrotated),
        2.0 * float(np.linalg.norm(ki_sample_Ainv) * np.linalg.norm(q_base)),
        1.0,
    )
    if amplitude <= tolerance * coefficient_scale:
        if abs(constant) <= tolerance * equation_scale:
            raise ValueError("layer-L center is a continuous beta manifold, not isolated points")
        return None
    cosine = -constant / amplitude
    if cosine < -1.0 - tolerance or cosine > 1.0 + tolerance:
        return None
    cosine = min(1.0, max(-1.0, cosine))
    phase = math.atan2(sine_coefficient, cosine_coefficient)
    delta = math.acos(cosine)
    if 1.0 - abs(cosine) <= tolerance:
        beta = ((phase + (np.pi if cosine < 0.0 else 0.0)) % (2.0 * np.pi),)
        root_sign = (0,)
    else:
        beta = ((phase - delta) % (2.0 * np.pi), (phase + delta) % (2.0 * np.pi))
        root_sign = (-1, 1)
    direction_sample = crystal_to_sample @ mean_axis
    signed_root = u_Ainv + float(q_parallel @ mean_axis) + float(ki_sample_Ainv @ direction_sample)
    root_scale = max(abs(u_Ainv), float(np.linalg.norm(ki_sample_Ainv)), 1.0)
    if abs(signed_root) <= tolerance * root_scale:
        return None
    return LayerLEwaldRoots(
        beta_rad=beta,
        root_sign=root_sign,
        branch=1 if signed_root < 0.0 else 2,
    )


def solve_layer_l_ewald_roots(
    *,
    rod: Rod,
    layer_order: CommensurateLayerOrder,
    reciprocal_basis_Ainv: FloatArray,
    crystal_to_sample: FloatArray,
    ki_sample_Ainv: FloatArray,
) -> LayerLEwaldRoots | None:
    """Solve one exact rational-layer section in the declared one-layer basis."""

    if not isinstance(layer_order, CommensurateLayerOrder):
        raise TypeError("layer_order must be CommensurateLayerOrder")
    return _solve_fixed_layer_l_ewald_roots(
        rod=rod,
        layer_l=layer_order.as_float(),
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        ki_sample_Ainv=ki_sample_Ainv,
    )


def solve_integer_l_ewald_roots(
    *,
    rod: Rod,
    integer_l: int,
    reciprocal_basis_Ainv: FloatArray,
    crystal_to_sample: FloatArray,
    ki_sample_Ainv: FloatArray,
) -> IntegerLEwaldRoots | None:
    """Solve the alpha=0 Ewald equation analytically for full-beta roots."""

    roots = _solve_fixed_layer_l_ewald_roots(
        rod=rod,
        layer_l=CommensurateLayerOrder(integer_l).as_float(),
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        ki_sample_Ainv=ki_sample_Ainv,
    )
    if roots is None:
        return None
    return IntegerLEwaldRoots(roots.beta_rad, roots.root_sign, roots.branch)


def evaluate_nominal_integer_l_markers(
    context: NominalEwaldContext,
) -> DetectorIntegerLMarkers:
    """Return exact visible integer-L centers for the nominal incident state.

    Each physical rod is evaluated independently at exact integer L before
    symmetry-coincident detector sites are grouped for legible display. The
    geometry-visible catalog is retained even when its diagnostic strength is zero.
    """

    if not isinstance(context, NominalEwaldContext):
        raise TypeError("context must be a NominalEwaldContext")
    geometry = context.geometry
    space = geometry.coating.bragg_space
    basis = space.config.reciprocal_basis_Ainv
    crystal_to_sample = space.config.crystal_to_sample
    ki_sample_Ainv = geometry.coating.ki_sample_Ainv
    k_norm_Ainv = float(np.linalg.norm(ki_sample_Ainv))
    b3_norm_Ainv = float(np.linalg.norm(basis[:, 2]))
    tolerance = 4096.0 * np.finfo(np.float64).eps
    integer_l_solver_tolerance = 131072.0 * np.finfo(np.float64).eps
    contributions: list[_IntegerLMarkerContribution] = []

    for rod in space.config.rods:
        if rod.family_m == 0:
            mapped = geometry.map_specular_geometry(
                rod=rod,
                alpha_rad=np.asarray([0.0]),
                beta_rad=np.asarray([0.0]),
            ).geometry
            if not bool(mapped.valid[0]):
                continue
            actual_l = float(mapped.ewald_geometry.L[0])
            integer_l = round(actual_l)
            if integer_l == 0 or abs(actual_l - integer_l) > tolerance * max(abs(actual_l), 1.0):
                continue
            strength = rod.population * space.strength_model.evaluate(
                rod=rod,
                L=float(integer_l),
                k_norm_Ainv=space.config.k_norm_Ainv,
            )
            exact_q = space.map_latent(
                rod=rod,
                alpha_rad=0.0,
                beta_rad=0.0,
                u_Ainv=integer_l * b3_norm_Ainv,
            )
            residual = abs(float(np.linalg.norm(ki_sample_Ainv + exact_q) - k_norm_Ainv))
            contributions.append(
                _IntegerLMarkerContribution(
                    family_m=0,
                    integer_L=integer_l,
                    branch=0,
                    root_sign=0,
                    column_px=float(mapped.column_px[0]),
                    row_px=float(mapped.row_px[0]),
                    q_sample_Ainv=np.asarray(exact_q, dtype=np.float64),
                    ewald_residual_Ainv=residual,
                    rod_hk=(rod.h, rod.k),
                    beta_rad=0.0,
                    strength_weight_A2=float(strength),
                )
            )
            continue

        lower_u_Ainv, upper_u_Ainv = space.rod_u_bounds_Ainv(rod)
        lower_l = lower_u_Ainv / b3_norm_Ainv
        upper_l = upper_u_Ainv / b3_norm_Ainv
        l_scale = max(abs(lower_l), abs(upper_l), 1.0)
        first_l = math.ceil(lower_l - tolerance * l_scale)
        last_l = math.floor(upper_l + tolerance * l_scale)
        candidates_by_branch: dict[int, list[tuple[int, float, int]]] = {1: [], 2: []}
        for integer_l in range(first_l, last_l + 1):
            roots = solve_integer_l_ewald_roots(
                rod=rod,
                integer_l=integer_l,
                reciprocal_basis_Ainv=basis,
                crystal_to_sample=crystal_to_sample,
                ki_sample_Ainv=ki_sample_Ainv,
            )
            if roots is None:
                continue
            candidates_by_branch[roots.branch].extend(
                (integer_l, beta_rad, root_sign)
                for beta_rad, root_sign in zip(roots.beta_rad, roots.root_sign, strict=True)
            )
        visible_roots: list[tuple[int, int, int, float, float, float]] = []
        for branch, candidates in candidates_by_branch.items():
            if not candidates:
                continue
            beta_values = np.asarray([item[1] for item in candidates])
            mapped = geometry.map_latent_geometry(
                rod=rod,
                branch=branch,
                alpha_rad=np.zeros(beta_values.size, dtype=np.float64),
                beta_rad=beta_values,
            )
            for beta_index, (integer_l, beta_rad, root_sign) in enumerate(candidates):
                if not bool(mapped.valid[beta_index]):
                    continue
                actual_l = float(mapped.ewald_geometry.L[beta_index])
                if abs(actual_l - integer_l) > integer_l_solver_tolerance * max(abs(actual_l), 1.0):
                    raise FloatingPointError("analytic integer-L root disagrees with Ewald solver")
                visible_roots.append(
                    (
                        integer_l,
                        branch,
                        root_sign,
                        beta_rad,
                        float(mapped.column_px[beta_index]),
                        float(mapped.row_px[beta_index]),
                    )
                )
        if not visible_roots:
            continue
        visible_integer_l = np.asarray(
            sorted({item[0] for item in visible_roots}),
            dtype=np.float64,
        )
        visible_strength = rod.population * space.strength_model.evaluate_profile(
            rod=rod,
            L=visible_integer_l,
            k_norm_Ainv=space.config.k_norm_Ainv,
        )
        strength_by_l = dict(zip(visible_integer_l.astype(np.int64), visible_strength, strict=True))
        for integer_l, branch, root_sign, beta_rad, column_px, row_px in visible_roots:
            strength = float(strength_by_l[integer_l])
            exact_q = space.map_latent(
                rod=rod,
                alpha_rad=0.0,
                beta_rad=beta_rad,
                u_Ainv=float(integer_l) * b3_norm_Ainv,
            )
            residual = abs(float(np.linalg.norm(ki_sample_Ainv + exact_q) - k_norm_Ainv))
            if residual > tolerance * max(k_norm_Ainv, 1.0):
                raise FloatingPointError("integer-L marker violates the Ewald identity")
            contributions.append(
                _IntegerLMarkerContribution(
                    family_m=rod.family_m,
                    integer_L=integer_l,
                    branch=branch,
                    root_sign=root_sign,
                    column_px=column_px,
                    row_px=row_px,
                    q_sample_Ainv=np.asarray(exact_q, dtype=np.float64),
                    ewald_residual_Ainv=residual,
                    rod_hk=(rod.h, rod.k),
                    beta_rad=beta_rad,
                    strength_weight_A2=strength,
                )
            )

    coordinate_scale = max(float(max(geometry.instrument.detector_shape_rc)), 1.0)
    coordinate_tolerance_px = 32768.0 * np.finfo(np.float64).eps * coordinate_scale
    q_tolerance_Ainv = 32768.0 * np.finfo(np.float64).eps * max(k_norm_Ainv, 1.0)
    grouped: list[list[_IntegerLMarkerContribution]] = []
    for contribution in sorted(
        contributions,
        key=lambda item: (
            item.family_m,
            item.integer_L,
            item.branch,
            item.root_sign,
            item.row_px,
            item.column_px,
            item.rod_hk,
        ),
    ):
        matched: list[_IntegerLMarkerContribution] | None = None
        for group in grouped:
            representative = group[0]
            if (
                contribution.family_m == representative.family_m
                and contribution.integer_L == representative.integer_L
                and contribution.branch == representative.branch
                and contribution.root_sign == representative.root_sign
                and math.hypot(
                    contribution.column_px - representative.column_px,
                    contribution.row_px - representative.row_px,
                )
                <= coordinate_tolerance_px
                and float(np.linalg.norm(contribution.q_sample_Ainv - representative.q_sample_Ainv))
                <= q_tolerance_Ainv
            ):
                matched = group
                break
        if matched is None:
            grouped.append([contribution])
        else:
            matched.append(contribution)

    grouped.sort(
        key=lambda group: (
            group[0].family_m,
            group[0].integer_L,
            group[0].branch,
            group[0].root_sign,
            group[0].row_px,
            group[0].column_px,
        )
    )
    representatives = [group[0] for group in grouped]
    ordered_groups = [sorted(group, key=lambda item: item.rod_hk) for group in grouped]
    return DetectorIntegerLMarkers(
        column_px=np.asarray([item.column_px for item in representatives]),
        row_px=np.asarray([item.row_px for item in representatives]),
        q_sample_Ainv=np.asarray([item.q_sample_Ainv for item in representatives]).reshape(-1, 3),
        family_m=np.asarray([item.family_m for item in representatives], dtype=np.int64),
        integer_L=np.asarray([item.integer_L for item in representatives], dtype=np.int64),
        branch=np.asarray([item.branch for item in representatives], dtype=np.int64),
        root_sign=np.asarray([item.root_sign for item in representatives], dtype=np.int64),
        ewald_residual_Ainv=np.asarray(
            [max(member.ewald_residual_Ainv for member in group) for group in grouped]
        ),
        family_strength_weight_A2=np.asarray(
            [math.fsum(member.strength_weight_A2 for member in group) for group in ordered_groups]
        ),
        contributing_rod_hk=tuple(
            tuple(member.rod_hk for member in group) for group in ordered_groups
        ),
        contributing_beta_rad=tuple(
            tuple(member.beta_rad for member in group) for group in ordered_groups
        ),
        per_rod_strength_weight_A2=tuple(
            tuple(member.strength_weight_A2 for member in group) for group in ordered_groups
        ),
        reference_wavelength_A=float(context.incident.states.wavelength_A[0]),
    )


@dataclass(frozen=True, slots=True)
class ReciprocalSpaceDisplay:
    q_sample_Ainv: FloatArray
    intensity_density_A2_rad2_inv: FloatArray
    family_m: NDArray[np.int64]
    reference_wavelength_A: float
    measure_id: str = "latent_bragg_density_A2_rad2_inv.v1"

    def __post_init__(self) -> None:
        points = _readonly_float_array(self.q_sample_Ainv, (None, 3), "q_sample_Ainv")
        intensity = _readonly_float_array(
            self.intensity_density_A2_rad2_inv,
            (points.shape[0],),
            "intensity_density_A2_rad2_inv",
        )
        if np.any(intensity < 0.0):
            raise ValueError("reciprocal display intensity must be nonnegative")
        family = np.array(self.family_m, dtype=np.int64, copy=True, order="C")
        if family.shape != (points.shape[0],) or np.any(family < 0):
            raise ValueError("family_m must contain one nonnegative value per point")
        wavelength = _positive(self.reference_wavelength_A, "reference_wavelength_A")
        family.setflags(write=False)
        object.__setattr__(self, "q_sample_Ainv", points)
        object.__setattr__(self, "intensity_density_A2_rad2_inv", intensity)
        object.__setattr__(self, "family_m", family)
        object.__setattr__(self, "reference_wavelength_A", wavelength)


def sample_reciprocal_space(inputs: ConfiguredSimulationInputs) -> ReciprocalSpaceDisplay:
    """Deterministically sample the continuous latent Bragg field for rendering only."""

    numerics = inputs.config.numerics
    alpha_width = math.radians(numerics.reciprocal_alpha_max_deg)
    alpha = (np.arange(numerics.reciprocal_alpha_count, dtype=np.float64) + 0.5) * (
        alpha_width / numerics.reciprocal_alpha_count
    )
    beta = (np.arange(numerics.reciprocal_beta_count, dtype=np.float64) + 0.5) * (
        2.0 * np.pi / numerics.reciprocal_beta_count
    )
    points: list[FloatArray] = []
    intensities: list[FloatArray] = []
    families: list[NDArray[np.int64]] = []
    for rod in inputs.bragg_space.config.rods:
        lower, upper = inputs.bragg_space.rod_u_bounds_Ainv(rod)
        axial = np.linspace(lower, upper, numerics.reciprocal_u_count)
        alpha_grid, beta_grid, axial_grid = np.meshgrid(alpha, beta, axial, indexing="ij")
        evaluated = inputs.bragg_space.evaluate_latent(
            rod=rod,
            alpha_rad=alpha_grid,
            beta_rad=beta_grid,
            u_Ainv=axial_grid,
        )
        points.append(evaluated.q_sample_Ainv.reshape(-1, 3))
        intensities.append(evaluated.intensity_density_A2_rad2_inv.reshape(-1))
        families.append(np.full(alpha_grid.size, rod.family_m, dtype=np.int64))
    return ReciprocalSpaceDisplay(
        q_sample_Ainv=np.concatenate(points, axis=0),
        intensity_density_A2_rad2_inv=np.concatenate(intensities),
        family_m=np.concatenate(families),
        reference_wavelength_A=2.0 * np.pi / inputs.bragg_space.config.k_norm_Ainv,
    )


@dataclass(frozen=True, slots=True)
class EwaldSurfaceDisplay:
    q_sample_Ainv: FloatArray
    coating_intensity_density_A2_rad2_inv: FloatArray
    family_m: NDArray[np.int64]
    branch: NDArray[np.int64]
    ewald_residual_Ainv: FloatArray
    detector_visible_m0_q_gap_Ainv: float | None
    measure_id: str = "detector_visible_intrinsic_ewald_latent_density_A2_rad2_inv.v1"

    def __post_init__(self) -> None:
        points = _readonly_float_array(self.q_sample_Ainv, (None, 3), "q_sample_Ainv")
        shape = (points.shape[0],)
        density = _readonly_float_array(
            self.coating_intensity_density_A2_rad2_inv,
            shape,
            "coating_intensity_density_A2_rad2_inv",
        )
        residual = _readonly_float_array(
            self.ewald_residual_Ainv,
            shape,
            "ewald_residual_Ainv",
        )
        family = np.array(self.family_m, dtype=np.int64, copy=True, order="C")
        branch = np.array(self.branch, dtype=np.int64, copy=True, order="C")
        if (
            family.shape != shape
            or branch.shape != shape
            or np.any(family < 0)
            or np.any(~np.isin(branch, (0, 1, 2)))
            or np.any(density < 0.0)
            or np.any(residual < 0.0)
        ):
            raise ValueError("Ewald display arrays have invalid values")
        has_m0 = np.any(family == 0)
        gap = self.detector_visible_m0_q_gap_Ainv
        if has_m0:
            if gap is None or not math.isfinite(float(gap)) or float(gap) <= 0.0:
                raise ValueError("a detector-visible m=0 display requires a positive Q gap")
            gap = float(gap)
        elif gap is not None:
            raise ValueError("an m=0 Q gap requires detector-visible m=0 points")
        family.setflags(write=False)
        branch.setflags(write=False)
        for name, value in (
            ("q_sample_Ainv", points),
            ("coating_intensity_density_A2_rad2_inv", density),
            ("family_m", family),
            ("branch", branch),
            ("ewald_residual_Ainv", residual),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", gap)


def evaluate_nominal_ewald_surface(
    context: NominalEwaldContext,
    *,
    alpha_count: int,
    beta_count: int,
    alpha_max_deg: float,
) -> EwaldSurfaceDisplay:
    """Sample the continuous intrinsic coating after detector visibility only."""

    alpha_count = _integer(alpha_count, "alpha_count", positive=True)
    beta_count = _integer(beta_count, "beta_count", positive=True)
    alpha_max = math.radians(_positive(alpha_max_deg, "alpha_max_deg"))
    if alpha_max > np.pi:
        raise ValueError("alpha_max_deg must not exceed 180 degrees")
    alpha = (np.arange(alpha_count, dtype=np.float64) + 0.5) * (alpha_max / alpha_count)
    beta = (np.arange(beta_count, dtype=np.float64) + 0.5) * (2.0 * np.pi / beta_count)
    alpha_grid, beta_grid = np.meshgrid(alpha, beta, indexing="ij")
    points: list[FloatArray] = []
    intensities: list[FloatArray] = []
    families: list[NDArray[np.int64]] = []
    branches: list[NDArray[np.int64]] = []
    residuals: list[FloatArray] = []
    m0_gap: float | None = None
    for rod in context.geometry.coating.bragg_space.config.rods:
        selected_branches = (0,) if rod.family_m == 0 else (1, 2)
        for branch_value in selected_branches:
            evaluated = context.geometry.map_detector_visible_coating(
                rod=rod,
                branch=branch_value,
                alpha_rad=alpha_grid,
                beta_rad=beta_grid,
            )
            valid = evaluated.geometry.valid
            if not np.any(valid):
                continue
            if evaluated.detector_visible_m0_q_gap_Ainv is not None:
                m0_gap = evaluated.detector_visible_m0_q_gap_Ainv
            ewald = evaluated.geometry.ewald_geometry
            count = int(np.count_nonzero(valid))
            points.append(ewald.q_sample_Ainv[valid])
            intensities.append(evaluated.coating_intensity_density_A2_rad2_inv[valid])
            families.append(np.full(count, rod.family_m, dtype=np.int64))
            branches.append(np.full(count, branch_value, dtype=np.int64))
            residuals.append(ewald.ewald_residual_Ainv[valid])
    if not points:
        raise RuntimeError("nominal Ewald coating has no detector-visible display points")
    q_sample = np.concatenate(points, axis=0)
    intensity = np.concatenate(intensities)
    family = np.concatenate(families)
    branch = np.concatenate(branches)
    residual = np.concatenate(residuals)
    if float(np.max(residual, initial=0.0)) > 2.0e-13:
        raise FloatingPointError("nominal detector patch violates the Ewald identity")
    return EwaldSurfaceDisplay(
        q_sample_Ainv=q_sample,
        coating_intensity_density_A2_rad2_inv=intensity,
        family_m=family,
        branch=branch,
        ewald_residual_Ainv=residual,
        detector_visible_m0_q_gap_Ainv=m0_gap,
    )


@dataclass(frozen=True, slots=True)
class DetectorMacrobinImage:
    image_A2: FloatArray
    column_center_px: FloatArray
    row_center_px: FloatArray
    valid_source_count_min: NDArray[np.int64]
    coordinate_evaluation_count: int
    measure_id: str = "raw_detector_macrobin_fixed_quadrature_estimate_A2.v1"
    execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None

    def __post_init__(self) -> None:
        image = _readonly_float_array(self.image_A2, (None, None), "image_A2")
        if np.any(image < 0.0):
            raise ValueError("image_A2 must be nonnegative")
        column = _readonly_float_array(
            self.column_center_px,
            (image.shape[1],),
            "column_center_px",
        )
        row = _readonly_float_array(
            self.row_center_px,
            (image.shape[0],),
            "row_center_px",
        )
        valid_count = np.array(
            self.valid_source_count_min,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        if valid_count.shape != image.shape or np.any(valid_count < 0):
            raise ValueError("valid_source_count_min has invalid values")
        count = _integer(self.coordinate_evaluation_count, "coordinate_evaluation_count")
        if count < 1:
            raise ValueError("coordinate_evaluation_count must be positive")
        if self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
        }:
            raise ValueError("unsupported detector macrobin execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        if (self.execution_backend == "numba_cuda_source_averaged.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("execution_device must identify exactly the CUDA backend")
        valid_count.setflags(write=False)
        for name, value in (
            ("image_A2", image),
            ("column_center_px", column),
            ("row_center_px", row),
            ("valid_source_count_min", valid_count),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "coordinate_evaluation_count", count)


@dataclass(frozen=True, slots=True)
class DetectorCoordinateDensityImage:
    """Exact samples of the combined detector function at native pixel centers."""

    image_A2_per_px2: FloatArray
    column_center_px: FloatArray
    row_center_px: FloatArray
    valid_source_count: NDArray[np.int64]
    coordinate_evaluation_count: int
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"
    sampling_grid_id: str = "native_pixel_centers.v1"
    execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None

    def __post_init__(self) -> None:
        image = _readonly_float_array(
            self.image_A2_per_px2,
            (None, None),
            "image_A2_per_px2",
        )
        if np.any(image < 0.0):
            raise ValueError("image_A2_per_px2 must be nonnegative")
        column = _readonly_float_array(
            self.column_center_px,
            (image.shape[1],),
            "column_center_px",
        )
        row = _readonly_float_array(
            self.row_center_px,
            (image.shape[0],),
            "row_center_px",
        )
        valid_count = np.array(
            self.valid_source_count,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        if valid_count.shape != image.shape or np.any(valid_count < 0):
            raise ValueError("valid_source_count has invalid values")
        count = _integer(self.coordinate_evaluation_count, "coordinate_evaluation_count")
        if count < 0:
            raise ValueError("coordinate_evaluation_count must be nonnegative")
        if self.measure_id != "raw_detector_coordinate_density_A2_per_px2.v1":
            raise ValueError("unsupported detector-coordinate density measure")
        if self.sampling_grid_id != "native_pixel_centers.v1":
            raise ValueError("unsupported detector-coordinate sampling grid")
        if self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
        }:
            raise ValueError("unsupported detector-coordinate execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        if (self.execution_backend == "numba_cuda_source_averaged.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("execution_device must identify exactly the CUDA backend")
        valid_count.setflags(write=False)
        for name, value in (
            ("image_A2_per_px2", image),
            ("column_center_px", column),
            ("row_center_px", row),
            ("valid_source_count", valid_count),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "coordinate_evaluation_count", count)


def sample_detector_pixel_center_density(
    detector: SourceAveragedDetectorEwaldMeasure,
    *,
    execution_backend: str = "cpu",
    cuda_coordinate_chunk_size: int | None = None,
) -> DetectorCoordinateDensityImage:
    """Sample the final all-source, all-rod, all-root detector function once per pixel."""

    if not isinstance(detector, SourceAveragedDetectorEwaldMeasure):
        raise TypeError("detector must be SourceAveragedDetectorEwaldMeasure")
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be cpu or cuda")
    if cuda_coordinate_chunk_size is not None:
        if (
            isinstance(cuda_coordinate_chunk_size, bool)
            or not isinstance(cuda_coordinate_chunk_size, (int, np.integer))
            or int(cuda_coordinate_chunk_size) < 1
        ):
            raise ValueError("cuda_coordinate_chunk_size must be a positive integer")
        if execution_backend != "cuda":
            raise ValueError("cuda_coordinate_chunk_size requires the CUDA execution backend")
    rows, columns = detector.instrument.detector_shape_rc
    column_center = np.arange(columns, dtype=np.float64)
    row_center = np.arange(rows, dtype=np.float64)
    image = np.zeros((rows, columns), dtype=np.float64)
    valid_source_count = np.zeros((rows, columns), dtype=np.int64)
    instrument = detector.instrument
    detector_rotation = instrument.lab_from_detector.rotation
    column_step_lab = detector_rotation[:, 0] * instrument.detector_column_pitch_m
    row_step_lab = detector_rotation[:, 1] * instrument.detector_row_pitch_m
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    detector_zero_lab = (
        instrument.lab_from_detector.translation_m
        - reference_column * column_step_lab
        - reference_row * row_step_lab
    )
    sample_normal_lab = instrument.sample_from_lab.rotation[2]
    valid_state = detector.incident.states.valid
    origin_normal_coordinate = (
        detector.incident.states.sample_intersection_lab_m[valid_state] @ sample_normal_lab
    )
    if not origin_normal_coordinate.size:
        raise ValueError("detector contains no valid source state")
    minimum_origin_normal_coordinate = float(np.min(origin_normal_coordinate))
    row_chunk_size = max(1, _MAXIMUM_MACROBIN_COORDINATES_PER_CALL // columns)
    result_backend: str | None = None
    result_device: str | None = None
    coordinate_evaluation_count = 0
    for row_start in range(0, rows, row_chunk_size):
        row_stop = min(row_start + row_chunk_size, rows)
        column_grid, row_grid = np.broadcast_arrays(
            column_center[None, :],
            row_center[row_start:row_stop, None],
        )
        point_normal_coordinate = (
            detector_zero_lab @ sample_normal_lab
            + column_grid * (column_step_lab @ sample_normal_lab)
            + row_grid * (row_step_lab @ sample_normal_lab)
        )
        projection_scale = max(
            float(np.max(np.abs(point_normal_coordinate))),
            abs(minimum_origin_normal_coordinate),
            1.0,
        )
        projection_tolerance = 1024.0 * np.finfo(np.float64).eps * projection_scale
        candidate = (
            point_normal_coordinate > minimum_origin_normal_coordinate - projection_tolerance
        )
        selected = np.flatnonzero(candidate.ravel())
        if not selected.size:
            continue
        evaluated = detector.evaluate_detector_density_all_roots(
            column_grid.ravel()[selected],
            row_grid.ravel()[selected],
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
        )
        if np.any(evaluated.caustic):
            raise FloatingPointError("a native pixel center lies exactly on a detector caustic")
        if result_backend is None:
            result_backend = evaluated.execution_backend
            result_device = evaluated.execution_device
        elif (
            evaluated.execution_backend != result_backend
            or evaluated.execution_device != result_device
        ):
            raise RuntimeError("detector center sampling changed execution backend")
        chunk_image = image[row_start:row_stop].ravel()
        chunk_valid_count = valid_source_count[row_start:row_stop].ravel()
        chunk_image[selected] = evaluated.density_A2_per_px2
        chunk_valid_count[selected] = evaluated.valid_source_count
        coordinate_evaluation_count += int(selected.size)
    if result_backend is None:
        empty = detector.evaluate_detector_density_all_roots(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            execution_backend=execution_backend,
        )
        result_backend = empty.execution_backend
        result_device = empty.execution_device
    return DetectorCoordinateDensityImage(
        image_A2_per_px2=image,
        column_center_px=column_center,
        row_center_px=row_center,
        valid_source_count=valid_source_count,
        coordinate_evaluation_count=coordinate_evaluation_count,
        execution_backend=result_backend,
        execution_device=result_device,
    )


def integrate_detector_macrobins(
    detector: SourceAveragedDetectorEwaldMeasure,
    *,
    bin_size_px: int,
    gauss_order: int,
    execution_backend: str = "cpu",
) -> DetectorMacrobinImage:
    """Return a fixed-rule preview estimate over detector-aligned macrobins.

    The continuous detector callable remains authoritative. This deliberately
    low-cost rendering rule has no convergence claim and must not be used as a
    quantitative fitting observable.
    """

    if isinstance(bin_size_px, bool) or not isinstance(bin_size_px, int) or bin_size_px < 1:
        raise ValueError("bin_size_px must be a positive integer")
    if isinstance(gauss_order, bool) or not isinstance(gauss_order, int) or gauss_order < 2:
        raise ValueError("gauss_order must be an integer of at least two")
    if gauss_order % 2:
        raise ValueError("gauss_order must be even to avoid center caustics")
    rows, columns = detector.instrument.detector_shape_rc
    if rows % bin_size_px or columns % bin_size_px:
        raise ValueError("bin_size_px must divide both detector dimensions")
    column_center = -0.5 + (np.arange(columns // bin_size_px) + 0.5) * bin_size_px
    row_center = -0.5 + (np.arange(rows // bin_size_px) + 0.5) * bin_size_px
    nodes, weights = np.polynomial.legendre.leggauss(gauss_order)
    half_width = 0.5 * bin_size_px
    offset = half_width * nodes
    mapped_weight = half_width * weights
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    evaluation_kwargs = (
        {} if execution_backend == "cpu" else {"execution_backend": execution_backend}
    )
    shape = (row_center.size, column_center.size)
    image = np.zeros(shape, dtype=np.float64)
    valid_source_count_min = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
    row_chunk_size = max(1, _MAXIMUM_MACROBIN_COORDINATES_PER_CALL // column_center.size)
    result_backend: str | None = None
    result_device: str | None = None
    for row_offset, row_weight in zip(offset, mapped_weight, strict=True):
        for column_offset, column_weight in zip(offset, mapped_weight, strict=True):
            for row_start in range(0, row_center.size, row_chunk_size):
                row_stop = min(row_start + row_chunk_size, row_center.size)
                column_grid, row_grid = np.broadcast_arrays(
                    column_center[None, :] + column_offset,
                    row_center[row_start:row_stop, None] + row_offset,
                )
                evaluated = detector.evaluate_detector_density_all_roots(
                    column_grid,
                    row_grid,
                    **evaluation_kwargs,
                )
                if np.any(evaluated.caustic):
                    raise FloatingPointError("a detector quadrature node lies exactly on a caustic")
                image[row_start:row_stop] += (
                    row_weight * column_weight * evaluated.density_A2_per_px2
                )
                np.minimum(
                    valid_source_count_min[row_start:row_stop],
                    evaluated.valid_source_count,
                    out=valid_source_count_min[row_start:row_stop],
                )
                current_backend = getattr(
                    evaluated,
                    "execution_backend",
                    "numba_cpu_source_averaged.v1",
                )
                current_device = getattr(evaluated, "execution_device", None)
                if result_backend is None:
                    result_backend = current_backend
                    result_device = current_device
                elif (current_backend, current_device) != (result_backend, result_device):
                    raise RuntimeError("detector execution backend changed during quadrature")
    return DetectorMacrobinImage(
        image_A2=image,
        column_center_px=column_center,
        row_center_px=row_center,
        valid_source_count_min=valid_source_count_min,
        coordinate_evaluation_count=image.size * gauss_order**2,
        execution_backend=(result_backend or "numba_cpu_source_averaged.v1"),
        execution_device=result_device,
    )


__all__ = [
    "CONFIGURED_RESULT_SCHEMA_VERSION",
    "ConfiguredGeometryInputs",
    "ConfiguredSimulationInputs",
    "DetectorCoordinateDensityImage",
    "DetectorIntegerLMarkers",
    "DetectorMacrobinImage",
    "EwaldSurfaceDisplay",
    "GeometryOnlyEwaldContext",
    "LayerLEwaldRoots",
    "NominalEwaldContext",
    "ReciprocalSpaceDisplay",
    "SimulationConfiguration",
    "build_configured_geometry_inputs",
    "build_configured_simulation_inputs",
    "build_geometry_only_ewald_context",
    "build_nominal_ewald_context",
    "build_single_source_ewald_context",
    "build_source_averaged_detector",
    "build_source_averaged_structure_detector",
    "configured_rod_catalog_revision",
    "evaluate_nominal_ewald_surface",
    "evaluate_nominal_integer_l_markers",
    "integrate_detector_macrobins",
    "load_simulation_config",
    "load_strict_yaml_mapping",
    "rebind_configured_geometry_direct_basis",
    "rebind_configured_geometry_instrument",
    "rebind_configured_simulation_instrument",
    "rebind_source_averaged_detector_incidence_scan",
    "sample_configured_nominal_geometry_source",
    "sample_detector_pixel_center_density",
    "sample_reciprocal_space",
    "solve_layer_l_ewald_roots",
]
