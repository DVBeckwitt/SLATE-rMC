"""Strict YAML boundary and reusable calculations for configured Bi2Se3 views."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from painted_ewald import (
    BraggSpaceConfig,
    ContinuousEwaldCoating,
    MosaicBraggSpace,
    MosaicParameters,
    Rod,
    enumerate_rods_within_ewald_sphere,
)
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    IncidentSampleBatch,
    MaterialOptics,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry import build_incident_states
from rasim_next.geometry.instrument import (
    AxisRotation,
    CompiledInstrument,
    InstrumentConfiguration,
    compile_instrument,
)
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.materials import CrystalStructure, material_optics, read_crystal
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure
from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.sampling.source import sample_gaussian_source_rays

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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
    layers: int
    normalization: str
    shared_disorder_epsilon: float


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
    physics_revision: str = field(init=False)
    render_revision: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "cif_sha256": hashlib.sha256(self.material.cif_path.read_bytes()).hexdigest(),
            "material_phase_id": self.material.phase_id,
            "source": asdict(self.source),
            "instrument": asdict(self.instrument),
            "mosaic": asdict(self.mosaic),
            "structure_factor": asdict(self.structure_factor),
            "bragg": asdict(self.bragg),
            "weights": asdict(self.weights),
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


def _artifact(value: Any, path: str) -> ArtifactConfiguration:
    data = _mapping(value, path, required={"enabled", "filename"})
    filename = _string(data["filename"], f"{path}.filename")
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".png":
        raise ValueError(f"{path}.filename must be one PNG basename")
    return ArtifactConfiguration(
        enabled=_boolean(data["enabled"], f"{path}.enabled"),
        filename=filename,
    )


def _load_one_yaml(path: Path) -> dict[str, Any]:
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
    """Load one strict, config-relative ``rasim-simulation-v1`` document."""

    config_path = Path(path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    document = _mapping(
        _load_one_yaml(config_path),
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
    if schema != "rasim-simulation-v1":
        raise ValueError("schema_version must be rasim-simulation-v1")

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
    )
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
    )
    if source.seed < 0:
        raise ValueError("source.seed must be nonnegative")

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
        lab_from_detector=_transform(
            instrument_data["lab_from_detector"], "instrument.lab_from_detector"
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
    if mosaic.lorentzian_probability < 1.0 and mosaic.gaussian_sigma_deg == 0.0:
        raise ValueError("active Gaussian mosaic width must be nonzero")
    if mosaic.lorentzian_probability > 0.0 and mosaic.lorentzian_hwhm_deg == 0.0:
        raise ValueError("active Lorentzian mosaic width must be nonzero")

    sf_data = _mapping(
        document["structure_factor"],
        "structure_factor",
        required={"model_id", "layers", "normalization", "shared_disorder_epsilon"},
    )
    structure_factor = StructureFactorConfiguration(
        model_id=_string(sf_data["model_id"], "structure_factor.model_id"),
        layers=_integer(sf_data["layers"], "structure_factor.layers", positive=True),
        normalization=_string(sf_data["normalization"], "structure_factor.normalization"),
        shared_disorder_epsilon=_finite(
            sf_data["shared_disorder_epsilon"], "structure_factor.shared_disorder_epsilon"
        ),
    )
    if structure_factor.model_id != "bi2se3_finite_2h.v1":
        raise ValueError("structure_factor.model_id must be bi2se3_finite_2h.v1")
    if structure_factor.normalization != "FINITE_TOTAL":
        raise ValueError("structure_factor.normalization must be FINITE_TOTAL")
    if not 0.0 <= structure_factor.shared_disorder_epsilon <= 1.0:
        raise ValueError("structure_factor.shared_disorder_epsilon must lie in [0, 1]")

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
        optional={"detector_execution_backend"},
    )
    numerics = NumericalConfiguration(
        worker_count=_integer(
            numerical_data["worker_count"], "numerics.worker_count", positive=True
        ),
        detector_execution_backend=_string(
            numerical_data.get("detector_execution_backend", "cpu"),
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


def _sample_source(
    source: SourceConfiguration,
    *,
    sample_count: int | None = None,
) -> IncidentSampleBatch:
    return sample_gaussian_source_rays(
        mean_origin_lab_m=np.asarray(source.mean_origin_lab_m),
        mean_direction_lab=np.asarray(source.mean_direction_lab),
        transverse_axes_lab=np.asarray(source.transverse_axes_lab),
        spatial_sigma_m=np.asarray(source.spatial_sigma_m),
        divergence_sigma_rad=np.asarray(source.divergence_sigma_rad),
        mean_wavelength_A=source.mean_wavelength_A,
        wavelength_sigma_A=source.wavelength_sigma_A,
        sample_count=source.sample_count if sample_count is None else sample_count,
        seed=source.seed,
        polarization_state_id=source.polarization_state_id,
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
class ConfiguredSimulationInputs:
    config: SimulationConfiguration
    samples: IncidentSampleBatch
    instrument: CompiledInstrument
    crystal: CrystalStructure
    incident: IncidentTransportResult
    reciprocal: ReciprocalLattice
    rods: tuple[Rod, ...]
    mosaic: MosaicParameters
    strength: Bi2Se3TwoHStrength
    bragg_space: MosaicBraggSpace
    material: MaterialOptics


def build_configured_simulation_inputs(
    config: SimulationConfiguration,
) -> ConfiguredSimulationInputs:
    """Build shared immutable physics once, without allocating any rendered field."""

    if not isinstance(config, SimulationConfiguration):
        raise TypeError("config must be SimulationConfiguration")
    samples = _sample_source(config.source)
    instrument = _compile_instrument(config.instrument)
    crystal = read_crystal(config.material.cif_path, phase_id=config.material.phase_id)
    material = material_optics(crystal, samples.wavelength_A)
    incident = build_incident_states(samples, material, instrument)
    valid_index = np.flatnonzero(incident.states.valid)
    if not valid_index.size:
        raise ValueError("configured source produced no valid incident state")
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    maximum_air_k = 2.0 * np.pi / float(np.min(incident.states.wavelength_A[valid_index]))
    rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=reciprocal.basis_Ainv,
        k_norm_Ainv=maximum_air_k,
        population=config.bragg.rod_population,
    )
    if not config.bragg.include_detector_visible_m0:
        rods = tuple(rod for rod in rods if rod.family_m != 0)
    mosaic = _mosaic(config.mosaic)
    strength = Bi2Se3TwoHStrength(
        crystal=crystal,
        layers=config.structure_factor.layers,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
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


def build_source_averaged_detector(
    inputs: ConfiguredSimulationInputs,
) -> SourceAveragedDetectorEwaldMeasure:
    """Compile the continuous all-state detector pullback only when requested."""

    return SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
        rods=inputs.rods,
        mosaic=inputs.mosaic,
        strength_model=inputs.strength,
        incident=inputs.incident,
        material=inputs.material,
        instrument=inputs.instrument,
        phase_population_weight=inputs.config.weights.phase_population,
        polarization_weight=inputs.config.weights.polarization,
        worker_count=inputs.config.numerics.worker_count,
    )


@dataclass(frozen=True, slots=True)
class NominalEwaldContext:
    geometry: DetectorEwaldMeasure
    incident: IncidentTransportResult


def build_nominal_ewald_context(inputs: ConfiguredSimulationInputs) -> NominalEwaldContext:
    """Build the explicitly nominal mean incident state for one Ewald visualization."""

    samples = _sample_source(inputs.config.source, sample_count=1)
    material = material_optics(inputs.crystal, samples.wavelength_A)
    incident = build_incident_states(samples, material, inputs.instrument)
    if not bool(incident.states.valid[0]):
        raise ValueError("nominal mean source state is invalid")
    air_k = 2.0 * np.pi / float(samples.wavelength_A[0])
    reachable_keys = {
        (rod.h, rod.k)
        for rod in enumerate_rods_within_ewald_sphere(
            reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
            k_norm_Ainv=air_k,
            population=inputs.config.bragg.rod_population,
        )
    }
    rods = tuple(rod for rod in inputs.rods if (rod.h, rod.k) in reachable_keys)
    bragg = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
            crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
            rods=rods,
            mosaic=inputs.mosaic,
            k_norm_Ainv=air_k,
        ),
        inputs.strength,
    )
    coating = ContinuousEwaldCoating(
        bragg,
        ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
    )
    geometry = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=inputs.instrument,
    )
    return NominalEwaldContext(geometry=geometry, incident=incident)


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
            if evaluated.detector_visible_m0_q_gap_Ainv is not None:
                m0_gap = evaluated.detector_visible_m0_q_gap_Ainv
            if not np.any(valid):
                continue
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
    per_rod_image_A2: FloatArray
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
        per_rod = _readonly_float_array(
            self.per_rod_image_A2,
            (*image.shape, None),
            "per_rod_image_A2",
        )
        if np.any(per_rod < 0.0) or not np.allclose(
            image,
            np.sum(per_rod, axis=-1, dtype=np.float64),
            rtol=0.0,
            atol=1024.0 * np.finfo(np.float64).eps * max(float(np.max(image, initial=0.0)), 1.0),
        ):
            raise ValueError("macrobin image must equal its physical rod sum")
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
        count = _integer(
            self.coordinate_evaluation_count,
            "coordinate_evaluation_count",
            positive=True,
        )
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
            ("per_rod_image_A2", per_rod),
            ("column_center_px", column),
            ("row_center_px", row),
            ("valid_source_count_min", valid_count),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "coordinate_evaluation_count", count)


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
    column_grid, row_grid, row_offset_grid, column_offset_grid = np.broadcast_arrays(
        column_center[None, :, None, None],
        row_center[:, None, None, None],
        offset[None, None, :, None],
        offset[None, None, None, :],
    )
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    evaluation_kwargs = (
        {} if execution_backend == "cpu" else {"execution_backend": execution_backend}
    )
    evaluated = detector.evaluate_detector_coordinates_all_roots(
        column_grid + column_offset_grid,
        row_grid + row_offset_grid,
        **evaluation_kwargs,
    )
    if np.any(evaluated.caustic):
        raise FloatingPointError("a detector quadrature node lies exactly on a caustic")
    node_weight = mapped_weight[:, None] * mapped_weight[None, :]
    per_rod = np.sum(
        evaluated.per_rod_density_A2_per_px2 * node_weight[None, None, :, :, None],
        axis=(2, 3),
        dtype=np.float64,
    )
    return DetectorMacrobinImage(
        image_A2=np.sum(per_rod, axis=-1, dtype=np.float64),
        per_rod_image_A2=per_rod,
        column_center_px=column_center,
        row_center_px=row_center,
        valid_source_count_min=np.min(evaluated.valid_source_count, axis=(2, 3)),
        coordinate_evaluation_count=int(evaluated.density_A2_per_px2.size),
        execution_backend=getattr(
            evaluated,
            "execution_backend",
            "numba_cpu_source_averaged.v1",
        ),
        execution_device=getattr(evaluated, "execution_device", None),
    )


__all__ = [
    "ConfiguredSimulationInputs",
    "DetectorMacrobinImage",
    "EwaldSurfaceDisplay",
    "NominalEwaldContext",
    "ReciprocalSpaceDisplay",
    "SimulationConfiguration",
    "build_configured_simulation_inputs",
    "build_nominal_ewald_context",
    "build_source_averaged_detector",
    "evaluate_nominal_ewald_surface",
    "integrate_detector_macrobins",
    "load_simulation_config",
    "sample_reciprocal_space",
]
