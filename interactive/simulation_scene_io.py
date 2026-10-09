"""Small canonical Simulator scene preparation; no intensity or reciprocal evaluation."""

import json
import math
from dataclasses import dataclass

import numpy as np
from archive_storage import storage_files
from experiment_scene import SceneGeometry
from job_lifecycle import JobResult
from native_simulation_state import native_draft_from_document
from simulation_io import _stop, canonical_configuration
from simulation_state import simulation_draft_from_document


@dataclass(frozen=True, slots=True)
class SimulatorGeometry:
    draft: object
    geometry: SceneGeometry
    instrument: object
    base_lab_m: tuple[float, float, float]
    mount_lab_m: tuple[float, float, float]
    mount_axes_lab: tuple[tuple[float, float, float], ...]
    incidence_deg: float

    @property
    def nbytes(self):
        return 65536


def prepare_simulator_geometry(argument, control):
    request = json.loads(argument)
    _stop(control)
    native = request["kind"] == "native"
    draft = (native_draft_from_document if native else simulation_draft_from_document)(
        request["draft"]
    )
    if native:
        from native_simulation_io import canonical_native

        bound = canonical_native(draft)[0]
        instrument, source = bound.instrument, bound.source_definition
        rotations = ()
        base = mount = instrument.lab_from_sample.translation_m
        mount_rotation = np.eye(3)
    else:
        from rasim_next.pipeline.configured_simulation import _compile_instrument

        config = canonical_configuration(draft, storage_files(request.get("storage_json", "")))
        instrument, source = _compile_instrument(config.instrument), config.source
        rotations = tuple(
            (r.pivot_lab_m, r.axis_lab, math.radians(r.angle_deg))
            for r in config.instrument.axis_rotations
        )
        base = config.instrument.lab_from_goniometer_zero.translation_m
        # The compiler owns the ordered rigid chain. Recover its moving mount frame.
        mount_rotation = (
            instrument.lab_from_sample.rotation
            @ np.asarray(config.instrument.goniometer_from_sample.rotation).T
        )
        mount = instrument.lab_from_sample.translation_m - mount_rotation @ np.asarray(
            config.instrument.goniometer_from_sample.translation_m
        )
    geometry = SceneGeometry.from_instrument(
        instrument, source.mean_origin_lab_m, source.mean_direction_lab, rotations
    )
    normal = np.asarray(geometry.sample_axes_lab[2])
    incidence = math.degrees(
        math.asin(float(np.clip(-normal @ np.asarray(geometry.beam_direction_lab), -1, 1)))
    )
    _stop(control)
    value = SimulatorGeometry(
        draft,
        geometry,
        instrument,
        tuple(base),
        tuple(mount),
        tuple(tuple(mount_rotation[:, i]) for i in range(3)),
        incidence,
    )
    return JobResult(value, value.nbytes)
