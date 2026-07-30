from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import gettempdir
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "figures" / "render_bi2se3_publication_mapping.py"
SPEC = importlib.util.spec_from_file_location("render_bi2se3_publication_mapping", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIGURE
SPEC.loader.exec_module(FIGURE)


def _external_scratch_directory(prefix: str) -> Path:
    base = Path(gettempdir()).resolve()
    if base == ROOT or base.is_relative_to(ROOT):
        base = ROOT.parent
    path = base / f"{prefix}-{uuid4().hex}"
    path.mkdir()
    return path


def test_publication_defaults_preserve_bi2se3_while_applying_declared_overrides() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )

    configured = FIGURE.apply_figure_settings(base, FIGURE.FigureSettings())

    assert base.instrument.axis_rotations[0].angle_deg == 5.0
    assert configured.material == base.material
    assert configured.instrument.axis_rotations[0].angle_deg == 10.0
    assert configured.source.sample_count == 1
    assert configured.mosaic.gaussian_sigma_deg == 2.0
    assert configured.mosaic.lorentzian_hwhm_deg == 0.2
    assert configured.mosaic.lorentzian_probability == 0.1
    assert configured.instrument.lab_from_detector == base.instrument.lab_from_detector
    inputs = build_configured_simulation_inputs(configured)
    np.testing.assert_array_equal(
        inputs.samples.origin_lab_m, (configured.source.mean_origin_lab_m,)
    )
    np.testing.assert_array_equal(
        inputs.samples.direction_lab,
        (configured.source.mean_direction_lab,),
    )
    np.testing.assert_array_equal(
        inputs.samples.wavelength_A, (configured.source.mean_wavelength_A,)
    )
    np.testing.assert_array_equal(inputs.samples.source_weight, (1.0,))

    wrong_phase = replace(base, material=replace(base.material, phase_id="bi2te3"))
    with pytest.raises(ValueError, match="only the Bi2Se3 phase"):
        FIGURE.apply_figure_settings(wrong_phase, FIGURE.FigureSettings())

    for invalid in (
        {"ewald_image_size": 3},
        {"ewald_tile_row_count": 0},
        {"ewald_worker_count": 0},
        {"ewald_worker_count": True},
        {"ewald_worker_count": 2.5},
        {"ewald_image_size": 12.0},
        {"detector_image_size": 64.5},
        {"dpi": 320.5},
    ):
        with pytest.raises(ValueError):
            FIGURE.FigureSettings(**invalid)

    numpy_scalar_settings = FIGURE.FigureSettings(
        incidence_deg=np.float64(10.0),
        ewald_image_size=np.int64(64),
    )
    json.dumps(asdict(numpy_scalar_settings))
    assert type(numpy_scalar_settings.incidence_deg) is float
    assert type(numpy_scalar_settings.ewald_image_size) is int


def test_illustrative_detector_tilt_changes_only_rotation_about_reference_pivot() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )
    inputs = build_configured_simulation_inputs(
        FIGURE.apply_figure_settings(base, FIGURE.FigureSettings())
    )
    ideal = inputs.instrument

    tilted = FIGURE.tilt_detector_about_reference(
        ideal,
        column_tilt_deg=0.0,
        row_tilt_deg=20.0,
    )
    angle = np.deg2rad(20.0)
    expected_about_local_row = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    expected_rotation = ideal.lab_from_detector.rotation @ expected_about_local_row

    np.testing.assert_array_equal(
        tilted.lab_from_detector.translation_m,
        ideal.lab_from_detector.translation_m,
    )
    np.testing.assert_allclose(
        tilted.lab_from_detector.rotation,
        expected_rotation,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        tilted.lab_from_detector.rotation.T @ tilted.lab_from_detector.rotation,
        np.eye(3),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.linalg.det(tilted.lab_from_detector.rotation),
        1.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert tilted.detector_shape_rc == ideal.detector_shape_rc
    assert tilted.detector_reference_coordinate_px == ideal.detector_reference_coordinate_px
    assert tilted.detector_column_pitch_m == ideal.detector_column_pitch_m
    assert tilted.detector_row_pitch_m == ideal.detector_row_pitch_m
    assert tilted.sample_geometry_revision == ideal.sample_geometry_revision


def test_detector_display_tiles_native_column_row_coordinates_without_transpose() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )
    instrument = build_configured_simulation_inputs(
        FIGURE.apply_figure_settings(base, FIGURE.FigureSettings())
    ).instrument

    calls: list[tuple[int, ...]] = []

    class CoordinateProbe:
        def evaluate_detector_density_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
        ) -> object:
            column = np.asarray(column_px)
            row = np.asarray(row_px)
            calls.append(column.shape)
            return SimpleNamespace(
                density_A2_per_px2=100.0 * row + column,
                valid_source_count=np.ones(column.shape, dtype=np.int64),
                caustic=np.zeros(column.shape, dtype=np.bool_),
                measure_id="coordinate_probe.v1",
                execution_backend=execution_backend,
            )

    display = FIGURE.evaluate_detector_display(
        CoordinateProbe(),
        instrument=instrument,
        image_size=3,
        tile_row_count=2,
    )
    rows, columns = instrument.detector_shape_rc
    column_centers = 0.5 * (
        np.linspace(-0.5, columns - 0.5, 4)[:-1] + np.linspace(-0.5, columns - 0.5, 4)[1:]
    )
    row_centers = 0.5 * (
        np.linspace(-0.5, rows - 0.5, 4)[:-1] + np.linspace(-0.5, rows - 0.5, 4)[1:]
    )
    expected = 100.0 * row_centers[:, None] + column_centers[None, :]

    np.testing.assert_array_equal(display.density_A2_per_px2, expected)
    assert calls == [(2, 3), (1, 3)]
    assert np.all(display.valid)
    assert not np.any(display.caustic)
    assert display.measure_id == "coordinate_probe.v1"
    assert display.execution_backend == "cpu"
    assert not display.density_A2_per_px2.flags.writeable
    assert not display.column_edges_px.flags.writeable
    assert not display.row_edges_px.flags.writeable

    class ScalarProbe:
        def evaluate_detector_density_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
        ) -> object:
            del column_px, row_px
            return SimpleNamespace(
                density_A2_per_px2=1.0,
                valid_source_count=1,
                caustic=False,
                measure_id="coordinate_probe.v1",
                execution_backend=execution_backend,
            )

    with pytest.raises(ValueError, match="tile arrays"):
        FIGURE.evaluate_detector_display(
            ScalarProbe(),
            instrument=instrument,
            image_size=3,
            tile_row_count=2,
        )

    class ChangingMetadataProbe(CoordinateProbe):
        def evaluate_detector_density_all_roots(
            self,
            column_px: object,
            row_px: object,
            *,
            execution_backend: str = "cpu",
        ) -> object:
            result = super().evaluate_detector_density_all_roots(
                column_px,
                row_px,
                execution_backend=execution_backend,
            )
            return SimpleNamespace(
                **{
                    **vars(result),
                    "measure_id": f"coordinate_probe_{len(calls)}.v1",
                }
            )

    calls.clear()
    with pytest.raises(ValueError, match="metadata changed"):
        FIGURE.evaluate_detector_display(
            ChangingMetadataProbe(),
            instrument=instrument,
            image_size=3,
            tile_row_count=2,
        )

    with pytest.raises(ValueError, match="nonempty"):
        FIGURE.DetectorDisplay(
            density_A2_per_px2=np.empty((0, 1)),
            valid=np.empty((0, 1), dtype=np.bool_),
            caustic=np.empty((0, 1), dtype=np.bool_),
            column_edges_px=np.asarray((0.0, 1.0)),
            row_edges_px=np.asarray((0.0,)),
            measure_id="coordinate_probe.v1",
            execution_backend="cpu",
        )


def test_detector_surface_raster_preserves_native_density_and_pose() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )
    ideal = build_configured_simulation_inputs(
        FIGURE.apply_figure_settings(base, FIGURE.FigureSettings())
    ).instrument
    tilted = FIGURE.tilt_detector_about_reference(
        ideal,
        column_tilt_deg=0.0,
        row_tilt_deg=20.0,
    )
    first_density = np.arange(6.0).reshape(2, 3)
    second_density = first_density + 100.0
    displays = tuple(
        FIGURE.DetectorDisplay(
            density_A2_per_px2=density,
            valid=np.ones((2, 3), dtype=np.bool_),
            caustic=np.zeros((2, 3), dtype=np.bool_),
            column_edges_px=np.asarray((10.0, 20.0, 30.0, 40.0)),
            row_edges_px=np.asarray((-1.0, 1.0, 3.0)),
            measure_id="detector_native_A2_per_px2.v1",
            execution_backend="cpu",
        )
        for density in (first_density, second_density)
    )

    ideal_surface = FIGURE.detector_surface_raster(displays[0], ideal)
    tilted_surface = FIGURE.detector_surface_raster(displays[1], tilted)

    assert ideal_surface.vertices_lab_m.shape == (3, 4, 3)
    np.testing.assert_array_equal(ideal_surface.density_A2_per_px2, first_density)
    np.testing.assert_array_equal(tilted_surface.density_A2_per_px2, second_density)
    np.testing.assert_array_equal(ideal_surface.valid, displays[0].valid)
    np.testing.assert_array_equal(ideal_surface.caustic, displays[0].caustic)
    assert ideal_surface.measure_id == displays[0].measure_id

    column_grid, row_grid = np.meshgrid(
        displays[0].column_edges_px,
        displays[0].row_edges_px,
        indexing="xy",
    )
    reference_column, reference_row = ideal.detector_reference_coordinate_px
    detector_local_m = np.stack(
        (
            (column_grid - reference_column) * ideal.detector_column_pitch_m,
            (row_grid - reference_row) * ideal.detector_row_pitch_m,
            np.zeros_like(column_grid),
        ),
        axis=-1,
    )
    expected_lab_m = (
        detector_local_m @ ideal.lab_from_detector.rotation.T
        + ideal.lab_from_detector.translation_m
    )
    np.testing.assert_allclose(
        ideal_surface.vertices_lab_m,
        expected_lab_m,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert not np.allclose(tilted_surface.vertices_lab_m, ideal_surface.vertices_lab_m)
    assert not ideal_surface.density_A2_per_px2.flags.writeable
    assert not ideal_surface.vertices_lab_m.flags.writeable


def test_schematic_detector_coarsening_uses_linear_cell_averages() -> None:
    density = np.arange(24.0).reshape(4, 6)
    display = FIGURE.DetectorDisplay(
        density_A2_per_px2=density,
        valid=np.ones((4, 6), dtype=np.bool_),
        caustic=np.eye(4, 6, dtype=np.bool_),
        column_edges_px=np.arange(7.0),
        row_edges_px=np.arange(5.0),
        measure_id="detector_native_A2_per_px2.v1",
        execution_backend="cpu",
    )

    coarse = FIGURE._coarsen_detector_display(display, maximum_cell_count=3)

    expected = density.reshape(2, 2, 3, 2).mean(axis=(1, 3))
    np.testing.assert_array_equal(coarse.density_A2_per_px2, expected)
    np.testing.assert_array_equal(coarse.column_edges_px, (0.0, 2.0, 4.0, 6.0))
    np.testing.assert_array_equal(coarse.row_edges_px, (0.0, 2.0, 4.0))
    assert coarse.valid.shape == expected.shape
    assert coarse.caustic.shape == expected.shape
    assert np.all(coarse.valid)
    assert np.count_nonzero(coarse.caustic) == 2
    assert not display.density_A2_per_px2.flags.writeable
    assert not display.valid.flags.writeable

    assert FIGURE._coarsened_cell_count(360, 180) == 180
    assert FIGURE._coarsened_cell_count(359, 180) == 180
    source = np.arange(77.0).reshape(7, 11)
    rebinned = FIGURE._conservative_uniform_rebin_2d(source, (4, 6))
    assert rebinned.shape == (4, 6)
    np.testing.assert_allclose(np.mean(rebinned), np.mean(source), rtol=2.0e-15)
    scalar_oracle = np.zeros((4, 6), dtype=np.float64)
    for target_row in range(4):
        target_row_bounds = (target_row / 4.0, (target_row + 1) / 4.0)
        for target_column in range(6):
            target_column_bounds = (target_column / 6.0, (target_column + 1) / 6.0)
            for source_row in range(7):
                row_overlap = max(
                    0.0,
                    min(target_row_bounds[1], (source_row + 1) / 7.0)
                    - max(target_row_bounds[0], source_row / 7.0),
                )
                for source_column in range(11):
                    column_overlap = max(
                        0.0,
                        min(target_column_bounds[1], (source_column + 1) / 11.0)
                        - max(target_column_bounds[0], source_column / 11.0),
                    )
                    scalar_oracle[target_row, target_column] += (
                        source[source_row, source_column] * row_overlap * column_overlap * 24.0
                    )
    np.testing.assert_allclose(rebinned, scalar_oracle, rtol=2.0e-15, atol=2.0e-14)
    np.testing.assert_allclose(
        FIGURE._conservative_uniform_rebin_2d(np.ones((7, 11)), (4, 6)),
        1.0,
    )


def test_detector_visible_ewald_patch_includes_m0_and_only_paints_active_panel() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        load_simulation_config,
    )

    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )
    configured = FIGURE.apply_figure_settings(base, FIGURE.FigureSettings())
    inputs = build_configured_simulation_inputs(configured)
    nominal = build_nominal_ewald_context(inputs)

    sphere = FIGURE.evaluate_detector_visible_ewald_patch(
        configured,
        nominal,
        image_size=16,
        tile_row_count=3,
        worker_count=1,
    )

    assert 2 <= sphere.intensity_density_A2_per_sr.shape[0] < 16
    assert sphere.intensity_density_A2_per_sr.shape[1] == 16
    assert sphere.q_vertices_sample_Ainv.shape == (
        sphere.intensity_density_A2_per_sr.shape[0] + 1,
        sphere.intensity_density_A2_per_sr.shape[1] + 1,
        3,
    )
    assert sphere.outgoing_direction_centers_sample.shape == (
        *sphere.intensity_density_A2_per_sr.shape,
        3,
    )
    assert sphere.inverse_branch_count.shape == sphere.intensity_density_A2_per_sr.shape
    assert sphere.caustic.shape == sphere.intensity_density_A2_per_sr.shape
    assert sphere.detector_visible.shape == sphere.intensity_density_A2_per_sr.shape
    assert sphere.m0_intensity_density_A2_per_sr.shape == (sphere.intensity_density_A2_per_sr.shape)
    assert np.any(sphere.detector_visible)
    assert np.all(sphere.intensity_density_A2_per_sr[~sphere.detector_visible] == 0.0)
    assert np.all(sphere.m0_intensity_density_A2_per_sr[~sphere.detector_visible] == 0.0)
    assert np.any(sphere.m0_intensity_density_A2_per_sr > 0.0)
    assert np.all(sphere.intensity_density_A2_per_sr >= sphere.m0_intensity_density_A2_per_sr)
    k_magnitude_Ainv = np.linalg.norm(nominal.ki_sample_Ainv)
    used_vertices = np.zeros(sphere.q_vertices_sample_Ainv.shape[:2], dtype=np.bool_)
    visible_rows, visible_columns = np.nonzero(sphere.detector_visible)
    for row_offset, column_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
        used_vertices[visible_rows + row_offset, visible_columns + column_offset] = True
    centered = sphere.q_vertices_sample_Ainv[used_vertices] + nominal.ki_sample_Ainv
    np.testing.assert_allclose(
        np.linalg.norm(centered, axis=-1),
        k_magnitude_Ainv,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.all(np.isfinite(sphere.intensity_density_A2_per_sr))
    assert np.all(sphere.intensity_density_A2_per_sr >= 0.0)
    assert sphere.detector_visible_m0_q_gap_Ainv > 0.0
    assert sphere.measure_id == "detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1"
    assert (
        sphere.selection_id
        == "configured_active_panel_regular_all_inverse_preimages_including_m0.v1"
    )
    assert sphere.execution_backend == "serial_cpu.v1"
    assert sphere.worker_count == 1
    assert not sphere.q_vertices_sample_Ainv.flags.writeable
    assert not sphere.outgoing_direction_centers_sample.flags.writeable
    assert not sphere.intensity_density_A2_per_sr.flags.writeable
    assert not sphere.caustic.flags.writeable

    for index in zip(*np.nonzero(sphere.detector_visible), strict=True):
        oracle = nominal.geometry.evaluate_detector_visible_ewald_directions(
            0.5 * (sphere.column_edges_px[index[1]] + sphere.column_edges_px[index[1] + 1]),
            0.5 * (sphere.row_edges_px[index[0]] + sphere.row_edges_px[index[0] + 1]),
            rods=nominal.rods,
        )
        np.testing.assert_allclose(
            sphere.intensity_density_A2_per_sr[index],
            oracle.density_A2_per_sr,
            rtol=5.0e-12,
            atol=0.0,
        )
        assert sphere.inverse_branch_count[index] == np.sum(oracle.per_rod_inverse_branch_count)
        assert sphere.caustic[index] == np.any(oracle.caustic)
        break


def test_detector_visible_ewald_patch_spawn_workers_match_serial() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        load_simulation_config,
    )

    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )
    configured = FIGURE.apply_figure_settings(base, FIGURE.FigureSettings())
    nominal = build_nominal_ewald_context(build_configured_simulation_inputs(configured))
    kwargs = dict(image_size=12, tile_row_count=2)
    serial = FIGURE.evaluate_detector_visible_ewald_patch(
        configured,
        nominal,
        worker_count=1,
        **kwargs,
    )
    parallel = FIGURE.evaluate_detector_visible_ewald_patch(
        configured,
        nominal,
        worker_count=2,
        **kwargs,
    )

    for name in (
        "q_vertices_sample_Ainv",
        "outgoing_direction_centers_sample",
        "intensity_density_A2_per_sr",
        "m0_intensity_density_A2_per_sr",
        "inverse_branch_count",
        "caustic",
        "detector_visible",
        "column_edges_px",
        "row_edges_px",
    ):
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
    assert parallel.maximum_ewald_residual_Ainv == serial.maximum_ewald_residual_Ainv
    assert parallel.detector_visible_m0_q_gap_Ainv == (serial.detector_visible_m0_q_gap_Ainv)
    assert parallel.execution_backend == "process_cpu_spawn.v1"
    assert parallel.multiprocessing_start_method == "spawn"
    assert parallel.worker_count == 2


def test_scale_resolved_display_nodes_cover_lorentzian_core_tail_and_sf_peaks() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        load_simulation_config,
    )

    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )
    inputs = build_configured_simulation_inputs(
        FIGURE.apply_figure_settings(base, FIGURE.FigureSettings())
    )
    coarse = FIGURE.select_scale_resolved_mosaic_alpha_nodes(inputs, count=32)
    refined = FIGURE.select_scale_resolved_mosaic_alpha_nodes(inputs, count=64)
    coarse_deg = np.rad2deg(coarse)
    refined_deg = np.rad2deg(refined)
    assert coarse_deg[0] < 0.01
    assert np.min(np.abs(coarse_deg - 0.2)) < 0.005
    assert coarse_deg[-1] > 179.0
    assert np.max(np.diff(refined_deg)) < np.max(np.diff(coarse_deg))

    nominal = build_nominal_ewald_context(inputs)
    basis = FIGURE._incident_aligned_basis(nominal.ki_sample_Ainv)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.linalg.det(basis), 1.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        nominal.ki_sample_Ainv @ basis,
        (0.0, 0.0, np.linalg.norm(nominal.ki_sample_Ainv)),
        rtol=0.0,
        atol=1.0e-14,
    )
    rod = inputs.bragg_space.config.rods[0]
    axial = FIGURE.structure_resolved_axial_nodes_Ainv(inputs, rod, background_count=48)
    b3_norm = np.linalg.norm(inputs.reciprocal.basis_Ainv[:, 2])
    ell = axial / b3_norm
    lower, upper = inputs.bragg_space.rod_u_bounds_Ainv(rod)
    integer_l = np.arange(np.ceil(lower / b3_norm), np.floor(upper / b3_norm) + 1)
    for value in integer_l:
        assert np.min(np.abs(ell - value)) < 1.0e-12


def test_publication_outputs_reject_repository_paths() -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        FIGURE._external_output_directory(ROOT)


def test_publication_output_names_formats_and_overwrite_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="case_name"):
        FIGURE._case_stem(FIGURE.FigureSettings(), "../bad")
    with pytest.raises(ValueError, match="formats"):
        FIGURE._validated_formats(("png", "png"))
    with pytest.raises(ValueError, match="formats"):
        FIGURE._validated_formats(("svg",))

    with pytest.raises(FileExistsError, match="--overwrite"):
        FIGURE._require_available_outputs((SCRIPT,), overwrite=False)
    FIGURE._require_available_outputs((SCRIPT,), overwrite=True)

    tmp_path = _external_scratch_directory("rasim-publication-test")
    try:
        case_stem = "atomic-case"
        output_directory = tmp_path / "output"
        output_directory.mkdir()
        known = FIGURE._known_case_artifact_paths(output_directory, case_stem)
        assert len(known) == 13
        stale_pdf = output_directory / f"{case_stem}-projection-schematic.pdf"
        replaced_png = output_directory / f"{case_stem}-publication-panels.png"
        stale_pdf.write_text("stale", encoding="utf-8")
        replaced_png.write_text("old", encoding="utf-8")
        with pytest.raises(FileExistsError, match="--overwrite"):
            FIGURE._require_available_outputs(known, overwrite=False)
        unsafe = output_directory / f"{case_stem}-04-tilted-detector-mapping.pdf"
        unsafe.mkdir()
        with pytest.raises(ValueError, match="must not be directories"):
            FIGURE._require_available_outputs(known, overwrite=True)
        assert unsafe.is_dir()
        unsafe.rmdir()

        staging = tmp_path / "staging"
        staging.mkdir()
        staged_png = staging / replaced_png.name
        staged_manifest = staging / f"{case_stem}-publication-manifest.json"
        staged_png.write_text("new", encoding="utf-8")
        staged_manifest.write_text("{}\n", encoding="utf-8")
        desired = (replaced_png, output_directory / staged_manifest.name)
        with pytest.raises(ValueError, match="exactly match"):
            FIGURE._install_staged_artifacts(
                (staged_manifest, staged_png),
                desired_paths=desired,
                known_case_paths=known,
                overwrite=True,
                backup_directory=staging / "unused-backup",
            )
        assert stale_pdf.read_text(encoding="utf-8") == "stale"
        assert replaced_png.read_text(encoding="utf-8") == "old"
        installed = FIGURE._install_staged_artifacts(
            (staged_png, staged_manifest),
            desired_paths=desired,
            known_case_paths=known,
            overwrite=True,
            backup_directory=staging / "backup",
        )

        assert installed == desired
        assert replaced_png.read_text(encoding="utf-8") == "new"
        assert installed[-1].read_text(encoding="utf-8") == "{}\n"
        assert not stale_pdf.exists()
        assert (staging / "backup" / stale_pdf.name).read_text(encoding="utf-8") == "stale"

        config = tmp_path / "config.yaml"
        config.write_text("revision: one\n", encoding="utf-8")
        provenance = FIGURE._capture_source_provenance(config)
        FIGURE._require_source_provenance_unchanged(provenance)
        config.write_text("revision: two\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="configuration changed"):
            FIGURE._require_source_provenance_unchanged(provenance)
    finally:
        shutil.rmtree(tmp_path)


def test_publication_install_failure_restores_prior_manifest_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _external_scratch_directory("rasim-publication-rollback")
    try:
        output_directory = root / "output"
        staging = root / "staging"
        output_directory.mkdir()
        staging.mkdir()
        case_stem = "rollback-case"
        old_manifest = output_directory / f"{case_stem}-publication-manifest.json"
        old_figure = output_directory / f"{case_stem}-publication-panels.png"
        old_manifest.write_text("old manifest\n", encoding="utf-8")
        old_figure.write_text("old figure\n", encoding="utf-8")
        staged_figure = staging / old_figure.name
        staged_manifest = staging / old_manifest.name
        staged_figure.write_text("new figure\n", encoding="utf-8")
        staged_manifest.write_text("new manifest\n", encoding="utf-8")
        known = FIGURE._known_case_artifact_paths(output_directory, case_stem)
        desired = (old_figure, old_manifest)

        original_replace = Path.replace
        replace_sources: list[Path] = []

        def fail_new_manifest(source: Path, target: Path) -> Path:
            replace_sources.append(source)
            if source == staged_manifest:
                raise OSError("injected manifest installation failure")
            return original_replace(source, target)

        monkeypatch.setattr(Path, "replace", fail_new_manifest)
        with pytest.raises(OSError, match="injected manifest"):
            FIGURE._install_staged_artifacts(
                (staged_figure, staged_manifest),
                desired_paths=desired,
                known_case_paths=known,
                overwrite=True,
                backup_directory=staging / "backup",
            )

        assert replace_sources[0] == old_manifest
        assert replace_sources.index(staged_figure) < replace_sources.index(staged_manifest)
        assert replace_sources[-1].name == old_manifest.name
        assert old_figure.read_text(encoding="utf-8") == "old figure\n"
        assert old_manifest.read_text(encoding="utf-8") == "old manifest\n"
    finally:
        shutil.rmtree(root)


def test_publication_manifest_declares_visible_m0_patch_and_spawn_execution() -> None:
    root = _external_scratch_directory("rasim-publication-manifest")
    try:
        output = root / "figure.png"
        output.write_bytes(b"figure")
        instrument = SimpleNamespace(
            lab_from_detector=SimpleNamespace(
                rotation=np.eye(3),
                translation_m=np.zeros(3),
            )
        )
        ewald = SimpleNamespace(
            measure_id="detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1",
            selection_id="configured_active_panel_regular_all_inverse_preimages_including_m0.v1",
            intensity_density_A2_per_sr=np.asarray(((1.0, 2.0), (3.0, 0.0))),
            m0_intensity_density_A2_per_sr=np.asarray(((0.1, 0.2), (0.3, 0.0))),
            detector_visible=np.asarray(((True, True), (True, False))),
            caustic=np.zeros((2, 2), dtype=np.bool_),
            inverse_branch_count=np.ones((2, 2), dtype=np.int64),
            column_edges_px=np.asarray((-0.5, 0.5, 1.5)),
            row_edges_px=np.asarray((-0.5, 0.5, 1.5)),
            detector_visible_m0_q_gap_Ainv=0.7,
            maximum_ewald_residual_Ainv=1.0e-15,
            execution_backend="process_cpu_spawn.v1",
            requested_worker_count=2,
            worker_count=2,
            multiprocessing_start_method="spawn",
        )
        detector = SimpleNamespace(
            measure_id="raw_detector_coordinate_density_A2_per_px2.v1",
            density_A2_per_px2=np.asarray(((1.0, 2.0), (3.0, 4.0))),
            caustic=np.zeros((2, 2), dtype=np.bool_),
        )
        data = SimpleNamespace(
            settings=FIGURE.FigureSettings(ewald_worker_count=2),
            inputs=SimpleNamespace(
                config=SimpleNamespace(physics_revision="test-physics"),
                samples=SimpleNamespace(source_revision="test-source"),
                instrument=instrument,
            ),
            tilted_instrument=instrument,
            reciprocal=SimpleNamespace(
                measure_id="latent_bragg_density_A2_rad2_inv.v1",
                q_sample_Ainv=np.zeros((3, 3)),
            ),
            ewald=ewald,
            ideal_detector=detector,
            tilted_detector=detector,
            nominal=SimpleNamespace(rods=(object(), object())),
            mosaic_alpha_nodes_rad=np.asarray((0.0, 1.0)),
            reciprocal_logical_candidate_count=4,
            reciprocal_candidate_evaluation_count=3,
            schematic_rays=SimpleNamespace(direction_lab=np.zeros((2, 3))),
            timings_s={"ewald_pointwise_density": 1.0},
        )
        payload = FIGURE._publication_manifest_payload(
            data,
            provenance=FIGURE.SourceProvenance(
                config_path=ROOT / "config.yaml",
                config_identity="config.yaml",
                config_sha256="config-sha",
                renderer_path=SCRIPT,
                renderer_identity="renderer.py",
                renderer_sha256="renderer-sha",
            ),
            outputs=(output,),
        )

        assert payload["schema_version"] == "rasim-bi2se3-publication-mapping-v5"
        assert payload["measures"]["ewald"] == ewald.measure_id
        assert payload["measures"]["ewald_selection"] == ewald.selection_id
        assert payload["ewald_point_sampling"]["sphere_policy"] == (
            "canonical_top_exit_and_configured_active_panel_visibility.v1"
        )
        assert payload["ewald_point_sampling"]["m0_policy"] == (
            "regular_nonzero_support_with_positive_Q_gap;collapsed_direct_Q0_excluded.v1"
        )
        assert payload["counts"]["ewald_visible_samples"] == 3
        assert payload["counts"]["ewald_m0_positive_samples"] == 3
        assert payload["execution"]["ewald"] == {
            "backend": "process_cpu_spawn.v1",
            "requested_worker_count": 2,
            "used_worker_count": 2,
            "start_method": "spawn",
            "child_blas_thread_limit": 1,
        }
    finally:
        shutil.rmtree(root)


def test_publication_cli_stages_complete_inventory_before_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _external_scratch_directory("rasim-publication-cli")
    builds: list[tuple[Path, object]] = []
    try:

        def fake_build(config_path: Path, settings: object) -> object:
            builds.append((config_path, settings))
            return SimpleNamespace(settings=settings)

        def fake_render(
            data: object,
            *,
            output_directory: Path,
            formats: tuple[str, ...],
            include_standalone: bool,
            case_stem: str,
        ) -> tuple[Path, ...]:
            del data
            paths = FIGURE._figure_paths(
                output_directory,
                case_stem=case_stem,
                formats=formats,
                include_standalone=include_standalone,
            )
            for path in paths:
                path.write_bytes(path.name.encode())
            return paths

        def fake_manifest(
            data: object,
            *,
            provenance: object,
            output_directory: Path,
            outputs: tuple[Path, ...],
            case_stem: str,
        ) -> Path:
            del data, provenance, outputs
            path = output_directory / f"{case_stem}-publication-manifest.json"
            path.write_text("{}\n", encoding="utf-8")
            return path

        monkeypatch.setattr(FIGURE, "_configure_matplotlib", lambda: None)
        monkeypatch.setattr(FIGURE, "build_publication_data", fake_build)
        monkeypatch.setattr(FIGURE, "_render_staged_figures", fake_render)
        monkeypatch.setattr(FIGURE, "_write_manifest", fake_manifest)
        arguments = (
            "--output-directory",
            str(root),
            "--case-name",
            "cli-case",
            "--formats",
            "png",
            "--ewald-worker-count",
            "2",
            "--skip-standalone",
        )

        assert FIGURE.main(arguments) == 0
        expected = (
            root / "cli-case-publication-panels.png",
            root / "cli-case-projection-schematic.png",
            root / "cli-case-publication-manifest.json",
        )
        assert all(path.is_file() for path in expected)
        assert len(builds) == 1
        assert builds[0][1].ewald_worker_count == 2
        assert not tuple(root.glob(".rmp-*"))

        with pytest.raises(FileExistsError, match="--overwrite"):
            FIGURE.main(arguments)
        assert len(builds) == 1
    finally:
        shutil.rmtree(root)


def test_ewald_only_cli_skips_reciprocal_and_detector_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _external_scratch_directory("rasim-ewald-only-cli")
    builds: list[object] = []
    try:

        def fake_build(config_path: Path, settings: object) -> object:
            del config_path
            builds.append(settings)
            return SimpleNamespace(settings=settings)

        def forbidden_full_build(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("full publication data must not be built in Ewald-only mode")

        def fake_render(
            data: object,
            *,
            output_directory: Path,
            formats: tuple[str, ...],
            case_stem: str,
        ) -> tuple[Path, ...]:
            del data, formats
            path = output_directory / f"{case_stem}-visible-m0-ewald-surface.png"
            path.write_text("figure\n", encoding="utf-8")
            return (path,)

        def fake_manifest(
            data: object,
            *,
            provenance: object,
            output_directory: Path,
            outputs: tuple[Path, ...],
            case_stem: str,
        ) -> Path:
            del data, provenance, outputs
            path = output_directory / f"{case_stem}-ewald-only-manifest.json"
            path.write_text("{}\n", encoding="utf-8")
            return path

        monkeypatch.setattr(FIGURE, "_configure_matplotlib", lambda: None)
        monkeypatch.setattr(FIGURE, "build_publication_data", forbidden_full_build)
        monkeypatch.setattr(FIGURE, "build_ewald_only_data", fake_build)
        monkeypatch.setattr(FIGURE, "_render_staged_ewald_only", fake_render)
        monkeypatch.setattr(FIGURE, "_write_ewald_only_manifest", fake_manifest)
        arguments = (
            "--output-directory",
            str(root),
            "--case-name",
            "ewald-case",
            "--formats",
            "png",
            "--ewald-worker-count",
            "2",
            "--only-ewald",
        )

        assert FIGURE.main(arguments) == 0
        assert (root / "ewald-case-visible-m0-ewald-surface.png").is_file()
        assert (root / "ewald-case-ewald-only-manifest.json").is_file()
        assert len(builds) == 1
        assert builds[0].ewald_worker_count == 2
    finally:
        shutil.rmtree(root)


def test_partial_render_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        FIGURE._parser().parse_args(("--only-ewald", "--only-schematic"))


def test_schematic_only_artifact_inventory_is_distinct_and_manifest_last() -> None:
    root = Path(gettempdir()).resolve()
    schematic = FIGURE._schematic_only_artifact_paths(
        root,
        case_stem="projection-case",
        formats=("png", "pdf"),
    )
    ewald = FIGURE._ewald_only_artifact_paths(
        root,
        case_stem="projection-case",
        formats=("png", "pdf"),
    )

    assert schematic == (
        root / "projection-case-projection-schematic-only.png",
        root / "projection-case-projection-schematic-only.pdf",
        root / "projection-case-schematic-only-manifest.json",
    )
    assert schematic[-1].name.endswith("-manifest.json")
    assert set(schematic).isdisjoint(ewald)


def test_schematic_only_builder_evaluates_detector_textures_at_display_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasim_next.pipeline import configured_simulation

    configured_instrument = object()
    tilted_instrument = object()
    inputs = SimpleNamespace(instrument=configured_instrument, incident=object())
    nominal = object()
    ewald = object()
    schematic_rays = object()
    detector_displays: list[tuple[object, int, int]] = []
    detector_model = SimpleNamespace(
        rebind_geometry=lambda *, incident, instrument: (incident, instrument)
    )

    monkeypatch.setattr(configured_simulation, "load_simulation_config", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        configured_simulation, "build_configured_simulation_inputs", lambda config: inputs
    )
    monkeypatch.setattr(configured_simulation, "build_nominal_ewald_context", lambda value: nominal)
    monkeypatch.setattr(
        configured_simulation,
        "build_source_averaged_detector",
        lambda value: detector_model,
    )
    monkeypatch.setattr(FIGURE, "apply_figure_settings", lambda base, settings: object())
    monkeypatch.setattr(
        FIGURE,
        "select_scale_resolved_mosaic_alpha_nodes",
        lambda value, *, count: np.zeros(count),
    )
    monkeypatch.setattr(
        FIGURE,
        "evaluate_detector_visible_ewald_patch",
        lambda *args, **kwargs: ewald,
    )
    monkeypatch.setattr(
        FIGURE,
        "sample_scale_resolved_reciprocal_space",
        lambda *args, **kwargs: pytest.fail("schematic-only mode sampled reciprocal space"),
    )
    monkeypatch.setattr(
        FIGURE,
        "tilt_detector_about_reference",
        lambda *args, **kwargs: tilted_instrument,
    )

    def fake_detector_display(
        detector: object,
        *,
        instrument: object,
        image_size: int,
        tile_row_count: int,
    ) -> object:
        del detector
        detector_displays.append((instrument, image_size, tile_row_count))
        return SimpleNamespace(instrument=instrument)

    monkeypatch.setattr(FIGURE, "evaluate_detector_display", fake_detector_display)
    monkeypatch.setattr(FIGURE, "_sample_schematic_rays", lambda *args, **kwargs: schematic_rays)
    settings = FIGURE.FigureSettings(
        detector_image_size=1440,
        detector_tile_row_count=32,
        schematic_detector_cell_count=7,
        ewald_image_size=4,
        ewald_worker_count=1,
    )

    data = FIGURE.build_schematic_only_data(ROOT / "unused.yaml", settings)

    assert detector_displays == ([(configured_instrument, 7, 7), (tilted_instrument, 7, 7)])
    assert data.ewald is ewald
    assert data.schematic_rays is schematic_rays
    assert not hasattr(data, "reciprocal")


def test_schematic_ray_endpoints_are_same_lab_rays_intersecting_both_detector_planes() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        load_simulation_config,
    )

    settings = FIGURE.FigureSettings(
        mosaic_alpha_count=16,
        ewald_image_size=4,
        ewald_worker_count=1,
    )
    base = load_simulation_config(
        ROOT / "configs" / "bi2se3_simulation.yaml",
        repository_root=ROOT,
    )
    inputs = build_configured_simulation_inputs(FIGURE.apply_figure_settings(base, settings))
    nominal = build_nominal_ewald_context(inputs)
    tilted_instrument = FIGURE.tilt_detector_about_reference(
        inputs.instrument,
        column_tilt_deg=settings.tilt_column_deg,
        row_tilt_deg=settings.tilt_row_deg,
    )
    rays = FIGURE._sample_schematic_rays(
        nominal,
        tilted_instrument=tilted_instrument,
        alpha_nodes_rad=FIGURE.select_scale_resolved_mosaic_alpha_nodes(
            inputs,
            count=settings.mosaic_alpha_count,
        ),
    )
    origins = np.broadcast_to(rays.origin_lab_m, rays.direction_lab.shape)

    for instrument, stored_points in (
        (rays.ideal_instrument, rays.ideal_point_lab_m),
        (rays.tilted_instrument, rays.tilted_point_lab_m),
    ):
        projected = FIGURE.project_detector_rays(origins, rays.direction_lab, instrument)
        assert np.all(projected.valid)
        assert np.all(projected.ray_distance_m > 0.0)
        np.testing.assert_allclose(stored_points, projected.point_lab_m, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            stored_points - origins,
            projected.ray_distance_m[:, None] * rays.direction_lab,
            rtol=0.0,
            atol=4.0 * np.finfo(np.float64).eps,
        )
        detector_points = instrument.lab_from_detector.inverse().apply_point(stored_points)
        np.testing.assert_allclose(
            detector_points[:, 2],
            0.0,
            rtol=0.0,
            atol=4.0 * np.finfo(np.float64).eps,
        )


def test_schematic_only_manifest_records_scope_coordinates_and_ewald_proof() -> None:
    root = _external_scratch_directory("rasim-schematic-manifest")
    output = root / "schematic.png"
    output.write_bytes(b"schematic")
    try:
        transform = SimpleNamespace(
            rotation=np.eye(3),
            translation_m=np.asarray((0.0, 0.075, 0.0)),
        )
        instrument = SimpleNamespace(lab_from_detector=transform)
        ewald = SimpleNamespace(
            measure_id="detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1",
            selection_id="configured_active_panel_regular_all_inverse_preimages_including_m0.v1",
            intensity_density_A2_per_sr=np.asarray(((1.0, 2.0), (3.0, 0.0))),
            m0_intensity_density_A2_per_sr=np.asarray(((0.1, 0.2), (0.3, 0.0))),
            detector_visible=np.asarray(((True, True), (True, False))),
            caustic=np.asarray(((False, True), (False, False))),
            inverse_branch_count=np.asarray(((1, 2), (3, 0))),
            column_edges_px=np.asarray((-0.5, 0.5, 1.5)),
            row_edges_px=np.asarray((-0.5, 0.5, 1.5)),
            detector_visible_m0_q_gap_Ainv=0.7077572188469623,
            maximum_ewald_residual_Ainv=8.881784197001252e-16,
            execution_backend="process_cpu_spawn.v1",
            requested_worker_count=2,
            worker_count=2,
            multiprocessing_start_method="spawn",
        )
        detector = SimpleNamespace(
            measure_id="raw_detector_coordinate_density_A2_per_px2.v1",
            density_A2_per_px2=np.asarray(((1.0, 2.0), (3.0, 4.0))),
            valid=np.asarray(((True, True), (True, False))),
            caustic=np.asarray(((False, False), (True, False))),
            execution_backend="numba_cpu_source_averaged.v1",
        )
        data = SimpleNamespace(
            settings=FIGURE.FigureSettings(ewald_worker_count=2),
            inputs=SimpleNamespace(
                config=SimpleNamespace(physics_revision="test-physics"),
                samples=SimpleNamespace(source_revision="test-source"),
                instrument=instrument,
            ),
            tilted_instrument=instrument,
            ewald=ewald,
            ideal_detector=detector,
            tilted_detector=detector,
            schematic_rays=SimpleNamespace(direction_lab=np.zeros((2, 3))),
            timings_s={"ewald_pointwise_density": 1.0},
        )

        payload = FIGURE._schematic_only_manifest_payload(
            data,
            provenance=FIGURE.SourceProvenance(
                config_path=ROOT / "config.yaml",
                config_identity="config.yaml",
                config_sha256="config-sha",
                renderer_path=SCRIPT,
                renderer_identity="renderer.py",
                renderer_sha256="renderer-sha",
            ),
            outputs=(output,),
        )

        assert payload["schema_version"] == "rasim-bi2se3-projection-schematic-v1"
        assert payload["build_scope"] == {
            "reciprocal_space": "not_evaluated",
            "ewald_patch": "evaluated_at_requested_schematic_resolution",
            "detector_maps": "evaluated_directly_at_schematic_detector_cell_count",
        }
        assert payload["coordinates"]["detector_coordinate_order"] == "(column_px, row_px)"
        assert payload["coordinates"]["array_index_order"] == "[row, column]"
        assert payload["ewald_proof"] == {
            "detector_visible_m0_q_gap_Ainv": 0.7077572188469623,
            "maximum_ewald_residual_Ainv": 8.881784197001252e-16,
        }
        assert payload["counts"]["ewald_caustic_samples"] == 1
        assert payload["counts"]["ewald_regular_inverse_preimages"] == 6
        assert payload["counts"]["configured_detector_valid_samples"] == 3
    finally:
        shutil.rmtree(root)


def test_schematic_only_cli_skips_full_and_ewald_only_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _external_scratch_directory("rasim-schematic-only-cli")
    builds: list[object] = []
    try:

        def fake_build(config_path: Path, settings: object) -> object:
            del config_path
            builds.append(settings)
            return SimpleNamespace(settings=settings)

        def forbidden_build(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("unselected publication data must not be built")

        def fake_render(
            data: object,
            *,
            output_directory: Path,
            formats: tuple[str, ...],
            case_stem: str,
        ) -> tuple[Path, ...]:
            del data, formats
            path = output_directory / f"{case_stem}-projection-schematic-only.png"
            path.write_text("figure\n", encoding="utf-8")
            return (path,)

        def fake_manifest(
            data: object,
            *,
            provenance: object,
            output_directory: Path,
            outputs: tuple[Path, ...],
            case_stem: str,
        ) -> Path:
            del data, provenance, outputs
            path = output_directory / f"{case_stem}-schematic-only-manifest.json"
            path.write_text("{}\n", encoding="utf-8")
            return path

        monkeypatch.setattr(FIGURE, "_configure_matplotlib", lambda: None)
        monkeypatch.setattr(FIGURE, "build_publication_data", forbidden_build)
        monkeypatch.setattr(FIGURE, "build_ewald_only_data", forbidden_build)
        monkeypatch.setattr(FIGURE, "build_schematic_only_data", fake_build)
        monkeypatch.setattr(FIGURE, "_render_staged_schematic_only", fake_render)
        monkeypatch.setattr(FIGURE, "_write_schematic_only_manifest", fake_manifest)
        arguments = (
            "--output-directory",
            str(root),
            "--case-name",
            "projection-case",
            "--formats",
            "png",
            "--ewald-worker-count",
            "3",
            "--only-schematic",
        )

        assert FIGURE.main(arguments) == 0
        assert (root / "projection-case-projection-schematic-only.png").is_file()
        assert (root / "projection-case-schematic-only-manifest.json").is_file()
        assert len(builds) == 1
        assert builds[0].ewald_worker_count == 3
    finally:
        shutil.rmtree(root)
