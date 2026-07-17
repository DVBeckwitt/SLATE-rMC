"""Generate the canonical Bi2Se3 detector image with the accepted default case."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry.instrument import (
    AxisRotation,
    CompiledInstrument,
    InstrumentConfiguration,
    compile_instrument,
)
from rasim_next.materials import material_optics, read_crystal
from rasim_next.pipeline.simulate import simulate_ordered
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.sampling.mosaic import (
    WrappedMosaicParameters,
    manuscript_axisymmetric_v1_orientation_quadrature,
)
from rasim_next.sampling.source import sample_gaussian_source_rays

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    Path(tempfile.gettempdir())
    / "rasim-next"
    / "slice6b_bi2se3_detector_101x32.png"
)
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def configure_matplotlib() -> None:
    """Select the non-interactive renderer or fail before the long simulation."""

    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise SystemExit(
            "This optional image tool requires Matplotlib; install it with "
            "`python -m pip install matplotlib`."
        ) from error
    matplotlib.use("Agg")


def build_default_instrument() -> CompiledInstrument:
    """Compile the fixed detector-native instrument used by the canonical image."""

    identity = np.eye(3)
    zero = np.zeros(3)
    detector_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    return compile_instrument(
        InstrumentConfiguration(
            axis_rotations=(
                AxisRotation(
                    axis_lab=np.array([1.0, 0.0, 0.0]),
                    angle_rad=0.20943951023931953,
                    pivot_lab_m=zero,
                ),
            ),
            lab_from_goniometer_zero=RigidTransform(
                identity,
                zero,
                FrameId.GONIOMETER,
                FrameId.LAB,
            ),
            goniometer_from_sample=RigidTransform(
                identity,
                zero,
                FrameId.SAMPLE,
                FrameId.GONIOMETER,
            ),
            sample_from_crystal=RigidTransform(
                identity,
                zero,
                FrameId.CRYSTAL,
                FrameId.SAMPLE,
            ),
            lab_from_detector=RigidTransform(
                detector_rotation,
                np.array([0.0, 0.075, 0.0]),
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
            detector_shape_rc=(3000, 3000),
            detector_row_pitch_m=1.0e-4,
            detector_column_pitch_m=1.0e-4,
            detector_reference_coordinate_px=(1453.12, 1596.422),
            sample_width_m=2.0e-4,
            sample_length_m=5.0e-4,
            film_thickness_A=500.0,
        )
    )


def write_detector_image(image_A2: NDArray[np.float64]) -> None:
    """Write the detector-native mass image with the original logarithmic styling."""

    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    positive = image_A2[image_A2 > 0.0]
    if positive.size == 0:
        raise RuntimeError("the default simulation produced no positive detector mass")

    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    shown = axis.imshow(
        np.ma.masked_less_equal(image_A2, 0.0),
        origin="upper",
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
    )
    axis.set_title(
        "Bi$_2$Se$_3$ detector — 20 rays \N{MULTIPLICATION SIGN} 40 MC draws "
        "from one mosaic distribution"
    )
    axis.set_xlabel("Detector column (px)")
    axis.set_ylabel("Detector row (px)")
    figure.colorbar(shown, ax=axis, label="Assigned detector mass (Å²; log scale)")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)


def main() -> None:
    configure_matplotlib()
    samples = sample_gaussian_source_rays(
        mean_origin_lab_m=np.array([0.0, -0.020, 0.0]),
        mean_direction_lab=np.array([0.0, 1.0, 0.0]),
        transverse_axes_lab=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        spatial_sigma_m=np.full(2, 0.05e-3 * FWHM_TO_SIGMA),
        divergence_sigma_rad=np.full(2, 0.0008726646259971648 * FWHM_TO_SIGMA),
        mean_wavelength_A=1.540592925,
        wavelength_sigma_A=1.540592925 * 0.007,
        sample_count=20,
        seed=1729,
        polarization_state_id="UNITY_APPROXIMATION",
    )
    crystal = read_crystal(
        ROOT / "examples" / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    material = material_optics(crystal, samples.wavelength_A)
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(
            gaussian_sigma_rad=math.radians(1.0) * FWHM_TO_SIGMA,
            lorentzian_half_width_rad=math.radians(0.5) / 2.0,
            lorentzian_probability=0.0,
        ),
        reciprocal_basis_Ainv=reciprocal.basis_Ainv,
        alpha_cell_count=1,
        azimuth_cell_count=32,
    )
    result = simulate_ordered(
        crystal=crystal,
        incident_samples=samples,
        material=material,
        instrument=build_default_instrument(),
        orientations=orientations,
        phase_population_weight=1.0,
        polarization_policy_id="UNITY_APPROXIMATION",
        polarization_provenance="explicit all-one event weights; no physical polarization model",
        selection_seed=20260716,
        draw_count=40,
    )

    image_A2 = result.deposition.image_A2
    write_detector_image(image_A2)
    print(f"Wrote {OUTPUT}")
    print(
        f"valid_rays={np.count_nonzero(result.incident.states.valid)} "
        f"selections={result.selection.event_id.size} "
        f"mosaic_support={np.unique(orientations.alpha_rad).size}x"
        f"{np.unique(orientations.azimuth_rad).size} "
        f"nonzero_pixels={np.count_nonzero(image_A2)} "
        f"detector_mass_A2={np.sum(image_A2, dtype=np.float64):.17g}"
    )


if __name__ == "__main__":
    main()
