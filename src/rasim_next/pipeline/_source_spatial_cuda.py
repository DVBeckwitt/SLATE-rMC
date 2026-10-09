"""Explicit CUDA terminal for the shared conditional Gaussian native pixel integral.

Physics and angular nodes remain on the host. Each block deposits one Gaussian;
float64 atomics only reorder independent, nonnegative event contributions.
"""

import math

import numba
import numpy as np
from numba import cuda

from rasim_next.pipeline.source_spatial import (
    _assemble_correlation_rectangle,
    _complementary_correlation_corner,
    _correlation_angle_coefficients,
    _correlation_corner,
    _normal_interval_probability,
    _pixel_kernel_parameters,
    _rectangle_probability,
    _signed_unit_rectangle_probability,
)


@numba.njit(nogil=True)
def _prepare_spatial_parameters(mean, factor, mass, nodes, weights):
    parameters = np.empty((len(mean), 6))
    # The canonical angle partition has at most two panels below |rho|=0.925.
    coefficients = np.zeros((len(mean), 3, 2 * len(nodes)))
    counts = np.zeros(len(mean), np.int64)
    for i in range(len(mean)):
        sx, beta, conditional_y = _pixel_kernel_parameters(factor[i])
        parameters[i] = (mean[i, 0], mean[i, 1], sx, beta, conditional_y, mass[i])
        rho = beta * sx / math.hypot(beta * sx, conditional_y)
        if 0.0 < abs(rho) < 0.925:
            values = _correlation_angle_coefficients(beta * sx, conditional_y, nodes, weights)
            counts[i] = values.shape[1]
            coefficients[i, :, : values.shape[1]] = values
    return (parameters, coefficients, counts)


@cuda.jit(device=True)
def _pixel_probability(p, column, row, coefficients, count, nodes, weights, radius):
    mx, my, sx, beta, conditional_y = (p[0], p[1], p[2], p[3], p[4])
    xlow, xhigh, ylow, yhigh = (column - 0.5, column + 0.5, row - 0.5, row + 0.5)
    lo, hi = (
        (xlow - mx) / sx if (xlow - mx) / sx > -radius else -radius,
        (xhigh - mx) / sx if (xhigh - mx) / sx < radius else radius,
    )
    if hi <= lo:
        return 0.0
    slope = beta * sx
    sy = math.hypot(slope, conditional_y)
    rho = slope / sy
    if slope == 0.0 or abs(rho) >= 1.0:
        return _rectangle_probability(
            mx, my, sx, beta, conditional_y, xlow, xhigh, ylow, yhigh, nodes, weights, radius
        )
    ma, mb = (my + slope * lo, my + slope * hi)
    if (
        ylow <= (ma if ma < mb else mb) - radius * conditional_y
        and yhigh >= (ma if ma > mb else mb) + radius * conditional_y
    ):
        return _normal_interval_probability(lo, hi)
    yl, yh = ((ylow - my) / sy, (yhigh - my) / sy)
    if abs(rho) >= 0.925 and conditional_y / sy > 1e-12:
        if abs(yl) > 12.0 or abs(yh) > 12.0 or abs(lo) > 12.0 or abs(hi) > 12.0:
            return _rectangle_probability(
                mx, my, sx, beta, conditional_y, xlow, xhigh, ylow, yhigh, nodes, weights, radius
            )
        sign = 1.0 if rho > 0.0 else -1.0
        base = _signed_unit_rectangle_probability(lo, hi, yl, yh, sign)
        a = _complementary_correlation_corner(hi, sign * yh, conditional_y / sy, nodes, weights)
        b = _complementary_correlation_corner(lo, sign * yh, conditional_y / sy, nodes, weights)
        c = _complementary_correlation_corner(hi, sign * yl, conditional_y / sy, nodes, weights)
        d = _complementary_correlation_corner(lo, sign * yl, conditional_y / sy, nodes, weights)
        value, accepted = _assemble_correlation_rectangle(base, a, b, c, d, sign, True)
        if accepted:
            return value
    elif count > 0:
        base = _normal_interval_probability(lo, hi) * _normal_interval_probability(yl, yh)
        a, b, c, d = (0.0, 0.0, 0.0, 0.0)
        for j in range(count):
            exponent, sine, weight = (coefficients[0, j], coefficients[1, j], coefficients[2, j])
            a += weight * _correlation_corner(exponent, sine, hi, yh)
            b += weight * _correlation_corner(exponent, sine, lo, yh)
            c += weight * _correlation_corner(exponent, sine, hi, yl)
            d += weight * _correlation_corner(exponent, sine, lo, yl)
        value, accepted = _assemble_correlation_rectangle(base, a, b, c, d, 1.0, False)
        if accepted:
            return value
    return _rectangle_probability(
        mx, my, sx, beta, conditional_y, xlow, xhigh, ylow, yhigh, nodes, weights, radius
    )


@cuda.jit(fastmath=False)
def _deposit(
    parameters, coefficients, counts, nodes, weights, radius, row_offset, column_offset, image
):
    i = cuda.blockIdx.x
    p = parameters[i]
    if p[5] == 0.0:
        return
    mx, my, sx, beta, conditional_y = (p[0], p[1], p[2], p[3], p[4])
    low = math.ceil(mx - radius * sx - 0.5)
    high = math.floor(mx + radius * sx + 0.5)
    if low < column_offset:
        low = column_offset
    if high >= column_offset + image.shape[1]:
        high = column_offset + image.shape[1] - 1
    for column in range(low + cuda.threadIdx.x, high + 1, cuda.blockDim.x):
        ym = my + beta * (column - mx)
        yr = 0.5 * abs(beta) + radius * conditional_y
        bottom, top = math.ceil(ym - yr - 0.5), math.floor(ym + yr + 0.5)
        if bottom < row_offset:
            bottom = row_offset
        if top >= row_offset + image.shape[0]:
            top = row_offset + image.shape[0] - 1
        for row in range(bottom, top + 1):
            value = p[5] * _pixel_probability(
                p, column, row, coefficients[i], counts[i], nodes, weights, radius
            )
            cuda.atomic.add(image, (row - row_offset, column - column_offset), value)


def deposit_gaussian_pixels_cuda(
    mean, factor, mass, shape, nodes, weights, radius, row_offset, column_offset, out
):
    """Run actual CUDA or raise; transfer, synchronization and readback are explicit."""
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA native spatial execution was requested, but no CUDA device is available"
        )
    if cuda.get_current_device().compute_capability < (6, 0):
        raise RuntimeError(
            "CUDA native spatial execution requires float64 atomic support (compute capability 6.0)"
        )
    parameters, coefficients, counts = _prepare_spatial_parameters(
        mean, factor, mass, nodes, weights
    )
    host_image = np.zeros(shape) if out is None else out
    device_image = cuda.to_device(host_image)
    device_inputs = tuple(
        cuda.to_device(a) for a in (parameters, coefficients, counts, nodes, weights)
    )
    _deposit[len(mean), 128](*device_inputs, radius, row_offset, column_offset, device_image)
    device_image.copy_to_host(host_image)
    cuda.synchronize()
    return host_image
