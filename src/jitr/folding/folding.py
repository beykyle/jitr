"""General folding kernels and numerical helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss


def gaussian_fold(
    R_grid: float | np.ndarray,
    M_callable: Callable[[np.ndarray], np.ndarray],
    t: float,
    r_max: float = 20.0,
    n_quad: int = 400,
) -> np.ndarray:
    """Fold a spherical scalar field with a 3D Gaussian kernel."""
    if t <= 0:
        raise ValueError("t must be positive.")
    if r_max <= 0:
        raise ValueError("r_max must be positive.")

    radii = np.atleast_1d(np.asarray(R_grid, dtype=float))
    nodes, weights = leggauss(n_quad)
    r_q = 0.5 * r_max * (nodes + 1.0)
    w_q = 0.5 * r_max * weights
    M_q = np.asarray(M_callable(r_q), dtype=float)

    sqrt_pi = np.sqrt(np.pi)
    R_col = radii[:, None]
    r_row = r_q[None, :]
    kernel = np.exp(-(((R_col - r_row) / t) ** 2)) - np.exp(
        -(((R_col + r_row) / t) ** 2)
    )
    integrand = r_row * M_q[None, :] * kernel * w_q[None, :]
    sum_int = integrand.sum(axis=1)

    safe_R = np.where(radii > 1e-10, radii, 1.0)
    U_general = sum_int / (sqrt_pi * t * safe_R)

    int_zero = (r_q**2) * M_q * np.exp(-((r_q / t) ** 2)) * w_q
    U_zero = (4.0 / (sqrt_pi * t**3)) * int_zero.sum()

    return np.where(radii > 1e-10, U_general, U_zero)


__all__ = ["gaussian_fold"]
