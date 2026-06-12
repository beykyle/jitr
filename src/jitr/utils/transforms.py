"""Quadrature-mesh transform utilities (Fourier-Bessel and integration).

Successor to the transform helpers of the deprecated ``jitr.quadrature``
module. These are setup-time reductions over Gauss quadrature nodes, so
they are plain NumPy by design (design doc §4 JAX-vs-NumPy rule); the mesh
helpers reproduce the lax/legacy Lagrange-Legendre nodes exactly (same
Gauss abscissa scaled to ``[0, R]``), so values sampled on
``IntegralWorkspace.radial_grid()`` can be transformed directly with
``legendre_mesh(workspace.nbasis, workspace.channel_radius_fm)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import spherical_jn

FloatArray = npt.NDArray[np.float64]


def legendre_mesh(n: int, radius: float) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre nodes and ``dr`` weights on ``[0, radius]``.

    Returns ``(radii, weights)`` with ``∫₀^R f(r) dr ≈ Σ_i w_i f(r_i)``.
    The nodes coincide with the lax ``MeshSpec("legendre", "x")`` radii.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return radius * (x + 1.0) / 2.0, radius * w / 2.0


def laguerre_mesh(n: int, scale: float) -> tuple[FloatArray, FloatArray]:
    """Gauss-Laguerre nodes and weights on ``[0, ∞)``.

    Returns ``(radii, weights)`` with
    ``∫₀^∞ f(r) e^{-r/scale} dr ≈ Σ_i w_i f(r_i)``.
    """
    x, w = np.polynomial.laguerre.laggauss(n)
    return scale * x, scale * w


def integrate_local(values: npt.ArrayLike, weights: FloatArray) -> np.floating:
    """Quadrature of node values: ``Σ_i w_i f(r_i)``."""
    return np.sum(np.asarray(values) * weights, axis=-1)


def fourier_bessel_transform(
    l: int,
    values: npt.ArrayLike,
    k_grid: npt.ArrayLike,
    radii: FloatArray,
    weights: FloatArray,
) -> np.ndarray:
    """Return ``∫ j_l(k r) r² f(r) dr`` on ``k_grid``.

    ``values`` are node samples ``f(r_i)``; matches the deprecated
    ``Kernel.fourier_bessel_transform`` convention.
    """
    k_arr = np.atleast_1d(np.asarray(k_grid, dtype=np.float64))
    bessel = spherical_jn(l, np.outer(k_arr, radii))  # (N_k, N)
    return bessel @ (radii**2 * weights * np.asarray(values))


def double_fourier_bessel_transform(
    l: int,
    values: npt.ArrayLike,
    k_grid: npt.ArrayLike,
    radii: FloatArray,
    weights: FloatArray,
) -> np.ndarray:
    """Return ``(2/π) ∫∫ j_l(k r) r² K(r, r') r'² j_l(k' r') dr dr'``.

    ``values`` is the node-sampled kernel ``K(r_i, r_j)``; matches the
    deprecated ``Kernel.double_fourier_bessel_transform`` convention.
    """
    k_arr = np.atleast_1d(np.asarray(k_grid, dtype=np.float64))
    bessel = spherical_jn(l, np.outer(k_arr, radii)) * (radii**2 * weights)
    return (2.0 / np.pi) * bessel @ np.asarray(values) @ bessel.T
