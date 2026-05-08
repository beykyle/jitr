"""Analytic local and nonlocal optical-potential building blocks."""

from __future__ import annotations

import numpy as np
from scipy import special as sc

from .._types import ArrayOrScalar, PotentialArray
from ..utils.constants import ALPHA, HBARC

MAX_ARG = np.log(1 / 1e-16)


def perey_buck_nonlocal(r: float, rp: float, *params: float) -> PotentialArray:
    """Return the Perey-Buck nonlocal kernel factor."""
    beta, ell = params
    z = 2 * np.pi * r * rp / beta**2
    Kl = 2 * 1j**ell * z * sc.spherical_jn(int(ell), -1j * z)
    return np.exp(-(r**2 + rp**2) / beta**2) * Kl / (beta * np.sqrt(np.pi))


def woods_saxon_potential(r: ArrayOrScalar, *params: float) -> PotentialArray:
    """Return a Woods-Saxon potential with complex depth ``V + iW``."""
    V, W, R, a = params
    potential = (V + 1j * W) * woods_saxon_safe(r, R, a)
    if isinstance(potential, np.ndarray):
        return np.asarray(potential, dtype=np.complex128)
    return complex(potential)


def woods_saxon_prime(r: ArrayOrScalar, *params: float) -> PotentialArray:
    """Return the radial derivative of a Woods-Saxon potential."""
    V, W, R, a = params
    potential = (V + 1j * W) * woods_saxon_prime_safe(r, R, a)
    if isinstance(potential, np.ndarray):
        return np.asarray(potential, dtype=np.complex128)
    return complex(potential)


def woods_saxon_safe(r: ArrayOrScalar, R: float, a: float) -> ArrayOrScalar:
    """Evaluate a Woods-Saxon shape while avoiding overflow in ``exp``."""
    if not isinstance(r, np.ndarray):
        x_scalar = (r - R) / a
        return 1.0 / (1.0 + np.exp(x_scalar)) if x_scalar < MAX_ARG else 0.0

    x_array = (r - R) / a
    mask = x_array <= MAX_ARG
    potential = np.zeros_like(r, dtype=np.float64)
    potential[mask] = 1.0 / (1.0 + np.exp(x_array[mask]))
    return potential


def woods_saxon_prime_safe(r: ArrayOrScalar, R: float, a: float) -> ArrayOrScalar:
    """Evaluate the radial derivative of the Woods-Saxon shape safely."""
    if not isinstance(r, np.ndarray):
        x_scalar = (r - R) / a
        return (
            -np.exp(x_scalar) / (a * (1 + np.exp(x_scalar)) ** 2)
            if x_scalar < MAX_ARG
            else 0.0
        )

    x_array = (r - R) / a
    mask = x_array <= MAX_ARG
    potential = np.zeros_like(r, dtype=np.float64)
    potential[mask] = -np.exp(x_array[mask]) / (a * (1 + np.exp(x_array[mask])) ** 2)
    return potential


def thomas_safe(r: ArrayOrScalar, R: float, a: float) -> ArrayOrScalar:
    """Evaluate the Thomas spin-orbit shape without overflow issues."""
    if not isinstance(r, np.ndarray):
        x_scalar = (r - R) / a
        y_scalar = 1.0 / r
        return (
            y_scalar * (-np.exp(x_scalar) / (a * (1 + np.exp(x_scalar)) ** 2))
            if x_scalar < MAX_ARG
            else 0.0
        )

    x_array = (r - R) / a
    y_array = 1.0 / r
    mask = x_array <= MAX_ARG
    potential = np.zeros_like(r, dtype=np.float64)
    potential[mask] = y_array[mask] * (
        -np.exp(x_array[mask]) / (a * (1 + np.exp(x_array[mask])) ** 2)
    )
    return potential


def surface_peaked_gaussian_potential(
    r: ArrayOrScalar, *params: float
) -> PotentialArray:
    """Return a simple surface-peaked Gaussian potential."""
    V, W, R, a = params
    potential = (V + 1j * W) * np.exp(-((r - R) ** 2) / (2 * np.pi * a) ** 2)
    if isinstance(potential, np.ndarray):
        return np.asarray(potential, dtype=np.complex128)
    return complex(potential)


def woods_saxon_volume_integral(V: float, R: float, a: float) -> float:
    """Return the volume integral for a Woods-Saxon term."""
    return 4 * np.pi / 3 * (V * R**3) * (1 + (np.pi * a / R) ** 2)


def woods_saxon_mean_square_radius(R: float, a: float) -> float:
    """Return the mean-square radius for a Woods-Saxon term."""
    return 3.0 / 5 * R**2 * (1 + 7.0 / 3 * (np.pi * a / R) ** 2)


def woods_saxon_prime_volume_integral(V: float, R: float, a: float) -> float:
    """Return the volume integral for a derivative Woods-Saxon term."""
    return (
        (4 * np.pi / 3) * V * R**3 * 12 * a / R * (1 + 1.0 / 3 * (np.pi * a / R) ** 2)
    )


def woods_saxon_prime_mean_square_radius(R: float, a: float) -> float:
    """Return the mean-square radius for a derivative Woods-Saxon term."""
    return R**2 * (1 + 5.0 / 3 * (np.pi * a / R) ** 2)


def thomas_volume_integral(V: float, R: float, a: float) -> float:
    """Return the volume integral for the Thomas spin-orbit shape."""
    return 4 * np.pi * V * (R + a * np.log(1 + np.exp(-R / a)))


def thomas_mean_square_radius(R: float, a: float) -> float:
    """Return the mean-square radius for the Thomas spin-orbit shape."""
    return R**2 * (1 + 7.0 / 3.0 * (np.pi * a / R) ** 2)


def coulomb_charged_sphere(r: ArrayOrScalar, zz: float, r_c: float) -> ArrayOrScalar:
    """Return the Coulomb potential of a uniformly charged sphere."""
    return zz * ALPHA * HBARC * regular_inverse_r(r, r_c)


def regular_inverse_r(r: ArrayOrScalar, r_c: float) -> ArrayOrScalar:
    """Return ``1/r`` regularized inside a sphere of radius ``r_c``."""
    if not isinstance(r, np.ndarray):
        return 1 / (2 * r_c) * (3 - (r / r_c) ** 2) if r < r_c else 1 / r

    mask = r <= r_c
    not_mask = np.logical_not(mask)
    potential = np.zeros_like(r, dtype=np.float64)
    potential[mask] = 1.0 / (2.0 * r_c) * (3.0 - (r[mask] / r_c) ** 2)
    potential[not_mask] = 1.0 / r[not_mask]
    return potential


def yamaguchi_potential(r: float, rp: float, *params: float) -> float:
    """Return the Yamaguchi separable nonlocal potential."""
    W0, beta, alpha = params
    return -W0 * 2 * beta * (beta + alpha) ** 2 * np.exp(-beta * (r + rp))


def yamaguchi_swave_delta(k: float, *params: float) -> np.float64:
    """Return the analytic s-wave phase shift for the Yamaguchi potential."""
    _, b, a = params
    denominator = 2 * (a + b) ** 2

    kcotdelta = (
        a * b * (a + 2 * b) / denominator
        + (a**2 + 2 * a * b + 3 * b**2) * k**2 / (b * denominator)
        + k**4 / (b * denominator)
    )
    return np.rad2deg(np.arctan(k / kcotdelta))
