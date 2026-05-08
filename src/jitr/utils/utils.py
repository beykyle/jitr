"""General-purpose numerical helpers used across :mod:`jitr`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit

from .free_solutions import (
    CoulombAsymptotics,
    H_minus,
    H_minus_prime,
    H_plus,
    H_plus_prime,
)

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


@njit
def complex_det(matrix: np.ndarray) -> np.complex128:
    """Return ``sqrt(det(A A^\u2020))`` for a complex matrix."""
    determinant = np.linalg.det(matrix @ np.conj(matrix).T)
    return np.sqrt(determinant)


@njit
def block(
    matrix: np.ndarray,
    block_index: tuple[int, int],
    block_size: tuple[int, int],
) -> np.ndarray:
    """Extract a sub-block from a block-structured matrix.

    :param matrix: Input matrix containing equally sized blocks.
    :param block_index: ``(row, column)`` index of the desired block.
    :param block_size: Shape of each block as ``(rows, columns)``.
    :returns:
    :rtype: The requested submatrix view."""
    i, j = block_index
    n, m = block_size
    return matrix[i * n : i * n + n, j * m : j * m + m]


def second_derivative_op(
    s: float,
    channel: Any,
    interaction: Callable[..., complex],
    args: tuple[Any, ...] = (),
) -> complex:
    """Evaluate the scaled radial Schrödinger operator."""
    return (
        eval_scaled_interaction(s, interaction, channel, args)
        + channel.l * (channel.l + 1) / s**2
        - 1.0
    )


def schrodinger_eqn_ivp_order1(
    s: float,
    y: Sequence[complex] | FloatArray | ComplexArray,
    channel: Any,
    interaction: Callable[..., complex],
    args: tuple[Any, ...] = (),
) -> list[complex]:
    """Convert the radial Schrödinger equation to a first-order system."""
    u, uprime = y
    return [uprime, second_derivative_op(s, channel, interaction, args) * u]


def smatrix(
    Rl: complex,
    a: float,
    l: int,
    eta: float,
    asym: type = CoulombAsymptotics,
) -> complex:
    """Compute an S-matrix element from a channel R-matrix value."""
    return (
        H_minus(a, l, eta, asym=asym) - a * Rl * H_minus_prime(a, l, eta, asym=asym)
    ) / (H_plus(a, l, eta, asym=asym) - a * Rl * H_plus_prime(a, l, eta, asym=asym))


def delta(Sl: complex) -> tuple[np.float64, np.float64]:
    """Return the phase shift and attenuation in degrees."""
    phase = np.log(Sl) / 2.0j
    return np.rad2deg(np.real(phase)), np.rad2deg(np.imag(phase))


def eval_scaled_interaction(
    s: float,
    interaction: Callable[..., complex],
    ch: Any,
    args: tuple[Any, ...],
) -> complex:
    """Evaluate a local interaction in dimensionless coordinates."""
    return interaction(s / ch.k, *args) / ch.E


def eval_scaled_nonlocal_interaction(
    s: float,
    sp: float,
    interaction: Callable[..., complex],
    ch: Any,
    args: tuple[Any, ...],
) -> complex:
    """Evaluate a nonlocal interaction in dimensionless coordinates."""
    return interaction(s / ch.k, sp / ch.k, *args) / ch.E


def interaction_range(A: int, rA: float = 1.2, r0: float = 0.0) -> float:
    """Estimate an interaction radius in femtometers."""
    return r0 + rA * A ** (1 / 3)


def suggested_dimensionless_channel_radius(interaction_range: float, k: float) -> float:
    """Suggest a dimensionless channel radius for a given interaction range.

    The estimate leaves roughly one asymptotic wavelength beyond the interaction
    region.
    """
    return interaction_range * k + 2 * np.pi


def suggested_basis_size(a: float, zeros_per_node: int = 5) -> int:
    """Suggest an R-matrix basis size for a dimensionless channel radius."""
    return zeros_per_node * int(np.ceil(a / np.pi))
