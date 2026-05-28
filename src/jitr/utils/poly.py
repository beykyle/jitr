"""Evaluate low-order polynomials and their derivatives."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
PolyValue: TypeAlias = FloatArray | np.float64


def poly1d(
    x: float | ArrayLike,
    coeffs: ArrayLike,
    start_i: int = 0,
) -> PolyValue:
    """Evaluate a one-dimensional polynomial in ascending-power form.

    Args:
        x: Evaluation point or points.
        coeffs: Polynomial coefficients where ``coeffs[i]`` multiplies
            ``x ** (start_i + i)``.
        start_i: Lowest polynomial power represented by ``coeffs``.

    Returns:
        Polynomial values evaluated at ``x``.
    """

    x_values = np.asarray(x, dtype=float)
    coeff_array: FloatArray = np.asarray(coeffs, dtype=float)
    powers = np.arange(start_i, len(coeff_array) + start_i)
    return (x_values[..., None] ** powers) @ coeff_array


def poly2d(
    x: float | ArrayLike,
    y: float | ArrayLike,
    coeffs: ArrayLike,
    start_i: int = 0,
    start_j: int = 0,
) -> PolyValue:
    """Evaluate a two-dimensional polynomial in ascending-power form.

    Args:
        x: First evaluation coordinate or coordinates.
        y: Second evaluation coordinate or coordinates.
        coeffs: Coefficient matrix where ``coeffs[i, j]`` multiplies
            ``x ** (start_i + i) * y ** (start_j + j)``.
        start_i: Lowest power represented along the ``x`` axis.
        start_j: Lowest power represented along the ``y`` axis.

    Returns:
        Polynomial values evaluated on the paired ``x`` and ``y`` inputs.
    """

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    coeff_array: FloatArray = np.asarray(coeffs, dtype=float)
    i = np.arange(start_i, coeff_array.shape[0] + start_i)
    j = np.arange(start_j, coeff_array.shape[1] + start_j)
    x_powers = x_values[..., None] ** i
    y_powers = y_values[..., None] ** j
    return np.einsum("...i,ij,...j->...", x_powers, coeff_array, y_powers)


def poly1d_deriv(coeffs: ArrayLike, start_i: int = 0) -> tuple[FloatArray, int]:
    """Differentiate a one-dimensional ascending-power polynomial.

    Args:
        coeffs: Polynomial coefficients where ``coeffs[i]`` multiplies
            ``x ** (start_i + i)``.
        start_i: Lowest polynomial power represented by ``coeffs``.

    Returns:
        Tuple of derivative coefficients and the new lowest represented power.
    """

    coeff_array: FloatArray = np.asarray(coeffs, dtype=float)
    i = np.arange(start_i, start_i + len(coeff_array))
    out: FloatArray = coeff_array * i
    if start_i == 0:
        return out[1:], 0
    return out, start_i - 1


def poly2d_deriv(
    coeffs: ArrayLike,
    start_i: int = 0,
    start_j: int = 0,
    wrt: str = "y",
) -> tuple[FloatArray, int, int]:
    """Differentiate a two-dimensional ascending-power polynomial.

    Args:
        coeffs: Coefficient matrix where ``coeffs[i, j]`` multiplies
            ``x ** (start_i + i) * y ** (start_j + j)``.
        start_i: Lowest power represented along the ``x`` axis.
        start_j: Lowest power represented along the ``y`` axis.
        wrt: Coordinate to differentiate with respect to, either ``"x"`` or
            ``"y"``.

    Returns:
        Tuple of derivative coefficients and the new ``start_i``/``start_j``
        values suitable for passing back into :func:`poly2d`.

    Raises:
        ValueError: If ``wrt`` is not ``"x"`` or ``"y"``.
    """

    coeff_array: FloatArray = np.asarray(coeffs, dtype=float)
    if wrt == "y":
        j = np.arange(start_j, start_j + coeff_array.shape[1])
        out: FloatArray = coeff_array * j[None, :]
        if start_j == 0:
            return out[:, 1:], start_i, 0
        return out, start_i, start_j - 1
    if wrt == "x":
        i = np.arange(start_i, start_i + coeff_array.shape[0])
        out = coeff_array * i[:, None]
        if start_i == 0:
            return out[1:, :], 0, start_j
        return out, start_i - 1, start_j
    raise ValueError(f"wrt must be 'x' or 'y', got {wrt!r}")
