"""Quadrature kernels and transforms on Lagrange meshes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.special as sc

from .quadrature import (
    ComplexArray,
    FloatArray,
    LagrangeLaguerreQuadrature,
    LagrangeLegendreQuadrature,
    generate_laguerre_quadrature,
    generate_legendre_quadrature,
    laguerre,
    legendre,
)

Quadrature: TypeAlias = LagrangeLaguerreQuadrature | LagrangeLegendreQuadrature
BasisFunction: TypeAlias = Callable[[int, float, float, Quadrature], complex]


class Kernel:
    """Convenience wrapper around a quadrature rule and its basis functions."""

    def __init__(self, nbasis: int, basis: str = "Legendre") -> None:
        """Construct a kernel for the requested Lagrange basis."""
        self.overlap = np.diag(np.ones(nbasis))
        self.quadrature: Quadrature
        self.basis_function: BasisFunction
        if basis == "Legendre":
            x, w = generate_legendre_quadrature(nbasis)
            self.quadrature = LagrangeLegendreQuadrature(x, w)
            self.basis_function = legendre
        elif basis == "Laguerre":
            x, w = generate_laguerre_quadrature(nbasis)
            self.quadrature = LagrangeLaguerreQuadrature(x, w)
            self.basis_function = laguerre
        else:
            raise NotImplementedError(
                "Currently only Legendre and Laguerre meshes are supported"
            )

        self.weight_matrix = np.outer(self.quadrature.weights, self.quadrature.weights)
        self.Xn: FloatArray
        self.Xm: FloatArray
        self.Xn, self.Xm = np.meshgrid(
            self.quadrature.abscissa, self.quadrature.abscissa
        )
        self.upper_mask = np.triu_indices(nbasis)
        self.lower_mask = np.tril_indices(nbasis, k=-1)

    def f(self, n: int, a: float, s: float) -> complex:
        """Evaluate the ``n``-th Lagrange basis function."""
        return self.basis_function(n, a, s, self.quadrature)

    def radial_grid(self, radius: float) -> FloatArray:
        """Return the one-dimensional quadrature grid on ``[0, radius]``."""
        return self.quadrature.abscissa * radius

    def nonlocal_radial_grids(self, radius: float) -> tuple[FloatArray, FloatArray]:
        """Return the tensor-product quadrature grids on ``[0, radius]^2``."""
        return self.Xn * radius, self.Xm * radius

    def integrate_local(self, values: npt.ArrayLike, radius: float) -> np.complex128:
        """Integrate local values on ``[0, radius]`` using Gauss quadrature."""
        values_array = np.asarray(values, dtype=np.complex128)
        if values_array.shape != (self.quadrature.nbasis,):
            raise ValueError(
                "local quadrature values must have shape "
                f"({self.quadrature.nbasis},)"
            )
        return np.sum(values_array * self.quadrature.weights) * radius

    def double_integrate_nonlocal(
        self,
        values: npt.ArrayLike,
        radius: float,
        is_symmetric: bool = True,
    ) -> np.complex128:
        """Integrate nonlocal values on ``[0, radius] × [0, radius]``."""
        values_array = np.asarray(values, dtype=np.complex128)
        expected_shape = (self.quadrature.nbasis, self.quadrature.nbasis)
        if values_array.shape != expected_shape:
            raise ValueError(
                "nonlocal quadrature values must have shape " f"{expected_shape}"
            )

        if is_symmetric:
            off_diag = np.sum(
                self.weight_matrix[self.lower_mask] * values_array[self.lower_mask]
            )
            diag = np.sum(self.quadrature.weights**2 * np.diag(values_array))
            return radius**2 * (2 * off_diag + diag)

        return radius**2 * np.sum(self.weight_matrix * values_array)

    def fourier_bessel_transform(
        self,
        l: int,  # noqa: E741
        values: npt.ArrayLike,
        k: FloatArray,
        radius: float,
    ) -> ComplexArray:
        """Perform a Fourier-Bessel transform of order ``l``."""
        values_array = np.asarray(values, dtype=np.complex128)
        r = self.radial_grid(radius)
        if values_array.shape != r.shape:
            raise ValueError(
                "local values for a Fourier-Bessel transform must have shape "
                f"{r.shape}"
            )
        kr = np.outer(k, r)
        return np.sum(
            sc.spherical_jn(l, kr) * r**2 * values_array * self.quadrature.weights,
            axis=1,
        )

    def double_fourier_bessel_transform(
        self,
        l: int,  # noqa: E741
        values: npt.ArrayLike,
        k: float,
        radius: float,
    ) -> ComplexArray:
        """Perform a double Fourier-Bessel transform of order ``l``."""
        values_array = np.asarray(values, dtype=np.complex128)
        expected_shape = (self.quadrature.nbasis, self.quadrature.nbasis)
        if values_array.shape != expected_shape:
            raise ValueError(
                "nonlocal values for a double Fourier-Bessel transform must have "
                f"shape {expected_shape}"
            )

        r = self.radial_grid(radius)
        jkr = sc.spherical_jn(l, np.outer(k, r))
        transformed_rkp = np.zeros_like(values_array, dtype=np.complex128)
        transformed_kkp = np.zeros_like(values_array, dtype=np.complex128)

        for i in range(self.quadrature.nbasis):
            transformed_rkp[i, :] = np.sum(
                jkr * r**2 * values_array[i, :] * self.quadrature.weights,
                axis=1,
            )

        for i in range(self.quadrature.nbasis):
            transformed_kkp[:, i] = np.sum(
                jkr * r**2 * transformed_rkp[:, i] * self.quadrature.weights,
                axis=1,
            )

        return transformed_kkp * 2 / np.pi

    def dwba_local(
        self,
        bra: ComplexArray,
        ket: ComplexArray,
        values: npt.ArrayLike,
    ) -> np.complex128:
        """Return a DWBA matrix element for a local operator."""
        return np.sum(bra * self.matrix_local(values) * ket)

    def dwba_nonlocal(
        self,
        bra: ComplexArray,
        ket: ComplexArray,
        values: npt.ArrayLike,
        radius: float,
    ) -> np.complex128:
        """Return a DWBA matrix element for a nonlocal operator."""
        operator = self.matrix_nonlocal(values, radius)
        return np.complex128(bra.T @ operator @ ket)

    def matrix_local(self, values: npt.ArrayLike) -> ComplexArray:
        """Validate and cast local quadrature values."""
        values_array = np.asarray(values, dtype=np.complex128)
        nbasis = self.quadrature.nbasis
        if values_array.ndim == 1 and values_array.shape == (nbasis,):
            return values_array
        if (
            values_array.ndim == 3
            and values_array.shape[0] == values_array.shape[1]
            and values_array.shape[2] == nbasis
        ):
            return values_array
        raise ValueError(
            "local potential values must have shape "
            f"({nbasis},) or (nchannels, nchannels, {nbasis})"
        )

    def matrix_nonlocal(self, values: npt.ArrayLike, radius: float) -> ComplexArray:
        """Validate and weight nonlocal quadrature values."""
        values_array = np.asarray(values, dtype=np.complex128)
        nbasis = self.quadrature.nbasis
        weight_sqrt = np.sqrt(self.weight_matrix)
        if values_array.ndim == 2 and values_array.shape == (nbasis, nbasis):
            return weight_sqrt * values_array * radius
        if (
            values_array.ndim == 4
            and values_array.shape[0] == values_array.shape[1]
            and values_array.shape[2:] == (nbasis, nbasis)
        ):
            return weight_sqrt[np.newaxis, np.newaxis, :, :] * values_array * radius
        raise ValueError(
            "nonlocal potential values must have shape "
            f"({nbasis}, {nbasis}) or "
            f"(nchannels, nchannels, {nbasis}, {nbasis})"
        )
