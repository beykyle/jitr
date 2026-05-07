"""Quadrature kernels and transforms on Lagrange meshes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

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

    def integrate_local(
        self,
        f: Callable[..., npt.ArrayLike],
        a: float,
        args: tuple[Any, ...] = (),
    ) -> np.complex128:
        """Integrate a local function on ``[0, a]`` using Gauss quadrature."""
        values = np.asarray(f(self.quadrature.abscissa * a, *args), dtype=np.complex128)
        return np.sum(values * self.quadrature.weights) * a

    def double_integrate_nonlocal(
        self,
        f: Callable[..., npt.ArrayLike],
        a: float,
        is_symmetric: bool = True,
        args: tuple[Any, ...] = (),
    ) -> np.complex128:
        """Integrate a nonlocal kernel on ``[0, a] × [0, a]``."""
        if is_symmetric:
            off_diag_values = np.asarray(
                f(self.Xn[self.lower_mask] * a, self.Xm[self.lower_mask] * a, *args),
                dtype=np.complex128,
            )
            off_diag = np.sum(self.weight_matrix[self.lower_mask] * off_diag_values)
            diag_values = np.asarray(
                f(self.quadrature.abscissa * a, self.quadrature.abscissa * a, *args),
                dtype=np.complex128,
            )
            diag = np.sum(self.quadrature.weights**2 * diag_values)
            return a**2 * (2 * off_diag + diag)

        values = np.asarray(f(self.Xn * a, self.Xm * a, *args), dtype=np.complex128)
        return a**2 * np.sum(self.weight_matrix * values)

    def fourier_bessel_transform(
        self,
        l: int,  # noqa: E741
        f: Callable[..., npt.ArrayLike],
        k: FloatArray,
        a: float,
        *args: Any,
    ) -> ComplexArray:
        """Perform a Fourier-Bessel transform of order ``l``."""
        r = self.quadrature.abscissa * a
        kr = np.outer(k, r)
        values = np.asarray(f(r, *args), dtype=np.complex128)
        return np.sum(
            sc.spherical_jn(l, kr) * r**2 * values * self.quadrature.weights,
            axis=1,
        )

    def double_fourier_bessel_transform(
        self,
        l: int,  # noqa: E741
        f: Callable[..., npt.ArrayLike],
        k: float,
        a: float,
        *args: Any,
    ) -> ComplexArray:
        """Perform a double Fourier-Bessel transform of order ``l``."""
        n_basis = self.quadrature.nbasis
        r = self.quadrature.abscissa * a
        jkr = sc.spherical_jn(l, np.outer(k, r))
        transformed_kkp = np.zeros((n_basis, n_basis), dtype=np.complex128)
        transformed_rkp = np.zeros((n_basis, n_basis), dtype=np.complex128)

        for i in range(n_basis):
            transformed_rkp[i, :] = np.sum(
                jkr * r**2 * np.asarray(f(r[i], r, *args)) * self.quadrature.weights,
                axis=1,
            )

        for i in range(n_basis):
            transformed_kkp[:, i] = np.sum(
                jkr * r**2 * transformed_rkp[:, i] * self.quadrature.weights,
                axis=1,
            )

        return transformed_kkp * 2 / np.pi

    def dwba_local(
        self,
        bra: ComplexArray,
        ket: ComplexArray,
        a: float,
        f: Callable[..., npt.ArrayLike],
        args: tuple[Any, ...],
    ) -> np.complex128:
        """Return a DWBA matrix element for a local operator."""
        return np.sum(bra * self.matrix_local(f, a, args) * ket)

    def dwba_nonlocal(
        self,
        bra: ComplexArray,
        ket: ComplexArray,
        a: float,
        f: Callable[..., npt.ArrayLike],
        args: tuple[Any, ...],
        is_symmetric: bool = True,
    ) -> np.complex128:
        """Return a DWBA matrix element for a nonlocal operator."""
        operator = self.matrix_nonlocal(f, a, is_symmetric=is_symmetric, args=args)
        return np.complex128(bra.T @ operator @ ket)

    def matrix_local(
        self,
        f: Callable[..., npt.ArrayLike],
        a: float,
        args: tuple[Any, ...] = (),
    ) -> ComplexArray:
        """Evaluate a diagonal local operator in the Lagrange basis."""
        return np.asarray(f(self.quadrature.abscissa * a, *args), dtype=np.complex128)

    def matrix_nonlocal(
        self,
        f: Callable[..., npt.ArrayLike],
        a: float,
        is_symmetric: bool = True,
        args: tuple[Any, ...] = (),
    ) -> ComplexArray:
        """Evaluate a nonlocal operator in the Lagrange basis."""
        values = np.asarray(f(self.Xn * a, self.Xm * a, *args), dtype=np.complex128)
        return np.sqrt(self.weight_matrix) * values * a
