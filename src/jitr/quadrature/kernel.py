import numpy as np
import scipy.special as sc

from .quadrature import (
    legendre,
    laguerre,
    LagrangeLegendreQuadrature,
    LagrangeLaguerreQuadrature,
    generate_laguerre_quadrature,
    generate_legendre_quadrature,
)


class Kernel:
    def __init__(
        self,
        nbasis: np.int32,
        basis="Legendre",
    ):
        self.overlap = np.diag(np.ones(nbasis))
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

        # precompute matrices of weights and abscissa for vectorized operations
        self.weight_matrix = np.outer(self.quadrature.weights, self.quadrature.weights)
        self.Xn, self.Xm = np.meshgrid(
            self.quadrature.abscissa, self.quadrature.abscissa
        )
        self.upper_mask = np.triu_indices(nbasis)
        self.lower_mask = np.tril_indices(nbasis, k=-1)

    def f(self, n: np.int32, a: np.float64, s: np.float64):
        return self.basis_function(n, a, s, self.quadrature)

    def integrate_local(self, f, a: np.float64, args=()):
        """
        @returns integral of local function f(x,*args)dx from [0,a] in Gauss
        quadrature
        """
        return (
            np.sum(f(self.quadrature.abscissa * a, *args) * self.quadrature.weights) * a
        )

    def double_integrate_nonlocal(
        self, f, a: np.float64, is_symmetric: bool = True, args=()
    ):
        """
        @returns double integral nonlocal function f(x,x',*args)dxdx' from
        [0,a] x [0,a] in Gauss quadrature
        """
        if is_symmetric:
            off_diag = np.sum(
                self.weight_matrix[self.lower_mask]
                * f(self.Xn[self.lower_mask] * a, self.Xm[self.lower_mask] * a, *args)
            )
            diag = np.sum(
                self.quadrature.weights**2
                * f(self.quadrature.abscissa * a, self.quadrature.abscissa * a, *args)
            )

            return a**2 * (2 * off_diag + diag)
        else:
            return a**2 * np.sum(
                self.weight_matrix * f(self.Xn * a, self.Xm * a, *args)
            )

    def fourier_bessel_transform(
        self, l: np.int32, f, k: np.array, a: np.float64, *args
    ):
        """
        performs a Fourier-Bessel transform of order l from r->k coordinates
        with r on [0,a]
        """
        r = self.quadrature.abscissa * a
        kr = np.outer(k, r)
        return np.sum(
            sc.spherical_jn(l, kr) * r**2 * f(r, *args) * self.quadrature.weights,
            axis=1,
        )

    def double_fourier_bessel_transform(
        self, l: np.int32, f, k: np.float64, a: np.float64, *args
    ):
        """
        performs a double Fourier-Bessel transform of f(r,r') of order l, going
        from f(r,r')->F(k,k') coordinates with r/r' on [0,a]
        """
        N = self.quadrature.nbasis
        r = self.quadrature.abscissa * a
        jkr = sc.spherical_j(l, np.outer(k, r))
        F_kkp = np.zeros((N, N), dtype=np.complex128)
        F_rkp = np.zeros((N, N), dtype=np.complex128)

        # integrate over rp at each r
        for i in range(N):
            F_rkp[i, :] = np.sum(
                jkr * r**2 * f(r[i], r, *args) * self.quadrature.weights, axis=1
            )

        # integrate over r at each kp
        for i in range(N):
            F_kkp[:, i] = np.sum(
                jkr * r**2 * F_rkp[:, i] * self.quadrature.weights, axis=1
            )

        return F_kkp * 2 / np.pi

    def dwba_local(
        self,
        bra: np.array,
        ket: np.array,
        a: np.float64,
        f,
        args,
    ):
        r"""
        @returns the DWBA matrix element for the local operator `interaction`,
        between distorted wave `bra` and `ket`, which are represented as a set
        of `self.nbasis` complex coefficients for the Lagrange functions,
        following Eq. 29 in Descouvemont, 2016 (or 2.85 in Baye, 2015).

        Note: integral is performed in s space, with ds = k dr. To get value of
        integral over r space, result should be scaled by 1/k
        """
        return np.sum(bra * self.matrix_local(f, a, args) * ket)

    def dwba_nonlocal(
        self,
        bra: np.array,
        ket: np.array,
        a: np.float64,
        f,
        args,
        is_symmetric: bool = True,
    ):
        r"""
        @returns DWBA (complex128): matrix element for the nonlocal operator
        `interaction`, between distorted wave `bra` and `ket`, which are
        represented as a set of `self.nbasis` complex coefficients for the
        Lagrange functions, generalizing Eq. 29 in Descouvemont, 2016 (or 2.85
        in Baye, 2015).

        Note: integral is performed in s,s' space, with ds = k dr. To get value
        of integral over r space, result should be scaled by 1/(kk')
        """
        # get operator in Lagrange coords as (nbasis x nbasis) matrix
        Vnm = self.matrix_nonlocal(f, a, is_symmetric=is_symmetric, args=args)
        # reduce
        return bra.T @ Vnm @ ket

    def matrix_local(self, f, a: np.float64, args=()):
        r"""
        @returns matrix (np.ndarray): diagonal elements of arbitrary vectorized
            operator f(x) in lagrange basis
        """
        return f(self.quadrature.abscissa * a, *args)

    def matrix_nonlocal(self, f, a: np.float64, is_symmetric=True, args=()):
        r"""
        @returns matrix (np.ndarray): arbitrary vectorized operator f(x,xp) in
            lagrange basis
        """
        return np.sqrt(self.weight_matrix) * f(self.Xn * a, self.Xm * a, *args) * a
