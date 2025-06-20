import numpy as np
import scipy.special as sc

from quadrature import (
    legendre,
    LagrangeLegendreQuadrature,
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

    def f(self, n: np.int32, a: np.float64, r: np.float64):
        return self.basis_function(n, a, r, self.quadrature)

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



    def matrix_local(self, f, a, args=()):
        r"""
        @returns diagonal elements of matrix for arbitrary local vectorized
        operator f(x)
        """
        return f(self.quadrature.abscissa * a, *args)

    def matrix_nonlocal(self, f, a, is_symmetric=True, args=()):
        r"""
        @returns matrix for arbitrary vectorized operator f(x,xp)
        """
        if is_symmetric:
            n = self.quadrature.nbasis
            M = np.zeros((n, n), dtype=np.complex128)
            xn = self.Xn[self.upper_mask] * a
            xm = self.Xm[self.upper_mask] * a
            M[self.upper_mask] = (
                np.sqrt(self.weight_matrix[self.upper_mask]) * f(xn, xm, *args) * a
            )
            M += np.triu(M, k=1).T
            return M
        else:
            return np.sqrt(self.weight_matrix) * f(self.Xn * a, self.Xm * a, *args) * a

    # def free_matrix(
    #     self,
    #     a: np.array,
    #     l: np.array,
    # ):
    #     r"""
    #     @returns the full (NchxNb)x(NchxNb) free Schrödinger equation 1/E (H-E)
    #     in the Lagrange basis, where each channel is an NbxNb block (Nb
    #     being the basis size), and there are NchxNch such blocks.
    #     """
    #     Nb = self.quadrature.nbasis
    #     Nch = np.size(a)
    #     sz = Nb * Nch
    #     F = np.zeros((sz, sz), dtype=np.complex128)
    #     for i in range(Nch):
    #         Fij = self.quadrature.kinetic_matrix(a[i], l[i]) - self.overlap
    #         F[(i * Nb) : (i + 1) * Nb, (i * Nb) : (i + 1) * Nb] += Fij
    #     return F
