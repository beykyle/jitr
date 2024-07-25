import numpy as np
import scipy.special as sc

from .utils import block
from .system import InteractionMatrix
from .rmatrix_kernel import (
    LagrangeLaguerreKernel,
    LagrangeLegendreKernel,
    laguerre_quadrature,
    legendre_quadrature,
    solution_coeffs,
    solution_coeffs_with_inverse,
    solve_smatrix_with_inverse,
    solve_smatrix_without_inverse,
)


class LagrangeRMatrixSolver:
    r"""
    A Schrödinger equation solver using the R-matrix method on a Lagrange mesh
    """

    def __init__(
        self,
        nbasis: np.int32,
        nchannels: np.int32,
        channel_radii: np.array,
        basis="Legendre",
        **args,
    ):
        r"""
        @parameters:
            nbasis (int) : size of basis; e.g. number of quadrature points for
            integration
            nchannels (int) : number of channels in full system
            ecom (float) : center of mass frame scattering energy
            basis (str): what basis/mesh to use (see Ch. 3 of Baye, 2015)
        """
        self.channel_radii = channel_radii
        assert self.channel_radii.shape == (nchannels,)
        self.overlap = np.diag(np.ones(nbasis))
        if basis == "Legendre":
            x, w = legendre_quadrature(nbasis)
            self.kernel = LagrangeLegendreKernel(nbasis, nchannels, x, w)
            self.f = self.legendre
        elif basis == "Laguerre":
            x, w = laguerre_quadrature(nbasis)
            self.f = self.laguerre
            self.kernel = LagrangeLaguerreKernel(nbasis, nchannels, x, w)
        else:
            raise NotImplementedError(
                "Currently only Legendre and Laguerre meshes are supported"
            )

        # precompute matrices of weights and abscissa for vectorized operations
        self.weight_matrix = np.outer(self.kernel.weights, self.kernel.weights)
        self.Xn, self.Xm = np.meshgrid(self.kernel.abscissa, self.kernel.abscissa)
        self.upper_mask = np.triu_indices(nbasis)
        self.lower_mask = np.tril_indices(nbasis, k=-1)

        # precompute basis functions at boundary
        self.precompute_boundaries(self.channel_radii)

    def get_channel_block(self, matrix: np.array, i: np.int32, j: np.int32 = None):
        N = self.kernel.nbasis
        if j is None:
            j = i
        return block(matrix, (i, j), (N, N))

    def integrate_local(self, f, a: np.float64, args=()):
        """
        @returns integral of local function f(x,*args)dx from [0,a] in Gauss
        quadrature
        """
        return np.sum(f(self.kernel.abscissa * a, *args) * self.kernel.weights) * a

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
                self.kernel.weights**2
                * f(self.kernel.abscissa * a, self.kernel.abscissa * a, *args)
            )

            return a**2 * (2 * off_diag + diag)
        else:
            return a**2 * np.sum(
                self.weight_matrix * f(self.Xn * a, self.Xm * a, *args)
            )

    def fourier_bessel_transform(
        self, l: np.int32, f, k: np.float64, a: np.float64, *args
    ):
        """
        performs a Fourier-Bessel transform of order l from r->k coordinates
        with r on [0,a]
        """
        r = self.kernel.abscissa * a
        kr = np.outer(k, r)
        return np.sum(
            sc.spherical_jn(l, kr) * r**2 * f(r, *args) * self.kernel.weights, axis=1
        )

    def double_fourier_bessel_transform(
        self, l: np.int32, f, k: np.float64, a: np.float64, *args
    ):
        """
        performs a double Fourier-Bessel transform of f(r,r') of order l, going
        from f(r,r')->F(k,k') coordinates with r/r' on [0,a]
        """
        N = self.kernel.nbasis
        r = self.kernel.abscissa * a
        jkr = sc.spherical_j(l, np.outer(k, r))
        F_kkp = np.zeros((N, N), dtype=np.complex128)
        F_rkp = np.zeros((N, N), dtype=np.complex128)

        # integrate over rp at each r
        for i in range(N):
            F_rkp[i, :] = np.sum(
                jkr * r**2 * f(r[i], r, *args) * self.kernel.weights, axis=1
            )

        # integrate over r at each kp
        for i in range(N):
            F_kkp[:, i] = np.sum(jkr * r**2 * F_rkp[:, i] * self.kernel.weights, axis=1)

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
        return np.sum(bra.conj() * self.matrix_local(f, a, args) * ket)

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
        @returns the DWBA matrix element for the nonlocal operator
        `interaction`, between distorted wave `bra` and `ket`, which are
        represented as a set of `self.nbasis` complex coefficients for the
        Lagrange functions, generalizing Eq. 29 in Descouvemont, 2016 (or 2.85
        in Baye, 2015).

        Note: integral is performed in s,s' space, with ds = k dr. To get value of
        integral over r space, result should be scaled by 1/(kk')
        """
        # get operator in Lagrange coords as (nbasis x nbasis) matrix
        Vnm = self.matrix_nonlocal(f, a, is_symmetric=is_symmetric, args=args)
        # reduce
        return bra.conj().T @ Vnm @ ket

    def matrix_local(self, f, a, args=()):
        r"""
        @returns diagonal elements of matrix for arbitrary local vectorized
        operator f(x)
        """
        return f(self.kernel.abscissa * a, *args)

    def matrix_nonlocal(self, f, a, is_symmetric=True, args=()):
        r"""
        @returns matrix for arbitrary vectorized operator f(x,xp)
        """
        if is_symmetric:
            n = self.kernel.nbasis
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

    def precompute_boundaries(self, a):
        r"""
        precompute boundary values of Lagrange-Legendre for each channel
        """
        self.b = np.hstack(
            [
                [self.f(n, a[i], a[i]) for n in range(1, self.kernel.nbasis + 1)]
                for i in range(self.kernel.nchannels)
            ],
            dtype=np.complex128,
        )

    def free_matrix(self, l: np.array, full_matrix=True):
        r"""
        precompute free matrices, which only depend on orbital angular momentum
        l and dimensionless channel radius a
        Parameters:
            l: orbital angular momentum quantum number for each channel
            full_matrix: whether to return the full matrix or just the block
            diagonal elements (elements off of the channel diagonal are all 0
            for the free matrix). If False, returns a list of Nch (Nb,Nb) matrices,
            where Nch is the number of channels and Nb is the number of basis elements
        """
        assert l.shape == (self.kernel.nchannels,)
        free_matrix = self.kernel.free_matrix(self.channel_radii, l)
        if full_matrix:
            return free_matrix
        else:
            return [
                self.get_channel_block(free_matrix, i)
                for i in range(self.kernel.nchannels)
            ]

    def laguerre(self, n: np.int32, a: np.float64, s: np.float64):
        r"""
        nth Lagrange-Laguerre function, scaled by a and regularized s. Eq. 3.70
        in Baye, 2015 with alpha = 0.

        Note: n is indexed from 1 (constant function is not part of basis)
        """
        assert n <= self.kernel.nbasis and n >= 1

        x = s / a
        xn = self.kernel.abscissa[n - 1]

        return (
            (-1) ** n
            / np.sqrt(xn)
            * sc.special.eval_laguerre(n, x)
            / (x - xn)
            * x
            * np.exp(-x / 2)
        )

    def legendre(self, n: np.int32, a: np.float64, s: np.float64):
        r"""
        nth Lagrange-Legendre polynomial shifted onto [0,a_i] and regularized by
        s.  Eq. 3.122 in Baye, 2015

        Note: n is indexed from 1 (constant function is not part of basis)
        """
        assert n <= self.kernel.nbasis and n >= 1

        x = s / a
        xn = self.kernel.abscissa[n - 1]

        return (
            (-1.0) ** (self.kernel.nbasis - n)
            * np.sqrt((1 - xn) / xn)
            * sc.eval_legendre(self.kernel.nbasis, 2.0 * x - 1.0)
            * x
            / (x - xn)
        )

    def interaction_matrix(
        self,
        interaction_matrix: InteractionMatrix,
        channels: np.array,
    ):
        r"""
        Returns the full (Nxn)x(Nxn) interaction in the Lagrange basis, where
        each channel is an nxn block (n being the basis size), and there are NxN
        such blocks, for N channels.  Uses the dimensionless version with s=kr
        and divided by E.
        """
        nb = self.kernel.nbasis
        sz = nb * self.kernel.nchannels
        C = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(self.kernel.nchannels):
            channel_radius_r = channels["a"][i] / channels["k"][i]
            E = channels["E"][i]
            for j in range(self.kernel.nchannels):
                Cij = C[i * nb : i * nb + nb, j * nb : j * nb + nb]
                int_local = interaction_matrix.local_matrix[i, j]
                int_nonlocal = interaction_matrix.nonlocal_matrix[i, j]
                if int_local is not None:
                    loc_args = interaction_matrix.local_args[i, j]
                    Cij += (
                        np.diag(
                            self.matrix_local(
                                int_local,
                                channel_radius_r,
                                loc_args,
                            )
                        )
                        / E
                    )
                if int_nonlocal is not None:
                    nloc_args = interaction_matrix.nonlocal_args[i, j]
                    is_symmetric = interaction_matrix.nonlocal_symmetric[i, j]
                    Cij += (
                        self.matrix_nonlocal(
                            int_nonlocal,
                            channel_radius_r,
                            is_symmetric,
                            nloc_args,
                        )
                        / channels["k"][i]  # extra factor of 1/k because dr = 1/k ds
                        / E
                    )
        return C

    def solve(
        self,
        interaction_matrix: InteractionMatrix,
        channels: np.array,
        free_matrix=None,
        wavefunction=None,
    ):
        if free_matrix is None:
            free_matrix = self.free_matrix(channels["l"])

        A = free_matrix + self.interaction_matrix(interaction_matrix, channels)
        R, S, Ainv, uext_prime_boundary = solve_smatrix_with_inverse(
            A,
            self.b,
            channels["Hp"],
            channels["Hm"],
            channels["Hpp"],
            channels["Hmp"],
            channels["weight"],
            self.channel_radii,
            self.kernel.nchannels,
            self.kernel.nbasis,
        )
        if wavefunction is None:
            return R, S, uext_prime_boundary
        else:
            x = solution_coeffs_with_inverse(
                Ainv,
                self.b,
                S,
                uext_prime_boundary,
                self.kernel.nchannels,
                self.kernel.nbasis,
            )
            return R, S, x, uext_prime_boundary
