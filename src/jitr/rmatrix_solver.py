import numpy as np
import scipy.special as sc
import scipy.linalg as la

from .utils import (
    hbarc,
    c,
    CoulombAsymptotics,
    FreeAsymptotics,
    H_plus,
    H_minus,
    H_plus_prime,
    H_minus_prime,
)
from .system import ProjectileTargetSystem, InteractionMatrix
from .rmatrix_kernel import LagrangeRMatrixKernel, rmsolve_smatrix, rmsolve_wavefunction


def build_kernel(nbasis: int, nchannels: int = 1):
    x, w = np.polynomial.legendre.leggauss(nbasis)
    abscissa = 0.5 * (x + 1)
    weights = 0.5 * w
    return LagrangeRMatrixKernel(nbasis, nchannels, abscissa, weights)


class LagrangeRMatrixSolver:
    r"""A solver valid for all energies"""

    def __init__(
        self,
        nbasis: np.int32,
        nchannels: np.int32,
        sys: ProjectileTargetSystem,
        ecom=None,
        asym=None,
    ):
        r"""
        Parameters:
            nbasis (int) : size of basis; e.g. number of quadrature points for integration
            nchannels (int) : number of channels in full system
            sys (ProjectileTargetSystem) : information about scattering system in question
            ecom (float) : center of mass frame scattering energy
            asym : Implementation of asymptotic free wavefunctions
        """
        self.sys = sys
        self.kernel = build_kernel(nbasis, nchannels)

        # precompute matrices of weights and abscissa for vectorized operations
        self.weight_matrix = np.sqrt(np.outer(self.kernel.weights, self.kernel.weights))
        self.Xn, self.Xm = np.meshgrid(self.kernel.abscissa, self.kernel.abscissa)
        self.upper_mask = np.triu_indices(nbasis)
        self.lower_mask = np.tril_indices(nbasis, k=1)

        # precompute asymptotic values
        if asym is None:
            if sys.Zproj * sys.Ztarget > 0:
                asym = CoulombAsymptotics
            else:
                asym = FreeAsymptotics

        self.asym = asym
        self.ecom = ecom
        if ecom is not None:
            self.precompute_asymptotics(
                self.sys.channel_radii, self.sys.l, self.sys.eta(ecom)
            )

        # precompute basis functions at boundary
        self.precompute_boundaries(self.sys.channel_radii)

        # precompute free matrix
        self.precompute_free_matrix(self.sys.channel_radii, self.sys.l)

    def integrate_local(self, f, a: np.float64, args=()):
        """
        integrates local operator of form f(x,*args)dx from [0,a] in Gauss quadrature
        """
        return np.sum(f(self.kernel.abscissa * a, *args) * self.kernel.weights) * a

    def double_integrate_nonlocal(
        self, f, a: np.float64, is_symmetric: bool = True, args=()
    ):
        """
        double integrates nonlocal operator of form f(x,x',*args)dxdx' from [0,a] x [0,a]
        in Gauss quadrature
        """
        w = self.kernel.weights
        x = self.kernel.abscissa
        d = 0

        if is_symmetric:
            off_diag = np.sum(
                self.weight_matrix[lower_mask]
                * f(self.Xn[lower_mask] * a, self.Xm[lower_mask] * a, *args)
            )
            diag = np.sum(
                np.diag(weight_matrix)
                * f(self.kernel.abscissa * a, self.kernel.abscissa * a)
            )

            return a * (2 * off_diag + diag)
        else:
            return a * np.sum(self.weight_matrix * f(self.Xn * a, self.Xm * a, *args))

    def fourier_bessel_transform(
        self, l: np.int32, f, k: np.float64, a: np.float64, *args
    ):
        """
        performs a Fourier-Bessel transform of order l from r->k coordinates with r on [0,a]
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
        performs a double Fourier-Bessel transform of f(r,r') of order l, going from f(r,r')->F(k,k')
        coordinates with r/r' on [0,a]
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
        Calculates the DWBA matrix element for the local operator `interaction`, between distorted
        wave `bra` and `ket`, which are represented as a set of `self.nbasis` complex coefficients
        for the Lagrange functions, following Eq. 29 in Descouvemont, 2016 (or 2.85 in Baye, 2015).
        """
        return np.sum(bra.conj() * self.matrix_local(interaction, a, args) * ket)

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
        Calculates the DWBA matrix element for the nonlocal operator `interaction`, between
        distorted wave `bra` and `ket`, which are represented as a set of `self.nbasis` complex
        coefficients for the Lagrange functions, generalizing Eq. 29 in Descouvemont, 2016
        (or 2.85 in Baye, 2015).
        """
        # get operator in Lagrange coords as (nbasis x nbasis) matrix
        Vnm = self.matrix_nonlocal(f, a, is_symmetric=is_symmetric, args=args)
        # reduce
        return bra.conj().T @ Vnm @ ket

    def matrix_local(self, f, a, args=()):
        r"""get diagonal elements of matrix for arbitrary local vectorized operator f(x)"""
        return f(self.kernel.abscissa * a, *args)

    def matrix_nonlocal(self, f, a, is_symmetric=True, args=()):
        r"""get matrix for arbitrary vectorized operator f(x,xp)"""
        if is_symmetric:
            n = self.kernel.nbasis
            M = np.zeros((n, n), dtype=np.complex128)
            xn = self.Xn[self.upper_mask] * a
            xm = self.Xm[self.upper_mask] * a
            M[self.upper_mask] = (
                self.weight_matrix[self.upper_mask] * f(xn, xm, *args) * a
            )
            M += np.triu(M, k=1).T
            return M
        else:
            return self.weight_matrix * f(self.Xn * a, self.Xm * a, *args) * a

    def precompute_boundaries(self, a):
        r"""precompute boundary values of Lagrange-Legendre for each channel"""
        self.b = np.hstack(
            [
                [self.f(n, a[i], a[i]) for n in range(1, self.kernel.nbasis + 1)]
                for i in range(self.kernel.nchannels)
            ],
            dtype=np.complex128,
        )

    def precompute_asymptotics(self, a, l, eta):
        r"""precompute asymoptotic wavefunction and derivative in each channel"""
        Hp = np.array(
            [H_plus(ai, li, etai, asym=self.asym) for (ai, li, etai) in zip(a, l, eta)],
            dtype=np.complex128,
        )
        Hm = np.array(
            [
                H_minus(ai, li, etai, asym=self.asym)
                for (ai, li, etai) in zip(a, l, eta)
            ],
            dtype=np.complex128,
        )
        Hpp = np.array(
            [
                H_plus_prime(ai, li, etai, asym=self.asym)
                for (ai, li, etai) in zip(a, l, eta)
            ],
            dtype=np.complex128,
        )
        Hmp = np.array(
            [
                H_minus_prime(ai, li, etai, asym=self.asym)
                for (ai, li, etai) in zip(a, l, eta)
            ],
            dtype=np.complex128,
        )
        self.asymptotics = (Hp, Hm, Hpp, Hmp)

    def precompute_free_matrix(self, a: np.array, l: np.array):
        r"""free matrices only depend on orbital angular momentum l and dimensionless channel
        radius a"""
        self.free_matrix = self.kernel.free_matrix(a, l)

    def reset_energy(self, ecom: np.float64):
        r"""update precomputed asymptotic values for new energy"""
        self.ecom = ecom
        if self.sys.Zproj * self.sys.Ztarget > 0:
            self.precompute_asymptotics(
                self.sys.channel_radii, self.sys.l, self.sys.eta(ecom)
            )

    def f(self, n: np.int32, a: np.float32, s: np.float64):
        """
        nth basis function in channel i - Lagrange-Legendre polynomial of degree n shifted onto
        [0,a_i] and regularized by s/( a_i * xn)
        Note: n is indexed from 1 (constant function is not part of basis)
        """
        assert n <= self.kernel.nbasis and n >= 1

        x = s / a
        xn = self.kernel.abscissa[n - 1]

        # Eqn 3.122 in [Baye, 2015], with s = kr
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
        channels: list,
    ):
        r"""
        Returns the full (Nxn)x(Nxn) interaction in the Lagrange basis,
        where each channel is an nxn block (n being the basis size), and there are NxN such blocks.
        Uses the dimensionless version with s=kr and divided by E.
        """
        nb = self.kernel.nbasis
        sz = nb * self.kernel.nchannels
        C = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(self.kernel.nchannels):
            ch = channels[i]
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
                                ch.domain[1] / ch.k,
                                loc_args,
                            )
                        )
                        / ch.E
                    )
                if int_nonlocal is not None:
                    nloc_args = interaction_matrix.nonlocal_args[i, j]
                    is_symmetric = interaction_matrix.nonlocal_symmetric[i, j]
                    Cij += (
                        self.matrix_nonlocal(
                            int_nonlocal,
                            ch.domain[1] / ch.k,
                            is_symmetric,
                            nloc_args,
                        )
                        / ch.E
                    )
        return C

    def solve(
        self,
        interaction_matrix: InteractionMatrix,
        channels: np.array,
        wavefunction=None,
    ):
        A = self.free_matrix + self.interaction_matrix(interaction_matrix, channels)

        args = (
            A,
            self.b,
            self.asymptotics,
            self.sys.incoming_weights,
            self.sys.channel_radii,
            self.kernel.nchannels,
            self.kernel.nbasis,
        )

        if wavefunction is None:
            return rmsolve_smatrix(*args)
        else:
            return rmsolve_wavefunction(*args)
