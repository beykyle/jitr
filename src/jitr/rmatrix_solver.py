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
            a : channel radii

        """
        self.kernel = build_kernel(nbasis)

        # precomputed matrices of weights and abscissa for vectorized operations
        self.weight_matrix = np.sqrt(np.outer(self.kernel.weights, self.kernel.weights))
        self.Xi, self.Xj = np.meshgrid(self.kernel.abscissa, self.kernel.abscissa)

        if asym is None:
            if sys.Zproj * sys.Ztarget > 0:
                asym = CoulombAsymptotics
            else:
                asym = FreeAsymptotics
        self.asym = asym

        self.sys = sys
        self.precompute_boundaries(self.sys.channel_radii)
        self.precompute_free_matrix(self.sys.channel_radii, self.sys.l)

        self.ecom = ecom
        if ecom is not None:
            self.precompute_asymptotics(
                self.sys.channel_radii, self.sys.l, self.sys.eta(ecom)
            )

    def integrate_local(self, f, a, args=()):
        """
        integrates local operator of form f(x,*args)dx from [0,a] in Gauss quadrature
        """
        return np.sum(f(self.kernel.abscissa * a, *args) * self.kernel.weights) * a

    def double_integrate_nonlocal(self, f, a, is_symmetric=True, args=()):
        """
        double integrates nonlocal operator of form f(x,x',*args)dxdx' from [0,a] x [0,a]
        in Gauss quadrature
        """
        w = self.kernel.weights
        x = self.kernel.abscissa
        d = 0

        # TODO vectorize
        if is_symmetric:
            for n in range(0, self.nbasis):
                d += f(x[n] * a, x[n] * a) * w[n]
                for m in range(n + 1, self.nbasis):
                    # account for both above and below diagonal
                    d += 2 * f(x[n] * a, x[m] * a) * np.sqrt(w[n] * w[m])
        else:
            for n in range(0, self.nbasis):
                for m in range(0, self.nbasis):
                    d += f(x[n] * a, x[m] * a) * np.sqrt(w[n] * w[m])

        return d * a

    def matrix_local(self, f, args=()):
        r"""get diagonal elements of matrix for arbitrary local vectorized operator f(x)"""
        return self.kernel.weights * f(self.kernel.abscissa, *args)

    def matrix_nonlocal(self, f, is_symmetric=True, args=()):
        r"""get matrix for arbitrary vectorized operator f(x,xp)"""
        if is_symmetric:
            n = self.kernel.nbasis
            umask = self.triu_indices(n)
            M = np.zeros(n, n)
            M[mask] *= self.weight_matrix[mask] * f(self.Xi[mask], self.Xj[mask])
            M += np.triu(M, k=1).T
            return M
        else:
            return self.weight_matrix * f(self.Xi, self.Xj, *args)

    def precompute_boundaries(self, a):
        r"""precompute boundary values of Lagrange-Legendre for each channel"""
        self.b = np.hstack(
            [
                [self.f(n, i, a[i]) for n in range(1, self.kernel.nbasis + 1)]
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

    def f(self, n: np.int32, i: np.int32, s: np.float64):
        """
        nth basis function in channel i - Lagrange-Legendre polynomial of degree n shifted onto
        [0,a_i] and regularized by s/( a_i * xn)
        Note: n is indexed from 1 (constant function is not part of basis)
        """
        assert n <= self.kernel.nbasis and n >= 1

        x = s / self.sys.channel_radii[i]
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
        where each channel is an nxn block (n being the basis size), and there are NxN such blocks
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
                    Cij += self.kernel.single_channel_local_interaction_matrix(
                        int_local,
                        ch,
                        loc_args,
                    )
                if int_nonlocal is not None:
                    nloc_args = interaction_matrix.nonlocal_args[i, j]
                    is_symmetric = interaction_matrix.nonlocal_symmetric[i, j]
                    Cij += self.kernel.single_channel_nonlocal_interaction_matrix(
                        int_nonlocal,
                        ch,
                        is_symmetric,
                        nloc_args,
                    )
        return C

    def bloch_se_matrix(
        self,
        interaction_matrix: InteractionMatrix,
        channels: list,
    ):
        return self.interaction_matrix(interaction_matrix, channels) + self.free_matrix

    def solve(
        self,
        interaction_matrix: InteractionMatrix,
        channels: np.array,
        wavefunction=None,
        added_operator=None,
    ):
        A = self.bloch_se_matrix(interaction_matrix, channels)

        # allow user to add arbitrary operator
        if added_operator is not None:
            A += added_operator

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
