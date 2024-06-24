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
        x, w = np.polynomial.legendre.leggauss(nbasis)
        abscissa = 0.5 * (x + 1)
        weights = 0.5 * w
        self.kernel = LagrangeRMatrixKernel(nbasis, nchannels, abscissa, weights)

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
    ):

        A = self.bloch_se_matrix(interaction_matrix, channels)

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
