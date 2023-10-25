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
from .rmatrix_kernel import LagrangeRMatrixKernel, rmsolve_smatrix


class LagrangeRMatrixSolver:
    def __init__(
        self,
        nbasis,
        nchannels,
        sys: ProjectileTargetSystem,
        ecom=None,
    ):
        r"""
        Parameters:
            a : channel radii

        """
        x, w = np.polynomial.legendre.leggauss(nbasis)
        abscissa = 0.5 * (x + 1)
        weights = 0.5 * w
        self.kernel = LagrangeRMatrixKernel(nbasis, nchannels, abscissa, weights)
        if sys.Zproj * sys.Ztarget > 0:
            self.asym = CoulombAsymptotics
        else:
            self.asym = FreeAsymptotics
        self.sys = sys
        self.ecom = ecom
        if ecom is not None:
            self.set_energy(ecom)

    def precompute_asymptotics(self, a, l, eta):
        # precompute asymptotic values of Lagrange-Legendre for each channel
        b = np.hstack(
            [
                [self.f(n, i, a[i]) for n in range(1, self.kernel.nbasis + 1)]
                for i in range(self.kernel.nchannels)
            ],
            dtype=np.complex128,
        )

        # precompute asymoptotic wavefunction and derivartive in each channel
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
        asymptotics = (Hp, Hm, Hpp, Hmp)
        return b, asymptotics

    def set_energy(self, ecom: np.float64):
        r"""update precomputed values for new energy"""
        self.ecom = ecom
        self.b, self.boundary_asymptotic_wf = self.precompute_asymptotics(
            self.sys.channel_radii, self.sys.l, self.sys.eta(ecom)
        )

    def f(self, n, i, s):
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

    def bloch_se_matrix(
        self,
        interaction_matrix: InteractionMatrix,
        channel_matrix: np.array,
        args=(),
    ):
        r"""Constructs the Bloch-Schrodinger equation in the Lagrange-Legendre basis"""
        nb = self.kernel.nbasis
        sz = nb * self.kernel.nchannels
        C = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(self.kernel.nchannels):
            for j in range(self.kernel.nchannels):
                C[
                    i * nb : i * nb + nb,
                    j * nb : j * nb + nb,
                ] = self.kernel.single_channel_bloch_se_matrix(
                    i,
                    j,
                    interaction_matrix.local_matrix[i, j],
                    interaction_matrix.nonlocal_matrix[i, j],
                    interaction_matrix.nonlocal_symmetric[i, j],
                    channel_matrix[i],
                    args,
                )
        return C

    def solve(
        self,
        interaction_matrix: InteractionMatrix,
        channel_matrix: np.array,
        ecom=None,
        args=(),
    ):
        if ecom is not None:
            self.set_energy(ecom)

        A = self.bloch_se_matrix(interaction_matrix, channel_matrix, args=args)

        return rmsolve_smatrix(
            A,
            self.b,
            self.boundary_asymptotic_wf,
            self.sys.incoming_weights,
            self.sys.channel_radii,
            self.kernel.nchannels,
            self.kernel.nbasis,
        )
