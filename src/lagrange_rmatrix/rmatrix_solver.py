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
        asym=CoulombAsymptotics,
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
        self.asym = asym
        self.sys = sys
        self.incoming_weights = sys.incoming_weights
        self.ecom = ecom
        if ecom is not None:
            self.b, self.asym = self.precompute_asymptotics(
                self.sys.channel_radii, self.sys.l, self.sys.eta(ecom)
            )

    def precompute_asymptotics(self, a, l, eta):
        # precompute asymptotic values of Lagrange-Legendre for each channel
        b = np.hstack(
            [
                [self.f(n, i, a[i]) / a[i] for n in range(1, self.kernel.nbasis + 1)]
                for i in range(self.kernel.nchannels)
            ]
        )

        # precompute asymoptotic wavefunction and derivartive in each channel
        Hp = np.array(
            [H_plus(ai, li, etai, asym=self.asym) for (ai, li, etai) in zip(a, l, eta)]
        )
        Hm = np.array(
            [H_minus(ai, li, etai, asym=self.asym) for (ai, li, etai) in zip(a, l, eta)]
        )
        Hpp = np.array(
            [
                H_plus_prime(ai, li, etai, asym=self.asym)
                for (ai, li, etai) in zip(a, l, eta)
            ]
        )
        Hmp = np.array(
            [
                H_minus_prime(ai, li, etai, asym=self.asym)
                for (ai, li, etai) in zip(a, l, eta)
            ]
        )
        asymptotics = (Hp, Hm, Hpp.Hmp)
        return b, asymptotics

    def set_energy(ecom: np.float64):
        r"""update precomputed values for new energy"""
        self.ecom = ecom
        self.b, self.asym = self.precompute_asymptotics(
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

    def solve(
        self,
        interaction_matrix: InteractionMatrix,
        channel_matrix: np.array,
        args=(),
        ecom=None,
    ):
        if ecom is not None:
            self.set_energy(ecom)

        local_matrix = interaction_matrix.local_matrix
        nonlocal_matrix = interaction_matrix.nonlocal_matrix
        nonlocal_symmetric = interaction_matrix.nonlocal_symmetric

        A = self.solve(
            local_matrix, nonlocal_matrix, is_symmetric, channel_matrix, args
        )

        return rmsolve_smatrix(
            A,
            self.b,
            self.asymptotics,
            self.incoming_weights,
            self.sys.a,
            self.kernel.nchannels,
            self.kernel.nbasis,
        )
