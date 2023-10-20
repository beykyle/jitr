import numpy as np
import scipy.special as sc
import scipy.linalg as la

from .util import (
    hbarc,
    c,
    CoulombAsymptotics,
    FreeAsymptotics,
    H_plus,
    H_minus,
    H_plus_prime,
    H_minus_prime,
)
from .system import ProjectileTargetSystem
from .rmatrix_kernel import LagrangeRMatrixKernel, rmsolve_smatrix


class LagrangeRMatrixSolver:
    def __init__(
        self, nbasis, nchannels, sys: ProjectileTargetSystem, asym=CoulombAsymptotics
    ):
        r"""
        Parameters:
            a : channel radii

        """
        if not isinstance(a, np.array):
            a = np.ones((nchannels,)) * a

        x, w = np.polynomial.legendre.leggauss(nbasis)
        abscissa = 0.5 * (x + 1)
        weights = 0.5 * w
        self.kernel = LagrangeRMatrixKernel(nbasis, nchannels, abscissa, weights)

        self.incoming_weights = sys.incoming_weights
        self.a = sys.channel_radius
        self.k = sys.k()
        self.eta = sys.eta()
        self.l = sys.l

        # precompute asymptotic values of Lagrange-Legendre for each channel
        self.b = np.hstack(
            [
                [self.f(n, self.a[i]) / self.a[i] for n in range(1, self.nbasis + 1)]
                for i in range(self.nchannels)
            ]
        )

        # precompute asymoptotic wavefunction and derivartive in each channel
        self.Hp = np.array(
            [
                H_plus(a, l, eta, asym=asym)
                for (a, l, eta) in zip(self.a, self.l, self.eta)
            ]
        )
        self.Hm = np.array(
            [
                H_minus(a, l, eta, asym=asym)
                for (a, l, eta) in zip(self.a, self.l, self.eta)
            ]
        )
        self.Hpp = np.array(
            [
                H_plus_prime(a, l, eta, asym=asym)
                for (a, l, eta) in zip(self.a, self.l, self.eta)
            ]
        )
        self.Hmp = np.array(
            [
                H_minus_prime(a, l, eta, asym=asym)
                for (a, l, eta) in zip(self.a, self.l, self.eta)
            ]
        )
        self.asymptotics = (self.Hp, self.Hm, self.Hpp.self.Hmp)

    def update_energy():
        r"""update precomputed values for new energy"""
        pass

    def f(self, n, s):
        """
        nth basis function in channel i - Lagrange-Legendre polynomial of degree n shifted onto
        [0,a_i] and regularized by s/( a_i * xn)
        Note: n is indexed from 1 (constant function is not part of basis)
        """
        assert n <= self.kernel.nbasis and n >= 1

        x = s / self.a
        xn = self.kernel.abscissa[n - 1]

        # Eqn 3.122 in [Baye, 2015], with s = kr
        return (
            (-1.0) ** (self.kernel.nbasis - n)
            * np.sqrt((1 - xn) / xn)
            * eval_legendre(self.kernel.nbasis, 2.0 * x - 1.0)
            * x
            / (x - xn)
        )

    def solve(self, interaction_matrix: np.array, channel_matrix: np.array, args=()):
        A = self.kernel.bloch_se_matrix(interaction_matrix, channel_matrix, args)
        return rmsolve_smatrix(
            A,
            self.b,
            self.asymptotics,
            self.incoming_weights,
            self.a,
            self.kernel.nchannels,
            self.kernel.nbasis,
        )
