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
        self.sys = sys
        self.incoming_weights = sys.incoming_weights
        b, asym = self.precompute_asymptotics(self.sys.a, self.sys.l, self.sys.eta())
        self.b = b
        self.asym = asym

    def precompute_asymptotics(self, a, l, eta):
        # precompute asymptotic values of Lagrange-Legendre for each channel
        b = np.hstack(
            [
                [self.f(n, a[i]) / a[i] for n in range(1, self.nbasis + 1)]
                for i in range(self.nchannels)
            ]
        )

        # precompute asymoptotic wavefunction and derivartive in each channel
        Hp = np.array(
            [H_plus(ai, li, etai, asym=asym) for (ai, li, etai) in zip(a, l, eta)]
        )
        Hm = np.array(
            [H_minus(ai, li, etai, asym=asym) for (ai, li, etai) in zip(a, l, eta)]
        )
        Hpp = np.array(
            [H_plus_prime(ai, li, etai, asym=asym) for (ai, li, etai) in zip(a, l, eta)]
        )
        Hmp = np.array(
            [
                H_minus_prime(ai, li, etai, asym=asym)
                for (ai, li, etai) in zip(a, l, eta)
            ]
        )
        asymptotics = (Hp, Hm, Hpp.Hmp)

        return b, asymptotics

    def update_energy(Ecom: np.float64):
        r"""update precomputed values for new energy"""
        self.sys.incident_energy = Ecom
        b, asym = self.precompute_asymptotics(self.sys.a, self.sys.l, self.sys.eta())
        self.b = b
        self.asym = asym

    def f(self, n, i, s):
        """
        nth basis function in channel i - Lagrange-Legendre polynomial of degree n shifted onto
        [0,a_i] and regularized by s/( a_i * xn)
        Note: n is indexed from 1 (constant function is not part of basis)
        """
        assert n <= self.kernel.nbasis and n >= 1

        x = s / self.sys.a[i]
        xn = self.kernel.abscissa[n - 1]

        # Eqn 3.122 in [Baye, 2015], with s = kr
        return (
            (-1.0) ** (self.kernel.nbasis - n)
            * np.sqrt((1 - xn) / xn)
            * eval_legendre(self.kernel.nbasis, 2.0 * x - 1.0)
            * x
            / (x - xn)
        )

    def solve(
        self, interaction_matrix: InteractionMatrix, channel_matrix: np.array, args=()
    ):
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
