import numpy as np
import scipy.special as sc

from .bloch_se import ProjectileTargetSystem
from .rmatrix_kernel import LagrangeRMatrixKernel


class LagrangeRMatrixSolver:
    def __init__(self, nbasis, nchannels, a):
        r"""
        Parameters:
            a : channel radii

        """
        if not isinstance(a, np.array):
            a = np.ones((nchannels,)) * a
        self.a = a
        self.kernel = LagrangeRMatrixKernel(nbasis, nchannels)
        # TODO precompute b for each channel
        # TODO precompute Zlpus, Zminus
        # TODO add wrapper for solve that passes in an interaction, energy, k, l and args

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
            (-1.0) ** (self.N - n)
            * np.sqrt((1 - xn) / xn)
            * eval_legendre(self.N, 2.0 * x - 1.0)
            * x
            / (x - xn)
        )

    def solve(
        self, system: ProjectileTargetSystem, interaction_matrix: InteractionMatrix
    ):
        pass
