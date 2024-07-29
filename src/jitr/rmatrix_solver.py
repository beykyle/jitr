import numpy as np

from .system import InteractionMatrix
from .rmatrix import (
    solution_coeffs,
    solution_coeffs_with_inverse,
    solve_smatrix_with_inverse,
    solve_smatrix_without_inverse,
)
from .kernel import QuadratureKernel


class RMatrixSolver:
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
        self.kernel = QuadratureKernel(nbasis, nchannels, basis)
        self.precompute_boundaries(self.channel_radii)

    def precompute_boundaries(self, a):
        r"""
        precompute boundary values of Lagrange-Legendre for each channel
        """
        nbasis = self.kernel.quadrature.nbasis
        nchannels = self.kernel.nchannels
        self.b = np.hstack(
            [
                [self.kernel.f(n, a[i], a[i]) for n in range(1, nbasis + 1)]
                for i in range(nchannels)
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
            for the free matrix). If False, returns a list of Nch (Nb,Nb)
            matrices, where Nch is the number of channels and Nb is the number
            of basis elements
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

    def interaction_matrix(
        self,
        interaction_matrix: InteractionMatrix,
        channels: np.array,
    ):
        r"""
        Returns the full (Nxn)x(Nxn) interaction in the Lagrange basis, where
        each channel is an nxn block (n being the basis size), and there are
        NxN such blocks, for N channels. Uses dimensionless coords with s=kr
        and divided by E.
        """
        nb = self.kernel.quadrature.nbasis
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
                            self.kernel.matrix_local(
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
                        self.kernel.matrix_nonlocal(
                            int_nonlocal,
                            channel_radius_r,
                            is_symmetric,
                            nloc_args,
                        )
                        / channels["k"][i]
                        / E
                    )
                    # extra factor of 1/k because dr = 1/k ds
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
            self.kernel.quadrature.nbasis,
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
                self.kernel.quadrature.nbasis,
            )
            return R, S, x, uext_prime_boundary
