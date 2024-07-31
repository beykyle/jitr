import numpy as np

from .system import InteractionMatrix
from .utils import block
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
        basis="Legendre",
        **args,
    ):
        r"""
        @parameters:
            nbasis (int) : size of basis; e.g. number of quadrature points for
            integration
            ecom (float) : center of mass frame scattering energy
            basis (str): what basis/mesh to use (see Ch. 3 of Baye, 2015)
        """
        self.kernel = QuadratureKernel(nbasis, basis)

    def precompute_boundaries(self, a):
        r"""
        precompute boundary values of Lagrange basis functions for a set of
        channel radii a
            a: dimensionless radii (e.g. a = k * r_max) for each channel
        """
        nbasis = self.kernel.quadrature.nbasis
        nchannels = np.size(a)
        return np.hstack(
            [
                [self.kernel.f(n, a[i], a[i]) for n in range(1, nbasis + 1)]
                for i in range(nchannels)
            ],
            dtype=np.complex128,
        )

    def get_channel_block(self, matrix: np.array, i: np.int32, j: np.int32 = None):
        N = self.kernel.quadrature.nbasis
        if j is None:
            j = i
        return block(matrix, (i, j), (N, N))

    def free_matrix(self, a: np.array, l: np.array, full_matrix=True):
        r"""
        precompute free matrix, which only depend on the channel orbital
        angular momenta l and dimensionless channel radii a
        Parameters:
            a: dimensionless radii (e.g. a = k * r_max) for each channel
            l: orbital angular momentum quantum number for each channel
            full_matrix: whether to return the full matrix or just the block
            diagonal elements (elements off of the channel diagonal are all 0
            for the free matrix). If False, returns a list of Nch (Nb,Nb)
            matrices, where Nch is the number of channels and Nb is the number
            of basis elements, othereise returns the full (Nch x Nb, Nch x Nb)
            matrix
        """
        assert a.size == l.size
        assert a.shape == (a.size,)

        free_matrix = self.kernel.free_matrix(a, l)

        if full_matrix:
            return free_matrix
        else:
            return [self.get_channel_block(free_matrix, i) for i in range(a.size)]

    def interaction_matrix(
        self,
        interaction: InteractionMatrix,
        channels: np.array,
    ):
        r"""
        Returns the full (Nxn)x(Nxn) interaction in the Lagrange basis, where
        each channel is an nxn block (n being the basis size), and there are
        NxN such blocks, for N channels. Uses dimensionless coords with s=kr
        and divided by E.
        """
        # ensure consistency in sizing between interaction and channels
        nchannels = interaction.nchannels
        assert channels.size == nchannels

        # allocate matrix to store full interaction in Lagrange basis
        nb = self.kernel.quadrature.nbasis
        sz = nb * nchannels
        C = np.zeros((sz, sz), dtype=np.complex128)

        for i in range(nchannels):
            channel_radius_r = channels["a"][i] / channels["k"][i]
            E = channels["E"][i]
            for j in range(nchannels):
                Cij = C[i * nb : i * nb + nb, j * nb : j * nb + nb]
                int_local = interaction.local_matrix[i, j]
                int_nonlocal = interaction.nonlocal_matrix[i, j]
                if int_local is not None:
                    loc_args = interaction.local_args[i, j]
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
                    nloc_args = interaction.nonlocal_args[i, j]
                    is_symmetric = interaction.nonlocal_symmetric[i, j]
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
        interaction: InteractionMatrix,
        channels: np.array,
        free_matrix=None,
        basis_boundary=None,
        wavefunction=None,
    ):
        if free_matrix is None:
            free_matrix = self.free_matrix(channels["a"], channels["l"])
        if basis_boundary is None:
            basis_boundary = self.precompute_boundaries(channels["a"])

        # check consistent sizes
        sz = interaction.nchannels * self.kernel.quadrature.nbasis
        assert channels.size == interaction.nchannels
        assert free_matrix.shape == (sz, sz)
        assert basis_boundary.shape == (sz,)

        # calculate full multichannel Schrödinger equation in the Lagrange basis
        A = free_matrix + self.interaction_matrix(interaction, channels)

        # solve system using the R-matrix method
        R, S, Ainv, uext_prime_boundary = solve_smatrix_with_inverse(
            A,
            basis_boundary,
            channels["Hp"],
            channels["Hm"],
            channels["Hpp"],
            channels["Hmp"],
            channels["weight"],
            channels["a"],
            channels.size,
            self.kernel.quadrature.nbasis,
        )

        # get the wavefunction expansion coefficients in the Lagrange
        # basis if needed
        if wavefunction is None:
            return R, S, uext_prime_boundary
        else:
            x = solution_coeffs_with_inverse(
                Ainv,
                basis_boundary,
                S,
                uext_prime_boundary,
                channels.size,
                self.kernel.quadrature.nbasis,
            )
            return R, S, x, uext_prime_boundary
