import numpy as np

from ..reactions.system import Channels, Asymptotics
from ..utils import block
from .rmatrix_solver import (
    solution_coeffs_with_inverse,
    solve_smatrix_with_inverse,
)
from ..quadrature import Kernel


class Solver:
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
        self.kernel = Kernel(nbasis, basis)

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
                np.array(
                    [self.kernel.f(n, a[i], a[i]) for n in range(1, nbasis + 1)],
                    dtype=np.complex128,
                )
                for i in range(nchannels)
            ]
        )

    def get_channel_block(self, matrix: np.array, i: np.int32, j: np.int32 = None):
        N = self.kernel.quadrature.nbasis
        if j is None:
            j = i
        return block(matrix, (i, j), (N, N))

    def precompute_free_matrix_energy_scaling(self, energies, coupled=True):
        r"""
        precomputes the block array [E_i / E_0], with each channel block having
        a length equal to the number of basis functions
        """
        # calculate channel scaling for free matrix
        E_scaling = energies / energies[0]
        Nb = self.kernel.quadrature.nbasis
        scaling = (E_scaling * np.ones(Nb)[:, np.newaxis]).T.reshape(
            (Nb * energies.size,)
        )
        if coupled:
            return scaling
        else:
            return [scaling[i * Nb : (i + 1) * Nb] for i in range(energies.size)]
        return

    def free_matrix(self, a: np.array, l: np.array, coupled=True):
        r"""
        precompute free matrix, which only depend on the channel orbital
        angular momenta l and dimensionless channel radii a
        Parameters:
            a: dimensionless radii (e.g. a = k * r_max) for each channel
            l: orbital angular momentum quantum number for each channel
            coupled: whether to return the full matrix or just the block
            diagonal elements (elements off of the channel diagonal are all 0
            for the free matrix). If False, returns a list of Nch (Nb,Nb)
            matrices, where Nch is the number of channels and Nb is the number
            of basis elements, othereise returns the full (Nch x Nb, Nch x Nb)
            matrix
        """
        assert a.size == l.size
        assert a.shape == (a.size,)

        free_matrix = self.kernel.free_matrix(a, l)

        if coupled:
            return free_matrix
        else:
            return [self.get_channel_block(free_matrix, i) for i in range(a.size)]

    def interaction_matrix(
        self,
        channels: Channels,
        local_interaction=None,
        local_args=None,
        nonlocal_interaction=None,
        nonlocal_args=None,
    ):
        r"""
        Returns the full (Nxn)x(Nxn) interaction in the Lagrange basis, where
        each channel is an nxn block (n being the basis size), and there are
        NxN such blocks, for N channels. Uses dimensionless coords with s=k0 r
        and divided by E0, 0 denoting the entrance channel.
        """
        # allocate matrix to store full interaction in Lagrange basis
        nb = self.kernel.quadrature.nbasis
        sz = nb * channels.size
        V = np.zeros((sz, sz), dtype=np.complex128)

        # scale to s = k_0 r, with k_0 the entrance channel wavenumber
        E0 = channels.E[0]
        k0 = channels.k[0]
        channel_radius_r = channels.a[0] / channels.k[0]

        if local_interaction is not None:
            # matrix_local just gives us the diagonal elements of each block ...
            Vl = self.kernel.matrix_local(
                local_interaction, channel_radius_r, args=local_args
            ).reshape(channels.size, channels.size, nb)
            # ... so we have to manually put them in the locations of the diagonals of each block
            for i in range(channels.size):
                for j in range(channels.size):
                    V[i * nb : (i + 1) * nb, j * nb : (j + 1) * nb] = np.diag(
                        Vl[i, j, ...]
                    )

        if nonlocal_interaction is not None:
            # matrix_nonlocal gives us an (nchannels, nchannels, nbasis, nbasis) array
            # which we can just reshape into the the block matrix we want
            V += (
                self.kernel.matrix_nonlocal(
                    nonlocal_interaction, channel_radius_r, args=nonlocal_args
                )
                .reshape(channels.size, channels.size, nb, nb)
                .swapaxes(1, 2)
                .reshape(sz, sz, order="C")
                / k0
            )
        V /= E0
        return V

    def solve(
        self,
        channels: Channels,
        asymptotics: Asymptotics,
        local_interaction=None,
        local_args=None,
        nonlocal_interaction=None,
        nonlocal_args=None,
        free_matrix=None,
        free_matrix_energy_scaling=None,
        basis_boundary=None,
        wavefunction=None,
    ):
        if free_matrix is None:
            free_matrix = self.free_matrix(channels.a, channels.l)
        if free_matrix_energy_scaling is None:
            free_matrix_energy_scaling = self.precompute_free_matrix_energy_scaling(
                channels.E
            )
        if basis_boundary is None:
            basis_boundary = self.precompute_boundaries(channels.a)

        # check consistent sizes
        sz = channels.size * self.kernel.quadrature.nbasis
        assert free_matrix.shape == (sz, sz)
        assert basis_boundary.shape == (sz,)

        # calculate full multichannel Schrödinger equation in the Lagrange basis
        A = free_matrix / free_matrix_energy_scaling
        A += self.interaction_matrix(
            channels, local_interaction, local_args, nonlocal_interaction, nonlocal_args
        )

        # solve system using the R-matrix method
        R, S, Ainv, uext_prime_boundary = solve_smatrix_with_inverse(
            A,
            basis_boundary,
            asymptotics.Hp,
            asymptotics.Hm,
            asymptotics.Hpp,
            asymptotics.Hmp,
            channels.weight,
            channels.a,
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
