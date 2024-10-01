import numpy as np

from ..reactions.system import Channels, Asymptotics
from ..utils import block
from .core import (
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
        Constructs an R-matrix solver on a Lagrange mesh
        @parameters:
            nbasis (int) : size of basis; e.g. number of quadrature points for
            integration
            ecom (float) : center of mass frame scattering energy
            basis (str): what basis/mesh to use (see Ch. 3 of Baye, 2015)
        """
        self.kernel = Kernel(nbasis, basis)

    def precompute_boundaries(self, a: np.float64):
        r"""
        precompute boundary values of Lagrange basis functions for a set of
        channel radii a
        @parameters:
            a: dimensionless radii (e.g. a = k * r_max) for each channel
        """
        nbasis = self.kernel.quadrature.nbasis
        return np.array(
            [self.kernel.f(n, a, a) for n in range(1, nbasis + 1)],
            dtype=np.complex128,
        )

    def get_channel_block(self, matrix: np.ndarray, i: np.int32, j: np.int32 = None):
        N = self.kernel.quadrature.nbasis
        if j is None:
            j = i
        return block(matrix, (i, j), (N, N))

    def kinetic_matrix(
        self,
        a: np.float64,
        l: np.ndarray,
        mu: np.float64 = None,
    ):
        r"""
        @returns:
            kinetic_matrix (np.ndarray): the full (Nb)x(Nb) kinetic energy
            matrix
        @parameters:
            a : dimensionless channel radius (r * k_0) with k_0 being the
                wavenumber in the entrance channel
            l : orbital angular momentum in each channel
            mu : reduced mass in each channel. For problems with same
                partition in each channel, can leave as none as the
                dimensionless reduced mass is 1.
        """
        if mu is None:
            mu = np.ones(l.shape, dtype=np.float64)

        Nb = self.kernel.quadrature.nbasis
        Nch = np.size(l)
        sz = Nb * Nch
        F = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(Nch):
            Fij = self.kernel.quadrature.kinetic_matrix(a, l[i]) * mu[0] / mu[i]
            F[(i * Nb) : (i + 1) * Nb, (i * Nb) : (i + 1) * Nb] += Fij
        return F

    def energy_matrix(
        self,
        a: np.float64,
        l: np.float64,
        E: np.float64 = None,
    ):
        r"""
        @returns:
            energy_matrix (np.ndarray): the full (nchannels x nbasis)^2.
                Diagonal in channel space but possibly not in Lagrange
                space. 1/E0 (Ei <f_n | f_m>)
        @parameters:
            a : dimensionless channel radius (r * k_0) with k_0 being
                the wavenumber in the entrance channel
            l : orbital angular momentum in each channel
            E : energy in each channel. For single channel problems,
                can leave as None, as the dimensionless energy is 1.
                Otherwise it is Ei/E0 for channel i with energy Ei
        """
        if E is None:
            E = np.ones(l.shape, dtype=np.float64)

        Nb = self.kernel.quadrature.nbasis
        Nch = np.size(l)
        sz = Nb * Nch
        F = np.zeros((sz, sz), dtype=np.complex128)
        for i in range(Nch):
            F[(i * Nb) : (i + 1) * Nb, (i * Nb) : (i + 1) * Nb] += (
                self.kernel.overlap * E[i]
            )

        return F / E[0]

    def free_matrix(
        self,
        a: np.float64,
        l: np.array,
        E: np.ndarray = None,
        mu: np.ndarray = None,
        coupled=True,
    ):
        r"""
        precompute free matrix (kinetic + energy), which only depend on the
        channel orbital angular momenta l and dimensionless channel radius a
        @parameters:
            a: dimensionless radii (e.g. a = k * r_max) for each channel
            l: orbital angular momentum quantum number for each channel
            E : energy in each channel. For single channel problems, can set
                as None, as the dimensionless energy is 1.
            mu : reduced mass in each channel. For problems with same partition
                in each channel, can leave as none as the dimensionless reduced
                mass is 1.
            coupled: whether to return the full matrix or just the block
                diagonal elements (elements off of the channel diagonal are all
                0 for the free matrix). If False, returns a list of Nch (Nb,Nb)
                matrices, where Nch is the number of channels and Nb is the
                number of basis elements, othereise returns the full
                (Nch x Nb, Nch x Nb) matrix
        """
        free_matrix = self.kinetic_matrix(a, l, mu) - self.energy_matrix(a, l, E)

        if coupled:
            return free_matrix
        else:
            return [self.get_channel_block(free_matrix, i) for i in range(l.size)]

    def interaction_matrix(
        self,
        k0: np.float64,
        E0: np.float64,
        a: np.float64,
        nch: np.int32,
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
        @parameters:
            k0 (float): fixed wavenumber [fm^-1] with which to scale the
                coordinate r. Typically this just the wavenumber in the 0th
                channel.
            E0 (float): fixed energy [MeV] with which to scale the system.
                Typically this just the energy in the 0th channel.
            a (float): dimensionless channel radius
            local_interaction (callable): the local potential, a function of
                r and *args
            local_args (tuple): the args that get passed into
                local_interaction
            nonlocal_interaction (callable): the nonlocal potential, a
                function of r, r', and *args
            nonlocal_args (tuple): the args that get passed into
                nonlocal_interaction
        """
        # allocate matrix to store full interaction in Lagrange basis
        nb = self.kernel.quadrature.nbasis
        sz = nb * nch
        V = np.zeros((sz, sz), dtype=np.complex128)

        # scaling
        channel_radius_r = a / k0

        if local_interaction is not None:
            # matrix_local just gives us the diagonal elements of each block ...
            Vl = self.kernel.matrix_local(
                local_interaction, channel_radius_r, args=local_args
            ).reshape(nch, nch, nb)
            # ... so we have to manually put them in the locations of the diagonals of each block
            for i in range(nch):
                for j in range(nch):
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
                .reshape(nch, nch, nb, nb)
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
        interaction_matrix=None,
        free_matrix=None,
        basis_boundary=None,
        weights=None,
        wavefunction=None,
    ):
        # calculate everything that hasn't been precomputed
        if free_matrix is None:
            free_matrix = self.free_matrix(
                channels.a,
                channels.l,
                channels.E,
                channels.mu,
                coupled=True,
            )
        if basis_boundary is None:
            basis_boundary = self.precompute_boundaries(channels.a)
        if weights is None:
            weights = np.zeros(channels.size, dtype=np.float64)
            weights[0] = 1
        if interaction_matrix is None:
            interaction_matrix = self.interaction_matrix(
                channels.k[0],
                channels.E[0],
                channels.a,
                channels.size,
                local_interaction,
                local_args,
                nonlocal_interaction,
                nonlocal_args,
            )

        # check consistent sizes
        sz = channels.size * self.kernel.quadrature.nbasis
        assert free_matrix.shape == (sz, sz)
        assert interaction_matrix.shape == (sz, sz)
        assert basis_boundary.shape == (self.kernel.quadrature.nbasis,)

        # this is the full multi-channel representation of 1/E_0 (H-E)
        A = free_matrix + interaction_matrix

        # solve system using the R-matrix method
        R, S, Ainv, uext_prime_boundary = solve_smatrix_with_inverse(
            A,
            basis_boundary,
            asymptotics.Hp,
            asymptotics.Hm,
            asymptotics.Hpp,
            asymptotics.Hmp,
            weights,
            channels.a,
            channels.size,
            self.kernel.quadrature.nbasis,
        )

        if wavefunction is None:
            return R, S, uext_prime_boundary
        else:
            # get the wavefunction expansion coefficients in the Lagrange basis
            x = solution_coeffs_with_inverse(
                Ainv,
                basis_boundary,
                S,
                uext_prime_boundary,
                channels.size,
                self.kernel.quadrature.nbasis,
            )
            return R, S, x, uext_prime_boundary
