"""High-level R-matrix solver built on a Lagrange mesh."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from ..quadrature import Kernel
from ..reactions.system import Asymptotics, Channels
from ..utils import block
from .core import solution_coeffs_with_inverse, solve_smatrix_with_inverse

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


class Solver:
    """Solve coupled-channel Schrödinger equations with the R-matrix method."""

    def __init__(self, nbasis: np.int32, basis: str = "Legendre", **args: Any) -> None:
        """Construct an R-matrix solver on a Lagrange mesh.

        Args:
            nbasis: Number of basis functions or quadrature points.
            basis: Mesh family to use.
            **args: Reserved for future extensions.
        """
        self.kernel = Kernel(nbasis, basis)

    def precompute_boundaries(self, a: np.float64) -> ComplexArray:
        """Precompute basis-function values at the channel radius ``a``."""
        nbasis = self.kernel.quadrature.nbasis
        return np.array(
            [self.kernel.f(n, a, a) for n in range(1, nbasis + 1)],
            dtype=np.complex128,
        )

    def get_channel_block(
        self,
        matrix: np.ndarray,
        i: np.int32,
        j: np.int32 | None = None,
    ) -> np.ndarray:
        """Extract a channel block from a full block-structured matrix."""
        n_basis = self.kernel.quadrature.nbasis
        if j is None:
            j = i
        return block(matrix, (i, j), (n_basis, n_basis))

    def kinetic_matrix(
        self,
        a: np.float64,
        l: np.ndarray,
        mu: np.ndarray | None = None,
    ) -> ComplexArray:
        """Assemble the full kinetic-energy matrix."""
        if mu is None:
            mu = np.ones(l.shape, dtype=np.float64)

        n_basis = self.kernel.quadrature.nbasis
        n_channels = np.size(l)
        size = n_basis * n_channels
        kinetic = np.zeros((size, size), dtype=np.complex128)
        for i in range(n_channels):
            block_ij = self.kernel.quadrature.kinetic_matrix(a, l[i]) * mu[0] / mu[i]
            kinetic[
                (i * n_basis) : (i + 1) * n_basis, (i * n_basis) : (i + 1) * n_basis
            ] += block_ij
        return kinetic

    def energy_matrix(
        self,
        a: np.float64,
        l: np.float64,
        E: np.ndarray | None = None,
    ) -> ComplexArray:
        """Assemble the full overlap-weighted energy matrix."""
        if E is None:
            E = np.ones(l.shape, dtype=np.float64)

        n_basis = self.kernel.quadrature.nbasis
        n_channels = np.size(l)
        size = n_basis * n_channels
        energy = np.zeros((size, size), dtype=np.complex128)
        for i in range(n_channels):
            energy[
                (i * n_basis) : (i + 1) * n_basis, (i * n_basis) : (i + 1) * n_basis
            ] += (self.kernel.overlap * E[i])

        return energy / E[0]

    def free_matrix(
        self,
        a: np.float64,
        l: np.ndarray,
        E: np.ndarray | None = None,
        mu: np.ndarray | None = None,
        coupled: bool = True,
    ) -> ComplexArray | list[np.ndarray]:
        """Precompute the free Hamiltonian matrix.

        Args:
            a: Dimensionless channel radius.
            l: Orbital angular momentum for each channel.
            E: Dimensionless energies for each channel.
            mu: Dimensionless reduced masses for each channel.
            coupled: If ``False``, return one diagonal channel block per channel.

        Returns:
            Either the full coupled matrix or a list of uncoupled channel blocks.
        """
        free_matrix = self.kinetic_matrix(a, l, mu) - self.energy_matrix(a, l, E)

        if coupled:
            return free_matrix
        return [self.get_channel_block(free_matrix, i) for i in range(l.size)]

    def interaction_matrix(
        self,
        k0: np.float64,
        E0: np.float64,
        a: np.float64,
        nch: np.int32,
        local_interaction: Callable[..., complex] | None = None,
        local_args: tuple[Any, ...] | None = None,
        nonlocal_interaction: Callable[..., complex] | None = None,
        nonlocal_args: tuple[Any, ...] | None = None,
    ) -> ComplexArray:
        """Build the interaction matrix in the Lagrange basis."""
        n_basis = self.kernel.quadrature.nbasis
        size = n_basis * nch
        interaction = np.zeros((size, size), dtype=np.complex128)

        channel_radius_r = a / k0

        if local_interaction is not None:
            local_terms = self.kernel.matrix_local(
                local_interaction, channel_radius_r, args=local_args
            ).reshape(nch, nch, n_basis)
            for i in range(nch):
                for j in range(nch):
                    interaction[
                        i * n_basis : (i + 1) * n_basis, j * n_basis : (j + 1) * n_basis
                    ] = np.diag(local_terms[i, j, ...])

        if nonlocal_interaction is not None:
            interaction += (
                self.kernel.matrix_nonlocal(
                    nonlocal_interaction, channel_radius_r, args=nonlocal_args
                )
                .reshape(nch, nch, n_basis, n_basis)
                .swapaxes(1, 2)
                .reshape(size, size, order="C")
                / k0
            )
        interaction /= E0
        return interaction

    def solve(
        self,
        channels: Channels,
        asymptotics: Asymptotics,
        local_interaction: Callable[..., complex] | None = None,
        local_args: tuple[Any, ...] | None = None,
        nonlocal_interaction: Callable[..., complex] | None = None,
        nonlocal_args: tuple[Any, ...] | None = None,
        interaction_matrix: ComplexArray | None = None,
        free_matrix: ComplexArray | None = None,
        basis_boundary: ComplexArray | None = None,
        weights: FloatArray | None = None,
        wavefunction: bool | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Solve the scattering problem for one coupled set of channels.

        Args:
            channels: Channel metadata for a single partial wave.
            asymptotics: Precomputed asymptotic solutions for the same channels.
            local_interaction: Optional local interaction.
            local_args: Positional arguments for the local interaction.
            nonlocal_interaction: Optional nonlocal interaction.
            nonlocal_args: Positional arguments for the nonlocal interaction.
            interaction_matrix: Optional precomputed interaction matrix.
            free_matrix: Optional precomputed free matrix.
            basis_boundary: Optional basis values evaluated at the boundary.
            weights: Incoming-channel weights.
            wavefunction: If truthy, also return internal basis coefficients.

        Returns:
            ``(R, S, uext_prime_boundary)`` by default, or
            ``(R, S, coeffs, uext_prime_boundary)`` when ``wavefunction`` is truthy.
        """
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

        size = channels.size * self.kernel.quadrature.nbasis
        assert free_matrix.shape == (size, size)
        assert interaction_matrix.shape == (size, size)
        assert basis_boundary.shape == (self.kernel.quadrature.nbasis,)

        A = free_matrix + interaction_matrix
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

        coeffs = solution_coeffs_with_inverse(
            Ainv,
            basis_boundary,
            S,
            uext_prime_boundary,
            channels.size,
            self.kernel.quadrature.nbasis,
        )
        return R, S, coeffs, uext_prime_boundary
