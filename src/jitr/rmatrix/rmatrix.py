"""High-level R-matrix solver built on a Lagrange mesh."""

from __future__ import annotations

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

    def __init__(self, nbasis: int, basis: str = "Legendre", **args: Any) -> None:
        """Construct an R-matrix solver on a Lagrange mesh."""
        self.kernel = Kernel(nbasis, basis)

    def radial_grid(self, a: float, k0: float) -> FloatArray:
        """Return the physical quadrature grid for a dimensionless radius ``a``."""
        return self.kernel.radial_grid(a / k0)

    def nonlocal_radial_grids(
        self, a: float, k0: float
    ) -> tuple[FloatArray, FloatArray]:
        """Return the physical tensor-product quadrature grids for ``a``."""
        return self.kernel.nonlocal_radial_grids(a / k0)

    def precompute_boundaries(self, a: float) -> ComplexArray:
        """Precompute basis-function values at the channel radius ``a``."""
        nbasis = self.kernel.quadrature.nbasis
        return np.array(
            [self.kernel.f(n, a, a) for n in range(1, nbasis + 1)],
            dtype=np.complex128,
        )

    def get_channel_block(
        self,
        matrix: npt.ArrayLike,
        i: int,
        j: int | None = None,
    ) -> ComplexArray:
        """Extract a channel block from a full block-structured matrix."""
        n_basis = self.kernel.quadrature.nbasis
        if j is None:
            j = i
        return np.asarray(block(np.asarray(matrix), (i, j), (n_basis, n_basis)))

    def kinetic_matrix(
        self,
        a: float,
        l: npt.ArrayLike,
        mu: npt.ArrayLike | None = None,
    ) -> ComplexArray:
        """Assemble the full kinetic-energy matrix."""
        l_array = np.asarray(l)
        mu_array = np.ones(l_array.shape, dtype=np.float64)
        if mu is not None:
            mu_array = np.asarray(mu, dtype=np.float64)

        n_basis = self.kernel.quadrature.nbasis
        n_channels = int(np.size(l_array))
        size = n_basis * n_channels
        kinetic = np.zeros((size, size), dtype=np.complex128)
        for i in range(n_channels):
            block_ij = (
                self.kernel.quadrature.kinetic_matrix(a, int(l_array[i]))
                * mu_array[0]
                / mu_array[i]
            )
            kinetic[
                (i * n_basis) : (i + 1) * n_basis, (i * n_basis) : (i + 1) * n_basis
            ] += block_ij
        return kinetic

    def energy_matrix(
        self,
        a: float,
        l: npt.ArrayLike,
        E: npt.ArrayLike | None = None,
    ) -> ComplexArray:
        """Assemble the full overlap-weighted energy matrix."""
        l_array = np.asarray(l)
        energy_scale = np.ones(l_array.shape, dtype=np.float64)
        if E is not None:
            energy_scale = np.asarray(E, dtype=np.float64)

        n_basis = self.kernel.quadrature.nbasis
        n_channels = int(np.size(l_array))
        size = n_basis * n_channels
        energy = np.zeros((size, size), dtype=np.complex128)
        for i in range(n_channels):
            energy[
                (i * n_basis) : (i + 1) * n_basis, (i * n_basis) : (i + 1) * n_basis
            ] += (self.kernel.overlap * energy_scale[i])

        return energy / energy_scale[0]

    def free_matrix(
        self,
        a: float,
        l: npt.ArrayLike,
        E: npt.ArrayLike | None = None,
        mu: npt.ArrayLike | None = None,
        coupled: bool = True,
    ) -> ComplexArray | list[ComplexArray]:
        """Precompute the free Hamiltonian matrix."""
        l_array = np.asarray(l)
        free_matrix = self.kinetic_matrix(a, l_array, mu) - self.energy_matrix(
            a, l_array, E
        )

        if coupled:
            return free_matrix
        return [
            self.get_channel_block(free_matrix, i) for i in range(int(np.size(l_array)))
        ]

    def interaction_matrix(
        self,
        k0: float,
        E0: float,
        a: float,
        nch: int,
        local_potential: npt.ArrayLike | None = None,
        nonlocal_potential: npt.ArrayLike | None = None,
    ) -> ComplexArray:
        """Build the interaction matrix from pre-evaluated potential arrays."""
        n_basis = self.kernel.quadrature.nbasis
        size = n_basis * nch
        interaction = np.zeros((size, size), dtype=np.complex128)
        radius = a / k0

        if local_potential is not None:
            local_terms = self.kernel.matrix_local(local_potential)
            if local_terms.ndim == 1:
                if nch != 1:
                    raise ValueError(
                        "1D local potentials are only valid for one channel"
                    )
                local_terms = local_terms.reshape(1, 1, n_basis)
            elif local_terms.shape[:2] != (nch, nch):
                raise ValueError(
                    f"local_potential leading dimensions {local_terms.shape[:2]} "
                    f"do not match nch={nch}; expected shape ({nch}, {nch}, {n_basis})"
                )
            for i in range(nch):
                for j in range(nch):
                    interaction[
                        i * n_basis : (i + 1) * n_basis, j * n_basis : (j + 1) * n_basis
                    ] = np.diag(local_terms[i, j, ...])

        if nonlocal_potential is not None:
            nonlocal_terms = self.kernel.matrix_nonlocal(nonlocal_potential, radius)
            if nonlocal_terms.ndim == 2:
                if nch != 1:
                    raise ValueError(
                        "2D nonlocal potentials are only valid for one channel"
                    )
                nonlocal_terms = nonlocal_terms.reshape(1, 1, n_basis, n_basis)
            elif nonlocal_terms.shape[:2] != (nch, nch):
                raise ValueError(
                    f"nonlocal_potential leading dimensions {nonlocal_terms.shape[:2]} "
                    f"do not match nch={nch}; "
                    f"expected shape ({nch}, {nch}, {n_basis}, {n_basis})"
                )
            interaction += (
                nonlocal_terms.swapaxes(1, 2).reshape(size, size, order="C") / k0
            )

        return interaction / E0

    def solve(
        self,
        channels: Channels,
        asymptotics: Asymptotics,
        local_potential: npt.ArrayLike | None = None,
        nonlocal_potential: npt.ArrayLike | None = None,
        interaction_matrix: ComplexArray | None = None,
        free_matrix: ComplexArray | None = None,
        basis_boundary: ComplexArray | None = None,
        weights: FloatArray | None = None,
        wavefunction: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Solve the scattering problem for one coupled set of channels."""
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
                local_potential=local_potential,
                nonlocal_potential=nonlocal_potential,
            )

        size = channels.size * self.kernel.quadrature.nbasis
        assert free_matrix.shape == (size, size)
        assert interaction_matrix.shape == (size, size)
        assert basis_boundary.shape == (self.kernel.quadrature.nbasis,)

        system_matrix = free_matrix + interaction_matrix
        R, S, inverse, uext_prime_boundary = solve_smatrix_with_inverse(
            system_matrix,
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

        if not wavefunction:
            return R, S, uext_prime_boundary

        coeffs = solution_coeffs_with_inverse(
            inverse,
            basis_boundary,
            S,
            uext_prime_boundary,
            channels.size,
            self.kernel.quadrature.nbasis,
        )
        return R, S, coeffs, uext_prime_boundary
