"""Wavefunction reconstruction for solved R-matrix channels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ..utils.free_solutions import H_minus, H_plus

ComplexArray = npt.NDArray[np.complex128]


class Wavefunctions:
    """Internal and external wavefunction representations for solved channels."""

    def __init__(
        self,
        solver,
        coeffs: ComplexArray,
        S: ComplexArray,
        uext_prime_boundary: ComplexArray,
        channels,
        incoming_weights: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Store the ingredients needed to reconstruct channel wavefunctions."""
        self.solver = solver
        self.coeffs = coeffs
        self.S = S
        self.uext_prime_boundary = uext_prime_boundary
        self.channels = channels
        if incoming_weights is None:
            incoming_weights = np.zeros(channels.size, dtype=np.float64)
            incoming_weights[0] = 1
        self.incoming_weights = incoming_weights

    def uext(self) -> list[Callable[[npt.ArrayLike], ComplexArray]]:
        """Return external-channel wavefunctions valid beyond the boundary.

        The returned callables take the channel-0 coordinate ``s = k_0 r``;
        channel ``i`` is evaluated at its own ``rho_i = k_i r``. ``S`` is taken
        to be the flux-normalized matrix returned by :meth:`Solver.solve`.
        """
        # amplitude of the outgoing wave in each channel, in the raw (not
        # flux-normalized) convention that the asymptotic forms use
        velocity = self.channels.k / self.channels.mu
        outgoing = (
            self.S * np.sqrt(velocity[np.newaxis, :] / velocity[:, np.newaxis])
        ) @ self.incoming_weights.astype(np.complex128)
        k_ratio = self.channels.k / self.channels.k[0]

        def uext_channel(i: int) -> Callable[[npt.ArrayLike], ComplexArray]:
            l = int(self.channels.l[i])
            eta = float(self.channels.eta[i])

            def u(s: float) -> complex:
                rho = s * k_ratio[i]
                return (
                    1j
                    / 2
                    * (
                        self.incoming_weights[i] * H_minus(rho, l, eta)
                        - outgoing[i] * H_plus(rho, l, eta)
                    )
                )

            return lambda s_mesh: np.array(
                [u(s) for s in np.atleast_1d(s_mesh)], dtype=np.complex128
            )

        return [uext_channel(i) for i in range(self.channels.size)]

    def uint(self) -> list[Callable[[float], complex]]:
        """Return internal wavefunctions expanded in the Lagrange basis."""

        def uint_channel(i: int) -> Callable[[float], complex]:
            return lambda s: np.sum(
                [
                    self.coeffs[i, n]
                    / self.channels.a
                    * self.solver.kernel.f(n + 1, self.channels.a, s)
                    for n in range(self.solver.kernel.quadrature.nbasis)
                ],
                axis=0,
            )

        return [uint_channel(i) for i in range(self.channels.size)]
