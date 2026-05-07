"""Wavefunction reconstruction for solved R-matrix channels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ..utils.free_solutions import CoulombAsymptotics, H_minus, H_plus

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
        asym=CoulombAsymptotics,
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
        self.asym = asym

    def uext(self) -> list[Callable[[npt.ArrayLike], ComplexArray]]:
        """Return external-channel wavefunctions valid beyond the boundary."""

        def uext_channel(i: int) -> Callable[[npt.ArrayLike], ComplexArray]:
            l = self.channels.l[i]
            eta = self.channels.eta[i]

            def asym_func_in(s: float) -> complex:
                return self.incoming_weights[i] * H_minus(s, l, eta, asym=self.asym)

            def asym_func_out(s: float) -> complex:
                return np.sum(
                    [
                        self.incoming_weights[j]
                        * self.S[i, j]
                        * H_plus(s, l, eta, asym=self.asym)
                        for j in range(len(self.channels))
                    ],
                    axis=0,
                )

            return lambda s_mesh: np.array(
                [1j / 2 * (asym_func_in(s) - asym_func_out(s)) for s in s_mesh],
                dtype=np.complex128,
            )

        return [uext_channel(i) for i in range(len(self.channels))]

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
