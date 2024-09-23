import numpy as np
from ..utils.free_solutions import (
    H_plus,
    H_minus,
    CoulombAsymptotics,
)


class Wavefunctions:
    """
    Represents a wavefunction, expressed internally to the channel as a
    linear combination of Lagrange-Legendre functions, and externally as
    a linear combination of incoming and outgoing Coulomb scattering
    wavefunctions
    """

    def __init__(
        self,
        solver,
        coeffs,
        S,
        uext_prime_boundary,
        channels,
        incoming_weights=None,
        asym=CoulombAsymptotics,
    ):
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

    def uext(self):
        r"""Returns a callable which evaluates the asymptotic form of
        wavefunction in each channel, valid external to the channel radii.
        Follows Eqn. 3.79 from P. Descouvemont and D. Baye 2010 Rep.
        Prog. Phys. 73 036301
        """

        def uext_channel(i):
            l = self.channels.l[i]
            eta = self.channels.eta[i]

            def asym_func_in(s):
                return self.incoming_weights[i] * H_minus(s, l, eta, asym=self.asym)

            def asym_func_out(s):
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

        uext = []
        for i in range(len(self.channels)):
            uext.append(uext_channel(i))

        return uext

    def uint(self):
        def uint_channel(i):
            return lambda s: np.sum(
                [
                    self.coeffs[i, n]
                    / self.channels.a
                    * self.solver.kernel.f(n + 1, self.channels.a, s)
                    for n in range(self.solver.kernel.quadrature.nbasis)
                ],
                axis=0,
            )

        uint = []
        for i in range(self.channels.size):
            uint.append(uint_channel(i))

        return uint
