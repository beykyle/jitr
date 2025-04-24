import numpy as np
from ..quadrature.kernel import Kernel
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
        channel_radius_fm: float,
        S: np.ndarray,
        coeffs: np.ndarray,
        channels: np.ndarray,
        kernel: Kernel,
        incoming_channel_idx: int = 0,
    ):
        self.channel_radius_fm = channel_radius_fm
        self.S = S
        self.coeffs = coeffs
        self.channels = channels
        self.incoming_channel_idx = 0

    def uext(self):
        r"""Returns a callable which evaluates the asymptotic form of
        wavefunction in each channel, valid external to the channel radii.
        Follows Eqn. 3.79 from P. Descouvemont and D. Baye 2010 Rep.
        Prog. Phys. 73 036301
        """

        def uext_channel(i):
            l = self.channels["l"][i]
            eta = self.channels["eta"][i]

            def asym_func_in(s):
                if i == self.incoming_channel_idx:
                    return 0
                else:
                    return H_minus(s, l, eta)

            def asym_func_out(s):
                return np.sum(
                    [
                        self.S[i, j] * H_plus(s, l, eta)
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
                    / (self.channel_radius_fm * self.channels["k"][i])
                    * self.kernel.f(
                        n + 1, self.channel_radius_fm * self.channels["k"][i], s
                    )
                    for n in range(self.kernel.quadrature.nbasis)
                ],
                axis=0,
            )

        uint = []
        for i in range(self.channels.size):
            uint.append(uint_channel(i))

        return uint
