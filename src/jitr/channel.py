from numba.experimental import jitclass
from numba import float64, int64
import numpy as np

from .utils import (
    Gamow_factor,
    H_plus,
    H_minus,
    alpha,
    hbarc,
    eval_scaled_interaction,
    null,
    CoulombAsymptotics,
)

spec_ch = [
    ("l", int64),
    ("mass", float64),
    ("E", float64),
    ("k", float64),
    ("eta", float64),
    ("domain", float64[:]),
]


@jitclass(spec_ch)
class ChannelData:
    """ """

    def __init__(
        self,
        l: np.int64,
        reduced_mass: np.float64,
        a: np.float64,
        E: np.float64,
        k: np.float64,
        eta: np.float64,
    ):
        """ """
        self.l = l
        self.mass = reduced_mass
        self.E = E
        self.k = k
        self.eta = eta
        self.domain = np.array([1.0e-3, a], dtype=np.float64)

    def initial_conditions(self):
        """
        initial conditions for numerical integration in coordinate (s) space
        """
        s_0 = self.domain[0]
        C_l = Gamow_factor(self.l, self.eta)
        u0 = C_l * s_0 ** (self.l + 1)
        uprime0 = C_l * (self.l + 1) * s_0**self.l
        return np.array([u0 * (1 + 0j), uprime0 * (1 + 0j)])

    def s_grid(self, size=200):
        return np.linspace(self.domain[0], self.domain[1], size)


class Wavefunctions:
    """
    Represents a wavefunction, expressed internally to the channel as a linear combination
    of Lagrange-Legendre functions, and externally as a linear combination of incoming and
    outgoing Coulomb scattering wavefunctions
    """

    def __init__(
        self,
        solver,
        coeffs,
        S,
        uext_prime_boundary,
        incoming_weights,
        channels,
        asym=CoulombAsymptotics,
    ):
        self.solver = solver
        self.coeffs = coeffs
        self.S = S
        self.channels = channels
        self.incoming_weights = incoming_weights
        self.uext_prime_boundary = uext_prime_boundary
        self.asym = asym

    def uext(self):
        r"""Returns a callable which evaluates the asymptotic form of wavefunction in each channel,
        valid external to the channel radii. Follows Eqn. 3.79 from P. Descouvemont and D. Baye
        2010 Rep. Prog. Phys. 73 036301
        """

        def uext_channel(i):
            # TODO optimize this
            l = self.channels[i].l
            eta = self.channels[i].eta
            asym_func_in = lambda s: (
                self.incoming_weights[i] * H_minus(s, l, eta, asym=self.asym)
            )
            asym_func_out = lambda s: np.sum(
                [
                    self.incoming_weights[j]
                    * self.S[i, j]
                    * H_plus(s, l, eta, asym=self.asym)
                    for j in self.solver.kernel.nchannels
                ],
                axis=0,
            )
            return lambda s: asym_func_in(s) + asym_func_out(s)

        uext = []
        for i in range(self.solver.kernel.nchannels):
            uext.append(uext_channel(i))

        return uext

    def uint(self):
        def uint_channel(i):
            return lambda s: np.sum(
                [
                    self.coeffs[i, n] * self.solver.f(n + 1, i, s)
                    for n in range(self.solver.kernel.nbasis)
                ],
                axis=0,
            )

        uint = []
        for i in range(self.solver.kernel.nchannels):
            uint.append(uint_channel(i))

        return uint
