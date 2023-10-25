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


class Wavefunction:
    """
    Represents a wavefunction, expressed internally to the channel as a linear combination
    of Lagrange-Legendre functions, and externally as a linear combination of incoming and
    outgoing Coulomb scattering wavefunctions
    """

    def __init__(self, lm, coeffs, S, se, is_entrance_channel=False):
        self.is_entrance_channel = is_entrance_channel
        self.lm = lm
        self.se = se
        self.coeffs = coeffs
        self.S = S

        self.callable = self.u()

    def __call__(self, s):
        return self.callable(s)

    def uext(self):
        out = lambda s: np.array(
            self.S * H_plus(s, self.se.l, self.se.eta),
            dtype=complex,
        )
        if self.is_entrance_channel:
            return lambda s: np.array(
                H_minus(s, self.se.l, self.se.eta) + out(s),
                dtype=complex,
            )
        else:
            return out

    def uint(self):
        return lambda s: np.sum(
            [self.coeffs[n - 1] * self.lm.f(n, s) for n in range(1, self.lm.N + 1)],
            axis=0,
        )

    def u(self):
        uint = self.uint()
        uext = self.uext()
        ch_radius = self.se.a
        return lambda s: np.where(s < ch_radius, uint(s), uext(s))
