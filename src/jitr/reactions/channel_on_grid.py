from numba.experimental import jitclass
from numba import float64, int64
import numpy as np

from .system import Channels
from ..utils.free_solutions import Gamow_factor

spec_ch = [
    ("l", int64),
    ("mass", float64),
    ("E", float64),
    ("k", float64),
    ("eta", float64),
    ("domain", float64[:]),
]


def make_channel_data(channels: Channels):
    r""" """
    return [
        SingleChannelData(
            channels.l[i],
            channels.mu[i],
            channels.a,
            channels.E[i],
            channels.k[i],
            channels.eta[i],
        )
        for i in range(channels.num_channels)
    ]


@jitclass(spec_ch)
class SingleChannelData:
    r"""
    Data and capabilities for traditional solvers using a discretized grid
    in coordinate (s=kr) space
    """

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
        # use asymptotic behavior of F to avoid floating point
        # stability issue in RK solver
        C_l = Gamow_factor(self.l, self.eta)
        min_rho_0 = (np.finfo(np.float64).eps * 10 / C_l) ** (1 / (self.l + 1))
        s_0 = max(self.domain[0], min_rho_0)
        u0 = C_l * s_0 ** (self.l + 1)
        uprime0 = C_l * (self.l + 1) * s_0**self.l
        domain = np.array([s_0, self.domain[1]], dtype=np.float64)
        init_con = np.array([u0, uprime0], dtype=np.complex128)
        assert domain[1] > domain[0]
        return domain, init_con

    def s_grid(self, size=200):
        return np.linspace(self.domain[0], self.domain[1], size)
