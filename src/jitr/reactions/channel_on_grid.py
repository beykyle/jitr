"""Helpers for representing a single reaction channel on a radial grid."""

from __future__ import annotations

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

from ..utils.free_solutions import Gamow_factor
from .system import Channels

spec_ch = [
    ("l", int64),
    ("mass", float64),
    ("E", float64),
    ("k", float64),
    ("eta", float64),
    ("domain", float64[:]),
]


def make_channel_data(channels: Channels) -> list[SingleChannelData]:
    """Split a coupled :class:`Channels` object into single-channel views."""
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
    """Single-channel data for traditional coordinate-space solvers."""

    def __init__(
        self,
        l: int,
        reduced_mass: float,
        a: float,
        E: float,
        k: float,
        eta: float,
    ) -> None:
        """Store the channel quantum numbers and integration domain."""
        self.l = l
        self.mass = reduced_mass
        self.E = E
        self.k = k
        self.eta = eta
        self.domain = np.array([1.0e-3, a], dtype=np.float64)

    def initial_conditions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stable inward-boundary initial conditions in ``s = k r``."""
        C_l = Gamow_factor(self.l, self.eta)
        min_rho_0 = (np.finfo(np.float64).eps * 10 / C_l) ** (1 / (self.l + 1))
        s_0 = max(self.domain[0], min_rho_0)
        u0 = C_l * s_0 ** (self.l + 1)
        uprime0 = C_l * (self.l + 1) * s_0**self.l
        domain = np.array([s_0, self.domain[1]], dtype=np.float64)
        init_con = np.array([u0, uprime0], dtype=np.complex128)
        assert domain[1] > domain[0]
        return domain, init_con

    def s_grid(self, size: int = 200) -> np.ndarray:
        """Return an evenly spaced grid in dimensionless radius."""
        return np.linspace(self.domain[0], self.domain[1], size)
