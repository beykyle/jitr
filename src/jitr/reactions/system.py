"""Channel-system objects used by the R-matrix solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from ..utils.free_solutions import H_minus, H_minus_prime, H_plus, H_plus_prime

FloatArray: TypeAlias = npt.NDArray[np.float64]
ComplexArray: TypeAlias = npt.NDArray[np.complex128]
CouplingFunction: TypeAlias = Callable[[int], FloatArray]


def scalar_couplings(l: int) -> FloatArray:
    """Return the default single-channel coupling matrix for partial wave ``l``."""
    return np.array([[1.0]])


def spin_half_orbit_coupling(l: int) -> FloatArray:
    """Return ``l · sigma`` expectation values for spin-1/2 on spin-0 scattering.

    Args:
        l: Orbital angular momentum.

    Returns:
        A diagonal matrix containing the coupling strength in each ``j`` channel.
    """
    js = [l + 1.0 / 2] if l == 0 else [l + 1.0 / 2, l - 1.0 / 2]
    return np.diag([(j * (j + 1) - l * (l + 1) - 0.5 * (0.5 + 1)) for j in js])


class Asymptotics:
    """Asymptotic solutions for a set of channels in one partial wave."""

    def __init__(
        self,
        Hp: ComplexArray,
        Hm: ComplexArray,
        Hpp: ComplexArray,
        Hmp: ComplexArray,
    ) -> None:
        self.Hp = Hp
        self.Hm = Hm
        self.Hpp = Hpp
        self.Hmp = Hmp
        self.size = Hp.shape[0]

    def decouple(self) -> list[Asymptotics]:
        """Split diagonal asymptotics into one-channel objects."""
        asymptotics: list[Asymptotics] = []
        for i in range(self.size):
            asymptotics.append(
                Asymptotics(
                    self.Hp[i : i + 1],
                    self.Hm[i : i + 1],
                    self.Hpp[i : i + 1],
                    self.Hmp[i : i + 1],
                )
            )
        return asymptotics


class Channels:
    """Channel metadata for one partial wave."""

    def __init__(
        self,
        E: FloatArray,
        k: FloatArray,
        mu: FloatArray,
        eta: FloatArray,
        a: float,
        l: FloatArray,
        couplings: FloatArray,
    ) -> None:
        self.num_channels = E.shape[0]
        assert couplings.shape == (self.num_channels, self.num_channels)
        assert k.shape[0] == self.num_channels
        assert mu.shape[0] == self.num_channels
        assert eta.shape[0] == self.num_channels
        self.E = E
        self.k = k
        self.mu = mu
        self.eta = eta
        self.a = a
        self.l = l
        self.size = couplings.shape[0]
        self.couplings = couplings

    def decouple(self) -> list[Channels]:
        """Split diagonal channel data into one-channel objects."""
        assert (
            np.count_nonzero(self.couplings - np.diag(np.diagonal(self.couplings))) == 0
        )
        diagonal_couplings = np.diag(self.couplings)
        channels: list[Channels] = []
        for i in range(self.size):
            channels.append(
                Channels(
                    self.E[i : i + 1],
                    self.k[i : i + 1],
                    self.mu[i : i + 1],
                    self.eta[i : i + 1],
                    self.a,
                    self.l[i : i + 1],
                    np.array([[diagonal_couplings[i]]]),
                )
            )
        return channels


class ProjectileTargetSystem:
    """System-level information for a projectile-target partition."""

    def __init__(
        self,
        channel_radius: float,
        lmax: int,
        mass_target: float = 0.0,
        mass_projectile: float = 0.0,
        Ztarget: float = 0.0,
        Zproj: float = 0.0,
        coupling: CouplingFunction = scalar_couplings,
        channel_levels: FloatArray | None = None,
    ) -> None:
        """Store channel-independent parameters for each partial wave.

        Args:
            channel_radius: Dimensionless channel radius ``k_0 r``.
            lmax: Maximum orbital angular momentum.
            mass_target: Target mass in MeV/c^2.
            mass_projectile: Projectile mass in MeV/c^2.
            Ztarget: Target charge.
            Zproj: Projectile charge.
            coupling: Function returning the coupling matrix for each ``l``.
            channel_levels: Optional excitation-energy offsets for coupled channels.
        """
        self.channel_radius = channel_radius
        self.lmax = lmax
        self.l = np.arange(0, lmax + 1, dtype=np.int64)
        self.couplings = [coupling(int(l)) for l in self.l]

        if channel_levels is None:
            channel_levels = np.zeros(self.couplings[0].shape[0], dtype=np.float64)
        self.channel_levels = channel_levels

        self.mass_target = mass_target
        self.mass_projectile = mass_projectile
        self.Ztarget = Ztarget
        self.Zproj = Zproj

    def get_partial_wave_channels(
        self,
        Elab: float | FloatArray,
        Ecm: float | FloatArray,
        mu: float | FloatArray,
        k: float | FloatArray,
        eta: float | FloatArray,
    ) -> tuple[list[Channels], list[Asymptotics]]:
        """Build channel and asymptotic objects for every partial wave."""
        channels: list[Channels] = []
        asymptotics: list[Asymptotics] = []
        for l in range(0, self.lmax + 1):
            num_channels = self.couplings[l].shape[0]
            eta_array = uniform_array_from_scalar_or_array(eta, num_channels)
            channels.append(
                Channels(
                    uniform_array_from_scalar_or_array(Ecm, num_channels),
                    uniform_array_from_scalar_or_array(k, num_channels),
                    uniform_array_from_scalar_or_array(mu, num_channels),
                    eta_array,
                    self.channel_radius,
                    np.ones(num_channels) * l,
                    self.couplings[l],
                )
            )
            asymptotics.append(
                Asymptotics(
                    Hp=np.array(
                        [
                            H_plus(self.channel_radius, l, channel_eta)
                            for channel_eta in eta_array
                        ],
                        dtype=np.complex128,
                    ),
                    Hm=np.array(
                        [
                            H_minus(self.channel_radius, l, channel_eta)
                            for channel_eta in eta_array
                        ],
                        dtype=np.complex128,
                    ),
                    Hpp=np.array(
                        [
                            H_plus_prime(self.channel_radius, l, channel_eta)
                            for channel_eta in eta_array
                        ],
                        dtype=np.complex128,
                    ),
                    Hmp=np.array(
                        [
                            H_minus_prime(self.channel_radius, l, channel_eta)
                            for channel_eta in eta_array
                        ],
                        dtype=np.complex128,
                    ),
                )
            )

        return channels, asymptotics


def uniform_array_from_scalar_or_array(
    scalar_or_array: float | list[float] | FloatArray,
    size: int,
) -> FloatArray:
    """Broadcast a scalar or list-like input to a one-dimensional array."""
    if isinstance(scalar_or_array, np.ndarray):
        return np.asarray(scalar_or_array, dtype=np.float64)
    if isinstance(scalar_or_array, list):
        return np.asarray(scalar_or_array, dtype=np.float64)
    return np.full(size, scalar_or_array, dtype=np.float64)
