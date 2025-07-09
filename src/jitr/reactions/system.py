import numpy as np

from ..utils.free_solutions import (
    H_plus,
    H_minus,
    H_plus_prime,
    H_minus_prime,
)


def scalar_couplings(l):
    r"""default case of uncoupled partial waves"""
    return np.array([[1.0]])


def spin_half_orbit_coupling(l):
    r"""For a spin-1/2 nucleon scattering off a spin-0 nucleus with spin-obit coupling,
    there are maximally 2 different total angular momentum couplings: l+1/2 and l-1/2.

    Parameters:
        l (int): angular momentum

    Returns:
        couplings (np.ndarray): expectation value of l dot s in each j channel
    """
    js = [l + 1.0 / 2] if l == 0 else [l + 1.0 / 2, l - 1.0 / 2]
    return np.diag([(j * (j + 1) - l * (l + 1) - 0.5 * (0.5 + 1)) for j in js]) / 2


class Asymptotics:
    r"""
    Stores the asymptotic behavior in a set of partial wave channels
    """

    def __init__(self, Hp, Hm, Hpp, Hmp):
        self.Hp = Hp
        self.Hm = Hm
        self.Hpp = Hpp
        self.Hmp = Hmp
        self.size = Hp.shape[0]

    def decouple(self):
        r"""
        If coupling is diagonal, this partial wave can be decoupled
        into individual Asymptotics objects
        """
        asym = []
        for i in range(self.size):
            asym.append(
                Asymptotics(
                    self.Hp[i : i + 1],
                    self.Hm[i : i + 1],
                    self.Hpp[i : i + 1],
                    self.Hmp[i : i + 1],
                )
            )
        return asym


class Channels:
    r"""
    Stores information about a set of channels at a given partial wave
    """

    def __init__(self, E, k, mu, eta, a, l, couplings):
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

    def decouple(self):
        r"""
        If self.couplings is diagonal, this partial wave can be decoupled
        into self.size Channels objects
        """
        assert (
            np.count_nonzero(self.couplings - np.diag(np.diagonal(self.couplings))) == 0
        )
        couplings = np.diag(self.couplings)
        channels = []
        for i in range(self.size):
            channels.append(
                Channels(
                    self.E[i : i + 1],
                    self.k[i : i + 1],
                    self.mu[i : i + 1],
                    self.eta[i : i + 1],
                    self.a,
                    self.l[i : i + 1],
                    np.array([[couplings[i]]]),
                )
            )
        return channels


class ProjectileTargetSystem:
    r"""
    Stores physics parameters of asystem. Calculates useful parameters for each partial wave.
    """

    def __init__(
        self,
        channel_radius: np.float64,
        lmax: np.int64,
        mass_target: np.float64 = 0,
        mass_projectile: np.float64 = 0,
        Ztarget: np.float64 = 0,
        Zproj: np.float64 = 0,
        coupling=scalar_couplings,
        channel_levels=None,
    ):
        r"""
        @params
            channel_radius (np.float64):  dimensionless channel radius k_0 * radius
        """

        self.channel_radius = channel_radius
        self.lmax = lmax
        self.l = np.arange(0, lmax + 1, dtype=np.int64)
        self.couplings = [coupling(l) for l in self.l]

        if channel_levels is None:
            channel_levels = np.zeros(self.couplings[0].shape[0], dtype=np.float64)
        self.channel_levels = channel_levels

        self.mass_target = mass_target
        self.mass_projectile = mass_projectile
        self.Ztarget = Ztarget
        self.Zproj = Zproj

    def get_partial_wave_channels(
        self,
        Elab,
        Ecm,
        mu,
        k,
        eta,
    ):
        r"""
        For each partial wave, returns a Channels object describing the array of channels
        in each wave
        """

        channels = []
        asymptotics = []
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


def uniform_array_from_scalar_or_array(scalar_or_array, size):
    if isinstance(scalar_or_array, np.ndarray):
        return scalar_or_array
    elif isinstance(scalar_or_array, list):
        return np.array(scalar_or_array)
    else:
        return np.ones(size) * scalar_or_array
