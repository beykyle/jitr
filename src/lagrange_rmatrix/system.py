from numba.experimental import jitclass
from numba import float64, int32
import numpy as np

from .utils import hbarc, c
from .channel import ChannelData


class InteractionMatrix:
    def __init__(self, nchannels):
        self.nchannels = nchannels
        self.matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.is_local = np.ones((self.nchannels, self.nchannels), dtype=bool)
        self.is_symmetric = np.ones((self.nchannels, self.nchannels), dtype=bool)

    def set_interaction(interaction, i: int = 0, j: int = 0, is_local=True, is_symmetric=True):
        self.matrix[i, j] = interaction
        self.is_local[i, j] = is_local
        self.is_symmetric[i, j] = is_symmetric

system_spec = [
    (incident_energy, float64),
    (reduced_mass, float64),
    (channel_radii, float64),
    (l, array),
    (Ztarget, float64),
    (Zproj, float64),
    (nchannels, int32),
    (level_energies, float64[:]),
    (incoming_weights, float64[:]),
]


@jitclass(system_spec)
class ProjectileTargetSystem:
    def __init__(
        self,
        incident_energy: np.float64,
        reduced_mass: np.float64,
        channel_radii: np.float64,
        l: np.array,
        Ztarget: np.float64,
        Zproj: np.float64,
        nchannels: np.int32 = 1,
        level_energies: np.array = None,
        incoming_weights: np.array = None,
    ):
        self.incident_energy = incident_energy
        self.reduced_mass = reduced_mass
        self.channel_radii = channel_radii
        self.Ztarget = Ztarget
        self.Zproj = Zproj
        self.nchannels = nchannels
        if level_energies is None:
            level_energies = np.zeros(self.nchannels)
        self.level_energies = level_energies
        if incoming_weights is None:
            incoming_weights = np.zeros(self.nchannels)
            incoming_weights[0] = 1
        self.incoming_weights = incoming_weights
        self.l = l

    def k(self):
        r"""
        Wavenumber in each channel
        """
        return (
            np.sqrt(
                2 * (self.incident_energy - self.level_energies) / self.reduced_mass
            )
            / hbarc
        )

    def velocity(self):
        return np.sqrt(2 * hbarc * self.k() / self.reduced_mass) * c

    def eta(self):
        r"""
        Sommerfield parameter in each channel
        """
        k = self.k()
        return (alpha * self.Zproj * self.Ztarget) * self.reduced_mass / (hbarc * k)

    def build_channels(self):
        k = self.k()
        eta = self.eta()
        channels = np.empty((self.num_channels, self.num_channels), dtype=object)
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                ChannelData[i, j] = RadialSEChannel(
                    l[i],
                    self.reduced_mass,
                    self.channel_radius[i],
                    self.incident_energy - self.level_energies[i],
                    k[i],
                    eta[i],
                )

        return channels
