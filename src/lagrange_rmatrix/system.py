from numba.experimental import jitclass
import numpy as np

from .channel import ChannelData


class InteractionMatrix:
    def __init__(self, nchannels):
        self.nchannels = nchannels
        self.matrix = np.empty((self.nchannels, self.nchannels), dtype=object)
        self.is_local = np.ones((self.nchannels, self.nchannels), dtype=bool)

    def set_local_interaction(interaction, i: int = 0, j: int = 0):
        self.matrix[i, j] = interaction
        self.is_local[i, j] = True

    def set_nonlocal_interaction(interaction, i: int = 0, j: int = 0):
        self.matrix[i, j] = interaction
        self.is_local[i, j] = False


@jitclass()
class ProjectileTargetSystem:
    incident_energy: np.float64  # [MeV]
    reduced_mass: np.float64  # [MeV]
    channel_radius: np.float64  # [dimensionless]
    Ztarget: np.float64 = 0
    Zproj: np.float64 = 0
    num_channels: np.int32 = 1
    level_energies: np.array
    l: np.array

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
                    self.channel_radius,
                    self.incident_energy - self.level_energies[i],
                    k[i],
                    eta[i],
                )

        return channels
